"""WT-PM Orchestrator: ModelRouter + InferenceEngine + the end-to-end pipeline.

Sensors/SCADA -> Ingestion -> Cleaning/Sync -> Features -> 25-Model Engine ->
Fusion -> Diagnosis -> Degradation -> RUL -> Risk -> XAI -> Twin -> Edge/Safety
-> records for dashboard/API.

The router dynamically decides which models are relevant for the incoming
condition (mode, deployment tier, available data views, questions asked,
latency budget). The inference engine runs them with parallelism, timeouts,
retries and fallbacks, and reports per-model health.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wtpm_platform.base import BaseWTModel, ModelRegistry
from wtpm_platform.contracts import (
    Deployment, FAULT_CLASSES, ModelOutput, OperatingContext, Question,
    SensorBatch, TaskType, WTDataSchema,
)
from wtpm_platform.data import FeaturePipeline, ingest_timeline
from wtpm_platform.engines import (
    AnomalyEngine, DiagnosisEngine, DigitalTwinInterface, EdgeInferenceManager,
    RULManager, SafetyManager, UncertaintyEngine, XAIEngine,
)
from wtpm_platform.fusion import FusionConfig, FusionEngine
from wtpm_platform.advanced import (
    AlertManager, CostRiskOptimizer, DecisionAuditLog, MaintenancePlanner,
    SensorQualityMonitor, downsample_series, health_index,
)
from wtpm_platform.hermes import HermesAgent


def build_default_registry() -> ModelRegistry:
    """Register all 25 adapters. Unavailable ones stay registered but report so."""
    from wtpm_platform.adapters.anomaly import (
        DeepSVDDBoundary, GRUScadaTelemetry, IsolationForestTelemetry,
        LSTMScadaAnomaly, TCNPowerCurve, VAEReconstruction,
    )
    from wtpm_platform.adapters.classification import (
        AeroZipCompressor, CNN1DBearingVibration, ContrastiveSSLVibration,
        DBNFeatureExtraction, RandomForestTelemetry, SNNEventVibration,
        SVMGeneratorStator, XGBoostTabularFaults,
    )
    from wtpm_platform.adapters.prognostics import (
        ConvLSTMWearPrognostics, HMMDegradationStates, InformerLongSequence,
        MLPRULRegression, ParticleFilterRUL,
    )
    from wtpm_platform.adapters.physics_graph_edge import (
        DigitalTwinSurrogate, GNNTurbineCascade, PGBNNWindTurbine,
        QuantizedMobileNetEdge, TinyMLSafetyRelay, XAIShapInterpretable,
    )

    reg = ModelRegistry()
    for cls in (
        CNN1DBearingVibration, ConvLSTMWearPrognostics, TCNPowerCurve,
        GRUScadaTelemetry, LSTMScadaAnomaly, InformerLongSequence,
        SNNEventVibration, ContrastiveSSLVibration, DBNFeatureExtraction,
        RandomForestTelemetry, XGBoostTabularFaults, SVMGeneratorStator,
        DeepSVDDBoundary, IsolationForestTelemetry, HMMDegradationStates,
        ParticleFilterRUL, MLPRULRegression, PGBNNWindTurbine,
        DigitalTwinSurrogate, GNNTurbineCascade, XAIShapInterpretable,
        VAEReconstruction, AeroZipCompressor, QuantizedMobileNetEdge,
        TinyMLSafetyRelay,
    ):
        reg.register(cls())
    return reg


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------
class ModelRouter:
    """Selects models by intended task, available data and conditions."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def route(self, ctx: OperatingContext, batch: SensorBatch) -> Dict[str, List[str]]:
        sel: Dict[str, List[str]] = {"anomaly": [], "classification": [],
                                     "degradation": [], "rul": [], "support": []}
        q = set(ctx.questions)

        def eligible(m: BaseWTModel) -> bool:
            if not m.available():
                return False
            # Hard rule: never put a non-MCU model on the ESP32.
            if ctx.deployment == Deployment.MCU:
                return Deployment.MCU in m.spec.deployment_targets
            if ctx.connect_all:
                return True
            if ctx.deployment not in m.spec.deployment_targets:
                return False
            if m.spec.typical_latency_ms > ctx.latency_budget_ms:
                return False
            return True

        if {"what", "severity", "rul", "action", "safety"} & q:
            for m in self.registry.by_task(TaskType.ANOMALY_DETECTION):
                if eligible(m):
                    sel["anomaly"].append(m.spec.model_id)
            for m in self.registry.by_task(TaskType.FORECASTING):
                if eligible(m):
                    sel["anomaly"].append(m.spec.model_id)  # residual stream

        if {"what", "where", "why"} & q:
            for m in self.registry.by_task(TaskType.FAULT_CLASSIFICATION):
                if not eligible(m):
                    continue
                # supervised classifiers need to have been fitted with labels
                if not m.fitted and not ctx.has_labels:
                    continue
                needs_vib = any("vib" in r for r in m.spec.input_requirements)
                if needs_vib and batch.vib_waveforms is None:
                    continue
                sel["classification"].append(m.spec.model_id)

        if {"severity", "rul", "action"} & q:
            for m in self.registry.by_task(TaskType.DEGRADATION_STATE):
                if eligible(m):
                    sel["degradation"].append(m.spec.model_id)
        if {"rul", "action"} & q:
            for m in self.registry.by_task(TaskType.RUL_ESTIMATION):
                if eligible(m):
                    sel["rul"].append(m.spec.model_id)

        for task in (TaskType.FEATURE_EXTRACTION, TaskType.COMPRESSION,
                     TaskType.SURROGATE, TaskType.SAFETY,
                     TaskType.GRAPH_ANALYSIS, TaskType.EXPLAINABILITY,
                     TaskType.EDGE_INFERENCE):
            for m in self.registry.by_task(task):
                if eligible(m):
                    sel["support"].append(m.spec.model_id)
        return sel


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------
@dataclass
class InferenceReport:
    outputs: Dict[str, ModelOutput] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    fallbacks_used: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    timeouts: List[str] = field(default_factory=list)


class InferenceEngine:
    """Parallel/sequential execution with timeout, retry and fallback."""

    def __init__(self, registry: ModelRegistry, max_workers: int = 4,
                 timeout_s: float = 300.0, retries: int = 1) -> None:
        self.registry = registry
        self.max_workers = max_workers
        self.timeout_s = timeout_s
        self.retries = retries

    def run(self, model_ids: Sequence[str], batch: SensorBatch,
            parallel: bool = True) -> InferenceReport:
        report = InferenceReport()
        ids = [i for i in model_ids if i in self.registry]

        def call(mid: str) -> ModelOutput:
            model = self.registry.get(mid)
            last: Optional[ModelOutput] = None
            for _ in range(self.retries + 1):
                last = model.predict(batch)
                if last.ok:
                    return last
            return last  # type: ignore[return-value]

        def finish(mid: str, out: ModelOutput) -> None:
            if not out.ok:
                report.errors[mid] = out.error
                fb = self.registry.resolve_fallback(mid)
                if fb is not None and fb.spec.model_id not in report.outputs:
                    fout = fb.predict(batch)
                    if fout.ok:
                        report.fallbacks_used[mid] = fb.spec.model_id
                        report.outputs[fb.spec.model_id] = fout
                        report.timings_ms[fb.spec.model_id] = fout.inference_time_ms
                return
            report.outputs[mid] = out
            report.timings_ms[mid] = out.inference_time_ms

        if parallel and len(ids) > 1:
            with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futs = {pool.submit(call, mid): mid for mid in ids}
                for fut in cf.as_completed(futs, timeout=None):
                    mid = futs[fut]
                    try:
                        finish(mid, fut.result(timeout=self.timeout_s))
                    except cf.TimeoutError:
                        report.timeouts.append(mid)
                    except Exception as exc:  # noqa: BLE001
                        report.errors[mid] = f"{type(exc).__name__}: {exc}"
        else:
            for mid in ids:
                t0 = time.perf_counter()
                finish(mid, call(mid))
                report.timings_ms.setdefault(mid, (time.perf_counter() - t0) * 1000)
        return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """The central WT-PM brain: fit once, then answer the seven questions."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        fusion_cfg: Optional[FusionConfig] = None,
        max_workers: int = 4,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.features = FeaturePipeline()
        self.fusion = FusionEngine(fusion_cfg)
        self.router = ModelRouter(self.registry)
        self.infer = InferenceEngine(self.registry, max_workers=max_workers)
        self.anomaly_engine = AnomalyEngine(self.fusion)
        self.diagnosis_engine = DiagnosisEngine(self.fusion)
        self.uncertainty_engine = UncertaintyEngine()
        self.rul_manager = RULManager(self.registry, self.fusion)
        self.xai = XAIEngine(self.registry)
        self.safety = SafetyManager(self.registry)
        self.twin = DigitalTwinInterface(self.registry)
        self.edge = EdgeInferenceManager(self.registry)
        self.alerts = AlertManager()
        self.quality = SensorQualityMonitor()
        self.planner = MaintenancePlanner()
        self.costing = CostRiskOptimizer()
        self.audit = DecisionAuditLog()
        self.hermes = HermesAgent(self)

    # -- data prep -----------------------------------------------------------
    def prepare(self, batch: SensorBatch) -> SensorBatch:
        return self.features.transform(batch)

    # -- training ------------------------------------------------------------
    def fit(self, batch: SensorBatch, ctx: OperatingContext,
            train_fraction: float = 1.0,
            model_ids: Optional[Sequence[str]] = None,
            verbose: bool = True) -> Dict[str, str]:
        """Fit on a caller-supplied training record, never the evaluation record.

        Supervised models see labelled training rows; normal-only detectors and
        encoders additionally exclude fault-labelled rows. Each adapter must
        honour its mask. train_fraction optionally limits the allowed prefix.
        """
        batch = self.prepare(batch) if batch.features is None else batch
        if not 0 < train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1]")
        n = batch.n_steps
        allowed = np.arange(n) < int(n * train_fraction)
        quality = batch.meta.get("mask")
        if quality is not None:
            allowed &= np.asarray(quality, bool).all(axis=1)
        lab = batch.meta.get("fault_label")
        healthy = allowed.copy()
        if lab is not None:
            healthy &= np.asarray(lab).astype(int) == 0
        healthy_tasks = {TaskType.ANOMALY_DETECTION, TaskType.FORECASTING,
                         TaskType.FEATURE_EXTRACTION, TaskType.COMPRESSION}
        wanted = set(model_ids) if model_ids else set(self.registry.ids())
        status: Dict[str, str] = {}
        for mid in sorted(wanted):
            if mid not in self.registry:
                continue
            m = self.registry.get(mid)
            m._fitted = False
            m._fit_error = ""
            m._fit_status = "checking dependencies"
            if not m.available():
                m._fit_status = "unavailable"
                status[mid] = f"unavailable: {m._unavailable_reason}"
                continue
            try:
                t0 = time.perf_counter()
                m._fit_status = "fitting"
                train_mask = healthy if m.spec.task in healthy_tasks else allowed
                if not train_mask.any():
                    raise ValueError("no permitted training samples")
                needs_labels = m.spec.task in {TaskType.FAULT_CLASSIFICATION, TaskType.EDGE_INFERENCE} or mid in {"m02-convlstm-wear", "m17-mlp-rul"}
                if needs_labels and not ctx.has_labels:
                    raise ValueError("supervised fitting requires an explicitly labelled training context")
                # Training happens sequentially. Seeds do not get reset during inference.
                seed = int(batch.meta.get("training_seed", 42)) + sum(map(ord, mid))
                import sys
                np.random.seed(seed)
                if "torch" in sys.modules:
                    sys.modules["torch"].manual_seed(seed)
                if "tensorflow" in sys.modules:
                    sys.modules["tensorflow"].keras.utils.set_random_seed(seed)
                m.fit(batch, train_mask.copy())
                m._training_rows = int(train_mask.sum())
                if not m.fitted:
                    raise RuntimeError("fit() returned without marking the model fitted")
                m._fit_status = "fitted"
                status[mid] = f"fitted in {(time.perf_counter()-t0)*1000:.0f} ms"
            except Exception as exc:  # noqa: BLE001
                m._fitted = False
                m._fit_status = "fit failed"
                m._fit_error = f"{type(exc).__name__}: {exc}"
                status[mid] = f"fit failed: {m._fit_error}"
            if verbose:
                print(f"  [fit] {mid}: {status[mid]}")
        return status

    # -- the seven questions ---------------------------------------------------
    def analyse(self, batch: SensorBatch, ctx: OperatingContext,
                parallel: bool = True) -> Dict[str, Any]:
        t_start = time.perf_counter()
        batch = self.prepare(batch) if batch.features is None else batch
        routed = self.router.route(ctx, batch)
        # inference can only use models that were actually fitted (a model whose
        # fit failed — e.g. m12 on a record without electrical faults — is
        # excluded here but its fit status remains visible via health_report)
        routed = {k: [mid for mid in v
                      if mid in self.registry and self.registry.get(mid).fitted]
                  for k, v in routed.items()}

        # 25-model engine — every routed bucket, including support (m08/m09/m19/
        # m20/m21/m23/m24/m25) so no registered adapter is left idle.
        rep_anom = self.infer.run(routed["anomaly"], batch, parallel=parallel)
        rep_clf = self.infer.run(routed["classification"], batch, parallel=parallel)
        rep_deg = self.infer.run(routed["degradation"], batch, parallel=False)
        rep_rul = self.infer.run([m for m in routed["rul"]
                                  if m != "m16-particle-filter-rul"], batch, parallel=parallel)
        # SHAP needs a genuinely fitted tree, not a "connected" placeholder.
        if "m21-xai-shap" in self.registry:
            xai = self.registry.get("m21-xai-shap")
            xai.bind_estimator(None, None)
            for mid in ("m10-random-forest", "m11-xgboost-tabular"):
                out = rep_clf.outputs.get(mid)
                if out is not None and out.ok and out.extra.get("estimator") is not None:
                    xai.bind_estimator(out.extra["estimator"], mid)
                    break
        rep_sup = self.infer.run(routed.get("support") or [], batch, parallel=parallel)

        # WHAT: fused anomaly
        fused = None
        anomaly_events: List[Dict[str, Any]] = []
        if rep_anom.outputs:
            try:
                fused = self.anomaly_engine.run(list(rep_anom.outputs.values()))
                anomaly_events = self.anomaly_engine.events(fused, batch.timestamps)
            except RuntimeError as exc:
                rep_anom.errors["fusion"] = str(exc)

        # WHAT/WHERE: diagnosis
        diagnosis = self.diagnosis_engine.run(list(rep_clf.outputs.values()), fused)

        # SEVERITY: degradation state
        degradation = next(iter(rep_deg.outputs.values()), None)

        # RUL
        rul_out, rul_info = (None, {})
        if routed["rul"]:
            rul_out, rul_info = self.rul_manager.run(
                batch, list(rep_rul.outputs.values()), degradation)

        # uncertainty + risk
        unc = self.uncertainty_engine.aggregate(
            fused, list(rep_anom.outputs.values()) + list(rep_rul.outputs.values()))
        risk = self._risk_score(fused, diagnosis, degradation, rul_out)

        # WHY: XAI
        explanation = self.xai.explain(batch, diagnosis,
                                       list(rep_clf.outputs.values()), fused)
        if degradation is not None and degradation.ok:
            explanation["degradation_state"] = int(degradation.degradation_state[-1])
        if rul_out is not None and rul_out.ok:
            explanation["estimated_rul_hours"] = round(float(rul_out.rul_hours[-1]), 1)
        explanation["uncertainty"] = {k: round(v, 4) for k, v in unc.items()
                                      if not k.startswith("u::")}

        # SAFETY
        safety = self.safety.evaluate(batch, diagnosis, rul_out)

        # ACTION
        action = self._recommend(diagnosis, degradation, rul_out, risk, safety)

        # ---- advanced ops layer ----
        quality = self.quality.run(batch)
        fused_arr = None if fused is None else fused.score
        deg_arr = None if degradation is None or not degradation.ok else degradation.degradation_state
        rul_arr = None if rul_out is None or not rul_out.ok else rul_out.rul_hours
        vib = batch.channel("bearing_vib_rms_mm_s") if "bearing_vib_rms_mm_s" in batch.channel_names else None
        hidx = health_index(fused_arr, deg_arr, rul_arr, vib)
        rul_now = None if rul_out is None or not rul_out.ok else float(rul_out.rul_hours[-1])
        work_order = self.planner.plan(diagnosis, rul_now, risk, safety["decision"])
        costing = self.costing.choose(risk, rul_now, safety["decision"], work_order)
        action["cost_optimal"] = costing["recommended"]
        action["cost_model"] = costing
        new_alerts = self.alerts.update(
            batch.turbine_id,
            0.0 if fused is None else float(fused.score[-1]),
            risk, safety["decision"], diagnosis.fault, int(batch.timestamps[-1]))
        twin_derate = self.twin.maintenance_scenario(
            batch, rul_out, derate_pct=20.0) if self.twin._twin() else {"error": "twin unfitted"}
        physics = {}
        if "physics_power_residual_kw" in batch.feature_names:
            pr = batch.features[:, list(batch.feature_names).index("physics_power_residual_kw")]
            physics = {
                "power_residual_kw_now": round(float(pr[-1]), 3),
                "power_residual_kw_mean": round(float(np.mean(pr)), 3),
                "series": downsample_series(pr),
            }
        series = {
            "fused_score": [] if fused is None else downsample_series(fused.score),
            "health_index": hidx["series"],
            "rul_hours": [] if rul_arr is None else downsample_series(rul_arr),
            "vibration": [] if vib is None else downsample_series(vib),
            "oil_temp": downsample_series(batch.channel("gearbox_oil_temp_c"))
            if "gearbox_oil_temp_c" in batch.channel_names else [],
            "power_kw": downsample_series(batch.channel("power_kw"))
            if "power_kw" in batch.channel_names else [],
            "stride": hidx["stride"],
        }
        self.audit.record("analyse", {
            "turbine_id": batch.turbine_id, "fault": diagnosis.fault,
            "risk": risk, "action": action["action"], "safety": safety["decision"],
        })

        result = {
            "schema_version": "wt-pm.platform.v1",
            "turbine_id": batch.turbine_id,
            "timestamp": int(batch.timestamps[-1]),
            "routing": routed,
            "what": {
                "anomaly_events": anomaly_events,
                "current_fused_score": None if fused is None else round(float(fused.score[-1]), 3),
                "alarm": None if fused is None else bool(fused.alarm[-1]),
                "conflicts": [] if fused is None else fused.conflicts,
                "fault": diagnosis.fault,
                "fault_confidence": round(diagnosis.confidence, 4),
                "votes": diagnosis.votes,
            },
            "where": {"subsystem": diagnosis.subsystem},
            "why": explanation,
            "severity": {
                "degradation_state": None if degradation is None or not degradation.ok
                else int(degradation.degradation_state[-1]),
                "state_confidence": None if degradation is None or not degradation.ok
                or degradation.uncertainty is None
                else round(1 - float(degradation.uncertainty[-1]), 4),
            },
            "rul": {
                "hours": None if rul_out is None or not rul_out.ok else round(float(rul_out.rul_hours[-1]), 1),
                "uncertainty_hours": None if rul_out is None or not rul_out.ok
                or rul_out.uncertainty is None else round(float(rul_out.uncertainty[-1]), 1),
                "source": None if rul_out is None else rul_out.model_id,
                **rul_info,
            },
            "risk_score": risk,
            "action": action,
            "safety": safety,
            "model_health": {
                "ran": sorted(set(rep_anom.outputs) | set(rep_clf.outputs)
                              | set(rep_deg.outputs) | set(rep_rul.outputs)
                              | set(rep_sup.outputs)
                              | ({rul_out.model_id} if rul_out is not None and rul_out.ok
                                 and rul_out.model_id in self.registry else set())),
                "errors": {**rep_anom.errors, **rep_clf.errors,
                           **rep_deg.errors, **rep_rul.errors, **rep_sup.errors,
                           **({"m16-particle-filter-rul": rul_info["particle_filter_error"]}
                              if "particle_filter_error" in rul_info else {})},
                "fallbacks": {**rep_anom.fallbacks_used, **rep_clf.fallbacks_used,
                              **rep_rul.fallbacks_used, **rep_sup.fallbacks_used},
                "timings_ms": {k: round(v, 1) for k, v in
                               {**rep_anom.timings_ms, **rep_clf.timings_ms,
                                **rep_deg.timings_ms, **rep_rul.timings_ms,
                                **rep_sup.timings_ms}.items()},
                "connected": sorted(self.registry.ids()),
                "details": self.registry.health_report(),
                "fitted": [mid for mid in self.registry.ids()
                           if self.registry.get(mid).fitted],
                "unavailable": [mid for mid in self.registry.ids()
                                if not self.registry.get(mid).available()],
            },
            "support": {
                mid: {
                    "ok": o.ok,
                    "explanation": o.explanation,
                    **({k: (v if not hasattr(v, "shape") else list(getattr(v, "shape", [])))
                        for k, v in (o.extra or {}).items()
                        if k in ("compression_ratio", "load_names", "tflite_path",
                                 "size_bytes", "status", "embeddings_shape",
                                 "turbine_ids", "input_dtype", "output_dtype", "shap_values",
                                 "source_model")}),
                }
                for mid, o in rep_sup.outputs.items()
            },
            "health_index": hidx,
            "sensor_quality": quality,
            "work_order": work_order,
            "cost_risk": costing,
            "alerts": {"raised_now": [a.__dict__ for a in new_alerts],
                       **self.alerts.snapshot()},
            "physics": physics,
            "what_if_derate_20pct": twin_derate,
            "series": series,
            "pipeline_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }
        hermes = self.hermes.run(batch, result)
        result["hermes"] = hermes.to_dict()
        result["why"]["narrative"] = hermes.final.get("why", {}).get("narrative")
        result["why"]["hermes_steps"] = hermes.to_dict()["n_steps"]
        problems = WTDataSchema.validate({
            "timestamp": result["timestamp"], "turbine_id": batch.turbine_id,
            "model_id": "wtpm-orchestrator",
        })
        result["contract_ok"] = not problems
        return result

    # -- risk & action ----------------------------------------------------------
    @staticmethod
    def _risk_score(fused, diagnosis, degradation, rul) -> float:
        """0..100: anomaly level, diagnosis confidence, severity, RUL urgency."""
        s = 0.0
        if fused is not None:
            s += 30 * min(float(fused.score[-1]) / 6.0, 1.0)
        if diagnosis.fault not in ("healthy",):
            s += 25 * diagnosis.confidence
        if degradation is not None and degradation.ok:
            s += 20 * float(degradation.degradation_state[-1]) / 3.0
        if rul is not None and rul.ok and rul.rul_hours is not None:
            r = float(rul.rul_hours[-1])
            s += 25 * max(0.0, 1.0 - r / 336.0)   # 2-week horizon
        return round(min(s, 100.0), 1)

    @staticmethod
    def _recommend(diagnosis, degradation, rul, risk: float, safety) -> Dict[str, str]:
        if safety["decision"] == "TRIP":
            return {"action": "stop_turbine", "priority": "immediate",
                    "detail": "; ".join(safety["reasons"])}
        if risk >= 70:
            return {"action": "schedule_inspection", "priority": "48h",
                    "detail": f"risk {risk}: {diagnosis.fault} in {diagnosis.subsystem}"}
        if risk >= 40 or safety["decision"] in ("DERATE", "INSPECT"):
            return {"action": "inspect_at_next_service", "priority": "2w",
                    "detail": f"risk {risk}; safety={safety['decision']}"}
        return {"action": "continue_monitoring", "priority": "routine",
                "detail": f"risk {risk}"}

    # -- fleet-level (m20) ---------------------------------------------------
    def analyse_fleet(self, batches: List[SensorBatch],
                      edge_index: np.ndarray) -> Optional[ModelOutput]:
        if "m20-gnn-cascade" not in self.registry:
            return None
        m20 = self.registry.get("m20-gnn-cascade")
        if not m20.available():
            return None
        scores: Dict[str, np.ndarray] = {}
        iso = self.registry.get("m14-isolation-forest")
        for b in batches:
            bb = self.prepare(b) if b.features is None else b
            if iso.available() and iso.fitted:
                out = iso.predict(bb)
                if out.ok:
                    scores[b.turbine_id] = out.anomaly_score
        if not m20.fitted:
            raise RuntimeError("fit m20 on a separate training record before fleet inference")
        return m20.predict_fleet(batches, scores, edge_index)
