"""Decision engines that sit on top of the model outputs.

AnomalyEngine     -> WHAT is happening (fused anomaly + alarm events)
DiagnosisEngine   -> WHAT fault + WHERE (subsystem) via consensus
UncertaintyEngine -> aggregate uncertainty + trust level
RULManager        -> HOW LONG until failure (fusion + particle filter)
XAIEngine         -> WHY (SHAP + evidence bundle)
SafetyManager     -> CAN it keep operating (TinyML gate + hard rules)
DigitalTwinInterface -> what-if / scenario analysis
EdgeInferenceManager -> cloud/edge/MCU deployment planning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wtpm_platform.base import ModelRegistry
from wtpm_platform.contracts import (
    Deployment, FAULT_CLASSES, FAULT_SUBSYSTEM, ModelOutput, SensorBatch,
    Subsystem, TaskType,
)
from wtpm_platform.fusion import FusedAnomaly, FusionEngine


# ---------------------------------------------------------------------------
# AnomalyEngine
# ---------------------------------------------------------------------------
class AnomalyEngine:
    def __init__(self, fusion: FusionEngine) -> None:
        self.fusion = fusion

    def run(self, outputs: Sequence[ModelOutput]) -> FusedAnomaly:
        return self.fusion.fuse_anomaly(outputs)

    @staticmethod
    def events(fused: FusedAnomaly, timestamps: np.ndarray) -> List[Dict[str, Any]]:
        """Contiguous alarm runs -> event records."""
        a = fused.alarm
        edges = np.flatnonzero(np.diff(a.astype(int)))
        starts = list(edges[::2] + 1) if a[0] == 0 else [0] + list(edges[1::2] + 1)
        events = []
        idx = np.flatnonzero(a)
        if len(idx) == 0:
            return events
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for r in runs:
            t0, t1 = int(r[0]), int(r[-1])
            events.append({
                "start_ts": int(timestamps[t0]), "end_ts": int(timestamps[t1]),
                "peak_score": float(fused.score[t0:t1 + 1].max()),
                "mean_disagreement": float(fused.disagreement[t0:t1 + 1].mean()),
                "duration_steps": t1 - t0 + 1,
            })
        return events


# ---------------------------------------------------------------------------
# DiagnosisEngine — WHAT fault, WHERE
# ---------------------------------------------------------------------------
@dataclass
class Diagnosis:
    fault: str
    subsystem: str
    confidence: float
    votes: Dict[str, int]
    per_class_prob: Dict[str, float]
    conflicting: bool
    evidence: Dict[str, Any] = field(default_factory=dict)


class DiagnosisEngine:
    def __init__(self, fusion: FusionEngine) -> None:
        self.fusion = fusion

    def run(
        self,
        clf_outputs: Sequence[ModelOutput],
        fused_anomaly: Optional[FusedAnomaly],
        at_step: int = -1,
    ) -> Diagnosis:
        ok = [o for o in clf_outputs if o.ok and o.probability]
        if not ok:
            # anomaly-only diagnosis: something is wrong but no classifier ran
            score = float(fused_anomaly.score[at_step]) if fused_anomaly is not None else 0.0
            return Diagnosis(
                fault="unknown_anomaly" if score > 3.0 else "healthy",
                subsystem=Subsystem.UNKNOWN.value,
                confidence=min(score / 6.0, 1.0) if score > 3.0 else 1.0 - score / 6.0,
                votes={}, per_class_prob={}, conflicting=False,
                evidence={"fused_anomaly_score": score},
            )
        fusedP, consensus, votes = self.fusion.fuse_probabilities(ok, FAULT_CLASSES)
        pc = {c: float(fusedP[c][at_step]) for c in FAULT_CLASSES}
        fault = max(pc, key=pc.get)
        conf = pc[fault]
        # conflict: two models voting for different non-healthy classes
        nz = [v for v in votes if v != "healthy"]
        conflicting = len(set(nz)) > 1
        # anomaly gate: classifiers can hallucinate a fault on healthy data;
        # require anomaly evidence before overriding 'healthy'
        if fused_anomaly is not None and fault != "healthy":
            if float(fused_anomaly.score[at_step]) < 1.0 and conf < 0.9:
                fault, conf = "healthy", pc.get("healthy", 1 - conf)
        subsystem = FAULT_SUBSYSTEM.get(fault, Subsystem.UNKNOWN).value
        return Diagnosis(fault=fault, subsystem=subsystem, confidence=conf,
                         votes=votes, per_class_prob=pc, conflicting=conflicting)


# ---------------------------------------------------------------------------
# UncertaintyEngine
# ---------------------------------------------------------------------------
class UncertaintyEngine:
    @staticmethod
    def aggregate(
        fused: Optional[FusedAnomaly],
        outputs: Sequence[ModelOutput],
        at_step: int = -1,
    ) -> Dict[str, float]:
        per_model = {}
        for o in outputs:
            if o.ok and o.uncertainty is not None and len(o.uncertainty):
                per_model[o.model_id] = float(np.asarray(o.uncertainty)[at_step])
        disagreement = float(fused.disagreement[at_step]) if fused is not None else 0.0
        # trust: high when models agree and individual uncertainties are low
        u_mean = float(np.mean(list(per_model.values()))) if per_model else 0.0
        trust = 1.0 / (1.0 + 0.5 * disagreement + 0.5 * u_mean)
        return {"model_disagreement": disagreement,
                "mean_model_uncertainty": u_mean,
                "trust": trust, **{f"u::{k}": v for k, v in per_model.items()}}


# ---------------------------------------------------------------------------
# RULManager
# ---------------------------------------------------------------------------
class RULManager:
    def __init__(self, registry: ModelRegistry, fusion: FusionEngine) -> None:
        self.registry = registry
        self.fusion = fusion

    def run(
        self,
        batch: SensorBatch,
        rul_outputs: Sequence[ModelOutput],
        degradation: Optional[ModelOutput],
    ) -> Tuple[Optional[ModelOutput], Dict[str, Any]]:
        """Fuse point RUL estimates, then refine through the particle filter."""
        point = [o for o in rul_outputs if o.ok and o.rul_hours is not None
                 and o.model_id != "m16-particle-filter-rul"]
        info: Dict[str, Any] = {"point_models": [o.model_id for o in point]}
        if not point:
            return None, {**info, "reason": "no RUL point estimates available"}
        fused_rul, spread, contrib = self.fusion.fuse_rul(point)
        info["fusion_contrib"] = contrib

        pf_out = None
        if "m16-particle-filter-rul" in self.registry:
            pf = self.registry.get("m16-particle-filter-rul")
            if pf.available():
                if not pf.fitted:
                    pf.fit(batch, np.ones(batch.n_steps, bool))
                batch.meta["rul_observations"] = fused_rul
                batch.meta["degradation_state"] = (
                    degradation.degradation_state if degradation is not None
                    and degradation.ok else None)
                pf_out = pf.predict(batch)
                if not pf_out.ok:
                    info["particle_filter_error"] = pf_out.error
                    pf_out = None
        if pf_out is None:
            pf_out = ModelOutput(
                model_id="fused-rul", task=TaskType.RUL_ESTIMATION,
                turbine_id=batch.turbine_id, timestamps=batch.timestamps,
                rul_hours=fused_rul, uncertainty=spread,
                explanation="inverse-variance fusion of point RUL models "
                            "(particle filter unavailable)",
            )
        return pf_out, info


# ---------------------------------------------------------------------------
# XAIEngine — WHY
# ---------------------------------------------------------------------------
class XAIEngine:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def explain(
        self,
        batch: SensorBatch,
        diagnosis: Diagnosis,
        clf_outputs: Sequence[ModelOutput],
        fused: Optional[FusedAnomaly],
        at_step: int = -1,
    ) -> Dict[str, Any]:
        bundle: Dict[str, Any] = {
            "predicted_fault": diagnosis.fault,
            "confidence": round(diagnosis.confidence, 4),
            "subsystem": diagnosis.subsystem,
            "models_responsible": [o.model_id for o in clf_outputs if o.ok],
            "conflicting_models": diagnosis.conflicting,
        }
        # anomaly evidence
        if fused is not None:
            step = at_step if at_step >= 0 else len(fused.score) + at_step
            bundle["anomaly_evidence"] = {
                "fused_score": round(float(fused.score[step]), 3),
                "alarm": bool(fused.alarm[step]),
                "per_model": {k: round(float(v[step]), 3) for k, v in fused.per_model.items()},
                "disagreement": round(float(fused.disagreement[step]), 3),
            }
        # SHAP attribution through m21 on the strongest tree classifier
        shap_attr: Dict[str, float] = {}
        if "m21-xai-shap" in self.registry:
            xai = self.registry.get("m21-xai-shap")
            if xai.available():
                for o in clf_outputs:
                    est = o.extra.get("estimator") if o.ok else None
                    if est is not None and hasattr(est, "predict_proba") and (
                            hasattr(est, "get_booster") or hasattr(est, "estimators_")):
                        try:
                            ci = FAULT_CLASSES.index(diagnosis.fault) if diagnosis.fault in FAULT_CLASSES else None
                            # map to model's class ordering if provided
                            shap_attr = xai.explain(
                                est, batch.features,
                                list(batch.feature_names),
                                row=len(batch.features) + at_step if at_step < 0 else at_step,
                                class_index=None,
                            )
                            bundle["shap_source_model"] = o.model_id
                            break
                        except Exception as exc:  # noqa: BLE001
                            bundle["shap_error"] = f"{type(exc).__name__}: {exc}"
        bundle["contributing_features"] = shap_attr
        # relevant sensor signals = channels behind top features
        chans = []
        for f in shap_attr:
            base = f.split("__")[0]
            if base in batch.channel_names and base not in chans:
                chans.append(base)
        bundle["relevant_sensor_signals"] = chans[:5]
        bundle["contrastive"] = self._contrastive(batch)
        bundle["counterfactual"] = self._counterfactual(batch, shap_attr)
        bundle["method"] = [
            "shap_tree_explainer (m21)",
            "contrastive healthy-band z",
            "counterfactual: restore top feature to healthy median",
        ]
        return bundle

    @staticmethod
    def _contrastive(batch: SensorBatch, frac: float = 0.4) -> Dict[str, Any]:
        n = batch.n_steps
        h = max(int(n * frac), 8)
        rows = []
        for ch in batch.channel_names:
            x = np.asarray(batch.channel(ch), float)
            med, sd = float(np.nanmedian(x[:h])), float(np.nanstd(x[:h]) + 1e-9)
            z = (float(x[-1]) - med) / sd
            rows.append((abs(z), ch, round(z, 3), round(med, 3), round(float(x[-1]), 3)))
        rows.sort(reverse=True)
        return {
            "top_deviations": [
                {"channel": ch, "z": z, "healthy_median": med, "now": now}
                for _, ch, z, med, now in rows[:6]
            ]
        }

    @staticmethod
    def _counterfactual(batch: SensorBatch, shap_attr: Dict[str, float]) -> Dict[str, Any]:
        """If we restored the strongest SHAP channel to its healthy median, what changes.

        This is a *feature-level* counterfactual (not a causal do-operator). It
        answers 'what would the operator look at first to return to normal'.
        """
        if not shap_attr or batch.features is None:
            return {"note": "no SHAP attributions available"}
        feat = next(iter(shap_attr))
        base = feat.split("__")[0]
        if base not in batch.channel_names:
            return {"feature": feat, "note": "not a raw channel"}
        x = np.asarray(batch.channel(base), float)
        h = max(len(x) // 3, 8)
        return {
            "feature": feat,
            "channel": base,
            "now": round(float(x[-1]), 3),
            "healthy_median": round(float(np.nanmedian(x[:h])), 3),
            "restore_delta": round(float(np.nanmedian(x[:h]) - x[-1]), 3),
            "claim": f"returning {base} to its healthy median is the first counterfactual lever",
        }


# ---------------------------------------------------------------------------
# SafetyManager — CAN the turbine continue operating?
# ---------------------------------------------------------------------------
class SafetyManager:
    """TinyML tree gate (m25) + non-negotiable hard limits.

    The hard limits exist because a learned tree must never be the only thing
    between a turbine and a burst bearing.
    """

    HARD_LIMITS = {"bearing_vib_rms_mm_s": 12.0, "gearbox_oil_temp_c": 78.0, "rotor_speed_rpm": 25.0}

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def evaluate(self, batch: SensorBatch, diagnosis: Diagnosis,
                 rul: Optional[ModelOutput]) -> Dict[str, Any]:
        decision = "CONTINUE"
        reasons: List[str] = []
        # Missing/invalid safety telemetry must never authorize CONTINUE.
        for ch in self.HARD_LIMITS:
            idx = list(batch.channel_names).index(ch) if ch in batch.channel_names else None
            quality = batch.meta.get("mask")
            if (idx is None or not np.isfinite(batch.values[-1, idx]) or
                    (quality is not None and not quality[-1, idx])):
                decision = "INSPECT"
                reasons.append(f"safety telemetry unavailable: {ch}; manual assessment required")
        # hard limits first
        for ch, lim in self.HARD_LIMITS.items():
            if ch in batch.channel_names and float(batch.channel(ch)[-1]) > lim:
                decision = "TRIP"
                reasons.append(f"hard limit: {ch}={batch.channel(ch)[-1]:.1f} > {lim}")
        # TinyML gate
        tiny_trip = False
        if "m25-tinyml-safety" in self.registry:
            m25 = self.registry.get("m25-tinyml-safety")
            if m25.available() and m25.fitted:
                out = m25.predict(batch)
                if out.ok and str(out.prediction[-1]) == "TRIP":
                    tiny_trip = True
                    reasons.append("TinyML safety tree voted TRIP")
        if tiny_trip and decision != "TRIP":
            decision = "DERATE"     # learned trip without hard-limit backing => derate
        # RUL-based derating
        if rul is not None and rul.ok and rul.rul_hours is not None:
            r = float(rul.rul_hours[-1])
            if r < 24 and decision == "CONTINUE":
                decision = "DERATE"
                reasons.append(f"fused RUL {r:.0f} h < 24 h")
        if diagnosis.fault in ("bearing_wear",) and diagnosis.confidence > 0.8 and decision == "CONTINUE":
            decision = "INSPECT"
            reasons.append("high-confidence bearing wear diagnosis")
        return {"decision": decision, "reasons": reasons,
                "hard_limits": self.HARD_LIMITS}

    def export_mcu_artifacts(self, out_dir: str) -> Dict[str, str]:
        arts = {}
        if "m25-tinyml-safety" in self.registry:
            m25 = self.registry.get("m25-tinyml-safety")
            if m25.available() and m25.fitted:
                try:
                    arts["esp32_header"] = m25.export_esp32(f"{out_dir}/model.h")
                except Exception as exc:  # noqa: BLE001
                    arts["esp32_header_error"] = str(exc)
        return arts


# ---------------------------------------------------------------------------
# DigitalTwinInterface — what-if / scenarios
# ---------------------------------------------------------------------------
class DigitalTwinInterface:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def _twin(self):
        if "m19-digital-twin" not in self.registry:
            return None
        t = self.registry.get("m19-digital-twin")
        return t if (t.available() and t.fitted) else None

    def what_if(self, batch: SensorBatch, overrides: Dict[str, float]) -> Dict[str, Any]:
        twin = self._twin()
        if twin is None:
            return {"error": "digital twin unavailable or unfitted"}
        base = twin.predict(batch)
        alt = twin.what_if(batch, overrides)
        b = base.extra["loads"]
        a = alt["loads"]
        names = base.extra["load_names"]
        return {
            "overrides": overrides,
            "baseline_mean": {n: float(b[:, i].mean()) for i, n in enumerate(names)},
            "scenario_mean": {n: float(a[:, i].mean()) for i, n in enumerate(names)},
            "delta_pct": {n: float((a[:, i].mean() - b[:, i].mean())
                                   / (abs(b[:, i].mean()) + 1e-9) * 100) for i, n in enumerate(names)},
        }

    def maintenance_scenario(self, batch: SensorBatch, rul: Optional[ModelOutput],
                             derate_pct: float = 20.0) -> Dict[str, Any]:
        """Predicted effect of derating on loads and (heuristically) on RUL."""
        twin = self._twin()
        if twin is None:
            return {"error": "digital twin unavailable or unfitted"}
        rpm = float(np.mean(batch.channel("rotor_speed_rpm")))
        res = self.what_if(batch, {"rotor_speed_rpm": rpm * (1 - derate_pct / 100)})
        out = {"scenario": f"derate {derate_pct:.0f}%", **res}
        if rul is not None and rul.ok and rul.rul_hours is not None:
            stress_gain = -res["delta_pct"].get("von_mises_proxy", 0.0) / 100.0
            r = float(rul.rul_hours[-1])
            out["rul_now_hours"] = r
            out["rul_extension_estimate_hours"] = r * max(stress_gain, 0.0) * 1.5
            out["note"] = "extension estimate is a heuristic from the stress-proxy delta, not a validated life model"
        return out


# ---------------------------------------------------------------------------
# EdgeInferenceManager — Cloud -> Edge -> MCU planning
# ---------------------------------------------------------------------------
class EdgeInferenceManager:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def deployment_plan(self) -> Dict[str, List[str]]:
        plan: Dict[str, List[str]] = {d.value: [] for d in Deployment}
        for spec in self.registry.specs():
            for d in spec.deployment_targets:
                plan[d.value].append(spec.model_id)
        return plan

    def plan_for(self, deployment: Deployment) -> List[str]:
        """Model ids that may legitimately run on the given tier.

        A large model is never pushed to the MCU merely because it exists —
        only models whose spec lists the tier are eligible.
        """
        return [s.model_id for s in self.registry.specs()
                if deployment in s.deployment_targets]

    def export_edge_bundle(self, batch: SensorBatch, out_dir: str) -> Dict[str, Any]:
        """Quantize the small edge net via m24's recipe + export m25's header."""
        arts: Dict[str, Any] = {}
        if "m24-quantized-edge" in self.registry:
            m24 = self.registry.get("m24-quantized-edge")
            if m24.available():
                try:
                    import tensorflow as tf
                    feats = batch.features.astype(np.float32)
                    net = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(feats.shape[1],)),
                        tf.keras.layers.Dense(16, activation="relu"),
                        tf.keras.layers.Dense(1, activation="sigmoid"),
                    ])
                    hi = (feats[:, 0] > np.percentile(feats[:, 0], 90)).astype(np.float32)
                    net.compile(optimizer="adam", loss="binary_crossentropy")
                    net.fit(feats, hi, epochs=2, batch_size=64, verbose=0)
                    arts["edge_net"] = m24.quantize(net, feats, f"{out_dir}/edge_net_int8.tflite")
                except Exception as exc:  # noqa: BLE001
                    arts["edge_net_error"] = f"{type(exc).__name__}: {exc}"
            else:
                arts["edge_net_error"] = "m24 unavailable (tensorflow missing)"
        return arts
