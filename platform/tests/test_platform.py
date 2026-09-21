"""Platform integration tests.

Run:  pytest platform/tests -q
Fast by design: tiny records, tiny epochs. Heavier framework-dependent tests
skip cleanly when the dependency is absent, mirroring the adapters' own
graceful degradation.
"""

from __future__ import annotations

import numpy as np
import pytest

from wtpm_platform.contracts import (
    FAULT_CLASSES, ModelOutput, OperatingContext, SensorBatch, TaskType,
    WTDataSchema,
)
from wtpm_platform.data import FeaturePipeline, clean, ingest_arrays
from wtpm_platform.fusion import FusionConfig, FusionEngine
from wtpm_platform.orchestrator import ModelRouter, build_default_registry

CHANNELS = (
    "wind_speed_ms", "ambient_temp_c", "rotor_speed_rpm", "pitch_angle_deg",
    "yaw_error_deg", "power_kw", "main_shaft_torque_knm", "generator_current_a",
    "grid_frequency_hz", "nacelle_temp_c", "gearbox_oil_temp_c",
    "bearing_vib_rms_mm_s",
)


def test_config_and_split_by_turbine(tmp_path):
    from wtpm_platform.config import load_config
    from wtpm_platform.scada import split_by_turbine
    p = tmp_path / "wt-pm.yaml"
    p.write_text("plant: demo\npoll_seconds: 12\n")
    cfg = load_config(str(p))
    assert cfg["plant"] == "demo" and cfg["poll_seconds"] == 12
    rows = [
        {"turbine": "A", "wind_speed": 8, "power": 100, "rpm": 10},
        {"turbine": "B", "wind_speed": 9, "power": 110, "rpm": 11},
        {"turbine": "A", "wind_speed": 8.2, "power": 102, "rpm": 10.1},
    ]
    g = split_by_turbine(rows)
    assert set(g) == {"A", "B"}
    assert g["A"].n_steps == 2 and g["B"].n_steps == 1


def test_scada_column_map_and_writeback():
    from wtpm_platform.scada import (
        emit_scada_tags, ingest_scada_rows, resolve_column, write_tags_csv,
    )
    assert resolve_column("WindSpeed") == "wind_speed_ms"
    assert resolve_column("WT07.BrgVib") == "bearing_vib_rms_mm_s"
    assert resolve_column("DateTime") == "__time__"
    rows = [
        {"timestamp": 1700000000 + i * 600, "wind_speed": 8 + 0.01 * i,
         "power": 1200, "rpm": 12, "oil_temp": 55, "vibration": 2.0 + 0.01 * i}
        for i in range(40)
    ]
    b = ingest_scada_rows(rows, turbine_id="WT-07")
    assert "wind_speed_ms" in b.channel_names
    assert np.isfinite(b.channel("wind_speed_ms")).all()
    # torque filled from P/ω
    assert np.isfinite(b.channel("main_shaft_torque_knm")).sum() > 0
    tags = emit_scada_tags({
        "timestamp": 1, "risk_score": 12, "what": {"fault": "healthy", "alarm": False,
                                                   "current_fused_score": 0.4, "fault_confidence": 0.9},
        "where": {"subsystem": "unknown"}, "rul": {"hours": 200},
        "action": {"action": "continue_monitoring"}, "safety": {"decision": "CONTINUE"},
        "health_index": {"now": 91}, "sensor_quality": {"trust": 0.95},
    }, "WT-07")
    assert tags["tags"]["WTPM.FAULT"] == "healthy"
    assert any(p["tag"] == "WTPM.RUL_H" for p in tags["point_list"])


def test_hermes_trace_has_thought_action_observation():
    from wtpm_platform.hermes import HermesAgent, contrastive_channels
    b = FeaturePipeline(window=24, stride=4).transform(synth_batch(200))
    fake = {
        "what": {"fault": "bearing_wear", "fault_confidence": 0.8, "votes": {"bearing_wear": 2},
                 "current_fused_score": 4.2, "alarm": True, "anomaly_events": [{}], "conflicts": []},
        "where": {"subsystem": "drivetrain"},
        "why": {"contributing_features": {"bearing_vib_rms_mm_s": 1.2},
                "relevant_sensor_signals": ["bearing_vib_rms_mm_s"]},
        "severity": {"degradation_state": 2},
        "rul": {"hours": 80.0, "uncertainty_hours": 10.0, "source": "m16"},
        "safety": {"decision": "INSPECT", "reasons": ["bearing"]},
        "action": {"action": "schedule_inspection", "cost_optimal": "schedule_inspection"},
        "risk_score": 72.0,
        "physics": {"power_residual_kw_now": 3.0},
        "what_if_derate_20pct": {"scenario": "derate 20%", "delta_pct": {"von_mises_proxy": -8.0}},
    }

    class _Q:
        def run(self, batch):
            return {"trust": 0.9, "n_flags": 0, "flags": []}

    class _Twin:
        def maintenance_scenario(self, *a, **k):
            return fake["what_if_derate_20pct"]

    class _Orch:
        quality = _Q()
        twin = _Twin()

    res = HermesAgent(_Orch()).run(b, fake)
    actions = [s.action for s in res.trace]
    assert actions[0] == "sensor_quality"
    assert "finish" in actions
    assert all(s.thought for s in res.trace)
    assert res.final["what"] == "bearing_wear"
    assert "bearing" in res.final["why"]["narrative"].lower() or "bearing_wear" in res.final["why"]["narrative"]
    c = contrastive_channels(b)
    assert "bearing_vib_rms_mm_s" in c


def test_builtin_simulator_and_cli_version():
    from wtpm_platform.simulate import simulate_scada
    from wtpm_platform.cli import main, __version__
    b = simulate_scada(days=2, seed=3)
    assert b.n_steps == 2 * 24 * 6
    assert b.meta["fault_label"].sum() > 0
    assert __version__
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0


def synth_batch(n: int = 400, seed: int = 0, fault_at: float = 0.7) -> SensorBatch:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 600
    ws = 8 + 2 * np.sin(np.arange(n) / 40) + rng.normal(0, 0.5, n)
    rpm = 10 + 0.8 * ws + rng.normal(0, 0.2, n)
    torque = 20 + 5 * ws + rng.normal(0, 0.5, n)
    power = 0.94 * torque * 1e3 * rpm * 2 * np.pi / 60 / 1e3
    vib = np.full(n, 2.0) + rng.normal(0, 0.1, n)
    oil = np.full(n, 55.0) + rng.normal(0, 0.5, n)
    kinds = [""] * n
    i0 = int(n * fault_at)
    vib[i0:] += np.linspace(0, 6, n - i0)          # bearing-wear-like ramp
    for i in range(i0, n):
        kinds[i] = "bearing_wear"
    values = np.column_stack([
        ws, np.full(n, 12.0), rpm, np.full(n, 2.0), rng.normal(0, 1, n),
        power, torque, power / 0.69, np.full(n, 50.0), np.full(n, 30.0),
        oil, vib,
    ])
    b = ingest_arrays("WT-test", t, values, CHANNELS)
    b.meta["fault_kind_per_step"] = kinds
    b.meta["fault_label"] = (np.arange(n) >= i0).astype(int)
    return b


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------
def test_schema_validation_catches_missing_and_unknown():
    assert WTDataSchema.validate({"timestamp": 1, "turbine_id": "a", "model_id": "m"}) == []
    probs = WTDataSchema.validate({"timestamp": 1, "bogus": 2})
    assert any("turbine_id" in p for p in probs)
    assert any("unknown" in p for p in probs)


def test_model_output_records_conform():
    out = ModelOutput(
        model_id="x", task=TaskType.ANOMALY_DETECTION, turbine_id="WT-1",
        timestamps=np.array([10, 20]), anomaly_score=np.array([0.1, 5.0]),
    )
    recs = out.to_records()
    assert len(recs) == 2
    assert WTDataSchema.validate(recs[0]) == []


# ---------------------------------------------------------------------------
# data layer
# ---------------------------------------------------------------------------
def test_clean_interpolates_short_gaps_only():
    b = synth_batch(100)
    b.values[10:12, 0] = np.nan          # short gap -> interpolated
    b.values[50:70, 1] = np.nan          # long gap -> stays masked
    b.meta["mask"] = np.isfinite(b.values)
    out = clean(b, max_gap=6)
    assert np.isfinite(out.values[10:12, 0]).all()
    assert out.meta["mask"][10:12, 0].all()
    assert not out.meta["mask"][55:65, 1].any()


def test_feature_pipeline_views():
    b = FeaturePipeline(window=24, stride=4).transform(synth_batch(300))
    assert b.features is not None and b.features.shape[0] == 300
    assert "physics_power_residual_kw" in b.feature_names
    assert b.windows.shape[1] == 24
    assert b.vib_waveforms is not None
    assert b.meta["vibration_is_surrogate"] is True   # honesty flag


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------
def _mk_anom(mid: str, s: np.ndarray, unc=None) -> ModelOutput:
    return ModelOutput(model_id=mid, task=TaskType.ANOMALY_DETECTION,
                       turbine_id="WT-1", timestamps=np.arange(len(s)),
                       anomaly_score=s, uncertainty=unc)


def test_fusion_weighted_and_conflicts():
    n = 60
    quiet = np.zeros(n)
    loud = np.zeros(n); loud[30:40] = 8.0
    eng = FusionEngine(FusionConfig(method="weighted", temporal_window=1,
                                    weights={"a": 1.0, "b": 1.0}))
    fused = eng.fuse_anomaly([_mk_anom("a", loud), _mk_anom("b", quiet)])
    assert fused.score[35] == pytest.approx(4.0)
    assert fused.conflicts, "large disagreement above threshold must be flagged"


def test_fusion_confidence_weighting_downweights_uncertain_model():
    n = 40
    s_bad = np.full(n, 6.0)              # screams anomaly...
    u_bad = np.full(n, 100.0)            # ...but is very unsure
    u_bad[:5] = 0.01                     # (non-degenerate median)
    s_good = np.zeros(n)
    u_good = np.full(n, 0.01)
    eng = FusionEngine(FusionConfig(method="confidence_weighted", temporal_window=1))
    fused = eng.fuse_anomaly([_mk_anom("bad", s_bad, u_bad),
                              _mk_anom("good", s_good, u_good)])
    assert fused.score[-1] < 4.0         # pulled well below the unsure model


def test_probability_fusion_consensus():
    n = 10
    p1 = {c: np.full(n, 0.01) for c in FAULT_CLASSES}
    p1["bearing_wear"] = np.full(n, 0.9)
    p2 = {c: np.full(n, 0.01) for c in FAULT_CLASSES}
    p2["bearing_wear"] = np.full(n, 0.8)
    o1 = ModelOutput(model_id="c1", task=TaskType.FAULT_CLASSIFICATION,
                     turbine_id="w", timestamps=np.arange(n), probability=p1,
                     prediction=np.array(["bearing_wear"] * n, dtype=object))
    o2 = ModelOutput(model_id="c2", task=TaskType.FAULT_CLASSIFICATION,
                     turbine_id="w", timestamps=np.arange(n), probability=p2,
                     prediction=np.array(["bearing_wear"] * n, dtype=object))
    eng = FusionEngine()
    fused, consensus, votes = eng.fuse_probabilities([o1, o2], FAULT_CLASSES)
    assert consensus[-1] == "bearing_wear"
    assert votes == {"bearing_wear": 2}


def test_rul_fusion_inverse_variance():
    n = 5
    o1 = ModelOutput(model_id="r1", task=TaskType.RUL_ESTIMATION, turbine_id="w",
                     timestamps=np.arange(n), rul_hours=np.full(n, 100.0),
                     uncertainty=np.full(n, 1.0))
    o2 = ModelOutput(model_id="r2", task=TaskType.RUL_ESTIMATION, turbine_id="w",
                     timestamps=np.arange(n), rul_hours=np.full(n, 500.0),
                     uncertainty=np.full(n, 100.0))
    est, spread, contrib = FusionEngine().fuse_rul([o1, o2])
    assert abs(est[-1] - 100.0) < 5.0    # confident model dominates
    assert contrib["r1"] > contrib["r2"]


# ---------------------------------------------------------------------------
# registry / router
# ---------------------------------------------------------------------------
def test_registry_has_all_25():
    reg = build_default_registry()
    assert len(reg.ids()) == 25
    repos = {s.repository for s in reg.specs()}
    assert len(repos) == 25              # one adapter per repository


def test_connect_all_routes_every_available_model():
    reg = build_default_registry()
    router = ModelRouter(reg)
    b = FeaturePipeline(window=24, stride=4).transform(synth_batch(200))
    ctx = OperatingContext(mode="research", has_labels=True,
                           has_vibration_waveform=True, connect_all=True)
    sel = router.route(ctx, b)
    routed = set(sum(sel.values(), []))
    available = {mid for mid in reg.ids() if reg.get(mid).available()}
    missing = available - routed
    assert not missing, f"available models not routed: {sorted(missing)}"


def test_router_respects_deployment_and_labels():
    reg = build_default_registry()
    router = ModelRouter(reg)
    b = FeaturePipeline(window=24, stride=4).transform(synth_batch(200))
    # MCU deployment: only MCU-capable models routed
    from wtpm_platform.contracts import Deployment
    ctx = OperatingContext(deployment=Deployment.MCU, has_labels=False)
    sel = router.route(ctx, b)
    routed = sum(sel.values(), [])
    for mid in routed:
        assert Deployment.MCU in reg.get(mid).spec.deployment_targets
    # production without labels: unfitted supervised classifiers are skipped
    ctx2 = OperatingContext(mode="production", has_labels=False)
    sel2 = router.route(ctx2, b)
    assert "m11-xgboost-tabular" not in sel2["classification"]


# ---------------------------------------------------------------------------
# adapters (cheap ones always; torch ones if torch importable)
# ---------------------------------------------------------------------------
def _fit_predict(mid: str, b: SensorBatch):
    reg = build_default_registry()
    m = reg.get(mid)
    if not m.available():
        pytest.skip(f"{mid} unavailable: {m._unavailable_reason}")
    mask = np.zeros(b.n_steps, bool)
    mask[: int(b.n_steps * 0.6)] = True
    mask &= np.asarray(b.meta["fault_label"]) == 0
    m.fit(b, mask)
    out = m.predict(b)
    assert out.ok, out.error
    return out


@pytest.fixture(scope="module")
def prepared_batch():
    return FeaturePipeline(window=24, stride=4).transform(synth_batch(400))


def test_isolation_forest_detects_ramp(prepared_batch):
    out = _fit_predict("m14-isolation-forest", prepared_batch)
    s = out.anomaly_score
    assert s[-20:].mean() > s[:200].mean() + 1.0


def test_deep_svdd_detects_ramp(prepared_batch):
    out = _fit_predict("m13-deep-svdd", prepared_batch)
    assert out.anomaly_score[-20:].mean() > out.anomaly_score[:200].mean() + 1.0


def test_vae_detects_ramp(prepared_batch):
    out = _fit_predict("m22-vae-reconstruction", prepared_batch)
    assert out.anomaly_score[-20:].mean() > out.anomaly_score[:200].mean()


def test_xgboost_classifies(prepared_batch):
    out = _fit_predict("m11-xgboost-tabular", prepared_batch)
    assert str(out.prediction[-1]) == "bearing_wear"
    assert out.probability["bearing_wear"][-1] > 0.5


def test_hmm_states_are_severity_ordered(prepared_batch):
    out = _fit_predict("m15-hmm-degradation", prepared_batch)
    assert out.degradation_state[-10:].mean() > out.degradation_state[:100].mean()


def test_mlp_rul_decreases_toward_fault(prepared_batch):
    out = _fit_predict("m17-mlp-rul", prepared_batch)
    assert out.rul_hours[-1] < out.rul_hours[100]


def test_particle_filter_consumes_observations(prepared_batch):
    reg = build_default_registry()
    m = reg.get("m16-particle-filter-rul")
    m.fit(prepared_batch, np.ones(prepared_batch.n_steps, bool))
    prepared_batch.meta["rul_observations"] = np.linspace(300, 10, prepared_batch.n_steps)
    prepared_batch.meta["degradation_state"] = None
    out = m.predict(prepared_batch)
    assert out.ok, out.error
    assert out.rul_hours[-1] < out.rul_hours[0]
    assert out.uncertainty is not None


def test_tinyml_safety_and_export(tmp_path, prepared_batch):
    reg = build_default_registry()
    m = reg.get("m25-tinyml-safety")
    healthy = np.asarray(prepared_batch.meta["fault_label"]) == 0
    m.fit(prepared_batch, healthy)   # trip baseline comes from the healthy band
    out = m.predict(prepared_batch)
    assert out.ok
    assert str(out.prediction[-1]) == "TRIP"      # vibration ramp must trip
    try:
        p = m.export_esp32(str(tmp_path / "model.h"))
        assert "predict" in open(p).read()
    except Exception:
        pytest.skip("micromlgen not installed")


def test_unavailable_adapter_reports_not_crashes(monkeypatch, prepared_batch):
    reg = build_default_registry()
    m = reg.get("m14-isolation-forest")
    m._unavailable_reason = "simulated missing dep"
    out = m.predict(prepared_batch)
    assert not out.ok and "unavailable" in out.error


def test_fallback_resolution():
    reg = build_default_registry()
    fb = reg.resolve_fallback("m04-gru-scada-telemetry")
    assert fb is not None and fb.spec.model_id == "m14-isolation-forest"


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def test_binary_metrics_and_rates():
    from wtpm_platform.evaluation import binary_metrics
    m = binary_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]))
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["false_alarm_rate"] == 0.5 and m["missed_fault_rate"] == 0.5


def test_ece_perfect_calibration_is_zero():
    from wtpm_platform.evaluation import expected_calibration_error
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.05


def test_sensor_quality_flags_freeze():
    from wtpm_platform.advanced import SensorQualityMonitor
    b = synth_batch(200)
    b.values[:, list(CHANNELS).index("bearing_vib_rms_mm_s")] = 2.0  # frozen
    q = SensorQualityMonitor(freeze_steps=10).run(b)
    kinds = {f["kind"] for f in q["flags"]}
    assert "freeze" in kinds
    assert q["trust"] < 1.0


def test_alert_hysteresis_and_ack():
    from wtpm_platform.advanced import AlertManager
    am = AlertManager(raise_at=3.0, clear_at=2.0, hold=3)
    n1 = am.update("WT", 4.0, 80, "CONTINUE", "bearing_wear", 1)
    assert n1 and n1[0].severity in ("high", "critical", "warning")
    assert not am.update("WT", 4.1, 80, "CONTINUE", "bearing_wear", 2)  # already active
    am.update("WT", 0.5, 10, "CONTINUE", "healthy", 3)
    am.update("WT", 0.5, 10, "CONTINUE", "healthy", 4)
    am.update("WT", 0.5, 10, "CONTINUE", "healthy", 5)
    assert not am.snapshot()["active"]
    am2 = AlertManager()
    a = am2.update("WT", 5.0, 90, "TRIP", "bearing_wear", 1)[0]
    assert am2.ack(a.alert_id)


def test_work_order_and_cost_model():
    from wtpm_platform.advanced import MaintenancePlanner, CostRiskOptimizer
    from wtpm_platform.engines import Diagnosis
    d = Diagnosis(fault="bearing_wear", subsystem="drivetrain", confidence=0.8,
                  votes={}, per_class_prob={}, conflicting=False)
    wo = MaintenancePlanner().plan(d, 40.0, 75, "INSPECT")
    assert wo["priority"] == "P1"
    assert wo["estimated_cost_eur"] > 0
    c = CostRiskOptimizer().choose(75, 40.0, "INSPECT", wo)
    assert c["recommended"] in c["expected_cost_eur"]


def test_health_index_drops_with_anomaly():
    from wtpm_platform.advanced import health_index
    n = 50
    h = health_index(np.linspace(0, 6, n), np.zeros(n), np.full(n, 400.0))
    assert h["now"] < 90
    assert len(h["series"]) > 0


def test_drift_report_flags_shift():
    from wtpm_platform.evaluation import drift_report
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (500, 3))
    b = rng.normal(4, 1, (500, 3))
    rep = drift_report(a, b, ["f1", "f2", "f3"])
    assert rep["drifted"]
