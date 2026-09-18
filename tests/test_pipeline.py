"""End-to-end protocol invariants on a deliberately tiny run.

The point is not to measure detection quality — that takes minutes and belongs
in ``wtpm demo``. The point is that the protocol cannot be silently violated:
faults must not reach training, calibration or the scaler; the same config must
produce the same numbers; a saved bundle must score identically to the object it
came from; and the reports must carry the caveats.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from wt_pm_lstm.config import RunConfig, config_fingerprint
from wt_pm_lstm.pipeline import (
    build_fault_schedule,
    fit_detector,
    generate_record,
    operating_point_table,
    run_experiment,
)
from wt_pm_lstm.registry import load_detector, save_detector
from wt_pm_lstm.windows import split_indices


@pytest.fixture(scope="module")
def tiny_cfg():
    cfg = RunConfig()
    cfg.name = "pytest-tiny"
    cfg.data.n_days = 12
    cfg.data.test_lead_days = 0.5
    cfg.model.epochs = 3
    cfg.model.n_ensemble = 1
    cfg.model.stride = 4
    cfg.detect.bootstrap_samples = 12
    cfg.evaluation.bootstrap_samples = 12
    return cfg


@pytest.fixture(scope="module")
def tiny_run(tiny_cfg):
    return run_experiment(tiny_cfg)


def test_fault_schedule_marks_a_healthy_lead_in(tiny_cfg):
    n = int(tiny_cfg.data.n_days * 24 * 60 / tiny_cfg.data.sample_minutes)
    test_start = int(n * 0.6)
    specs = build_fault_schedule(tiny_cfg, n, test_start, seed=1)
    assert specs, "a 12-day record must still fit the whole fault schedule"
    assert min(s.onset_step for s in specs) >= test_start
    # Every family appears exactly once, and none overlaps the calibration band.
    kinds = [s.kind for s in specs]
    assert sorted(kinds) == sorted(tiny_cfg.data.fault_kinds)
    for spec in specs:
        assert spec.onset_step + spec.duration_steps <= n


def test_training_and_calibration_bands_contain_no_faults(tiny_cfg):
    """The core leakage guarantee: the model never sees a labelled fault."""
    n = int(tiny_cfg.data.n_days * 24 * 60 / tiny_cfg.data.sample_minutes)
    plan = split_indices(n, tiny_cfg.data.healthy_fraction, tiny_cfg.data.calibration_fraction, None, tiny_cfg.model.window)
    timeline = generate_record(tiny_cfg, plan)
    label = timeline.fault_label
    assert label is not None
    assert not label[plan.train[0] : plan.train[1]].any()
    assert not label[plan.calibration[0] : plan.calibration[1]].any()
    assert label[plan.test[0] : plan.test[1]].any()
    # The split planner refuses to build a fault-contaminated split at all.
    with pytest.raises(ValueError):
        split_indices(n, 0.9, 0.05, label, tiny_cfg.model.window)


def test_run_metrics_are_complete_and_bounded(tiny_run):
    metrics = tiny_run.metrics
    for key in ("point", "point_adjusted", "event", "pr_auc", "roc_auc", "f1_ci",
                "false_alarms_per_day", "alarm_minutes_per_day"):
        assert key in metrics
    for key in ("precision", "recall", "f1", "specificity"):
        assert 0.0 <= metrics["point"][key] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["n_samples"] > 0
    summary = tiny_run.summary()
    assert summary["threshold"] > 0
    assert summary["n_parameters"] > 0
    # Point-adjusted metrics may flatter but never reduce recall below raw.
    assert metrics["point_adjusted"]["recall"] >= metrics["point"]["recall"] - 1e-12


def test_repeated_runs_are_reproducible(tiny_cfg):
    first = run_experiment(tiny_cfg)
    second = run_experiment(tiny_cfg)
    assert first.fingerprint == second.fingerprint == config_fingerprint(tiny_cfg)
    assert first.summary()["threshold"] == pytest.approx(second.summary()["threshold"], rel=1e-9)
    np.testing.assert_allclose(first.detection.score, second.detection.score, rtol=1e-9, atol=1e-12)


def test_fingerprint_changes_when_the_protocol_changes(tiny_cfg):
    other = RunConfig()
    other.model.hidden_size = tiny_cfg.model.hidden_size + 8
    assert config_fingerprint(other) != config_fingerprint(tiny_cfg)


def test_saved_bundle_scores_identically(tiny_run, tmp_path):
    prefix = os.path.join(tmp_path, "det")
    save_detector(tiny_run.detector, prefix)
    restored = load_detector(prefix)
    test = tiny_run.timeline.slice(*tiny_run.plan.test)
    a = tiny_run.detector.score_timeline(test, stride=4)
    b = restored.score_timeline(test, stride=4)
    np.testing.assert_allclose(a.score, b.score, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(a.alarm, b.alarm)
    assert restored.threshold.value == pytest.approx(tiny_run.detector.threshold.value)
    assert restored.drift_scale.shape == tiny_run.detector.drift_scale.shape


def test_threshold_override_changes_only_the_decision(tiny_run):
    """The sweep must not perturb the score, only the alarm decision."""
    test = tiny_run.timeline.slice(*tiny_run.plan.test)
    base = tiny_run.detector.score_timeline(test, stride=4)
    loose = tiny_run.detector.score_timeline(test, stride=4, threshold_override=float(base.threshold) * 3.0)
    np.testing.assert_allclose(base.score, loose.score, rtol=0, atol=0)
    assert loose.alarm.sum() <= base.alarm.sum()
    assert loose.threshold == pytest.approx(float(base.threshold) * 3.0)


def test_operating_point_table_is_monotone_in_alarm_time(tiny_run):
    rows = operating_point_table(tiny_run.detector, tiny_run.timeline, tiny_run.plan,
                                 multipliers=(0.5, 1.0, 2.0))
    assert [r["threshold_multiplier"] for r in rows] == [0.5, 1.0, 2.0]
    minutes = [r["alarm_minutes_per_day"] for r in rows]
    # Raising the threshold can only reduce alarm time (assertions inside rows
    # are the deployed decision path, not a reimplementation).
    assert minutes[0] >= minutes[-1]
    assert all(0.0 <= r["f1"] <= 1.0 for r in rows)


def test_records_are_event_level_and_explainable(tiny_run):
    records = tiny_run.detection.records
    assert records, "a run with injected faults must emit at least one record"
    for record in records:
        assert record.alarm is True
        assert record.recommended_action
        assert record.explanation
        payload = record.to_dict()
        assert payload["schema_version"]
        json.dumps(payload)  # must be JSON-serialisable for the API contract
        assert record.attribution, "a record must name the channels it blames"


def test_report_states_its_limitations(tiny_run):
    from wt_pm_lstm.report import markdown_report

    text = markdown_report(tiny_run)
    assert "does *not* establish" in text
    assert "simulated" in text
    assert "false alarm time" in text
    assert "total alarm time" in text
    # The operating point is presented as evidence, never as a selection rule.
    assert "not a selection procedure" in text
    assert tiny_run.summary()["fingerprint"] in text


def test_early_stopping_checkpoint_is_actually_used(tiny_cfg):
    """The restored best epoch must be the weights the model runs with.

    A regression test for a silent failure: early stopping copied the best
    parameter *arrays* into the parameter dict, which rebound the dict entries
    while the RNN cells kept the last epoch's weights. Training reported a
    best epoch, the bundle reported a best epoch, and inference used neither.
    """
    from wt_pm_lstm.train import train_model

    n = 60
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(n, 6, 4))
    masks = np.ones_like(windows)
    weights = np.ones(4)
    cfg = RunConfig().model
    cfg.epochs = 4
    cfg.patience = 1
    model, history = train_model(
        cfg, windows, masks, weights, np.random.default_rng(1), seed=2,
        channel_names=("wind_speed_ms", "ambient_temp_c", "grid_frequency_hz", "power_kw"),
    )
    # The invariant that makes the restore meaningful.
    assert model.params["enc.W_ih"] is model.enc.W_ih
    assert model.params["dec.W_hh"] is model.dec.W_hh
    assert model.params["W_recon"] is not None
    assert history.best_epoch <= len(history.train_loss)
    # And the deployed weights are finite and non-trivial.
    assert np.isfinite(model.params["enc.W_ih"]).all()
    assert np.abs(model.params["enc.W_ih"]).sum() > 0.0


def test_warmup_guard_suppresses_cold_start_alarms():
    """A batch that begins mid-record must not alarm on its own warm-up.

    The causal window statistics and the EWMA are undefined for the first few
    samples of any batch, so a backfill or a batch API call used to report an
    alarm in its first hour purely because the score had not settled.
    """
    import numpy as np

    from wt_pm_lstm.api import ScoringService

    cfg = RunConfig()
    cfg.data.n_days = 8
    cfg.model.epochs = 2
    cfg.model.n_ensemble = 1
    cfg.detect.bootstrap_samples = 5
    service = ScoringService(cfg=cfg)

    warmup = max(cfg.model.window, cfg.detect.trend_window)
    healthy = service.sample(n_samples=288)
    timeline = service._timeline_from_payload(healthy)
    result = service.detector.score_timeline(timeline, stride=1)
    assert not result.alarm[:warmup].any(), "cold-start samples must not alarm"
    assert result.components["warmup"][:warmup].all()
    assert not result.components["warmup"][warmup:].any()


def test_sampled_batch_has_history_behind_it():
    """The demo batch is the tail of a longer record, not a cold start."""
    from wt_pm_lstm.api import ScoringService

    cfg = RunConfig()
    cfg.data.n_days = 8
    cfg.model.epochs = 2
    cfg.model.n_ensemble = 1
    service = ScoringService(cfg=cfg)
    batch = service.sample(fault="bearing_wear", n_samples=288)
    assert len(batch["samples"]) == 288
    assert batch["injected_fault"] == "bearing_wear"
    assert 0 < batch["fault_starts_at_sample"] < 288
    # The fault is inside the returned window, and every channel is present.
    assert all(len(s) >= 13 for s in batch["samples"])
