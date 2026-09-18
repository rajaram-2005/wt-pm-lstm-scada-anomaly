"""End-to-end orchestration: simulate (or load) -> fit -> calibrate -> score.

This module is the single place where the experiment protocol lives, so the CLI,
the API and the tests all execute exactly the same sequence of steps:

1. build the time-ordered train / calibration / test plan,
2. generate (or load) a labelled record whose faults start after the test band
   opens, leaving a healthy lead-in,
3. fit the scaler on the training band only,
4. train the ensemble on training windows only,
5. fit the residual normaliser and calibrate the threshold on the calibration
   band only,
6. score the test band and evaluate against ground truth,
7. run the drift monitors.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wt_pm_lstm.config import RunConfig, config_fingerprint
from wt_pm_lstm.detect import (
    AlarmStateMachine,
    DetectionResult,
    Detector,
    ResidualNormaliser,
    channel_reduce,
    ewma,
    member_residuals,
)
from wt_pm_lstm.drift import DriftReport, monitor_drift
from wt_pm_lstm.evaluate import evaluate_detection, events_from_labels
from wt_pm_lstm.registry import model_id
from wt_pm_lstm.schema import SCHEMA_VERSION, TurbineTimeline
from wt_pm_lstm.simulate import (
    SENSOR_FAULT_TARGETS,
    FaultSpec,
    default_fault_schedule,
    simulate_fleet,
    simulate_timeline,
)
from wt_pm_lstm.train import TrainHistory, train_ensemble
from wt_pm_lstm.windows import ChannelScaler, SplitPlan, make_windows, prepare_timeline, split_indices

Array = np.ndarray


@dataclass
class ExperimentResult:
    """Everything a run produces, in memory."""

    cfg: RunConfig
    detector: Detector
    plan: SplitPlan
    histories: List[TrainHistory]
    detection: DetectionResult
    metrics: Dict[str, Any]
    drift: DriftReport
    timeline: TurbineTimeline
    seconds: float
    fingerprint: str = ""
    model_id: str = ""
    #: Optional sensitivity/false-alarm table (see :func:`operating_point_table`).
    sweep: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        point = self.metrics.get("point", {})
        return {
            "run": self.cfg.name,
            "model_id": self.model_id,
            "fingerprint": self.fingerprint,
            "cell": self.cfg.model.cell,
            "n_parameters": self.detector.models[0].n_params,
            "ensemble": len(self.detector.models),
            "threshold": self.detector.threshold.value,
            "f1": point.get("f1"),
            "precision": point.get("precision"),
            "recall": point.get("recall"),
            "pr_auc": self.metrics.get("pr_auc"),
            "roc_auc": self.metrics.get("roc_auc"),
            "event_recall": self.metrics.get("event", {}).get("recall"),
            "median_latency_samples": self.metrics.get("event", {}).get("median_latency_samples"),
            "false_alarms_per_day": self.metrics.get("false_alarms_per_day"),
            "drift_detected": self.drift.drift_detected,
            "seconds": round(self.seconds, 2),
        }


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_fault_schedule(cfg: RunConfig, n_steps: int, test_start: int, seed: Optional[int] = None) -> List[FaultSpec]:
    """Schedule faults across the whole test band, after a healthy lead-in.

    Faults are *spread over the test period* rather than packed at its start.
    Packing them made the evaluation measure the detector's first few days only;
    spreading them means recall and latency are averaged over the whole band,
    including periods far from the calibration window — which is where a
    non-stationary model actually degrades.
    """
    rng = np.random.default_rng(cfg.data.seed if seed is None else seed)
    steps_per_day = 24 * 60 / cfg.data.sample_minutes
    kinds = list(cfg.data.fault_kinds)
    hi = n_steps - 6
    # Shrink the healthy lead-in when the record is short: a fixed multi-day
    # lead would leave no room for events at all and the run would silently
    # evaluate on a fault-free band.
    lead = int(cfg.data.test_lead_days * steps_per_day)
    lead = min(lead, max(0, (hi - test_start) // 4))
    lo = test_start + lead
    if hi - lo < 12 * len(kinds):
        return default_fault_schedule(n_steps, min(lo, hi - 1), kinds=kinds, seed=seed or cfg.data.seed)

    # Evenly spaced slots across the test band, jittered so events do not align
    # with any periodic artefact of the simulator.
    usable = hi - lo
    slot = usable / len(kinds)
    specs: List[FaultSpec] = []
    for i, kind in enumerate(kinds):
        duration = int(rng.integers(48, 180))
        jitter = int(rng.integers(0, max(1, int(slot * 0.5))))
        onset = int(lo + i * slot) + jitter
        if onset + duration >= hi:
            duration = max(24, hi - onset - 1)
        channel = ""
        if kind in SENSOR_FAULT_TARGETS:
            channel = str(rng.choice(SENSOR_FAULT_TARGETS[kind]))
        specs.append(
            FaultSpec(
                kind=kind,
                onset_step=onset,
                duration_steps=duration,
                magnitude=float(rng.uniform(0.7, 1.3)),
                channel=channel,
            )
        )
    return specs


def generate_record(cfg: RunConfig, plan: SplitPlan) -> TurbineTimeline:
    """Simulate one labelled turbine record matching the split plan."""
    specs = build_fault_schedule(cfg, plan.n_steps, plan.test[0])
    return simulate_timeline(cfg.data, fault_specs=specs)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_detector(cfg: RunConfig, timeline: TurbineTimeline, verbose: bool = False) -> Tuple[Detector, SplitPlan, List[TrainHistory]]:
    """Fit scaler, ensemble, residual normaliser and threshold.

    Only the train and calibration bands of ``timeline`` are touched; the caller
    is responsible for evaluating on the test band afterwards. If the timeline
    carries labels, :func:`~wt_pm_lstm.windows.split_indices` rejects any split
    that would contain a fault.
    """
    values, mask = prepare_timeline(timeline)
    window = cfg.model.window
    plan = split_indices(
        n_steps=values.shape[0],
        healthy_fraction=cfg.data.healthy_fraction,
        calibration_fraction=cfg.data.calibration_fraction,
        fault_label=timeline.fault_label,
        window=window,
    )
    # Shrink the training band so no window straddles into calibration.
    train_stop = plan.train[1] - window + 1
    # The scaler may be fitted on train + calibration, because both bands are
    # fault-free by construction and the test band is untouched. Fitting it on
    # the (shorter) training band alone leaves the operating envelope half
    # covered, which is how pitched operation ends up looking anomalous.
    #
    # The *model* is still trained on the training band only: if it saw the
    # calibration band, calibration residuals would be optimistically small and
    # the threshold would be set too low, producing false alarms in operation.
    scaler_stop = plan.calibration[1]
    scaler = ChannelScaler.fit(
        values[:scaler_stop], mask[:scaler_stop], timeline.channel_names
    )
    z = scaler.transform(values)

    train_windows = make_windows(z, mask, window, cfg.model.stride, 0, train_stop)
    if verbose:
        print(
            f"  training windows: {len(train_windows)} "
            f"(band [{0}, {train_stop}), mask coverage {train_windows.mask.mean():.4%})"
        )
    models, histories = train_ensemble(
        cfg, train_windows.values, train_windows.mask, timeline.channel_names, verbose=verbose
    )

    # Residual normaliser: fitted on the calibration band, never on the test band.
    cal_windows = make_windows(
        z, mask, window, stride=2, start=plan.calibration[0], stop=plan.calibration[1]
    )
    recon, pred = member_residuals(models, np.ascontiguousarray(cal_windows.values))
    residuals = np.concatenate(
        [recon.reshape(-1, recon.shape[-1]), pred.reshape(-1, pred.shape[-1])], axis=0
    )
    normaliser = ResidualNormaliser.fit(residuals, timeline.channel_names)

    # Normal-behaviour model: fitted on the same healthy band as the scaler
    # (train + calibration), never on the test band.
    from wt_pm_lstm.baseline import NormalBehaviourModel
    from wt_pm_lstm.schema import CONTEXT_CHANNELS
    from wt_pm_lstm.train import score_channel_mask

    nbm = NormalBehaviourModel.fit(
        values[:scaler_stop],
        mask[:scaler_stop],
        timeline.channel_names,
        CONTEXT_CHANNELS,
        score_channel_mask(cfg.model, timeline.channel_names),
    )
    detector = Detector(
        cfg=cfg,
        models=models,
        nbm=nbm,
        scaler=scaler,
        residual_normaliser=normaliser,
        threshold=_placeholder_threshold(),
        channel_names=timeline.channel_names,
        metadata={
            "model_id": model_id(cfg),
            "fingerprint": config_fingerprint(cfg),
            "schema_version": SCHEMA_VERSION,
        },
    )

    rng = np.random.default_rng(cfg.model.offline_seed)
    detector.calibrate(
        timeline.slice(*plan.calibration),
        quantile=cfg.detect.threshold_quantile,
        rng=rng,
        bootstrap_samples=cfg.detect.bootstrap_samples,
    )
    if verbose:
        print(
            f"  threshold={detector.threshold.value:.4f} "
            f"({detector.threshold.quantile:.4f} quantile, "
            f"95% CI [{detector.threshold.ci_low:.4f}, {detector.threshold.ci_high:.4f}])"
        )
    return detector, plan, histories


def _placeholder_threshold():
    from wt_pm_lstm.detect import Threshold

    return Threshold(
        value=float("inf"),
        quantile=0.0,
        exceedance_rate=0.0,
        ci_low=float("inf"),
        ci_high=float("inf"),
        n_calibration=0,
        method="uncalibrated",
    )


# --------------------------------------------------------------------------
# Full experiment
# --------------------------------------------------------------------------
def run_experiment(
    cfg: RunConfig,
    timeline: Optional[TurbineTimeline] = None,
    verbose: bool = False,
) -> ExperimentResult:
    """Run the complete protocol and return every artefact."""
    t0 = time.perf_counter()
    fingerprint = config_fingerprint(cfg)
    if timeline is None:
        n_steps = int(cfg.data.n_days * 24 * 60 / cfg.data.sample_minutes)
        plan = split_indices(
            n_steps,
            cfg.data.healthy_fraction,
            cfg.data.calibration_fraction,
            fault_label=None,
            window=cfg.model.window,
        )
        timeline = generate_record(cfg, plan)
    detector, plan, histories = fit_detector(cfg, timeline, verbose=verbose)

    test = timeline.slice(*plan.test)
    detection = detector.score_timeline(test, stride=cfg.model.stride)
    kinds = _event_kinds(test)
    metrics = evaluate_detection(
        y_true=test.fault_label if test.fault_label is not None else np.zeros(test.n_steps, dtype=bool),
        alarm=detection.alarm,
        score=detection.score,
        sample_seconds=int(timeline.sampling_seconds or 600),
        tolerance=cfg.evaluation.detection_tolerance,
        kinds=kinds,
        bootstrap_samples=cfg.evaluation.bootstrap_samples,
        alpha=cfg.evaluation.alpha,
        rng=np.random.default_rng(cfg.model.offline_seed + 1),
        recovery_samples=cfg.detect.clear_after,
    )
    drift = _drift_check(cfg, detector, timeline, plan, detection)
    sweep = (
        operating_point_table(detector, timeline, plan, multipliers=cfg.evaluation.sweep_multipliers, verbose=verbose)
        if cfg.evaluation.sweep_multipliers
        else []
    )
    return ExperimentResult(
        cfg=cfg,
        detector=detector,
        plan=plan,
        histories=histories,
        detection=detection,
        metrics=metrics,
        drift=drift,
        timeline=timeline,
        seconds=time.perf_counter() - t0,
        fingerprint=fingerprint,
        model_id=model_id(cfg),
        sweep=sweep,
    )


def operating_point_table(
    detector: Detector,
    timeline: TurbineTimeline,
    plan: SplitPlan,
    multipliers: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Score the test band once and re-threshold it, to expose the trade-off.

    A single threshold hides the decision that produced it. Re-running the alarm
    policy at several multipliers of the calibrated threshold shows what an
    operator buys — recall and event recall — and what they pay for it in alarm
    time, which is the comparison that should drive the operating point. The
    score itself does not depend on the threshold, so this is cheap: one forward
    pass, several state machines.
    """
    cfg = detector.cfg
    test = timeline.slice(*plan.test)
    y_true = test.fault_label if test.fault_label is not None else np.zeros(test.n_steps, dtype=bool)
    kinds = _event_kinds(test)
    sample_seconds = int(timeline.sampling_seconds or 600)
    base = float(detector.threshold.value)
    rows: List[Dict[str, Any]] = []
    for m in multipliers:
        value = base * float(m)
        # The production path is re-run per multiplier, so the sensor-health
        # rule and the hysteresis are included exactly as deployed. Re-deriving
        # the decision here instead would silently diverge from it.
        detection = detector.score_timeline(test, stride=cfg.model.stride, threshold_override=value)
        alarm = detection.alarm
        metrics = evaluate_detection(
            y_true=y_true,
            alarm=alarm,
            score=detection.score,
            sample_seconds=sample_seconds,
            tolerance=cfg.evaluation.detection_tolerance,
            kinds=kinds,
            bootstrap_samples=0,
            rng=np.random.default_rng(cfg.model.offline_seed + 2),
            recovery_samples=cfg.detect.clear_after,
        )
        rows.append(
            {
                "threshold_multiplier": float(m),
                "threshold": value,
                "precision": metrics["point"]["precision"],
                "recall": metrics["point"]["recall"],
                "f1": metrics["point"]["f1"],
                "event_recall": metrics["event"]["recall"],
                "alarm_minutes_per_day": metrics["alarm_minutes_per_day"],
                "false_alarms_per_day": metrics["false_alarms_per_day"],
                "median_latency_samples": metrics["event"]["median_latency_samples"],
            }
        )
        if verbose:
            print(
                f"    x{m:<4g} thr={value:8.2f} F1={rows[-1]['f1']:.3f} R={rows[-1]['recall']:.3f} "
                f"event R={rows[-1]['event_recall']:.2f} alarm={rows[-1]['alarm_minutes_per_day']:7.1f} min/day"
            )
    return rows


def _event_kinds(timeline: TurbineTimeline) -> List[str]:
    """Fault kind for each labelled event, in onset order (for per-kind recall)."""
    kinds_map = timeline.meta.get("fault_kind_per_step") or []
    events = events_from_labels(timeline.fault_label) if timeline.fault_label is not None else []
    out = []
    for a, _b in events:
        kind = str(kinds_map[a]) if a < len(kinds_map) else ""
        out.append(kind or "unlabelled")
    return out


def _drift_check(
    cfg: RunConfig,
    detector: Detector,
    timeline: TurbineTimeline,
    plan: SplitPlan,
    detection: DetectionResult,
) -> DriftReport:
    """Compare the calibration band with the first two days of the test band.

    The comparison window deliberately excludes the later test band, which
    contains real faults: a fault is not drift, and conflating them would make
    the monitor useless.
    """
    values, mask = prepare_timeline(timeline)
    z = detector.scaler.transform(values)
    cal = slice(*plan.calibration)
    n_recent = min(
        int(2 * 24 * 60 / cfg.data.sample_minutes),
        max(1, plan.test[1] - plan.test[0] - 1),
    )
    recent = slice(plan.test[0], plan.test[0] + n_recent)
    valid_cal = mask[cal].all(axis=1)
    valid_rec = mask[recent].all(axis=1)
    return monitor_drift(
        reference_features=z[cal][valid_cal],
        recent_features=z[recent][valid_rec],
        channel_names=timeline.channel_names,
        reference_scores=_calibration_scores(cfg, detector, timeline, plan),
        recent_scores=detection.score[:n_recent],
        drift_sigma=cfg.detect.drift_sigma,
        normalise_window=cfg.detect.normalise_window,
    )


def _calibration_scores(
    cfg: RunConfig, detector: Detector, timeline: TurbineTimeline, plan: SplitPlan
) -> Array:
    """Score trace of the calibration band (reference distribution for drift)."""
    result = detector.score_timeline(timeline.slice(*plan.calibration), stride=2)
    covered = result.count > 0
    return result.score[covered]


# --------------------------------------------------------------------------
# Fleet
# --------------------------------------------------------------------------
def run_fleet_experiment(cfg: RunConfig, n_turbines: int = 6, verbose: bool = False) -> Dict[str, Any]:
    """Run one detector per turbine over a wake-coupled farm.

    The per-turbine scores are what model 08 (graph layer) consumes: it needs to
    see that two adjacent turbines alarmed together to decide whether a fault is
    local or farm-wide. This function therefore returns the raw score matrix
    rather than only the metrics.
    """
    t0 = time.perf_counter()
    timelines = simulate_fleet(cfg.data, n_turbines=n_turbines)
    n_steps = timelines[0].n_steps
    plan = split_indices(
        n_steps, cfg.data.healthy_fraction, cfg.data.calibration_fraction, None, cfg.model.window
    )
    results: List[Dict[str, Any]] = []
    scores = np.zeros((n_turbines, plan.test[1] - plan.test[0]))
    alarms = np.zeros_like(scores, dtype=bool)
    for idx, timeline in enumerate(timelines):
        detector, _, _ = fit_detector(cfg, timeline, verbose=verbose)
        test = timeline.slice(*plan.test)
        detection = detector.score_timeline(test)
        scores[idx] = detection.score
        alarms[idx] = detection.alarm
        metrics = evaluate_detection(
            y_true=test.fault_label,
            alarm=detection.alarm,
            score=detection.score,
            sample_seconds=int(timeline.sampling_seconds or 600),
            tolerance=cfg.evaluation.detection_tolerance,
            kinds=_event_kinds(test),
            bootstrap_samples=max(50, cfg.evaluation.bootstrap_samples // 4),
        )
        results.append(
            {
                "turbine_id": timeline.turbine_id,
                "wake_deficit_mean": timeline.meta.get("wake_deficit_mean"),
                "metrics": metrics,
                "threshold": detector.threshold.value,
            }
        )
    return {
        "n_turbines": n_turbines,
        "seconds": time.perf_counter() - t0,
        "per_turbine": results,
        "score_matrix": scores.tolist(),
        "alarm_matrix": alarms.astype(int).tolist(),
        "timestamps": timelines[0].timestamps[plan.test[0] :].tolist(),
        "cross_turbine_alarm_correlation": _alarm_correlation(alarms),
    }


def _alarm_correlation(alarms: Array) -> float:
    """Mean pairwise correlation of alarm traces — the graph layer's trigger."""
    n = alarms.shape[0]
    if n < 2:
        return 0.0
    a = alarms.astype(np.float64)
    std = a.std(axis=1)
    valid = std > 0
    if valid.sum() < 2:
        return 0.0
    corr = np.corrcoef(a[valid])
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(np.nanmean(corr[iu]))


__all__ = [
    "ExperimentResult",
    "build_fault_schedule",
    "generate_record",
    "fit_detector",
    "run_experiment",
    "run_fleet_experiment",
]
