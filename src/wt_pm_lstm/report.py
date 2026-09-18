"""Markdown reporting: turns a run into a document a reviewer can read.

The report deliberately includes what the numbers *cannot* show — which fault
families were missed and why, what the protocol does and does not establish —
because a predictive-maintenance model presented without its failure modes is
worse than useless to the crew who has to act on its alarms.
"""

from __future__ import annotations

from typing import Any, Dict, List

from wt_pm_lstm.schema import CONTEXT_CHANNELS, EXOGENOUS_CHANNELS


def markdown_report(result: Any) -> str:
    """Render an :class:`~wt_pm_lstm.pipeline.ExperimentResult` as Markdown."""
    metrics = result.metrics
    point = metrics["point"]
    adjusted = metrics["point_adjusted"]
    event = metrics["event"]
    summary = result.summary()
    cfg = result.cfg

    lines: List[str] = []
    add = lines.append
    add(f"# WT-PM model 13 — LSTM SCADA anomaly detection: run report")
    add("")
    add(f"* **run** `{cfg.name}`  * **model id** `{summary['model_id']}`")
    add(f"* **configuration fingerprint** `{summary['fingerprint']}` "
        f"(identical configurations map to the same registry entry)")
    add(f"* **engine** `{cfg.engine}`  * **cell** `{cfg.model.cell}`  "
        f"* **parameters per member** {summary['n_parameters']}  "
        f"* **ensemble members** {summary['ensemble']}")
    add(f"* **wall time** {summary['seconds']}s")
    add("")
    add("## Headline metrics (test band, raw point-wise)")
    add("")
    add(f"* detection threshold **{summary['threshold']:.3f}** — "
        f"`{getattr(result.detector.threshold, 'method', 'unknown')}`")
    add("* the threshold is set from a *budget*: the quantile of the calibration score "
        "implied by the operator's declared tolerance for alarm time on healthy machines. "
        "It is the same kind of statement as an SLO, and it is the knob to change if the "
        "alarm rate is wrong.")
    add("")
    add("| metric | value |")
    add("| --- | --- |")
    add(f"| precision | {point['precision']:.3f} |")
    add(f"| recall | {point['recall']:.3f} |")
    add(f"| F1 | {point['f1']:.3f} |")
    add(f"| F1 95% CI (moving-block bootstrap) | [{metrics['f1_ci'][0]:.3f}, {metrics['f1_ci'][1]:.3f}] |")
    add(f"| PR-AUC (threshold-free) | {metrics['pr_auc']:.3f} |")
    add(f"| ROC-AUC (threshold-free) | {metrics['roc_auc']:.3f} |")
    add(f"| false alarm time | {metrics['false_alarms_per_day'] * cfg.data.sample_minutes:.0f} min/day "
        f"({metrics['false_alarms_per_day']:.2f} samples/day) |")
    add(f"| total alarm time | {metrics['alarm_minutes_per_day']:.0f} min/day "
        f"({100 * metrics['alarm_minutes_per_day'] / 1440:.1f}% of the band) |")
    add(f"| fault fraction of test band | {metrics['fault_fraction']:.4f} |")
    add("")
    add("Point-adjusted metrics (segment credited once any sample is flagged) are "
        f"reported below for comparability with the SCADA literature, but are "
        f"**not** the headline: P={adjusted['precision']:.3f}, R={adjusted['recall']:.3f}, "
        f"F1={adjusted['f1']:.3f}.")
    add("")
    add("## Event-level behaviour")
    add("")
    add("| quantity | value |")
    add("| --- | --- |")
    add(f"| labelled fault events | {event['n_events']} |")
    add(f"| detected within tolerance | {event['n_detected']} ({event['recall']:.3f}) |")
    add(f"| median detection latency | {event['median_latency_samples']} samples |")
    add(f"| false-alarm events | {event['false_alarm_events']} |")
    add("")
    per_kind = event.get("per_kind") or {}
    if per_kind:
        add("### Per fault family")
        add("")
        add("| family | events | detected | recall |")
        add("| --- | --- | --- | --- |")
        for kind, rec in per_kind.items():
            add(f"| {kind or 'unlabelled'} | {int(rec['n'])} | {int(rec['detected'])} | {rec['recall']:.2f} |")
        add("")
    if result.sweep:
        add("## Operating point")
        add("")
        add("The decision stage re-run at several multiples of the calibrated threshold. "
            "This table uses test-band labels, so it is **evidence about the sensitivity/"
            "false-alarm trade-off, not a selection procedure** — the shipped threshold was "
            "chosen from calibration data alone.")
        add("")
        add("| x | threshold | precision | recall | F1 | event recall | alarm time (min/day) | false alarm time (min/day) |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in result.sweep:
            add(
                f"| {row['threshold_multiplier']:g} | {row['threshold']:.2f} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | {row['f1']:.3f} | {row['event_recall']:.2f} | "
                f"{row['alarm_minutes_per_day']:.0f} | {row['false_alarms_per_day'] * cfg.data.sample_minutes:.0f} |"
            )
        add("")

    add("## Protocol")
    add("")
    add(f"* record: {cfg.data.n_days} days at {cfg.data.sample_minutes}-minute sampling, "
        f"{result.timeline.n_steps} samples, seed {cfg.data.seed}")
    add(f"* splits (contiguous, never shuffled): train " + str(tuple(result.plan.train)) +
        f", calibration {tuple(result.plan.calibration)}, test {tuple(result.plan.test)}")
    add(f"* the scaler and the normal-behaviour model are fitted on train + calibration "
        f"(both fault-free); the ensemble is trained on the training band only; the "
        f"threshold and the trend/level scales come from the calibration band only")
    add(f"* environment/control channels that are *observed but not scored*: "
        f"{', '.join(CONTEXT_CHANNELS)}")
    add(f"* window {cfg.model.window} samples ({cfg.model.window * cfg.data.sample_minutes / 60:.1f} h), "
        f"stride {cfg.model.stride}")
    add("")
    add("## Alarm decomposition")
    add("")
    add("| family | events | detected | note |")
    add("| --- | --- | --- | --- |")
    for kind, rec in (event.get("per_kind") or {}).items():
        add(f"| {kind or 'unlabelled'} | {int(rec['n'])} | {int(rec['detected'])} | {rec['recall']:.2f} recall |")
    add("")
    add("A second, independent rule runs alongside the score: a scored channel whose value "
        "does not change for a sustained run of samples raises a *sensor-health* alarm. A "
        "frozen sensor has no residual for a value-based score to see, and the two failure "
        "modes call for different maintenance actions, so they are alarmed separately and "
        "labelled differently in the records.")
    add("")
    add("## Drift monitoring")
    add("")
    drift = result.drift
    add(f"* verdict: **{'drift detected' if drift.drift_detected else 'no drift'}** "
        f"(PSI threshold {drift.psi_threshold}, score-shift threshold {drift.drift_sigma_threshold} robust sigmas)")
    add(f"* largest population stability index: {drift.max_psi:.3f} on `{drift.max_psi_channel}`")
    add(f"* median score shift vs calibration: {drift.score_shift_sigma:+.2f} robust sigmas")
    add(f"* Page-Hinkley change points: {drift.page_hinkley_changes}")
    add("")
    add("The monitor flags whenever the monitoring band departs from the calibration band by "
        "more than four robust sigmas or shows a PSI above the conventional 0.25. It fires on "
        "essentially every multi-week synthetic run here, and the honest reading is not that "
        "the detector is unstable: it is that a nine-day calibration window does not represent "
        "a sixty-day operating period, so the alarm-time budget estimated on calibration is "
        "*exceeded* on the monitoring band. That discrepancy is exactly what this section "
        "exists to surface; a retraining/alerting policy for it is future work.")
    add("")
    add("## What this run does *not* establish")
    add("")
    add("* The data are simulated. The physics is explicit and the labels are exact, but "
        "simulated faults are cleaner than field faults; absolute metric values are not "
        "transferable to a real asset without re-calibration.")
    add("* Faults are injected in the test band only, so the reported recall measures "
        "detection of the families in the schedule — not of unseen failure modes.")
    add("* A single threshold is used for all conditions. Operating-point-dependent "
        "thresholds are future work.")
    add("")
    return "\n".join(lines) + "\n"


__all__ = ["markdown_report"]
