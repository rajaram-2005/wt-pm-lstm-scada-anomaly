"""Evaluation protocol shared by the WT-PM models.

Reporting choices that are easy to get wrong in anomaly detection:

* **Raw point metrics first.** Point-adjusted F1 (where every sample inside a
  labelled anomaly segment is credited if *any* sample in the segment is
  flagged) is reported, but second and clearly labelled, because it makes weak
  detectors look strong. Raw F1 is the headline number.
* **Rank metrics alongside threshold metrics.** PR-AUC and ROC-AUC measure the
  score ordering independently of the threshold, so a detector is not judged on
  a single lucky quantile.
* **Event-level recall and latency.** A fault that is detected 3 days late is a
  different operational outcome from one detected 2 hours late, even at equal
  point F1.
* **False alarms per day**, not just precision: that is the number a
  maintenance team actually feels.
* **Block bootstrap CIs.** Metrics are computed on autocorrelated series, so
  confidence intervals resample contiguous blocks (whole days), not individual
  samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------
# Core metric primitives
# --------------------------------------------------------------------------
def confusion(y_true: Array, y_pred: Array) -> Dict[str, int]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def point_metrics(y_true: Array, y_pred: Array) -> Dict[str, float]:
    c = confusion(y_true, y_pred)
    p, r, f = prf(c["tp"], c["fp"], c["fn"])
    return {
        **{k: float(v) for k, v in c.items()},
        "precision": p,
        "recall": r,
        "f1": f,
        "specificity": c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else 0.0,
    }


def point_adjusted_metrics(y_true: Array, y_pred: Array) -> Dict[str, float]:
    """Point-adjusted metrics (with the standard caveat).

    Every sample of a true anomaly segment counts as detected once the detector
    flags *any* sample inside that segment. This rewards segment-level
    detection but cannot distinguish early from late alarms, and inflates F1 on
    long faults — hence it is reported as a secondary metric only.
    """
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool).copy()
    idx = np.flatnonzero(np.diff(np.concatenate([[0], yt.view(np.int8), [0]])))
    for a, b in zip(idx[0::2], idx[1::2]):
        if yp[a:b].any():
            yp[a:b] = True
    return point_metrics(yt, yp)


def average_precision(y_true: Array, score: Array) -> float:
    """PR-AUC via the step-wise precision/recall integral (no interpolation bias)."""
    yt = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    if yt.sum() == 0:
        return float("nan")
    order = np.argsort(-s)
    yt_sorted = yt[order]
    tp = np.cumsum(yt_sorted)
    fp = np.cumsum(~yt_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(yt.sum()), 1)
    # Sum over ranked positions where a positive was retrieved.
    ap = 0.0
    prev_recall = 0.0
    for i in range(yt_sorted.size):
        if yt_sorted[i]:
            ap += precision[i] * (recall[i] - prev_recall)
            prev_recall = recall[i]
    return float(ap)


def roc_auc(y_true: Array, score: Array) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties get average rank)."""
    yt = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    n_pos, n_neg = int(yt.sum()), int((~yt).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Average ranks within tie groups.
    sorted_s = s[order]
    i = 0
    while i < sorted_s.size:
        j = i
        while j + 1 < sorted_s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    return float((ranks[yt].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# Event-level metrics
# --------------------------------------------------------------------------
@dataclass
class EventMetrics:
    """Detection behaviour per labelled fault event."""

    n_events: int
    n_detected: int
    recall: float
    latencies: List[int]
    median_latency: Optional[float]
    missed: List[int]
    false_alarm_events: int
    per_kind: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_events": self.n_events,
            "n_detected": self.n_detected,
            "recall": self.recall,
            "median_latency_samples": self.median_latency,
            "latencies": self.latencies,
            "missed_onsets": self.missed,
            "false_alarm_events": self.false_alarm_events,
            "per_kind": self.per_kind,
        }


def events_from_labels(y_true: Array) -> List[Tuple[int, int]]:
    yt = np.asarray(y_true, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], yt.view(np.int8), [0]])))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def event_metrics(
    y_true: Array,
    alarm: Array,
    tolerance: int,
    sample_seconds: int,
    kinds: Sequence[str] = (),
    times: Optional[Array] = None,
) -> EventMetrics:
    """Score alarm events against labelled fault events.

    An event counts as detected if the first alarm falls within
    ``[onset, onset + tolerance]``; a later alarm still credits the *point*
    metrics but not the latency metric, so "detected but too late to act" is
    visible rather than hidden.
    """
    yt = np.asarray(y_true, dtype=bool)
    al = np.asarray(alarm, dtype=bool)
    events = events_from_labels(yt)
    latencies: List[int] = []
    missed: List[int] = []
    per_kind: Dict[str, Dict[str, float]] = {}
    n_detected = 0
    for ei, (a, b) in enumerate(events):
        window_end = min(yt.size, a + tolerance + 1)
        flagged = np.flatnonzero(al[a:window_end])
        if flagged.size:
            n_detected += 1
            latencies.append(int(flagged[0]))
        else:
            missed.append(a)
        if kinds and ei < len(kinds):
            kind = kinds[ei]
            per_kind.setdefault(kind, {"n": 0.0, "detected": 0.0})
            per_kind[kind]["n"] += 1
            per_kind[kind]["detected"] += 1.0 if flagged.size else 0.0
    for kind, rec in per_kind.items():
        rec["recall"] = rec["detected"] / rec["n"] if rec["n"] else 0.0

    # False-alarm events: alarm runs that do not overlap any labelled fault.
    fa = 0
    for a, b in events_from_labels(al):
        if not yt[a:b].any():
            fa += 1
    return EventMetrics(
        n_events=len(events),
        n_detected=n_detected,
        recall=n_detected / len(events) if events else float("nan"),
        latencies=latencies,
        median_latency=float(np.median(latencies)) if latencies else None,
        missed=missed,
        false_alarm_events=fa,
        per_kind=per_kind,
    )


def false_alarms_per_day(
    alarm: Array,
    y_true: Array,
    sample_seconds: int,
    recovery_samples: int = 0,
) -> float:
    """False-positive *samples* per day — the operator-facing alarm-time rate.

    This is a duration, not a count: ``k`` samples per day at a ten-minute
    sampling interval is ``10k`` minutes per day of alarm time, which is the
    quantity a maintenance team actually spends. Event counts are reported
    separately in :class:`EventMetrics`.

    ``recovery_samples`` excludes the tail of a fault event, because the alarm
    policy *requires* the alarm to persist after the fault ends (hysteresis:
    ``clear_after`` samples). Counting that deliberate persistence as a false
    positive charges every real detection twice and makes the rate depend on how
    many faults the schedule happens to contain.
    """
    al = np.asarray(alarm, dtype=bool)
    yt = np.asarray(y_true, dtype=bool)
    if recovery_samples > 0:
        inflated = yt.copy()
        for a, b in events_from_labels(yt):
            inflated[a : min(yt.size, b + int(recovery_samples))] = True
        yt = inflated
    fp = int(np.sum(al & ~yt))
    days = al.size * sample_seconds / 86400.0
    return fp / days if days > 0 else float("nan")


def alarm_minutes_per_day(alarm: Array, sample_seconds: int) -> float:
    """Total alarm time per day, including time correctly spent in alarm.

    Paired with :func:`false_alarms_per_day` this separates "how much alarm time
    does the system produce" from "how much of it was unnecessary".
    """
    al = np.asarray(alarm, dtype=bool)
    days = al.size * sample_seconds / 86400.0
    return float(np.sum(al) * sample_seconds / 60.0 / days) if days > 0 else float("nan")


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def block_bootstrap_ci(
    y_true: Array,
    alarm: Array,
    metric: str = "f1",
    block: int = 144,
    samples: int = 400,
    rng: Optional[np.random.Generator] = None,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Confidence interval for a point metric via moving-block bootstrap.

    Contiguous blocks (default 144 samples = one day at 10-minute sampling)
    preserve the autocorrelation structure that i.i.d. resampling destroys.
    """
    rng = rng or np.random.default_rng(0)
    yt = np.asarray(y_true, dtype=bool)
    al = np.asarray(alarm, dtype=bool)
    n = yt.size
    block = int(max(1, min(block, n)))
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, max(1, n - block + 1))
    out = np.empty(samples)
    for s in range(samples):
        idx = np.concatenate([np.arange(i, i + block) for i in rng.choice(starts_pool, n_blocks)])[:n]
        m = point_metrics(yt[idx], al[idx])
        out[s] = m[metric]
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


# --------------------------------------------------------------------------
# Full report
# --------------------------------------------------------------------------
def evaluate_detection(
    y_true: Array,
    alarm: Array,
    score: Array,
    sample_seconds: int,
    tolerance: int = 18,
    kinds: Sequence[str] = (),
    bootstrap_samples: int = 400,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    recovery_samples: int = 0,
) -> Dict[str, Any]:
    """Complete evaluation report for one detection run.

    ``recovery_samples`` should be the alarm policy's hold time
    (``DetectConfig.clear_after``) so that the false-alarm rate does not count
    the hysteresis tail after a genuine detection.
    """
    yt = np.asarray(y_true, dtype=bool)
    point = point_metrics(yt, alarm)
    adjusted = point_adjusted_metrics(yt, alarm)
    events = event_metrics(yt, alarm, tolerance, sample_seconds, kinds)
    if bootstrap_samples and bootstrap_samples > 0:
        f1_lo, f1_hi = block_bootstrap_ci(
            yt, alarm, "f1", samples=bootstrap_samples, rng=rng, alpha=alpha
        )
    else:
        # Sweeps that only rank operating points skip the resampling; an empty
        # bootstrap would otherwise raise deep inside numpy.
        f1_lo = f1_hi = float("nan")
    return {
        "point": point,
        "point_adjusted": adjusted,
        "event": events.to_dict(),
        "pr_auc": average_precision(yt, score),
        "roc_auc": roc_auc(yt, score),
        "false_alarms_per_day": false_alarms_per_day(
            alarm, yt, sample_seconds, recovery_samples=recovery_samples
        ),
        "alarm_minutes_per_day": alarm_minutes_per_day(alarm, sample_seconds),
        "f1_ci": [f1_lo, f1_hi],
        "n_samples": int(yt.size),
        "fault_fraction": float(yt.mean()),
        "note": (
            "point_adjusted metrics credit a whole fault segment once any sample is "
            "flagged and must not be reported without the raw point metrics."
        ),
    }


__all__ = [
    "confusion",
    "prf",
    "point_metrics",
    "point_adjusted_metrics",
    "average_precision",
    "roc_auc",
    "event_metrics",
    "EventMetrics",
    "events_from_labels",
    "false_alarms_per_day",
    "alarm_minutes_per_day",
    "block_bootstrap_ci",
    "evaluate_detection",
]
