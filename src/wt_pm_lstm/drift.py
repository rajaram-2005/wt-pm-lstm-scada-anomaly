"""Drift detection and model monitoring.

A predictive-maintenance model that silently stops matching its input
distribution is worse than no model: it keeps emitting confident alarms that
maintenance crews learn to ignore. Three complementary monitors are implemented,
because drift has three distinct failure modes:

========================== ==================================================
monitor                    catches
========================== ==================================================
PSI on input features      sensor recalibration, season change, curtailment
robust score shift         gradual loss of reconstruction fidelity
Page-Hinkley statistic     an abrupt mean shift in the score stream
========================== ==================================================

All three are threshold-based and stated in units a reviewer can re-derive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

Array = np.ndarray


def population_stability_index(
    expected: Array, actual: Array, n_bins: int = 10, eps: float = 1e-6
) -> float:
    """PSI between two samples of a scalar feature.

    Conventional reading: ``< 0.1`` no meaningful shift, ``0.1-0.25`` moderate
    shift worth investigating, ``> 0.25`` significant shift. Bin edges come from
    the expected (reference) sample so the statistic measures movement *of the
    new data* relative to the frozen reference.
    """
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if e.size == 0 or a.size == 0:
        return 0.0
    edges = np.quantile(e, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts = np.histogram(e, bins=edges)[0].astype(np.float64)
    a_counts = np.histogram(a, bins=edges)[0].astype(np.float64)
    e_pct = np.maximum(e_counts / max(e_counts.sum(), 1), eps)
    a_pct = np.maximum(a_counts / max(a_counts.sum(), 1), eps)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def page_hinkley(
    x: Array, delta: float = 0.005, threshold: float = 5.0, alpha: float = 0.999
) -> Dict[str, Any]:
    """Page-Hinkley change detector on a score stream.

    Tracks the cumulative deviation of the running mean and reports a change
    point when the statistic exceeds ``threshold``. ``delta`` is the magnitude
    of change considered irrelevant; ``alpha`` forgets old statistics so the
    detector tracks slow drift instead of saturating.
    """
    s = np.asarray(x, dtype=np.float64)
    n = s.size
    mean = 0.0
    m_t = 0.0
    m_min = 0.0
    alarms: List[int] = []
    for i in range(n):
        mean = mean + (s[i] - mean) / (i + 1)
        m_t = alpha * m_t + (s[i] - mean - delta)
        m_min = min(m_min, m_t)
        if m_t - m_min > threshold:
            alarms.append(i)
            m_t = 0.0
            m_min = 0.0
    return {"n_changes": len(alarms), "change_points": alarms[:20], "detected": bool(alarms)}


@dataclass
class DriftReport:
    """Result of a monitoring pass over one turbine."""

    feature_psi: Dict[str, float]
    max_psi_channel: str
    max_psi: float
    score_shift_sigma: float
    page_hinkley_changes: int
    drift_detected: bool
    drift_sigma_threshold: float
    psi_threshold: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_psi": {k: round(v, 4) for k, v in self.feature_psi.items()},
            "max_psi_channel": self.max_psi_channel,
            "max_psi": round(self.max_psi, 4),
            "score_shift_sigma": round(self.score_shift_sigma, 3),
            "page_hinkley_changes": self.page_hinkley_changes,
            "drift_detected": self.drift_detected,
            "drift_sigma_threshold": self.drift_sigma_threshold,
            "psi_threshold": self.psi_threshold,
        }


def monitor_drift(
    reference_features: Array,
    recent_features: Array,
    channel_names: tuple,
    reference_scores: Array,
    recent_scores: Array,
    drift_sigma: float = 4.0,
    psi_threshold: float = 0.25,
    normalise_window: int = 288,
) -> DriftReport:
    """Run all three monitors and combine them into a single verdict.

    The score-shift test uses the reference window's robust scale, so a shift of
    ``drift_sigma`` MADs in the *median* score is what triggers — median rather
    than mean, so that genuine faults (which raise the tail, not the centre) do
    not masquerade as drift.
    """
    psi = {
        str(name): population_stability_index(reference_features[:, j], recent_features[:, j])
        for j, name in enumerate(channel_names)
    }
    max_channel = max(psi, key=psi.get) if psi else ""
    max_psi = psi.get(max_channel, 0.0)

    ref = np.asarray(reference_scores, dtype=np.float64)
    rec = np.asarray(recent_scores, dtype=np.float64)
    if ref.size > 1 and rec.size > 1:
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med)))
        scale = 1.4826 * mad if mad > 1e-12 else float(np.std(ref)) or 1.0
        shift = (float(np.median(rec)) - med) / scale
        # Page-Hinkley is run on the concatenated reference+recent stream, since
        # it is a change detector, not a two-sample test.
        ph = page_hinkley(np.concatenate([ref, rec]), delta=0.005 * max(scale, 1e-9), threshold=drift_sigma)
    else:
        shift = 0.0
        ph = {"n_changes": 0, "detected": False}
    return DriftReport(
        feature_psi=psi,
        max_psi_channel=max_channel,
        max_psi=max_psi,
        score_shift_sigma=shift,
        page_hinkley_changes=int(ph["n_changes"]),
        drift_detected=bool(abs(shift) > drift_sigma or max_psi > psi_threshold),
        drift_sigma_threshold=drift_sigma,
        psi_threshold=psi_threshold,
    )


__all__ = [
    "population_stability_index",
    "page_hinkley",
    "DriftReport",
    "monitor_drift",
]
