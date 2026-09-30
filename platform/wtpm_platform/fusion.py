"""FusionEngine — configurable combination of the 25 models' outputs.

Implements: weighted ensemble, probability fusion, anomaly-score fusion,
Bayesian fusion (where probabilistic outputs exist), temporal consensus,
model-confidence weighting, uncertainty-aware decisions, majority/consensus
diagnosis and conflict detection. No model is assumed universally superior:
weights are either supplied by configuration or learned from validation
performance in research mode (see evaluation.select_weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wtpm_platform.contracts import ModelOutput, TaskType
from wtpm_platform.protocol import validate_output

def _aligned(outputs):
    reference = outputs[0]
    for out in outputs:
        validate_output(out)
        if out.turbine_id != reference.turbine_id or not np.array_equal(out.timestamps, reference.timestamps):
            raise ValueError("fusion streams must match turbine and timestamps")
    if len({o.model_id for o in outputs}) != len(outputs):
        raise ValueError("duplicate fusion model IDs")



@dataclass
class FusionConfig:
    method: str = "confidence_weighted"     # weighted | confidence_weighted | max | median
    weights: Dict[str, float] = field(default_factory=dict)  # model_id -> weight
    temporal_window: int = 9                 # steps of temporal consensus (EWMA-ish)
    alarm_threshold: float = 3.0             # robust-z units
    conflict_margin: float = 2.0             # score disagreement flagged as conflict
    min_models: int = 1


@dataclass
class FusedAnomaly:
    score: np.ndarray                        # (T,) fused robust z
    alarm: np.ndarray                        # (T,) bool
    per_model: Dict[str, np.ndarray]
    disagreement: np.ndarray                 # (T,) std across models
    conflicts: List[Dict]                    # detected conflicts
    contributors: Dict[str, float]           # model -> effective weight


class FusionEngine:
    def __init__(self, cfg: Optional[FusionConfig] = None) -> None:
        self.cfg = cfg or FusionConfig()

    # ------------------------------------------------------------------
    # anomaly-score fusion
    # ------------------------------------------------------------------
    def fuse_anomaly(self, outputs: Sequence[ModelOutput]) -> FusedAnomaly:
        cfg = self.cfg
        streams: Dict[str, np.ndarray] = {}
        uncertainties: Dict[str, np.ndarray] = {}
        valid = [o for o in outputs if o.ok and o.anomaly_score is not None]
        if valid:
            _aligned(valid)
        for o in valid:
            streams[o.model_id] = np.asarray(o.anomaly_score, float)
            if o.uncertainty is not None:
                uncertainties[o.model_id] = np.asarray(o.uncertainty, float)
        if len(streams) < cfg.min_models:
            raise RuntimeError(f"fusion needs >= {cfg.min_models} anomaly streams, got {len(streams)}")

        M = np.vstack(list(streams.values()))            # (K, T)
        ids = list(streams.keys())

        # ---- weights ----
        if cfg.method == "max":
            fused = M.max(axis=0)
            eff = {i: 1.0 / len(ids) for i in ids}
        elif cfg.method == "median":
            fused = np.median(M, axis=0)
            eff = {i: 1.0 / len(ids) for i in ids}
        else:
            w = np.array([cfg.weights.get(i, 1.0) for i in ids], float)
            if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
                raise ValueError("fusion weights must be finite, nonnegative and have positive total")
            if cfg.method == "confidence_weighted":
                # models reporting uncertainty get down-weighted where unsure
                conf = np.ones_like(M)
                for k, i in enumerate(ids):
                    if i in uncertainties:
                        u = uncertainties[i]
                        conf[k] = 1.0 / (1.0 + u)
                wm = w[:, None] * conf
                fused = (M * wm).sum(axis=0) / (wm.sum(axis=0) + 1e-12)
            else:
                w = w / w.sum()
                fused = (M * w[:, None]).sum(axis=0)
            eff = {i: float(w[k]) / float(w.sum()) for k, i in enumerate(ids)}

        # ---- temporal consensus (trailing mean over window) ----
        tw = max(cfg.temporal_window, 1)
        sums = np.r_[0.0, np.cumsum(fused)]
        ends = np.arange(1, len(fused) + 1)
        starts = np.maximum(ends - tw, 0)
        smoothed = (sums[ends] - sums[starts]) / (ends - starts)

        disagreement = M.std(axis=0)
        alarm = smoothed > cfg.alarm_threshold

        # ---- conflict detection ----
        conflicts: List[Dict] = []
        if len(ids) >= 2:
            hi = M.max(axis=0)
            lo = M.min(axis=0)
            conflict_steps = np.where((hi > cfg.alarm_threshold) &
                                      (hi - lo > cfg.conflict_margin) &
                                      (lo < cfg.alarm_threshold * 0.5))[0]
            if len(conflict_steps):
                # summarize contiguous runs
                runs = np.split(conflict_steps, np.where(np.diff(conflict_steps) > 1)[0] + 1)
                for r in runs[:20]:
                    t = int(r[len(r) // 2])
                    hi_m = ids[int(M[:, t].argmax())]
                    lo_m = ids[int(M[:, t].argmin())]
                    conflicts.append({
                        "step_range": (int(r[0]), int(r[-1])),
                        "alarming_model": hi_m, "dissenting_model": lo_m,
                        "scores": {hi_m: float(M[:, t].max()), lo_m: float(M[:, t].min())},
                    })
        return FusedAnomaly(score=smoothed, alarm=alarm, per_model=streams,
                            disagreement=disagreement, conflicts=conflicts,
                            contributors=eff)

    # ------------------------------------------------------------------
    # probability fusion (fault classification)
    # ------------------------------------------------------------------
    def fuse_probabilities(
        self, outputs: Sequence[ModelOutput], classes: Sequence[str],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int]]:
        """Coverage-aware weighted probability mean; absent classes abstain.

        Returns (fused class->prob arrays, consensus labels, votes per class at
        the final step).
        """
        cfg = self.cfg
        streams = [o for o in outputs if o.ok and o.probability]
        if not streams:
            raise RuntimeError("no probabilistic outputs to fuse")
        _aligned(streams)
        n = len(streams[0].timestamps)
        # A specialist abstains on classes it does not emit. Missing classes
        # must not receive log(1)=0 and defeat all actual negative log evidence.
        evidence = np.zeros((len(classes), n))
        coverage = np.zeros((len(classes), n))
        for o in streams:
            w = cfg.weights.get(o.model_id, 1.0)
            if not np.isfinite(w) or w < 0:
                raise ValueError("fusion weights must be finite and nonnegative")
            for ci, c in enumerate(classes):
                p = o.probability.get(c)
                if p is None:
                    continue
                p = np.asarray(p, float)
                if len(p) != n:
                    raise ValueError("classification streams must be timestamp aligned")
                evidence[ci] += w * p
                coverage[ci] += w
        P = np.divide(evidence, coverage, out=np.zeros_like(evidence), where=coverage > 0)
        mass = P.sum(axis=0, keepdims=True)
        if np.any(mass <= 0):
            raise ValueError("no class evidence with positive weight")
        P /= mass
        fused = {c: P[ci] for ci, c in enumerate(classes)}
        consensus = np.array([classes[i] for i in P.argmax(axis=0)], dtype=object)

        # majority vote at the final step (diagnosis-time consensus)
        votes: Dict[str, int] = {}
        for o in streams:
            if o.prediction is not None and len(o.prediction):
                lab = str(o.prediction[-1])
                votes[lab] = votes.get(lab, 0) + 1
        return fused, consensus, votes

    # ------------------------------------------------------------------
    # RUL fusion
    # ------------------------------------------------------------------
    def fuse_rul(self, outputs: Sequence[ModelOutput]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Inverse-variance weighting when uncertainty exists, else mean."""
        valid = [o for o in outputs if o.ok and o.rul_hours is not None]
        if valid:
            _aligned(valid)
        streams = [(o.model_id, np.asarray(o.rul_hours, float),
                    None if o.uncertainty is None else np.asarray(o.uncertainty, float))
                   for o in outputs if o.ok and o.rul_hours is not None]
        if not streams:
            raise RuntimeError("no RUL outputs to fuse")
        n = max(len(s[1]) for s in streams)
        est = np.zeros(n)
        wtot = np.zeros(n)
        contrib: Dict[str, float] = {}
        for mid, r, u in streams:
            if u is not None and len(u) == len(r):
                w = 1.0 / (u ** 2 + 1.0)
            else:
                w = np.full(n, 1.0 / len(streams))
            est += w * r
            wtot += w
            contrib[mid] = float(np.mean(w))
        est /= np.maximum(wtot, 1e-12)
        # spread across models as uncertainty proxy
        R = np.vstack([s[1] for s in streams])
        spread = R.std(axis=0)
        tot = sum(contrib.values()) + 1e-12
        contrib = {k: v / tot for k, v in contrib.items()}
        return est, spread, contrib
