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
        n = 0
        for o in outputs:
            if not o.ok or o.anomaly_score is None:
                continue
            n = max(n, len(o.anomaly_score))
        for o in outputs:
            if not o.ok or o.anomaly_score is None:
                continue
            s = np.asarray(o.anomaly_score, float)
            if len(s) < n:               # left-pad shorter streams with first value
                s = np.concatenate([np.full(n - len(s), s[0] if len(s) else 0.0), s])
            streams[o.model_id] = s
            if o.uncertainty is not None and len(o.uncertainty) == len(o.anomaly_score):
                u = np.asarray(o.uncertainty, float)
                if len(u) < n:
                    u = np.concatenate([np.full(n - len(u), u[0] if len(u) else 0.0), u])
                uncertainties[o.model_id] = u
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
            if cfg.method == "confidence_weighted":
                # models reporting uncertainty get down-weighted where unsure
                conf = np.ones_like(M)
                for k, i in enumerate(ids):
                    if i in uncertainties:
                        u = uncertainties[i]
                        conf[k] = 1.0 / (1.0 + u / (np.median(u) + 1e-9))
                wm = w[:, None] * conf
                fused = (M * wm).sum(axis=0) / (wm.sum(axis=0) + 1e-12)
            else:
                w = w / w.sum()
                fused = (M * w[:, None]).sum(axis=0)
            eff = {i: float(w[k]) / float(w.sum()) for k, i in enumerate(ids)}

        # ---- temporal consensus (trailing mean over window) ----
        tw = max(cfg.temporal_window, 1)
        kernel = np.ones(tw) / tw
        smoothed = np.convolve(fused, kernel, mode="full")[:len(fused)]

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
        """Bayesian product-of-experts where possible, weighted mean otherwise.

        Returns (fused class->prob arrays, consensus labels, votes per class at
        the final step).
        """
        cfg = self.cfg
        streams = [o for o in outputs if o.ok and o.probability]
        if not streams:
            raise RuntimeError("no probabilistic outputs to fuse")
        n = max(len(o.timestamps) for o in streams)
        logp = np.zeros((len(classes), n))
        wsum = 0.0
        for o in streams:
            w = cfg.weights.get(o.model_id, 1.0)
            wsum += w
            for ci, c in enumerate(classes):
                p = o.probability.get(c)
                if p is None:
                    continue
                p = np.asarray(p, float)
                if len(p) < n:
                    p = np.concatenate([np.full(n - len(p), p[0] if len(p) else 0.0), p])
                logp[ci] += w * np.log(np.clip(p, 1e-6, 1.0))   # Bayesian PoE
        logp /= max(wsum, 1e-12)
        P = np.exp(logp - logp.max(axis=0, keepdims=True))
        P /= P.sum(axis=0, keepdims=True)
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
            if len(r) < n:
                r = np.concatenate([np.full(n - len(r), r[0]), r])
            if u is not None and len(u) == len(r):
                w = 1.0 / (u ** 2 + 1.0)
            else:
                w = np.full(n, 1.0 / len(streams))
            est += w * r
            wtot += w
            contrib[mid] = float(np.mean(w))
        est /= np.maximum(wtot, 1e-12)
        # spread across models as uncertainty proxy
        R = np.vstack([np.concatenate([np.full(n - len(s[1]), s[1][0]), s[1]])
                       if len(s[1]) < n else s[1] for s in streams])
        spread = R.std(axis=0)
        tot = sum(contrib.values()) + 1e-12
        contrib = {k: v / tot for k, v in contrib.items()}
        return est, spread, contrib
