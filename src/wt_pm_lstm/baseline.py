"""Normal-behaviour model (NBM): the transparent baseline SCADA detectors use.

Why a second model, inside the LSTM repository?
-----------------------------------------------
A pure autoencoder is *relative*: it reconstructs the window it was given, so a
channel that drifts to a new, stable level is perfectly reconstructible — the
latent simply encodes the new level. That is a real blind spot, demonstrated in
this repository's own diagnostics (a slow gearbox-oil offset was invisible to
both the reconstruction and the forecast head no matter how long it trained).

The NBM fixes that class of fault by being *absolute*: it predicts each response
channel from the contemporaneous conditions (wind, ambient temperature, pitch,
grid frequency) using a small ridge regression fitted on healthy data. A channel
sitting 1.7 robust sigmas away from what the operating point implies is
detectable immediately, no matter how smoothly it got there.

It also serves the platform's evaluation discipline: an LSTM detector without a
baseline is an unfalsifiable claim. The NBM is deliberately simple — closed-form,
interpretable, and cheap enough to run inside a gateway — so that the deep model
has to beat something honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

Array = np.ndarray


def _design_matrix(context: Array, degree: int = 2) -> Tuple[Array, List[str]]:
    """Polynomial feature expansion of the context channels (no bias column).

    Quadratic terms are what let a linear model represent a power curve and the
    temperature/load interaction. Higher orders were not used: the point of a
    baseline is to be weak enough that it cannot be accused of stealing the deep
    model's contribution.
    """
    n, k = context.shape
    cols = [context]
    names = [f"c{j}" for j in range(k)]
    if degree >= 2:
        cols.append(context**2)
        names += [f"c{j}^2" for j in range(k)]
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append(context[:, i] * context[:, j])
                names.append(f"c{i}*c{j}")
        if pairs:
            cols.append(np.stack(pairs, axis=1))
    return np.concatenate(cols, axis=1), names


@dataclass
class NormalBehaviourModel:
    """Ridge regression from operating conditions to response channels."""

    channel_names: Tuple[str, ...]
    context_names: Tuple[str, ...]
    target_indices: Array
    context_indices: Array
    coef: Array  # (n_features + 1, n_targets)
    feature_names: List[str]
    ridge: float = 1e-3

    @classmethod
    def fit(
        cls,
        values: Array,
        mask: Array,
        channel_names: Sequence[str],
        context_names: Sequence[str],
        score_mask: Array,
        ridge: float = 1e-3,
        degree: int = 2,
    ) -> "NormalBehaviourModel":
        names = tuple(channel_names)
        ctx_names = tuple(context_names)
        ctx_idx = np.array([names.index(n) for n in ctx_names], dtype=int)
        tgt_idx = np.array([j for j in range(len(names)) if score_mask[j]], dtype=int)

        # Fit only on rows where every context channel is valid; rows where a
        # target is invalid are dropped per target inside the solve below.
        rows = mask[:, ctx_idx].all(axis=1)
        X = values[rows][:, ctx_idx]
        feats, feat_names = _design_matrix(X, degree=degree)
        A = np.concatenate([np.ones((feats.shape[0], 1)), feats], axis=1)

        coef = np.zeros((A.shape[1], len(tgt_idx)))
        for pos, j in enumerate(tgt_idx):
            valid = mask[rows, j]
            if valid.sum() < A.shape[1] * 2:
                continue
            Aj, yj = A[valid], values[rows][valid, j]
            gram = Aj.T @ Aj + ridge * np.eye(A.shape[1])
            coef[:, pos] = np.linalg.solve(gram, Aj.T @ yj)
        return cls(
            channel_names=names,
            context_names=ctx_names,
            target_indices=tgt_idx,
            context_indices=ctx_idx,
            coef=coef,
            feature_names=["bias"] + [f"{n}" for n in feat_names],
            ridge=ridge,
        )

    # -- inference --------------------------------------------------------
    def predict(self, values: Array) -> Array:
        """Predicted values for the target channels: ``(n, n_targets)``."""
        feats, _ = _design_matrix(values[:, self.context_indices], degree=2)
        A = np.concatenate([np.ones((feats.shape[0], 1)), feats], axis=1)
        return A @ self.coef

    def residuals(self, values: Array) -> Array:
        """Signed residuals ``observed - predicted`` on the target channels."""
        return values[:, self.target_indices] - self.predict(values)

    def r2(self, values: Array, mask: Array) -> Dict[str, float]:
        """Coefficient of determination per target channel (fit diagnostics)."""
        out: Dict[str, float] = {}
        pred = self.predict(values)
        for pos, j in enumerate(self.target_indices):
            valid = mask[:, j]
            y = values[valid, j]
            if y.size < 2:
                continue
            ss_res = float(np.sum((y - pred[valid, pos]) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            out[self.channel_names[j]] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_names": list(self.channel_names),
            "context_names": list(self.context_names),
            "target_indices": self.target_indices.tolist(),
            "context_indices": self.context_indices.tolist(),
            "coef": self.coef.tolist(),
            "feature_names": self.feature_names,
            "ridge": self.ridge,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalBehaviourModel":
        return cls(
            channel_names=tuple(payload["channel_names"]),
            context_names=tuple(payload["context_names"]),
            target_indices=np.asarray(payload["target_indices"], dtype=int),
            context_indices=np.asarray(payload["context_indices"], dtype=int),
            coef=np.asarray(payload["coef"], dtype=np.float64),
            feature_names=list(payload["feature_names"]),
            ridge=float(payload.get("ridge", 1e-3)),
        )


__all__ = ["NormalBehaviourModel"]
