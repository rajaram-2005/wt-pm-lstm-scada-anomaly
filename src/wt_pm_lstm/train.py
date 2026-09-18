"""Training loop for the LSTM/GRU sequence autoencoder (fabric Layer 4/6).

Protocol decisions that matter for reproducibility:

* **Time-ordered validation split.** The last ``val_fraction`` of the *training*
  windows is held out for early stopping. Nothing else is ever used for model
  selection.
* **Best-weight restore.** Validation loss is noisy at this model size; the
  returned model is the best checkpoint, not the last epoch, and the epoch at
  which it was taken is reported.
* **Ensemble.** ``ModelConfig.n_ensemble`` independently seeded models are
  trained. Their disagreement is the epistemic-uncertainty channel reported on
  every :class:`~wt_pm_lstm.schema.AnomalyRecord`.
* **Determinism.** All randomness (initialisation, batch order, dropout masks)
  comes from a seeded ``Generator``; a rerun with the same config reproduces
  the same weights bit-for-bit on the same NumPy build.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from wt_pm_lstm.config import ModelConfig, RunConfig
from wt_pm_lstm.nn import Adam, SequenceAutoencoder
from wt_pm_lstm.schema import CONTEXT_CHANNELS, EXOGENOUS_CHANNELS, default_score_channels

Array = np.ndarray


@dataclass
class TrainHistory:
    """Per-epoch record — the raw material for the training-curve figure."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    epoch_seconds: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    stopped_early: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "epoch_seconds": self.epoch_seconds,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "stopped_early": self.stopped_early,
            "n_epochs": len(self.train_loss),
        }


def channel_weight_vector(ccfg: ModelConfig, channel_names: Tuple[str, ...]) -> Array:
    """Per-channel loss weights.

    Standardisation already equalises channel magnitudes, so machine channels
    carry weight 1.0. Environment channels (wind, ambient temperature, grid
    frequency) carry ``ModelConfig.exogenous_weight`` because the turbine cannot
    cause them: the model must observe them to explain the machine's response,
    but a poor one-step wind forecast is not evidence of a fault, and letting it
    dominate the loss makes the score drift with the weather.

    ``ModelConfig.channel_weights`` overrides individual channels, which is how
    an ablation asks "how much does the detector rely on the vibration channel?".
    """
    w = np.ones(len(channel_names))
    for j, name in enumerate(channel_names):
        if name in CONTEXT_CHANNELS:
            # Context channels still get gradient (the encoder must represent
            # them), just far less weight than the response channels.
            w[j] = float(ccfg.exogenous_weight)
        if name in ccfg.channel_weights:
            w[j] = float(ccfg.channel_weights[name])
    return w


def context_index(
    ccfg: ModelConfig, channel_names: Tuple[str, ...]
) -> Tuple[int, ...]:
    """Indices of the channels fed to the decoder as context.

    Defaults to the environment and control-command channels. Conditioning the
    decoder on them is what turns the model from "predict the weather" into
    "explain the machine's response given the conditions and commands", which is
    both easier to fit and far more sensitive to genuine faults.
    """
    selected = tuple(ccfg.context_channels) if ccfg.context_channels else CONTEXT_CHANNELS
    return tuple(j for j, name in enumerate(channel_names) if name in selected)


def score_channel_mask(ccfg: ModelConfig, channel_names: Tuple[str, ...]) -> Array:
    """Boolean mask (D,) selecting the channels whose residuals are scored.

    Defaults to every non-exogenous channel. This is the single most important
    modelling decision in the detector: scoring the wind channel measures the
    weather, not the turbine.
    """
    selected = set(ccfg.score_channels)
    default = set(default_score_channels(channel_names))
    mask = np.ones(len(channel_names), dtype=bool)
    for j, name in enumerate(channel_names):
        mask[j] = (name in selected) if selected else (name in default)
    if not mask.any():
        raise ValueError("score_channel_mask selected no channels")
    return mask


def train_model(
    ccfg: ModelConfig,
    windows: Array,
    masks: Array,
    weights: Array,
    rng: np.random.Generator,
    seed: int,
    channel_names: Tuple[str, ...],
    verbose: bool = False,
) -> Tuple[SequenceAutoencoder, TrainHistory]:
    """Train one model; returns it and its history."""
    n, T, D = windows.shape
    n_val = max(1, int(n * ccfg.val_fraction))
    n_train = n - n_val
    if n_train < 2:
        raise ValueError("not enough windows to train; increase the record length")
    train_x, train_m = windows[:n_train], masks[:n_train]
    val_x, val_m = windows[n_train:], masks[n_train:]

    model = SequenceAutoencoder(
        input_size=D,
        hidden_size=ccfg.hidden_size,
        latent_size=max(2, ccfg.latent_size),
        cell=ccfg.cell,
        seed=seed,
        dropout=ccfg.dropout,
        forecast_weight=ccfg.forecast_weight,
        exogenous_index=context_index(ccfg, channel_names),
    )
    opt = Adam(
        model.params,
        lr=ccfg.learning_rate,
        weight_decay=ccfg.weight_decay,
        max_grad_norm=ccfg.gradient_clip,
    )
    history = TrainHistory()
    best_params = {k: v.copy() for k, v in model.params.items()}
    batch = min(ccfg.batch_size, n_train)
    epochs_without_improvement = 0

    for epoch in range(ccfg.epochs):
        t0 = time.perf_counter()
        opt.lr = ccfg.learning_rate * (ccfg.lr_decay ** (epoch // max(1, ccfg.lr_decay_every)))
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        norm_accum = 0.0
        for start in range(0, n_train, batch):
            idx = order[start : start + batch]
            xb = np.ascontiguousarray(train_x[idx])
            mb = np.ascontiguousarray(train_m[idx])
            loss, grads = model.loss_and_grads(xb, mb, weights, rng=rng)
            norm_accum += opt.step(model.params, grads)
            epoch_loss += loss
            n_batches += 1

        val_loss = _eval_loss(model, val_x, val_m, weights)
        history.train_loss.append(epoch_loss / max(1, n_batches))
        history.val_loss.append(val_loss)
        history.grad_norm.append(norm_accum / max(1, n_batches))
        history.lr.append(opt.lr)
        history.epoch_seconds.append(time.perf_counter() - t0)

        if val_loss < history.best_val_loss - 1e-6:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_params = {k: v.copy() for k, v in model.params.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= ccfg.patience:
                history.stopped_early = True
                break
        if verbose:
            print(
                f"    epoch {epoch:3d}  train={history.train_loss[-1]:.5f}  "
                f"val={val_loss:.5f}  |g|={history.grad_norm[-1]:.3f}  "
                f"lr={opt.lr:.2e}  {history.epoch_seconds[-1]:.2f}s"
            )

    # Restore the best-validation checkpoint *in place*. Adam updates the
    # parameter buffers in place (`params[k] -= ...`), so the dict entries and
    # the cells' own attributes are the same arrays; assigning `model.params[k]
    # = v` here would rebind the dict and leave the cells holding the weights
    # from the last epoch. Early stopping would then have no effect at all and
    # the deployed model would be the overfit one — silently.
    model.load_params(best_params)
    return model, history


def _eval_loss(
    model: SequenceAutoencoder, x: Array, m: Array, weights: Array, max_windows: int = 512
) -> float:
    """Mean validation loss over at most ``max_windows`` windows (no shuffling)."""
    if x.shape[0] == 0:
        return float("nan")
    idx = np.linspace(0, x.shape[0] - 1, min(max_windows, x.shape[0])).astype(int)
    out = model.forward(np.ascontiguousarray(x[idx]))
    loss_r, _ = model.masked_mse(np.ascontiguousarray(x[idx]), out["x_hat"], weights, np.ascontiguousarray(m[idx]))
    loss_p = 0.0
    if x.shape[1] > 1:
        loss_p, _ = model.masked_mse(
            np.ascontiguousarray(x[idx][:, 1:]), out["y_hat"][:, :-1], weights, np.ascontiguousarray(m[idx][:, 1:])
        )
    return loss_r + model.forecast_weight * loss_p


def train_ensemble(
    cfg: RunConfig,
    windows: Array,
    masks: Array,
    channel_names: Tuple[str, ...],
    verbose: bool = False,
) -> Tuple[List[SequenceAutoencoder], List[TrainHistory]]:
    """Train the ensemble described by ``cfg.model.n_ensemble``."""
    weights = channel_weight_vector(cfg.model, channel_names)
    models: List[SequenceAutoencoder] = []
    histories: List[TrainHistory] = []
    for member in range(cfg.model.n_ensemble):
        seed = cfg.model.offline_seed + 1000 * member
        rng = np.random.default_rng(seed)
        if verbose:
            print(f"  [model {member + 1}/{cfg.model.n_ensemble}] seed={seed} cell={cfg.model.cell}")
        model, history = train_model(
            cfg.model, windows, masks, weights, rng, seed, channel_names, verbose
        )
        models.append(model)
        histories.append(history)
        if verbose:
            print(
                f"    best epoch {history.best_epoch} "
                f"val={history.best_val_loss:.5f} "
                f"params={model.n_params} "
                f"time={sum(history.epoch_seconds):.1f}s"
            )
    return models, histories


def predict_all(models: List[SequenceAutoencoder], x: Array) -> Tuple[Array, Array]:
    """Stack reconstruction and forecast errors across the ensemble.

    Returns ``(recon, forecast)`` with shape ``(M, T, B, D)``: the ensemble axis
    is explicit so both the mean score and the member disagreement are
    available without a second forward pass.
    """
    recons, forecasts = [], []
    for model in models:
        out = model.forward(np.ascontiguousarray(x))
        recons.append(out["x_hat"])
        forecasts.append(out["y_hat"])
    return np.stack(recons), np.stack(forecasts)


__all__ = [
    "TrainHistory",
    "context_index",
    "score_channel_mask",
    "train_model",
    "train_ensemble",
    "predict_all",
    "channel_weight_vector",
]
