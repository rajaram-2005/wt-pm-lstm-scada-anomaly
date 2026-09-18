"""Minimal recurrent network engine with hand-derived gradients.

Why not a framework?
--------------------
Model 13 has to run at four fidelities: research (data centre), on-premise
inference, edge gateway and, eventually, an MCU. A dependency-light engine
keeps the reference implementation *checkable* — :mod:`tests.test_nn` verifies
every analytic gradient against finite differences — and gives the
quantisation work in models 17/22 a well-defined numerical baseline.

Contents
--------
* :func:`sigmoid`, :func:`tanh_act`
* :class:`LSTMCell` / :class:`GRUCell` — forward *and* backward, float64
* :class:`Adam` — bias-corrected Adam with global-norm clipping
* :class:`SequenceAutoencoder` — the model-13 detector

Gate ordering is fixed as ``[input, forget, candidate, output]`` for the LSTM
and ``[reset, update, candidate]`` for the GRU, matching the conventions used
by the major frameworks so that weight exports stay interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------
def sigmoid(x: Array) -> Array:
    """Numerically stable logistic function."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def tanh_act(x: Array) -> Array:
    return np.tanh(x)


def d_sigmoid_from_output(y: Array) -> Array:
    """Derivative of the logistic function expressed via its output."""
    return y * (1.0 - y)


def d_tanh_from_output(y: Array) -> Array:
    return 1.0 - y * y


# --------------------------------------------------------------------------
# Recurrent cells
# --------------------------------------------------------------------------
@dataclass
class LSTMCell:
    """Single-layer LSTM: ``W_ih`` is ``(4H, D)``, ``W_hh`` is ``(4H, H)``."""

    input_size: int
    hidden_size: int
    rng: np.random.Generator

    def __post_init__(self) -> None:
        H, D = self.hidden_size, self.input_size
        bound = np.sqrt(6.0 / (4 * H + D))
        self.W_ih = self.rng.uniform(-bound, bound, size=(4 * H, D))
        # Orthogonal recurrent matrix keeps gradient norm stable over the
        # window length; scaled by 0.8 as in the standard recipe.
        q, _ = np.linalg.qr(self.rng.standard_normal((H, H)))
        self.W_hh = np.concatenate([q, q, q, q], axis=0) * 0.8
        self.b = np.zeros(4 * H)
        # Forget-gate bias of 1.0: standard trick that makes long-range
        # dependencies learnable from a cold start.
        self.b[H : 2 * H] = 1.0

    def params(self) -> Dict[str, Array]:
        return {"W_ih": self.W_ih, "W_hh": self.W_hh, "b": self.b}

    def forward(
        self,
        x: Array,
        h0: Optional[Array] = None,
        c0: Optional[Array] = None,
        return_cache: bool = False,
    ) -> Tuple[Array, Array, Optional[List[Tuple]]]:
        """Run over ``x`` of shape ``(T, B, D)``.

        Returns ``(hs, c_final, cache)`` where ``hs`` is ``(T, B, H)``. The
        cache holds one tuple per timestep — ``(x_t, h_prev, c_prev, i, f, g,
        o, tanh(c))`` — which is the minimum state needed for exact BPTT.
        """
        T, B, _ = x.shape
        H = self.hidden_size
        h = np.zeros((B, H)) if h0 is None else np.asarray(h0, dtype=np.float64)
        c = np.zeros((B, H)) if c0 is None else np.asarray(c0, dtype=np.float64)
        hs = np.empty((T, B, H))
        gates: Optional[List[Tuple]] = [] if return_cache else None
        W_in_T, Wh_T = self.W_ih.T, self.W_hh.T
        for t in range(T):
            h_prev, c_prev = h, c
            pre = x[t] @ W_in_T + h_prev @ Wh_T + self.b
            i = sigmoid(pre[:, :H])
            f = sigmoid(pre[:, H : 2 * H])
            g = tanh_act(pre[:, 2 * H : 3 * H])
            o = sigmoid(pre[:, 3 * H :])
            c = f * c_prev + i * g
            tc = tanh_act(c)
            h = o * tc
            hs[t] = h
            if gates is not None:
                gates.append((x[t], h_prev, c_prev, i, f, g, o, tc))
        return hs, c, gates

    def backward(
        self, dhs: Array, cache: List[Tuple]
    ) -> Tuple[Array, Dict[str, Array], Array, Array]:
        """Backprop through time.

        Returns ``(dx, param_grads, dh0, dc0)`` where ``dh0``/``dc0`` are the
        gradients w.r.t. the *initial* states — needed because the decoder's
        initial hidden state is produced by a learnable projection, not by the
        sequence.
        """
        T, B, _ = dhs.shape
        H = self.hidden_size
        dx = np.empty((T, B, self.input_size))
        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        db = np.zeros_like(self.b)
        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        for t in range(T - 1, -1, -1):
            x_t, h_prev, c_prev, i, f, g, o, tc = cache[t]
            dh = dhs[t] + dh_next
            do_pre = dh * tc * d_sigmoid_from_output(o)
            dc = dh * o * d_tanh_from_output(tc) + dc_next
            df_pre = dc * c_prev * d_sigmoid_from_output(f)
            di_pre = dc * g * d_sigmoid_from_output(i)
            dg_pre = dc * i * d_tanh_from_output(g)
            dpre = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=1)
            dW_ih += dpre.T @ x_t
            dW_hh += dpre.T @ h_prev
            db += dpre.sum(axis=0)
            dx[t] = dpre @ self.W_ih
            dh_next = dpre @ self.W_hh
            dc_next = dc * f
        return dx, {"W_ih": dW_ih, "W_hh": dW_hh, "b": db}, dh_next, dc_next


@dataclass
class GRUCell:
    """Single-layer GRU with gates ordered ``[reset, update, candidate]``."""

    input_size: int
    hidden_size: int
    rng: np.random.Generator

    def __post_init__(self) -> None:
        H, D = self.hidden_size, self.input_size
        bound = np.sqrt(6.0 / (3 * H + D))
        self.W_ih = self.rng.uniform(-bound, bound, size=(3 * H, D))
        self.W_hh = self.rng.uniform(-bound, bound, size=(3 * H, H))
        self.b_ih = np.zeros(3 * H)
        self.b_hh = np.zeros(3 * H)

    def params(self) -> Dict[str, Array]:
        return {"W_ih": self.W_ih, "W_hh": self.W_hh, "b_ih": self.b_ih, "b_hh": self.b_hh}

    def forward(
        self,
        x: Array,
        h0: Optional[Array] = None,
        return_cache: bool = False,
    ) -> Tuple[Array, Optional[List[Tuple]]]:
        T, B, _ = x.shape
        H = self.hidden_size
        h = np.zeros((B, H)) if h0 is None else np.asarray(h0, dtype=np.float64)
        hs = np.empty((T, B, H))
        gates: Optional[List[Tuple]] = [] if return_cache else None
        W_in_T, Wh_T = self.W_ih.T, self.W_hh.T
        for t in range(T):
            h_prev = h
            xg = x[t] @ W_in_T + self.b_ih
            hr = h_prev @ Wh_T
            r = sigmoid(xg[:, :H] + hr[:, :H] + self.b_hh[:H])
            z = sigmoid(xg[:, H : 2 * H] + hr[:, H : 2 * H] + self.b_hh[H : 2 * H])
            n = tanh_act(xg[:, 2 * H :] + (r * h_prev) @ Wh_T[:, 2 * H :] + self.b_hh[2 * H :])
            h = (1.0 - z) * n + z * h_prev
            hs[t] = h
            if gates is not None:
                gates.append((x[t], h_prev, r, z, n))
        return hs, gates

    def backward(
        self, dhs: Array, cache: List[Tuple]
    ) -> Tuple[Array, Dict[str, Array], Array]:
        T, B, _ = dhs.shape
        H = self.hidden_size
        dx = np.empty((T, B, self.input_size))
        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        db_ih = np.zeros_like(self.b_ih)
        db_hh = np.zeros_like(self.b_hh)
        dh_next = np.zeros((B, H))
        for t in range(T - 1, -1, -1):
            x_t, h_prev, r, z, n = cache[t]
            dh = dhs[t] + dh_next
            dn_pre = dh * (1.0 - z) * d_tanh_from_output(n)
            dz_pre = dh * (h_prev - n) * d_sigmoid_from_output(z)
            # r enters only through the candidate branch.
            dr_pre = (dn_pre @ self.W_hh[2 * H :, :] * h_prev) * d_sigmoid_from_output(r)
            dpre = np.concatenate([dr_pre, dz_pre, dn_pre], axis=1)
            dW_ih += dpre.T @ x_t
            # Row blocks, not column blocks: W_hh is (3H, H), so the reset and
            # update blocks are dW_hh[:2H] and the candidate block is dW_hh[2H:].
            dW_hh[: 2 * H, :] += dpre[:, : 2 * H].T @ h_prev
            dW_hh[2 * H :, :] += dn_pre.T @ (r * h_prev)
            db_ih += dpre.sum(axis=0)
            db_hh += dpre.sum(axis=0)
            dx[t] = dpre @ self.W_ih
            dh_next = dpre[:, : 2 * H] @ self.W_hh[: 2 * H, :]
            dh_next += (dn_pre @ self.W_hh[2 * H :, :]) * r
            # Direct linear path: h = (1 - z) * n + z * h_prev contributes
            # dh * z to the previous hidden state.
            dh_next += dh * z
        return dx, {"W_ih": dW_ih, "W_hh": dW_hh, "b_ih": db_ih, "b_hh": db_hh}, dh_next


# --------------------------------------------------------------------------
# Optimiser
# --------------------------------------------------------------------------
class Adam:
    """Adam with decoupled weight decay and global-norm gradient clipping."""

    def __init__(
        self,
        params: Dict[str, Array],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 5.0,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
        self.last_grad_norm = 0.0

    def step(self, params: Dict[str, Array], grads: Dict[str, Array]) -> float:
        """Apply one update; returns the pre-clipping global gradient norm."""
        self.t += 1
        total = 0.0
        for g in grads.values():
            total += float(np.sum(g * g))
        norm = float(np.sqrt(total))
        self.last_grad_norm = norm
        scale = 1.0 if norm <= self.max_grad_norm or norm == 0.0 else self.max_grad_norm / norm
        for k in params:
            g = grads[k] * scale
            if self.weight_decay:
                g = g + self.weight_decay * params[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return norm


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class SequenceAutoencoder:
    """LSTM/GRU sequence autoencoder with reconstruction and forecast heads.

    Architecture::

        x_1..x_T -> encoder RNN -> h_T -> z = W_z h_T + b_z  (latent)
                                        -> h0 = W_seed z + b_seed
                   decoder RNN (input = z broadcast over time)
                        head_recon:  x_hat_t        (window reconstruction)
                        head_pred:   x_hat_{t+1}    (one-step forecast)

    Two heads share one latent, and that is the mechanism behind the fabric's
    fused anomaly score: the reconstruction term catches out-of-distribution
    *states*, the forecast term catches violations of the *dynamics*. Both are
    needed — a stuck sensor is ambiguous under reconstruction but violates the
    dynamics, while a genuine load ramp is predictable but poorly
    reconstructed.

    The decoder never sees the input sequence. Teacher forcing would make the
    reconstruction task nearly trivial and collapse the residual signal.
    """

    input_size: int
    hidden_size: int = 32
    latent_size: int = 8
    cell: str = "lstm"
    seed: int = 0
    dropout: float = 0.0
    forecast_weight: float = 0.5
    #: Indices of environment channels (wind, ambient temperature, grid
    #: frequency) handed to the *decoder* at every timestep.
    exogenous_index: Sequence[int] = ()
    params: Dict[str, Array] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.cell not in ("lstm", "gru"):
            raise ValueError(f"cell must be 'lstm' or 'gru', got {self.cell!r}")
        rng = np.random.default_rng(self.seed)
        H, D, Z = self.hidden_size, self.input_size, self.latent_size
        self.exogenous_index = tuple(int(i) for i in self.exogenous_index)
        self.params = {}
        # The decoder input is [latent, environment]: the model must explain the
        # machine's response *given* the conditions it operated in, not predict
        # the weather. Encoder input is still the full channel vector.
        self.dec_input_size = Z + len(self.exogenous_index)
        self.enc = LSTMCell(D, H, rng) if self.cell == "lstm" else GRUCell(D, H, rng)
        self.dec = LSTMCell(self.dec_input_size, H, rng) if self.cell == "lstm" else GRUCell(self.dec_input_size, H, rng)
        self.params.update({f"enc.{k}": v for k, v in self.enc.params().items()})
        self.params.update({f"dec.{k}": v for k, v in self.dec.params().items()})
        scale = 1.0 / np.sqrt(H)
        self.params["W_z"] = rng.normal(0.0, scale, size=(Z, H))
        self.params["b_z"] = np.zeros(Z)
        self.params["W_seed"] = rng.normal(0.0, scale, size=(H, Z))
        self.params["b_seed"] = np.zeros(H)
        self.params["W_recon"] = rng.normal(0.0, scale, size=(D, H))
        self.params["b_recon"] = np.zeros(D)
        self.params["W_pred"] = rng.normal(0.0, scale, size=(D, H))
        self.params["b_pred"] = np.zeros(D)

    # -- helpers ----------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def summary(self) -> Dict[str, object]:
        return {
            "cell": self.cell,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "latent_size": self.latent_size,
            "decoder_input_size": self.dec_input_size,
            "exogenous_index": list(self.exogenous_index),
            "n_parameters": self.n_params,
            "forecast_weight": self.forecast_weight,
        }

    # -- forward ----------------------------------------------------------
    def load_params(self, arrays: Mapping[str, Array]) -> None:
        """Copy saved arrays back *into* the existing parameter buffers.

        The dict values are aliases of the cell attributes
        (``model.params["enc.W_ih"] is model.enc.W_ih``), and Adam updates them
        in place (``params[k] -= ...``) precisely so that the alias survives
        training. Rebinding the dict entry to a freshly loaded array instead
        breaks that alias: the restored bundle then looks correct, reports the
        right parameter shapes, and silently scores with the *initialisation*
        the constructor just created. Assigning with ``[...] =`` preserves it.
        """
        missing = [k for k in self.params if k not in arrays]
        if missing:
            raise ValueError(f"bundle is missing parameters: {sorted(missing)}")
        for key, buffer in self.params.items():
            value = np.asarray(arrays[key], dtype=np.float64)
            if value.shape != buffer.shape:
                raise ValueError(
                    f"{key}: bundle has shape {value.shape}, model expects {buffer.shape}"
                )
            buffer[...] = value

    def forward(self, x: Array, cache: bool = False, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Run on ``x`` of shape ``(T, B, D)``."""
        T, B, _ = x.shape
        H = self.hidden_size
        if self.cell == "lstm":
            hs_enc, _, enc_cache = self.enc.forward(x, return_cache=cache)
        else:
            hs_enc, enc_cache = self.enc.forward(x, return_cache=cache)

        h_T = hs_enc[-1]
        z = h_T @ self.params["W_z"].T + self.params["b_z"]

        mask = None
        if self.dropout > 0.0 and rng is not None:
            mask = (rng.random(z.shape) >= self.dropout).astype(np.float64) / (1.0 - self.dropout)
            z_used = z * mask
        else:
            z_used = z
        h0 = z_used @ self.params["W_seed"].T + self.params["b_seed"]

        if self.exogenous_index:
            exo = x[:, :, list(self.exogenous_index)]
            dec_in = np.concatenate(
                [np.broadcast_to(z_used[None, :, :], (T, B, self.latent_size)), exo], axis=2
            ).copy()
        else:
            dec_in = np.broadcast_to(z_used[None, :, :], (T, B, self.latent_size)).copy()
        if self.cell == "lstm":
            hs_dec, _, dec_cache = self.dec.forward(dec_in, h0=h0, return_cache=cache)
        else:
            hs_dec, dec_cache = self.dec.forward(dec_in, h0=h0, return_cache=cache)

        out: Dict[str, Any] = {
            "x_hat": hs_dec @ self.params["W_recon"].T + self.params["b_recon"],
            "y_hat": hs_dec @ self.params["W_pred"].T + self.params["b_pred"],
            "z": z,
            "h_T": h_T,
            "h_dec": hs_dec,
        }
        if cache:
            out["cache"] = {
                "h_T": h_T,
                "enc": enc_cache,
                "dec": dec_cache,
                "h0": h0,
                "z_used": z_used,
                "dropout_mask": mask,
                "hs_enc": hs_enc,
                "hs_dec": hs_dec,
            }
        return out

    # -- loss -------------------------------------------------------------
    @staticmethod
    def masked_mse(
        x: Array, x_hat: Array, channel_weight: Array, sample_mask: Array
    ) -> Tuple[float, Array]:
        """Masked, channel-weighted MSE.

        ``channel_weight`` is ``(D,)`` (typically the inverse robust spread, so
        that the 3 MW power channel does not drown out the 1 mm/s vibration
        channel); ``sample_mask`` is ``(T, B, D)`` in ``{0, 1}`` marking the
        samples the contract declares valid. Returns the loss and the gradient
        of the loss w.r.t. ``x_hat``.
        """
        w = sample_mask * channel_weight
        diff = (x - x_hat) * w
        denom = max(float(np.sum(w)), 1e-9)
        loss = float(np.sum(diff * diff) / denom)
        # dL/dx_hat = -2 (x - x_hat) w^2 / denom; the extra w matters.
        return loss, -2.0 * diff * w / denom

    def loss_and_grads(
        self,
        x: Array,
        sample_mask: Array,
        channel_weight: Array,
        rng: Optional[np.random.Generator] = None,
        forecast_weight: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Array]]:
        """Full training step for one batch: returns ``(loss, param_grads)``.

        The forecast head is trained on shifted targets, so its mask is the
        mask of the *predicted* step: a prediction of an imputed/invalid sample
        must not contribute to the loss.
        """
        if forecast_weight is None:
            forecast_weight = self.forecast_weight
        out = self.forward(x, cache=True, rng=rng)
        loss_r, d_recon = self.masked_mse(x, out["x_hat"], channel_weight, sample_mask)

        loss_p = 0.0
        d_pred = np.zeros_like(x)
        if forecast_weight > 0.0 and x.shape[0] > 1:
            loss_p, d_pred_shifted = self.masked_mse(
                x[1:], out["y_hat"][:-1], channel_weight, sample_mask[1:]
            )
            d_pred[:-1] = d_pred_shifted
        loss = loss_r + forecast_weight * loss_p

        # Chain the weights through the two heads.
        grads = self.backward(out["cache"], d_recon, d_pred * forecast_weight)
        return loss, grads

    # -- backward ---------------------------------------------------------
    def backward(self, cache: Dict[str, Any], d_recon: Array, d_pred: Array) -> Dict[str, Array]:
        """Backpropagate head gradients to every parameter."""
        grads: Dict[str, Array] = {}
        hs_dec = cache["hs_dec"]
        hh = hs_dec.reshape(-1, self.hidden_size)
        grads["W_recon"] = d_recon.reshape(-1, self.input_size).T @ hh
        grads["b_recon"] = d_recon.reshape(-1, self.input_size).sum(axis=0)
        grads["W_pred"] = d_pred.reshape(-1, self.input_size).T @ hh
        grads["b_pred"] = d_pred.reshape(-1, self.input_size).sum(axis=0)

        dhs_dec = d_recon @ self.params["W_recon"] + d_pred @ self.params["W_pred"]

        # Decoder BPTT. The returned dh0/dc0 are the gradients w.r.t. the
        # initial states, which is where the latent projection enters.
        if self.cell == "lstm":
            ddec_in, g_dec, dh0, _ = self.dec.backward(dhs_dec, cache["dec"])
        else:
            ddec_in, g_dec, dh0 = self.dec.backward(dhs_dec, cache["dec"])
        for k, v in g_dec.items():
            grads[f"dec.{k}"] = v

        # Only the latent slice of the decoder input carries gradient to the
        # encoder; the environment slice is an input, not a parameter, and no
        # learned weight depends on it directly.
        dz_used = ddec_in[:, :, : self.latent_size].sum(axis=0) + dh0 @ self.params["W_seed"]
        grads["W_seed"] = dh0.T @ cache["z_used"]
        grads["b_seed"] = dh0.sum(axis=0)
        if cache["dropout_mask"] is not None:
            dz_used = dz_used * cache["dropout_mask"]

        grads["W_z"] = dz_used.T @ cache["h_T"]
        grads["b_z"] = dz_used.sum(axis=0)

        # Encoder BPTT: only the final hidden state feeds the latent.
        dhs_enc = np.zeros_like(cache["hs_enc"])
        dhs_enc[-1] += dz_used @ self.params["W_z"]
        if self.cell == "lstm":
            _, g_enc, _, _ = self.enc.backward(dhs_enc, cache["enc"])
        else:
            _, g_enc, _ = self.enc.backward(dhs_enc, cache["enc"])
        for k, v in g_enc.items():
            grads[f"enc.{k}"] = v
        return grads


__all__ = [
    "Array",
    "sigmoid",
    "tanh_act",
    "LSTMCell",
    "GRUCell",
    "Adam",
    "SequenceAutoencoder",
]
