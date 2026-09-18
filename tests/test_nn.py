"""Finite-difference verification of the hand-derived backwards.

``nn.py`` implements LSTM/GRU forward *and* backward passes in NumPy. Nothing in
this repository is more dangerous to change: a sign error in one gate does not
raise, does not slow anything down, and simply produces a slightly worse
detector that still trains and still scores. Every gradient the training loop
uses is therefore checked here against a central finite difference of the
training loss itself — ``SequenceAutoencoder.loss_and_grads``, the same call
``train.py`` makes.

The GRU case keeps the ``T=2, B=D=H=1`` trace: with a single hidden unit the
recurrent weight block is 1-by-1, and errors in the ``dh_next`` path — the
direct ``dh * z`` term versus the ``n``-row terms — cancel at larger sizes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from wt_pm_lstm.nn import GRUCell, LSTMCell, SequenceAutoencoder

EPS = 1e-6


def numeric_grad(f: Callable[[], float], like: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Central-difference gradient of a scalar function of an array."""
    out = np.zeros_like(like)
    flat = like.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = f()
        flat[i] = original - eps
        minus = f()
        flat[i] = original
        out.reshape(-1)[i] = (plus - minus) / (2 * eps)
    return out


def check_cell(cell_cls, T=4, B=3, D=5, H=4, seed=0):
    rng = np.random.default_rng(seed)
    if cell_cls is LSTMCell:
        cell = LSTMCell(input_size=D, hidden_size=H, rng=rng)
    else:
        cell = GRUCell(input_size=D, hidden_size=H, rng=rng)
    x = rng.normal(size=(T, B, D))
    h0 = rng.normal(size=(B, H))

    g = rng.normal(size=(T, B, H))
    if cell_cls is LSTMCell:
        _, _, cache = cell.forward(x, h0=h0, return_cache=True)
        dx, grads, dh0, _ = cell.backward(g, cache)
    else:
        _, cache = cell.forward(x, h0=h0, return_cache=True)
        dx, grads, dh0 = cell.backward(g, cache)

    def loss():
        return float(np.sum(cell.forward(x, h0=h0)[0] * g))

    for name, param in cell.params().items():
        np.testing.assert_allclose(
            grads[name], numeric_grad(loss, param), rtol=1e-5, atol=1e-7,
            err_msg=f"{name} gradient mismatch",
        )
    np.testing.assert_allclose(dx, numeric_grad(loss, x), rtol=1e-5, atol=1e-7, err_msg="dx mismatch")
    np.testing.assert_allclose(dh0, numeric_grad(loss, h0), rtol=1e-5, atol=1e-7, err_msg="dh0 mismatch")


@pytest.mark.parametrize("cell_cls", [LSTMCell, GRUCell])
def test_cell_gradients(cell_cls):
    check_cell(cell_cls)


def test_gru_single_unit_trace():
    """The 1x1 recurrent-weight case that hides dh_next errors at larger sizes."""
    check_cell(GRUCell, T=2, B=1, D=1, H=1, seed=3)


@pytest.mark.parametrize("cell", ["lstm", "gru"])
@pytest.mark.parametrize("exogenous", [False, True])
def test_model_gradients(cell, exogenous):
    """Every parameter gradient must match a finite difference of the loss."""
    rng = np.random.default_rng(5)
    T, B, D, H, L = 3, 2, 6, 4, 3
    exo = (0, 2) if exogenous else ()
    model = SequenceAutoencoder(
        input_size=D, hidden_size=H, latent_size=L, cell=cell, exogenous_index=exo, seed=17
    )
    x = rng.normal(size=(T, B, D))
    mask = np.ones((T, B, D))
    mask[1, 0, 2] = 0.0
    weight = rng.uniform(0.5, 2.0, size=D)

    loss, grads = model.loss_and_grads(x, mask, weight, forecast_weight=0.5)

    for name, param in model.params.items():
        numeric = numeric_grad(lambda: model.loss_and_grads(x, mask, weight, forecast_weight=0.5)[0], param)
        np.testing.assert_allclose(
            grads[name], numeric, rtol=2e-4, atol=2e-6, err_msg=f"{name} gradient mismatch"
        )
    assert np.isfinite(loss)


def test_masked_mse_ignores_invalid_samples():
    """A masked-out sample must not change the loss or its gradient."""
    rng = np.random.default_rng(3)
    T, B, D = 3, 2, 4
    x = rng.normal(size=(T, B, D))
    x_hat = rng.normal(size=(T, B, D))
    weight = np.ones(D)
    mask = np.ones((T, B, D))
    base_loss, base_grad = SequenceAutoencoder.masked_mse(x, x_hat, weight, mask)

    # With the mask *fixed*, the values at masked-out positions are irrelevant:
    # they cannot reach the loss. (Changing the mask itself does change the
    # denominator, which is why the two are tested separately.)
    mask[0, 1, :] = 0.0
    loss, grad = SequenceAutoencoder.masked_mse(x, x_hat, weight, mask)
    polluted = x.copy()
    polluted[0, 1, :] += 100.0
    loss_polluted, grad_polluted = SequenceAutoencoder.masked_mse(polluted, x_hat, weight, mask)
    assert loss_polluted == pytest.approx(loss)
    np.testing.assert_allclose(grad[0, 1, :], 0.0)
    np.testing.assert_allclose(grad_polluted[0, 1, :], 0.0)
    assert np.isfinite(base_grad).all()


def test_parameter_dict_aliases_cell_attributes():
    """``params["enc.W_ih"] is enc.W_ih`` must hold for training to work.

    Adam updates parameters in place, so the dict and the cells share buffers.
    Breaking the alias (for instance by assigning loaded arrays into the dict)
    makes a model that looks right, reports the right shapes, and scores with
    its random initialisation.
    """
    model = SequenceAutoencoder(input_size=5, hidden_size=3, latent_size=2, cell="lstm", seed=4)
    assert model.params["enc.W_ih"] is model.enc.W_ih
    assert model.params["dec.b"] is model.dec.b
    assert model.params["W_recon"] is not None

    # load_params must copy into the buffers, not rebind them.
    snapshot = {k: v.copy() for k, v in model.params.items()}
    model.load_params(snapshot)
    assert model.params["enc.W_ih"] is model.enc.W_ih
    np.testing.assert_allclose(model.enc.W_hh, snapshot["enc.W_hh"])

    with pytest.raises(ValueError):
        model.load_params({k: v for k, v in snapshot.items() if k != "W_z"})


def test_parameter_count_and_head_shapes():
    rng = np.random.default_rng(1)
    model = SequenceAutoencoder(input_size=6, hidden_size=4, latent_size=3, cell="lstm", seed=1)
    assert model.n_params == int(sum(v.size for v in model.params.values()))
    assert model.n_params > 0
    out = model.forward(np.ascontiguousarray(rng.normal(size=(4, 2, 6))))
    assert out["x_hat"].shape == (4, 2, 6)
    assert out["y_hat"].shape == (4, 2, 6)
    # The decoder never sees the encoder input: its width is latent + context.
    assert model.dec.input_size == model.latent_size + len(model.exogenous_index)
