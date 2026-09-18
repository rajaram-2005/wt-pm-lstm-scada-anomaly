"""The normal-behaviour model: the only *absolute* component of the score.

Reconstruction and forecast residuals are blind to a channel that settles at a
new, stable level — a slowly fouling gearbox oil cooler raises a temperature by
10 °C over two days, and a model that only asks "is this hard to reconstruct?"
answers no. The normal-behaviour model asks a different question ("what should
this channel be, given the conditions?") and these tests pin that behaviour.
"""

from __future__ import annotations

import numpy as np

from wt_pm_lstm.baseline import NormalBehaviourModel
from wt_pm_lstm.schema import CONTEXT_CHANNELS, default_score_channels


def _toy_problem(n=4000, seed=0, noise=0.05):
    """A response channel driven by two context channels, plus a quiet one."""
    rng = np.random.default_rng(seed)
    wind = rng.uniform(3.0, 20.0, size=n)
    ambient = rng.uniform(-10.0, 30.0, size=n)
    grid = 50.0 + rng.normal(0.0, 0.01, size=n)
    pitch = np.clip((wind - 12.0) * 0.6, 0.0, 25.0)
    # Power is a smooth, strongly wind-driven response; oil temperature follows
    # ambient plus load. Both are exactly what an NBM should capture.
    power = 30.0 * wind**2 * np.cos(np.radians(pitch)) + rng.normal(0, noise * 100, size=n)
    oil = 30.0 + 0.5 * ambient + 0.4 * power / 100.0 + rng.normal(0, noise, size=n)
    torque = power / 10.0 + rng.normal(0, noise, size=n)
    channels = ("wind_speed_ms", "ambient_temp_c", "grid_frequency_hz", "pitch_angle_deg",
                "power_kw", "gearbox_oil_temp_c", "main_shaft_torque_knm")
    values = np.column_stack([wind, ambient, grid, pitch, power, oil, torque])
    mask = np.ones_like(values, dtype=bool)
    return values, mask, channels


def _residual_column(model: NormalBehaviourModel, channels, name: str) -> int:
    """Residuals are returned for the *scored* channels only, in model order."""
    absolute = channels.index(name)
    hits = np.flatnonzero(np.asarray(model.target_indices) == absolute)
    assert hits.size == 1, f"{name} is not a scored target of this model"
    return int(hits[0])


def test_nbm_recovers_a_condition_driven_channel():
    values, mask, channels = _toy_problem()
    score_mask = np.array([c in set(default_score_channels(channels)) for c in channels])
    model = NormalBehaviourModel.fit(values, mask, channels, CONTEXT_CHANNELS, score_mask)
    r2 = model.r2(values, mask)
    assert r2["power_kw"] > 0.99
    assert r2["gearbox_oil_temp_c"] > 0.95
    assert all(np.isfinite(list(r2.values())))


def test_nbm_flags_a_settled_level_shift_that_residuals_cannot_see():
    """The failure mode this model exists for: a new, perfectly stable level."""
    values, mask, channels = _toy_problem()
    score_mask = np.array([c in set(default_score_channels(channels)) for c in channels])
    model = NormalBehaviourModel.fit(values, mask, channels, CONTEXT_CHANNELS, score_mask)

    healthy = model.residuals(values)
    oil_abs = channels.index("gearbox_oil_temp_c")
    oil = _residual_column(model, channels, "gearbox_oil_temp_c")
    shifted = values.copy()
    shifted[:, oil_abs] += 8.0  # a stable +8 degC offset, no extra variance
    abnormal = model.residuals(shifted)

    h = np.abs(healthy[:, oil] - np.median(healthy[:, oil]))
    a = np.abs(abnormal[:, oil] - np.median(healthy[:, oil]))
    assert a.mean() > 10.0 * h.mean() + 1.0


def test_nbm_round_trip_is_exact():
    values, mask, channels = _toy_problem(n=800)
    score_mask = np.array([c in set(default_score_channels(channels)) for c in channels])
    model = NormalBehaviourModel.fit(values, mask, channels, CONTEXT_CHANNELS, score_mask)
    payload = model.to_dict()
    restored = NormalBehaviourModel.from_dict(payload)
    np.testing.assert_allclose(model.predict(values), restored.predict(values))
    assert restored.target_indices.tolist() == model.target_indices.tolist()


def test_nbm_is_linear_in_its_features_and_ridge_regularised():
    """A fit on a small, perfectly-linear problem stays in the right place."""
    rng = np.random.default_rng(2)
    n = 500
    wind = rng.uniform(4.0, 18.0, size=n)
    ambient = rng.uniform(0.0, 25.0, size=n)
    grid = np.full(n, 50.0)
    pitch = np.zeros(n)
    power = 12.0 * wind**2 + rng.normal(0, 1.0, size=n)
    channels = ("wind_speed_ms", "ambient_temp_c", "grid_frequency_hz", "pitch_angle_deg", "power_kw")
    values = np.column_stack([wind, ambient, grid, pitch, power])
    mask = np.ones_like(values, dtype=bool)
    score_mask = np.array([c in set(default_score_channels(channels)) for c in channels])
    model = NormalBehaviourModel.fit(values, mask, channels, CONTEXT_CHANNELS, score_mask)
    assert model.r2(values, mask)["power_kw"] > 0.999
    # Residuals are centred on the fitted data, not left with a systematic bias.
    resid = model.residuals(values)
    assert abs(float(np.mean(resid[:, _residual_column(model, channels, "power_kw")]))) < 1.0
