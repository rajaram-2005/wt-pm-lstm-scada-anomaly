"""Built-in SCADA simulator so ``pip install wt-pm`` works without extra repos.

When ``wt_pm_lstm`` (model 05) is installed it is preferred. This fallback
matches the 12 canonical channels and injects a labelled fault from 55% of
the record so research/production/serve still run after a bare pip install.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

CHANNELS = (
    "wind_speed_ms", "ambient_temp_c", "rotor_speed_rpm", "pitch_angle_deg",
    "yaw_error_deg", "power_kw", "main_shaft_torque_knm", "generator_current_a",
    "grid_frequency_hz", "nacelle_temp_c", "gearbox_oil_temp_c",
    "bearing_vib_rms_mm_s",
)

FAULT_KINDS = (
    "gearbox_thermal", "bearing_wear", "pitch_misalignment", "yaw_error",
    "converter_fault", "sensor_freeze", "sensor_drift",
)


def simulate_scada(days: float = 8.0, seed: int = 7, turbine_id: str = "WT-01"):
    from wtpm_platform.data import ingest_arrays

    rng = np.random.default_rng(seed)
    n = max(int(days * 24 * 6), 48)
    t = np.arange(n) * 600
    i0 = int(n * 0.55)
    kind = FAULT_KINDS[seed % len(FAULT_KINDS)]

    ws = 8 + 3 * np.sin(np.arange(n) / 36) + rng.normal(0, 0.6, n)
    ws = np.clip(ws, 0.5, 25)
    ambient = 12 + 4 * np.sin(np.arange(n) / 144) + rng.normal(0, 0.3, n)
    rpm = np.clip(0.9 * ws + rng.normal(0, 0.25, n), 0, 22)
    pitch = np.clip((ws - 12) * 2.0, 0, 25) + rng.normal(0, 0.2, n)
    yaw = rng.normal(0, 1.2, n)
    torque = 4.5 * ws + rng.normal(0, 0.8, n)
    omega = rpm * 2 * np.pi / 60.0
    power = 0.94 * (torque * 1e3) * omega / 1e3 + rng.normal(0, 8, n)
    current = np.maximum(power / 0.69, 0) + rng.normal(0, 5, n)
    freq = 50.0 + rng.normal(0, 0.02, n)
    nacelle = ambient + 8 + 0.02 * power / 10 + rng.normal(0, 0.3, n)
    oil = 52 + 0.04 * power / 10 + rng.normal(0, 0.4, n)
    vib = 1.8 + 0.05 * rpm + rng.normal(0, 0.08, n)

    kinds: List[str] = [""] * n
    ramp = np.linspace(0, 1, n - i0)
    if kind == "bearing_wear":
        vib[i0:] += 6 * ramp
    elif kind == "gearbox_thermal":
        oil[i0:] += 18 * ramp
    elif kind == "pitch_misalignment":
        pitch[i0:] += 8 * ramp
        power[i0:] *= (1 - 0.15 * ramp)
    elif kind == "yaw_error":
        yaw[i0:] += 12 * ramp
        power[i0:] *= (1 - 0.1 * ramp)
    elif kind == "converter_fault":
        current[i0:] += 80 * ramp
        freq[i0:] += rng.normal(0, 0.15, n - i0)
    elif kind == "sensor_freeze":
        vib[i0:] = vib[i0]
    else:
        oil[i0:] += np.linspace(0, 6, n - i0)
    for i in range(i0, n):
        kinds[i] = kind

    values = np.column_stack([
        ws, ambient, rpm, pitch, yaw, power, torque, current, freq, nacelle, oil, vib,
    ])
    batch = ingest_arrays(turbine_id, t, values, CHANNELS)
    batch.meta["fault_kind_per_step"] = kinds
    batch.meta["fault_label"] = (np.arange(n) >= i0).astype(int)
    batch.meta["simulator"] = "wt-pm-builtin"
    return batch


def make_batch(days: float, seed: int = 7, turbine_id: str = "WT-01"):
    """Prefer the reference physics simulator; fall back to the built-in one."""
    try:
        from wt_pm_lstm.config import DataConfig
        from wt_pm_lstm.simulate import simulate_timeline, default_fault_schedule
        from wtpm_platform.data import ingest_timeline

        cfg = DataConfig()
        cfg.n_days = days
        cfg.seed = seed
        n_steps = int(days * 24 * 6)
        specs = default_fault_schedule(n_steps, first_fault_step=int(n_steps * 0.55), seed=seed)
        tl = simulate_timeline(cfg, fault_specs=specs, turbine_id=turbine_id)
        batch = ingest_timeline(tl)
        batch.meta["fault_kind_per_step"] = tl.meta.get("fault_kind_per_step")
        batch.meta["fault_label"] = tl.fault_label
        batch.meta["simulator"] = "wt_pm_lstm"
        return batch
    except Exception:
        return simulate_scada(days, seed, turbine_id)
