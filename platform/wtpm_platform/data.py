"""Data ingestion, cleaning/synchronization and the FeaturePipeline.

Sensors/SCADA -> Ingestion -> Cleaning/Sync -> Feature & Representation layer.

The canonical source of truth is model 05's ``TurbineTimeline`` (12 channels,
10-minute cadence, quality codes). This module converts any supported source
into a clean, synchronized ``SensorBatch`` and attaches the derived views the
adapters need: tabular features, sequence windows and vibration surrogates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wtpm_platform.contracts import SensorBatch

try:  # the reference repo is the schema authority when importable
    from wt_pm_lstm.schema import TurbineTimeline, canonical_channels, EXOGENOUS_CHANNELS
    HAVE_REFERENCE = True
except ImportError:  # platform still works from raw arrays / CSV
    HAVE_REFERENCE = False
    def canonical_channels() -> Tuple[str, ...]:  # type: ignore[misc]
        return (
            "wind_speed_ms", "ambient_temp_c", "rotor_speed_rpm", "pitch_angle_deg",
            "yaw_error_deg", "power_kw", "main_shaft_torque_knm", "generator_current_a",
            "grid_frequency_hz", "nacelle_temp_c", "gearbox_oil_temp_c",
            "bearing_vib_rms_mm_s",
        )
    EXOGENOUS_CHANNELS = ("wind_speed_ms", "ambient_temp_c", "grid_frequency_hz")

RESPONSE_CHANNELS: Tuple[str, ...] = tuple(
    c for c in canonical_channels()
    if c not in EXOGENOUS_CHANNELS and c != "pitch_angle_deg"
)

#: channels the safety layer is allowed to depend on (must exist on the MCU)
SAFETY_CHANNELS: Tuple[str, ...] = (
    "bearing_vib_rms_mm_s", "gearbox_oil_temp_c", "rotor_speed_rpm", "power_kw",
)

ELECTRICAL_CHANNELS: Tuple[str, ...] = (
    "generator_current_a", "power_kw", "grid_frequency_hz", "main_shaft_torque_knm",
)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_timeline(timeline: "TurbineTimeline") -> SensorBatch:
    """Ingest a reference-schema timeline (the preferred path)."""
    mask = timeline.mask if timeline.mask is not None else np.ones(timeline.values.shape, bool)
    batch = SensorBatch(
        turbine_id=timeline.turbine_id,
        timestamps=np.asarray(timeline.timestamps),
        channel_names=tuple(timeline.channel_names),
        values=np.asarray(timeline.values, dtype=float),
        meta={
            "mask": np.asarray(mask, bool),
            "fault_label": None if timeline.fault_label is None else np.asarray(timeline.fault_label),
            "fault_kind_per_step": timeline.meta.get("fault_kind_per_step"),
            "site_id": timeline.site_id,
        },
    )
    return batch


def ingest_arrays(
    turbine_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    channel_names: Sequence[str],
) -> SensorBatch:
    """Ingest raw arrays (e.g. from a CSV or an external historian)."""
    values = np.asarray(values, dtype=float)
    if values.shape[1] != len(channel_names):
        raise ValueError("values width does not match channel_names")
    return SensorBatch(
        turbine_id=turbine_id,
        timestamps=np.asarray(timestamps),
        channel_names=tuple(channel_names),
        values=values,
        meta={"mask": np.isfinite(values)},
    )


# ---------------------------------------------------------------------------
# Cleaning / synchronization
# ---------------------------------------------------------------------------
def clean(batch: SensorBatch, max_gap: int = 6) -> SensorBatch:
    """Interpolate short gaps, flag long ones; never invent a sentinel value.

    Rows whose gap exceeds ``max_gap`` samples stay masked; downstream
    consumers receive the mask and must not score masked rows.
    """
    v = batch.values.copy()
    mask = batch.meta.get("mask", np.isfinite(v)).copy()
    mask &= np.isfinite(v)
    for c in range(v.shape[1]):
        col, m = v[:, c], mask[:, c]
        if m.all():
            continue
        idx = np.arange(len(col))
        good = idx[m]
        if len(good) < 2:
            continue
        holes = idx[~m]
        v[holes, c] = np.interp(holes, good, col[good])
        # re-mask holes that sit inside gaps longer than max_gap
        gap_starts = np.where(np.diff(good) > max_gap)[0]
        for g in gap_starts:
            lo, hi = good[g], good[g + 1]
            mask[lo + 1:hi, c] = False
        # otherwise the interpolation is accepted
        short = np.setdiff1d(holes, np.concatenate([np.arange(good[g] + 1, good[g + 1])
                                                    for g in gap_starts]) if len(gap_starts) else [])
        mask[short, c] = True
    out = SensorBatch(
        turbine_id=batch.turbine_id, timestamps=batch.timestamps,
        channel_names=batch.channel_names, values=v,
        operating_state=batch.operating_state,
        meta={**batch.meta, "mask": mask},
    )
    return out


def operating_state(batch: SensorBatch) -> np.ndarray:
    """Classify each step: 0=idle, 1=partial load, 2=rated, 3=curtailed/other."""
    ws = batch.channel("wind_speed_ms") if "wind_speed_ms" in batch.channel_names else None
    pw = batch.channel("power_kw") if "power_kw" in batch.channel_names else None
    n = batch.n_steps
    state = np.full(n, 3, dtype=int)
    if ws is None or pw is None:
        return state
    rated = np.nanpercentile(pw, 97)
    state[(ws < 3.5) | (pw < 0.02 * max(rated, 1e-9))] = 0
    state[(pw >= 0.02 * rated) & (pw < 0.85 * rated)] = 1
    state[pw >= 0.85 * rated] = 2
    return state


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------
@dataclass
class FeaturePipeline:
    """Derives every representation the 25 adapters need, exactly once.

    - tabular features (rolling stats + physics residual) for m10..m15, m17
    - sequence windows for m02..m06, m22
    - vibration surrogate waveforms for m01, m07, m08 (honest surrogates:
      synthesised from the 10-min vibration RMS channel because no repo ships
      a high-frequency DAQ; flagged in metadata so nobody mistakes them for
      real waveforms)
    """

    window: int = 36
    stride: int = 2
    roll: int = 12
    vib_len: int = 1024

    def transform(self, batch: SensorBatch) -> SensorBatch:
        t0 = time.perf_counter()
        batch = clean(batch)
        batch.operating_state = operating_state(batch)
        feats, names = self._tabular(batch)
        batch.features, batch.feature_names = feats, names
        batch.windows, batch.window_index = self._windows(batch)
        batch.vib_waveforms, batch.vib_index = self._vibration_surrogate(batch)
        batch.meta["feature_pipeline_ms"] = (time.perf_counter() - t0) * 1000
        batch.meta["vibration_is_surrogate"] = True
        return batch

    # -- tabular ------------------------------------------------------------
    def _tabular(self, batch: SensorBatch) -> Tuple[np.ndarray, List[str]]:
        cols: List[np.ndarray] = []
        names: List[str] = []
        r = self.roll
        for ch in batch.channel_names:
            x = batch.channel(ch)
            cols.append(x); names.append(ch)
            cols.append(_rolling_mean(x, r)); names.append(f"{ch}__mean{r}")
            cols.append(_rolling_std(x, r)); names.append(f"{ch}__std{r}")
        # physics residual: P_mech = tau * omega  (model 18's constraint,
        # promoted to a shared feature because several models benefit)
        if {"main_shaft_torque_knm", "rotor_speed_rpm", "power_kw"} <= set(batch.channel_names):
            tau = batch.channel("main_shaft_torque_knm") * 1e3  # kNm -> Nm
            omega = batch.channel("rotor_speed_rpm") * 2 * np.pi / 60.0
            p_mech_kw = tau * omega / 1e3
            resid = batch.channel("power_kw") - 0.94 * p_mech_kw  # generator eff.
            cols.append(resid); names.append("physics_power_residual_kw")
        if {"power_kw", "wind_speed_ms"} <= set(batch.channel_names):
            ws = np.maximum(batch.channel("wind_speed_ms"), 0.5)
            cols.append(batch.channel("power_kw") / ws ** 3); names.append("power_per_v3")
        return np.column_stack(cols), names

    # -- windows -------------------------------------------------------------
    def _windows(self, batch: SensorBatch) -> Tuple[np.ndarray, np.ndarray]:
        W, S = self.window, self.stride
        T = batch.n_steps
        if T < W + 1:
            return np.zeros((0, W, batch.values.shape[1])), np.zeros(0, int)
        ends = np.arange(W, T, S)
        wins = np.stack([batch.values[e - W:e] for e in ends])
        return wins, ends

    # -- vibration surrogate --------------------------------------------------
    def _vibration_surrogate(self, batch: SensorBatch) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if "bearing_vib_rms_mm_s" not in batch.channel_names:
            return None, None
        rms = batch.channel("bearing_vib_rms_mm_s")
        rpm = batch.channel("rotor_speed_rpm") if "rotor_speed_rpm" in batch.channel_names else np.full_like(rms, 12.0)
        L = self.vib_len
        idx = np.arange(self.window, batch.n_steps, max(self.stride * 4, 8))
        rng = np.random.default_rng(1234)
        t = np.arange(L) / L
        waves = np.empty((len(idx), L), dtype=np.float32)
        healthy_rms = np.nanpercentile(rms, 30)
        for j, i in enumerate(idx):
            f_rot = max(rpm[i], 1.0) / 60.0 * 64.0     # rotations per waveform
            base = np.sin(2 * np.pi * f_rot * t) + 0.4 * np.sin(2 * np.pi * 3.2 * f_rot * t)
            sev = max(rms[i] / max(healthy_rms, 1e-6) - 1.0, 0.0)
            # bearing defect signature: periodic impulses whose energy scales
            # with the observed RMS elevation
            impulses = (np.sin(2 * np.pi * 4.7 * f_rot * t) > 0.995).astype(float)
            wave = base + sev * 3.0 * impulses * np.exp(-3 * t) + 0.25 * rng.standard_normal(L)
            waves[j] = (rms[i] / max(healthy_rms, 1e-6)) * wave
        return waves, idx


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(np.nan_to_num(x, nan=0.0), 0, 0.0))
    out = (c[w:] - c[:-w]) / w                       # length: len(x) - w + 1
    pad = len(x) - len(out)
    return np.concatenate([np.full(pad, out[0] if len(out) else 0.0), out])


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    m = _rolling_mean(x, w)
    m2 = _rolling_mean(x ** 2, w)
    return np.sqrt(np.maximum(m2 - m ** 2, 0.0))


# ---------------------------------------------------------------------------
# Labels (research mode only)
# ---------------------------------------------------------------------------
def labels_from_batch(batch: SensorBatch) -> Optional[np.ndarray]:
    lab = batch.meta.get("fault_label")
    return None if lab is None else np.asarray(lab).astype(int)


def fault_kinds_from_batch(batch: SensorBatch) -> Optional[List[str]]:
    kinds = batch.meta.get("fault_kind_per_step")
    return None if kinds is None else list(kinds)


def degradation_target(batch: SensorBatch, healthy_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Health indicator in [0, 1.5] used to supervise RUL/degradation surrogates.

    Built from vibration + gearbox-oil deviation from their *healthy* baselines
    (the ``healthy_mask`` rows when given, else the record's lower quartile).
    This is a *proxy target*, clearly labelled as such — no repo ships real
    run-to-failure data.
    """
    hi = np.zeros(batch.n_steps)
    for ch, wt in (("bearing_vib_rms_mm_s", 0.6), ("gearbox_oil_temp_c", 0.4)):
        if ch in batch.channel_names:
            x = batch.channel(ch)
            ref = x[healthy_mask] if healthy_mask is not None and healthy_mask.any() else x
            base = np.nanpercentile(ref, 50)
            spread = max(np.nanpercentile(ref, 95) - np.nanpercentile(ref, 5), 1e-6)
            # per-channel clip is generous (2.5) so a single-channel extreme —
            # e.g. severe vibration with normal oil temp — can still push the
            # indicator past the trip level on its own
            hi += wt * np.clip((x - base) / (3.0 * spread), 0, 2.5)
    return np.clip(hi, 0, 1.5)
