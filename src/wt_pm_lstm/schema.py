"""WT-PM unified SCADA data contract (``wt-pm.scada.v1``).

This module is the *only* thing other WT-PM repositories need in order to
exchange data with model 13. It defines three things:

1. :data:`CHANNELS` — the canonical SCADA channel dictionary: name, unit,
   physical plausibility bounds and the physical role each signal plays.
2. :class:`TurbineTimeline` — the transport object for one turbine's
   synchronised sensor timeline (Layer 1 of the fabric).
3. :class:`AnomalyRecord` — the output contract emitted by the anomaly
   layer, designed to be consumed by the diagnosis, prognostics, XAI and
   graph layers downstream.

Design rules
------------
* Units are SI-adjacent and explicit in every channel name suffix. Silent unit
  changes are the most common cause of multi-repository drift, so a channel
  name must never be reinterpreted with a different unit.
* Missing data is represented by ``mask=False`` plus a quality flag, never by a
  magic value such as ``-999``. Detectors must be able to distinguish "sensor
  said zero" from "sensor said nothing".
* Everything serialises to plain JSON so the contract is language-neutral.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "wt-pm.scada.v1"
ANOMALY_SCHEMA_VERSION = "wt-pm.anomaly.v1"

# --------------------------------------------------------------------------
# Quality flags
# --------------------------------------------------------------------------
QUALITY_OK = 0
QUALITY_MISSING = 1
QUALITY_IMPUTED = 2
QUALITY_SUSPECT = 3
QUALITY_STUCK = 4
QUALITY_OUT_OF_RANGE = 5

QUALITY_FLAGS: Dict[int, str] = {
    QUALITY_OK: "ok",
    QUALITY_MISSING: "missing",
    QUALITY_IMPUTED: "imputed",
    QUALITY_SUSPECT: "suspect",
    QUALITY_STUCK: "stuck",
    QUALITY_OUT_OF_RANGE: "out_of_range",
}

# Flags that mean "do not learn from or score this sample".
INVALID_QUALITY = (QUALITY_MISSING, QUALITY_OUT_OF_RANGE)

#: Sampling convention for SCADA aggregates. Vibration is *not* represented
#: here at SCADA resolution; high-frequency accelerometer streams belong to the
#: vibration plane (models 01/03/19) and are summarised into this channel.
CHANNELS_META = {
    "native_frequency": "10min_mean",
    "timezone": "UTC",
    "timestamp_units": "unix_seconds",
}


@dataclass(frozen=True)
class ChannelSpec:
    """A single canonical SCADA channel."""

    name: str
    unit: str
    role: str
    description: str
    plausible_min: float
    plausible_max: float

    def in_range(self, value: float) -> bool:
        return self.plausible_min <= value <= self.plausible_max

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Canonical channel dictionary. Roles follow the fabric's physical-asset
#: breakdown (rotor / gearbox / main bearing / generator / converter / grid).
CHANNELS: Tuple[ChannelSpec, ...] = (
    ChannelSpec(
        "wind_speed_ms", "m/s", "environment",
        "Hub-height wind speed, 10-minute mean.", 0.0, 45.0,
    ),
    ChannelSpec(
        "ambient_temp_c", "degC", "environment",
        "Ambient air temperature.", -45.0, 60.0,
    ),
    ChannelSpec(
        "rotor_speed_rpm", "rpm", "rotor",
        "Rotor speed. Region-2 tip-speed-ratio tracking, region-3 rated speed.",
        0.0, 30.0,
    ),
    ChannelSpec(
        "pitch_angle_deg", "deg", "rotor",
        "Collective blade pitch angle. Non-zero only above rated wind speed.",
        -5.0, 95.0,
    ),
    ChannelSpec(
        "yaw_error_deg", "deg", "rotor",
        "Absolute nacelle-to-wind misalignment. Small when the yaw system works.",
        0.0, 180.0,
    ),
    ChannelSpec(
        "power_kw", "kW", "response",
        "Active electrical power. Negative while idling / consuming auxiliaries.",
        -500.0, 20000.0,
    ),
    ChannelSpec(
        "main_shaft_torque_knm", "kNm", "generator",
        "Main (low-speed) shaft torque, derived from power and rotor speed.",
        0.0, 3000.0,
    ),
    ChannelSpec(
        "generator_current_a", "A", "electrical",
        "Stator current magnitude. Coupled to torque through the drive train.",
        0.0, 3500.0,
    ),
    ChannelSpec(
        "grid_frequency_hz", "Hz", "grid",
        "Grid frequency at the point of interconnection.", 47.0, 53.0,
    ),
    ChannelSpec(
        "nacelle_temp_c", "degC", "thermal",
        "Nacelle interior air temperature (thermal lag of ambient + load).",
        -45.0, 90.0,
    ),
    ChannelSpec(
        "gearbox_oil_temp_c", "degC", "thermal",
        "Gearbox oil sump temperature. Load-heated, ambient-cooled.",
        -45.0, 110.0,
    ),
    ChannelSpec(
        "bearing_vib_rms_mm_s", "mm/s", "vibration",
        "Main-bearing vibration RMS velocity, band 10-1000 Hz, 10-minute mean.",
        0.0, 60.0,
    ),
)

CHANNEL_INDEX: Dict[str, ChannelSpec] = {c.name: c for c in CHANNELS}

#: Channels whose behaviour is imposed by the environment rather than by the
#: control system. Used by the residual attribution to separate "the wind
#: changed" from "the machine changed".
EXOGENOUS_CHANNELS: Tuple[str, ...] = ("wind_speed_ms", "ambient_temp_c", "grid_frequency_hz")

#: Actuation/control-state channels. The controller *commands* these in
#: response to the conditions, so they describe what the machine was asked to
#: do, not how healthy it is. Blade pitch is the clear case: it is 0 below rated
#: wind speed, rises steeply above it and parks feathered in storms, which makes
#: its distribution trimodal. It is therefore context for the model, never a
#: scored signal — the *response* to that command (power, torque, vibration,
#: temperatures) is what carries health information.
CONTROL_CHANNELS: Tuple[str, ...] = ("pitch_angle_deg",)

#: Channels handed to the decoder as context: the model must explain the
#: machine's response *given* the conditions and the commands it received.
CONTEXT_CHANNELS: Tuple[str, ...] = EXOGENOUS_CHANNELS + CONTROL_CHANNELS


def default_score_channels(channel_names: Sequence[str] = ()) -> Tuple[str, ...]:
    """Channels whose residuals form the anomaly score.

    Everything that is neither environment nor control command: the machine's
    own response. Scoring the wind measures the weather; scoring the pitch
    command measures the controller.
    """
    names = tuple(channel_names) if len(channel_names) else canonical_channels()
    return tuple(n for n in names if n not in CONTEXT_CHANNELS)


def canonical_channels() -> Tuple[str, ...]:
    """Return the canonical channel ordering."""
    return tuple(c.name for c in CHANNELS)


def get_channel(name: str) -> ChannelSpec:
    """Look up a channel, raising a helpful error if it is not canonical."""
    try:
        return CHANNEL_INDEX[name]
    except KeyError:  # pragma: no cover - message quality matters more than the branch
        raise KeyError(
            f"unknown channel {name!r}; known channels: {', '.join(canonical_channels())}"
        ) from None


# --------------------------------------------------------------------------
# Transport object
# --------------------------------------------------------------------------
@dataclass
class TurbineTimeline:
    """One turbine's synchronised multi-channel SCADA timeline.

    Parameters
    ----------
    turbine_id, site_id
        Asset identifiers. ``site_id`` matters for the wind-farm graph layer
        (model 08): turbines at the same site are coupled through wakes.
    channel_names
        Ordered channel names, must all be canonical.
    values
        ``(T, D)`` float array of sensor values.
    timestamps
        ``(T,)`` integer Unix seconds, strictly increasing, regular spacing.
    mask
        ``(T, D)`` bool array. ``False`` marks a sample that must not be used
        for fitting and must be excluded from the reconstruction loss.
    quality
        ``(T, D)`` uint8 array of :data:`QUALITY_FLAGS` codes.
    fault_label, fault_type
        Optional ground truth. Present in simulation and in labelled benchmark
        exports; absent in production.
    """

    turbine_id: str
    channel_names: Sequence[str]
    values: np.ndarray
    timestamps: np.ndarray
    site_id: str = "unknown-site"
    mask: Optional[np.ndarray] = None
    quality: Optional[np.ndarray] = None
    fault_label: Optional[np.ndarray] = None
    fault_type: Tuple[str, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        self.timestamps = np.asarray(self.timestamps, dtype=np.int64)
        self.channel_names = tuple(self.channel_names)

        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D (T, D); got shape {self.values.shape}")
        n_steps, n_channels = self.values.shape
        if len(self.channel_names) != n_channels:
            raise ValueError(
                f"channel_names has {len(self.channel_names)} entries but values has "
                f"{n_channels} columns"
            )
        if len(self.timestamps) != n_steps:
            raise ValueError(
                f"timestamps has {len(self.timestamps)} entries but values has "
                f"{n_steps} rows"
            )
        for name in self.channel_names:
            get_channel(name)

        if self.mask is None:
            self.mask = np.ones((n_steps, n_channels), dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.values.shape:
                raise ValueError("mask shape must match values shape")

        if self.quality is None:
            self.quality = np.where(self.mask, QUALITY_OK, QUALITY_MISSING).astype(np.uint8)
        else:
            self.quality = np.asarray(self.quality, dtype=np.uint8)
            if self.quality.shape != self.values.shape:
                raise ValueError("quality shape must match values shape")

        codes = set(np.unique(self.quality).tolist())
        unknown = codes - set(QUALITY_FLAGS)
        if unknown:
            raise ValueError(f"unknown quality codes: {sorted(unknown)}")

        if self.fault_label is not None:
            self.fault_label = np.asarray(self.fault_label, dtype=bool)
            if self.fault_label.shape != (n_steps,):
                raise ValueError("fault_label shape must be (T,)")

    # -- geometry ---------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[1])

    @property
    def sampling_seconds(self) -> Optional[int]:
        """Modal sampling interval, or ``None`` if fewer than two samples."""
        if self.n_steps < 2:
            return None
        deltas = np.diff(self.timestamps)
        return int(np.median(deltas))

    def channel(self, name: str) -> np.ndarray:
        """Return one channel as a ``(T,)`` array."""
        return self.values[:, self.channel_names.index(name)]

    def channel_slice(self, names: Iterable[str]) -> np.ndarray:
        """Return a ``(T, len(names))`` view over the requested channels."""
        idx = [self.channel_names.index(n) for n in names]
        return self.values[:, idx]

    # -- validation -------------------------------------------------------
    def validate(self, strict_bounds: bool = False) -> List[str]:
        """Return a list of contract violations (empty means conformant).

        ``strict_bounds`` additionally flags values outside the channel's
        physical plausibility envelope. The simulator produces occasional
        out-of-range spikes on purpose, so production callers normally leave
        this off and rely on the quality flags instead.
        """
        problems: List[str] = []
        if self.n_steps == 0:
            problems.append("timeline is empty")
            return problems
        if np.any(np.diff(self.timestamps) <= 0):
            problems.append("timestamps are not strictly increasing")
        if not np.all(np.isfinite(self.values[self.mask])):
            problems.append("masked-valid samples contain NaN or Inf")
        if np.any(np.isnan(self.values[~self.mask])):
            # NaN is allowed only where the mask already declares invalidity.
            pass
        if strict_bounds:
            for j, name in enumerate(self.channel_names):
                spec = get_channel(name)
                col = self.values[:, j]
                ok = self.mask[:, j] & np.isfinite(col)
                if ok.any():
                    lo, hi = float(col[ok].min()), float(col[ok].max())
                    if lo < spec.plausible_min or hi > spec.plausible_max:
                        problems.append(
                            f"{name}: range [{lo:.3f}, {hi:.3f}] outside plausible "
                            f"[{spec.plausible_min}, {spec.plausible_max}] {spec.unit}"
                        )
        return problems

    # -- slicing ----------------------------------------------------------
    def slice(self, start: int, stop: int) -> "TurbineTimeline":
        """Return the half-open ``[start, stop)`` sub-timeline."""
        if start < 0 or stop > self.n_steps or start >= stop:
            raise ValueError(f"invalid slice [{start}, {stop}) for {self.n_steps} steps")
        return TurbineTimeline(
            turbine_id=self.turbine_id,
            site_id=self.site_id,
            channel_names=self.channel_names,
            values=self.values[start:stop].copy(),
            timestamps=self.timestamps[start:stop].copy(),
            mask=self.mask[start:stop].copy(),
            quality=self.quality[start:stop].copy(),
            fault_label=None if self.fault_label is None else self.fault_label[start:stop].copy(),
            fault_type=self.fault_type,
            # Per-step meta arrays must be sliced with the data, or step-relative
            # lookups (e.g. the fault kind of an event onset) silently read the
            # wrong position of the parent record.
            meta={
                k: (v[start:stop] if isinstance(v, (list, np.ndarray)) and len(v) == self.n_steps else v)
                for k, v in self.meta.items()
            },
        )

    # -- serialisation ----------------------------------------------------
    def to_dict(self, include_values: bool = True) -> Dict[str, Any]:
        """JSON-safe representation of the timeline."""
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "turbine_id": self.turbine_id,
            "site_id": self.site_id,
            "channel_names": list(self.channel_names),
            "timestamps": [int(t) for t in self.timestamps],
            "meta": dict(self.meta),
        }
        if include_values:
            payload["values"] = [
                [None if not self.mask[i, j] else float(self.values[i, j]) for j in range(self.n_channels)]
                for i in range(self.n_steps)
            ]
            payload["quality"] = self.quality.tolist()
            if self.fault_label is not None:
                payload["fault_label"] = [bool(v) for v in self.fault_label]
            if self.fault_type:
                payload["fault_type"] = list(self.fault_type)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TurbineTimeline":
        """Rebuild a timeline from :meth:`to_dict` output."""
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"schema mismatch: payload is {version!r}, this build speaks {SCHEMA_VERSION!r}"
            )
        channel_names = tuple(payload["channel_names"])
        raw_values = payload["values"]
        mask = np.ones((len(raw_values), len(channel_names)), dtype=bool)
        values = np.zeros((len(raw_values), len(channel_names)), dtype=np.float64)
        for i, row in enumerate(raw_values):
            for j, v in enumerate(row):
                if v is None:
                    mask[i, j] = False
                else:
                    values[i, j] = float(v)
        quality = payload.get("quality")
        return cls(
            turbine_id=payload["turbine_id"],
            site_id=payload.get("site_id", "unknown-site"),
            channel_names=channel_names,
            values=values,
            timestamps=np.asarray(payload["timestamps"], dtype=np.int64),
            mask=mask,
            quality=None if quality is None else np.asarray(quality, dtype=np.uint8),
            fault_label=(
                None
                if payload.get("fault_label") is None
                else np.asarray(payload["fault_label"], dtype=bool)
            ),
            fault_type=tuple(payload.get("fault_type", ())),
            meta=dict(payload.get("meta", {})),
        )

    def to_csv(self, path: str) -> str:
        """Write a flat CSV (timestamp first, then channels, then quality)."""
        header = ["timestamp"] + list(self.channel_names) + [f"q_{c}" for c in self.channel_names]
        rows = []
        for i in range(self.n_steps):
            row = [str(int(self.timestamps[i]))]
            row += [
                "" if not self.mask[i, j] else f"{self.values[i, j]:.6f}"
                for j in range(self.n_channels)
            ]
            row += [str(int(self.quality[i, j])) for j in range(self.n_channels)]
            rows.append(",".join(row))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join([",".join(header), *rows]) + "\n")
        return path


# --------------------------------------------------------------------------
# Anomaly output contract
# --------------------------------------------------------------------------
@dataclass
class AnomalyRecord:
    """One scored timestep, as emitted by WT-PM model 13.

    This is the hand-off object for the downstream fabric. The diagnosis layer
    (models 20/18/25) consumes ``score``/``attribution``; the temporal layer
    (models 09/13/11/21) consumes the time series of records; the XAI layer
    (model 24) consumes ``attribution`` and ``explanation``.
    """

    turbine_id: str
    timestamp: int
    score: float
    threshold: float
    alarm: bool
    model_id: str
    schema_version: str = ANOMALY_SCHEMA_VERSION
    uncertainty_std: float = 0.0
    attribution: Dict[str, float] = field(default_factory=dict)
    recommended_action: str = "continue"
    explanation: str = ""
    drift_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "turbine_id": self.turbine_id,
            "timestamp": int(self.timestamp),
            "score": round(float(self.score), 6),
            "threshold": round(float(self.threshold), 6),
            "alarm": bool(self.alarm),
            "uncertainty_std": round(float(self.uncertainty_std), 6),
            "attribution": {k: round(float(v), 6) for k, v in self.attribution.items()},
            "recommended_action": self.recommended_action,
            "explanation": self.explanation,
            "drift_detected": bool(self.drift_detected),
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnomalyRecord":
        return cls(
            turbine_id=payload["turbine_id"],
            timestamp=int(payload["timestamp"]),
            score=float(payload["score"]),
            threshold=float(payload["threshold"]),
            alarm=bool(payload["alarm"]),
            model_id=payload["model_id"],
            schema_version=payload.get("schema_version", ANOMALY_SCHEMA_VERSION),
            uncertainty_std=float(payload.get("uncertainty_std", 0.0)),
            attribution=dict(payload.get("attribution", {})),
            recommended_action=payload.get("recommended_action", "continue"),
            explanation=payload.get("explanation", ""),
            drift_detected=bool(payload.get("drift_detected", False)),
        )


def describe_schema() -> Dict[str, Any]:
    """Full machine-readable schema description, served at ``/v1/schema``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "anomaly_schema_version": ANOMALY_SCHEMA_VERSION,
        "conventions": dict(CHANNELS_META),
        "quality_flags": {str(k): v for k, v in QUALITY_FLAGS.items()},
        "channels": [c.to_dict() for c in CHANNELS],
        "exogenous_channels": list(EXOGENOUS_CHANNELS),
    }


def quality_name(code: int) -> str:
    return QUALITY_FLAGS.get(int(code), f"unknown({int(code)})")


def _finite(x: float) -> bool:  # pragma: no cover - small helper
    return isinstance(x, (int, float)) and math.isfinite(float(x))
