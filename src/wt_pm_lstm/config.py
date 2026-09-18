"""Configuration objects for the WT-PM model-13 pipeline.

Every knob that affects a reported number lives here so that an experiment is
fully described by one JSON file (experiment tracking requirement of the
platform contract). Nested dataclasses, ``from_dict``/``to_dict`` round-trip,
and unknown keys are rejected loudly rather than silently ignored.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Mapping, Sequence, Type, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------
# Fidelity tiers
# --------------------------------------------------------------------------
#: Rung 1 of the fidelity ladder: the dependency-light NumPy reference engine.
#: Every rung must reproduce the same scores to within a stated tolerance —
#: see ``docs/fidelity_ladder.md``.
REFERENCE_ENGINE = "numpy-reference"


def _build(cls: Type[T], data: Mapping[str, Any]) -> T:
    """Construct a nested dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"unknown configuration key(s) for {cls.__name__}: {sorted(unknown)}; "
            f"known keys: {sorted(known)}"
        )
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        ftype = known[name].type
        # Nested dataclasses are declared with string annotations from
        # ``from __future__ import annotations``; resolve them lazily.
        target = _resolve_dataclass(ftype)
        if target is not None and isinstance(value, Mapping):
            kwargs[name] = _build(target, value)
        elif target is not None and isinstance(value, list):
            kwargs[name] = [_build(target, v) for v in value]
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def _resolve_dataclass(annotation: Any) -> Any:
    if isinstance(annotation, type):
        return annotation if is_dataclass(annotation) else None
    if isinstance(annotation, str):
        candidate = _REGISTRY.get(annotation)
        return candidate
    return None


_REGISTRY: Dict[str, Any] = {}


def register(cls: Type[T]) -> Type[T]:
    _REGISTRY[cls.__name__] = cls
    return cls


@register
@dataclass
class DataConfig:
    """Synthetic SCADA data generation settings (Layer 1/2)."""

    n_days: int = 60
    sample_minutes: int = 10
    turbine_id: str = "WT-001"
    site_id: str = "SITE-NORTH"
    seed: int = 20260918
    # Machine definition (3.45 MW / 112 m rotor class, 690 V).
    rated_power_kw: float = 3450.0
    rotor_radius_m: float = 56.0
    rated_rotor_rpm: float = 15.2
    cut_in_wind_ms: float = 3.0
    rated_wind_ms: float = 11.0
    cut_out_wind_ms: float = 25.0
    mean_wind_ms: float = 8.6
    mean_ambient_c: float = 9.0
    drivetrain_efficiency: float = 0.945
    sensor_noise: float = 1.0
    #: Fraction of the record that is fault-free and may be used for fitting.
    healthy_fraction: float = 0.45
    #: Fraction of the record reserved for threshold calibration (also healthy).
    calibration_fraction: float = 0.15
    #: Probability per sample of a channel dropping out (quality flagging).
    dropout_rate: float = 0.0015
    #: Probability per sample of a wild measurement spike.
    spike_rate: float = 0.0004
    #: Healthy lead-in at the start of the test band, in days. Without it the
    #: test record would open inside a fault and false-alarm rate could not be
    #: measured on a quiet baseline.
    test_lead_days: float = 3.0
    #: Sensor quantisation scale (physical units) applied after noise.
    quantisation: Dict[str, float] = field(
        default_factory=lambda: {
            "wind_speed_ms": 0.1,
            "ambient_temp_c": 0.1,
            "rotor_speed_rpm": 0.01,
            "pitch_angle_deg": 0.1,
            "yaw_error_deg": 0.1,
            "power_kw": 1.0,
            "main_shaft_torque_knm": 0.1,
            "generator_current_a": 0.5,
            "grid_frequency_hz": 0.001,
            "nacelle_temp_c": 0.1,
            "gearbox_oil_temp_c": 0.1,
            "bearing_vib_rms_mm_s": 0.001,
        }
    )
    fault_kinds: Sequence[str] = (
        "gearbox_thermal",
        "bearing_wear",
        "pitch_misalignment",
        "yaw_error",
        "converter_fault",
        "sensor_freeze",
        "sensor_drift",
    )


@register
@dataclass
class ModelConfig:
    """LSTM sequence-autoencoder architecture and training settings."""

    #: ``lstm`` (default) or ``gru`` — both are implemented in the reference
    #: engine, so the temporal-layer ablation (models 09/13) is reproducible
    #: without a second code base.
    cell: str = "lstm"
    hidden_size: int = 32
    #: Latent width. This is the bottleneck that makes the autoencoder a
    #: detector: too wide and it copies its input, too narrow and it cannot
    #: represent normal operating states.
    latent_size: int = 16
    num_layers: int = 1
    window: int = 36
    stride: int = 1
    #: If True the model reconstructs normalised inputs directly; if False it
    #: predicts the next step from the past (forecasting autoencoder).
    reconstruction: str = "autoencoder"
    learning_rate: float = 2e-3
    batch_size: int = 64
    epochs: int = 40
    gradient_clip: float = 5.0
    weight_decay: float = 1e-5
    #: Dropout applied to the latent representation (variational-free).
    dropout: float = 0.0
    lr_decay: float = 0.5
    lr_decay_every: int = 10
    offline_seed: int = 7
    #: Weight of the one-step forecast head relative to the reconstruction head.
    forecast_weight: float = 0.5
    #: Fraction of the *training* windows held out for early stopping.
    val_fraction: float = 0.12
    #: Epochs without validation improvement before restoring the best weights.
    patience: int = 8
    #: Number of independently initialised models in the ensemble. The spread
    #: across members is the epistemic uncertainty reported per record.
    n_ensemble: int = 3
    #: Optional per-channel loss weights, overriding the defaults below.
    channel_weights: Dict[str, float] = field(default_factory=dict)
    #: Loss weight applied to environment channels (wind, ambient temperature,
    #: grid frequency). They are inputs the model must *observe* so it can
    #: explain the machine's response, but the machine cannot cause them, so
    #: reconstructing them poorly must not dominate the loss.
    exogenous_weight: float = 0.3
    #: Channels handed to the decoder as context (environment + control
    #: commands). Empty means "environment channels plus control channels"; see
    #: :data:`wt_pm_lstm.schema.CONTEXT_CHANNELS`.
    context_channels: Sequence[str] = ()
    #: Channels whose residuals are scored. Empty means "every channel that is
    #: neither environment nor control command": an anomaly score built on
    #: whether the model could predict the wind, or reproduce a pitch command,
    #: is not a machine-health score.
    score_channels: Sequence[str] = ()


@register
@dataclass
class DetectConfig:
    """Threshold calibration, temporal fusion and alarm policy."""

    #: EWMA smoothing factor applied to the per-sample anomaly score.
    ewma_alpha: float = 0.15
    #: Rolling median/mad window for robust normalisation of the score stream.
    normalise_window: int = 288
    #: Quantile of the *calibration* score distribution used as threshold.
    #: When :attr:`threshold_method` is ``"time-budget"`` this is derived at
    #: calibration time from :attr:`target_false_alarm_minutes_per_day`.
    threshold_quantile: float = 0.999
    #: How the alarm threshold is chosen.
    #:
    #: ``"time-budget"`` (default) sets the threshold at the quantile implied by
    #: :attr:`target_false_alarm_minutes_per_day` on the calibration band, so the
    #: operating point is an explicit, reviewable statement about how much alarm
    #: time a crew is willing to spend on healthy machines. ``"quantile"`` uses
    #: :attr:`threshold_quantile` directly.
    #:
    #: Note that the realised rate on the *monitoring* band generally exceeds the
    #: calibration budget: alarm-time is the monotone, budgetable quantity, but
    #: the score distribution itself is non-stationary over weeks (see
    #: :mod:`wt_pm_lstm.drift`). The report prints both so the overshoot is
    #: visible rather than assumed away.
    threshold_method: str = "time-budget"
    #: Operational alarm-time budget for healthy operation, in minutes per day.
    #: The default is one ten-minute sample per day — a rate a crew can absorb
    #: without ignoring the system, and the strictest budget whose resulting
    #: threshold still sits in the sensitive half of the measured
    #: sensitivity/false-alarm curve (see ``operating_point_table``).
    target_false_alarm_minutes_per_day: float = 10.0
    #: Minimum consecutive alarms before an event is declared (alarm counter).
    min_consecutive: int = 3
    #: Alarm clears only after this many consecutive normal samples (hysteresis).
    clear_after: int = 12
    #: Bootstrap resamples for the threshold confidence interval.
    bootstrap_samples: int = 200
    #: Fusion weights over the three score components. They are normalised to
    #: sum to one at use time.
    #:
    #: ``reconstruction`` catches out-of-distribution *states* (a bearing that
    #: starts vibrating), ``temporal`` catches violations of the *dynamics*
    #: (a stuck sensor), and ``trend`` catches slow level shifts that both
    #: residual heads are blind to by construction — a smooth 10-degree oil
    #: temperature ramp is *easy* to reconstruct and *easy* to forecast, so no
    #: residual detector can see it without an explicit trend statistic.
    #: ``level`` compares each response channel with what the operating point
    #: (wind, ambient temperature, pitch, grid frequency) implies, via the
    #: normal-behaviour model. It is the only *absolute* component, and the only
    #: one that sees a channel settling at a new steady level.
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "reconstruction": 0.3,
            "temporal": 0.2,
            "trend": 0.2,
            "level": 0.3,
        }
    )
    #: Sensor-health alarm: a channel whose value does not change by even one
    #: quantisation step for this many consecutive samples is stuck. Sensor and
    #: machine faults are different failure modes and are alarmed separately,
    #: then merged — the way a real condition-monitoring system separates "the
    #: measurement is broken" from "the machine is damaged".
    #:
    #: Exact constancy is used rather than a variance threshold because
    #: legitimately quiet channels exist (rotor speed is constant at rated in
    #: region 3, blade pitch sits at 0 or feathered for hours), and a variance
    #: rule fires on all of them.
    stuck_samples: int = 18
    #: Winsorisation limit (robust sigmas) applied to the level component, so a
    #: single wild glitch cannot dominate an 8-channel average.
    level_clip: float = 6.0
    #: Operator reducing the per-channel fusion to one score per timestep:
    #: ``max`` (worst channel wins — the default for multi-channel alarming),
    #: ``topk`` (mean of the k largest) or ``mean`` (most noise-robust, least
    #: sensitive to single-channel faults).
    channel_reduction: str = "max"
    channel_reduction_k: int = 3
    #: Window (samples) for the trend statistic, in the units of the data. The
    #: default 36 samples is 6 h at 10-minute sampling.
    trend_window: int = 36
    #: Weight for the temporal (prediction) term when fusing.
    temporal_weight: float = 0.5
    #: Persistence marker: when the score's median shift exceeds this many
    #: robust sigmas, flag drift.
    drift_sigma: float = 4.0


@register
@dataclass
class EvalConfig:
    """Evaluation protocol settings (shared across WT-PM models)."""

    #: Samples after a fault onset within which an alarm counts as an event
    #: detection. Defaults to 36 samples = 6 h at 10-minute sampling, which
    #: reflects how fast a maintenance organisation can act — not how fast the
    #: fault develops. Injected faults ramp in over hours by construction, so a
    #: tolerance tighter than the ramp would score physics, not the detector.
    detection_tolerance: int = 36
    #: Legacy alias kept for the event-level report (same units as above).
    event_tolerance: int = 144
    bootstrap_samples: int = 400
    alpha: float = 0.05
    #: Multiples of the calibrated threshold at which the operating-point table
    #: is reported (see :func:`wt_pm_lstm.pipeline.operating_point_table`). An
    #: empty sequence disables the sweep.
    sweep_multipliers: Sequence[float] = (0.6, 0.8, 1.0, 1.25, 1.5, 2.0)


@register
@dataclass
class RunConfig:
    """Top-level experiment description."""

    name: str = "lstm-scada-reference"
    engine: str = REFERENCE_ENGINE
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    #: Where artifacts are written (git-ignored).
    output_dir: str = "artifacts"

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunConfig":
        return _build(cls, payload)

    def to_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        return path

    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def lr_at(self, epoch: int) -> float:
        """Step-decayed learning rate for ``epoch`` (0-indexed)."""
        m = self.model
        steps = epoch // max(1, m.lr_decay_every)
        return m.learning_rate * (m.lr_decay**steps)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def config_fingerprint(cfg: RunConfig) -> str:
    """Stable short hash of a run configuration, used as the model registry key."""
    import hashlib

    blob = json.dumps(cfg.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ``asdict`` re-export keeps older experiment scripts working.
__all__ = [
    "DataConfig",
    "ModelConfig",
    "DetectConfig",
    "EvalConfig",
    "RunConfig",
    "REFERENCE_ENGINE",
    "config_fingerprint",
    "asdict",
    "_build",
]
