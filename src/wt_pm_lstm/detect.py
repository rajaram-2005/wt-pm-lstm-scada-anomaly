"""Detection layer: scoring, fusion, threshold calibration and alarming.

The detector is deliberately built from named, individually testable pieces —
residual normalisation, score fusion, threshold calibration, alarm hysteresis —
because each one is a defensible modelling decision that a reviewer should be
able to challenge in isolation.

Scoring chain
-------------
1. Standardise inputs with :class:`~wt_pm_lstm.windows.ChannelScaler`
   (fit on the healthy training band).
2. Per ensemble member, compute reconstruction and one-step forecast residuals
   for every window.
3. Normalise each channel's residual by its own robust scale, fitted on the
   **calibration** band (never on the test band). Without this step the score is
   dominated by whichever channel happens to have the largest physical units.
4. Fuse the two heads and average over channels, then over the ensemble.
5. Aggregate overlapping windows onto the timeline.
6. EWMA-smooth, and alarm with hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wt_pm_lstm.baseline import NormalBehaviourModel
from wt_pm_lstm.config import DetectConfig, RunConfig
from wt_pm_lstm.nn import SequenceAutoencoder
from wt_pm_lstm.schema import AnomalyRecord, TurbineTimeline, get_channel
from wt_pm_lstm.windows import ChannelScaler, Windows, make_windows

Array = np.ndarray


# --------------------------------------------------------------------------
# Residual normalisation
# --------------------------------------------------------------------------
@dataclass
class ResidualNormaliser:
    """Per-channel robust scale of the residual stream.

    Fit on the calibration band: ``median`` removes a systematic bias (an
    autoencoder is rarely perfectly unbiased per channel) and ``scale`` converts
    residuals to comparable units. ``floor`` guards against a channel whose
    calibration residual is exactly constant.
    """

    channel_names: Tuple[str, ...]
    median: Array
    scale: Array
    n_samples: int = 0

    @classmethod
    def fit(cls, residuals: Array, channel_names: Tuple[str, ...], floor: float = 1e-6) -> "ResidualNormaliser":
        # ``residuals`` is (K, D) after flattening windows.
        r = np.asarray(residuals, dtype=np.float64).reshape(-1, residuals.shape[-1])
        med = np.median(r, axis=0)
        mad = np.median(np.abs(r - med), axis=0)
        scale = 1.4826 * mad
        scale = np.where(scale < floor, np.maximum(np.std(r, axis=0), floor), scale)
        return cls(channel_names=tuple(channel_names), median=med, scale=scale, n_samples=int(r.shape[0]))

    def z(self, residuals: Array) -> Array:
        """Signed robust z-score of a residual array."""
        return (residuals - self.median) / self.scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_names": list(self.channel_names),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualNormaliser":
        return cls(
            channel_names=tuple(payload["channel_names"]),
            median=np.asarray(payload["median"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            n_samples=int(payload.get("n_samples", 0)),
        )


# --------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------
@dataclass
class Threshold:
    """Alarm threshold with a block-bootstrap confidence interval."""

    value: float
    quantile: float
    exceedance_rate: float
    ci_low: float
    ci_high: float
    n_calibration: int
    method: str = "calibration-quantile"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "quantile": self.quantile,
            "exceedance_rate": self.exceedance_rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_calibration": self.n_calibration,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Threshold":
        return cls(**{k: payload[k] for k in (
            "value", "quantile", "exceedance_rate", "ci_low", "ci_high", "n_calibration",
        )}, method=payload.get("method", "calibration-quantile"))


def alarm_event_count(scores: Array, value: float, min_consecutive: int, clear_after: int) -> int:
    """Number of alarm *events* a threshold would raise on a score trace."""
    stack = AlarmStateMachine(min_consecutive, clear_after)
    alarm, _ = stack.run(np.asarray(scores, dtype=np.float64) > float(value))
    if alarm.size == 0:
        return 0
    return int(np.count_nonzero(alarm[1:] & ~alarm[:-1]) + int(alarm[0]))


def calibrate_threshold(
    scores: Array,
    quantile: float,
    rng: np.random.Generator,
    bootstrap_samples: int = 200,
    block: int = 288,
) -> Threshold:
    """Pick the alarm threshold from the calibration score distribution.

    The confidence interval uses a **moving-block bootstrap**: SCADA scores are
    strongly autocorrelated, so an i.i.d. bootstrap understates the variance of
    the quantile and would make the threshold look far more certain than it is.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 10:
        raise ValueError("need at least 10 calibration scores to set a threshold")
    value = float(np.quantile(s, quantile))
    n = s.size
    block = int(max(1, min(block, n // 4 if n >= 8 else 1)))
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, max(1, n - block + 1))
    boots = np.empty(max(1, bootstrap_samples))
    for b in range(boots.size):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        sample = np.concatenate([s[i : i + block] for i in starts])[:n]
        boots[b] = np.quantile(sample, quantile)
    exceed = float(np.mean(s > value)) if value < np.inf else 0.0
    return Threshold(
        value=value,
        quantile=float(quantile),
        exceedance_rate=exceed,
        ci_low=float(np.quantile(boots, 0.025)),
        ci_high=float(np.quantile(boots, 0.975)),
        n_calibration=int(n),
    )


# --------------------------------------------------------------------------
# Temporal fusion helpers
# --------------------------------------------------------------------------
def aggregate_windows(
    window_values: Array, starts: Array, n_steps: int, window: int
) -> Tuple[Array, Array]:
    """Average overlapping window values onto the timeline.

    Every timestep is covered by up to ``window`` windows; averaging all of them
    (rather than, say, taking the first) is what makes the per-timestep score
    stable near a fault onset, where coverage would otherwise be ragged.
    Returns ``(mean, count)``.
    """
    steps = starts[:, None] + np.arange(window)[None, :]
    trailing = window_values.shape[2:]
    acc = np.zeros((n_steps,) + trailing, dtype=np.float64)
    cnt = np.zeros(n_steps, dtype=np.float64)
    np.add.at(acc, steps, window_values)
    np.add.at(cnt, steps, np.ones_like(steps, dtype=np.float64))
    cnt_safe = np.maximum(cnt, 1.0).reshape((n_steps,) + (1,) * len(trailing))
    return acc / cnt_safe, cnt


def stuck_run_length(values: Array, atol: float = 0.0) -> Array:
    """Consecutive unchanged samples per channel, computed on *raw* values.

    A healthy quantised channel changes by at least one LSB within a few
    samples; a channel that has not moved in hours has stopped being measured.
    Operating on raw values matters — after standardisation a legitimately
    constant channel is still constant, but the comparison must use the
    instrument's own resolution.
    """
    v = np.asarray(values, dtype=np.float64)
    n, d = v.shape
    out = np.zeros((n, d), dtype=np.float64)
    run = np.zeros(d, dtype=np.float64)
    for i in range(1, n):
        run = np.where(np.abs(v[i] - v[i - 1]) <= atol, run + 1.0, 0.0)
        out[i] = run
    return out


def rolling_std(x: Array, window: int, min_periods: int = 4) -> Array:
    """Causal rolling standard deviation, computed from cumulative moments.

    Used for sensor-health monitoring: a channel that stops changing is stuck,
    and a *frozen* signal has no deviation from expectation, so no residual
    statistic can see it. Variance collapse is the observable.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    c1 = np.cumsum(x, axis=0)
    c2 = np.cumsum(x * x, axis=0)
    out = np.zeros_like(x)
    for i in range(n):
        lo = max(0, i - window + 1)
        cnt = i - lo + 1
        if cnt < min_periods:
            continue
        s1 = c1[i] - (c1[lo - 1] if lo > 0 else 0.0)
        s2 = c2[i] - (c2[lo - 1] if lo > 0 else 0.0)
        var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0)
        out[i] = np.sqrt(var)
    return out


def rolling_mean(x: Array, window: int) -> Array:
    """Causal rolling mean over the last ``window`` samples (cumulative-sum form).

    Causal on purpose: at time ``t`` only samples up to ``t`` exist. A centred
    window would make the trend statistic non-causal and therefore impossible to
    reproduce online.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    c = np.cumsum(x, axis=0)
    out = np.empty_like(c)
    n_lo = min(window, n)
    # Leading edge: mean over everything available so far.
    for i in range(n_lo):
        out[i] = c[i] / (i + 1)
    if n > n_lo:
        out[n_lo:] = (c[n_lo:] - c[:-n_lo]) / window
    return out


def trend_statistic(signed_z: Array, window: int) -> Array:
    """Mean signed residual per channel over the trailing window.

    This is the statistic that sees slow faults. A smooth drift keeps the
    per-sample residual small, but its *sign* is consistent, so the window mean
    grows without bound while the per-sample error stays flat.
    """
    return rolling_mean(signed_z, window)


def ewma(x: Array, alpha: float) -> Array:
    """Exponentially weighted moving average (recursive, O(n))."""
    out = np.empty_like(x, dtype=np.float64)
    acc = x[0]
    for i in range(x.size):
        acc = alpha * x[i] + (1.0 - alpha) * acc
        out[i] = acc
    return out


def rolling_robust_z(x: Array, window: int) -> Array:
    """Score expressed as a rolling robust z-value, for operator readability.

    The raw fused error is not on a human-interpretable scale; mapping it to
    "median absolute deviations above the recent baseline" is what makes a
    threshold of e.g. 6.2σ meaningful to a maintenance engineer.
    """
    n = x.size
    z = np.empty(n, dtype=np.float64)
    half = max(1, window // 2)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half)
        seg = x[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        scale = 1.4826 * mad if mad > 1e-12 else (np.std(seg) or 1.0)
        z[i] = (x[i] - med) / max(scale, 1e-12)
    return z


# --------------------------------------------------------------------------
# Alarm state machine
# --------------------------------------------------------------------------
@dataclass
class AlarmStateMachine:
    """Consecutive-sample alarm counter with hysteresis.

    A raw per-sample threshold crossing produces an unusable alert stream on
    noisy SCADA data. Requiring ``min_consecutive`` crossings to raise and
    ``clear_after`` clean samples to clear converts a rate of isolated
    crossings into a small number of actionable events.
    """

    min_consecutive: int = 3
    clear_after: int = 12

    def run(self, above: Array) -> Tuple[Array, List[Tuple[int, int]]]:
        """Return the alarm trace and the list of ``(start, end)`` alarm events."""
        n = above.size
        state = np.zeros(n, dtype=bool)
        events: List[Tuple[int, int]] = []
        count = 0
        clean = 0
        active = False
        onset = -1
        for i in range(n):
            if above[i]:
                count += 1
                clean = 0
            else:
                clean += 1
                count = 0
            if not active and count >= self.min_consecutive:
                active = True
                onset = i - count + 1
            elif active and clean >= self.clear_after:
                active = False
                events.append((onset, i - clean + 1))
            state[i] = active
        if active:
            events.append((onset, n))
        return state, events


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
@dataclass
class DetectionResult:
    """Everything the scoring pass produced, ready for the API and evaluator."""

    turbine_id: str
    timestamps: Array
    score: Array
    smoke_score: Array
    threshold: float
    alarm: Array
    uncertainty: Array
    attribution: Array  # (n_steps, D) robust z^2 per channel
    attribution_z: Array  # (n_steps, D) signed robust z, for explanations
    events: List[Tuple[int, int]]
    records: List[AnomalyRecord]
    channel_names: Tuple[str, ...]
    components: Dict[str, Array] = field(default_factory=dict)
    count: Array = field(default_factory=lambda: np.zeros(0))

    @property
    def n_steps(self) -> int:
        return int(self.score.size)

    def summary(self) -> Dict[str, Any]:
        return {
            "turbine_id": self.turbine_id,
            "n_steps": self.n_steps,
            "n_alarms": int(self.alarm.sum()),
            "n_events": len(self.events),
            "threshold": self.threshold,
            "mean_score": float(np.mean(self.score)),
            "max_score": float(np.max(self.score)),
        }


class Detector:
    """Fitted detector: ensemble + scalers + threshold, and the scoring path."""

    def __init__(
        self,
        cfg: RunConfig,
        models: Sequence[SequenceAutoencoder],
        scaler: ChannelScaler,
        residual_normaliser: ResidualNormaliser,
        threshold: Threshold,
        channel_names: Tuple[str, ...],
        metadata: Optional[Dict[str, Any]] = None,
        score_mask: Optional[Array] = None,
        drift_scale: Optional[Array] = None,
        nbm: Optional[NormalBehaviourModel] = None,
        level_centre: Optional[Array] = None,
        level_scale: Optional[Array] = None,
        flatline_ref: Optional[Array] = None,
    ) -> None:
        self.cfg = cfg
        self.models = list(models)
        self.scaler = scaler
        self.residual_normaliser = residual_normaliser
        self.threshold = threshold
        self.channel_names = tuple(channel_names)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        if score_mask is None:
            from wt_pm_lstm.train import score_channel_mask

            score_mask = score_channel_mask(cfg.model, self.channel_names)
        self.score_mask = np.asarray(score_mask, dtype=bool)
        if self.score_mask.shape != (len(self.channel_names),):
            raise ValueError("score_mask must have one entry per channel")
        # Per-channel robust scale of the trend statistic, fitted during
        # calibration. Ones until :meth:`calibrate` runs.
        self.drift_scale = (
            np.ones(len(self.channel_names)) if drift_scale is None else np.asarray(drift_scale, dtype=np.float64)
        )
        # Level component: normal-behaviour model plus the robust centre/scale
        # of its residuals, both fitted on healthy data only.
        self.nbm = nbm
        # Reference rolling variability, filled in during calibration.
        self.flatline_ref = (
            np.ones(len(self.channel_names))
            if flatline_ref is None
            else np.asarray(flatline_ref, dtype=np.float64)
        )
        n_targets = 0 if nbm is None else len(nbm.target_indices)
        self.self_targets = (
            np.zeros(0, dtype=int) if nbm is None else np.asarray(nbm.target_indices, dtype=int)
        )
        self.level_centre = (
            np.zeros(n_targets) if level_centre is None else np.asarray(level_centre, dtype=np.float64)
        )
        self.level_scale = (
            np.ones(n_targets) if level_scale is None else np.asarray(level_scale, dtype=np.float64)
        )

    # -- scoring ----------------------------------------------------------
    def _windows(self, timeline: TurbineTimeline, stride: Optional[int] = None) -> Tuple[Array, Array, Windows]:
        values, mask = _timeline_arrays(timeline)
        z = self.scaler.transform(values)
        stride = self.cfg.model.stride if stride is None else stride
        win = make_windows(z, mask, self.cfg.model.window, stride=stride)
        return z, mask, win

    def score_timeline(
        self,
        timeline: TurbineTimeline,
        stride: Optional[int] = None,
        threshold_override: Optional[float] = None,
    ) -> DetectionResult:
        """Score a timeline end to end and produce an alarm trace + records.

        The four components are computed *per channel* and fused before any
        reduction over channels. Reducing first (averaging over eight channels)
        dilutes a fault that is strong in one channel — a gearbox oil
        temperature sitting 3.3 robust sigmas high contributes only 1.4 once
        averaged over the other seven — which is why the reduction is a
        configurable operator rather than a fixed mean.

        ``threshold_override`` re-runs the alarm policy at a different decision
        level while keeping the same score: used by
        :func:`~wt_pm_lstm.pipeline.operating_point_table` to expose the
        sensitivity/false-alarm trade-off without duplicating the decision logic
        (and therefore without the two drifting apart). The returned records are
        the ones the overridden threshold would emit; the detector's own
        calibrated threshold is unchanged.
        """
        dcfg = self.cfg.detect
        values, mask = _timeline_arrays(timeline)
        z = self.scaler.transform(values)
        stride = self.cfg.model.stride if stride is None else stride
        win = make_windows(z, mask, self.cfg.model.window, stride=stride)
        n_steps = z.shape[0]
        T = win.window
        x, m = win.values, win.mask
        if x.shape[0] == 0:
            raise ValueError("timeline is shorter than one window")

        N, M = x.shape[0], len(self.models)
        w_recon, w_pred, w_trend, w_level = _score_weights(dcfg)
        m = m & self.score_mask[None, None, :]

        recon_c = np.zeros((N, T, z.shape[1]))
        pred_c = np.zeros((N, T, z.shape[1]))
        signed_c = np.zeros((N, T, z.shape[1]))
        member_scores = np.zeros((M, N, T))

        for mi, model in enumerate(self.models):
            out = model.forward(np.ascontiguousarray(x))
            z_recon = self.residual_normaliser.z(x - out["x_hat"])
            sq_recon = z_recon**2
            sq_pred = np.zeros_like(sq_recon)
            if T > 1:
                sq_pred[:, 1:] = self.residual_normaliser.z(x[:, 1:] - out["y_hat"][:, :-1]) ** 2
            denom = np.maximum(m.sum(axis=2), 1.0)
            s_recon = (sq_recon * m).sum(axis=2) / denom
            s_pred = (sq_pred * m).sum(axis=2) / denom
            member_scores[mi] = w_recon * s_recon + w_pred * s_pred
            recon_c += sq_recon
            pred_c += sq_pred
            signed_c += z_recon
        recon_c /= M
        pred_c /= M
        signed_c /= M

        recon_ts, count = aggregate_windows(recon_c, win.starts, n_steps, T)
        pred_ts, _ = aggregate_windows(pred_c, win.starts, n_steps, T)
        signed_ts, _ = aggregate_windows(signed_c, win.starts, n_steps, T)
        unc_ts, _ = aggregate_windows(member_scores.std(axis=0) if M > 1 else np.zeros((N, T)), win.starts, n_steps, T)

        trend_ts = (trend_statistic(signed_ts, dcfg.trend_window) / self.drift_scale[None, :]) ** 2
        level_ts = np.zeros((n_steps, z.shape[1]))
        if self.nbm is not None:
            z_level = (self.nbm.residuals(values) - self.level_centre) / self.level_scale
            z_level = np.clip(z_level, -dcfg.level_clip, dcfg.level_clip) ** 2
            level_ts[:, self.self_targets] = z_level

        quiet_ratio = self.quietness_ratio(z)

        fused_channels = (
            w_recon * recon_ts
            + w_pred * pred_ts
            + w_trend * trend_ts
            + w_level * level_ts
        )
        fused_channels = np.where(self.score_mask[None, :], fused_channels, 0.0)
        # An imputed sample carries no evidence: a masked timestep cannot alarm.
        valid = mask.any(axis=1)
        score_raw = channel_reduce(fused_channels, dcfg.channel_reduction) * valid

        smoke = ewma(score_raw, dcfg.ewma_alpha)
        # Machine-health rule (fused score) OR sensor-health rule (a channel that
        # has gone quiet). Flatline is an independent rule because a frozen
        # channel carries no residual for the fused score to see, and because
        # the two failure modes call for different maintenance actions.
        sensor_health, stuck_run = self._sensor_health_alarm(values)
        value = float(self.threshold.value if threshold_override is None else threshold_override)
        above = (smoke > value) | sensor_health

        # Warm-up guard. The causal window statistics (trend, sensor health) are
        # only defined once their window has filled, and the EWMA initialises on
        # the first sample: the head of *any* batch therefore scores differently
        # from steady state. On a continuous feed this is invisible, but a
        # backfill, a re-scored window or a batch API all begin mid-record, and
        # without this guard every such call reports an alarm in its first hour
        # (measured: a healthy batch scored 15.4 against a threshold of 11.4 at
        # sample 0, decaying to 3.3 after ~200 samples). Alarms are suppressed
        # until the statistics are warm; the samples themselves are still
        # scored and reported.
        warmup = max(int(self.cfg.model.window), int(dcfg.trend_window))
        warm = np.arange(n_steps) < warmup
        if warm.any():
            above = above.copy()
            above[warm] = False
            sensor_health = sensor_health.copy()
            sensor_health[warm] = False
        machine = AlarmStateMachine(dcfg.min_consecutive, dcfg.clear_after)
        alarm, events = machine.run(above)

        records = self._build_records(
            timeline,
            smoke,
            alarm,
            unc_ts,
            fused_channels,
            signed_ts,
            stuck_run,
            threshold_value=value,
        )
        return DetectionResult(
            turbine_id=timeline.turbine_id,
            timestamps=timeline.timestamps,
            score=smoke,
            smoke_score=smoke,
            threshold=value,
            alarm=alarm,
            uncertainty=unc_ts,
            attribution=fused_channels,
            attribution_z=signed_ts,
            events=events,
            records=records,
            channel_names=self.channel_names,
            components={
                "raw_score": score_raw,
                "recon": recon_ts,
                "pred": pred_ts,
                "trend": trend_ts,
                "level": level_ts,
                "quietness_ratio": quiet_ratio,
                "stuck_run": stuck_run,
                "sensor_health_alarm": sensor_health,
                "warmup": warm,
                "signed_z": signed_ts,
            },
            count=count,
        )

    def quietness_ratio(self, z: Array) -> Array:
        """How many times *quieter* each channel is than its healthy variability.

        A diagnostic aid rather than a score term: it is deliberately not fused
        into the anomaly score, because legitimately quiet channels exist
        (rotor speed at rated, blade pitch held at an operating point) and any
        threshold on it raises false alarms on those. The alarm rule for a
        genuinely stuck sensor is :func:`stuck_run_length`.
        """
        window = max(4, self.cfg.detect.trend_window)
        rs = rolling_std(z, window)
        ref = np.maximum(self.flatline_ref, 1e-9)
        return np.clip(ref[None, :] / np.maximum(rs, 1e-9) - 1.0, 0.0, self.cfg.detect.level_clip)

    def _sensor_health_alarm(self, values: Array) -> Tuple[Array, Array]:
        """Per-sample "the measurement has stopped changing" rule.

        Returns ``(alarm, run_length)``. ``run_length`` is per channel and in
        samples, so an alarm can name the sensor that stopped reporting — the
        attribution a machine-health alarm cannot give, because its cause is an
        absence of signal rather than a deviation in one.
        """
        dcfg = self.cfg.detect
        run_length = stuck_run_length(values)
        scored = run_length[:, self.score_mask]
        longest = scored.max(axis=1) if scored.size else np.zeros(values.shape[0])
        # Require persistence, exactly as the score rule does; one held sample
        # is a stale cache, eighteen is a sensor that has stopped.
        stack = AlarmStateMachine(1, dcfg.clear_after)
        alarm, _ = stack.run(longest >= dcfg.stuck_samples)
        return alarm, run_length

    # -- calibration ------------------------------------------------------
    def calibrate(
        self,
        calibration_timeline: TurbineTimeline,
        quantile: float,
        rng: np.random.Generator,
        bootstrap_samples: int = 200,
        stride: int = 2,
    ) -> Threshold:
        """Fit the trend scale and the alarm threshold on the calibration band.

        Two passes are needed: the first measures the trend statistic's own
        spread on healthy data (so the trend component is on the same scale as
        the residual components), the second sets the threshold. Both passes use
        only the calibration band, which the model never saw during training.
        """
        # Reference rolling variability per channel, from healthy data.
        cal_values, cal_mask = _timeline_arrays(calibration_timeline)
        cal_z = self.scaler.transform(cal_values)
        rs = rolling_std(cal_z, max(4, self.cfg.detect.trend_window))
        self.flatline_ref = np.maximum(np.median(rs, axis=0), 1e-9)

        if self.nbm is not None:
            values, mask = _timeline_arrays(calibration_timeline)
            res = self.nbm.residuals(values)
            valid = mask.any(axis=1)
            centre = np.median(res[valid], axis=0)
            mad = 1.4826 * np.median(np.abs(res[valid] - centre), axis=0)
            std = np.std(res[valid], axis=0)
            self.level_centre = centre
            self.level_scale = np.maximum(np.where(mad < 1e-9, std, mad), 1e-9)

        first = self.score_timeline(calibration_timeline, stride=stride)
        covered = first.count > 0
        stat = first.components["signed_z"][covered]
        scale = 1.4826 * np.median(np.abs(stat - np.median(stat, axis=0)), axis=0)
        fallback = np.std(stat, axis=0)
        scale = np.where(scale < 1e-9, np.maximum(fallback, 1e-9), scale)
        self.drift_scale = np.maximum(scale, 1e-9)

        second = self.score_timeline(calibration_timeline, stride=stride)
        covered = second.count > 0
        scores = second.score[covered]
        # Provisional threshold: the score quantile. Used directly when the
        # event-rate method is disabled, and as the fallback when the target
        # false-alarm budget cannot be met on this band.
        quantile_threshold = calibrate_threshold(
            scores,
            quantile=quantile,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )
        dcfg = self.cfg.detect
        if dcfg.threshold_method != "time-budget":
            self.threshold = quantile_threshold
            return self.threshold

        # The budget is a fraction of *time*: minutes per day out of 1440.
        # Translating it to a score quantile is exact for the calibration band,
        # and monotone in the budget, which is what makes it a usable knob.
        budget_quantile = float(
            np.clip(1.0 - dcfg.target_false_alarm_minutes_per_day / 1440.0, 1e-6, 1.0 - 1e-9)
        )
        self.threshold = calibrate_threshold(
            scores,
            quantile=budget_quantile,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )
        self.threshold.method = (
            f"time-budget({dcfg.target_false_alarm_minutes_per_day:g} min/day -> "
            f"q={budget_quantile:.6f} of the calibration score)"
        )
        return self.threshold

    # -- records ----------------------------------------------------------
    def _build_records(
        self,
        timeline: TurbineTimeline,
        score: Array,
        alarm: Array,
        uncertainty: Array,
        attr: Array,
        attr_z: Array,
        stuck_run: Optional[Array] = None,
        threshold_value: Optional[float] = None,
    ) -> List[AnomalyRecord]:
        threshold_value = float(self.threshold.value if threshold_value is None else threshold_value)
        model_id = self.metadata.get("model_id", "wtpm-lstm-scada-v1@unknown")
        records: List[AnomalyRecord] = []
        last_alarm_state = False
        for i in range(score.size):
            if not alarm[i]:
                last_alarm_state = False
                continue
            # Emit one record per alarm event (at its onset) rather than one per
            # sample: the SCADA API contract is meant for maintenance systems,
            # which consume events, not 10-minute rows.
            if last_alarm_state:
                continue
            last_alarm_state = True
            masked_attr = np.where(self.score_mask, attr[i], -np.inf)
            top = np.argsort(masked_attr)[::-1][:3]
            attribution = {
                self.channel_names[j]: float(attr[i, j]) for j in top if attr[i, j] > 0
            }
            ratio = float(score[i] / threshold_value) if threshold_value > 0 else float("inf")
            if stuck_run is not None:
                held = np.where(self.score_mask, stuck_run[i], 0.0)
                j_held = int(np.argmax(held))
                if held[j_held] >= self.cfg.detect.stuck_samples:
                    records.append(
                        AnomalyRecord(
                            turbine_id=timeline.turbine_id,
                            timestamp=int(timeline.timestamps[i]),
                            score=float(score[i]),
                            threshold=threshold_value,
                            alarm=True,
                            model_id=model_id,
                            uncertainty_std=float(uncertainty[i]),
                            attribution={self.channel_names[j_held]: float(held[j_held])},
                            recommended_action=(
                                "inspect sensor and wiring; verify against a redundant measurement"
                            ),
                            explanation=(
                                f"sensor-health alarm: {self.channel_names[j_held]} has not changed "
                                f"for {int(held[j_held])} consecutive samples (stuck / held signal)"
                            ),
                        )
                    )
                    last_alarm_state = True
                    continue
            records.append(
                AnomalyRecord(
                    turbine_id=timeline.turbine_id,
                    timestamp=int(timeline.timestamps[i]),
                    score=float(score[i]),
                    threshold=threshold_value,
                    alarm=True,
                    model_id=model_id,
                    uncertainty_std=float(uncertainty[i]),
                    attribution=attribution,
                    recommended_action=_recommend(ratio),
                    explanation=self._explain(i, score[i], attr_z[i]),
                )
            )
        return records

    def _explain(self, i: int, score: float, z_row: Array) -> str:
        """Human-readable cause statement — the XAI layer's input (model 24)."""
        masked_z = np.where(self.score_mask, z_row, 0.0)
        order = np.argsort(np.abs(masked_z))[::-1]
        parts = []
        for j in order[:3]:
            if not np.isfinite(z_row[j]):
                continue
            spec = get_channel(self.channel_names[j])
            parts.append(f"{self.channel_names[j]} {z_row[j]:+.1f}σ ({spec.role})")
        ratio = score / self.threshold.value if self.threshold.value > 0 else float("inf")
        return (
            f"fused score {score:.2f} = {ratio:.2f}x threshold; "
            f"largest robust residual deviations: " + ", ".join(parts)
        )

    # -- persistence ------------------------------------------------------
    def to_metadata(self) -> Dict[str, Any]:
        return {
            "channel_names": list(self.channel_names),
            "scaler": self.scaler.to_dict(),
            "residual_normaliser": self.residual_normaliser.to_dict(),
            "threshold": self.threshold.to_dict(),
            "model": {
                "architecture": "lstm-sequence-autoencoder" if self.cfg.model.cell == "lstm" else "gru-sequence-autoencoder",
                "cell": self.cfg.model.cell,
                "hidden_size": self.cfg.model.hidden_size,
                "latent_size": self.models[0].latent_size,
                "window": self.cfg.model.window,
                "n_members": len(self.models),
                "n_parameters_per_member": self.models[0].n_params,
                "members": [m.summary() for m in self.models],
            },
            "score_mask": self.score_mask.tolist(),
            "drift_scale": self.drift_scale.tolist(),
            "flatline_ref": self.flatline_ref.tolist(),
            "level_centre": self.level_centre.tolist(),
            "level_scale": self.level_scale.tolist(),
            "nbm": None if self.nbm is None else self.nbm.to_dict(),
            "exogenous_index_pool": list(getattr(self.models[0], "exogenous_index", ())),
            "score_channels": [n for n, keep in zip(self.channel_names, self.score_mask) if keep],
            # The decision and fusion settings, recorded as they are *used*. A
            # sidecar that describes a configuration the code no longer has is
            # worse than no sidecar: it makes an unreproducible bundle look
            # reproducible.
            "detect_config": {
                "score_weights": dict(self.cfg.detect.score_weights),
                "channel_reduction": self.cfg.detect.channel_reduction,
                "channel_reduction_k": self.cfg.detect.channel_reduction_k,
                "trend_window": self.cfg.detect.trend_window,
                "level_clip": self.cfg.detect.level_clip,
                "stuck_samples": self.cfg.detect.stuck_samples,
                "ewma_alpha": self.cfg.detect.ewma_alpha,
                "threshold_method": self.cfg.detect.threshold_method,
                "threshold_quantile": self.cfg.detect.threshold_quantile,
                "target_false_alarm_minutes_per_day": self.cfg.detect.target_false_alarm_minutes_per_day,
                "min_consecutive": self.cfg.detect.min_consecutive,
                "clear_after": self.cfg.detect.clear_after,
            },
            "metadata": self.metadata,
        }


def channel_reduce(x: Array, how: str = "max", top_k: int = 3) -> Array:
    """Reduce per-channel statistics to one score per timestep.

    ``max`` is the default for multi-channel alarm systems: each channel keeps
    its own sensitivity and the score reports the worst one. Averaging over
    channels instead lets a fault that is strong in a single channel be diluted
    by the seven healthy ones, which is the failure mode that made a sustained
    3.3-sigma bearing-temperature offset look like noise.
    """
    if x.ndim != 2:
        raise ValueError("channel_reduce expects a (n_steps, n_channels) array")
    if how == "mean":
        return x.mean(axis=1)
    if how == "max":
        return x.max(axis=1)
    if how == "topk":
        k = min(top_k, x.shape[1])
        return np.sort(x, axis=1)[:, -k:].mean(axis=1)
    raise ValueError(f"unknown channel reduction {how!r}")


def _score_weights(dcfg: DetectConfig) -> Tuple[float, float, float, float]:
    """Normalised fusion weights for the four score components."""
    raw = dict(dcfg.score_weights or {})
    total = sum(max(0.0, float(v)) for v in raw.values()) or 1.0
    return (
        max(0.0, float(raw.get("reconstruction", 0.0))) / total,
        max(0.0, float(raw.get("temporal", 0.0))) / total,
        max(0.0, float(raw.get("trend", 0.0))) / total,
        max(0.0, float(raw.get("level", 0.0))) / total,
    )


def _timeline_arrays(timeline: TurbineTimeline) -> Tuple[Array, Array]:
    from wt_pm_lstm.windows import prepare_timeline

    return prepare_timeline(timeline)


def member_residuals(
    models: Sequence[SequenceAutoencoder], x: Array
) -> Tuple[Array, Array]:
    """Reconstruction and forecast residuals for every ensemble member.

    Both are returned with shape ``(M, N, T, D)``; the forecast residual of the
    first timestep is zero because no prediction targets it. Used to fit the
    :class:`ResidualNormaliser` on the calibration band and by
    :meth:`Detector.score_timeline`.
    """
    recons, preds = [], []
    for model in models:
        out = model.forward(np.ascontiguousarray(x))
        recons.append(x - out["x_hat"])
        pred = np.zeros_like(x)
        if x.shape[1] > 1:
            pred[:, 1:] = x[:, 1:] - out["y_hat"][:, :-1]
        preds.append(pred)
    return np.stack(recons), np.stack(preds)


def _recommend(ratio: float) -> str:
    """Map alarm severity onto a maintenance action (fabric Layer 13)."""
    if ratio < 1.5:
        return "continue with monitoring"
    if ratio < 3.0:
        return "inspect at next scheduled service"
    if ratio < 8.0:
        return "schedule inspection within 7 days"
    return "dispatch inspection; consider derating"


__all__ = [
    "rolling_std",
    "channel_reduce",
    "ResidualNormaliser",
    "Threshold",
    "calibrate_threshold",
    "AlarmStateMachine",
    "Detector",
    "DetectionResult",
    "aggregate_windows",
    "ewma",
    "rolling_robust_z",
]
