"""Windowing, robust scaling and leakage-safe splitting (fabric Layer 2).

Three rules are enforced here, because violating any of them silently inflates
every downstream metric:

1. **Fit statistics on the training split only.** The scaler and the error
   normaliser are ``fit`` calls on the healthy training band; the calibration
   band is used once, for the threshold, and never for fitting.
2. **Faults never leak into fitting.** :func:`split_indices` refuses to hand out
   a split that contains labelled fault samples.
3. **Masked samples never contribute.** Imputed or flagged values are carried
   through the pipeline with weight 0 rather than being silently interpolated
   into the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from wt_pm_lstm.schema import INVALID_QUALITY, TurbineTimeline

Array = np.ndarray


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitPlan:
    """Half-open index ranges for the three protocol splits."""

    train: Tuple[int, int]
    calibration: Tuple[int, int]
    test: Tuple[int, int]

    @property
    def n_steps(self) -> int:
        return self.test[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train": list(self.train),
            "calibration": list(self.calibration),
            "test": list(self.test),
        }


def split_indices(
    n_steps: int,
    healthy_fraction: float,
    calibration_fraction: float,
    fault_label: Optional[Array] = None,
    window: int = 0,
) -> SplitPlan:
    """Time-ordered train / calibration / test split — never random.

    Random splits are invalid for time series: neighbouring SCADA samples are
    strongly autocorrelated, so a shuffled split lets the model interpolate
    between train and test points and reports metrics that cannot be reproduced
    in operation. The splits are therefore contiguous time blocks.

    ``window`` shrinks the training band by ``window - 1`` samples so that no
    training window can overlap the calibration band.
    """
    if healthy_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("healthy_fraction and calibration_fraction must be positive")
    if healthy_fraction + calibration_fraction >= 1.0:
        raise ValueError("healthy + calibration fractions must leave a test band")
    n_train = int(n_steps * healthy_fraction)
    n_cal = int(n_steps * calibration_fraction)
    if n_train <= window or n_cal <= window:
        raise ValueError(
            f"record too short: {n_steps} steps give train={n_train} and cal={n_cal}, "
            f"both must exceed the window length {window}"
        )
    plan = SplitPlan(
        train=(0, n_train),
        calibration=(n_train, n_train + n_cal),
        test=(n_train + n_cal, n_steps),
    )
    if fault_label is not None:
        for name, (a, b) in (("train", plan.train), ("calibration", plan.calibration)):
            if np.asarray(fault_label[a:b]).any():
                raise ValueError(
                    f"{name} split [{a}, {b}) contains labelled fault samples; the "
                    "detector must never be fitted or calibrated on faults"
                )
    return plan


# --------------------------------------------------------------------------
# Robust scaling
# --------------------------------------------------------------------------
@dataclass
class ChannelScaler:
    """Per-channel robust standardisation (median / 1.4826 MAD).

    Median and MAD are used instead of mean and standard deviation because
    SCADA records contain glitches and shutdown transients: a single 3 MW
    spike would shift a mean/std scaler enough to hide genuine slow faults.
    """

    channel_names: Tuple[str, ...]
    median: Array
    scale: Array
    clip: float = 8.0

    @classmethod
    def fit(
        cls,
        values: Array,
        mask: Array,
        channel_names: Tuple[str, ...],
        clip: float = 8.0,
    ) -> "ChannelScaler":
        """Fit median and scale per channel on *fault-free* data only.

        The scale is the larger of the MAD-based and IQR-based sigma estimates.
        MAD alone is wrong for zero-inflated control channels: blade pitch sits
        at 0 below rated wind speed and rises to 90 degrees above it, so more
        than half the samples equal the median, the MAD collapses towards zero,
        and every pitched sample normalises to a huge value that then dominates
        both the training loss and the anomaly score. The IQR spans the
        operating envelope and gives a usable scale in exactly that case, while
        agreeing with MAD for unimodal channels.
        """
        if values.shape != mask.shape:
            raise ValueError("values and mask must have the same shape")
        med = np.zeros(values.shape[1])
        scale = np.ones(values.shape[1])
        for j in range(values.shape[1]):
            col = values[mask[:, j], j]
            if col.size < 4:
                raise ValueError(f"channel {channel_names[j]!r} has fewer than 4 valid samples")
            med[j] = float(np.median(col))
            mad_sigma = 1.4826 * float(np.median(np.abs(col - med[j])))
            q75, q25 = np.percentile(col, [75, 25])
            iqr_sigma = float(q75 - q25) / 1.349
            sigma = max(mad_sigma, iqr_sigma)
            if sigma < 1e-9:  # constant channel
                sigma = float(np.std(col)) or 1.0
            scale[j] = sigma
        return cls(channel_names=tuple(channel_names), median=med, scale=scale, clip=clip)

    def transform(self, values: Array) -> Array:
        z = (values - self.median) / self.scale
        return np.clip(z, -self.clip, self.clip)

    def inverse(self, z: Array) -> Array:
        return z * self.scale + self.median

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_names": list(self.channel_names),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChannelScaler":
        return cls(
            channel_names=tuple(payload["channel_names"]),
            median=np.asarray(payload["median"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            clip=float(payload.get("clip", 8.0)),
        )


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Windows:
    """A batch of windows plus the bookkeeping needed to map scores back in time."""

    values: Array  # (N, T, D), standardised
    mask: Array  # (N, T, D) bool
    starts: Array  # (N,) first timestep index of each window

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def window(self) -> int:
        return int(self.values.shape[1])


def make_windows(
    values: Array,
    mask: Array,
    window: int,
    stride: int = 1,
    start: int = 0,
    stop: Optional[int] = None,
) -> Windows:
    """Cut ``[start, stop)`` into overlapping windows.

    Windows are returned as views (``sliding_window_view``), so a 60-day record
    costs no extra memory until batches are gathered. All windows are complete:
    partial windows at the end are dropped rather than padded, because a
    zero-padded window is out-of-distribution and would be scored as an anomaly.
    """
    stop = values.shape[0] if stop is None else stop
    if window < 1:
        raise ValueError("window must be >= 1")
    if stop - start < window:
        raise ValueError(f"range [{start}, {stop}) is shorter than the window ({window})")
    view = np.lib.stride_tricks.sliding_window_view(values[start:stop], window, axis=0)
    mview = np.lib.stride_tricks.sliding_window_view(mask[start:stop], window, axis=0)
    # sliding_window_view(y, w, axis=0) has shape (N - w + 1, D, w); move the
    # time axis to position 1 for the (N, T, D) convention used everywhere else.
    view = np.moveaxis(view, -1, 1)
    mview = np.moveaxis(mview, -1, 1)
    idx = np.arange(0, view.shape[0], stride)
    starts = (start + idx).astype(np.int64)
    return Windows(values=view[idx], mask=mview[idx].astype(bool), starts=starts)


def valid_mask_from_quality(quality: Array) -> Array:
    """Samples that may be used for fitting: quality flag says they are real."""
    mask = np.ones(quality.shape, dtype=bool)
    for code in INVALID_QUALITY:
        mask &= quality != code
    return mask


def prepare_timeline(
    timeline: TurbineTimeline, mask_policy: str = "quality"
) -> Tuple[Array, Array]:
    """Return ``(values, mask)`` ready for scaling.

    ``mask_policy``
        ``quality`` (default) trusts the contract's quality flags; ``finite``
        additionally rejects non-finite values; ``all`` is for ablation runs
        that deliberately ignore data quality.
    """
    values = timeline.values.copy()
    if mask_policy == "all":
        return values, np.ones(values.shape, dtype=bool)
    mask = np.isfinite(values)
    if mask_policy == "quality":
        mask &= valid_mask_from_quality(timeline.quality)
    # Values at masked positions are irrelevant but must not be NaN, since they
    # still flow through the recurrent layers (with zero loss weight).
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values, mask


__all__ = [
    "SplitPlan",
    "split_indices",
    "ChannelScaler",
    "Windows",
    "make_windows",
    "prepare_timeline",
    "valid_mask_from_quality",
]
