"""Shared fit boundaries and numerical contracts for all research adapters."""
from __future__ import annotations

import numpy as np


def training_mask(batch, mask):
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (batch.n_steps,):
        raise ValueError('train_mask must be a boolean vector with one entry per sample')
    if not mask.any():
        raise ValueError('train_mask selects no samples')
    return mask


def window_training_mask(batch, mask):
    """A training window may not include even one excluded row."""
    mask = training_mask(batch, mask)
    width = batch.windows.shape[1]
    ends = np.asarray(batch.window_index)
    bad = np.r_[0, np.cumsum(~mask)]
    keep = (bad[ends + 1] - bad[ends - width + 1]) == 0
    if not keep.any():
        raise ValueError('no complete windows inside train_mask')
    return keep


def sample_hours(batch):
    ts = np.asarray(batch.timestamps, float)
    if ts.shape != (batch.n_steps,) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('timestamps must be finite and strictly increasing')
    return np.r_[0.0, np.diff(ts) / 3600.0]


def hold_probabilities(probabilities, indices, n):
    """Past-only hold; before the first observation return a uniform prior."""
    p = np.asarray(probabilities)
    out = np.full((n, p.shape[1]), 1 / p.shape[1])
    pos = np.searchsorted(indices, np.arange(n), side='right') - 1
    valid = pos >= 0
    out[valid] = p[pos[valid]]
    return out


def validate_output(out):
    if not out.ok:
        return
    n = len(out.timestamps)
    if not n or not np.isfinite(out.timestamps).all() or np.any(np.diff(out.timestamps) <= 0):
        raise ValueError('output requires finite timestamps')
    for name in ('prediction', 'anomaly_score', 'rul_hours', 'degradation_state', 'uncertainty'):
        value = getattr(out, name)
        if value is None:
            continue
        value = np.asarray(value)
        if value.shape != (n,):
            raise ValueError(f'{name}: expected ({n},), got {value.shape}')
        if value.dtype.kind in 'iuf' and not np.isfinite(value).all():
            raise ValueError(f'{name}: non-finite output')
        if name in ('rul_hours', 'uncertainty') and np.any(value < 0):
            raise ValueError(f'{name}: negative output')
    if out.probability:
        p = np.asarray(list(out.probability.values()), float)
        if p.shape != (len(out.probability), n) or not np.isfinite(p).all():
            raise ValueError('invalid probability shape or non-finite values')
        if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(0), 1, atol=1e-5):
            raise ValueError('probabilities must lie in [0,1] and sum to one')


# Some original NumPy research modules use global RNGs. Isolate those calls so
# repeated inference does not depend on request order or alter caller RNG state.
from functools import wraps
import threading
_numpy_rng_lock = threading.RLock()


def isolated_numpy_rng(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        with _numpy_rng_lock:
            state = np.random.get_state()
            try:
                np.random.seed(42)
                return fn(*args, **kwargs)
            finally:
                np.random.set_state(state)
    return call
