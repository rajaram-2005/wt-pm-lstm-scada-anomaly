"""Detector mechanics: causality, fusion, the alarm policy, sensor health.

These test the pieces that a plausible-looking refactor can silently break —
a centred instead of causal window, a mean instead of a max over channels, a
sensor-health rule that fires on legitimately quiet channels.
"""

from __future__ import annotations

import numpy as np
import pytest

from wt_pm_lstm.config import RunConfig
from wt_pm_lstm.detect import (
    AlarmStateMachine,
    alarm_event_count,
    channel_reduce,
    rolling_mean,
    rolling_std,
    stuck_run_length,
    trend_statistic,
)


def test_rolling_statistics_are_causal():
    """A centred window would be unreproducible online and would leak the future."""
    x = np.zeros((50, 2))
    x[25:, 0] = 10.0
    rm = rolling_mean(x, window=5)
    assert np.all(rm[:25, 0] == 0.0), "the mean must not see the future step"
    assert rm[25, 0] == pytest.approx(2.0)  # (0,0,0,0,10)/5

    rs = rolling_std(x, window=10)
    assert np.all(rs[:25, 1] == 0.0), "a constant channel has zero rolling spread"
    # Variance appears only as the step enters the trailing window.
    assert rs[26, 0] > rs[25, 0] > 0.0


def test_rolling_mean_matches_bruteforce():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    window = 7
    got = rolling_mean(x, window)
    for i in range(window - 1, x.shape[0]):
        np.testing.assert_allclose(got[i], x[i - window + 1 : i + 1].mean(axis=0), rtol=1e-9)


def test_trend_statistic_ignores_noise_but_sees_a_ramp():
    """The statistic exists to catch slow shifts that residual heads cannot."""
    rng = np.random.default_rng(1)
    window = 36
    stationary = rng.normal(0.0, 1.0, size=(400, 1))
    ramp = np.zeros((400, 1))
    ramp[200:] = np.linspace(0.0, 0.8, 200)[:, None]  # 0.8 sigma over 33 h

    # Noise averages down by sqrt(window); a consistent drift accumulates.
    t_stationary = np.abs(trend_statistic(stationary, window)).mean()
    t_ramp = np.abs(trend_statistic(ramp, window))[240:].mean()
    assert t_ramp > 3.0 * t_stationary
    # By the end of a 0.8-sigma ramp the trailing mean is a large fraction of it.
    assert trend_statistic(ramp, window)[-1, 0] > 0.5
    # The statistic is exactly zero until the window fills: no warm-up artefacts.
    assert np.allclose(trend_statistic(ramp, window)[: window - 1], 0.0)


def test_channel_reduce_operators():
    x = np.array([[1.0, 5.0, 9.0], [2.0, 2.0, 2.0]])
    np.testing.assert_allclose(channel_reduce(x, "max"), [9.0, 2.0])
    np.testing.assert_allclose(channel_reduce(x, "mean"), [5.0, 2.0])
    np.testing.assert_allclose(channel_reduce(x, "topk", top_k=2), [7.0, 2.0])
    # A single strong channel must survive reduction: this is why the default is
    # max and not mean (a 3.3-sigma single-channel fault becomes 1.4 averaged).
    spike = np.array([[3.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])
    assert channel_reduce(spike, "max")[0] == pytest.approx(3.3)
    assert channel_reduce(spike, "mean")[0] < 1.0
    with pytest.raises(ValueError):
        channel_reduce(x, "median")


def test_stuck_run_length_detects_a_held_signal():
    x = np.zeros((30, 2))
    x[:, 0] = np.arange(30.0)  # healthy, always moving
    x[:10, 1] = np.linspace(1.0, 2.0, 10)  # healthy, moving
    x[10:, 1] = 42.0  # stuck from step 10 onwards
    runs = stuck_run_length(x)
    assert runs[:, 0].max() == 0.0
    assert runs[9, 1] == 0.0
    assert runs[10, 1] == 0.0, "the first held sample is where the value stopped moving"
    assert runs[11, 1] == 1.0
    assert runs[29, 1] == 19.0
    # A one-step quantisation tolerance is available for signals that dither.
    noisy = np.array([[1.0], [1.0 + 1e-12], [1.0 + 2e-12]])
    assert stuck_run_length(noisy, atol=1e-9)[-1, 0] == 2.0
    assert stuck_run_length(noisy, atol=0.0)[-1, 0] == 0.0


def test_alarm_state_machine_hysteresis():
    machine = AlarmStateMachine(min_consecutive=3, clear_after=4)
    above = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=bool)
    alarm, events = machine.run(above)
    # Three consecutive crossings raise the alarm.
    assert not alarm[0] and not alarm[1] and alarm[2]
    # Two clean samples do not clear it: the alarm stays latched through a short
    # drop-out, which is what stops one noisy window producing two events.
    assert alarm[3] and alarm[4] and alarm[5] and alarm[6]
    # It clears exactly after clear_after consecutive clean samples.
    assert alarm[9] and not alarm[10]
    # Crossing the threshold twice at the end is not an alarm.
    assert not alarm[11] and not alarm[12]
    assert events == [(0, 7)]


def test_alarm_event_count_is_not_monotone_in_the_threshold():
    """Documenting the trap that invalidated an earlier calibration attempt.

    A very low threshold puts the score permanently above it: that is *one*
    long alarm event, not many. Event counts are therefore not monotone in the
    threshold and cannot be bisected. The threshold is calibrated on alarm
    *time*, which is monotone.
    """
    from wt_pm_lstm.detect import ewma

    rng = np.random.default_rng(4)
    scores = ewma(np.abs(rng.normal(size=2000)), 0.15)
    low = alarm_event_count(scores, 0.01, 3, 12)
    mid = alarm_event_count(scores, 0.6, 3, 12)
    high = alarm_event_count(scores, 10.0, 3, 12)
    assert high == 0
    assert mid > low, "a mid threshold fragments one long alarm into several events"


def test_threshold_budget_is_a_time_fraction():
    """The time budget maps onto a monotone quantile: 10 min/day => q ~ 0.9931."""
    cfg = RunConfig()
    assert cfg.detect.threshold_method == "time-budget"
    budget = cfg.detect.target_false_alarm_minutes_per_day
    quantile = 1.0 - budget / 1440.0
    assert 1.0 - quantile == pytest.approx(budget / 1440.0)
    assert 0.99 < quantile < 0.999
