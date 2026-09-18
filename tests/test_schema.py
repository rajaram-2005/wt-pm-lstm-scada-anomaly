"""The data contract, and the invariants that keep it honest.

Every finding recorded here was a real failure at some point in this project's
development: the fault-kind lookup that silently read the wrong position of the
parent record, the missing values that were almost encoded as -999, the
exogenous channels that were once scored. They are tests now.
"""

from __future__ import annotations

import numpy as np
import pytest

from wt_pm_lstm.schema import (
    CONTEXT_CHANNELS,
    EXOGENOUS_CHANNELS,
    QUALITY_MISSING,
    QUALITY_OK,
    QUALITY_OUT_OF_RANGE,
    SCHEMA_VERSION,
    TurbineTimeline,
    default_score_channels,
    describe_schema,
    get_channel,
)


def _timeline(n=20, channels=("power_kw", "wind_speed_ms")):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(n, len(channels)))
    return TurbineTimeline(
        turbine_id="WT-T1",
        site_id="SITE-T",
        channel_names=channels,
        values=values,
        timestamps=1_700_000_000 + 600 * np.arange(n, dtype=np.int64),
    )


def test_schema_declares_every_channel_once():
    payload = describe_schema()
    assert payload["schema_version"] == SCHEMA_VERSION
    names = [c["name"] for c in payload["channels"]]
    assert len(names) == len(set(names)) == 12
    assert set(EXOGENOUS_CHANNELS).issubset(names)
    assert set(CONTEXT_CHANNELS).issubset(names)
    # Every channel declares a unit and a physical envelope; a contract without
    # ranges cannot reject an out-of-range value.
    for channel in payload["channels"]:
        assert channel["unit"]
        assert channel["plausible_min"] < channel["plausible_max"]


def test_scored_channels_exclude_exogenous_and_pitch():
    scored = default_score_channels()
    assert set(scored).isdisjoint(EXOGENOUS_CHANNELS)
    assert "pitch_angle_deg" not in scored
    assert len(scored) == 8


def test_missing_data_is_flagged_not_faked():
    """The contract has no sentinel value: absence is a flag plus a mask."""
    tl = _timeline()
    tl.mask[3, 0] = False
    tl.quality[3, 0] = QUALITY_MISSING
    payload = tl.to_dict()
    assert payload["values"][3][0] is None
    assert payload["quality"][3][0] == QUALITY_MISSING
    restored = TurbineTimeline.from_dict(payload)
    assert not restored.mask[3, 0]
    assert restored.quality[3, 0] == QUALITY_MISSING


def test_out_of_range_values_are_kept_and_flagged():
    tl = _timeline()
    spec = get_channel("power_kw")
    tl.values[5, 0] = spec.plausible_max * 10.0
    tl.quality[5, 0] = QUALITY_OUT_OF_RANGE
    payload = tl.to_dict()
    assert payload["values"][5][0] == pytest.approx(spec.plausible_max * 10.0)
    assert payload["quality"][5][0] == QUALITY_OUT_OF_RANGE


def test_validate_reports_problems_instead_of_raising():
    """Validation returns diagnostics: a bad record is reported, never raised.

    A SCADA export with a clock glitch must still be loadable — the caller
    decides what to do with a non-monotone timestamp, and a hard failure deep in
    a batch job is worse than a diagnostic at the boundary.
    """
    tl = _timeline()
    tl.timestamps[5] = tl.timestamps[4]  # repeated timestamp (clock glitch)
    problems = tl.validate()
    assert problems and any("increasing" in p for p in problems)

    tl2 = _timeline()
    tl2.values[3, 0] = np.nan  # NaN where the mask claims the sample is valid
    problems2 = tl2.validate()
    assert problems2 and any("NaN" in p for p in problems2)

    tl3 = _timeline()
    spec = get_channel("power_kw")
    tl3.values[0, 0] = spec.plausible_max * 5.0
    assert not any("range" in p for p in tl3.validate()), "bounds are opt-in"
    assert any("range" in p for p in tl3.validate(strict_bounds=True))


def test_slice_carries_per_step_metadata():
    """Slicing must slice per-step metadata, or step-relative lookups lie.

    This was a live bug: the fault kind of an event onset was read from the
    parent record's array using the slice's index, so every event was attributed
    to the wrong family and the per-family recall table collapsed to one key.
    """
    tl = _timeline(n=40)
    kinds = [""] * 40
    for i in range(10, 20):
        kinds[i] = "bearing_wear"
    tl.meta["fault_kind_per_step"] = kinds
    tl.meta["fault_schedule"] = [{"kind": "bearing_wear"}]  # not per-step

    sub = tl.slice(10, 20)
    assert sub.n_steps == 10
    assert sub.meta["fault_kind_per_step"] == ["bearing_wear"] * 10
    # Non-per-step metadata is carried through untouched.
    assert sub.meta["fault_schedule"] == [{"kind": "bearing_wear"}]


def test_timeline_requires_aligned_shapes():
    with pytest.raises(ValueError):
        TurbineTimeline(
            turbine_id="WT-T1",
            channel_names=("power_kw",),
            values=np.zeros((10, 2)),
            timestamps=np.arange(10, dtype=np.int64),
        )


def test_sampling_seconds_derived_from_timestamps():
    tl = _timeline(n=10)
    assert tl.sampling_seconds == 600
    assert tl.n_steps == 10 and tl.n_channels == 2
