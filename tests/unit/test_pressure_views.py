"""Time at Pressure bucketing, time weighting and event placement."""

from __future__ import annotations

import numpy as np
import pytest

from snore.services.pressure_views import _bucket_ids, _time_at_pressure

pytestmark = pytest.mark.unit


def _series(values: list[float]) -> tuple[np.ndarray, np.ndarray, float]:
    """A 0.5 Hz series whose ``values`` start at 10 s, after the settling window.

    The window's five samples repeat the first value.
    """
    padded = [values[0]] * 5 + values
    return np.arange(len(padded)) * 2.0, np.array(padded), 0.5


@pytest.mark.parametrize(
    ("pressure", "bucket_centre"),
    [(9.9, 10.0), (10.0, 10.0), (10.09, 10.0), (10.1, 10.2), (75.0, 60.0), (-2.0, 0.0)],
)
def test_pressure_lands_in_bucket_centred_within_a_tenth(
    pressure: float, bucket_centre: float
) -> None:
    assert _bucket_ids(np.array([pressure]))[0] / 5 == bucket_centre


def test_each_sample_holds_until_the_next_and_the_last_adds_nothing() -> None:
    tap = _time_at_pressure({1: _series([8.0, 8.0, 9.0, 12.0])}, None, {1: []})

    assert tap is not None
    assert tap.pressures[0] == 8.0
    assert tap.pressures[-1] == 9.0
    assert tap.pressure_seconds[0] == 4.0
    assert tap.pressure_seconds[-1] == 2.0
    assert sum(tap.pressure_seconds) == 6.0
    assert tap.epap_seconds is None


def _two_segments() -> tuple[np.ndarray, np.ndarray, float]:
    """8.0 from 0 to 20 s, a recording gap, then 10.0 from 600 to 620 s."""
    offsets = np.concatenate([np.arange(0.0, 21.0, 2.0), np.arange(600.0, 621.0, 2.0)])
    return offsets, np.repeat([8.0, 10.0], 11), 0.5


def test_each_segment_skips_its_first_ten_seconds_and_ends_at_the_gap() -> None:
    tap = _time_at_pressure({1: _two_segments()}, None, {1: []})

    assert tap is not None
    by_pressure = dict(zip(tap.pressures, tap.pressure_seconds, strict=True))
    # 10, 12, 14, 16 and 18 s hold 2 s each; 20 s ends at the gap.
    assert by_pressure[8.0] == 10.0
    assert by_pressure[10.0] == 10.0
    assert sum(tap.pressure_seconds) == 20.0


def test_event_in_a_recording_gap_or_after_the_last_sample_is_not_counted() -> None:
    flags = {1: [("OA", 20.0), ("OA", 300.0), ("H", 620.0), ("H", 650.0)]}

    tap = _time_at_pressure({1: _two_segments()}, None, flags)

    assert tap is not None
    by_pressure = dict(zip(tap.pressures, tap.event_counts["OA"], strict=True))
    # At the last sample before the gap counts; inside the gap does not.
    assert by_pressure[8.0] == 1
    assert sum(tap.event_counts["OA"]) == 1
    # At the last sample counts; after it does not.
    assert sum(tap.event_counts["H"]) == 1


def test_non_finite_sample_adds_no_time() -> None:
    tap = _time_at_pressure({1: _series([8.0, np.nan, 8.0, 8.0])}, None, {1: []})

    assert tap is not None
    assert tap.pressure_seconds == [4.0]


def test_event_counts_at_the_last_sample_at_or_before_its_flag() -> None:
    flags = {1: [("OA", 13.9), ("OA", 14.0), ("H", -5.0), ("CA", 15.0)]}

    tap = _time_at_pressure({1: _series([8.0, 9.0, np.nan, 10.0])}, None, flags)

    assert tap is not None
    assert tap.pressures[0] == 8.0
    by_pressure = dict(zip(tap.pressures, tap.event_counts["OA"], strict=True))
    # 13.9 s holds the 9.0 sample (at 12 s); 14.0 s is the NaN one, not counted.
    assert by_pressure[9.0] == 1
    assert sum(tap.event_counts["OA"]) == 1
    # Flagged before the first sample: the first sample's bucket.
    assert tap.event_counts["H"][0] == 1
    # The CA at the NaN sample is not counted, so the type has no column.
    assert "CA" not in tap.event_counts


def test_range_spans_both_channels() -> None:
    tap = _time_at_pressure(
        {1: _series([12.0, 12.0])}, {1: _series([6.0, 6.0])}, {1: []}
    )

    assert tap is not None
    assert tap.pressures[0] == 6.0
    assert tap.pressures[-1] == 12.0
    assert tap.epap_seconds is not None
    assert tap.epap_seconds[0] == 2.0
    assert tap.pressure_seconds[-1] == 2.0


def test_single_sample_without_events_has_no_time_at_pressure() -> None:
    assert _time_at_pressure({1: _series([8.0])}, None, {1: []}) is None
