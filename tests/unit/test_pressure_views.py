"""Time at Pressure bucketing, time weighting and event placement."""

from __future__ import annotations

import numpy as np
import pytest

from snore.services.pressure_views import _bucket_ids, _time_at_pressure

pytestmark = pytest.mark.unit


def _series(
    values: list[float], step: float = 2.0
) -> tuple[np.ndarray, np.ndarray, float]:
    offsets = np.arange(len(values)) * step
    return offsets, np.array(values), 0.5


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


def test_recording_gap_ends_the_sample_before_it() -> None:
    offsets = np.array([0.0, 2.0, 600.0, 602.0])
    tap = _time_at_pressure(
        {1: (offsets, np.array([8.0, 8.0, 10.0, 10.0]), 0.5)}, None, {1: []}
    )

    assert tap is not None
    assert sum(tap.pressure_seconds) == 4.0


def test_non_finite_sample_adds_no_time() -> None:
    tap = _time_at_pressure({1: _series([8.0, np.nan, 8.0, 8.0])}, None, {1: []})

    assert tap is not None
    assert tap.pressure_seconds == [4.0]


def test_event_counts_at_the_last_sample_at_or_before_its_flag() -> None:
    flags = {1: [("OA", 3.9), ("OA", 4.0), ("H", -5.0), ("CA", 5.0)]}

    tap = _time_at_pressure({1: _series([8.0, 9.0, np.nan, 10.0])}, None, flags)

    assert tap is not None
    assert tap.pressures[0] == 8.0
    by_pressure = dict(zip(tap.pressures, tap.event_counts["OA"], strict=True))
    # 3.9 s holds the 9.0 sample (at 2 s); 4.0 s is the NaN sample and not counted.
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
