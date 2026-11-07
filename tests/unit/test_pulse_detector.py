"""Tests for pulse change detection (OSCAR calcPulseChange parity)."""

import numpy as np
import pytest

from snore.analysis.shared.pulse_detector import PulseChangeDetector


def _step_series(
    steps: list[tuple[float, float]], duration_s: float = 120.0
) -> tuple[np.ndarray, np.ndarray]:
    """1 Hz pulse at 60 bpm, then each (time, bpm) step holds until the next."""
    timestamps = np.arange(0.0, duration_s, 1.0)
    values = np.full_like(timestamps, 60.0)
    for step_time, bpm in steps:
        values[timestamps >= step_time] = bpm
    return timestamps, values


@pytest.fixture
def detector() -> PulseChangeDetector:
    return PulseChangeDetector(bpm_threshold=5.0, duration_threshold=8.0)


def test_single_step_change_yields_one_event_at_change_time(detector):
    timestamps, values = _step_series([(30.0, 70.0)])

    events = detector.detect(timestamps, values)

    assert len(events) == 1
    assert events[0].start_time == 30.0
    assert events[0].duration == 8.0  # scan window started 8 s before the change
    assert events[0].bpm_delta == 10.0


def test_two_separated_step_changes_yield_two_events(detector):
    timestamps, values = _step_series([(30.0, 70.0), (80.0, 60.0)])

    events = detector.detect(timestamps, values)

    assert [e.start_time for e in events] == [30.0, 80.0]


def test_event_is_stamped_at_last_qualifying_delta_in_window(detector):
    """Scanning resumes at 31 s (70 bpm), whose window sees 80 bpm at 33-39 s.

    Every one of those deltas qualifies; the change is the last, at 39 s.
    """
    timestamps, values = _step_series([(30.0, 70.0), (33.0, 80.0)])

    events = detector.detect(timestamps, values)

    assert [(e.start_time, e.duration) for e in events] == [(30.0, 8.0), (39.0, 8.0)]


def test_delta_equal_to_threshold_is_not_an_event(detector):
    timestamps, values = _step_series([(30.0, 65.0)])

    assert detector.detect(timestamps, values) == []


def test_change_beyond_window_is_not_seen_across_nan_gap(detector):
    timestamps, values = _step_series([(30.0, 70.0)])
    values[(timestamps >= 25.0) & (timestamps < 35.0)] = np.nan

    events = detector.detect(timestamps, values)

    assert events == []


def test_nan_samples_inside_window_are_skipped(detector):
    timestamps, values = _step_series([(30.0, 70.0)])
    values[(timestamps >= 27.0) & (timestamps < 30.0)] = np.nan

    events = detector.detect(timestamps, values)

    assert len(events) == 1
    assert events[0].start_time == 30.0


@pytest.mark.parametrize("no_reading", [-1.0, 0.0])
def test_no_reading_dropouts_on_flat_pulse_return_no_events(detector, no_reading):
    """ResMed's -1 / 0 "no reading" samples are not heart-rate changes."""
    timestamps, values = _step_series([], duration_s=3600.0)
    for dropout_start in (600.0, 1800.0, 3000.0):
        values[(timestamps >= dropout_start) & (timestamps < dropout_start + 10)] = (
            no_reading
        )

    assert detector.detect(timestamps, values) == []


def test_nan_timestamp_is_skipped(detector):
    timestamps, values = _step_series([(30.0, 70.0)])
    timestamps[10] = np.nan

    events = detector.detect(timestamps, values)

    assert [e.start_time for e in events] == [30.0]


def test_all_nan_pulse_returns_no_events(detector):
    timestamps = np.arange(0.0, 60.0, 1.0)

    assert detector.detect(timestamps, np.full_like(timestamps, np.nan)) == []
