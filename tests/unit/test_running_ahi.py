"""OSCAR's running AHI over a session's mask-on segments."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from snore.services.running_ahi import RunningAhiSource, running_ahi

pytestmark = pytest.mark.unit

_START = datetime(2025, 3, 1, 22, 0)


def _source(
    hours: float, segments: list[tuple[float, float]] | None
) -> RunningAhiSource:
    return RunningAhiSource(
        session_id=1,
        start_time=_START,
        end_time=_START + timedelta(hours=hours),
        mask_on_segments=segments,
    )


def test_event_counts_for_the_next_hour_then_drops_out() -> None:
    [segment] = running_ahi([_source(2, [(0, 7200)])], {1: [("OA", 90.0)]})

    assert len(segment.values) == 120
    # Flagged at 1:30; counted from the 2-minute step through the 61-minute one.
    assert segment.values[:2] == [0, 0]
    assert segment.values[2:62] == [1] * 60
    assert segment.values[62:] == [0] * 58


def test_window_includes_both_ends() -> None:
    [segment] = running_ahi([_source(2, [(0, 7200)])], {1: [("H", 60.0)]})

    # Flagged exactly at the 1-minute step, and exactly 60 minutes before the 61st.
    assert segment.values[1] == 1
    assert segment.values[61] == 1
    assert segment.values[62] == 0


def test_window_never_reaches_into_an_earlier_segment() -> None:
    segments = running_ahi(
        [_source(3, [(0, 3600), (5400, 10800)])],
        {1: [("CA", 3500.0), ("OA", 5500.0)]},
    )

    assert [s.start for s in segments] == [_START, _START + timedelta(seconds=5400)]
    assert len(segments[0].values) == 60
    assert segments[0].values[-1] == 1
    # The second segment counts only its own event.
    assert max(segments[1].values) == 1
    assert segments[1].values[0] == 0


def test_unknown_segments_cover_the_whole_session() -> None:
    [segment] = running_ahi([_source(1, None)], {1: [("UA", 30.0)]})

    assert segment.start == _START
    assert len(segment.values) == 60
    assert segment.values[1] == 1


def test_stored_segments_are_clipped_to_the_session_span() -> None:
    [segment] = running_ahi([_source(1, [(0, 1e12), (2e12, 3e12)])], {1: []})

    assert len(segment.values) == 60


def test_sessions_are_merged_by_segment_start() -> None:
    late = RunningAhiSource(
        session_id=2,
        start_time=_START - timedelta(hours=1),
        end_time=_START,
        mask_on_segments=[(0, 600)],
    )

    segments = running_ahi([_source(1, [(0, 600)]), late], {1: [], 2: []})

    assert [s.session_id for s in segments] == [2, 1]


def test_segment_longer_than_two_days_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        segments = running_ahi([_source(24 * 365, None)], {1: []})

    assert segments == []
    assert "Session 1" in caplog.text
