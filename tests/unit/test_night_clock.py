"""Hours after noon of a night's date, and back to a clock time."""

from datetime import date, datetime, time

import pytest

from snore.parsers.base import hours_into_night, night_clock_time


@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        (0.0, time(12, 0)),
        (11.5, time(23, 30)),
        (12.0, time(0, 0)),
        (19.0, time(7, 0)),
        (25.0, time(13, 0)),
    ],
)
def test_night_clock_time_wraps_past_midnight_and_noon(
    hours: float, expected: time
) -> None:
    assert night_clock_time(hours) == expected


def test_night_clock_time_half_minute_rounds_up() -> None:
    # 22.5 and 52.5 minutes past noon, which Python's round() takes to the even 22
    # and 52 where JavaScript's Math.round (the web UI) gives 23 and 53.
    assert night_clock_time(0.375) == time(12, 23)
    assert night_clock_time(0.875) == time(12, 53)


def test_night_clock_time_inverts_hours_into_night() -> None:
    moment = datetime(2024, 3, 16, 6, 45)

    assert night_clock_time(hours_into_night(moment, date(2024, 3, 15))) == time(6, 45)
