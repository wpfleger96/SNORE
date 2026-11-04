"""OSCAR importer timezone conversion tests (A6).

OSCAR stores epoch-ms instants.  With a profile-declared IANA timezone the
parser converts them to that zone's wall clock (matching ResMed device-local
semantics); without one it keeps the legacy naive UTC wall-clock.  These tests
pin both behaviors and the noon-cutoff therapy-date off-by-one regression.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfoNotFoundError

import pytest

from snore.constants import CPAP_OBSTRUCTIVE
from snore.database.day_manager import DayManager
from snore.parsers.oscar_device import OscarDeviceParser, _epoch_to_wall_clock
from tests.helpers.oscar_builders import (
    make_event_list,
    make_events,
    make_summary,
    parse_oscar_session,
)

pytestmark = pytest.mark.unit

# Fixed winter instant: 2024-01-15 00:30 UTC == 2024-01-14 19:30 EST (UTC-5,
# no DST ambiguity).
WINTER_START_MS = int(datetime(2024, 1, 15, 0, 30, tzinfo=UTC).timestamp() * 1000)
WINTER_END_MS = WINTER_START_MS + 6 * 3600 * 1000  # 06:30 UTC == 01:30 EST


class TestDeclaredTimezoneConversion:
    def test_declared_timezone_yields_local_wall_clock(self, tmp_path):
        session = parse_oscar_session(
            tmp_path,
            make_summary(WINTER_START_MS, WINTER_END_MS),
            timezone_name="America/New_York",
        )

        # 00:30 UTC Jan 15 == 19:30 EST Jan 14 — naive, matching ResMed semantics
        assert session.start_time == datetime(2024, 1, 14, 19, 30)
        assert session.start_time.tzinfo is None
        assert session.end_time == datetime(2024, 1, 15, 1, 30)
        assert session.end_time.tzinfo is None

        # 19:30 is after the noon cutoff → therapy night of Jan 14
        assert DayManager.get_day_for_session(session.start_time) == date(2024, 1, 14)

    def test_no_timezone_keeps_utc_wall_clock(self, tmp_path):
        session = parse_oscar_session(
            tmp_path, make_summary(WINTER_START_MS, WINTER_END_MS)
        )

        assert session.start_time == datetime(2024, 1, 15, 0, 30)
        assert session.start_time.tzinfo is None
        assert session.end_time == datetime(2024, 1, 15, 6, 30)

        # Legacy behavior: UTC clock puts 00:30 before noon → Jan 14 by luck,
        # but the wall-clock hour itself is UTC, not local.
        assert DayManager.get_day_for_session(session.start_time) == date(2024, 1, 14)

    def test_event_times_use_declared_timezone(self, tmp_path):
        event_at_ms = WINTER_START_MS + 3600 * 1000  # 01:30 UTC == 20:30 EST Jan 14
        events = make_events(
            WINTER_START_MS,
            WINTER_END_MS,
            make_event_list(CPAP_OBSTRUCTIVE, event_at_ms, [15]),
        )

        session = parse_oscar_session(
            tmp_path,
            make_summary(WINTER_START_MS, WINTER_END_MS),
            events,
            timezone_name="America/New_York",
        )

        assert len(session.events) == 1
        event = session.events[0]
        # OSCAR timestamps are end-of-event; parser subtracts duration (15s) to
        # store the true start: 20:30:00 - 15s = 20:29:45.
        assert event.start_time == datetime(2024, 1, 14, 20, 29, 45)
        assert event.start_time.tzinfo is None
        assert event.duration_seconds == 15.0

    def test_noon_cutoff_off_by_one_regression(self, tmp_path):
        """16:30 UTC == 11:30 EST: opposite sides of the noon cutoff.

        UTC clock assigns the session to Jan 15; the declared-timezone local
        clock correctly assigns it to the night of Jan 14.
        """
        start_ms = int(datetime(2024, 1, 15, 16, 30, tzinfo=UTC).timestamp() * 1000)
        end_ms = start_ms + 2 * 3600 * 1000

        utc_session = parse_oscar_session(tmp_path, make_summary(start_ms, end_ms))
        local_session = parse_oscar_session(
            tmp_path, make_summary(start_ms, end_ms), timezone_name="America/New_York"
        )

        assert DayManager.get_day_for_session(utc_session.start_time) == date(
            2024, 1, 15
        )
        assert DayManager.get_day_for_session(local_session.start_time) == date(
            2024, 1, 14
        )


class TestDstTransitionNight:
    """A session straddling the real US fall-back transition of 2024-11-03.

    America/New_York falls back at 06:00 UTC (02:00 EDT → 01:00 EST): each
    epoch instant must convert with its own offset — UTC-4 before the
    transition, UTC-5 after — and the therapy date must stay the night of
    Nov 2 throughout.
    """

    # 05:30 UTC == 01:30 EDT (UTC-4), before the fall-back.
    START_MS = int(datetime(2024, 11, 3, 5, 30, tzinfo=UTC).timestamp() * 1000)
    # 07:30 UTC == 02:30 EST (UTC-5), after the fall-back.
    END_MS = int(datetime(2024, 11, 3, 7, 30, tzinfo=UTC).timestamp() * 1000)

    def test_each_instant_converts_with_its_own_offset(self, tmp_path):
        # Event at 06:30 UTC == 01:30 EST — the repeated wall-clock hour.
        event_at_ms = int(datetime(2024, 11, 3, 6, 30, tzinfo=UTC).timestamp() * 1000)
        events = make_events(
            self.START_MS,
            self.END_MS,
            make_event_list(CPAP_OBSTRUCTIVE, event_at_ms, [15]),
        )

        session = parse_oscar_session(
            tmp_path,
            make_summary(self.START_MS, self.END_MS),
            events,
            timezone_name="America/New_York",
        )

        # Start converts with the pre-transition EDT offset (UTC-4).
        assert session.start_time == datetime(2024, 11, 3, 1, 30)
        # End converts with the post-transition EST offset (UTC-5): 2 h of
        # elapsed time advance the wall clock by only 1 h across the fold.
        assert session.end_time == datetime(2024, 11, 3, 2, 30)

        # The event lands in the repeated 01:30 wall-clock hour (EST fold),
        # one real hour after the session start's identical wall-clock value.
        # OSCAR timestamps are end-of-event; parser subtracts duration (15s) to
        # store the true start: 01:30:00 - 15s = 01:29:45.
        assert len(session.events) == 1
        assert session.events[0].start_time == datetime(2024, 11, 3, 1, 29, 45)

    def test_therapy_date_stays_correct_across_transition(self, tmp_path):
        session = parse_oscar_session(
            tmp_path,
            make_summary(self.START_MS, self.END_MS),
            timezone_name="America/New_York",
        )

        # Both endpoints are pre-noon local → therapy night of Nov 2 on both
        # sides of the transition.
        assert DayManager.get_day_for_session(session.start_time) == date(2024, 11, 2)
        assert DayManager.get_day_for_session(session.end_time) == date(2024, 11, 2)

    def test_offsets_flip_at_the_transition_instant(self):
        before = datetime(2024, 11, 3, 5, 59, tzinfo=UTC).timestamp()
        after = datetime(2024, 11, 3, 6, 1, tzinfo=UTC).timestamp()
        # 05:59 UTC → 01:59 EDT; two minutes later, 06:01 UTC → 01:01 EST:
        # the wall clock steps backwards because each instant uses its own
        # offset.
        assert _epoch_to_wall_clock(before, "America/New_York") == datetime(
            2024, 11, 3, 1, 59
        )
        assert _epoch_to_wall_clock(after, "America/New_York") == datetime(
            2024, 11, 3, 1, 1
        )


class TestEpochToWallClock:
    def test_dst_summer_offset(self):
        # 2024-07-15 00:30 UTC == 20:30 EDT (UTC-4) July 14
        seconds = datetime(2024, 7, 15, 0, 30, tzinfo=UTC).timestamp()
        assert _epoch_to_wall_clock(seconds, "America/New_York") == datetime(
            2024, 7, 14, 20, 30
        )

    def test_none_is_utc_wall_clock(self):
        seconds = datetime(2024, 7, 15, 0, 30, tzinfo=UTC).timestamp()
        result = _epoch_to_wall_clock(seconds, None)
        assert result == datetime(2024, 7, 15, 0, 30)
        assert result.tzinfo is None

    def test_session_date_filter_uses_same_clock(self):
        # The date-filter sites derive session_date from epoch-seconds session
        # IDs with the same helper, so filtering agrees with day assignment.
        session_id = int(datetime(2024, 1, 15, 0, 30, tzinfo=UTC).timestamp())
        assert _epoch_to_wall_clock(session_id, "America/New_York").date() == date(
            2024, 1, 14
        )
        assert _epoch_to_wall_clock(session_id, None).date() == date(2024, 1, 15)


class TestInvalidTimezone:
    def test_invalid_zone_fails_loudly(self, tmp_path):
        parser = OscarDeviceParser()
        with pytest.raises(ZoneInfoNotFoundError):
            list(parser.parse_sessions(tmp_path, timezone_name="Not/A_Zone"))
