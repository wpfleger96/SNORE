"""Integration tests for the user-declared profile timezone (A6).

A profile may declare an IANA timezone name.  When set, MCP responses carry
``timezone_status: "user_declared"`` plus ``timezone_name`` beside every tier-2
wall-clock anchor.  When timezone is declared, wall-clock strings are
offset-qualified ISO 8601 (e.g. "2024-08-18T22:00:00-04:00"); when unknown,
strings stay offset-free (naive ISO 8601).  The DB stores naive datetimes;
no UTC offset is ever fabricated via ``.timestamp()`` / ``astimezone()``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Event
from tests.integration.conftest import (
    _make_analysis_result,
    _make_day_session,
    _make_device,
    _make_event,
)
from tests.integration.test_mcp_breath_table import _make_breath
from tests.integration.test_mcp_waveform import _make_waveform

TZ = "America/New_York"


def _assert_naive(wall_clock: str) -> None:
    """Tier-2 wall-clock strings stay offset-free when no TZ is declared."""
    assert "+" not in wall_clock
    assert not wall_clock.endswith("Z")
    # Must also not end in an explicit negative offset like "-05:00"
    parsed = datetime.fromisoformat(wall_clock)
    assert parsed.tzinfo is None


def _assert_offset_aware(wall_clock: str) -> None:
    """Tier-2 wall-clock strings carry a UTC offset when TZ is user_declared."""
    parsed = datetime.fromisoformat(wall_clock)
    assert parsed.tzinfo is not None, (
        f"Expected offset-aware timestamp, got: {wall_clock!r}"
    )


@pytest.fixture
async def tz_profile(async_db_session: AsyncSession, async_test_profile: Any) -> Any:
    """The standard test profile with a declared IANA timezone."""
    async_test_profile.timezone = TZ
    await async_db_session.flush()
    return async_test_profile


class TestUserDeclaredTimezone:
    async def test_get_events_carries_user_declared_timezone(
        self, async_db_session: AsyncSession, tz_profile: Any
    ) -> None:
        from snore.mcp.tools.events import get_events  # noqa: PLC0415

        target_date = date(2024, 8, 18)
        device = await _make_device(async_db_session, tz_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=sess.start_time,
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session, target_date, profile_id=tz_profile.id
        )

        assert result.timezone_status == "user_declared"
        assert result.timezone_name == TZ
        ev = result.events[0]
        assert ev.timezone_status == "user_declared"
        assert ev.timezone_name == TZ
        _assert_offset_aware(ev.start_time_wall_clock)
        _assert_offset_aware(ev.session_start_wall_clock)

    async def test_get_breath_table_carries_user_declared_timezone(
        self, async_db_session: AsyncSession, tz_profile: Any
    ) -> None:
        from snore.mcp.tools.breath_table import get_breath_table  # noqa: PLC0415

        target_date = date(2024, 2, 1)
        device = await _make_device(async_db_session, tz_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)
        await _make_breath(async_db_session, ar, sess, breath_number=1)
        await async_db_session.flush()

        result = await get_breath_table(
            async_db_session,
            target_date,
            profile_id=tz_profile.id,
            offset_start=0.0,
            offset_end=900.0,
        )

        assert result.timezone_status == "user_declared"
        assert result.timezone_name == TZ
        row = result.rows[0]
        assert row.timezone_status == "user_declared"
        assert row.timezone_name == TZ
        _assert_offset_aware(row.session_start_wall_clock)

    async def test_find_windows_carries_user_declared_timezone(
        self, async_db_session: AsyncSession, tz_profile: Any
    ) -> None:
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 2, 1)
        device = await _make_device(async_db_session, tz_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        await _make_event(async_db_session, sess, offset_seconds=300.0)

        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=tz_profile.id,
            criterion="ca_centered",
        )

        assert len(result.windows) == 1
        win = result.windows[0]
        assert win.timezone_status == "user_declared"
        assert win.timezone_name == TZ
        _assert_offset_aware(win.session_start_wall_clock)

    async def test_get_waveform_carries_user_declared_timezone(
        self, async_db_session: AsyncSession, tz_profile: Any
    ) -> None:
        from snore.mcp.tools.waveform import (  # noqa: PLC0415
            fetch_waveform_raw,
            waveform_response_from_raw,
        )

        target_date = date(2024, 3, 1)
        device = await _make_device(async_db_session, tz_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        await _make_waveform(async_db_session, sess, "flow", 25.0, "L/min", 750)

        raw = await fetch_waveform_raw(
            async_db_session,
            target_date,
            profile_id=tz_profile.id,
            offset_start=0.0,
            offset_end=10.0,
            window_cap_seconds=120.0,
        )
        response = waveform_response_from_raw(raw)

        assert response.timezone_status == "user_declared"
        assert response.timezone_name == TZ
        assert response.session_start_wall_clock is not None
        _assert_offset_aware(response.session_start_wall_clock)

    async def test_get_data_overview_carries_timezone_status(
        self, async_db_session: AsyncSession, tz_profile: Any
    ) -> None:
        """get_data_overview stamps timezone_status/timezone_name when TZ is declared."""
        from snore.mcp.tools.overview import get_data_overview  # noqa: PLC0415

        target_date = date(2024, 8, 18)
        device = await _make_device(async_db_session, tz_profile.id)
        await _make_day_session(async_db_session, device, target_date)

        result = await get_data_overview(async_db_session, tz_profile.id)

        assert result.timezone_status == "user_declared"
        assert result.timezone_name == TZ

    async def test_get_data_overview_unknown_when_no_timezone(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Control: no declared timezone → get_data_overview returns status "unknown"."""
        from snore.mcp.tools.overview import get_data_overview  # noqa: PLC0415

        target_date = date(2024, 8, 18)
        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(async_db_session, device, target_date)

        result = await get_data_overview(async_db_session, async_test_profile.id)

        assert result.timezone_status == "unknown"
        assert result.timezone_name is None

    async def test_undeclared_profile_stays_unknown_with_null_name(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Control: no declared timezone → status "unknown", timestamps stay naive."""
        from snore.mcp.tools.events import get_events  # noqa: PLC0415

        target_date = date(2024, 8, 18)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=sess.start_time,
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session, target_date, profile_id=async_test_profile.id
        )

        assert result.timezone_status == "unknown"
        assert result.timezone_name is None
        ev = result.events[0]
        assert ev.timezone_status == "unknown"
        assert ev.timezone_name is None
        _assert_naive(ev.start_time_wall_clock)
        _assert_naive(ev.session_start_wall_clock)


class TestCorruptedTimezoneGracefulDegradation:
    """A corrupted profile timezone degrades to naive timestamps instead of raising."""

    async def test_invalid_iana_zone_degrades_to_naive(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Profile with invalid IANA timezone: get_events returns naive timestamps, no error."""
        from snore.mcp.tools.events import get_events  # noqa: PLC0415

        async_test_profile.timezone = "Not/AZone"
        await async_db_session.flush()

        target_date = date(2024, 8, 18)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=sess.start_time,
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session, target_date, profile_id=async_test_profile.id
        )

        # The bad zone name propagates through to timezone_name (it's the raw profile value),
        # but wall-clock strings fall back to naive rather than raising ZoneInfoNotFoundError.
        assert result.timezone_status == "user_declared"
        ev = result.events[0]
        _assert_naive(ev.start_time_wall_clock)
        _assert_naive(ev.session_start_wall_clock)
