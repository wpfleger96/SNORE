"""Integration tests: get_night_detail and get_period_stats against a real SQLite DB."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import (
    Day,
    Device,
    HealthNightlySummary,
    Session,
    Statistics,
)
from snore.mcp.errors import ValidationError
from snore.mcp.tools.night_detail import get_night_detail
from snore.mcp.tools.period_stats import get_period_stats
from tests.integration.conftest import (
    _make_day_session,
    _make_device,
    _make_event,
    _make_profile,
)

pytestmark = pytest.mark.integration

NIGHT = date(2026, 3, 10)


async def _add_session(
    db: AsyncSession, device: Device, day: Day, start: datetime, *, enabled: bool
) -> Session:
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"extra_{start.isoformat()}",
        start_time=start,
        end_time=start + timedelta(hours=1),
        duration_seconds=3600.0,
        therapy_mode="APAP",
        enabled=enabled,
    )
    db.add(sess)
    await db.flush()
    return sess


async def _add_apple_night(
    db: AsyncSession, profile_id: int, night: date, total_sleep_seconds: float
) -> None:
    db.add(
        HealthNightlySummary(
            profile_id=profile_id,
            night_date=night,
            total_sleep_seconds=total_sleep_seconds,
            sleep_efficiency_pct=90.0,
        )
    )
    await db.flush()


class TestGetNightDetail:
    async def test_sessions_carry_device_statistics_in_ml(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        device = await _make_device(async_db_session, async_test_profile.id)
        day, first = await _make_day_session(
            async_db_session, device, NIGHT, ahi=2.5, index_source="device"
        )
        disabled = await _add_session(
            async_db_session, device, day, first.end_time, enabled=False
        )
        async_db_session.add(
            Statistics(
                session_id=first.id,
                ahi_device=2.5,
                rin=0.4,
                csr_pct=1.5,
                amb_humidity_median=45.0,
                tidal_volume_mean=0.45,  # ResMed STR litres
            )
        )
        await async_db_session.flush()

        result = await get_night_detail(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )

        assert result.device is not None
        assert result.device.ahi == 2.5
        assert result.device.index_source == "device"
        assert [s.id for s in result.sessions] == [first.id, disabled.id]
        assert [s.enabled for s in result.sessions] == [True, False]
        stats = result.sessions[0].statistics
        assert stats is not None
        assert (stats.ahi_device, stats.rin, stats.csr_pct) == (2.5, 0.4, 1.5)
        assert stats.amb_humidity_median == 45.0
        assert stats.tidal_volume_mean == 450.0
        assert stats.reras is None  # the device never scored a RERA
        assert result.sessions[1].statistics is None
        assert result.sessions[1].therapy_mode == "APAP"
        assert result.apple_health is None

    async def test_scored_rera_type_reports_zero_counts(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, NIGHT)
        await _make_event(async_db_session, sess, event_type="RE")
        async_db_session.add(Statistics(session_id=sess.id, reras=0))
        await async_db_session.flush()

        result = await get_night_detail(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )

        stats = result.sessions[0].statistics
        assert stats is not None
        assert stats.reras == 0

    async def test_apple_only_night_returns_apple_block(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        await _add_apple_night(async_db_session, async_test_profile.id, NIGHT, 25200.0)

        result = await get_night_detail(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )

        assert result.device is None
        assert result.sessions == []
        assert result.apple_health is not None
        assert result.apple_health.total_sleep_seconds == 25200.0

    async def test_apple_night_is_not_repeated_in_device_block(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(async_db_session, device, NIGHT)
        await _add_apple_night(async_db_session, async_test_profile.id, NIGHT, 25200.0)

        result = await get_night_detail(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )

        assert result.device is not None
        assert result.device.health_sleep is None
        assert result.apple_health is not None

    async def test_night_with_no_data_raises(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        with pytest.raises(ValidationError, match="No therapy or Apple Health data"):
            await get_night_detail(
                async_db_session, NIGHT, profile_id=async_test_profile.id
            )

    async def test_other_profiles_night_is_not_returned(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        other = await _make_profile(async_db_session)
        other_device = await _make_device(async_db_session, other.id)
        await _make_day_session(async_db_session, other_device, NIGHT)
        await _add_apple_night(async_db_session, other.id, NIGHT, 25200.0)

        with pytest.raises(ValidationError):
            await get_night_detail(
                async_db_session, NIGHT, profile_id=async_test_profile.id
            )


class TestGetPeriodStats:
    async def test_month_periods_include_apple_sleep_averages(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        device = await _make_device(async_db_session, async_test_profile.id)
        for night, ahi in ((date(2026, 2, 3), 1.0), (date(2026, 3, 4), 3.0)):
            await _make_day_session(async_db_session, device, night, ahi=ahi)
        await _add_apple_night(
            async_db_session, async_test_profile.id, date(2026, 3, 4), 7.5 * 3600
        )

        result = await get_period_stats(
            async_db_session, profile_id=async_test_profile.id
        )

        assert [p.period_start for p in result.periods] == [
            date(2026, 2, 1),
            date(2026, 3, 1),
        ]
        assert [p.avg_ahi for p in result.periods] == [1.0, 3.0]
        assert result.periods[0].avg_total_sleep_hours is None
        assert result.periods[1].avg_total_sleep_hours == 7.5
        assert result.periods[1].avg_sleep_efficiency_pct == 90.0

    async def test_range_bounds_and_profile_scope_apply(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        device = await _make_device(async_db_session, async_test_profile.id)
        for night in (date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3)):
            await _make_day_session(async_db_session, device, night)
        other_device = await _make_device(
            async_db_session, (await _make_profile(async_db_session)).id
        )
        await _make_day_session(async_db_session, other_device, date(2026, 3, 2))

        result = await get_period_stats(
            async_db_session,
            profile_id=async_test_profile.id,
            period_type="day",
            start=date(2026, 3, 2),
            end=date(2026, 3, 3),
        )

        assert [(p.period_start, p.days_used) for p in result.periods] == [
            (date(2026, 3, 2), 1),
            (date(2026, 3, 3), 1),
        ]
