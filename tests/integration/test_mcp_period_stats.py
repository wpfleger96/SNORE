"""Integration tests: get_period_stats against a real SQLite DB."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.mcp.tools.period_stats import get_period_stats
from tests.integration.conftest import (
    _make_apple_night,
    _make_day_session,
    _make_device,
    _make_profile,
)

pytestmark = pytest.mark.integration


async def test_month_periods_include_apple_sleep_averages(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    device = await _make_device(async_db_session, async_test_profile.id)
    for night, ahi in ((date(2026, 2, 3), 1.0), (date(2026, 3, 4), 3.0)):
        await _make_day_session(async_db_session, device, night, ahi=ahi)
    await _make_apple_night(
        async_db_session, async_test_profile.id, date(2026, 3, 4), 7.5 * 3600
    )

    result = await get_period_stats(async_db_session, profile_id=async_test_profile.id)

    assert [p.period_start for p in result.periods] == [
        date(2026, 2, 1),
        date(2026, 3, 1),
    ]
    assert [p.avg_ahi for p in result.periods] == [1.0, 3.0]
    assert result.periods[0].avg_total_sleep_hours is None
    assert result.periods[1].avg_total_sleep_hours == 7.5
    assert result.periods[1].avg_sleep_efficiency_pct == 90.0


async def test_periods_carry_compliance_days_and_percentage(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    device = await _make_device(async_db_session, async_test_profile.id)
    for night, hours in ((date(2026, 3, 2), 4.0), (date(2026, 3, 4), 3.0)):
        await _make_day_session(async_db_session, device, night, duration_hours=hours)

    [march] = (
        await get_period_stats(async_db_session, profile_id=async_test_profile.id)
    ).periods

    # One of the 3 calendar days from March 2 to March 4.
    assert march.days_compliant == 1
    assert march.compliance_pct == 33.3


async def test_range_bounds_and_profile_scope_apply(
    async_db_session: AsyncSession, async_test_profile: Any
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
