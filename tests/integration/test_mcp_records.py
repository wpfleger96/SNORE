"""Integration tests: get_records against a real SQLite DB."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.mcp.tools.records import get_records
from snore.services.stats_service import StatsService
from tests.integration.conftest import (
    _make_apple_night,
    _make_day_session,
    _make_device,
    _make_profile,
)

pytestmark = pytest.mark.integration


async def _seed_nights(db: AsyncSession, profile_id: int) -> None:
    device = await _make_device(db, profile_id)
    for night, ahi, leak, hours in (
        (date(2026, 3, 1), 4.0, 12.0, 6.0),
        (date(2026, 3, 2), 1.0, 3.0, 8.0),
        (date(2026, 3, 3), 2.5, 20.0, 7.0),
        (date(2026, 3, 4), 0.2, 1.0, 0.5),  # under 1 h: never ranked
    ):
        await _make_day_session(
            db, device, night, duration_hours=hours, ahi=ahi, leak_median=leak
        )
    await _make_apple_night(db, profile_id, date(2026, 3, 2), 7.25 * 3600)


async def test_records_rank_qualifying_nights_like_the_stats_service(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    await _seed_nights(async_db_session, async_test_profile.id)

    result = await get_records(
        async_db_session, profile_id=async_test_profile.id, top_n=2
    )

    expected = await StatsService(async_db_session, async_test_profile.id).get_records(
        top_n=2
    )
    assert result.model_dump() == expected.model_dump()
    assert result.ahi is not None
    assert result.ahi.best == [(date(2026, 3, 2), 1.0), (date(2026, 3, 3), 2.5)]
    assert result.ahi.worst == [(date(2026, 3, 1), 4.0), (date(2026, 3, 3), 2.5)]
    assert result.therapy_hours is not None
    assert result.therapy_hours.worst[0] == (date(2026, 3, 1), 6.0)
    assert result.total_sleep_hours is not None
    assert result.total_sleep_hours.best == [(date(2026, 3, 2), 7.25)]
    assert result.spo2_min is None


async def test_range_bounds_and_profile_scope_apply(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    await _seed_nights(async_db_session, async_test_profile.id)
    other_device = await _make_device(
        async_db_session, (await _make_profile(async_db_session)).id
    )
    await _make_day_session(async_db_session, other_device, date(2026, 3, 2), ahi=9.0)
    after_end = date(2026, 3, 5)
    own_device = await _make_device(async_db_session, async_test_profile.id)
    await _make_day_session(
        async_db_session,
        own_device,
        after_end,
        duration_hours=9.5,
        ahi=6.0,
        leak_median=30.0,
    )
    await _make_apple_night(
        async_db_session, async_test_profile.id, after_end, 9.75 * 3600
    )

    result = await get_records(
        async_db_session,
        profile_id=async_test_profile.id,
        start=date(2026, 3, 2),
        end=date(2026, 3, 3),
    )

    assert result.ahi is not None
    assert result.ahi.worst == [(date(2026, 3, 3), 2.5), (date(2026, 3, 2), 1.0)]
    assert result.leak is not None
    assert result.leak.worst == [(date(2026, 3, 3), 20.0), (date(2026, 3, 2), 3.0)]
    assert result.therapy_hours is not None
    assert result.therapy_hours.best == [
        (date(2026, 3, 2), 8.0),
        (date(2026, 3, 3), 7.0),
    ]
    assert result.total_sleep_hours is not None
    assert result.total_sleep_hours.best == [(date(2026, 3, 2), 7.25)]


async def test_range_without_data_returns_every_metric_null(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    await _seed_nights(async_db_session, async_test_profile.id)

    result = await get_records(
        async_db_session,
        profile_id=async_test_profile.id,
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
    )

    assert result.model_dump() == dict.fromkeys(type(result).model_fields)


async def test_two_devices_on_one_night_rank_as_separate_entries(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    first = await _make_device(async_db_session, async_test_profile.id)
    second = await _make_device(async_db_session, async_test_profile.id)
    await _make_day_session(async_db_session, first, date(2026, 3, 1), ahi=3.0)
    await _make_day_session(async_db_session, second, date(2026, 3, 1), ahi=2.0)
    await _make_day_session(async_db_session, first, date(2026, 3, 2), ahi=1.5)

    result = await get_records(async_db_session, profile_id=async_test_profile.id)

    assert result.ahi is not None
    assert result.ahi.best == [
        (date(2026, 3, 2), 1.5),
        (date(2026, 3, 1), 2.0),
        (date(2026, 3, 1), 3.0),
    ]


async def test_night_without_a_metric_value_is_skipped_for_that_metric_only(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    device = await _make_device(async_db_session, async_test_profile.id)
    await _make_day_session(
        async_db_session,
        device,
        date(2026, 3, 1),
        duration_hours=7.0,
        ahi=3.0,
        leak_median=10.0,
    )
    await _make_day_session(
        async_db_session,
        device,
        date(2026, 3, 2),
        duration_hours=6.5,
        ahi=1.5,
        leak_median=None,
    )

    result = await get_records(async_db_session, profile_id=async_test_profile.id)

    assert result.leak is not None
    assert result.leak.best == [(date(2026, 3, 1), 10.0)]
    assert result.leak.worst == [(date(2026, 3, 1), 10.0)]
    assert result.ahi is not None
    assert result.ahi.best == [(date(2026, 3, 2), 1.5), (date(2026, 3, 1), 3.0)]
    assert result.therapy_hours is not None
    assert result.therapy_hours.best == [
        (date(2026, 3, 1), 7.0),
        (date(2026, 3, 2), 6.5),
    ]


async def test_apple_night_without_device_night_ranks_in_total_sleep_hours(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    device = await _make_device(async_db_session, async_test_profile.id)
    await _make_day_session(async_db_session, device, date(2026, 3, 1), ahi=3.0)
    await _make_apple_night(
        async_db_session, async_test_profile.id, date(2026, 3, 1), 7.5 * 3600
    )
    await _make_apple_night(
        async_db_session, async_test_profile.id, date(2026, 3, 3), 6.75 * 3600
    )

    result = await get_records(async_db_session, profile_id=async_test_profile.id)

    assert result.total_sleep_hours is not None
    assert result.total_sleep_hours.best == [
        (date(2026, 3, 1), 7.5),
        (date(2026, 3, 3), 6.75),
    ]
    assert result.ahi is not None
    assert result.ahi.best == [(date(2026, 3, 1), 3.0)]
