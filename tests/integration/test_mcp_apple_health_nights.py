"""Integration tests: get_apple_health_nights against a real SQLite DB."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.mcp.tools.apple_health_nights import get_apple_health_nights
from tests.integration.conftest import _make_apple_night, _make_profile

pytestmark = pytest.mark.integration

NIGHTS = [date(2026, 3, d) for d in (1, 2, 4, 7, 9)]


async def _seed_nights(db: AsyncSession, profile_id: int) -> None:
    for i, night in enumerate(NIGHTS):
        await _make_apple_night(db, profile_id, night, 25200.0 + i)


async def test_lists_nights_newest_first_with_summaries(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    await _seed_nights(async_db_session, async_test_profile.id)

    result = await get_apple_health_nights(
        async_db_session, profile_id=async_test_profile.id
    )

    assert [n.night_date for n in result.nights] == NIGHTS[::-1]
    assert result.nights[0].total_sleep_seconds == 25204.0
    assert result.nights[0].sleep_efficiency_pct == 90.0
    assert (result.total_nights, result.page, result.page_size) == (5, 1, 30)


async def test_date_range_is_inclusive_and_total_counts_the_range(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    await _seed_nights(async_db_session, async_test_profile.id)

    result = await get_apple_health_nights(
        async_db_session,
        profile_id=async_test_profile.id,
        start=date(2026, 3, 2),
        end=date(2026, 3, 7),
    )

    assert [n.night_date for n in result.nights] == [
        date(2026, 3, 7),
        date(2026, 3, 4),
        date(2026, 3, 2),
    ]
    assert result.total_nights == 3


async def test_pages_continue_where_the_previous_page_ended(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    await _seed_nights(async_db_session, async_test_profile.id)

    pages = [
        await get_apple_health_nights(
            async_db_session, profile_id=async_test_profile.id, page=p, page_size=2
        )
        for p in (1, 2, 3, 4)
    ]

    assert [[n.night_date for n in page.nights] for page in pages] == [
        NIGHTS[4:2:-1],
        NIGHTS[2:0:-1],
        NIGHTS[:1],
        [],
    ]
    assert {page.total_nights for page in pages} == {5}


async def test_other_profiles_nights_are_excluded(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    other = await _make_profile(async_db_session)
    await _seed_nights(async_db_session, other.id)
    await _make_apple_night(
        async_db_session, async_test_profile.id, date(2026, 3, 9), 20000.0
    )

    result = await get_apple_health_nights(
        async_db_session, profile_id=async_test_profile.id
    )

    assert [n.total_sleep_seconds for n in result.nights] == [20000.0]
    assert result.total_nights == 1


async def test_profile_without_apple_health_returns_empty_page(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    result = await get_apple_health_nights(
        async_db_session, profile_id=async_test_profile.id
    )

    assert result.nights == []
    assert result.total_nights == 0
