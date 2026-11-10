"""Integration tests: get_apple_sleep_stages against a real SQLite DB."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import HealthSample
from snore.mcp.errors import ValidationError
from snore.mcp.tools.apple_sleep_stages import get_apple_sleep_stages
from snore.parsers.apple_health.type_handlers import SLEEP_TYPE, SPO2_TYPE
from tests.integration.conftest import _make_apple_night, _make_profile

pytestmark = pytest.mark.integration

NIGHT = date(2026, 3, 10)
WATCH = "Watch"
PHONE = "Phone"
BEDTIME = datetime(2026, 3, 10, 23, 0)


def _add_sample(
    db: AsyncSession,
    profile_id: int,
    source: str,
    start_minute: int,
    *,
    stage: str | None = "AsleepCore",
    record_type: str = SLEEP_TYPE,
) -> None:
    start = BEDTIME + timedelta(minutes=start_minute)
    db.add(
        HealthSample(
            profile_id=profile_id,
            record_type=record_type,
            source_name=source,
            start_time=start,
            end_time=start + timedelta(minutes=30),
            value_text=stage,
            value_num=None if stage else 96.0,
            night_date=NIGHT,
            ingest_channel="export_xml",
        )
    )


async def _seed_two_source_night(
    db: AsyncSession, profile_id: int, preferred_source: str | None
) -> None:
    """Watch stages (inserted out of order), a phone InBed and a Watch SpO2 sample."""
    await _make_apple_night(
        db, profile_id, NIGHT, 25200.0, preferred_source=preferred_source
    )
    _add_sample(db, profile_id, WATCH, 30, stage="AsleepDeep")
    _add_sample(db, profile_id, WATCH, 0, stage="AsleepCore")
    _add_sample(db, profile_id, PHONE, -10, stage="InBed")
    _add_sample(db, profile_id, WATCH, 5, stage=None, record_type=SPO2_TYPE)
    await db.flush()


async def test_returns_only_preferred_source_stages(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    await _seed_two_source_night(async_db_session, async_test_profile.id, WATCH)

    result = await get_apple_sleep_stages(
        async_db_session, NIGHT, profile_id=async_test_profile.id
    )

    assert result.date == NIGHT
    assert [(s.source_name, s.value_text) for s in result.stages] == [
        (WATCH, "AsleepCore"),
        (WATCH, "AsleepDeep"),
    ]


async def test_night_without_preferred_source_returns_every_source(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    await _seed_two_source_night(async_db_session, async_test_profile.id, None)

    result = await get_apple_sleep_stages(
        async_db_session, NIGHT, profile_id=async_test_profile.id
    )

    assert [s.value_text for s in result.stages] == [
        "InBed",
        "AsleepCore",
        "AsleepDeep",
    ]


async def test_night_with_no_apple_data_raises(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    with pytest.raises(
        ValidationError, match=f"^No Apple Health sleep data found for {NIGHT}$"
    ):
        await get_apple_sleep_stages(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )


async def test_other_profiles_night_raises(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    other = await _make_profile(async_db_session)
    await _seed_two_source_night(async_db_session, other.id, WATCH)

    with pytest.raises(ValidationError, match="No Apple Health sleep data found"):
        await get_apple_sleep_stages(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )


async def test_other_profiles_stages_on_same_night_are_excluded(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    other = await _make_profile(async_db_session)
    await _seed_two_source_night(async_db_session, other.id, WATCH)
    await _make_apple_night(
        async_db_session, async_test_profile.id, NIGHT, 25200.0, preferred_source=WATCH
    )
    _add_sample(async_db_session, async_test_profile.id, WATCH, 60, stage="AsleepREM")
    await async_db_session.flush()

    result = await get_apple_sleep_stages(
        async_db_session, NIGHT, profile_id=async_test_profile.id
    )

    assert [s.value_text for s in result.stages] == ["AsleepREM"]
