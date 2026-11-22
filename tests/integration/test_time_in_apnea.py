"""Total time in apnea on the session, day and MCP night surfaces."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Day, Device, Event, Session
from snore.mcp.tools.night_detail import get_night_detail
from snore.services.day_service import DayService
from snore.services.session_service import SessionService
from tests.integration.conftest import (
    _make_day_session,
    _make_device,
    _make_event,
    _make_profile,
)

pytestmark = pytest.mark.integration

_NIGHT = date(2025, 3, 1)
_SECOND_START = datetime(2025, 3, 2, 3, 0)


@pytest.fixture
async def profile_device(async_db_session: AsyncSession) -> tuple[int, Device]:
    profile = await _make_profile(async_db_session)
    return profile.id, await _make_device(async_db_session, profile.id)


async def _undurated_event(db: AsyncSession, session: Session, event_type: str) -> None:
    """Store an event with no recorded duration, as ResMed writes H and RE."""
    db.add(
        Event(
            session_id=session.id,
            event_type=event_type,
            start_time=session.start_time + timedelta(minutes=5),
            duration_seconds=None,
        )
    )
    await db.flush()


async def _second_session(
    db: AsyncSession, device: Device, day: Day, *, enabled: bool
) -> Session:
    session = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"second-{enabled}",
        start_time=_SECOND_START,
        end_time=_SECOND_START + timedelta(hours=1),
        duration_seconds=3600.0,
        enabled=enabled,
    )
    db.add(session)
    await db.flush()
    return session


async def test_session_time_in_apnea_sums_ahi_event_durations_only(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    for event_type, seconds in [
        ("OA", 12.0),
        ("CA", 20.5),
        ("UA", 11.0),
        ("CAA", 10.0),
        ("H", 9.0),
        ("RE", 7.0),
        ("FL", 30.0),
    ]:
        await _make_event(
            async_db_session, session, event_type=event_type, duration_seconds=seconds
        )
    await _undurated_event(async_db_session, session, "H")

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    # OA + CA + UA + CAA + H; the RERA, the FL and the undurated H add nothing.
    assert detail.time_in_apnea_seconds == pytest.approx(62.5)


async def test_session_with_only_undurated_events_has_zero_time_in_apnea(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _undurated_event(async_db_session, session, "H")

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id, include_channel_stats=False
    )

    assert detail.time_in_apnea_seconds == 0.0


async def test_day_time_in_apnea_covers_enabled_sessions_only(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, first = await _make_day_session(async_db_session, device, _NIGHT)
    second = await _second_session(async_db_session, device, day, enabled=True)
    disabled = await _second_session(async_db_session, device, day, enabled=False)
    await _make_event(async_db_session, first, event_type="OA", duration_seconds=15.0)
    await _make_event(async_db_session, second, event_type="CA", duration_seconds=25.0)
    await _make_event(
        async_db_session, disabled, event_type="OA", duration_seconds=60.0
    )

    detail = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    assert detail.time_in_apnea_seconds == pytest.approx(40.0)


async def test_day_with_no_enabled_session_has_null_time_in_apnea(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    session.enabled = False
    await _make_event(async_db_session, session, event_type="OA", duration_seconds=15.0)

    detail = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    assert detail.time_in_apnea_seconds is None


async def test_night_detail_carries_day_and_per_session_time_in_apnea(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, first = await _make_day_session(async_db_session, device, _NIGHT)
    disabled = await _second_session(async_db_session, device, day, enabled=False)
    await _make_event(async_db_session, first, event_type="OA", duration_seconds=15.0)
    await _make_event(async_db_session, disabled, event_type="CA", duration_seconds=9.0)

    response = await get_night_detail(async_db_session, _NIGHT, profile_id)

    assert response.device is not None
    assert response.device.time_in_apnea_seconds == pytest.approx(15.0)
    assert {s.id: s.time_in_apnea_seconds for s in response.sessions} == {
        first.id: pytest.approx(15.0),
        disabled.id: pytest.approx(9.0),
    }
