"""Pressure, EPAP and leak 99.5th, Time at Pressure and running AHI on the
session, day and MCP night surfaces."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Device, Event, Session, Statistics
from snore.mcp.tools.night_detail import get_night_detail
from snore.services.day_service import DayService
from snore.services.session_service import SessionService
from tests.integration.conftest import (
    _make_channel_waveform,
    _make_day_session,
    _make_device,
    _make_profile,
)

pytestmark = pytest.mark.integration

_NIGHT = date(2025, 3, 1)
_SECOND_START = datetime(2025, 3, 2, 3, 0)


@pytest.fixture
async def profile_device(async_db_session: AsyncSession) -> tuple[int, Device]:
    profile = await _make_profile(async_db_session)
    return profile.id, await _make_device(async_db_session, profile.id)


async def _add_stats(db: AsyncSession, session: Session, **values: float) -> None:
    db.add(Statistics(session_id=session.id, usage_hours=1.0, **values))
    await db.flush()


async def _second_session(db: AsyncSession, first: Session) -> Session:
    second = Session(
        device_id=first.device_id,
        day_id=first.day_id,
        device_session_id="second",
        start_time=_SECOND_START,
        end_time=_SECOND_START + timedelta(hours=1),
        duration_seconds=3600.0,
        enabled=True,
    )
    db.add(second)
    await db.flush()
    return second


async def test_day_99_5th_pools_every_session_samples(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first = await _make_day_session(async_db_session, device, _NIGHT, 1.0)
    second = await _second_session(async_db_session, first)
    first_pressure = [float(v) for v in range(4, 14)]
    second_pressure = [float(v) for v in range(10, 30)]
    for session, pressure in ((first, first_pressure), (second, second_pressure)):
        await _add_stats(
            async_db_session, session, pressure_95th=1.0, epap_95th=1.0, leak_95th=1.0
        )
        await _make_channel_waveform(
            async_db_session, session, "therapy_pressure", pressure
        )
        await _make_channel_waveform(
            async_db_session, session, "epap", [p - 4 for p in pressure]
        )
        await _make_channel_waveform(async_db_session, session, "leak", pressure)

    day = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    pooled = np.array(first_pressure + second_pressure)
    assert day.pressure_99_5th == pytest.approx(np.percentile(pooled, 99.5))
    assert day.epap_99_5th == pytest.approx(np.percentile(pooled - 4, 99.5))
    assert day.leak_99_5th == pytest.approx(np.percentile(pooled, 99.5))


async def test_99_5th_is_null_when_a_covered_session_has_no_waveform(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first = await _make_day_session(async_db_session, device, _NIGHT, 1.0)
    second = await _second_session(async_db_session, first)
    await _add_stats(async_db_session, first, pressure_95th=10.0)
    await _add_stats(async_db_session, second, pressure_95th=12.0)
    await _make_channel_waveform(
        async_db_session, first, "therapy_pressure", [10.0] * 5
    )

    day = await DayService(async_db_session, profile_id).get_day(_NIGHT)
    session = await SessionService(async_db_session, profile_id).get_session_detail(
        first.id
    )

    assert day.pressure_99_5th is None
    assert day.time_at_pressure is None
    # The session alone is complete.
    assert session.pressure_99_5th == 10.0
    assert session.time_at_pressure is not None


async def test_session_without_stored_value_is_not_covered(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first = await _make_day_session(async_db_session, device, _NIGHT, 1.0)
    await _add_stats(async_db_session, first)
    await _make_channel_waveform(
        async_db_session, first, "therapy_pressure", [10.0] * 5
    )

    session = await SessionService(async_db_session, profile_id).get_session_detail(
        first.id
    )

    assert session.pressure_99_5th is None
    assert session.time_at_pressure is None


async def test_day_time_at_pressure_counts_events_at_their_flag_pressure(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first = await _make_day_session(async_db_session, device, _NIGHT, 1.0)
    await _add_stats(async_db_session, first, pressure_95th=1.0)
    # 8 cmH2O for 10 s, then 12 cmH2O.
    await _make_channel_waveform(
        async_db_session, first, "therapy_pressure", [8.0] * 5 + [12.0] * 5
    )
    # Starts at 8 s but is flagged at its end, 14 s, during 12 cmH2O.
    async_db_session.add_all(
        [
            Event(
                session_id=first.id,
                event_type="OA",
                start_time=first.start_time + timedelta(seconds=8),
                duration_seconds=6.0,
            ),
            Event(
                session_id=first.id,
                event_type="RE",
                start_time=first.start_time + timedelta(seconds=2),
            ),
        ]
    )
    await async_db_session.flush()

    tap = (
        await DayService(async_db_session, profile_id).get_day(_NIGHT)
    ).time_at_pressure

    assert tap is not None
    seconds = dict(zip(tap.pressures, tap.pressure_seconds, strict=True))
    assert seconds[8.0] == 10.0
    assert seconds[12.0] == 8.0
    assert set(tap.event_counts) == {"OA"}
    assert dict(zip(tap.pressures, tap.event_counts["OA"], strict=True))[12.0] == 1


async def test_running_ahi_covers_enabled_sessions_only(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first = await _make_day_session(async_db_session, device, _NIGHT, 1.0)
    first.mask_on_segments = [[0.0, 1800.0]]
    second = await _second_session(async_db_session, first)
    second.enabled = False
    async_db_session.add(
        Event(
            session_id=first.id,
            event_type="H",
            start_time=first.start_time + timedelta(seconds=150),
        )
    )
    await async_db_session.flush()

    day = await DayService(async_db_session, profile_id).get_day(_NIGHT)
    session = await SessionService(async_db_session, profile_id).get_session_detail(
        second.id
    )

    assert day.running_ahi is not None
    [segment] = day.running_ahi
    assert segment.session_id == first.id
    assert len(segment.values) == 30
    assert segment.values[2:4] == [0, 1]
    # A disabled session still has its own running AHI.
    assert session.running_ahi is not None
    assert len(session.running_ahi[0].values) == 60


async def test_patch_style_session_detail_leaves_views_null(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first = await _make_day_session(async_db_session, device, _NIGHT, 1.0)
    await _add_stats(async_db_session, first, pressure_95th=1.0)
    await _make_channel_waveform(async_db_session, first, "therapy_pressure", [9.0] * 5)

    session = await SessionService(async_db_session, profile_id).get_session_detail(
        first.id, include_channel_stats=False
    )

    assert session.pressure_99_5th is None
    assert session.time_at_pressure is None
    assert session.running_ahi is None


async def test_mcp_night_detail_device_carries_the_views(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first = await _make_day_session(async_db_session, device, _NIGHT, 1.0)
    await _add_stats(async_db_session, first, pressure_95th=1.0)
    await _make_channel_waveform(async_db_session, first, "therapy_pressure", [9.0] * 5)

    night = await get_night_detail(async_db_session, _NIGHT, profile_id)
    day = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    assert night.device is not None
    assert night.device.pressure_99_5th == day.pressure_99_5th == 9.0
    assert night.device.time_at_pressure == day.time_at_pressure
    assert night.device.running_ahi == day.running_ahi
