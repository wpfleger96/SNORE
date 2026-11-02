"""A zero count of an event type a device never scores reads as absent, not zero.

Covers ``device_event_types`` / ``reported_count`` and the surfaces built on them:
session detail, day detail, the MCP nightly summary and period statistics.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Device, Statistics
from snore.mcp.tools.summary import get_nightly_summary
from snore.metrics import reported_count
from snore.services.day_service import DayService
from snore.services.event_service import device_event_types
from snore.services.session_service import SessionService
from snore.services.stats_service import StatsService
from tests.integration.conftest import (
    _make_day_session,
    _make_device,
    _make_event,
    _make_profile,
)

_NIGHT = date(2025, 3, 1)
_OTHER_NIGHT = date(2025, 3, 2)


def test_reported_count_hides_only_an_unreported_zero() -> None:
    assert reported_count(0, "RE", set()) is None
    assert reported_count(0, "RE", {"RE"}) == 0
    assert reported_count(3, "RE", set()) == 3
    assert reported_count(None, "RE", {"RE"}) is None


async def test_device_event_types_covers_disabled_sessions_per_device(
    async_db_session: AsyncSession,
) -> None:
    profile = await _make_profile(async_db_session)
    device = await _make_device(async_db_session, profile.id)
    quiet_device = await _make_device(async_db_session, profile.id)
    _, disabled = await _make_day_session(async_db_session, device, _NIGHT)
    disabled.enabled = False
    await _make_event(async_db_session, disabled, event_type="RE")
    await _make_event(async_db_session, disabled, event_type="OA")

    reported = await device_event_types(async_db_session, [device.id, quiet_device.id])

    assert reported == {device.id: {"RE", "OA"}, quiet_device.id: set()}


@contextmanager
def _captured_sql(db: AsyncSession) -> Iterator[list[tuple[str, Any]]]:
    """Collect the (statement, parameters) pairs ``db`` executes."""
    captured: list[tuple[str, Any]] = []

    def capture(
        _conn: Any, _cursor: Any, statement: str, parameters: Any, *_args: Any
    ) -> None:
        captured.append((statement, parameters))

    engine = db.get_bind()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        yield captured
    finally:
        event.remove(engine, "before_cursor_execute", capture)


async def test_device_event_types_without_devices_runs_no_query(
    async_db_session: AsyncSession,
) -> None:
    with _captured_sql(async_db_session) as captured:
        reported = await device_event_types(async_db_session, [])

    assert reported == {}
    assert captured == []


async def test_device_event_types_query_does_not_scan_events(
    async_db_session: AsyncSession,
) -> None:
    profile = await _make_profile(async_db_session)
    device = await _make_device(async_db_session, profile.id)
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_event(async_db_session, session, event_type="RE")

    with _captured_sql(async_db_session) as captured:
        await device_event_types(async_db_session, [device.id])

    [(statement, parameters)] = captured
    connection = await async_db_session.connection()
    plan = await connection.exec_driver_sql(
        f"EXPLAIN QUERY PLAN {statement}", parameters
    )
    details = [row.detail for row in plan]
    assert details
    assert not [detail for detail in details if "SCAN events" in detail]


async def _seed_night(
    db: AsyncSession, device: Device, *, reras: int = 0, unclassified: int = 0
) -> Any:
    """A night with OA, plus matching session statistics and Day counts."""
    day, session = await _make_day_session(
        db, device, _NIGHT, reras=reras, unclassified_apneas=unclassified
    )
    db.add(
        Statistics(
            session_id=session.id,
            obstructive_apneas=1,
            reras=reras,
            unclassified_apneas=unclassified,
        )
    )
    await _make_event(db, session, event_type="OA")
    await db.flush()
    return session


async def _seed_reporting_device(db: AsyncSession, device: Device) -> Any:
    """A zero night on a device that scored RE and UA on another night."""
    session = await _seed_night(db, device)
    _, other = await _make_day_session(db, device, _OTHER_NIGHT)
    await _make_event(db, other, event_type="RE")
    await _make_event(db, other, event_type="UA")
    return session


async def test_counts_are_null_on_a_device_that_never_scored_them(
    async_db_session: AsyncSession,
) -> None:
    profile = await _make_profile(async_db_session)
    device = await _make_device(async_db_session, profile.id)
    session = await _seed_night(async_db_session, device)

    detail = await SessionService(async_db_session, profile.id).get_session_detail(
        session.id
    )
    day = await DayService(async_db_session, profile.id).get_day(_NIGHT)
    summary = await get_nightly_summary(
        async_db_session, _NIGHT, _NIGHT, profile_id=profile.id
    )

    assert detail.statistics is not None
    assert detail.statistics.reras is None
    assert detail.statistics.unclassified_apneas is None
    assert detail.statistics.mixed_apneas is None
    assert detail.statistics.flow_limitations is None
    assert detail.statistics.obstructive_apneas == 1
    assert (day.reras, day.unclassified_apneas) == (None, None)
    night = summary.nights[0]
    assert (night.reras, night.unclassified_apneas) == (None, None)
    assert night.reras_reason == "not_reported_by_device"
    assert night.unclassified_apneas_reason == "not_reported_by_device"


async def test_zero_stays_zero_on_a_device_that_scored_one_on_another_session(
    async_db_session: AsyncSession,
) -> None:
    profile = await _make_profile(async_db_session)
    device = await _make_device(async_db_session, profile.id)
    session = await _seed_reporting_device(async_db_session, device)

    detail = await SessionService(async_db_session, profile.id).get_session_detail(
        session.id
    )
    day = await DayService(async_db_session, profile.id).get_day(_NIGHT)
    summary = await get_nightly_summary(
        async_db_session, _NIGHT, _NIGHT, profile_id=profile.id
    )

    assert detail.statistics is not None
    assert detail.statistics.reras == 0
    assert detail.statistics.unclassified_apneas == 0
    assert (day.reras, day.unclassified_apneas) == (0, 0)
    night = summary.nights[0]
    assert (night.reras, night.unclassified_apneas) == (0, 0)
    assert (night.reras_reason, night.unclassified_apneas_reason) == (None, None)


async def test_non_zero_count_is_never_hidden(
    async_db_session: AsyncSession,
) -> None:
    """Only zeros are hidden, so a stored count shows even with no event rows."""
    profile = await _make_profile(async_db_session)
    device = await _make_device(async_db_session, profile.id)
    session = await _seed_night(async_db_session, device, reras=2, unclassified=3)

    detail = await SessionService(async_db_session, profile.id).get_session_detail(
        session.id
    )
    day = await DayService(async_db_session, profile.id).get_day(_NIGHT)

    assert detail.statistics is not None
    assert detail.statistics.reras == 2
    assert detail.statistics.unclassified_apneas == 3
    assert (day.reras, day.unclassified_apneas) == (2, 3)


async def test_nightly_decides_per_device(async_db_session: AsyncSession) -> None:
    profile = await _make_profile(async_db_session)
    reporting = await _make_device(async_db_session, profile.id)
    quiet = await _make_device(async_db_session, profile.id)
    await _seed_reporting_device(async_db_session, reporting)
    await _seed_night(async_db_session, quiet)

    rows = {
        device.id: (
            await get_nightly_summary(
                async_db_session,
                _NIGHT,
                _NIGHT,
                profile_id=profile.id,
                device_id=device.id,
            )
        ).nights[0]
        for device in (reporting, quiet)
    }

    assert rows[reporting.id].reras == 0
    assert rows[quiet.id].reras is None


async def test_period_rera_is_none_without_a_reporting_device(
    async_db_session: AsyncSession,
) -> None:
    profile = await _make_profile(async_db_session)
    device = await _make_device(async_db_session, profile.id)
    await _seed_night(async_db_session, device)
    stats = StatsService(async_db_session, profile.id)

    periods = await stats.get_period_statistics("day")
    trends = await stats.get_trends("day")

    assert [p.avg_rera for p in periods] == [None]
    assert [value for _, value in trends.rera] == [None]


async def test_period_rera_averages_only_reporting_devices(
    async_db_session: AsyncSession,
) -> None:
    """Same month, two devices: the quiet device's zero days are left out."""
    profile = await _make_profile(async_db_session)
    reporting = await _make_device(async_db_session, profile.id)
    quiet = await _make_device(async_db_session, profile.id)
    _, session = await _make_day_session(async_db_session, reporting, _NIGHT, reras=4)
    await _make_event(async_db_session, session, event_type="RE")
    await _make_day_session(async_db_session, reporting, _OTHER_NIGHT, reras=0)
    await _make_day_session(async_db_session, quiet, date(2025, 3, 3), reras=0)

    periods = await StatsService(async_db_session, profile.id).get_period_statistics(
        "month"
    )

    assert [p.avg_rera for p in periods] == [0.25]
