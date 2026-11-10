"""Device channel statistics and mask-on spans on the session, day, period and
MCP nightly surfaces."""

from __future__ import annotations

import logging
import threading

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pytest

from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.calculations import PeriodType
from snore.database.models import Day, Device, Session, Statistics
from snore.database.waveform_blobs import decode_channel_blobs
from snore.mcp.schemas import EpochSpec, NightlyRow
from snore.mcp.tools.epochs import compare_epochs
from snore.mcp.tools.night_detail import get_night_detail
from snore.mcp.tools.summary import get_nightly_summary
from snore.services.breath_service import BreathService
from snore.services.day_service import DayService
from snore.services.report_service import ReportService
from snore.services.session_service import SessionService
from snore.services.stats_service import StatsService
from tests.integration.conftest import (
    _make_channel_waveform,
    _make_day_session,
    _make_device,
    _make_profile,
)

_NIGHT = date(2025, 3, 1)
_NEXT_NIGHT = date(2025, 3, 2)
# The waveform channels DeviceChannelStats reports.
_REPORTED_CHANNELS = {"fl", "snore", "rr", "tv", "mv", "ti", "ie_ratio", "pulse"}


async def _add_session(
    db: AsyncSession,
    device: Device,
    day: Day,
    *,
    start: datetime,
    hours: float = 1.0,
    enabled: bool = True,
    mask_on_segments: list[list[float]] | None = None,
) -> Session:
    """A second session on an existing Day row."""
    session = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"extra_{start.isoformat()}",
        start_time=start,
        end_time=start + timedelta(hours=hours),
        duration_seconds=hours * 3600,
        enabled=enabled,
        mask_on_segments=mask_on_segments,
    )
    db.add(session)
    await db.flush()
    return session


async def _night_session(
    db: AsyncSession,
    device: Device,
    night: date,
    start: datetime,
    *,
    fl: list[float] | None = None,
    snore: list[float] | None = None,
) -> Session:
    """A session starting at ``start`` on ``night``'s Day row, created if new."""
    day = (
        await db.execute(
            select(Day).where(Day.device_id == device.id, Day.date == night)
        )
    ).scalar_one_or_none()
    if day is None:
        day = Day(device_id=device.id, date=night, total_therapy_hours=8.0)
        db.add(day)
        await db.flush()
    session = await _add_session(db, device, day, start=start)
    if fl is not None:
        await _make_channel_waveform(db, session, "fl", fl)
    if snore is not None:
        await _make_channel_waveform(db, session, "snore", snore)
    return session


async def _seed_night_with_a_disabled_session(db: AsyncSession, device: Device) -> None:
    """Two enabled sessions and a disabled one, sentinels and zeros on the channels."""
    day, first = await _make_day_session(db, device, _NIGHT)
    second = await _add_session(db, device, day, start=datetime(2025, 3, 2, 3, 0))
    disabled = await _add_session(
        db, device, day, start=datetime(2025, 3, 2, 5, 0), enabled=False
    )
    await _make_channel_waveform(
        db, first, "fl", [-0.01, 0.0, 0.13, 0.27, 0.0, 0.61, 0.07]
    )
    await _make_channel_waveform(db, second, "fl", [0.33, -0.01, 0.41, 0.19, 0.0, 0.92])
    await _make_channel_waveform(db, disabled, "fl", [1.0])
    await _make_channel_waveform(db, first, "snore", [0.0, 0.2, 1.7, -0.01, 3.3, 0.9])
    await _make_channel_waveform(db, second, "snore", [0.4, 0.0, 2.2, 4.8])


@pytest.fixture
async def profile_device(async_db_session: AsyncSession) -> tuple[int, Device]:
    profile = await _make_profile(async_db_session)
    return profile.id, await _make_device(async_db_session, profile.id)


async def test_session_detail_reports_its_own_channels(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(
        async_db_session, session, "fl", [-0.01, 0.0, 0.0, 0.2, 0.4]
    )
    await _make_channel_waveform(async_db_session, session, "snore", [0.0, 1.0, 3.0])

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    stats = detail.channel_stats
    assert stats.device_flg_median == pytest.approx(0.1)
    assert stats.device_flg_95th == pytest.approx(0.4)
    assert stats.device_flg_max == pytest.approx(0.4)
    assert stats.snore_median == 1.0
    assert stats.snore_95th == 3.0
    assert stats.snore_max == 3.0


async def test_session_detail_includes_a_disabled_session(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    session.enabled = False
    await _make_channel_waveform(async_db_session, session, "fl", [0.3])

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    assert detail.channel_stats.device_flg_max == pytest.approx(0.3)


async def test_session_without_either_channel_has_null_channel_stats(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "pressure", [9.0])

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    assert detail.channel_stats is None


async def test_session_missing_one_channel_nulls_only_that_channel(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [0.3])

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    assert detail.channel_stats.device_flg_max == pytest.approx(0.3)
    assert detail.channel_stats.snore_max is None


async def test_corrupt_blob_skips_that_channel_and_warns(
    async_db_session: AsyncSession,
    profile_device: tuple[int, Device],
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    corrupt = await _make_channel_waveform(async_db_session, session, "fl", [0.3])
    corrupt.data_blob = b"not a whole sample"
    await _make_channel_waveform(async_db_session, session, "snore", [2.0])
    await async_db_session.flush()

    with caplog.at_level(logging.WARNING):
        detail = await SessionService(async_db_session, profile_id).get_session_detail(
            session.id
        )

    assert detail.channel_stats.device_flg_max is None
    assert detail.channel_stats.snore_max == 2.0
    assert f"session {session.id}" in caplog.text
    assert (
        sum(
            r.levelno >= logging.WARNING
            for r in caplog.records
            if r.name == "snore.database.waveform_blobs"
        )
        == 1
    )


async def test_empty_blob_is_skipped(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [])

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    assert detail.channel_stats is None


async def test_session_detail_carries_device_reported_ahi(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    async_db_session.add(Statistics(session_id=session.id, ahi_device=4.2))
    await async_db_session.flush()

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    assert detail.statistics.ahi_device == pytest.approx(4.2)


async def test_day_pools_the_samples_of_its_sessions(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, first = await _make_day_session(async_db_session, device, _NIGHT)
    second = await _add_session(
        async_db_session, device, day, start=datetime(2025, 3, 2, 3, 0)
    )
    await _make_channel_waveform(async_db_session, first, "fl", [0.0] * 19)
    await _make_channel_waveform(async_db_session, second, "fl", [1.0])

    stats = (
        await DayService(async_db_session, profile_id).get_day(_NIGHT)
    ).channel_stats

    # Averaging the two sessions' 95ths would read 0.5.
    assert stats.device_flg_95th == 1.0
    assert stats.device_flg_median == 0.0


async def test_day_excludes_disabled_sessions(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, enabled = await _make_day_session(async_db_session, device, _NIGHT)
    enabled.mask_on_segments = [[0.0, 60.0]]
    disabled = await _add_session(
        async_db_session,
        device,
        day,
        start=datetime(2025, 3, 2, 3, 0),
        enabled=False,
        mask_on_segments=[[0.0, 600.0]],
    )
    await _make_channel_waveform(async_db_session, enabled, "fl", [0.1])
    await _make_channel_waveform(async_db_session, disabled, "fl", [0.9])

    result = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    assert result.channel_stats.device_flg_max == pytest.approx(0.1)
    assert [span.session_id for span in result.mask_on_spans] == [enabled.id]
    assert result.session_ids == [enabled.id, disabled.id]


async def test_day_with_only_disabled_sessions_has_no_channel_stats(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    session.enabled = False
    await _make_channel_waveform(async_db_session, session, "fl", [0.9])

    result = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    assert result.channel_stats is None
    assert result.mask_on_spans == []


async def test_session_mask_on_spans_cover_every_segment(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    session.mask_on_segments = [[0.0, 1800.0], [3600.0, 5400.0]]

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    assert [(s.start, s.end, s.duration_seconds) for s in detail.mask_on_spans] == [
        (datetime(2025, 3, 1, 22, 0), datetime(2025, 3, 1, 22, 30), 1800.0),
        (datetime(2025, 3, 1, 23, 0), datetime(2025, 3, 1, 23, 30), 1800.0),
    ]


async def test_session_unknown_segments_have_null_spans(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)

    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    assert detail.mask_on_spans is None


async def test_day_spans_ascend_across_overlapping_sessions(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, first = await _make_day_session(async_db_session, device, _NIGHT)
    first.mask_on_segments = [[0.0, 600.0], [7200.0, 7800.0]]
    await _add_session(
        async_db_session,
        device,
        day,
        start=datetime(2025, 3, 1, 23, 0),
        mask_on_segments=[[0.0, 60.0]],
    )

    spans = (
        await DayService(async_db_session, profile_id).get_day(_NIGHT)
    ).mask_on_spans

    # In session order the second session's span would come last.
    assert [s.start for s in spans] == [
        datetime(2025, 3, 1, 22, 0),
        datetime(2025, 3, 1, 23, 0),
        datetime(2025, 3, 2, 0, 0),
    ]


async def test_day_spans_are_null_when_any_enabled_session_is_unknown(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, known = await _make_day_session(async_db_session, device, _NIGHT)
    known.mask_on_segments = [[0.0, 600.0]]
    await _add_session(async_db_session, device, day, start=datetime(2025, 3, 2, 3, 0))

    result = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    assert result.mask_on_spans is None


async def test_day_spans_ignore_unknown_segments_of_a_disabled_session(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, known = await _make_day_session(async_db_session, device, _NIGHT)
    known.mask_on_segments = [[0.0, 600.0]]
    await _add_session(
        async_db_session,
        device,
        day,
        start=datetime(2025, 3, 2, 3, 0),
        enabled=False,
    )

    result = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    assert [span.session_id for span in result.mask_on_spans] == [known.id]


async def test_day_with_known_zero_segments_has_empty_spans(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    session.mask_on_segments = []

    result = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    assert result.mask_on_spans == []


async def test_period_pools_samples_across_nights(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, quiet = await _make_day_session(async_db_session, device, _NIGHT)
    _, loud = await _make_day_session(async_db_session, device, _NEXT_NIGHT)
    await _make_channel_waveform(async_db_session, quiet, "fl", [0.0] * 19)
    await _make_channel_waveform(async_db_session, loud, "fl", [1.0])
    await _make_channel_waveform(async_db_session, quiet, "snore", [0.0] * 19)
    await _make_channel_waveform(async_db_session, loud, "snore", [5.0])

    [period] = await StatsService(async_db_session, profile_id).get_period_statistics(
        "month"
    )

    # Averaging the two nights' 95ths would read 0.5 and 2.5.
    assert period.device_flg_95th == 1.0
    assert period.snore_95th == 5.0


async def test_period_excludes_disabled_sessions(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, enabled = await _make_day_session(async_db_session, device, _NIGHT)
    disabled = await _add_session(
        async_db_session,
        device,
        day,
        start=datetime(2025, 3, 2, 3, 0),
        enabled=False,
    )
    await _make_channel_waveform(async_db_session, enabled, "fl", [0.1])
    await _make_channel_waveform(async_db_session, disabled, "fl", [0.9])

    [period] = await StatsService(async_db_session, profile_id).get_period_statistics(
        "month"
    )

    assert period.device_flg_95th == pytest.approx(0.1)


async def test_period_with_only_disabled_sessions_keeps_null_channels(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    session.enabled = False
    await _make_channel_waveform(async_db_session, session, "fl", [0.9])

    [period] = await StatsService(async_db_session, profile_id).get_period_statistics(
        "month"
    )

    assert period.device_flg_95th is None
    assert period.snore_95th is None


async def test_period_bins_nights_on_their_side_of_a_month_boundary(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, last_of_jan = await _make_day_session(
        async_db_session, device, date(2025, 1, 31)
    )
    _, first_of_feb = await _make_day_session(
        async_db_session, device, date(2025, 2, 1)
    )
    await _make_channel_waveform(async_db_session, last_of_jan, "fl", [0.1])
    await _make_channel_waveform(async_db_session, first_of_feb, "fl", [0.9])

    january, february = await StatsService(
        async_db_session, profile_id
    ).get_period_statistics("month")

    assert january.period_start == date(2025, 1, 1)
    assert january.device_flg_95th == pytest.approx(0.1)
    assert february.period_start == date(2025, 2, 1)
    assert february.device_flg_95th == pytest.approx(0.9)


async def test_period_filtered_range_leaves_out_nights_outside_it(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, included = await _make_day_session(async_db_session, device, _NIGHT)
    _, excluded = await _make_day_session(async_db_session, device, _NEXT_NIGHT)
    await _make_channel_waveform(async_db_session, included, "fl", [0.1])
    await _make_channel_waveform(async_db_session, excluded, "fl", [0.9])

    [period] = await StatsService(async_db_session, profile_id).get_period_statistics(
        "month", from_date=_NIGHT, to_date=_NIGHT
    )

    assert period.device_flg_95th == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("period_type", "night", "next_night", "first_period", "second_period"),
    [
        (
            "day",
            date(2024, 12, 31),
            date(2025, 1, 1),
            date(2024, 12, 31),
            date(2025, 1, 1),
        ),
        # Sunday into Monday: weeks start on Monday, and 2024-12-31 does not cross one.
        (
            "week",
            date(2025, 3, 2),
            date(2025, 3, 3),
            date(2025, 2, 24),
            date(2025, 3, 3),
        ),
        (
            "month",
            date(2024, 12, 31),
            date(2025, 1, 1),
            date(2024, 12, 1),
            date(2025, 1, 1),
        ),
        (
            "6month",
            date(2024, 12, 31),
            date(2025, 1, 1),
            date(2024, 7, 1),
            date(2025, 1, 1),
        ),
        (
            "year",
            date(2024, 12, 31),
            date(2025, 1, 1),
            date(2024, 1, 1),
            date(2025, 1, 1),
        ),
    ],
)
async def test_noon_split_bins_a_session_by_its_day_not_its_start(
    async_db_session: AsyncSession,
    profile_device: tuple[int, Device],
    period_type: PeriodType,
    night: date,
    next_night: date,
    first_period: date,
    second_period: date,
) -> None:
    profile_id, device = profile_device
    # Starts at 03:00 on the next period's first date, but belongs to the night before.
    await _night_session(
        async_db_session,
        device,
        night,
        datetime.combine(next_night, time(3, 0)),
        fl=[0.9],
    )
    await _night_session(
        async_db_session,
        device,
        next_night,
        datetime.combine(next_night, time(22, 0)),
        fl=[0.1],
    )

    rows = await StatsService(async_db_session, profile_id).get_period_statistics(
        period_type
    )

    assert {row.period_start: row.device_flg_95th for row in rows} == {
        first_period: 0.9,
        second_period: 0.1,
    }


async def test_period_days_limit_leaves_out_older_nights(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    old = date.today() - timedelta(days=40)
    recent = date.today() - timedelta(days=2)
    await _night_session(
        async_db_session, device, old, datetime.combine(old, time(22, 0)), fl=[0.9]
    )
    await _night_session(
        async_db_session,
        device,
        recent,
        datetime.combine(recent, time(22, 0)),
        fl=[0.1],
    )

    rows = await StatsService(async_db_session, profile_id).get_period_statistics(
        "year", days_limit=30
    )

    assert [row.device_flg_95th for row in rows] == [0.1]


async def test_period_pools_every_device_and_day_detail_reads_one(
    async_db_session: AsyncSession,
) -> None:
    profile = await _make_profile(async_db_session)
    quiet = await _make_device(async_db_session, profile.id)
    loud = await _make_device(async_db_session, profile.id)
    start = datetime(2025, 3, 1, 22, 0)
    await _night_session(async_db_session, quiet, _NIGHT, start, fl=[0.0] * 19)
    await _night_session(async_db_session, loud, _NIGHT, start, fl=[1.0], snore=[2.0])

    [period] = await StatsService(async_db_session, profile.id).get_period_statistics(
        "month"
    )
    days = DayService(async_db_session, profile.id)
    quiet_day = await days.get_day(_NIGHT)
    loud_day = await days.get_day(_NIGHT, device_id=loud.id)

    assert period.device_flg_95th == 1.0
    assert period.snore_95th == 2.0
    assert quiet_day.channel_stats.device_flg_95th == 0.0
    assert quiet_day.channel_stats.snore_95th is None
    assert loud_day.channel_stats.device_flg_95th == 1.0
    assert loud_day.channel_stats.snore_95th == 2.0


async def test_experimental_stats_do_not_read_channel_blobs(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    await _make_day_session(async_db_session, device, _NIGHT)

    with _captured_sql(async_db_session) as captured:
        await StatsService(async_db_session, profile_id).get_experimental_stats()

    assert not _blob_queries(captured)


async def test_period_statistics_decode_blobs_once(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [0.2])

    with _captured_sql(async_db_session) as captured:
        await StatsService(async_db_session, profile_id).get_period_statistics("week")

    assert len(_blob_queries(captured)) == 1


async def test_period_statistics_can_skip_the_channels(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [0.2])

    with _captured_sql(async_db_session) as captured:
        [period] = await StatsService(
            async_db_session, profile_id
        ).get_period_statistics("month", include_channel_stats=False)

    assert not _blob_queries(captured)
    assert period.device_flg_95th is None
    assert period.days_used == 1


async def test_summary_report_does_not_read_blobs(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [0.2])

    with _captured_sql(async_db_session) as captured:
        await ReportService(async_db_session, profile_id).generate_summary_report(
            _NIGHT, _NIGHT
        )

    assert not _blob_queries(captured)


async def test_session_detail_can_skip_the_channels_and_keep_spans(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    session.mask_on_segments = [[0.0, 60.0]]
    await _make_channel_waveform(async_db_session, session, "fl", [0.2])

    with _captured_sql(async_db_session) as captured:
        detail = await SessionService(async_db_session, profile_id).get_session_detail(
            session.id, include_channel_stats=False
        )

    assert not _blob_queries(captured)
    assert detail.channel_stats is None
    assert [span.duration_seconds for span in detail.mask_on_spans] == [60.0]


async def test_loader_decodes_off_the_event_loop_thread(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    _, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [0.2])
    decode_threads: list[threading.Thread] = []

    def reduce(_channel: str, samples: np.ndarray) -> int:
        decode_threads.append(threading.current_thread())
        return samples.size

    values, corrupt = await decode_channel_blobs(
        async_db_session, [session.id], ["fl", "snore"], reduce
    )

    assert values == {"fl": {session.id: 1}, "snore": {}}
    assert corrupt == {"fl": set(), "snore": set()}
    assert decode_threads
    assert threading.current_thread() not in decode_threads


async def test_loader_fetches_blobs_in_batches(
    async_db_session: AsyncSession,
    profile_device: tuple[int, Device],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, device = profile_device
    monkeypatch.setattr("snore.database.waveform_blobs.DECODE_BATCH_SESSIONS", 2)
    session_ids = []
    for offset in range(5):
        _, session = await _make_day_session(
            async_db_session, device, _NIGHT + timedelta(days=offset)
        )
        await _make_channel_waveform(async_db_session, session, "fl", [0.1 * offset])
        session_ids.append(session.id)

    with _captured_sql(async_db_session) as captured:
        values, _ = await decode_channel_blobs(
            async_db_session, session_ids, ["fl"], lambda _channel, s: s.size
        )

    assert len(_blob_queries(captured)) == 3
    assert values["fl"] == dict.fromkeys(session_ids, 1)


async def test_session_detail_loads_only_the_reported_channels(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)

    with _captured_sql(async_db_session) as captured:
        await SessionService(async_db_session, profile_id).get_session_detail(
            session.id
        )

    [(statement, parameters)] = _blob_queries(captured)
    assert "waveform_type IN" in statement
    assert set(parameters) - {session.id} == _REPORTED_CHANNELS


async def test_day_pools_respiratory_and_pulse_channels(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, first = await _make_day_session(async_db_session, device, _NIGHT)
    second = await _add_session(
        async_db_session, device, day, start=datetime(2025, 3, 2, 3, 0)
    )
    await _make_channel_waveform(async_db_session, first, "rr", [12.0, 0.0, -1.0])
    await _make_channel_waveform(async_db_session, second, "rr", [16.0, 14.0, 20.0])
    await _make_channel_waveform(async_db_session, first, "tv", [450.0, 500.0])
    await _make_channel_waveform(async_db_session, second, "pulse", [-1.0, 60.0, 70.0])

    stats = (
        await DayService(async_db_session, profile_id).get_day(_NIGHT)
    ).channel_stats

    # rr pools [0, 12, 14, 16, 20]: the zero counts, the -1 sentinel does not.
    assert (
        stats.respiratory_rate_median,
        stats.respiratory_rate_95th,
        stats.respiratory_rate_max,
    ) == (14.0, 20.0, 20.0)
    assert (stats.tidal_volume_median, stats.tidal_volume_max) == (475.0, 500.0)
    assert (stats.pulse_median, stats.pulse_max) == (65.0, 70.0)
    assert stats.minute_ventilation_median is None
    assert stats.device_flg_median is None


async def test_period_statistics_decode_only_fl_and_snore(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [0.2])
    await _make_channel_waveform(async_db_session, session, "rr", [15.0])

    with _captured_sql(async_db_session) as captured:
        [period] = await StatsService(
            async_db_session, profile_id
        ).get_period_statistics("month")

    [(_, parameters)] = _blob_queries(captured)
    assert set(parameters) - {session.id} == {"fl", "snore"}
    assert period.device_flg_95th == pytest.approx(0.2)


async def test_list_endpoints_do_not_read_blobs(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [0.2])

    with _captured_sql(async_db_session) as captured:
        await DayService(async_db_session, profile_id).list_days()
        await SessionService(async_db_session, profile_id).list_sessions()
        await StatsService(async_db_session, profile_id).get_trends("week")

    assert not _blob_queries(captured)


async def test_rest_day_values_equal_mcp_nightly_values(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    """The REST day and the MCP nightly summary pool the same samples alike."""
    profile_id, device = profile_device
    await _seed_night_with_a_disabled_session(async_db_session, device)

    stats = (
        await DayService(async_db_session, profile_id).get_day(_NIGHT)
    ).channel_stats
    nightly = await BreathService(
        async_db_session, profile_id
    ).get_nightly_device_summary(_NIGHT, _NIGHT)

    [night] = nightly.nights
    assert stats.device_flg_median == round(night.device_flg_median, 4)
    assert stats.device_flg_95th == round(night.device_flg_95th, 4)
    assert stats.device_flg_max == round(night.device_flg_max, 4)
    assert stats.snore_median == round(night.snore_median, 4)
    assert stats.snore_95th == round(night.snore_95th, 4)


async def test_compare_epochs_values_equal_day_values(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    """``compare_epochs`` reads the same pooled ranks as the REST day.

    It shares ``pool_histograms`` and ``value_at_rank`` but builds its own
    distribution object, so this pins its median and 95th to the day's.
    """
    profile_id, device = profile_device
    await _seed_night_with_a_disabled_session(async_db_session, device)

    stats = (
        await DayService(async_db_session, profile_id).get_day(_NIGHT)
    ).channel_stats
    compared = await compare_epochs(
        db_session=async_db_session,
        profile_id=profile_id,
        epochs=[
            EpochSpec(
                label="night",
                date_start=_NIGHT.isoformat(),
                date_end=_NIGHT.isoformat(),
            )
        ],
    )

    [epoch] = compared.epochs
    assert stats.device_flg_median == round(epoch.device_flg.median, 4)
    assert stats.device_flg_95th == round(epoch.device_flg.p95, 4)
    assert stats.snore_median == round(epoch.snore_dist.median, 4)
    assert stats.snore_95th == round(epoch.snore_dist.p95, 4)
    # The floors agree: FL drops its two sentinels, snore keeps its negative sample.
    assert epoch.device_flg.n_samples == 11
    assert epoch.snore_dist.n_samples == 10


_RESPIRATORY_CHANNELS = {"rr", "tv", "mv", "ti", "ie_ratio"}
_AGREED_STATISTICS = {
    "respiratory_rate_mean": 15.0,
    "tidal_volume_mean": 0.5,
    "minute_ventilation_mean": 7.5,
    "ti_median": 1.0,
    "ie_ratio_median": 50.0,
}


async def _two_session_night(
    db: AsyncSession,
    device: Device,
    first_stats: dict[str, float | None],
    second_stats: dict[str, float | None],
) -> tuple[Session, Session]:
    """A night of two enabled sessions with the given Statistics columns."""
    day, first = await _make_day_session(db, device, _NIGHT)
    second = await _add_session(db, device, day, start=datetime(2025, 3, 2, 3, 0))
    db.add_all(
        [
            Statistics(session_id=first.id, **first_stats),
            Statistics(session_id=second.id, **second_stats),
        ]
    )
    await db.flush()
    return first, second


async def _nightly_row(db: AsyncSession, profile_id: int) -> NightlyRow:
    [night] = (await get_nightly_summary(db, _NIGHT, _NIGHT, profile_id)).nights
    return night


_SOURCE_FIELDS = (
    "rr_mean_source",
    "tv_mean_source",
    "mv_mean_source",
    "ti_median_source",
    "ie_ratio_source",
)


def _sources(night: NightlyRow) -> dict[str, str | None]:
    return {field: getattr(night, field) for field in _SOURCE_FIELDS}


def _respiratory_blob_queries(captured: list[tuple[str, Any]]) -> list[set[Any]]:
    """Parameters of the blob queries that read a respiratory channel."""
    return [
        set(params)
        for _, params in _blob_queries(captured)
        if _RESPIRATORY_CHANNELS & set(params)
    ]


async def test_nightly_summary_agreeing_sessions_use_the_stored_value(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    first, _ = await _two_session_night(
        async_db_session, device, _AGREED_STATISTICS, _AGREED_STATISTICS
    )
    await _make_channel_waveform(async_db_session, first, "rr", [30.0])

    with _captured_sql(async_db_session) as captured:
        night = await _nightly_row(async_db_session, profile_id)

    assert (
        night.rr_mean_bpm,
        night.tv_mean_ml,
        night.mv_mean_lpm,
        night.ti_median_s,
        night.ie_ratio_median,
    ) == (15.0, 500.0, 7.5, 1.0, 50.0)
    assert _sources(night) == dict.fromkeys(_SOURCE_FIELDS, "device")
    assert night.ti_median_reason is None
    assert night.ie_ratio_reason is None
    assert not _respiratory_blob_queries(captured)


async def test_nightly_summary_disagreeing_sessions_use_the_pooled_median(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    first, second = await _two_session_night(
        async_db_session,
        device,
        {**_AGREED_STATISTICS, "respiratory_rate_mean": 14.0, "ti_median": 1.0},
        {**_AGREED_STATISTICS, "respiratory_rate_mean": 16.0, "ti_median": 2.0},
    )
    await _make_channel_waveform(async_db_session, first, "rr", [10.0, 12.0])
    await _make_channel_waveform(async_db_session, second, "rr", [20.0])
    await _make_channel_waveform(async_db_session, first, "ti", [1.5])
    await _make_channel_waveform(async_db_session, second, "ti", [1.5, 2.0])

    night = await _nightly_row(async_db_session, profile_id)

    assert night.rr_mean_bpm == 12.0
    assert night.ti_median_s == 1.5
    assert night.ti_median_reason is None
    assert night.mv_mean_lpm == 7.5
    assert (night.rr_mean_source, night.ti_median_source) == ("derived", "derived")
    assert night.mv_mean_source == "device"


async def test_nightly_summary_one_null_session_uses_the_pooled_median(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    first, second = await _two_session_night(
        async_db_session,
        device,
        _AGREED_STATISTICS,
        {**_AGREED_STATISTICS, "minute_ventilation_mean": None},
    )
    await _make_channel_waveform(async_db_session, first, "mv", [6.0])
    await _make_channel_waveform(async_db_session, second, "mv", [9.0, 10.0])

    night = await _nightly_row(async_db_session, profile_id)

    assert night.mv_mean_lpm == 9.0


async def test_nightly_summary_disagreeing_sessions_without_waveforms_are_null(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    await _two_session_night(
        async_db_session,
        device,
        {"respiratory_rate_mean": 14.0, "ti_median": 1.0, "ie_ratio_median": 40.0},
        {"respiratory_rate_mean": 16.0, "ti_median": 2.0, "ie_ratio_median": 50.0},
    )

    night = await _nightly_row(async_db_session, profile_id)

    assert night.rr_mean_bpm is None
    assert night.ti_median_s is None
    assert night.ti_median_reason == "sessions_disagree"
    assert night.ie_ratio_median is None
    assert night.ie_ratio_reason == "sessions_disagree"
    assert _sources(night) == dict.fromkeys(_SOURCE_FIELDS)


async def test_nightly_summary_without_stored_values_or_waveforms_is_channel_absent(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    await _two_session_night(async_db_session, device, {}, {})

    night = await _nightly_row(async_db_session, profile_id)

    assert (night.ti_median_s, night.ti_median_reason) == (None, "channel_absent")
    assert (night.ie_ratio_median, night.ie_ratio_reason) == (None, "channel_absent")


async def test_nightly_summary_corrupt_fallback_waveform_nulls_the_value(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    first, second = await _two_session_night(
        async_db_session,
        device,
        {**_AGREED_STATISTICS, "ti_median": 1.0},
        {**_AGREED_STATISTICS, "ti_median": 2.0},
    )
    await _make_channel_waveform(async_db_session, first, "ti", [1.5])
    corrupt = await _make_channel_waveform(async_db_session, second, "ti", [1.2])
    corrupt.data_blob = b"not a whole sample"
    await async_db_session.flush()

    night = await _nightly_row(async_db_session, profile_id)

    assert (night.ti_median_s, night.ti_median_source) == (None, None)
    assert night.ti_median_reason == "channel_corrupt"


async def test_nightly_summary_night_of_disabled_sessions_is_no_sessions(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, session = await _make_day_session(async_db_session, device, _NIGHT)
    session.enabled = False
    async_db_session.add(Statistics(session_id=session.id, **_AGREED_STATISTICS))
    await _make_channel_waveform(async_db_session, session, "rr", [15.0])
    await async_db_session.flush()

    night = await _nightly_row(async_db_session, profile_id)

    assert (night.rr_mean_bpm, night.ti_median_s, night.ie_ratio_median) == (
        None,
        None,
        None,
    )
    assert _sources(night) == dict.fromkeys(_SOURCE_FIELDS)
    assert night.ti_median_reason == "no_sessions"
    assert night.ie_ratio_reason == "no_sessions"


async def test_nightly_summary_disabled_session_is_left_out_of_agreement_and_pooling(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, enabled = await _make_day_session(async_db_session, device, _NIGHT)
    disabled = await _add_session(
        async_db_session, device, day, start=datetime(2025, 3, 2, 3, 0), enabled=False
    )
    async_db_session.add_all(
        [
            Statistics(
                session_id=enabled.id,
                **{**_AGREED_STATISTICS, "minute_ventilation_mean": None},
            ),
            Statistics(
                session_id=disabled.id,
                **{
                    **_AGREED_STATISTICS,
                    "respiratory_rate_mean": 20.0,
                    "minute_ventilation_mean": 9.0,
                },
            ),
        ]
    )
    await _make_channel_waveform(async_db_session, enabled, "mv", [6.0])
    await _make_channel_waveform(async_db_session, disabled, "mv", [20.0, 30.0])
    await _make_channel_waveform(async_db_session, disabled, "rr", [40.0])
    await async_db_session.flush()

    night = await _nightly_row(async_db_session, profile_id)

    assert (night.rr_mean_bpm, night.rr_mean_source) == (15.0, "device")
    assert (night.mv_mean_lpm, night.mv_mean_source) == (6.0, "derived")


async def test_nightly_summary_disagreeing_tidal_volume_pools_the_waveform_in_ml(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    first, second = await _two_session_night(
        async_db_session,
        device,
        {**_AGREED_STATISTICS, "tidal_volume_mean": 0.45},
        {**_AGREED_STATISTICS, "tidal_volume_mean": 0.55},
    )
    await _make_channel_waveform(async_db_session, first, "tv", [450.0, 500.0])
    await _make_channel_waveform(async_db_session, second, "tv", [520.0])

    night = await _nightly_row(async_db_session, profile_id)

    assert (night.tv_mean_ml, night.tv_mean_source) == (500.0, "derived")


async def test_nightly_summary_pools_only_the_sessions_with_a_waveform(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    first, _ = await _two_session_night(
        async_db_session,
        device,
        {**_AGREED_STATISTICS, "respiratory_rate_mean": 14.0},
        {**_AGREED_STATISTICS, "respiratory_rate_mean": 16.0},
    )
    await _make_channel_waveform(async_db_session, first, "rr", [10.0, 12.0])

    night = await _nightly_row(async_db_session, profile_id)

    assert (night.rr_mean_bpm, night.rr_mean_source) == (11.0, "derived")


async def test_nightly_summary_session_without_statistics_forces_the_fallback(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, summarized = await _make_day_session(async_db_session, device, _NIGHT)
    unsummarized = await _add_session(
        async_db_session, device, day, start=datetime(2025, 3, 2, 3, 0)
    )
    async_db_session.add(Statistics(session_id=summarized.id, **_AGREED_STATISTICS))
    await _make_channel_waveform(async_db_session, summarized, "rr", [18.0])
    await _make_channel_waveform(async_db_session, unsummarized, "rr", [20.0])
    await async_db_session.flush()

    night = await _nightly_row(async_db_session, profile_id)

    assert (night.rr_mean_bpm, night.rr_mean_source) == (19.0, "derived")
    assert night.ti_median_reason == "sessions_disagree"


async def test_nightly_fallback_median_equals_night_detail_channel_median(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    first, second = await _two_session_night(
        async_db_session,
        device,
        {**_AGREED_STATISTICS, "ie_ratio_median": 40.0},
        {**_AGREED_STATISTICS, "ie_ratio_median": 50.0},
    )
    await _make_channel_waveform(async_db_session, first, "ie_ratio", [41.3, 47.1])
    await _make_channel_waveform(async_db_session, second, "ie_ratio", [52.9, 44.4])

    night = await _nightly_row(async_db_session, profile_id)
    detail = await get_night_detail(async_db_session, _NIGHT, profile_id)

    assert night.ie_ratio_source == "derived"
    assert night.ie_ratio_median == detail.device.channel_stats.ie_ratio_median


async def test_nightly_summary_compares_tidal_volume_in_ml(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    resmed, oscar = await _two_session_night(
        async_db_session,
        device,
        {**_AGREED_STATISTICS, "tidal_volume_mean": 0.45},
        {**_AGREED_STATISTICS, "tidal_volume_mean": 450.0},
    )
    resmed.import_source = "resmed_edf"
    oscar.import_source = "oscar_binary"
    await async_db_session.flush()

    with _captured_sql(async_db_session) as captured:
        night = await _nightly_row(async_db_session, profile_id)

    assert night.tv_mean_ml == 450.0
    assert not _respiratory_blob_queries(captured)


async def test_nightly_summary_decodes_only_the_nights_that_fall_back(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, agreed = await _make_day_session(async_db_session, device, _NIGHT)
    async_db_session.add(Statistics(session_id=agreed.id, **_AGREED_STATISTICS))
    _, unsummarized = await _make_day_session(async_db_session, device, _NEXT_NIGHT)
    await _make_channel_waveform(async_db_session, agreed, "rr", [30.0])
    await _make_channel_waveform(async_db_session, unsummarized, "rr", [18.0])
    await async_db_session.flush()

    with _captured_sql(async_db_session) as captured:
        result = await get_nightly_summary(
            async_db_session, _NIGHT, _NEXT_NIGHT, profile_id
        )

    assert {night.date: night.rr_mean_bpm for night in result.nights} == {
        _NIGHT: 15.0,
        _NEXT_NIGHT: 18.0,
    }
    queries = _respiratory_blob_queries(captured)
    assert queries
    assert all(agreed.id not in params for params in queries)


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


def _blob_queries(captured: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    return [(sql, params) for sql, params in captured if "data_blob" in sql]
