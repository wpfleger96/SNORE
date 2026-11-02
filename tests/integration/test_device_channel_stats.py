"""FL and snore statistics and mask-on spans on the session, day and period surfaces."""

from __future__ import annotations

import logging

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Day, Device, Session, Statistics
from snore.services.breath_service import BreathService
from snore.services.day_service import DayService
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

    with caplog.at_level(logging.WARNING, logger="snore.services.channel_stats"):
        detail = await SessionService(async_db_session, profile_id).get_session_detail(
            session.id
        )

    assert detail.channel_stats.device_flg_max is None
    assert detail.channel_stats.snore_max == 2.0
    assert f"session {session.id}" in caplog.text


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
    assert result.mask_on_spans == []
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


async def test_day_spans_ascend_and_skip_sessions_with_unknown_segments(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    day, first = await _make_day_session(async_db_session, device, _NIGHT)
    first.mask_on_segments = [[0.0, 600.0], [1200.0, 1800.0]]
    await _add_session(
        async_db_session,
        device,
        day,
        start=datetime(2025, 3, 2, 3, 0),
        mask_on_segments=[[0.0, 60.0]],
    )
    await _add_session(async_db_session, device, day, start=datetime(2025, 3, 2, 4, 0))

    spans = (
        await DayService(async_db_session, profile_id).get_day(_NIGHT)
    ).mask_on_spans

    assert [s.start for s in spans] == [
        datetime(2025, 3, 1, 22, 0),
        datetime(2025, 3, 1, 22, 20),
        datetime(2025, 3, 2, 3, 0),
    ]


async def test_day_without_known_segments_has_empty_spans(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    await _make_day_session(async_db_session, device, _NIGHT)

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


async def test_trends_series_align_with_period_starts(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, with_channels = await _make_day_session(
        async_db_session, device, date(2025, 1, 31)
    )
    await _make_day_session(async_db_session, device, date(2025, 2, 1))
    await _make_channel_waveform(async_db_session, with_channels, "fl", [0.2])
    await _make_channel_waveform(async_db_session, with_channels, "snore", [1.5])

    trends = await StatsService(async_db_session, profile_id).get_trends("month")

    assert trends.device_flg_95th == [
        (date(2025, 1, 1), pytest.approx(0.2)),
        (date(2025, 2, 1), None),
    ]
    assert trends.snore_95th == [(date(2025, 1, 1), 1.5), (date(2025, 2, 1), None)]
    assert [d for d, _ in trends.ahi] == [d for d, _ in trends.device_flg_95th]


async def test_experimental_stats_do_not_read_channel_blobs(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    await _make_day_session(async_db_session, device, _NIGHT)

    with _captured_sql(async_db_session) as captured:
        await StatsService(async_db_session, profile_id).get_experimental_stats()

    assert not _blob_queries(captured)


async def test_trends_decode_blobs_once(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [0.2])

    with _captured_sql(async_db_session) as captured:
        await StatsService(async_db_session, profile_id).get_trends("week")

    assert len(_blob_queries(captured)) == 1


async def test_session_detail_loads_only_the_two_channels(
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
    assert {"fl", "snore"} <= set(parameters)
    assert "flow" not in parameters


async def test_list_endpoints_do_not_read_blobs(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    await _make_day_session(async_db_session, device, _NIGHT)

    with _captured_sql(async_db_session) as captured:
        await DayService(async_db_session, profile_id).list_days()
        await SessionService(async_db_session, profile_id).list_sessions()

    assert not _blob_queries(captured)


async def test_rest_day_values_equal_mcp_nightly_values(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    """The two implementations of the same statistics must not drift."""
    profile_id, device = profile_device
    day, first = await _make_day_session(async_db_session, device, _NIGHT)
    second = await _add_session(
        async_db_session, device, day, start=datetime(2025, 3, 2, 3, 0)
    )
    disabled = await _add_session(
        async_db_session,
        device,
        day,
        start=datetime(2025, 3, 2, 5, 0),
        enabled=False,
    )
    await _make_channel_waveform(
        async_db_session, first, "fl", [-0.01, 0.0, 0.13, 0.27, 0.0, 0.61, 0.07]
    )
    await _make_channel_waveform(
        async_db_session, second, "fl", [0.33, -0.01, 0.41, 0.19, 0.0, 0.92]
    )
    await _make_channel_waveform(async_db_session, disabled, "fl", [1.0])
    await _make_channel_waveform(
        async_db_session, first, "snore", [0.0, 0.2, 1.7, -0.01, 3.3, 0.9]
    )
    await _make_channel_waveform(
        async_db_session, second, "snore", [0.4, 0.0, 2.2, 4.8]
    )

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
