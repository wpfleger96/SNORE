"""Time over the leak redline and Large Leak spans on the session, day and MCP
night surfaces."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Day, Device, Session, Statistics, Waveform
from snore.mcp.tools.night_detail import get_night_detail
from snore.mcp.tools.summary import get_nightly_summary
from snore.services.day_service import DayService
from snore.services.schemas import MAX_LARGE_LEAK_SPANS
from snore.services.session_service import SessionService
from tests.integration.conftest import (
    _make_channel_waveform,
    _make_day_session,
    _make_device,
    _make_profile,
)

pytestmark = pytest.mark.integration

_NIGHT = date(2025, 3, 1)
_FIRST_START = datetime(2025, 3, 1, 22, 0)
_SECOND_START = datetime(2025, 3, 2, 3, 0)


@pytest.fixture
async def profile_device(async_db_session: AsyncSession) -> tuple[int, Device]:
    profile = await _make_profile(async_db_session)
    return profile.id, await _make_device(async_db_session, profile.id)


async def _add_leak(
    db: AsyncSession, session: Session, pairs: list[tuple[float, float]]
) -> None:
    """Store ``(offset, L/min)`` leak samples on ``session``."""
    db.add(
        Waveform(
            session_id=session.id,
            waveform_type="leak",
            sample_rate=0.5,
            data_blob=np.array(pairs, dtype=np.float32).tobytes(),
            sample_count=len(pairs),
        )
    )
    await db.flush()


async def _two_session_night(
    db: AsyncSession, device: Device
) -> tuple[Day, Session, Session]:
    """Two enabled one-hour sessions on one 2-hour night."""
    day, first = await _make_day_session(db, device, _NIGHT, duration_hours=1.0)
    day.total_therapy_hours = 2.0
    second = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id="second",
        start_time=_SECOND_START,
        end_time=_SECOND_START + timedelta(hours=1),
        duration_seconds=3600.0,
        enabled=True,
    )
    db.add(second)
    await db.flush()
    return day, first, second


async def test_day_leak_above_24_pct_equals_nightly_summary_value(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first, second = await _two_session_night(async_db_session, device)
    await _make_channel_waveform(async_db_session, first, "leak", [-1.0, 10.0, 24.0])
    await _make_channel_waveform(
        async_db_session, second, "leak", [30.0, 5.0, 26.0, 0.0]
    )

    day = await DayService(async_db_session, profile_id).get_day(_NIGHT)
    [nightly] = (
        await get_nightly_summary(async_db_session, _NIGHT, _NIGHT, profile_id)
    ).nights

    # 3 of the 6 valid pooled samples reach 24 L/min; the -1 sentinel is no sample.
    assert day.leak_redline is not None
    assert day.leak_redline.leak_above_24_pct == nightly.leak_above_24_pct == 50.0


async def test_multi_session_night_lists_spans_on_the_wall_clock(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first, second = await _two_session_night(async_db_session, device)
    # Two mask-on segments in the first session; leak stays high across the gap.
    await _add_leak(
        async_db_session,
        first,
        [(0.0, 10.0), (2.0, 30.0), (4.0, 30.0), (600.0, 30.0), (602.0, 10.0)],
    )
    # Still above the redline when the recording ends.
    await _add_leak(async_db_session, second, [(0.0, 10.0), (2.0, 40.0), (4.0, 40.0)])

    leak = (await DayService(async_db_session, profile_id).get_day(_NIGHT)).leak_redline

    assert leak is not None
    assert [
        (span.session_id, span.offset_seconds, span.start, span.end)
        for span in leak.large_leak_spans
    ] == [
        # Open at the gap: ends one sample period after the segment's last sample.
        (
            first.id,
            2.0,
            _FIRST_START + timedelta(seconds=2),
            _FIRST_START + timedelta(seconds=6),
        ),
        (
            first.id,
            600.0,
            _FIRST_START + timedelta(seconds=600),
            _FIRST_START + timedelta(seconds=602),
        ),
        # Open at the end of the recording.
        (
            second.id,
            2.0,
            _SECOND_START + timedelta(seconds=2),
            _SECOND_START + timedelta(seconds=6),
        ),
    ]
    # 10 seconds of Large Leak over the night's 2 mask-on hours.
    assert leak.large_leak_span_count == 3
    assert leak.large_leak_pct == round(100 * 10 / 7200, 2)


async def test_session_large_leak_pct_uses_its_mask_on_hours(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(
        async_db_session, device, _NIGHT, duration_hours=0.01
    )
    await _make_channel_waveform(async_db_session, session, "leak", [30.0] * 10 + [0.0])

    leak = (
        await SessionService(async_db_session, profile_id).get_session_detail(
            session.id
        )
    ).leak_redline

    # A 20 s span over a 36 s session.
    assert leak is not None
    assert leak.large_leak_pct == round(100 * 20 / 36, 2)
    assert leak.leak_above_24_pct == round(100 * 10 / 11, 1)


async def test_session_large_leak_pct_prefers_stored_usage_hours(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    # One hour recorded, 72 s of it mask-on by the device's count.
    async_db_session.add(Statistics(session_id=session.id, usage_hours=0.02))
    await _make_channel_waveform(async_db_session, session, "leak", [30.0, 30.0, 0.0])

    leak = (
        await SessionService(async_db_session, profile_id).get_session_detail(
            session.id
        )
    ).leak_redline

    assert leak is not None
    assert leak.large_leak_pct == round(100 * 4 / 72, 2)


async def test_session_without_leak_is_left_out_of_the_day_index(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first, _second = await _two_session_night(async_db_session, device)
    await _make_channel_waveform(async_db_session, first, "leak", [30.0, 0.0])

    leak = (await DayService(async_db_session, profile_id).get_day(_NIGHT)).leak_redline

    # 2 s over the first session's hour, not the night's two hours.
    assert leak is not None
    assert leak.large_leak_pct == round(100 * 2 / 3600, 2)


async def test_all_invalid_leak_samples_give_no_leak_redline(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(
        async_db_session, session, "leak", [-1.0, float("nan"), -1.0]
    )

    day = await DayService(async_db_session, profile_id).get_day(_NIGHT)

    assert day.leak_redline is None


@pytest.mark.parametrize(
    "pairs",
    [
        [(0.0, 30.0), (float("nan"), 30.0), (4.0, 0.0)],
        [(0.0, 30.0), (2.0, 30.0), (2.0, 30.0)],
        [(0.0, 30.0), (4.0, 30.0), (2.0, 0.0)],
        # Finite, but past the end of the wall clock.
        [(0.0, 10.0), (1e30, 30.0)],
    ],
)
async def test_unusable_sample_times_null_the_leak_redline(
    async_db_session: AsyncSession,
    profile_device: tuple[int, Device],
    pairs: list[tuple[float, float]],
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _add_leak(async_db_session, session, pairs)

    day = await DayService(async_db_session, profile_id).get_day(_NIGHT)
    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    assert day.leak_redline is None
    assert detail.leak_redline is None


async def test_long_span_lists_keep_the_longest_in_time_order(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    short = MAX_LARGE_LEAK_SPANS + 100
    # Many 2 s spans, then three longer ones: 10 s, 6 s and 8 s.
    values = [30.0, 0.0] * short
    values += [30.0] * 5 + [0.0] + [30.0] * 3 + [0.0] + [30.0] * 4 + [0.0]
    await _make_channel_waveform(async_db_session, session, "leak", values)

    leak = (
        await SessionService(async_db_session, profile_id).get_session_detail(
            session.id
        )
    ).leak_redline

    assert leak is not None
    assert leak.large_leak_span_count == short + 3
    spans = leak.large_leak_spans
    assert len(spans) == MAX_LARGE_LEAK_SPANS
    assert [span.duration_seconds for span in spans[-3:]] == [10.0, 6.0, 8.0]
    assert spans == sorted(spans, key=lambda span: span.start)
    # The index counts every span, listed or not.
    total_seconds = 2 * short + 24
    assert leak.large_leak_pct == round(100 * total_seconds / (8 * 3600), 2)


async def test_night_without_leak_channel_has_no_leak_redline(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "fl", [0.2])

    day = await DayService(async_db_session, profile_id).get_day(_NIGHT)
    detail = await SessionService(async_db_session, profile_id).get_session_detail(
        session.id
    )

    assert day.leak_redline is None
    assert detail.leak_redline is None


async def test_night_below_redline_has_zero_not_null(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "leak", [4.0, 8.0])

    leak = (await DayService(async_db_session, profile_id).get_day(_NIGHT)).leak_redline

    assert leak is not None
    assert (leak.leak_above_24_pct, leak.large_leak_pct, leak.large_leak_spans) == (
        0.0,
        0.0,
        [],
    )


async def test_corrupt_leak_blob_nulls_the_night(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first, second = await _two_session_night(async_db_session, device)
    await _make_channel_waveform(async_db_session, first, "leak", [30.0, 0.0])
    corrupt = await _make_channel_waveform(async_db_session, second, "leak", [30.0])
    corrupt.data_blob = b"\x00\x01"
    await async_db_session.flush()

    day = await DayService(async_db_session, profile_id).get_day(_NIGHT)
    [nightly] = (
        await get_nightly_summary(async_db_session, _NIGHT, _NIGHT, profile_id)
    ).nights

    assert day.leak_redline is None
    assert nightly.leak_above_24_pct is None


async def test_disabled_session_leak_is_left_out_of_the_day(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, first, second = await _two_session_night(async_db_session, device)
    second.enabled = False
    await _make_channel_waveform(async_db_session, first, "leak", [10.0, 10.0])
    await _make_channel_waveform(async_db_session, second, "leak", [30.0, 30.0])

    leak = (await DayService(async_db_session, profile_id).get_day(_NIGHT)).leak_redline

    assert leak is not None
    assert (leak.leak_above_24_pct, leak.large_leak_spans) == (0.0, [])


async def test_get_night_detail_carries_the_day_leak_redline(
    async_db_session: AsyncSession, profile_device: tuple[int, Device]
) -> None:
    profile_id, device = profile_device
    _, session = await _make_day_session(async_db_session, device, _NIGHT)
    await _make_channel_waveform(async_db_session, session, "leak", [10.0, 30.0, 10.0])

    result = await get_night_detail(async_db_session, _NIGHT, profile_id)

    assert result.device is not None
    leak = result.device.leak_redline
    assert leak is not None
    assert leak.leak_above_24_pct == 33.3
    assert [
        (span.session_id, span.duration_seconds) for span in leak.large_leak_spans
    ] == [(session.id, 2.0)]
