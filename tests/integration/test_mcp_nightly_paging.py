"""Integration tests: what one page of get_nightly_summary and one experimental day read.

- a page's channel statistics equal the unpaged response's, and only that
  page's sessions are decoded
- compliance covers the whole range on every page, including past the end
- the experimental day metrics read the ``days`` table once
"""

from __future__ import annotations

import re

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any

import pytest

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Day
from snore.mcp.tools.summary import get_nightly_summary
from snore.services.day_service import DayService
from tests.integration.conftest import (
    _make_analysis_result,
    _make_channel_waveform,
    _make_day_session,
    _make_device,
)

_START = date(2025, 10, 1)
_N_NIGHTS = 5
_END = _START + timedelta(days=_N_NIGHTS + 1)  # two trailing nights without data


async def _seed_nights(db: AsyncSession, profile_id: int) -> dict[date, int]:
    """Five nights with distinct channel samples; every other night is compliant."""
    device = await _make_device(db, profile_id)
    session_by_date: dict[date, int] = {}
    for offset in range(_N_NIGHTS):
        night = _START + timedelta(days=offset)
        _, session = await _make_day_session(
            db, device, night, duration_hours=7.0 if offset % 2 == 0 else 2.0
        )
        await _make_channel_waveform(db, session, "fl", [-0.01, 0.1 * offset, 0.5])
        await _make_channel_waveform(db, session, "snore", [0.0, 1.0 + offset])
        await _make_channel_waveform(db, session, "leak", [30.0, 0.0, 5.0 * offset])
        session_by_date[night] = session.id
    return session_by_date


@contextmanager
def _recorded_decodes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Session ids the nightly service hands to the blob decoder."""
    from snore.services.breath import nightly  # noqa: PLC0415

    decoded: list[int] = []
    real_decode = nightly.decode_channel_blobs

    async def recording(
        db: AsyncSession, session_ids: Sequence[int], *args: Any
    ) -> Any:
        decoded.extend(session_ids)
        return await real_decode(db, session_ids, *args)

    monkeypatch.setattr(nightly, "decode_channel_blobs", recording)
    yield decoded


async def _summary(db: AsyncSession, profile_id: int, **kwargs: Any) -> Any:
    return await get_nightly_summary(
        db_session=db, start=_START, end=_END, profile_id=profile_id, **kwargs
    )


@pytest.mark.integration
class TestNightlySummaryPaging:
    async def test_page_two_matches_unpaged_rows_and_decodes_only_its_sessions(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_by_date = await _seed_nights(async_db_session, async_test_profile.id)
        unpaged = await _summary(async_db_session, async_test_profile.id, page_size=90)

        with _recorded_decodes(monkeypatch) as decoded:
            page_two = await _summary(
                async_db_session, async_test_profile.id, page=2, page_size=2
            )

        assert [n.model_dump() for n in page_two.nights] == [
            n.model_dump() for n in unpaged.nights[2:4]
        ]
        assert page_two.nights[0].device_flg_max == 0.5
        assert sorted(decoded) == sorted(
            session_by_date[n.date] for n in page_two.nights
        )

    async def test_compliance_is_full_range_on_every_page(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _seed_nights(async_db_session, async_test_profile.id)
        unpaged = await _summary(async_db_session, async_test_profile.id, page_size=90)

        with _recorded_decodes(monkeypatch) as decoded:
            pages = [
                await _summary(
                    async_db_session, async_test_profile.id, page=page, page_size=2
                )
                for page in (1, 2, 3, 4)
            ]

        assert unpaged.compliance is not None
        assert unpaged.compliance.days_compliant == 3
        assert unpaged.compliance.days_total == _N_NIGHTS + 2
        assert [page.compliance for page in pages] == [unpaged.compliance] * 4
        assert [page.total_nights for page in pages] == [_N_NIGHTS] * 4
        assert pages[3].nights == []
        # Pages 1-3 hold the five nights; the empty fourth page decodes nothing.
        assert len(decoded) == _N_NIGHTS


@pytest.mark.integration
async def test_experimental_day_metrics_read_days_once(
    async_db_session: AsyncSession, async_test_profile: Any
) -> None:
    device = await _make_device(async_db_session, async_test_profile.id)
    _, session = await _make_day_session(async_db_session, device, _START)
    await _make_analysis_result(async_db_session, session)
    statements: list[str] = []

    def capture(_conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
        statements.append(statement)

    engine = async_db_session.get_bind()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        metrics = await DayService(
            async_db_session, async_test_profile.id
        ).get_day_experimental(_START)
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    # The OK analysis was read: a uniform-mode night counts its (zero) RERAs.
    assert metrics.rera_count == 0
    assert metrics.rera_count_reason is None
    days_table = re.compile(rf"\b{Day.__tablename__}\b", re.IGNORECASE)
    assert len([s for s in statements if days_table.search(s)]) == 1
