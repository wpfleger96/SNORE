"""Integration tests: compare_epochs and experimental_compare_epochs over overlapping
epochs and growing session counts.

- overlapping epochs report the same per-epoch stats as comparing each alone
- the default tool decodes the waveforms of overlapping epochs in one query
- the experimental tool reads the breaths of overlapping epochs in one query
- the experimental tool's query count does not grow with the number of sessions
- a failure while streaming breaths still releases the breath cursor
- the service refuses an epoch longer than MAX_EPOCH_NIGHTS
"""

from __future__ import annotations

import re
import sqlite3

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Breath, Profile, Setting, Waveform
from snore.services.breath.dtos import MAX_EPOCH_NIGHTS, EpochRequest
from tests.integration.conftest import (
    _make_analysis_result,
    _make_day_session,
    _make_device,
    _make_profile,
)
from tests.integration.test_mcp_compare_epochs import _make_breath
from tests.integration.test_mcp_device_waveform_aggregates import (
    _add_waveform,
    _epoch,
)

_FIRST_NIGHT = date(2025, 3, 1)


@contextmanager
def _captured_statements(db: AsyncSession) -> Iterator[list[str]]:
    statements: list[str] = []

    def capture(_conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
        statements.append(statement)

    engine = db.get_bind()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", capture)


def _nights(n: int) -> list[date]:
    return [_FIRST_NIGHT + timedelta(days=k) for k in range(n)]


async def _seed_device_nights(db: AsyncSession, profile_id: int, n: int) -> None:
    """One session per night with distinct FL (sentinel included) and snore samples."""
    device = await _make_device(db, profile_id)
    for k, night in enumerate(_nights(n)):
        _, sess = await _make_day_session(db, device, night)
        await _add_waveform(db, sess.id, "fl", [-0.01, 0.1 * k, 0.05 + 0.1 * k, 0.3])
        await _add_waveform(db, sess.id, "snore", [0.0, float(k), 0.5])


async def _seed_analyzed_nights(db: AsyncSession, profile_id: int, n: int) -> None:
    """One analyzed session per night; each night scores one RERA-proxy run.

    Breaths 1-2 are a flow-limited run (breath 2 leak-invalid, so it counts for
    the RERA proxy but not the distributions) and breath 3 recovers.
    """
    device = await _make_device(db, profile_id)
    for k, night in enumerate(_nights(n)):
        _, sess = await _make_day_session(db, device, night)
        db.add(Setting(session_id=sess.id, key="pressure_min", value="8.0"))
        ar = await _make_analysis_result(db, sess)
        first = await _make_breath(
            db,
            ar,
            sess,
            1,
            mid_insp_flattening=0.1 * k + 0.05,
            flatness_index=0.2 + 0.01 * k,
            tidal_volume_ml=400.0 + k,
            i_e_ratio=0.4 + 0.02 * k,
            flow_class=4,
        )
        first.flow_confidence = 0.8
        await _make_breath(db, ar, sess, 2, leak_valid=False, flow_class=5)
        recovery = await _make_breath(
            db,
            ar,
            sess,
            3,
            mid_insp_flattening=0.9,
            flow_class=1,
            is_recovery_breath=True,
        )
        recovery.flow_confidence = 0.5
    await db.flush()


CompareFn = Callable[..., Awaitable[Any]]


def _compare_fn(tool: str) -> CompareFn:
    from snore.mcp.tools import epochs  # noqa: PLC0415

    fn: CompareFn = getattr(epochs, tool)
    return fn


@pytest.mark.integration
@pytest.mark.parametrize(
    ("tool", "seed"),
    [
        pytest.param("compare_epochs", _seed_device_nights, id="default"),
        pytest.param(
            "experimental_compare_epochs", _seed_analyzed_nights, id="experimental"
        ),
    ],
)
class TestOverlappingEpochs:
    async def test_overlapping_epochs_match_comparing_each_alone(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Profile,
        tool: str,
        seed: Callable[[AsyncSession, int, int], Awaitable[None]],
    ) -> None:
        await seed(async_db_session, async_test_profile.id, 4)
        nights = _nights(4)
        epochs = [
            _epoch("early", nights[0], nights[2]),
            _epoch("late", nights[1], nights[3]),
            _epoch("all", nights[0], nights[3]),
        ]
        compare = _compare_fn(tool)

        together = await compare(
            async_db_session, profile_id=async_test_profile.id, epochs=epochs
        )
        alone = [
            (
                await compare(
                    async_db_session, profile_id=async_test_profile.id, epochs=[e]
                )
            ).epochs[0]
            for e in epochs
        ]

        assert all(e.null_reason is None for e in alone)
        assert together.epochs == alone


@pytest.mark.integration
class TestCompareEpochsQueryCost:
    async def test_overlapping_device_epochs_decode_waveforms_in_one_query(
        self, async_db_session: AsyncSession, async_test_profile: Profile
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        await _seed_device_nights(async_db_session, async_test_profile.id, 4)
        nights = _nights(4)
        waveforms_table = re.compile(rf"\bFROM {Waveform.__tablename__}\b", re.I)

        with _captured_statements(async_db_session) as statements:
            result = await compare_epochs(
                async_db_session,
                profile_id=async_test_profile.id,
                epochs=[
                    _epoch("early", nights[0], nights[2]),
                    _epoch("late", nights[1], nights[3]),
                ],
            )

        assert [e.device_flg.n_samples for e in result.epochs] == [9, 9]
        assert len([s for s in statements if waveforms_table.search(s)]) == 1

    async def test_overlapping_experimental_epochs_read_breaths_in_one_query(
        self, async_db_session: AsyncSession, async_test_profile: Profile
    ) -> None:
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        await _seed_analyzed_nights(async_db_session, async_test_profile.id, 4)
        nights = _nights(4)
        breaths_table = re.compile(rf"\bFROM {Breath.__tablename__}\b", re.I)

        with _captured_statements(async_db_session) as statements:
            result = await experimental_compare_epochs(
                async_db_session,
                profile_id=async_test_profile.id,
                epochs=[
                    _epoch("early", nights[0], nights[2]),
                    _epoch("late", nights[1], nights[3]),
                ],
            )

        assert [e.rera_proxy_count for e in result.epochs] == [3, 3]
        assert len([s for s in statements if breaths_table.search(s)]) == 1

    async def test_experimental_query_count_is_independent_of_session_count(
        self, async_db_session: AsyncSession
    ) -> None:
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        async def statements_for(n_nights: int) -> int:
            profile = await _make_profile(async_db_session)
            await _seed_analyzed_nights(async_db_session, profile.id, n_nights)
            with _captured_statements(async_db_session) as statements:
                result = await experimental_compare_epochs(
                    async_db_session,
                    profile_id=profile.id,
                    epochs=[_epoch("all", _FIRST_NIGHT, _nights(n_nights)[-1])],
                )
            epoch = result.epochs[0]
            assert epoch.nights_with_data == n_nights
            assert epoch.rera_proxy_count == n_nights
            return len(statements)

        assert await statements_for(2) == await statements_for(8)


@pytest.mark.integration
class TestBreathStreamRelease:
    async def test_failure_mid_stream_releases_the_breath_cursor(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Profile,
        temp_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An open breath cursor holds a read lock that blocks other writers."""
        from snore.services.breath import epochs as epochs_module  # noqa: PLC0415
        from snore.services.breath_service import BreathService  # noqa: PLC0415

        await _seed_analyzed_nights(async_db_session, async_test_profile.id, 3)
        await async_db_session.commit()

        def fail_reduce(*_args: Any) -> None:
            raise RuntimeError("reduce failed")

        # Two-row fetches leave the cursor mid-result when the first run fails.
        monkeypatch.setattr(epochs_module, "_BREATH_FETCH_ROWS", 2)
        monkeypatch.setattr(epochs_module, "_reduce_run_breaths", fail_reduce)

        with pytest.raises(RuntimeError) as failure:
            await BreathService(
                async_db_session, async_test_profile.id
            ).compare_experimental_epochs(
                [
                    EpochRequest(
                        label="all", date_start=_FIRST_NIGHT, date_end=_nights(3)[-1]
                    )
                ]
            )

        # ``failure`` keeps the traceback, and with it the failed call's frames
        # and any result they hold, alive while another connection writes.
        writer = sqlite3.connect(temp_db, timeout=0)
        try:
            writer.execute("UPDATE profiles SET name = 'renamed'")
            writer.commit()
        finally:
            writer.close()
        assert str(failure.value) == "reduce failed"


@pytest.mark.integration
@pytest.mark.parametrize(
    "method", ["compare_device_epochs", "compare_experimental_epochs"]
)
class TestServiceEpochSpanCap:
    async def test_epoch_over_the_cap_raises_value_error(
        self, async_db_session: AsyncSession, async_test_profile: Profile, method: str
    ) -> None:
        from snore.services.breath_service import BreathService  # noqa: PLC0415

        await _seed_analyzed_nights(async_db_session, async_test_profile.id, 1)
        compare = getattr(
            BreathService(async_db_session, async_test_profile.id), method
        )
        too_long = EpochRequest(
            label="long",
            date_start=_FIRST_NIGHT,
            date_end=_FIRST_NIGHT + timedelta(days=MAX_EPOCH_NIGHTS),
        )

        with pytest.raises(
            ValueError, match=rf"'long' spans {MAX_EPOCH_NIGHTS + 1} nights"
        ):
            await compare([too_long])

    async def test_epoch_at_the_cap_is_compared(
        self, async_db_session: AsyncSession, async_test_profile: Profile, method: str
    ) -> None:
        from snore.services.breath_service import BreathService  # noqa: PLC0415

        await _seed_analyzed_nights(async_db_session, async_test_profile.id, 1)
        compare = getattr(
            BreathService(async_db_session, async_test_profile.id), method
        )
        at_cap = EpochRequest(
            label="year",
            date_start=_FIRST_NIGHT,
            date_end=_FIRST_NIGHT + timedelta(days=MAX_EPOCH_NIGHTS - 1),
        )

        result = await compare([at_cap])

        assert result.epochs[0].nights_with_data == 1
