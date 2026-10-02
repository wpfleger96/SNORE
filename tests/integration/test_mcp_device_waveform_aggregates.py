"""Integration tests: device FL/snore aggregates in get_nightly_summary and compare_epochs.

Tests seed waveform rows and verify:
- device_flg_* and snore_* fields appear in NightlyRow
- CHANNEL_ABSENT reason when channel not recorded
- snore_pct_time computed correctly
- FL negative sentinel filter applied
- no_sessions reason when Day has no enabled sessions
- compare_epochs device_flg and snore_dist distributions populated
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.conftest import (
    _make_analysis_result,
    _make_day_session,
    _make_device,
)


def _make_waveform_blob(values: list[float]) -> tuple[bytes, int]:
    """Serialize a list of float values into the [timestamp, value] blob format."""
    n = len(values)
    arr = np.zeros((n, 2), dtype=np.float32)
    arr[:, 0] = np.arange(n, dtype=np.float32)  # timestamps: 0, 1, 2, ...
    arr[:, 1] = np.array(values, dtype=np.float32)
    return arr.tobytes(), n


async def _add_waveform(
    db: AsyncSession,
    session_id: int,
    waveform_type: str,
    values: list[float],
) -> None:
    from snore.database.models import Waveform  # noqa: PLC0415

    blob, sample_count = _make_waveform_blob(values)
    wf = Waveform(
        session_id=session_id,
        waveform_type=waveform_type,
        sample_rate=1.0,
        sample_count=sample_count,
        data_blob=blob,
    )
    db.add(wf)
    await db.flush()


@pytest.mark.integration
class TestNightlySummaryDeviceWaveforms:
    async def test_fl_channel_aggregates_populated(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """FL waveform values appear in device_flg_* fields of NightlyRow."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 1)
        _day, sess = await _make_day_session(async_db_session, device, d)

        # FL values: 0.0, 0.2, 0.4, 0.6, 0.8 — no negatives
        await _add_waveform(async_db_session, sess.id, "fl", [0.0, 0.2, 0.4, 0.6, 0.8])

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        assert len(result.nights) == 1
        night = result.nights[0]
        assert night.device_flg_median is not None
        assert night.device_flg_median == pytest.approx(0.4)
        assert night.device_flg_95th is not None
        assert night.device_flg_max == pytest.approx(0.8)
        assert night.device_flg_reason is None

    async def test_fl_negative_sentinels_filtered(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """FL values < 0 are filtered; remaining values are aggregated correctly."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 2)
        _day, sess = await _make_day_session(async_db_session, device, d)

        # Two sentinel values (−0.01 from digital −1) mixed with real data
        await _add_waveform(
            async_db_session, sess.id, "fl", [-0.01, 0.3, 0.5, -0.01, 0.7]
        )

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        # Only [0.3, 0.5, 0.7] remain; median = 0.5
        assert night.device_flg_median == pytest.approx(0.5)
        assert night.device_flg_max == pytest.approx(0.7)

    async def test_snore_channel_aggregates_populated(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Snore waveform values appear in snore_* fields including snore_pct_time."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 3)
        _day, sess = await _make_day_session(async_db_session, device, d)

        # 4 out of 10 samples > 0.5: 0.6, 0.7, 0.8, 1.0
        await _add_waveform(
            async_db_session,
            sess.id,
            "snore",
            [0.0, 0.0, 0.5, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 1.0],
        )

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        assert night.snore_median is not None
        assert night.snore_pct_time == pytest.approx(0.4)  # 4/10
        assert night.snore_reason is None

    async def test_channel_absent_gives_null_with_reason(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Night with no FL or snore waveform rows has null fields and reason string."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 4)
        await _make_day_session(async_db_session, device, d)

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        assert night.device_flg_median is None
        assert night.device_flg_reason == "channel_absent"
        assert night.snore_median is None
        assert night.snore_reason == "channel_absent"

    async def test_multi_session_night_merges_samples(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """FL values from two sessions of the same night are aggregated together."""
        from snore.database.models import Session  # noqa: PLC0415
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 5)
        day, sess1 = await _make_day_session(async_db_session, device, d)

        # Create a second session for the same day
        import uuid  # noqa: PLC0415

        from datetime import datetime, timedelta  # noqa: PLC0415

        start2 = datetime(d.year, d.month, d.day, 23, 0, 0)
        sess2 = Session(
            device_id=device.id,
            day_id=day.id,
            device_session_id=f"test_{d.isoformat()}_{uuid.uuid4().hex[:6]}",
            start_time=start2,
            end_time=start2 + timedelta(hours=2),
            duration_seconds=7200,
            enabled=True,
        )
        async_db_session.add(sess2)
        await async_db_session.flush()

        # Session 1 FL: [0.1, 0.2], Session 2 FL: [0.3, 0.4]
        await _add_waveform(async_db_session, sess1.id, "fl", [0.1, 0.2])
        await _add_waveform(async_db_session, sess2.id, "fl", [0.3, 0.4])

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        # All 4 merged: median of [0.1, 0.2, 0.3, 0.4] = 0.25
        assert night.device_flg_median == pytest.approx(0.25)
        assert night.device_flg_max == pytest.approx(0.4)


@pytest.mark.integration
class TestNightlySummaryNoSessions:
    async def test_day_with_no_sessions_gives_no_sessions_reason(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Day that has no enabled sessions gets device_flg_reason='no_sessions'."""
        from snore.database.models import Day  # noqa: PLC0415
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 8, 1)
        # Create a Day with no sessions attached to it
        day = Day(device_id=device.id, date=d, total_therapy_hours=0)
        async_db_session.add(day)
        await async_db_session.flush()

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        assert len(result.nights) == 1
        night = result.nights[0]
        assert night.device_flg_median is None
        assert night.device_flg_reason == "no_sessions"
        assert night.snore_median is None
        assert night.snore_reason == "no_sessions"


@pytest.mark.integration
class TestCompareEpochsDeviceWaveforms:
    async def test_compare_epochs_device_flg_and_snore_distributions(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """compare_epochs populates device_flg/snore_dist for epoch with waveform data.

        Epoch A: sessions with FL (including a negative sentinel to exercise the filter)
        and snore waveform rows; OK analysis results.
        Epoch B: sessions with OK analysis but no waveform channels.
        Verifies distributions on epoch A and absent-channel behavior on epoch B.
        """
        from snore.mcp.schemas import EpochSpec  # noqa: PLC0415
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)

        # Epoch A: two nights with FL and snore waveforms + OK analysis
        d_a1 = date(2025, 9, 1)
        d_a2 = date(2025, 9, 2)
        _day_a1, sess_a1 = await _make_day_session(async_db_session, device, d_a1)
        _day_a2, sess_a2 = await _make_day_session(async_db_session, device, d_a2)

        # FL values include one negative sentinel (should be filtered) and real values
        await _add_waveform(async_db_session, sess_a1.id, "fl", [-0.01, 0.2, 0.4, 0.6])
        await _add_waveform(async_db_session, sess_a1.id, "snore", [0.0, 0.6, 0.8, 1.0])
        await _add_waveform(async_db_session, sess_a2.id, "fl", [0.1, 0.3, 0.5])
        await _add_waveform(async_db_session, sess_a2.id, "snore", [0.0, 0.0, 0.7])

        await _make_analysis_result(async_db_session, sess_a1)
        await _make_analysis_result(async_db_session, sess_a2)

        # Epoch B: one night with OK analysis but no waveform channels
        d_b = date(2025, 10, 1)
        _day_b, sess_b = await _make_day_session(async_db_session, device, d_b)
        await _make_analysis_result(async_db_session, sess_b)

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(label="A", date_start="2025-09-01", date_end="2025-09-30"),
                EpochSpec(label="B", date_start="2025-10-01", date_end="2025-10-31"),
            ],
        )

        assert len(result.epochs) == 2
        epoch_a = next(e for e in result.epochs if e.label == "A")
        epoch_b = next(e for e in result.epochs if e.label == "B")

        # Epoch A: FL distribution — negative sentinel filtered; real values aggregated
        # Epoch A FL real values: [0.2, 0.4, 0.6] from sess_a1 + [0.1, 0.3, 0.5] from sess_a2
        assert epoch_a.device_flg.n_breaths == 6
        assert epoch_a.device_flg.median is not None
        assert epoch_a.device_flg.n_nights == 2

        # Epoch A: snore distribution populated
        # Snore values: [0.0, 0.6, 0.8, 1.0] + [0.0, 0.0, 0.7] = 7 samples
        assert epoch_a.snore_dist.n_breaths == 7
        assert epoch_a.snore_dist.median is not None

        # Epoch B: no waveform channels — distributions have n_breaths=0, null stats
        assert epoch_b.device_flg.n_breaths == 0
        assert epoch_b.device_flg.median is None
        assert epoch_b.snore_dist.n_breaths == 0
        assert epoch_b.snore_dist.median is None
