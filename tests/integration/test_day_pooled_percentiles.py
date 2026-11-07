"""Day pressure/EPAP/leak percentiles pool the imported waveforms (issue #352).

Drives the real import path, ``UnifiedSession.finalize_statistics`` →
``SessionImporter.import_sessions_batch`` → day aggregation, so the samples the
Day percentiles read are the stored float32 blobs, not the parser's arrays.
"""

from __future__ import annotations

import uuid

from datetime import datetime, timedelta

import numpy as np
import pytest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database import models
from snore.database.day_manager import DayManager
from snore.database.importers import SessionImporter
from snore.parsers.unified import (
    DeviceInfo,
    SessionStatistics,
    UnifiedSession,
    WaveformData,
    WaveformType,
)
from tests.integration.conftest import _make_profile

_SAMPLE_RATE_HZ = 0.5


def _leak_session(serial: str, start: datetime, leak: np.ndarray) -> UnifiedSession:
    """A parsed session whose only waveform is ``leak`` (float64, 0.5 Hz)."""
    duration_s = len(leak) / _SAMPLE_RATE_HZ
    session = UnifiedSession(
        device_info=DeviceInfo(
            manufacturer="ResMed", model="AirSense 10", serial_number=serial
        ),
        device_session_id=start.strftime("%Y%m%d_%H%M%S"),
        start_time=start,
        end_time=start + timedelta(seconds=duration_s),
        import_source="resmed_edf",
        statistics=SessionStatistics(),
    )
    session.mask_on_segments = [(0.0, duration_s)]
    session.add_waveform(
        WaveformData(
            waveform_type=WaveformType.LEAK_RATE,
            sample_rate=_SAMPLE_RATE_HZ,
            unit="L/min",
            timestamps=np.arange(len(leak)) / _SAMPLE_RATE_HZ,
            values=leak,
        )
    )
    session.finalize_statistics()
    return session


async def _import_day(db: AsyncSession, sessions: list[UnifiedSession]) -> models.Day:
    profile = await _make_profile(db)
    imported, _, failed, _ = await SessionImporter(
        profile_id=profile.id
    ).import_sessions_batch(sessions, db=db)
    assert (imported, failed) == (len(sessions), 0)
    return (await db.execute(select(models.Day))).scalars().one()


class TestDayPooledPercentilesOnImport:
    async def test_unequal_sessions_day_leak_95th_is_pooled_blob_percentile(
        self, async_db_session: AsyncSession
    ) -> None:
        rng = np.random.default_rng(352)
        high = rng.uniform(40.0, 80.0, int(1 * 3600 * _SAMPLE_RATE_HZ))
        low = rng.uniform(0.0, 10.0, int(7 * 3600 * _SAMPLE_RATE_HZ))
        serial = f"PP_{uuid.uuid4().hex[:8]}"
        sessions = [
            _leak_session(serial, datetime(2026, 3, 1, 22, 0, 0), high),
            _leak_session(serial, datetime(2026, 3, 1, 23, 30, 0), low),
        ]

        day = await _import_day(async_db_session, sessions)

        stored = np.concatenate([high, low]).astype(np.float32)
        weighted = (
            sessions[0].statistics.leak_95th * 1.0
            + sessions[1].statistics.leak_95th * 7.0
        ) / 8.0
        assert day.leak_95th == pytest.approx(np.percentile(stored, 95), rel=1e-9)
        assert day.leak_median == pytest.approx(np.median(stored), rel=1e-9)
        assert day.leak_95th != pytest.approx(weighted, abs=1.0)

    async def test_recalculated_day_keeps_pooled_value(
        self, async_db_session: AsyncSession
    ) -> None:
        rng = np.random.default_rng(353)
        serial = f"PP_{uuid.uuid4().hex[:8]}"
        sessions = [
            _leak_session(
                serial, datetime(2026, 3, 1, 22, 0, 0), rng.uniform(40.0, 80.0, 1800)
            ),
            _leak_session(
                serial, datetime(2026, 3, 1, 23, 30, 0), rng.uniform(0.0, 10.0, 12600)
            ),
        ]
        day = await _import_day(async_db_session, sessions)
        imported_95th = day.leak_95th

        assert await DayManager.recalculate_day(day, async_db_session)

        assert day.leak_95th == imported_95th

    async def test_single_session_day_matches_session_leak_95th(
        self, async_db_session: AsyncSession
    ) -> None:
        leak = np.random.default_rng(354).uniform(0.0, 30.0, 14400)
        serial = f"PP_{uuid.uuid4().hex[:8]}"

        day = await _import_day(
            async_db_session,
            [_leak_session(serial, datetime(2026, 3, 1, 22, 0, 0), leak)],
        )

        stats = (
            (await async_db_session.execute(select(models.Statistics))).scalars().one()
        )
        assert stats.leak_95th is not None
        # Equal to float32 precision: the session value interpolates in float32.
        assert day.leak_95th == pytest.approx(stats.leak_95th, rel=1e-6)
        assert day.leak_median == pytest.approx(stats.leak_median, rel=1e-6)
