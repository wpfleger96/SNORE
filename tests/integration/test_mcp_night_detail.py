"""Integration tests: get_night_detail against a real SQLite DB."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Day, Device, Session, Statistics
from snore.mcp.errors import ValidationError
from snore.mcp.tools.night_detail import get_night_detail
from tests.integration.conftest import (
    _make_apple_night,
    _make_day_session,
    _make_device,
    _make_event,
    _make_profile,
)

pytestmark = pytest.mark.integration

NIGHT = date(2026, 3, 10)


async def _add_session(
    db: AsyncSession, device: Device, day: Day, start: datetime, *, enabled: bool
) -> Session:
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"extra_{start.isoformat()}",
        start_time=start,
        end_time=start + timedelta(hours=1),
        duration_seconds=3600.0,
        therapy_mode="APAP",
        enabled=enabled,
    )
    db.add(sess)
    await db.flush()
    return sess


class TestGetNightDetail:
    async def test_sessions_carry_device_statistics_in_ml(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        device = await _make_device(async_db_session, async_test_profile.id)
        day, first = await _make_day_session(
            async_db_session, device, NIGHT, ahi=2.5, index_source="device"
        )
        disabled = await _add_session(
            async_db_session, device, day, first.end_time, enabled=False
        )
        async_db_session.add(
            Statistics(
                session_id=first.id,
                ahi_device=2.5,
                rin=0.4,
                csr_pct=1.5,
                amb_humidity_median=45.0,
                tidal_volume_mean=0.45,  # ResMed STR litres
            )
        )
        await async_db_session.flush()

        result = await get_night_detail(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )

        assert result.device is not None
        assert result.device.ahi == 2.5
        assert result.device.index_source == "device"
        assert [s.id for s in result.sessions] == [first.id, disabled.id]
        assert [s.enabled for s in result.sessions] == [True, False]
        stats = result.sessions[0].statistics
        assert stats is not None
        assert (stats.ahi_device, stats.rin, stats.csr_pct) == (2.5, 0.4, 1.5)
        assert stats.amb_humidity_median == 45.0
        assert stats.tidal_volume_mean == 450.0
        assert stats.reras is None  # the device never scored a RERA
        assert result.sessions[1].statistics is None
        assert result.sessions[1].therapy_mode == "APAP"
        assert result.apple_health is None

    async def test_scored_rera_type_reports_zero_counts(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, NIGHT)
        await _make_event(async_db_session, sess, event_type="RE")
        async_db_session.add(Statistics(session_id=sess.id, reras=0))
        await async_db_session.flush()

        result = await get_night_detail(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )

        stats = result.sessions[0].statistics
        assert stats is not None
        assert stats.reras == 0

    async def test_apple_only_night_returns_apple_block(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        await _make_apple_night(async_db_session, async_test_profile.id, NIGHT, 25200.0)

        result = await get_night_detail(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )

        assert result.device is None
        assert result.sessions == []
        assert result.apple_health is not None
        assert result.apple_health.total_sleep_seconds == 25200.0

    async def test_device_block_keeps_day_page_apple_summary(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, NIGHT)
        await _make_apple_night(async_db_session, async_test_profile.id, NIGHT, 25200.0)

        result = await get_night_detail(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )

        assert result.device is not None
        assert result.device.session_ids == [sess.id]
        assert result.device.health_sleep is not None
        assert result.device.health_sleep.total_sleep_seconds == 25200.0
        assert result.apple_health is not None

    @pytest.mark.parametrize("with_apple_night", [False, True])
    @pytest.mark.parametrize("target", ["nonexistent", "other_profile", "no_night"])
    async def test_unmatched_device_id_raises(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Any,
        target: str,
        with_apple_night: bool,
    ) -> None:
        own = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(async_db_session, own, NIGHT)
        other = await _make_device(
            async_db_session, (await _make_profile(async_db_session)).id
        )
        await _make_day_session(async_db_session, other, NIGHT)
        nightless = await _make_device(async_db_session, async_test_profile.id)
        device_id = {
            "nonexistent": other.id + 1000,
            "other_profile": other.id,
            "no_night": nightless.id,
        }[target]
        if with_apple_night:
            await _make_apple_night(
                async_db_session, async_test_profile.id, NIGHT, 25200.0
            )

        with pytest.raises(
            ValidationError,
            match=f"^device_id={device_id} has no night on {NIGHT} in this session$",
        ):
            await get_night_detail(
                async_db_session,
                NIGHT,
                profile_id=async_test_profile.id,
                device_id=device_id,
            )

    async def test_two_devices_without_device_id_raises_ambiguity(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        first = await _make_device(async_db_session, async_test_profile.id)
        second = await _make_device(async_db_session, async_test_profile.id)
        for device in (first, second):
            await _make_day_session(async_db_session, device, NIGHT)

        with pytest.raises(ValidationError) as exc_info:
            await get_night_detail(
                async_db_session, NIGHT, profile_id=async_test_profile.id
            )

        assert str(exc_info.value) == (
            f"Multiple devices have sessions on {NIGHT}: pass device_id to "
            f"disambiguate. Devices: device_id={first.id} "
            f"(serial={first.serial_number}), device_id={second.id} "
            f"(serial={second.serial_number})"
        )

    async def test_device_id_on_shared_date_returns_that_devices_night(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        first = await _make_device(async_db_session, async_test_profile.id)
        second = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(async_db_session, first, NIGHT, ahi=1.0)
        _, second_sess = await _make_day_session(
            async_db_session, second, NIGHT, ahi=4.0
        )

        result = await get_night_detail(
            async_db_session,
            NIGHT,
            profile_id=async_test_profile.id,
            device_id=second.id,
        )

        assert result.device is not None
        assert result.device.device_id == second.id
        assert result.device.ahi == 4.0
        assert [s.id for s in result.sessions] == [second_sess.id]

    async def test_invalid_stored_mask_on_segments_returns_sessions(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, NIGHT)
        await async_db_session.execute(
            text("UPDATE sessions SET mask_on_segments = :stored WHERE id = :id"),
            {"stored": "{not json", "id": sess.id},
        )
        async_db_session.expunge_all()

        result = await get_night_detail(
            async_db_session, NIGHT, profile_id=async_test_profile.id
        )

        assert [s.id for s in result.sessions] == [sess.id]
        assert result.device is not None
        assert result.device.mask_on_spans is None

    async def test_night_with_no_data_raises(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        with pytest.raises(ValidationError, match="No therapy or Apple Health data"):
            await get_night_detail(
                async_db_session, NIGHT, profile_id=async_test_profile.id
            )

    async def test_other_profiles_night_is_not_returned(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        other = await _make_profile(async_db_session)
        other_device = await _make_device(async_db_session, other.id)
        await _make_day_session(async_db_session, other_device, NIGHT)
        await _make_apple_night(async_db_session, other.id, NIGHT, 25200.0)

        with pytest.raises(ValidationError):
            await get_night_detail(
                async_db_session, NIGHT, profile_id=async_test_profile.id
            )
