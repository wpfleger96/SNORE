"""Unit tests for RxTracker high-level period queries."""

import uuid

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.rx_tracker import (
    TIMELINE_KEYS,
    RxTracker,
    _describe_mask,
    _diff_settings,
    changed_setting_keys,
    merge_changes_with_mask_log,
)
from snore.database.models import Day, Device, Session, Setting
from snore.services.schemas import MaskLogEntryResponse, RxSettingChange


async def _create_device(
    db_session: AsyncSession,
    profile_id: int,
    manufacturer: str = "ResMed",
    model: str = "AirSense 10",
) -> Device:
    """Create a device with a unique serial number."""
    device = Device(
        profile_id=profile_id,
        manufacturer=manufacturer,
        model=model,
        serial_number=f"SN_{uuid.uuid4().hex[:8]}",
    )
    db_session.add(device)
    await db_session.flush()
    return device


async def _create_day_with_session(
    db_session: AsyncSession,
    device: Device,
    day_date: date,
    ahi: float | None = None,
    total_therapy_hours: float = 8.0,
    settings: dict | None = None,
    enabled: bool = True,
    leak_median: float = 5.0,
) -> Day:
    """Helper to create a Day with a linked Session and optional RX settings."""
    day = Day(
        device_id=device.id,
        date=day_date,
        session_count=1,
        total_therapy_hours=total_therapy_hours,
        ahi=ahi,
        leak_median=leak_median,
    )
    db_session.add(day)
    await db_session.flush()

    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"rx_test_{device.id}_{day_date.isoformat()}_{uuid.uuid4().hex[:4]}",
        start_time=datetime.combine(day_date, datetime.min.time()),
        end_time=datetime.combine(day_date, datetime.min.time())
        + timedelta(hours=total_therapy_hours),
        duration_seconds=total_therapy_hours * 3600,
        enabled=enabled,
    )
    db_session.add(sess)
    await db_session.flush()

    if settings:
        for key, value in settings.items():
            db_session.add(Setting(session_id=sess.id, key=key, value=value))
        await db_session.flush()

    return day


RX_SETTINGS = {
    "mode": "APAP",
    "pressure_min": "4.0",
    "pressure_max": "20.0",
}


class TestRxTrackerHistory:
    async def test_history_empty_db(self, async_db_session):
        """Empty database returns empty list."""
        tracker = RxTracker(1)
        result = await tracker.get_history(async_db_session)
        assert result == []

    async def test_history_single_period(self, async_db_session, async_test_device):
        """Single cohesive RX period is returned as one entry."""
        base = date(2025, 1, 1)
        for i in range(10):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=3.0 + i * 0.1,
                settings=RX_SETTINGS,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_history(async_db_session)

        assert len(result) == 1
        assert result[0].days_count == 10
        assert result[0].settings == RX_SETTINGS
        assert result[0].start_date == base
        assert result[0].end_date == base + timedelta(days=9)
        assert result[0].device_id == async_test_device.id
        assert result[0].device_name == "Test Manufacturer Test Model"

    async def test_history_two_periods_different_settings(
        self, async_db_session, async_test_device
    ):
        """Settings change creates two distinct periods."""
        base = date(2025, 1, 1)
        settings_a = {"mode": "CPAP", "pressure_fixed": "8.0"}
        settings_b = {"mode": "APAP", "pressure_min": "6.0", "pressure_max": "20.0"}

        for i in range(5):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=5.0,
                settings=settings_a,
            )
        for i in range(5, 10):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=3.0,
                settings=settings_b,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_history(async_db_session)

        assert len(result) == 2
        assert result[0].days_count == 5
        assert result[1].days_count == 5

    async def test_history_stats_computed(self, async_db_session, async_test_device):
        """Stats fields are populated for periods with data."""
        base = date(2025, 3, 1)
        for i in range(7):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=4.0,
                total_therapy_hours=7.5,
                settings=RX_SETTINGS,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_history(async_db_session)

        assert len(result) == 1
        assert result[0].avg_ahi == pytest.approx(4.0)
        assert result[0].median_ahi == pytest.approx(4.0)
        assert result[0].avg_hours == pytest.approx(7.5)
        assert result[0].total_hours == pytest.approx(7.5 * 7)

    async def test_history_avg_ahi_and_leak_weighted_by_usage_hours(
        self, async_db_session, async_test_device
    ):
        """Period AHI and leak averages weight each night by therapy hours."""
        base = date(2025, 3, 1)
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base,
            ahi=20.0,
            total_therapy_hours=1.0,
            settings=RX_SETTINGS,
            leak_median=40.0,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base + timedelta(days=1),
            ahi=2.0,
            total_therapy_hours=8.0,
            settings=RX_SETTINGS,
            leak_median=4.0,
        )

        result = await RxTracker(1).get_history(async_db_session)

        assert len(result) == 1
        # (20*1 + 2*8) / 9 = 4.0, not the per-night mean 11.0.
        assert result[0].avg_ahi == pytest.approx(4.0)
        # (40*1 + 4*8) / 9 = 8.0, not the per-night mean 22.0.
        assert result[0].avg_leak == pytest.approx(8.0)
        # Median stays per-night.
        assert result[0].median_ahi == pytest.approx(11.0)

    async def test_history_two_devices_same_settings_produce_separate_periods(
        self, async_db_session, async_test_profile
    ):
        """Two devices with identical settings on the same dates yield two separate periods."""
        device_a = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
        )
        device_b = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirCurve 10",
        )
        base = date(2025, 6, 1)
        settings = {"mode": "CPAP", "pressure_fixed": "10.0"}

        for i in range(5):
            await _create_day_with_session(
                async_db_session, device_a, base + timedelta(days=i), settings=settings
            )
            await _create_day_with_session(
                async_db_session, device_b, base + timedelta(days=i), settings=settings
            )
        await async_db_session.flush()

        result = await RxTracker(1).get_history(async_db_session)

        assert len(result) == 2
        device_ids = {r.device_id for r in result}
        assert device_ids == {device_a.id, device_b.id}
        for r in result:
            assert r.days_count == 5
            if r.device_id == device_a.id:
                assert r.device_name == "ResMed AirSense 10"
            else:
                assert r.device_name == "ResMed AirCurve 10"

    async def test_history_ps_change_splits_period(
        self, async_db_session, async_test_device
    ):
        """Changing the 'ps' bilevel key splits the RX period."""
        base = date(2025, 4, 1)
        settings_before = {"mode": "VAuto", "ps": "4.0", "epap": "6.0"}
        settings_after = {"mode": "VAuto", "ps": "6.0", "epap": "6.0"}

        for i in range(3):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=settings_before,
            )
        for i in range(3, 6):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=settings_after,
            )
        await async_db_session.flush()

        result = await RxTracker(1).get_history(async_db_session)

        assert len(result) == 2
        assert result[0].settings["ps"] == "4.0"
        assert result[1].settings["ps"] == "6.0"

    async def test_history_disabled_session_ignored(
        self, async_db_session, async_test_device
    ):
        """Disabled sessions are not used to extract RX settings."""
        base = date(2025, 5, 1)
        settings_enabled = {"mode": "CPAP", "pressure_fixed": "8.0"}
        settings_disabled = {"mode": "APAP", "pressure_min": "4.0"}

        day = Day(
            device_id=async_test_device.id,
            date=base,
            session_count=2,
            total_therapy_hours=8.0,
        )
        async_db_session.add(day)
        await async_db_session.flush()

        # Enabled session — longer, contributes settings
        sess_enabled = Session(
            device_id=async_test_device.id,
            day_id=day.id,
            device_session_id=f"enabled_{uuid.uuid4().hex[:4]}",
            start_time=datetime.combine(base, datetime.min.time()),
            end_time=datetime.combine(base, datetime.min.time()) + timedelta(hours=8),
            duration_seconds=8 * 3600,
            enabled=True,
        )
        async_db_session.add(sess_enabled)
        await async_db_session.flush()
        for k, v in settings_enabled.items():
            async_db_session.add(Setting(session_id=sess_enabled.id, key=k, value=v))

        # Disabled session — different settings, should be ignored
        sess_disabled = Session(
            device_id=async_test_device.id,
            day_id=day.id,
            device_session_id=f"disabled_{uuid.uuid4().hex[:4]}",
            start_time=datetime.combine(base, datetime.min.time()),
            end_time=datetime.combine(base, datetime.min.time()) + timedelta(hours=6),
            duration_seconds=6 * 3600,
            enabled=False,
        )
        async_db_session.add(sess_disabled)
        await async_db_session.flush()
        for k, v in settings_disabled.items():
            async_db_session.add(Setting(session_id=sess_disabled.id, key=k, value=v))

        await async_db_session.flush()

        result = await RxTracker(1).get_history(async_db_session)

        assert len(result) == 1
        assert result[0].settings == settings_enabled

    async def test_history_day_with_only_disabled_sessions_skipped(
        self, async_db_session, async_test_device
    ):
        """A day whose only sessions are disabled contributes no period days."""
        base = date(2025, 7, 1)
        # Day 0: all disabled — skipped
        day_disabled = Day(
            device_id=async_test_device.id,
            date=base,
            session_count=1,
            total_therapy_hours=4.0,
        )
        async_db_session.add(day_disabled)
        await async_db_session.flush()
        sess_off = Session(
            device_id=async_test_device.id,
            day_id=day_disabled.id,
            device_session_id=f"off_{uuid.uuid4().hex[:4]}",
            start_time=datetime.combine(base, datetime.min.time()),
            end_time=datetime.combine(base, datetime.min.time()) + timedelta(hours=4),
            duration_seconds=4 * 3600,
            enabled=False,
        )
        async_db_session.add(sess_off)
        await async_db_session.flush()
        async_db_session.add(Setting(session_id=sess_off.id, key="mode", value="CPAP"))
        await async_db_session.flush()

        # Day 1: enabled
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base + timedelta(days=1),
            settings=RX_SETTINGS,
        )
        await async_db_session.flush()

        result = await RxTracker(1).get_history(async_db_session)

        assert len(result) == 1
        assert result[0].start_date == base + timedelta(days=1)


class TestRxTrackerCurrent:
    async def test_current_returns_last_period(
        self, async_db_session, async_test_device
    ):
        """Returns the most recent RX period when multiple exist."""
        base = date(2025, 1, 1)
        settings_a = {"mode": "CPAP", "pressure_fixed": "8.0"}
        settings_b = {"mode": "APAP", "pressure_min": "6.0", "pressure_max": "20.0"}

        for i in range(5):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=settings_a,
            )
        for i in range(5, 10):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=settings_b,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_current(async_db_session)

        assert result is not None
        assert result.settings == settings_b
        assert result.end_date == base + timedelta(days=9)


class TestRxTrackerCurrentTailWalk:
    async def test_current_empty_db_returns_none(
        self, async_db_session: AsyncSession
    ) -> None:
        """Empty database returns None."""
        assert await RxTracker(1).get_current(async_db_session) is None

    async def test_current_equals_history_last_single_device(
        self, async_db_session: AsyncSession, async_test_device: Device
    ) -> None:
        """Single device with multiple periods: get_current equals get_history()[-1]."""
        base = date(2025, 1, 1)
        settings_a = {"mode": "CPAP", "pressure_fixed": "8.0"}
        settings_b = {"mode": "APAP", "pressure_min": "6.0", "pressure_max": "20.0"}
        for i in range(5):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=settings_a,
            )
        for i in range(5, 10):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=settings_b,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        assert (
            await tracker.get_current(async_db_session)
            == (await tracker.get_history(async_db_session))[-1]
        )

    async def test_current_equals_history_last_multi_device(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Two devices with interleaved dates: get_current equals get_history()[-1]."""
        device_a = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
        )
        device_b = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirCurve 10",
        )
        base = date(2025, 3, 1)
        settings_a = {"mode": "CPAP", "pressure_fixed": "10.0"}
        settings_b = {"mode": "APAP", "pressure_min": "5.0", "pressure_max": "20.0"}

        for i in range(7):
            await _create_day_with_session(
                async_db_session,
                device_a,
                base + timedelta(days=i),
                settings=settings_a,
            )
        # device_b starts later, so its period will have the higher start_date
        for i in range(14, 21):
            await _create_day_with_session(
                async_db_session,
                device_b,
                base + timedelta(days=i),
                settings=settings_b,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        assert (
            await tracker.get_current(async_db_session)
            == (await tracker.get_history(async_db_session))[-1]
        )

    async def test_current_later_start_on_lower_device_id_wins(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A later start_date wins over a higher device_id with an earlier start."""
        device_low = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
        )
        device_high = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirCurve 10",
        )
        assert device_low.id < device_high.id

        base = date(2025, 5, 1)
        settings_low = {"mode": "CPAP", "pressure_fixed": "8.0"}
        settings_high = {"mode": "APAP", "pressure_min": "4.0", "pressure_max": "20.0"}

        # device_high starts at base (earlier), device_low starts 10 days later
        for i in range(5):
            await _create_day_with_session(
                async_db_session,
                device_high,
                base + timedelta(days=i),
                settings=settings_high,
            )
        for i in range(10, 15):
            await _create_day_with_session(
                async_db_session,
                device_low,
                base + timedelta(days=i),
                settings=settings_low,
            )
        await async_db_session.flush()

        result = await RxTracker(1).get_current(async_db_session)
        assert result is not None
        assert result.device_id == device_low.id
        assert result.start_date == base + timedelta(days=10)

    async def test_current_skips_none_settings_days_at_tail(
        self, async_db_session: AsyncSession, async_test_device: Device
    ) -> None:
        """Newest days with no valid settings are skipped; real period below is returned."""
        base = date(2025, 6, 1)
        real_settings = {"mode": "CPAP", "pressure_fixed": "8.0"}

        # Days 0–4: real period with valid settings
        for i in range(5):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=real_settings,
            )
        # Days 5–7: disabled sessions → _get_day_period_settings returns None
        for i in range(5, 8):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=real_settings,
                enabled=False,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_current(async_db_session)
        assert result is not None
        assert result == (await tracker.get_history(async_db_session))[-1]

    async def test_current_skips_none_settings_day_mid_period(
        self, async_db_session: AsyncSession, async_test_device: Device
    ) -> None:
        """A None-settings day sandwiched inside the last period is skipped, not a boundary."""
        base = date(2025, 7, 1)
        settings_old = {"mode": "CPAP", "pressure_fixed": "6.0"}
        settings_new = {"mode": "APAP", "pressure_min": "4.0", "pressure_max": "20.0"}

        # Older different-fingerprint days
        for i in range(3):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=settings_old,
            )
        # Newer period: valid day, then a None-settings day, then a valid day (newest)
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base + timedelta(days=3),
            settings=settings_new,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base + timedelta(days=4),
            settings=settings_new,
            enabled=False,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base + timedelta(days=5),
            settings=settings_new,
        )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_current(async_db_session)
        assert result is not None
        assert result == (await tracker.get_history(async_db_session))[-1]

    async def test_current_all_days_same_settings_single_period(
        self, async_db_session: AsyncSession, async_test_device: Device
    ) -> None:
        """When fingerprint never changes, all batches exhaust and whole history is one period."""
        base = date(2025, 2, 1)
        settings = {"mode": "CPAP", "pressure_fixed": "10.0"}
        for i in range(30):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=settings,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_current(async_db_session)
        assert result is not None
        assert result.days_count == 30
        assert result == (await tracker.get_history(async_db_session))[-1]

    async def test_current_period_spanning_batch_boundary(
        self, async_db_session: AsyncSession, async_test_device: Device
    ) -> None:
        """Current period >90 days spans a batch boundary; result equals get_history()[-1]."""
        base = date(2024, 1, 1)
        settings_old = {"mode": "CPAP", "pressure_fixed": "6.0"}
        settings_new = {"mode": "APAP", "pressure_min": "4.0", "pressure_max": "20.0"}

        # 10 older days with different settings
        for i in range(10):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=settings_old,
            )
        # 95 consecutive days with current settings (crosses the 90-day batch boundary)
        new_start = base + timedelta(days=10)
        for i in range(95):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                new_start + timedelta(days=i),
                settings=settings_new,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_current(async_db_session)
        assert result is not None
        assert result.start_date == new_start
        assert result.days_count == 95
        assert result == (await tracker.get_history(async_db_session))[-1]


class TestRxTrackerComparison:
    async def test_comparison_empty_db(self, async_db_session):
        """Empty database returns empty comparison."""
        tracker = RxTracker(1)
        result = await tracker.get_comparison(async_db_session)
        assert result.periods == []
        assert result.best_index is None
        assert result.worst_index is None

    async def test_comparison_insufficient_days(
        self, async_db_session, async_test_device
    ):
        """Periods with fewer days than min_days are excluded from best/worst."""
        base = date(2025, 1, 1)
        for i in range(3):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=2.0,
                settings=RX_SETTINGS,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_comparison(async_db_session, min_days=7)

        assert len(result.periods) == 1
        # Period has fewer days than min_days threshold, so no best/worst
        assert result.best_index is None
        assert result.worst_index is None

    async def test_comparison_best_worst_identified(
        self, async_db_session, async_test_device
    ):
        """Best (lowest AHI) and worst (highest AHI) periods identified correctly."""
        base = date(2025, 1, 1)
        settings_a = {"mode": "CPAP", "pressure_fixed": "8.0"}
        settings_b = {"mode": "APAP", "pressure_min": "5.0", "pressure_max": "20.0"}
        settings_c = {"mode": "APAP", "pressure_min": "8.0", "pressure_max": "20.0"}

        for i in range(10):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=5.0,
                settings=settings_a,
            )
        for i in range(10, 20):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=1.5,
                settings=settings_b,
            )
        for i in range(20, 30):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=8.0,
                settings=settings_c,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_comparison(async_db_session, min_days=7)

        assert len(result.periods) == 3
        assert result.best_index == 1
        assert result.worst_index == 2

    async def test_comparison_zero_ahi_period_selected_as_best(
        self, async_db_session, async_test_device
    ):
        """A period averaging AHI 0.0 is the best period, not treated as missing."""
        base = date(2025, 1, 1)
        settings_a = {"mode": "CPAP", "pressure_fixed": "8.0"}
        settings_b = {"mode": "CPAP", "pressure_fixed": "10.0"}
        for i in range(7):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=3.0,
                settings=settings_a,
            )
        for i in range(7, 14):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                ahi=0.0,
                settings=settings_b,
            )

        result = await RxTracker(1).get_comparison(async_db_session, min_days=7)

        assert len(result.periods) == 2
        assert result.best_index == 1
        assert result.worst_index == 0


class TestDiffSettings:
    def test_empty_dicts(self):
        assert _diff_settings({}, {}) == []

    def test_no_change(self):
        s = {"mode": "CPAP", "pressure_fixed": "8.0"}
        assert _diff_settings(s, s.copy()) == []

    def test_value_changed(self):
        prev = {"mode": "CPAP", "pressure_fixed": "8.0"}
        curr = {"mode": "CPAP", "pressure_fixed": "10.0"}
        result = _diff_settings(prev, curr)
        assert result == [("pressure_fixed", "8.0", "10.0")]

    def test_key_added(self):
        result = _diff_settings({}, {"ps": "4.0"})
        assert result == [("ps", None, "4.0")]

    def test_key_removed(self):
        result = _diff_settings({"ps": "4.0"}, {})
        assert result == [("ps", "4.0", None)]

    def test_multiple_changes_sorted_by_key(self):
        prev = {"mode": "CPAP", "ps": "4.0", "epap": "6.0"}
        curr = {"mode": "VAuto", "ps": "6.0", "epap": "6.0"}
        result = _diff_settings(prev, curr)
        keys = [r[0] for r in result]
        assert keys == sorted(keys)
        assert ("mode", "CPAP", "VAuto") in result
        assert ("ps", "4.0", "6.0") in result


class TestRxTrackerChanges:
    async def test_changes_empty_db(self, async_db_session):
        """Empty database returns empty changes list."""
        result = await RxTracker(1).get_changes(async_db_session)
        assert result.changes == []

    async def test_single_day_emits_no_changes(
        self, async_db_session, async_test_device
    ):
        """First day with settings has no previous to diff against."""
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            date(2025, 1, 1),
            settings={"mode": "CPAP", "pressure_fixed": "8.0"},
        )
        await async_db_session.flush()

        result = await RxTracker(1).get_changes(async_db_session)
        assert result.changes == []

    async def test_multi_key_change_emits_one_entry_per_key(
        self, async_db_session, async_test_device
    ):
        """A day where two keys change yields two separate RxSettingChange entries."""
        base = date(2025, 2, 1)
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base,
            settings={"mode": "CPAP", "pressure_fixed": "8.0"},
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base + timedelta(days=1),
            settings={"mode": "APAP", "pressure_min": "6.0"},
        )
        await async_db_session.flush()

        result = await RxTracker(1).get_changes(async_db_session)
        keys = {c.key for c in result.changes}
        assert "mode" in keys
        assert len(result.changes) >= 2
        for c in result.changes:
            assert c.date == base + timedelta(days=1)
            assert c.device_id == async_test_device.id

    async def test_days_without_settings_are_bridged(
        self, async_db_session, async_test_device
    ):
        """A gap day (no enabled sessions with settings) is skipped; diff uses last day WITH settings."""
        base = date(2025, 3, 1)
        # Day 0: settings A
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base,
            settings={"mode": "CPAP", "pressure_fixed": "8.0"},
        )
        # Day 1: no settings (only disabled session)
        day_gap = Day(
            device_id=async_test_device.id,
            date=base + timedelta(days=1),
            session_count=1,
            total_therapy_hours=0.0,
        )
        async_db_session.add(day_gap)
        await async_db_session.flush()
        sess_gap = Session(
            device_id=async_test_device.id,
            day_id=day_gap.id,
            device_session_id=f"gap_{uuid.uuid4().hex[:4]}",
            start_time=datetime.combine(base + timedelta(days=1), datetime.min.time()),
            end_time=datetime.combine(base + timedelta(days=1), datetime.min.time())
            + timedelta(hours=1),
            duration_seconds=3600,
            enabled=False,
        )
        async_db_session.add(sess_gap)
        await async_db_session.flush()
        # Day 2: settings B (change from day 0)
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base + timedelta(days=2),
            settings={"mode": "APAP", "pressure_min": "4.0"},
        )
        await async_db_session.flush()

        result = await RxTracker(1).get_changes(async_db_session)
        # Only day 2 triggers diffs (day 1 is skipped), so changes happen at day 2
        assert all(c.date == base + timedelta(days=2) for c in result.changes)
        assert len(result.changes) >= 1

    async def test_changes_across_two_devices_are_independent(
        self, async_db_session, async_test_profile
    ):
        """Changes from different devices are not cross-diffed."""
        device_a = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
        )
        device_b = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirCurve 10",
        )
        base = date(2025, 4, 1)

        # Device A: CPAP → APAP
        await _create_day_with_session(
            async_db_session,
            device_a,
            base,
            settings={"mode": "CPAP", "pressure_fixed": "8.0"},
        )
        await _create_day_with_session(
            async_db_session,
            device_a,
            base + timedelta(days=1),
            settings={"mode": "APAP", "pressure_min": "6.0"},
        )
        # Device B: stable settings, no change
        await _create_day_with_session(
            async_db_session,
            device_b,
            base,
            settings={"mode": "CPAP", "pressure_fixed": "8.0"},
        )
        await _create_day_with_session(
            async_db_session,
            device_b,
            base + timedelta(days=1),
            settings={"mode": "CPAP", "pressure_fixed": "8.0"},
        )
        await async_db_session.flush()

        result = await RxTracker(1).get_changes(async_db_session)
        # All changes should be from device_a only
        assert all(c.device_id == device_a.id for c in result.changes)
        assert len(result.changes) >= 1

    async def test_changes_sorted_by_date_device_key(
        self, async_db_session, async_test_profile
    ):
        """Changes are sorted (date, device_id, key) ascending."""
        device_a = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
        )
        device_b = await _create_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="ResMed",
            model="AirCurve 10",
        )
        base = date(2025, 5, 1)

        for device in [device_a, device_b]:
            await _create_day_with_session(
                async_db_session, device, base, settings={"mode": "CPAP", "ps": "4.0"}
            )
            await _create_day_with_session(
                async_db_session,
                device,
                base + timedelta(days=1),
                settings={"mode": "APAP", "ps": "6.0"},
            )
        await async_db_session.flush()

        result = await RxTracker(1).get_changes(async_db_session)
        changes = result.changes
        for i in range(len(changes) - 1):
            a, b = changes[i], changes[i + 1]
            assert (a.date, a.device_id, a.key) <= (b.date, b.device_id, b.key)

    async def test_non_rx_key_change_is_tracked(
        self, async_db_session, async_test_device
    ):
        """A comfort-setting change (humidity_level) not in RX_KEYS appears in get_changes."""
        base = date(2025, 6, 1)
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base,
            settings={"mode": "CPAP", "pressure_fixed": "8.0", "humidity_level": "3"},
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            base + timedelta(days=1),
            settings={"mode": "CPAP", "pressure_fixed": "8.0", "humidity_level": "5"},
        )
        await async_db_session.flush()

        result = await RxTracker(1).get_changes(async_db_session)

        humidity_changes = [c for c in result.changes if c.key == "humidity_level"]
        assert len(humidity_changes) == 1
        change = humidity_changes[0]
        assert change.date == base + timedelta(days=1)
        assert change.old_value == "3"
        assert change.new_value == "5"
        assert change.device_id == async_test_device.id


class TestTimelineKeys:
    async def test_timeline_keys_mask_type_change_splits_period(
        self, async_db_session, async_test_device
    ):
        """get_history(keys=TIMELINE_KEYS) splits a period when mask_type changes."""
        base = date(2025, 8, 1)
        for i in range(3):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings={**RX_SETTINGS, "mask_type": "Pillows"},
            )
        for i in range(3, 6):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings={**RX_SETTINGS, "mask_type": "Full Face"},
            )
        await async_db_session.flush()

        result = await RxTracker(1).get_history(async_db_session, keys=TIMELINE_KEYS)

        assert len(result) == 2
        assert result[0].settings["mask_type"] == "Pillows"
        assert result[1].settings["mask_type"] == "Full Face"
        assert result[1].start_date == base + timedelta(days=3)

    async def test_timeline_keys_mask_type_removal_splits_period(
        self, async_db_session, async_test_device
    ):
        """A mask_type present→absent transition splits TIMELINE_KEYS periods.

        Under the MCP timeline semantics (absent keys read as None), the
        removal is flagged as a mask_type change; default RX_KEYS history is
        unaffected.
        """
        base = date(2025, 8, 1)
        for i in range(3):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings={**RX_SETTINGS, "mask_type": "Pillows"},
            )
        for i in range(3, 6):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings=RX_SETTINGS,
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        result = await tracker.get_history(async_db_session, keys=TIMELINE_KEYS)

        assert len(result) == 2
        assert result[0].settings["mask_type"] == "Pillows"
        assert "mask_type" not in result[1].settings
        assert result[1].start_date == base + timedelta(days=3)

        # MCP layer semantics: absent keys become None, flagging the removal
        prev = {k: result[0].settings.get(k) for k in TIMELINE_KEYS}
        curr = {k: result[1].settings.get(k) for k in TIMELINE_KEYS}
        assert changed_setting_keys(prev, curr) == {"mask_type"}

        # Default RX_KEYS history ignores the mask_type removal
        assert len(await tracker.get_history(async_db_session)) == 1

    async def test_default_history_does_not_split_on_mask_type(
        self, async_db_session, async_test_device
    ):
        """Default get_history() ignores a mask_type-only change (RX_KEYS behavior)."""
        base = date(2025, 8, 1)
        for i in range(3):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings={**RX_SETTINGS, "mask_type": "Pillows"},
            )
        for i in range(3, 6):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings={**RX_SETTINGS, "mask_type": "Full Face"},
            )
        await async_db_session.flush()

        result = await RxTracker(1).get_history(async_db_session)

        assert len(result) == 1
        assert result[0].days_count == 6

    async def test_humidity_level_change_never_splits_period(
        self, async_db_session, async_test_device
    ):
        """humidity_level is excluded from both RX_KEYS and TIMELINE_KEYS epochs."""
        base = date(2025, 8, 1)
        for i in range(3):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings={**RX_SETTINGS, "humidity_level": "3"},
            )
        for i in range(3, 6):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings={**RX_SETTINGS, "humidity_level": "5"},
            )
        await async_db_session.flush()

        tracker = RxTracker(1)
        assert len(await tracker.get_history(async_db_session)) == 1
        assert len(await tracker.get_history(async_db_session, keys=TIMELINE_KEYS)) == 1

    async def test_get_current_unaffected_by_mask_type_change(
        self, async_db_session, async_test_device
    ):
        """get_current keeps RX-only fingerprints: mask_type change is no boundary."""
        base = date(2025, 8, 1)
        for i in range(3):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings={**RX_SETTINGS, "mask_type": "Pillows"},
            )
        for i in range(3, 6):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                settings={**RX_SETTINGS, "mask_type": "Full Face"},
            )
        await async_db_session.flush()

        result = await RxTracker(1).get_current(async_db_session)

        assert result is not None
        assert result.days_count == 6
        assert result.start_date == base
        assert "mask_type" not in result.settings


def _make_mask_entry(**kwargs: object) -> MaskLogEntryResponse:
    """Build a MaskLogEntryResponse with sensible defaults, overridable per-test."""
    defaults: dict[str, object] = {
        "id": 1,
        "brand": "ResMed",
        "model": "AirFit P10",
        "style": "pillows",
        "start_date": date(2025, 6, 1),
        "size": None,
        "notes": None,
    }
    defaults.update(kwargs)
    return MaskLogEntryResponse.model_validate(defaults)


class TestDescribeMask:
    """Tests for _describe_mask() with the new nullable brand/model/style contract."""

    def test_full_brand_and_model_no_size(self):
        entry = _make_mask_entry(
            brand="ResMed", model="AirFit P10", style="pillows", size=None
        )
        assert _describe_mask(entry) == "ResMed AirFit P10"

    def test_full_brand_and_model_with_size(self):
        entry = _make_mask_entry(
            brand="ResMed", model="AirFit P10", style="pillows", size="M"
        )
        assert _describe_mask(entry) == "ResMed AirFit P10 (M)"

    def test_brand_only_no_model(self):
        entry = _make_mask_entry(brand="ResMed", model=None, style="pillows", size=None)
        assert _describe_mask(entry) == "ResMed"

    def test_model_only_no_brand(self):
        entry = _make_mask_entry(
            brand=None, model="AirFit P10", style="pillows", size=None
        )
        assert _describe_mask(entry) == "AirFit P10"

    def test_style_fallback_when_no_brand_or_model(self):
        entry = _make_mask_entry(brand=None, model=None, style="pillows", size=None)
        assert _describe_mask(entry) == "pillows"

    def test_style_fallback_with_size(self):
        entry = _make_mask_entry(brand=None, model=None, style="pillows", size="M")
        assert _describe_mask(entry) == "pillows (M)"

    def test_unspecified_mask_when_all_none(self):
        entry = _make_mask_entry(brand=None, model=None, style=None, size=None)
        assert _describe_mask(entry) == "unspecified mask"

    def test_model_only_with_size(self):
        entry = _make_mask_entry(
            brand=None, model="AirFit P10", style="pillows", size="S"
        )
        assert _describe_mask(entry) == "AirFit P10 (S)"


class TestMergeChangesWithMaskLog:
    """Tests for merge_changes_with_mask_log() with null-date entries skipped."""

    def test_null_date_entry_skipped_entirely(self):
        """An entry with start_date=None is excluded from the merged timeline."""
        null_entry = _make_mask_entry(
            id=1,
            brand="ResMed",
            model="AirFit P10",
            style="pillows",
            start_date=None,
        )
        window_start = date(2025, 6, 1)
        window_end = date(2025, 6, 30)

        result = merge_changes_with_mask_log(
            device_changes=[],
            mask_entries=[null_entry],
            start=window_start,
            end=window_end,
        )

        assert result == []

    def test_null_date_entry_does_not_perturb_prev_desc(self):
        """A null-date entry sandwiched before a dated entry leaves prev_desc unaffected.

        The dated entry after the null-date entry should see the last DATED
        entry's description as old_value, as if the null-date entry never existed.
        """
        first_dated = _make_mask_entry(
            id=1,
            brand="ResMed",
            model="AirFit P10",
            style="pillows",
            start_date=date(2025, 5, 1),
        )
        null_entry = _make_mask_entry(
            id=2,
            brand="Philips",
            model="DreamWear",
            style="nasal",
            start_date=None,
        )
        second_dated = _make_mask_entry(
            id=3,
            brand="Fisher & Paykel",
            model="Evora",
            style="full_face",
            start_date=date(2025, 6, 15),
        )
        window_start = date(2025, 6, 1)
        window_end = date(2025, 6, 30)

        result = merge_changes_with_mask_log(
            device_changes=[],
            mask_entries=[first_dated, null_entry, second_dated],
            start=window_start,
            end=window_end,
        )

        # Only second_dated falls in the window; old_value must come from
        # first_dated (the last dated entry before the window), NOT from
        # null_entry which is skipped entirely.
        assert len(result) == 1
        assert result[0].new_value == "Fisher & Paykel Evora"
        assert result[0].old_value == "ResMed AirFit P10"

    def test_first_dated_entry_after_null_has_none_old_value(self):
        """When no prior dated entry exists, old_value is None even if a null-date entry precedes."""
        null_entry = _make_mask_entry(
            id=1,
            brand="Philips",
            model="DreamWear",
            style="nasal",
            start_date=None,
        )
        dated = _make_mask_entry(
            id=2,
            brand="ResMed",
            model="AirFit P10",
            style="pillows",
            start_date=date(2025, 6, 10),
        )
        window_start = date(2025, 6, 1)
        window_end = date(2025, 6, 30)

        result = merge_changes_with_mask_log(
            device_changes=[],
            mask_entries=[null_entry, dated],
            start=window_start,
            end=window_end,
        )

        assert len(result) == 1
        assert result[0].new_value == "ResMed AirFit P10"
        assert result[0].old_value is None

    def test_same_date_device_change_sorts_before_mask_entry(self):
        """On the same date, source="device_settings" sorts before source="mask_log"."""
        device_change = RxSettingChange(
            date=date(2025, 3, 2),
            device_id=1,
            device_name="ResMed AirSense 10",
            key="pressure_fixed",
            old_value="8.0",
            new_value="9.0",
        )
        mask_entry = _make_mask_entry(
            id=2,
            brand="ResMed",
            model="AirFit P10",
            style="pillows",
            start_date=date(2025, 3, 2),
        )
        window_start = date(2025, 3, 1)
        window_end = date(2025, 3, 31)

        result = merge_changes_with_mask_log(
            device_changes=[device_change],
            mask_entries=[mask_entry],
            start=window_start,
            end=window_end,
        )

        assert len(result) == 2
        assert result[0].source == "device_settings"
        assert result[1].source == "mask_log"
