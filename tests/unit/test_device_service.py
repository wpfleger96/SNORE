"""Unit tests for DeviceService settings-history diffing and detail retrieval."""

from datetime import datetime

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.day_manager import DayManager
from snore.database.models import Setting
from snore.exceptions import NotFoundError
from snore.services.device_service import DeviceService
from snore.services.session_service import SessionService


async def _add_settings(
    db_session: AsyncSession, session_id: int, settings: dict[str, str]
) -> None:
    """Insert Setting rows for a session."""
    for key, value in settings.items():
        db_session.add(Setting(session_id=session_id, key=key, value=value))
    await db_session.flush()


class TestSettingsHistoryDiffing:
    async def test_no_sessions_returns_empty_history(
        self, async_db_session, async_test_device
    ):
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.settings_history == []

    async def test_single_session_with_settings_is_baseline_no_history(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        s = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        await _add_settings(
            async_db_session, s.id, {"mode": "AutoSet", "pressure_min": "4.0"}
        )
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.settings_history == []

    async def test_two_sessions_identical_settings_no_history(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        s1 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        s2 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 2, 22, 0)
        )
        await _add_settings(
            async_db_session, s1.id, {"mode": "AutoSet", "pressure_min": "4.0"}
        )
        await _add_settings(
            async_db_session, s2.id, {"mode": "AutoSet", "pressure_min": "4.0"}
        )
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.settings_history == []

    async def test_changed_key_emits_entry(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        s1 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        s2 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 2, 22, 0)
        )
        await _add_settings(async_db_session, s1.id, {"pressure_max": "12.0"})
        await _add_settings(async_db_session, s2.id, {"pressure_max": "10.0"})
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert len(detail.settings_history) == 1
        entry = detail.settings_history[0]
        assert entry.session_id == s2.id
        assert len(entry.changes) == 1
        change = entry.changes[0]
        assert change.key == "pressure_max"
        assert change.old_value == "12.0"
        assert change.new_value == "10.0"

    async def test_added_key_has_null_old_value(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        s1 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        s2 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 2, 22, 0)
        )
        await _add_settings(async_db_session, s1.id, {"mode": "CPAP"})
        await _add_settings(async_db_session, s2.id, {"mode": "CPAP", "epr_level": "2"})
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert len(detail.settings_history) == 1
        changes = {c.key: c for c in detail.settings_history[0].changes}
        assert "epr_level" in changes
        assert changes["epr_level"].old_value is None
        assert changes["epr_level"].new_value == "2"

    async def test_removed_key_has_null_new_value(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        s1 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        s2 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 2, 22, 0)
        )
        await _add_settings(async_db_session, s1.id, {"mode": "CPAP", "epr_level": "2"})
        await _add_settings(async_db_session, s2.id, {"mode": "CPAP"})
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert len(detail.settings_history) == 1
        changes = {c.key: c for c in detail.settings_history[0].changes}
        assert "epr_level" in changes
        assert changes["epr_level"].old_value == "2"
        assert changes["epr_level"].new_value is None

    async def test_session_without_settings_between_two_is_skipped(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        s1 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        # s2 has no settings rows — should be skipped entirely
        await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 2, 22, 0)
        )
        s3 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 3, 22, 0)
        )
        await _add_settings(async_db_session, s1.id, {"mode": "AutoSet"})
        await _add_settings(async_db_session, s3.id, {"mode": "CPAP"})
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        # s2 is skipped; diff is s1 vs s3
        assert len(detail.settings_history) == 1
        assert detail.settings_history[0].session_id == s3.id
        assert detail.settings_history[0].changes[0].key == "mode"
        assert detail.settings_history[0].changes[0].old_value == "AutoSet"
        assert detail.settings_history[0].changes[0].new_value == "CPAP"


class TestGetDeviceDetail:
    async def test_raises_not_found_for_unknown_id(self, async_db_session):
        svc = DeviceService(async_db_session, profile_id=1)
        with pytest.raises(NotFoundError):
            await svc.get_device_detail(99999)

    async def test_usage_summary_counts_enabled_sessions_only(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        disabled = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 2, 22, 0)
        )
        disabled.enabled = False
        await async_db_session.flush()
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.usage.session_count == 1

    async def test_current_settings_from_latest_session(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        s1 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        s2 = await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 2, 22, 0)
        )
        await _add_settings(async_db_session, s1.id, {"mode": "CPAP"})
        await _add_settings(async_db_session, s2.id, {"mode": "AutoSet"})
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.current_settings == {"mode": "AutoSet"}

    async def test_current_settings_none_when_no_settings(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        await async_test_session_factory(
            async_test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.current_settings is None

    async def test_no_sessions_gives_zero_usage(
        self, async_db_session, async_test_device
    ):
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.usage.session_count == 0
        assert detail.usage.first_session_date is None
        assert detail.usage.last_session_date is None
        assert detail.usage.total_therapy_hours == 0.0

    async def test_therapy_hours_use_mask_on_time_not_session_span(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        s = await async_test_session_factory(
            async_test_device.id,
            start_time=datetime(2024, 1, 1, 22, 0),
            duration_hours=8.0,
            usage_hours=6.5,
        )
        await DayManager.link_session_to_day(s, async_test_device.id, async_db_session)
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.usage.total_therapy_hours == 6.5

    async def test_therapy_hours_exclude_disabled_sessions(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        enabled = await async_test_session_factory(
            async_test_device.id,
            start_time=datetime(2024, 1, 1, 22, 0),
            duration_hours=8.0,
            usage_hours=6.5,
        )
        disabled = await async_test_session_factory(
            async_test_device.id,
            start_time=datetime(2024, 1, 2, 6, 30),
            duration_hours=1.0,
            usage_hours=0.75,
        )
        for s in (enabled, disabled):
            await DayManager.link_session_to_day(
                s, async_test_device.id, async_db_session
            )
        assert enabled.day_id == disabled.day_id
        await SessionService(async_db_session, profile_id=1).set_session_enabled(
            disabled.id, False
        )
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.usage.session_count == 1
        assert detail.usage.total_therapy_hours == 6.5

    async def test_identity_fields_included(self, async_db_session, async_test_device):
        svc = DeviceService(async_db_session, profile_id=1)
        detail = await svc.get_device_detail(async_test_device.id)
        assert detail.id == async_test_device.id
        assert detail.manufacturer == async_test_device.manufacturer
        assert detail.model == async_test_device.model
        assert detail.serial_number == async_test_device.serial_number


class TestListDevices:
    async def test_list_empty(self, async_db_session):
        svc = DeviceService(async_db_session, profile_id=1)
        assert await svc.list_devices() == []

    async def test_list_returns_device_info(self, async_db_session, async_test_device):
        svc = DeviceService(async_db_session, profile_id=1)
        devices = await svc.list_devices()
        assert len(devices) == 1
        assert devices[0].id == async_test_device.id
        assert devices[0].serial_number == async_test_device.serial_number
