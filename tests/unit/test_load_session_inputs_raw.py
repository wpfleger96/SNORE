"""Tests for AnalysisService.load_session_inputs_raw() single-query refactor.

Verifies that session, device_manufacturer, and ramp settings are resolved
correctly from the combined Session+Device+Setting query, and that not-found /
profile-scoping semantics are preserved.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.service import AnalysisService
from snore.database.models import Day, Device, Event, Session, Setting

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_FLOW_BLOB = b"flowbytes"
_DUMMY_FLOW_META: dict = {"sample_rate_hz": 25.6}
_DUMMY_FLOW_COUNT = 100


def _patch_waveform(*, flow_count: int = _DUMMY_FLOW_COUNT) -> Any:
    """Patch fetch_waveform_blob and _load_machine_events for the duration of a test."""
    return patch(
        "snore.analysis.service.fetch_waveform_blob",
        new=AsyncMock(return_value=(_DUMMY_FLOW_BLOB, flow_count, _DUMMY_FLOW_META)),
    )


def _patch_events() -> Any:
    return patch(
        "snore.analysis.service.AnalysisService._load_machine_events",
        new=AsyncMock(return_value=[]),
    )


async def _seed_session(
    db_session: AsyncSession,
    device: Device,
    *,
    device_session_id: str = "RAWTEST",
    duration_seconds: float | None = 28800.0,
) -> Session:
    """Insert a Day + Session and return the flushed Session."""
    day = Day(
        device_id=device.id,
        date=datetime(2025, 1, 1).date(),
        total_therapy_hours=8.0,
    )
    db_session.add(day)
    await db_session.flush()

    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=device_session_id,
        start_time=datetime(2025, 1, 1, 21, 0, 0),
        end_time=datetime(2025, 1, 2, 5, 0, 0),
        duration_seconds=duration_seconds,
    )
    db_session.add(sess)
    await db_session.flush()
    return sess


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadMachineEvents:
    async def test_null_duration_loads_as_zero(
        self, async_db_session, async_test_device
    ):
        """No duration is invented for an event the device recorded none for."""
        sess = await _seed_session(async_db_session, async_test_device)
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="H",
                start_time=sess.start_time + timedelta(seconds=90),
            )
        )
        await async_db_session.commit()

        svc = AnalysisService(async_db_session, profile_id=async_test_device.profile_id)
        events = await svc._load_machine_events(sess.id, sess.start_time.timestamp())

        assert [(e.event_type, e.start_time, e.duration) for e in events] == [
            ("H", 90.0, 0.0)
        ]


@pytest.mark.unit
class TestLoadSessionInputsRaw:
    """load_session_inputs_raw() resolves session, device, and settings in one query."""

    async def test_device_manufacturer_resolved(
        self, async_db_session, async_test_device
    ):
        """device_manufacturer is read from the Device row joined to the session."""
        sess = await _seed_session(async_db_session, async_test_device)
        await async_db_session.commit()

        svc = AnalysisService(async_db_session, profile_id=async_test_device.profile_id)
        with _patch_waveform(), _patch_events():
            raw = await svc.load_session_inputs_raw(sess.id)

        assert raw.device_manufacturer == async_test_device.manufacturer

    async def test_settings_resolved_when_present(
        self, async_db_session, async_test_device
    ):
        """All three ramp settings are resolved when Setting rows exist."""
        sess = await _seed_session(async_db_session, async_test_device)
        for key, value in (
            ("ramp_enabled", "true"),
            ("ramp_time", "20"),
            ("smart_ramp", "false"),
        ):
            async_db_session.add(Setting(session_id=sess.id, key=key, value=value))
        await async_db_session.commit()

        svc = AnalysisService(async_db_session, profile_id=async_test_device.profile_id)
        with _patch_waveform(), _patch_events():
            raw = await svc.load_session_inputs_raw(sess.id)

        assert raw.ramp_enabled is True
        assert raw.ramp_time_minutes == 20
        assert raw.smart_ramp is False

    async def test_settings_default_when_absent(
        self, async_db_session, async_test_device
    ):
        """All ramp fields are None/False when no Setting rows exist for the session."""
        sess = await _seed_session(
            async_db_session, async_test_device, device_session_id="RAWTEST_NOSETTINGS"
        )
        await async_db_session.commit()

        svc = AnalysisService(async_db_session, profile_id=async_test_device.profile_id)
        with _patch_waveform(), _patch_events():
            raw = await svc.load_session_inputs_raw(sess.id)

        assert raw.ramp_enabled is None
        assert raw.ramp_time_minutes is None
        assert raw.smart_ramp is False

    async def test_partial_settings_resolved(self, async_db_session, async_test_device):
        """Only the keys that exist are populated; missing keys remain None/False."""
        sess = await _seed_session(
            async_db_session, async_test_device, device_session_id="RAWTEST_PARTIAL"
        )
        async_db_session.add(
            Setting(session_id=sess.id, key="ramp_enabled", value="false")
        )
        await async_db_session.commit()

        svc = AnalysisService(async_db_session, profile_id=async_test_device.profile_id)
        with _patch_waveform(), _patch_events():
            raw = await svc.load_session_inputs_raw(sess.id)

        assert raw.ramp_enabled is False
        assert raw.ramp_time_minutes is None  # row absent
        assert raw.smart_ramp is False  # row absent → default

    async def test_not_found_raises_value_error(
        self, async_db_session, async_test_profile
    ):
        """An unknown session_id raises ValueError with the session id in the message."""
        svc = AnalysisService(async_db_session, profile_id=async_test_profile.id)
        with pytest.raises(ValueError, match="9999"):
            await svc.load_session_inputs_raw(9999)

    async def test_foreign_profile_session_raises_value_error(
        self, async_db_session, async_test_device, async_test_profile
    ):
        """A session belonging to a different profile raises ValueError under profile scoping."""
        import uuid

        from snore.database.models import Device, Profile, User

        # Create a second profile+device that owns a session.
        other_user = User(
            canonical_email=f"other_{uuid.uuid4().hex[:8]}@example.com",
            role="member",
        )
        async_db_session.add(other_user)
        await async_db_session.flush()

        other_profile = Profile(user_id=other_user.id, name="Other Profile")
        async_db_session.add(other_profile)
        await async_db_session.flush()

        other_device = Device(
            profile_id=other_profile.id,
            manufacturer="OtherCo",
            model="Model X",
            serial_number=f"OTHER_{uuid.uuid4().hex[:8]}",
        )
        async_db_session.add(other_device)
        await async_db_session.flush()

        other_sess = await _seed_session(
            async_db_session, other_device, device_session_id="FOREIGN_SESSION"
        )
        await async_db_session.commit()

        # Service scoped to the original profile — must not see the other profile's session.
        svc = AnalysisService(async_db_session, profile_id=async_test_device.profile_id)
        with pytest.raises(ValueError, match=str(other_sess.id)):
            await svc.load_session_inputs_raw(other_sess.id)

    async def test_all_three_setting_rows_aggregated(
        self, async_db_session, async_test_device
    ):
        """When all three Setting keys exist, all three are correctly parsed."""
        sess = await _seed_session(
            async_db_session, async_test_device, device_session_id="RAWTEST_ALL3"
        )
        for key, value in (
            ("ramp_enabled", "1"),
            ("ramp_time", "30"),
            ("smart_ramp", "true"),
        ):
            async_db_session.add(Setting(session_id=sess.id, key=key, value=value))
        await async_db_session.commit()

        svc = AnalysisService(async_db_session, profile_id=async_test_device.profile_id)
        with _patch_waveform(), _patch_events():
            raw = await svc.load_session_inputs_raw(sess.id)

        assert raw.ramp_enabled is True
        assert raw.ramp_time_minutes == 30
        assert raw.smart_ramp is True

    async def test_session_id_in_raw_blob(self, async_db_session, async_test_device):
        """The returned RawSessionBlobs carries the correct session_id."""
        sess = await _seed_session(
            async_db_session, async_test_device, device_session_id="RAWTEST_SESSIONID"
        )
        await async_db_session.commit()

        svc = AnalysisService(async_db_session, profile_id=async_test_device.profile_id)
        with _patch_waveform(), _patch_events():
            raw = await svc.load_session_inputs_raw(sess.id)

        assert raw.session_id == sess.id
