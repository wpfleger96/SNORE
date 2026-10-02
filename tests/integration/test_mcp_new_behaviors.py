"""Integration tests for new MCP tool behaviors added in the mcp-skeleton review cycle.

Covers:
- rera_index / rdi reason="duration_zero" when Day.total_therapy_hours == 0 but
  analysis IS present (NullReason.DURATION_ZERO branch).
- Compliance block present even on empty range-mode responses (no day rows).
- get_events max_events truncation: total_events keeps untruncated count; truncated=True
  when cut; validate_max_events raises ValidationError for max_events < 1.
- DeviceNotOwnedError maps to a client-safe error message that does NOT leak the
  internal profile id ("profile" must not appear in the message).
- DeviceCapabilities identity fields (manufacturer, model, serial_number) populated
  from the owned Device row.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import (
    AnalysisResult,
    Breath,
    Event,
    Session,
)
from tests.integration.conftest import (
    _make_analysis_result,
    _make_day_session,
    _make_device,
    _make_profile,
)


async def _make_breath(
    db: AsyncSession,
    ar: AnalysisResult,
    session: Session,
    breath_number: int,
    inspiration_time_s: float = 1.2,
    i_e_ratio: float = 0.4,
    leak_valid: bool = True,
) -> Breath:
    start_offset = float(breath_number) * 5.0
    breath = Breath(
        analysis_result_id=ar.id,
        session_id=session.id,
        breath_number=breath_number,
        start_offset_s=start_offset,
        end_offset_s=start_offset + 4.0,
        inspiration_time_s=inspiration_time_s,
        i_e_ratio=i_e_ratio,
        leak_valid=leak_valid,
    )
    db.add(breath)
    await db.flush()
    return breath


# ---------------------------------------------------------------------------
# TestDurationZeroReason
# ---------------------------------------------------------------------------


class TestDurationZeroReason:
    async def test_rera_index_and_rdi_null_with_duration_zero_reason(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """rera_index and rdi are null with reason 'duration_zero' when analysis is
        present but Day.total_therapy_hours == 0 (cannot divide RERA count by hours).
        """
        from snore.mcp.tools.summary import get_nightly_summary

        target_date = date(2024, 3, 10)
        device = await _make_device(async_db_session, async_test_profile.id)

        # total_therapy_hours=0 is the trigger: analysis is present so rera_count=0
        # (not None), but dividing by 0 is disallowed → DURATION_ZERO.
        day, sess = await _make_day_session(
            async_db_session, device, target_date, duration_hours=0.0, ahi=3.5
        )
        ar = await _make_analysis_result(async_db_session, sess)
        await _make_breath(async_db_session, ar, sess, breath_number=1)

        result = await get_nightly_summary(
            async_db_session,
            target_date,
            target_date,
            profile_id=async_test_profile.id,
        )

        assert len(result.nights) == 1
        night = result.nights[0]
        assert night.rera_index is None
        assert night.rera_index_reason == "duration_zero"
        assert night.rdi is None
        assert night.rdi_reason == "duration_zero"

    async def test_rera_index_computed_when_therapy_hours_nonzero(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Positive total_therapy_hours with analysis produces rera_index (not null)."""
        from snore.mcp.tools.summary import get_nightly_summary

        target_date = date(2024, 3, 11)
        device = await _make_device(async_db_session, async_test_profile.id)
        day, sess = await _make_day_session(
            async_db_session,
            device,
            target_date,
            duration_hours=8.0,
            ahi=4.0,
            ahi_computed=4.0,
        )
        ar = await _make_analysis_result(async_db_session, sess)
        await _make_breath(async_db_session, ar, sess, breath_number=1)

        result = await get_nightly_summary(
            async_db_session,
            target_date,
            target_date,
            profile_id=async_test_profile.id,
        )

        assert len(result.nights) == 1
        night = result.nights[0]
        # rera_count=0 / 8h = 0.0; no DURATION_ZERO
        assert night.rera_index is not None
        assert night.rera_index_reason is None
        assert night.rdi is not None
        assert night.rdi_reason is None
        assert night.rera_proxy_version == "v2"


# ---------------------------------------------------------------------------
# TestEmptyRangeCompliance
# ---------------------------------------------------------------------------


class TestEmptyRangeCompliance:
    async def test_compliance_block_present_when_range_has_no_data(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """In range mode (start != end) with zero matching day rows, the compliance
        block is still included (previously it was omitted, causing an inconsistent
        response shape).
        """
        from snore.mcp.tools.summary import get_nightly_summary

        # No data seeded for async_test_profile — the range is entirely empty.
        result = await get_nightly_summary(
            async_db_session,
            date(2024, 4, 1),
            date(2024, 4, 10),
            compliance_threshold_hours=4.0,
            profile_id=async_test_profile.id,
        )

        assert result.nights == []
        assert result.total_nights == 0
        # Compliance block must be present even with no data
        assert result.compliance is not None
        # 10 calendar nights in the range (Apr 1–10 inclusive)
        assert result.compliance.days_total == 10
        assert result.compliance.days_compliant == 0
        assert result.compliance.compliance_pct == 0.0

    async def test_compliance_block_absent_on_single_night_query(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Single-date queries (start == end) with no data must NOT include compliance."""
        from snore.mcp.tools.summary import get_nightly_summary

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 4, 15),
            date(2024, 4, 15),
            profile_id=async_test_profile.id,
        )

        assert result.nights == []
        # Compliance is not meaningful for a single point
        assert result.compliance is None


# ---------------------------------------------------------------------------
# TestMaxEventsTruncation
# ---------------------------------------------------------------------------


class TestMaxEventsTruncation:
    async def test_max_events_truncates_list_but_preserves_total_count(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """get_events with max_events=2 on a 4-event night returns 2 events,
        total_events=4, and truncated=True.
        """
        from snore.mcp.tools.events import get_events

        target_date = date(2024, 6, 20)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)

        for i, ev_type in enumerate(["OA", "CA", "H", "OA"]):
            async_db_session.add(
                Event(
                    session_id=sess.id,
                    event_type=ev_type,
                    start_time=sess.start_time + timedelta(minutes=10 + i * 5),
                    duration_seconds=15.0,
                )
            )
        await async_db_session.flush()

        result = await get_events(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            max_events=2,
        )

        assert result.total_events == 4
        assert len(result.events) == 2
        assert result.truncated is True

    async def test_max_events_no_truncation_when_count_below_limit(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """When total events are within max_events, truncated is False and all events
        are returned.
        """
        from snore.mcp.tools.events import get_events

        target_date = date(2024, 6, 21)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)

        for i, ev_type in enumerate(["OA", "CA", "H"]):
            async_db_session.add(
                Event(
                    session_id=sess.id,
                    event_type=ev_type,
                    start_time=sess.start_time + timedelta(minutes=10 + i * 5),
                    duration_seconds=15.0,
                )
            )
        await async_db_session.flush()

        result = await get_events(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            max_events=500,
        )

        assert result.total_events == 3
        assert len(result.events) == 3
        assert result.truncated is False

    def test_validate_max_events_below_one_raises(self) -> None:
        """validate_max_events raises ValidationError with the expected message."""
        from snore.mcp.errors import ValidationError
        from snore.mcp.validation import validate_max_events

        with pytest.raises(ValidationError, match="max_events must be >= 1"):
            validate_max_events(0)

        with pytest.raises(ValidationError, match="max_events must be >= 1"):
            validate_max_events(-5)


# ---------------------------------------------------------------------------
# TestNotOwnedDeviceMessage
# ---------------------------------------------------------------------------


class TestNotOwnedDeviceMessage:
    async def test_get_events_not_owned_device_hides_profile_id(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """DeviceNotOwnedError in get_events is re-raised as a ValidationError whose
        message names the device_id but does NOT contain 'profile', preventing the
        internal profile id from leaking to the client.
        """
        from snore.mcp.errors import ValidationError
        from snore.mcp.tools.events import get_events

        profile_b = await _make_profile(async_db_session)
        device_b = await _make_device(async_db_session, profile_b.id)

        with pytest.raises(ValidationError) as exc_info:
            await get_events(
                async_db_session,
                date(2024, 7, 1),
                profile_id=async_test_profile.id,
                device_id=device_b.id,
            )

        message = str(exc_info.value)
        assert f"device_id={device_b.id}" in message
        assert "is not available in this session" in message
        assert "profile" not in message.lower()

    async def test_get_nightly_summary_not_owned_device_hides_profile_id(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """DeviceNotOwnedError in get_nightly_summary maps to a client-safe message."""
        from snore.mcp.errors import ValidationError
        from snore.mcp.tools.summary import get_nightly_summary

        profile_b = await _make_profile(async_db_session)
        device_b = await _make_device(async_db_session, profile_b.id)

        with pytest.raises(ValidationError) as exc_info:
            await get_nightly_summary(
                async_db_session,
                date(2024, 7, 1),
                date(2024, 7, 7),
                profile_id=async_test_profile.id,
                device_id=device_b.id,
            )

        message = str(exc_info.value)
        assert f"device_id={device_b.id}" in message
        assert "is not available in this session" in message
        assert "profile" not in message.lower()


# ---------------------------------------------------------------------------
# TestCapabilitiesIdentityFields
# ---------------------------------------------------------------------------


class TestCapabilitiesIdentityFields:
    async def test_nightly_summary_device_capabilities_has_identity_fields(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """device_capabilities in NightlySummaryResponse includes manufacturer, model,
        and serial_number matching the seeded Device row.
        """
        from snore.mcp.tools.summary import get_nightly_summary

        target_date = date(2024, 9, 20)
        device = await _make_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="AcmeCPAP",
            model="AirSense 11",
            serial_number="SN-ACME-20240920",
        )
        await _make_day_session(async_db_session, device, target_date)

        result = await get_nightly_summary(
            async_db_session,
            target_date,
            target_date,
            profile_id=async_test_profile.id,
        )

        caps = result.device_capabilities
        assert caps is not None
        assert caps.manufacturer == "AcmeCPAP"
        assert caps.model == "AirSense 11"
        assert caps.serial_number == "SN-ACME-20240920"

    async def test_events_device_capabilities_has_identity_fields(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """device_capabilities in EventsResponse includes identity fields from the
        owned Device row.
        """
        from snore.mcp.tools.events import get_events

        target_date = date(2024, 9, 21)
        device = await _make_device(
            async_db_session,
            async_test_profile.id,
            manufacturer="PhilipsRespironics",
            model="DreamStation 2",
            serial_number="SN-PH-20240921",
        )
        _, sess = await _make_day_session(async_db_session, device, target_date)
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=sess.start_time + timedelta(minutes=20),
                duration_seconds=12.0,
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session, target_date, profile_id=async_test_profile.id
        )

        caps = result.device_capabilities
        assert caps is not None
        assert caps.manufacturer == "PhilipsRespironics"
        assert caps.model == "DreamStation 2"
        assert caps.serial_number == "SN-PH-20240921"
