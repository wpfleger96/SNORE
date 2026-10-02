"""Adversarial profile-isolation and new-behavior integration tests for SNORE MCP tools.

Covers:
- Two-profile adversarial isolation: each tool returns only the caller's profile data.
- Device ambiguity: two devices on the same date → ValidationError, not silent fallback.
- Compliance denominator: days_total is calendar nights, not data nights.
- RDI/Ti/IE: populated when analysis present, null+reason when absent.
- Capability blocks: nightly/events responses carry device_capabilities when resolved.
- Multi-session night: per-event anchors correct; response-level anchors null.
"""

from __future__ import annotations

import uuid

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import (
    AnalysisResult,
    Breath,
    Day,
    Event,
    Session,
    Setting,
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
    """Create a Breath row with leak_valid breaths for Ti/IE aggregation."""
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
# TestProfileIsolation — two-profile adversarial tests
# ---------------------------------------------------------------------------


class TestProfileIsolation:
    """Assert that each tool returns only data belonging to the calling profile.

    Seeds two independent profiles (A = async_test_profile, B = fresh profile)
    each with their own device, sessions, events, waveforms, and settings.
    Every assertion checks both that A's data is present for A's caller AND
    that B's data is absent (not just that some data exists).
    """

    async def test_overview_shows_only_own_devices_and_event_types(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.overview import get_data_overview

        profile_a = async_test_profile
        profile_b = await _make_profile(async_db_session)

        target_date = date(2024, 9, 1)

        device_a = await _make_device(
            async_db_session, profile_a.id, manufacturer="MfrA"
        )
        _, sess_a = await _make_day_session(async_db_session, device_a, target_date)
        async_db_session.add(
            Event(
                session_id=sess_a.id,
                event_type="OA",
                start_time=sess_a.start_time + timedelta(minutes=10),
                duration_seconds=15.0,
            )
        )

        device_b = await _make_device(
            async_db_session, profile_b.id, manufacturer="MfrB"
        )
        _, sess_b = await _make_day_session(async_db_session, device_b, target_date)
        async_db_session.add(
            Event(
                session_id=sess_b.id,
                event_type="CA",
                start_time=sess_b.start_time + timedelta(minutes=10),
                duration_seconds=15.0,
            )
        )
        await async_db_session.flush()

        result_a = await get_data_overview(async_db_session, profile_id=profile_a.id)
        result_b = await get_data_overview(async_db_session, profile_id=profile_b.id)

        # Profile A sees only its device and OA event type
        assert len(result_a.devices) == 1
        assert result_a.devices[0].manufacturer == "MfrA"
        assert "OA" in result_a.available_event_types
        assert "CA" not in result_a.available_event_types
        assert result_a.total_sessions == 1

        # Profile B sees only its device and CA event type
        assert len(result_b.devices) == 1
        assert result_b.devices[0].manufacturer == "MfrB"
        assert "CA" in result_b.available_event_types
        assert "OA" not in result_b.available_event_types
        assert result_b.total_sessions == 1

    async def test_nightly_summary_shows_only_own_nights(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.summary import get_nightly_summary

        profile_a = async_test_profile
        profile_b = await _make_profile(async_db_session)

        target_date = date(2024, 9, 5)

        device_a = await _make_device(async_db_session, profile_a.id)
        await _make_day_session(async_db_session, device_a, target_date, ahi=5.2)

        device_b = await _make_device(async_db_session, profile_b.id)
        await _make_day_session(async_db_session, device_b, target_date, ahi=9.9)

        result_a = await get_nightly_summary(
            async_db_session, target_date, target_date, profile_id=profile_a.id
        )
        result_b = await get_nightly_summary(
            async_db_session, target_date, target_date, profile_id=profile_b.id
        )

        # A sees one night with A's AHI; B's AHI does not appear
        assert len(result_a.nights) == 1
        assert result_a.nights[0].ahi == pytest.approx(5.2, abs=0.01)

        # B sees one night with B's AHI; A's AHI does not appear
        assert len(result_b.nights) == 1
        assert result_b.nights[0].ahi == pytest.approx(9.9, abs=0.01)

    async def test_events_shows_only_own_events(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.events import get_events

        profile_a = async_test_profile
        profile_b = await _make_profile(async_db_session)

        target_date = date(2024, 9, 10)

        device_a = await _make_device(async_db_session, profile_a.id)
        _, sess_a = await _make_day_session(async_db_session, device_a, target_date)
        async_db_session.add(
            Event(
                session_id=sess_a.id,
                event_type="OA",
                start_time=sess_a.start_time + timedelta(minutes=20),
                duration_seconds=15.0,
            )
        )

        device_b = await _make_device(async_db_session, profile_b.id)
        _, sess_b = await _make_day_session(async_db_session, device_b, target_date)
        async_db_session.add(
            Event(
                session_id=sess_b.id,
                event_type="CA",
                start_time=sess_b.start_time + timedelta(minutes=20),
                duration_seconds=15.0,
            )
        )
        await async_db_session.flush()

        result_a = await get_events(
            async_db_session, target_date, profile_id=profile_a.id
        )
        result_b = await get_events(
            async_db_session, target_date, profile_id=profile_b.id
        )

        # A's caller sees only OA
        assert result_a.total_events == 1
        assert result_a.events[0].event_type == "OA"
        assert all(e.event_type != "CA" for e in result_a.events)

        # B's caller sees only CA
        assert result_b.total_events == 1
        assert result_b.events[0].event_type == "CA"
        assert all(e.event_type != "OA" for e in result_b.events)

    async def test_settings_timeline_shows_only_own_epochs(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.settings import get_settings_timeline

        profile_a = async_test_profile
        profile_b = await _make_profile(async_db_session)

        target_date = date(2024, 9, 15)

        device_a = await _make_device(async_db_session, profile_a.id)
        _, sess_a = await _make_day_session(async_db_session, device_a, target_date)
        async_db_session.add(Setting(session_id=sess_a.id, key="mode", value="AutoSet"))

        device_b = await _make_device(async_db_session, profile_b.id)
        _, sess_b = await _make_day_session(async_db_session, device_b, target_date)
        async_db_session.add(
            Setting(session_id=sess_b.id, key="mode", value="CPAP_Distinct")
        )
        await async_db_session.flush()

        result_a = await get_settings_timeline(
            async_db_session,
            date(2024, 9, 1),
            date(2024, 9, 30),
            profile_id=profile_a.id,
        )
        result_b = await get_settings_timeline(
            async_db_session,
            date(2024, 9, 1),
            date(2024, 9, 30),
            profile_id=profile_b.id,
        )

        # A's caller: at least one epoch with mode=AutoSet, no CPAP_Distinct
        a_modes = [ep.settings.get("mode") for ep in result_a.epochs]
        assert "AutoSet" in a_modes
        assert "CPAP_Distinct" not in a_modes

        # B's caller: at least one epoch with mode=CPAP_Distinct, no AutoSet
        b_modes = [ep.settings.get("mode") for ep in result_b.epochs]
        assert "CPAP_Distinct" in b_modes
        assert "AutoSet" not in b_modes

    async def test_settings_changes_shows_only_own_mask_log_entries(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.database.models import MaskLogEntry
        from snore.mcp.tools.changes import get_settings_changes

        profile_a = async_test_profile
        profile_b = await _make_profile(async_db_session)

        async_db_session.add(
            MaskLogEntry(
                profile_id=profile_a.id,
                brand="BrandA",
                model="ModelA",
                size="M",
                style="nasal",
                start_date=date(2024, 9, 20),
            )
        )
        async_db_session.add(
            MaskLogEntry(
                profile_id=profile_b.id,
                brand="BrandB",
                model="ModelB",
                size="L",
                style="full_face",
                start_date=date(2024, 9, 21),
            )
        )
        await async_db_session.flush()

        result_a = await get_settings_changes(
            async_db_session,
            date(2024, 9, 1),
            date(2024, 9, 30),
            profile_id=profile_a.id,
        )
        result_b = await get_settings_changes(
            async_db_session,
            date(2024, 9, 1),
            date(2024, 9, 30),
            profile_id=profile_b.id,
        )

        # A sees only its own mask entry; B's never appears
        a_values = [c.new_value for c in result_a.changes]
        assert a_values == ["BrandA ModelA (M)"]
        assert all("BrandB" not in (v or "") for v in a_values)

        # B sees only its own mask entry; A's never appears
        b_values = [c.new_value for c in result_b.changes]
        assert b_values == ["BrandB ModelB (L)"]


# ---------------------------------------------------------------------------
# TestDeviceAmbiguity — two devices same date, no device_id filter
# ---------------------------------------------------------------------------


class TestDeviceAmbiguity:
    async def test_two_devices_same_date_raises_validation_error(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Two devices with sessions on the same date and no device_id filter → ValidationError."""
        from snore.mcp.errors import ValidationError
        from snore.mcp.tools.summary import get_nightly_summary

        target_date = date(2024, 10, 1)

        device_1 = await _make_device(async_db_session, async_test_profile.id, "Mfr1")
        device_2 = await _make_device(async_db_session, async_test_profile.id, "Mfr2")
        await _make_day_session(async_db_session, device_1, target_date)
        await _make_day_session(async_db_session, device_2, target_date)

        with pytest.raises(ValidationError):
            await get_nightly_summary(
                async_db_session,
                target_date,
                target_date,
                profile_id=async_test_profile.id,
                # no device_id — should trigger DeviceAmbiguityError → ValidationError
            )


# ---------------------------------------------------------------------------
# TestComplianceDenominator — calendar nights vs data nights
# ---------------------------------------------------------------------------


class TestComplianceDenominator:
    async def test_compliance_uses_calendar_nights_not_data_nights(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """compliance.days_total equals calendar nights in range, not nights with data.

        Range: 7 calendar nights (Nov 1–7).
        Data on: Nov 1, 3, 5 only (3 nights with sessions).
        Expected: days_total == 7, days_compliant == 3 (all data nights >= 4 h).
        """
        from snore.mcp.tools.summary import get_nightly_summary

        device = await _make_device(async_db_session, async_test_profile.id)
        data_dates = [date(2024, 11, 1), date(2024, 11, 3), date(2024, 11, 5)]
        for d in data_dates:
            await _make_day_session(async_db_session, device, d, duration_hours=8.0)

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 11, 1),
            date(2024, 11, 7),
            compliance_threshold_hours=4.0,
            profile_id=async_test_profile.id,
        )

        assert result.compliance is not None
        # Calendar denominator: 7 nights in Nov 1–7 inclusive
        assert result.compliance.days_total == 7
        # Only 3 nights have data, all compliant
        assert result.compliance.days_compliant == 3
        # compliance_pct = round(3/7 * 100, 1) = 42.9
        assert result.compliance.compliance_pct == pytest.approx(
            round(3 / 7 * 100, 1), abs=0.05
        )

    async def test_compliance_pct_consistent_with_denominator(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """compliance_pct == round(days_compliant / days_total * 100, 1)."""
        from snore.mcp.tools.summary import get_nightly_summary

        device = await _make_device(async_db_session, async_test_profile.id)
        # 4 compliant nights in a 10-night range
        for i in range(4):
            await _make_day_session(
                async_db_session,
                device,
                date(2024, 12, 1) + timedelta(days=i),
                duration_hours=6.0,
            )

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 12, 1),
            date(2024, 12, 10),
            compliance_threshold_hours=4.0,
            profile_id=async_test_profile.id,
        )

        assert result.compliance is not None
        expected_pct = round(
            result.compliance.days_compliant / result.compliance.days_total * 100, 1
        )
        assert result.compliance.compliance_pct == pytest.approx(expected_pct, abs=0.05)


# ---------------------------------------------------------------------------
# TestRdiTiIe — analysis-derived fields
# ---------------------------------------------------------------------------


class TestRdiTiIe:
    async def test_rdi_ti_ie_populated_when_analysis_present(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """rdi, ti_median_s, and ie_ratio are non-null when valid analysis exists.

        Seeds: day with ahi=5.0, AnalysisResult with current identity,
        two leak-valid breaths with inspiration_time_s=1.2 and i_e_ratio=0.4.

        With no recovery breaths the RERA count is 0, so:
        rera_index = round(0 / 8.0, 2) = 0.0
        rdi        = round(ahi_computed 5.0 + 0.0, 2) = 5.0

        ti_median_s = median([1.2, 1.2]) = 1.2
        ie_ratio    = median([0.4, 0.4]) = 0.4
        """
        from snore.mcp.tools.summary import get_nightly_summary

        target_date = date(2024, 7, 1)
        device = await _make_device(async_db_session, async_test_profile.id)
        day, sess = await _make_day_session(
            async_db_session,
            device,
            target_date,
            ahi=5.0,
            ahi_computed=5.0,
            duration_hours=8.0,
        )
        ar = await _make_analysis_result(async_db_session, sess)
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=1,
            inspiration_time_s=1.2,
            i_e_ratio=0.4,
            leak_valid=True,
        )
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=2,
            inspiration_time_s=1.2,
            i_e_ratio=0.4,
            leak_valid=True,
        )

        result = await get_nightly_summary(
            async_db_session,
            target_date,
            target_date,
            profile_id=async_test_profile.id,
        )

        assert len(result.nights) == 1
        night = result.nights[0]

        assert night.rdi is not None
        assert night.rdi_reason is None
        # Arithmetic: rdi == round(ahi_computed + rera_index, 2), rera_index == 0.0
        assert night.rdi == pytest.approx(
            round((night.ahi_computed or 0.0) + 0.0, 2), abs=0.01
        )

        assert night.ti_median_s is not None
        assert night.ti_median_reason is None
        assert night.ti_median_s == pytest.approx(1.2, abs=0.01)

        assert night.ie_ratio is not None
        assert night.ie_ratio_reason is None
        assert night.ie_ratio == pytest.approx(0.4, abs=0.01)

    async def test_rdi_ti_ie_null_when_analysis_absent(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """rdi, ti_median_s, ie_ratio are null with appropriate reasons when no analysis exists."""
        from snore.mcp.tools.summary import get_nightly_summary

        target_date = date(2024, 7, 5)
        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(
            async_db_session,
            device,
            target_date,
            ahi=5.0,
            ahi_computed=5.0,
            duration_hours=8.0,
        )
        # No AnalysisResult seeded

        result = await get_nightly_summary(
            async_db_session,
            target_date,
            target_date,
            profile_id=async_test_profile.id,
        )

        assert len(result.nights) == 1
        night = result.nights[0]

        assert night.rdi is None
        assert night.rdi_reason is not None
        assert night.ti_median_s is None
        assert night.ti_median_reason is not None
        assert night.ie_ratio is None
        assert night.ie_ratio_reason is not None


# ---------------------------------------------------------------------------
# TestCapabilityBlocks — device_capabilities on tool responses
# ---------------------------------------------------------------------------


class TestCapabilityBlocks:
    async def test_nightly_summary_has_device_capabilities_block(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """NightlySummaryResponse carries a device_capabilities block for the resolved device."""
        from snore.mcp.tools.summary import get_nightly_summary

        target_date = date(2024, 6, 1)
        device = await _make_device(
            async_db_session, async_test_profile.id, manufacturer="CapMfr"
        )
        await _make_day_session(async_db_session, device, target_date)

        result = await get_nightly_summary(
            async_db_session,
            target_date,
            target_date,
            profile_id=async_test_profile.id,
        )

        assert result.device_capabilities is not None
        assert result.device_capabilities.manufacturer == "CapMfr"

    async def test_events_response_has_device_capabilities_block(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """EventsResponse carries a device_capabilities block when events are returned."""
        from snore.mcp.tools.events import get_events

        target_date = date(2024, 6, 5)
        device = await _make_device(
            async_db_session, async_test_profile.id, manufacturer="EvtCapMfr"
        )
        _, sess = await _make_day_session(async_db_session, device, target_date)
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=sess.start_time + timedelta(minutes=15),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session, target_date, profile_id=async_test_profile.id
        )

        assert result.device_capabilities is not None
        assert result.device_capabilities.manufacturer == "EvtCapMfr"

    async def test_device_capabilities_none_for_unowned_device(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """build_device_capabilities returns None when device is not owned by the profile."""
        from snore.mcp.tools._capabilities import build_device_capabilities

        profile_b = await _make_profile(async_db_session)
        device_b = await _make_device(async_db_session, profile_b.id)

        # Profile A tries to get capabilities for profile B's device
        caps = await build_device_capabilities(
            async_db_session,
            async_test_profile.id,
            device_b.id,
        )
        assert caps is None


# ---------------------------------------------------------------------------
# TestMultiSessionNight — response-level anchors null when events span sessions
# ---------------------------------------------------------------------------


class TestMultiSessionNight:
    async def test_multi_session_events_have_per_event_anchors_and_null_response_anchors(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Two sessions on the same night: per-event session anchors are populated;
        response-level session_id and session_start_wall_clock are null.
        """
        from snore.mcp.tools.events import get_events

        target_date = date(2024, 8, 22)
        device = await _make_device(async_db_session, async_test_profile.id)

        # First session: starts at 22:00
        day = Day(
            device_id=device.id,
            date=target_date,
            total_therapy_hours=10.0,
        )
        async_db_session.add(day)
        await async_db_session.flush()

        sess1 = Session(
            device_id=device.id,
            day_id=day.id,
            device_session_id=f"multi_s1_{uuid.uuid4().hex[:6]}",
            start_time=datetime(
                target_date.year, target_date.month, target_date.day, 22, 0, 0
            ),
            end_time=datetime(
                target_date.year, target_date.month, target_date.day, 22, 0, 0
            )
            + timedelta(hours=4),
            duration_seconds=4 * 3600,
            enabled=True,
        )
        async_db_session.add(sess1)
        await async_db_session.flush()

        # Second session: starts 1 hour after the first ends
        sess2 = Session(
            device_id=device.id,
            day_id=day.id,
            device_session_id=f"multi_s2_{uuid.uuid4().hex[:6]}",
            start_time=datetime(
                target_date.year, target_date.month, target_date.day, 22, 0, 0
            )
            + timedelta(hours=5),
            end_time=datetime(
                target_date.year, target_date.month, target_date.day, 22, 0, 0
            )
            + timedelta(hours=10),
            duration_seconds=5 * 3600,
            enabled=True,
        )
        async_db_session.add(sess2)
        await async_db_session.flush()

        # One event in each session
        async_db_session.add(
            Event(
                session_id=sess1.id,
                event_type="OA",
                start_time=sess1.start_time + timedelta(minutes=30),
                duration_seconds=10.0,
            )
        )
        async_db_session.add(
            Event(
                session_id=sess2.id,
                event_type="CA",
                start_time=sess2.start_time + timedelta(minutes=30),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session, target_date, profile_id=async_test_profile.id
        )

        assert result.total_events == 2

        # Response-level anchors are null when events span multiple sessions
        assert result.session_id is None
        assert result.session_start_wall_clock is None

        # Per-event anchors are always populated
        for ev in result.events:
            assert ev.session_id is not None
            assert ev.session_start_wall_clock is not None

        # The two events belong to different sessions
        event_session_ids = {ev.session_id for ev in result.events}
        assert len(event_session_ids) == 2
        assert sess1.id in event_session_ids
        assert sess2.id in event_session_ids
