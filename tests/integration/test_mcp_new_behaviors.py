"""Integration tests for new MCP tool behaviors added in the mcp-skeleton review cycle.

Covers:
- rera_index / rdi reason="duration_zero" when the analyzed sessions have zero
  mask-on hours but analysis IS present (NullReason.DURATION_ZERO branch).
- rera_index divides by the mask-on hours of analyzed (OK, enabled) sessions only.
- Compliance block present even on empty range-mode responses (no day rows).
- get_events max_events truncation: total_events keeps untruncated count; truncated=True
  when cut; validate_max_events raises ValidationError for max_events < 1.
- DeviceNotOwnedError maps to a client-safe error message that does NOT leak the
  internal profile id ("profile" must not appear in the message).
- DeviceCapabilities identity fields (manufacturer, model, serial_number) populated
  from the owned Device row.
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
    Event,
    Session,
    Statistics,
)
from snore.services.breath_service import BreathService, NoSessionsInRangeError
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
        present but the analyzed sessions last 0 h (cannot divide RERA count by hours).
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
# TestReraIndexAnalyzedHours
# ---------------------------------------------------------------------------


async def _add_session(
    db: AsyncSession,
    device: Any,
    day: Any,
    *,
    start: datetime,
    duration_hours: float,
    enabled: bool = True,
) -> Session:
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"rera_{uuid.uuid4().hex[:8]}",
        start_time=start,
        end_time=start + timedelta(hours=duration_hours),
        duration_seconds=duration_hours * 3600,
        enabled=enabled,
    )
    db.add(sess)
    await db.flush()
    return sess


async def _seed_rera_proxies(
    db: AsyncSession, ar: AnalysisResult, session: Session, count: int
) -> None:
    """Seed ``count`` RERA proxies: two FL breaths, then a recovery breath."""
    for n in range(count * 3):
        is_recovery = n % 3 == 2
        start_offset = float(n) * 5.0
        db.add(
            Breath(
                analysis_result_id=ar.id,
                session_id=session.id,
                breath_number=n,
                start_offset_s=start_offset,
                end_offset_s=start_offset + 4.0,
                leak_valid=True,
                flow_class=1 if is_recovery else 5,
                is_recovery_breath=is_recovery,
            )
        )
    await db.flush()


class TestReraIndexAnalyzedHours:
    async def test_unanalyzed_session_hours_excluded_from_denominator(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """20 RERAs over the one analyzed 4 h session → 5.0/h, not 20 / 8 h."""
        from snore.mcp.tools.summary import get_nightly_summary

        target_date = date(2024, 3, 12)
        device = await _make_device(async_db_session, async_test_profile.id)
        day, analyzed = await _make_day_session(
            async_db_session, device, target_date, duration_hours=4.0
        )
        day.total_therapy_hours = 8.0
        await _add_session(
            async_db_session,
            device,
            day,
            start=analyzed.start_time + timedelta(hours=4),
            duration_hours=4.0,
        )
        ar = await _make_analysis_result(async_db_session, analyzed)
        await _seed_rera_proxies(async_db_session, ar, analyzed, count=20)

        result = await get_nightly_summary(
            async_db_session,
            target_date,
            target_date,
            profile_id=async_test_profile.id,
        )

        night = result.nights[0]
        assert night.rera_index == pytest.approx(5.0)
        assert night.usage_hours == pytest.approx(8.0)

    async def test_rera_index_divides_by_usage_hours_not_span(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """12 RERAs over a 4 h span with 3 h mask-on → 4.0/h, and rdi adds the
        mask-on day_ahi to that same per-mask-on-hour rate."""
        target_date = date(2024, 3, 15)
        device = await _make_device(async_db_session, async_test_profile.id)
        day, sess = await _make_day_session(
            async_db_session, device, target_date, duration_hours=4.0, ahi_computed=2.0
        )
        day.total_therapy_hours = 3.0
        async_db_session.add(Statistics(session_id=sess.id, usage_hours=3.0))
        ar = await _make_analysis_result(async_db_session, sess)
        await _seed_rera_proxies(async_db_session, ar, sess, count=12)

        night = await BreathService(
            async_db_session, profile_id=async_test_profile.id
        ).get_nightly_summary(target_date)

        assert night.rera_count == 12
        assert night.rera_index == pytest.approx(4.0)
        assert night.rdi == pytest.approx(6.0)

    @pytest.mark.parametrize("explicit_device", [False, True])
    async def test_disabled_session_with_analysis_contributes_nothing(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Any,
        explicit_device: bool,
    ) -> None:
        """A disabled session's RERAs and hours stay out of rera_index, whether
        the device is auto-selected or passed explicitly."""
        target_date = date(2024, 3, 13)
        device = await _make_device(async_db_session, async_test_profile.id)
        day, enabled = await _make_day_session(
            async_db_session, device, target_date, duration_hours=4.0
        )
        disabled = await _add_session(
            async_db_session,
            device,
            day,
            start=enabled.start_time + timedelta(hours=4),
            duration_hours=4.0,
            enabled=False,
        )
        enabled_ar = await _make_analysis_result(async_db_session, enabled)
        await _seed_rera_proxies(async_db_session, enabled_ar, enabled, count=4)
        disabled_ar = await _make_analysis_result(async_db_session, disabled)
        await _seed_rera_proxies(async_db_session, disabled_ar, disabled, count=20)

        night = await BreathService(
            async_db_session, profile_id=async_test_profile.id
        ).get_nightly_summary(
            target_date, device_id=device.id if explicit_device else None
        )

        assert night.rera_count == 4
        assert night.rera_index == pytest.approx(1.0)
        assert [c.session_id for c in night.session_coverage] == [enabled.id]

    async def test_all_disabled_range_raises_no_sessions(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A date whose only session is disabled has no sessions to summarise."""
        target_date = date(2024, 3, 14)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        sess.enabled = False
        await async_db_session.flush()

        with pytest.raises(NoSessionsInRangeError):
            await BreathService(
                async_db_session, profile_id=async_test_profile.id
            ).get_nightly_summary(target_date)


# ---------------------------------------------------------------------------
# TestDisabledSessionsExcluded
# ---------------------------------------------------------------------------


class TestDisabledSessionsExcluded:
    async def test_breath_table_rejects_disabled_session_id(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """An explicit session_id of a disabled session errors, as get_waveform does."""
        from snore.mcp.errors import ValidationError
        from snore.mcp.tools.breath_table import get_breath_table

        target_date = date(2024, 4, 2)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)
        await _make_breath(async_db_session, ar, sess, breath_number=1)
        sess.enabled = False
        await async_db_session.flush()

        with pytest.raises(ValidationError, match="is disabled"):
            await get_breath_table(
                async_db_session,
                target_date,
                profile_id=async_test_profile.id,
                session_id=sess.id,
            )

    async def test_capabilities_count_only_enabled_sessions(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A disabled-only night and its events are absent from capabilities."""
        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(async_db_session, device, date(2024, 4, 3))
        _, dropped = await _make_day_session(async_db_session, device, date(2024, 4, 4))
        dropped.enabled = False
        async_db_session.add(
            Event(
                session_id=dropped.id,
                event_type="CA",
                start_time=dropped.start_time + timedelta(minutes=5),
                duration_seconds=12.0,
            )
        )
        await async_db_session.flush()

        caps = await BreathService(
            async_db_session, profile_id=async_test_profile.id
        ).get_device_capabilities(device.id)

        assert caps.session_count == 1
        assert caps.nights_with_data == 1
        assert caps.actual_date_end == date(2024, 4, 3)
        assert caps.event_types_present == []

    @pytest.mark.parametrize(
        "tool", ["summary", "events", "table", "windows", "epochs", "waveform"]
    )
    async def test_explicit_device_disabled_only_night_degrades_cleanly(
        self, async_db_session: AsyncSession, async_test_profile: Any, tool: str
    ) -> None:
        """With device_id given and the night's only session disabled, every tool
        returns an empty result with a reason or a mapped client error."""
        from snore.mcp.errors import ValidationError
        from snore.mcp.schemas import EpochSpec
        from snore.mcp.tools.breath_table import get_breath_table
        from snore.mcp.tools.epochs import compare_epochs
        from snore.mcp.tools.events import get_events
        from snore.mcp.tools.summary import get_nightly_summary
        from snore.mcp.tools.waveform import fetch_waveform_raw
        from snore.mcp.tools.windows import find_windows

        target_date = date(2024, 4, 5)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        await _make_analysis_result(async_db_session, sess)
        sess.enabled = False
        await async_db_session.flush()
        db, pid, dev_id = async_db_session, async_test_profile.id, device.id

        if tool == "summary":
            night = (
                await get_nightly_summary(
                    db, target_date, target_date, profile_id=pid, device_id=dev_id
                )
            ).nights[0]
            assert night.session_count == 0
            assert night.rera_index_reason == "analysis_not_run"
        elif tool == "events":
            events = await get_events(db, target_date, profile_id=pid, device_id=dev_id)
            assert events.events == []
        elif tool == "table":
            with pytest.raises(ValidationError, match="No therapy data found"):
                await get_breath_table(
                    db, target_date, profile_id=pid, device_id=dev_id
                )
        elif tool == "windows":
            windows = await find_windows(
                db,
                target_date,
                profile_id=pid,
                criterion="ca_centered",
                device_id=dev_id,
            )
            assert windows.windows == []
            assert windows.null_reason == "analysis_not_run"
        elif tool == "epochs":
            spec = EpochSpec(
                label="e",
                date_start=target_date.isoformat(),
                date_end=target_date.isoformat(),
                device_id=dev_id,
            )
            epochs = await compare_epochs(db, pid, [spec, spec.model_copy()])
            assert {e.null_reason for e in epochs.epochs} == {"no_data_in_range"}
        else:
            window = await fetch_waveform_raw(
                db,
                target_date,
                pid,
                0.0,
                60.0,
                device_id=dev_id,
                window_cap_seconds=120.0,
            )
            assert window.channels == []


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
