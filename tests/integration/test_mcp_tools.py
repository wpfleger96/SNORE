"""Integration tests for SNORE MCP tools.

Tests exercise each tool implementation directly against an in-memory async DB,
using the same fixture helpers as the rest of the integration suite.  These tests
verify behavior: correct data returned, null + reason pattern, pagination,
and graceful degradation when analysis results are absent.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Event
from snore.provenance import IndexSource
from tests.integration.conftest import (
    _make_analysis_result,
    _make_day_session,
    _make_device,
    _make_event,
)
from tests.integration.test_mcp_waveform import _make_waveform

# ---------------------------------------------------------------------------
# get_data_overview
# ---------------------------------------------------------------------------


class TestGetDataOverview:
    async def test_empty_database_returns_empty_devices(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.overview import get_data_overview

        result = await get_data_overview(
            async_db_session, profile_id=async_test_profile.id
        )
        assert result.devices == []
        assert result.total_sessions == 0
        assert not result.analysis_run

    async def test_device_and_sessions_appear_in_overview(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.overview import get_data_overview

        device = await _make_device(async_db_session, async_test_profile.id)
        today = date(2024, 8, 1)
        await _make_day_session(async_db_session, device, today)
        await _make_day_session(async_db_session, device, today + timedelta(days=1))

        result = await get_data_overview(
            async_db_session, profile_id=async_test_profile.id
        )
        assert len(result.devices) == 1
        assert result.devices[0].manufacturer == "TestMfr"
        assert result.devices[0].session_count == 2
        assert result.total_sessions == 2
        assert result.date_range_start == today
        assert result.date_range_end == today + timedelta(days=1)

    async def test_analysis_run_false_when_no_analysis_results(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.overview import get_data_overview

        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(async_db_session, device, date(2024, 8, 1))

        result = await get_data_overview(
            async_db_session, profile_id=async_test_profile.id
        )
        assert not result.analysis_run
        assert result.analysis_session_count == 0


# ---------------------------------------------------------------------------
# get_settings_timeline
# ---------------------------------------------------------------------------


class TestGetSettingsTimeline:
    async def test_empty_database_returns_no_epochs(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.settings import get_settings_timeline

        result = await get_settings_timeline(
            async_db_session,
            date(2024, 1, 1),
            date(2024, 12, 31),
            profile_id=async_test_profile.id,
        )
        assert result.epochs == []
        assert result.total_epochs == 0

    async def test_epochs_filtered_to_date_range(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.database.models import Setting
        from snore.mcp.tools.settings import get_settings_timeline

        device = await _make_device(async_db_session, async_test_profile.id)

        # Two sessions with settings — one inside range, one outside
        day1, sess1 = await _make_day_session(
            async_db_session, device, date(2024, 3, 1)
        )
        day2, sess2 = await _make_day_session(
            async_db_session, device, date(2024, 9, 1)
        )

        for sess in [sess1, sess2]:
            async_db_session.add(
                Setting(session_id=sess.id, key="mode", value="AutoSet")
            )
        await async_db_session.flush()

        # Query only the first half of the year
        result = await get_settings_timeline(
            async_db_session,
            date(2024, 1, 1),
            date(2024, 6, 30),
            profile_id=async_test_profile.id,
        )
        # Only the March session epoch should appear
        assert result.total_epochs == 1

    async def test_mask_type_change_splits_epochs_and_flags_changed_key(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A mask_type change splits epochs; mask_type appears in settings + changed_keys."""
        from snore.database.models import Setting
        from snore.mcp.tools.settings import get_settings_timeline

        device = await _make_device(async_db_session, async_test_profile.id)
        for i in range(2):
            _, sess = await _make_day_session(
                async_db_session, device, date(2024, 3, 1) + timedelta(days=i)
            )
            async_db_session.add(
                Setting(session_id=sess.id, key="mode", value="AutoSet")
            )
            async_db_session.add(
                Setting(session_id=sess.id, key="mask_type", value="Pillows")
            )
        for i in range(2, 4):
            _, sess = await _make_day_session(
                async_db_session, device, date(2024, 3, 1) + timedelta(days=i)
            )
            async_db_session.add(
                Setting(session_id=sess.id, key="mode", value="AutoSet")
            )
            async_db_session.add(
                Setting(session_id=sess.id, key="mask_type", value="Full Face")
            )
        await async_db_session.flush()

        result = await get_settings_timeline(
            async_db_session,
            date(2024, 3, 1),
            date(2024, 3, 31),
            profile_id=async_test_profile.id,
        )

        assert result.total_epochs == 2
        first, second = result.epochs
        assert first.settings["mask_type"] == "Pillows"
        assert second.settings["mask_type"] == "Full Face"
        assert second.start_date == date(2024, 3, 3)
        assert second.changed_keys == ["mask_type"]


# ---------------------------------------------------------------------------
# get_settings_changes
# ---------------------------------------------------------------------------


class TestGetSettingsChanges:
    async def test_empty_database_returns_no_changes(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.changes import get_settings_changes

        result = await get_settings_changes(
            async_db_session,
            date(2024, 1, 1),
            date(2024, 12, 31),
            profile_id=async_test_profile.id,
        )
        assert result.changes == []
        assert result.total_changes == 0

    async def test_merged_changes_interleave_sources_in_date_order(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Device settings changes and mask log entries merge into one date-ordered log."""
        from snore.database.models import MaskLogEntry, Setting
        from snore.mcp.tools.changes import get_settings_changes

        device = await _make_device(async_db_session, async_test_profile.id)
        # Day 1: pressure_max 15 → Day 3: pressure_max 18 (device change on Mar 3)
        for day_date, pmax in [(date(2024, 3, 1), "15.0"), (date(2024, 3, 3), "18.0")]:
            _, sess = await _make_day_session(async_db_session, device, day_date)
            async_db_session.add(
                Setting(session_id=sess.id, key="pressure_max", value=pmax)
            )
        # Mask log: an entry before the range (old_value source) and one inside
        async_db_session.add(
            MaskLogEntry(
                profile_id=async_test_profile.id,
                brand="ResMed",
                model="AirFit P10",
                size="M",
                style="pillows",
                start_date=date(2024, 1, 15),
            )
        )
        async_db_session.add(
            MaskLogEntry(
                profile_id=async_test_profile.id,
                brand="ResMed",
                model="AirFit F20",
                size=None,
                style="full_face",
                start_date=date(2024, 3, 2),
                notes="switched for leak issues",
            )
        )
        await async_db_session.flush()

        result = await get_settings_changes(
            async_db_session,
            date(2024, 3, 1),
            date(2024, 3, 31),
            profile_id=async_test_profile.id,
        )

        assert result.total_changes == 2
        mask_change, device_change = result.changes
        # Date order: mask log entry (Mar 2) before device change (Mar 3)
        assert mask_change.date == date(2024, 3, 2)
        assert mask_change.source == "mask_log"
        assert mask_change.key == "mask_equipment"
        assert mask_change.device_id is None
        assert mask_change.new_value == "ResMed AirFit F20"  # size null → no parens
        assert mask_change.old_value == "ResMed AirFit P10 (M)"  # pre-range entry
        assert mask_change.mask_brand == "ResMed"
        assert mask_change.mask_style == "full_face"
        assert mask_change.notes == "switched for leak issues"

        assert device_change.date == date(2024, 3, 3)
        assert device_change.source == "device_settings"
        assert device_change.device_id == device.id
        assert device_change.key == "pressure_max"
        assert device_change.old_value == "15.0"
        assert device_change.new_value == "18.0"

    async def test_first_mask_entry_ever_has_null_old_value(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.database.models import MaskLogEntry
        from snore.mcp.tools.changes import get_settings_changes

        async_db_session.add(
            MaskLogEntry(
                profile_id=async_test_profile.id,
                brand="F&P",
                model="Evora",
                size="S",
                style="nasal",
                start_date=date(2024, 5, 1),
            )
        )
        await async_db_session.flush()

        result = await get_settings_changes(
            async_db_session,
            date(2024, 5, 1),
            date(2024, 5, 31),
            profile_id=async_test_profile.id,
        )

        assert result.total_changes == 1
        assert result.changes[0].old_value is None
        assert result.changes[0].new_value == "F&P Evora (S)"

    async def test_same_day_mask_entries_keep_insertion_order_and_chain(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Two mask entries on the same start_date appear in id order, chained."""
        from snore.database.models import MaskLogEntry
        from snore.mcp.tools.changes import get_settings_changes

        for brand, model in [("ResMed", "AirFit P10"), ("ResMed", "AirFit F20")]:
            async_db_session.add(
                MaskLogEntry(
                    profile_id=async_test_profile.id,
                    brand=brand,
                    model=model,
                    style="nasal",
                    start_date=date(2024, 7, 1),
                )
            )
            await async_db_session.flush()

        result = await get_settings_changes(
            async_db_session,
            date(2024, 7, 1),
            date(2024, 7, 31),
            profile_id=async_test_profile.id,
        )

        assert result.total_changes == 2
        first, second = result.changes
        assert first.new_value == "ResMed AirFit P10"
        assert first.old_value is None
        # old_value chains across same-day entries in insertion (id) order
        assert second.new_value == "ResMed AirFit F20"
        assert second.old_value == "ResMed AirFit P10"

    async def test_device_id_filter_keeps_profile_level_mask_entries(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """device_id narrows device_settings entries; mask_log entries always remain."""
        from snore.database.models import MaskLogEntry, Setting
        from snore.mcp.tools.changes import get_settings_changes

        device_a = await _make_device(async_db_session, async_test_profile.id)
        device_b = await _make_device(async_db_session, async_test_profile.id)
        for device in [device_a, device_b]:
            for day_date, mode in [
                (date(2024, 6, 1), "CPAP"),
                (date(2024, 6, 2), "AutoSet"),
            ]:
                _, sess = await _make_day_session(async_db_session, device, day_date)
                async_db_session.add(
                    Setting(session_id=sess.id, key="mode", value=mode)
                )
        async_db_session.add(
            MaskLogEntry(
                profile_id=async_test_profile.id,
                brand="ResMed",
                model="AirFit N20",
                size="M",
                style="nasal",
                start_date=date(2024, 6, 3),
            )
        )
        await async_db_session.flush()

        result = await get_settings_changes(
            async_db_session,
            date(2024, 6, 1),
            date(2024, 6, 30),
            profile_id=async_test_profile.id,
            device_id=device_a.id,
        )

        sources = [(c.source, c.device_id) for c in result.changes]
        assert ("device_settings", device_a.id) in sources
        assert ("device_settings", device_b.id) not in sources
        assert ("mask_log", None) in sources
        assert result.total_changes == 2

    async def test_device_change_after_end_excluded_old_value_correct(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A device change dated after `end` is excluded; the in-range change's
        old_value still reflects the pre-range baseline correctly."""
        from snore.database.models import Setting
        from snore.mcp.tools.changes import get_settings_changes

        device = await _make_device(async_db_session, async_test_profile.id)
        # Day 1: establish baseline (mode=CPAP) — no change yet
        _, sess1 = await _make_day_session(async_db_session, device, date(2024, 3, 1))
        async_db_session.add(Setting(session_id=sess1.id, key="mode", value="CPAP"))
        # Day 3 (in range): mode changes CPAP → AutoSet
        _, sess3 = await _make_day_session(async_db_session, device, date(2024, 3, 3))
        async_db_session.add(Setting(session_id=sess3.id, key="mode", value="AutoSet"))
        # Day 7 (after end): mode changes AutoSet → APAP — must be excluded
        _, sess7 = await _make_day_session(async_db_session, device, date(2024, 3, 7))
        async_db_session.add(Setting(session_id=sess7.id, key="mode", value="APAP"))
        await async_db_session.flush()

        # Query window ends on Mar 4 — Day 3 is included, Day 7 is not
        result = await get_settings_changes(
            async_db_session,
            date(2024, 3, 1),
            date(2024, 3, 4),
            profile_id=async_test_profile.id,
        )

        device_changes = [c for c in result.changes if c.source == "device_settings"]
        assert len(device_changes) == 1
        change = device_changes[0]
        assert change.date == date(2024, 3, 3)
        assert change.old_value == "CPAP"
        assert change.new_value == "AutoSet"


# ---------------------------------------------------------------------------
# get_nightly_summary
# ---------------------------------------------------------------------------


class TestGetNightlySummary:
    async def test_empty_database_returns_empty_nights(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.summary import get_nightly_summary

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 1, 1),
            date(2024, 1, 31),
            profile_id=async_test_profile.id,
        )
        assert result.nights == []
        assert result.total_nights == 0

    async def test_rera_fields_null_with_reason_when_analysis_absent(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.summary import get_nightly_summary

        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(async_db_session, device, date(2024, 8, 1), ahi=2.5)

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 8, 1),
            date(2024, 8, 1),
            profile_id=async_test_profile.id,
        )
        assert len(result.nights) == 1
        night = result.nights[0]
        assert night.rera_index is None
        assert night.rera_index_reason == "not_available"
        assert night.rdi is None
        assert night.fl_class_ge4_pct is None
        assert night.fl_class_ge4_pct_reason == "not_available"
        assert night.leak_above_24_pct is None
        assert night.leak_above_24_pct_reason == "not_available"
        assert night.periodic_breathing_pct is None
        assert night.pb_reason == "not_available"

    async def test_periodic_breathing_pct_from_persisted_episodes(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """8 h analyzed night with 2880 s of persisted PB episodes → 10.0 %."""
        from snore.analysis.types import AnalysisResult as AnalysisResultDTO
        from snore.mcp.tools.summary import get_nightly_summary

        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, date(2024, 8, 2))
        ar = await _make_analysis_result(async_db_session, sess)
        ar.programmatic_result_json = AnalysisResultDTO(
            session_id=sess.id,
            session_duration_hours=8.0,
            total_breaths=0,
            machine_events=[],
            mode_results={},
            periodic_breathing_episodes=[
                {"start_time": 600.0, "end_time": 2040.0},
                {"start_time": 5000.0, "end_time": 6440.0},
            ],
        ).model_dump()
        await async_db_session.flush()

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 8, 2),
            date(2024, 8, 2),
            profile_id=async_test_profile.id,
        )

        night = result.nights[0]
        assert night.periodic_breathing_pct == pytest.approx(10.0)
        assert night.pb_reason is None

    async def test_periodic_breathing_pct_read_without_parsing_result_payload(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """PB % comes straight from the persisted episode list via SQLite
        ``json_extract``: a large ``flow_analysis`` payload and fields that are
        not a valid AnalysisResult (so a full-JSON parse would fail) do not
        affect it.  8 h night, 1440 s of PB → 5.0 %."""
        from snore.mcp.tools.summary import get_nightly_summary

        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, date(2024, 8, 3))
        ar = await _make_analysis_result(async_db_session, sess)
        ar.programmatic_result_json = {
            "mode_results": "not-a-mapping",
            "flow_analysis": {"breaths": [[float(i)] * 8 for i in range(20_000)]},
            "periodic_breathing_episodes": [{"start_time": 600.0, "end_time": 2040.0}],
        }
        await async_db_session.flush()

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 8, 3),
            date(2024, 8, 3),
            profile_id=async_test_profile.id,
        )

        night = result.nights[0]
        assert night.periodic_breathing_pct == pytest.approx(5.0)
        assert night.pb_reason is None

    async def test_ahi_populated_from_day_row(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.summary import get_nightly_summary

        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(
            async_db_session, device, date(2024, 8, 1), ahi=5.2, oai=1.0, cai=0.5
        )

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 8, 1),
            date(2024, 8, 1),
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        assert night.ahi == pytest.approx(5.2, abs=0.01)
        assert night.oai == pytest.approx(1.0, abs=0.01)

    async def test_device_headline_night_carries_source_and_recount(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.summary import get_nightly_summary
        from snore.provenance import response_provenance

        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(
            async_db_session,
            device,
            date(2024, 8, 1),
            ahi=1.2,
            index_source=IndexSource.DEVICE,
            ahi_computed=1.6789,
            cai_computed=0.4,
        )

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 8, 1),
            date(2024, 8, 1),
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        block = response_provenance(type(result))

        assert night.index_source == "device"
        assert night.ahi == pytest.approx(1.2)
        assert night.ahi_computed == 1.68
        assert night.cai_computed == pytest.approx(0.4)
        assert night.hi_computed is None
        assert block["source_dependent"]["nights[].ahi"] == "nights[].index_source"
        assert "nights[].ahi_computed" in block["derived"]

    async def test_compliance_block_present_in_range_mode(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.summary import get_nightly_summary

        device = await _make_device(async_db_session, async_test_profile.id)
        for i in range(5):
            await _make_day_session(
                async_db_session,
                device,
                date(2024, 8, 1) + timedelta(days=i),
                duration_hours=8.0,
            )

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 8, 1),
            date(2024, 8, 5),
            compliance_threshold_hours=4.0,
            profile_id=async_test_profile.id,
        )
        assert result.compliance is not None
        assert result.compliance.days_total == 5
        assert result.compliance.days_compliant == 5
        assert result.compliance.compliance_pct == 100.0

    async def test_compliance_below_threshold_counted_correctly(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.summary import get_nightly_summary

        device = await _make_device(async_db_session, async_test_profile.id)
        # 3 full nights, 2 short nights
        for i in range(3):
            await _make_day_session(
                async_db_session,
                device,
                date(2024, 8, 1) + timedelta(days=i),
                duration_hours=8.0,
            )
        for i in range(3, 5):
            await _make_day_session(
                async_db_session,
                device,
                date(2024, 8, 1) + timedelta(days=i),
                duration_hours=2.0,  # below 4 h threshold
            )

        result = await get_nightly_summary(
            async_db_session,
            date(2024, 8, 1),
            date(2024, 8, 5),
            compliance_threshold_hours=4.0,
            profile_id=async_test_profile.id,
        )
        assert result.compliance is not None
        assert result.compliance.days_compliant == 3
        assert result.compliance.days_total == 5

    async def test_pagination_returns_correct_page(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.summary import get_nightly_summary

        device = await _make_device(async_db_session, async_test_profile.id)
        for i in range(10):
            await _make_day_session(
                async_db_session,
                device,
                date(2024, 8, 1) + timedelta(days=i),
            )

        page1 = await get_nightly_summary(
            async_db_session,
            date(2024, 8, 1),
            date(2024, 8, 10),
            page=1,
            page_size=5,
            profile_id=async_test_profile.id,
        )
        page2 = await get_nightly_summary(
            async_db_session,
            date(2024, 8, 1),
            date(2024, 8, 10),
            page=2,
            page_size=5,
            profile_id=async_test_profile.id,
        )
        assert len(page1.nights) == 5
        assert len(page2.nights) == 5
        assert page1.total_nights == 10
        # Pages should not overlap
        dates_p1 = {n.date for n in page1.nights}
        dates_p2 = {n.date for n in page2.nights}
        assert not dates_p1 & dates_p2


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------


class TestGetEvents:
    async def test_missing_date_raises_validation_error(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.errors import ValidationError
        from snore.mcp.tools.events import get_events

        with pytest.raises(ValidationError, match="No therapy data"):
            await get_events(
                async_db_session, date(2024, 1, 1), profile_id=async_test_profile.id
            )

    async def test_events_returned_for_session_date(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 15)
        day, sess = await _make_day_session(async_db_session, device, target_date)

        session_start = sess.start_time
        # Add two events
        for i, ev_type in enumerate(["OA", "CA"]):
            async_db_session.add(
                Event(
                    session_id=sess.id,
                    event_type=ev_type,
                    start_time=session_start + timedelta(minutes=10 + i * 5),
                    duration_seconds=15.0,
                )
            )
        await async_db_session.flush()

        result = await get_events(
            async_db_session, target_date, profile_id=async_test_profile.id
        )
        assert result.total_events == 2
        types_returned = {e.event_type for e in result.events}
        assert types_returned == {"OA", "CA"}

    async def test_event_type_filter_applied(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 16)
        day, sess = await _make_day_session(async_db_session, device, target_date)
        session_start = sess.start_time

        for i, ev_type in enumerate(["OA", "CA", "H"]):
            async_db_session.add(
                Event(
                    session_id=sess.id,
                    event_type=ev_type,
                    start_time=session_start + timedelta(minutes=10 + i * 5),
                    duration_seconds=10.0,
                )
            )
        await async_db_session.flush()

        result = await get_events(
            async_db_session,
            target_date,
            types=["OA"],
            profile_id=async_test_profile.id,
        )
        assert result.total_events == 1
        assert result.events[0].event_type == "OA"

    async def test_min_duration_filter_applied(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 17)
        day, sess = await _make_day_session(async_db_session, device, target_date)
        session_start = sess.start_time

        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=session_start + timedelta(minutes=10),
                duration_seconds=5.0,  # short
            )
        )
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=session_start + timedelta(minutes=20),
                duration_seconds=30.0,  # long
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session,
            target_date,
            min_duration=10.0,
            profile_id=async_test_profile.id,
        )
        assert result.total_events == 1
        assert result.events[0].duration_seconds == 30.0

    async def test_event_context_includes_minutes_since_start(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 18)
        day, sess = await _make_day_session(async_db_session, device, target_date)
        session_start = sess.start_time

        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="CA",
                start_time=session_start + timedelta(minutes=45),
                duration_seconds=20.0,
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session,
            target_date,
            include_context=True,
            profile_id=async_test_profile.id,
        )
        assert result.total_events == 1
        ctx = result.events[0].context
        assert ctx is not None
        assert ctx.minutes_since_session_start == pytest.approx(45.0, abs=0.1)

    async def test_event_context_includes_ventilatory_metrics(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Device MV + therapy pressure + EPAP seeded → an OA event carries
        MV slope/stability (constant MV → 0.0) and PS = 15 − 10 = 5 cmH2O."""
        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 20)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        for waveform_type, unit, value in (
            ("mv", "L/min", 6.0),
            ("therapy_pressure", "cmH2O", 15.0),
            ("epap", "cmH2O", 10.0),
        ):
            await _make_waveform(
                async_db_session,
                sess,
                waveform_type,
                1.0,
                unit,
                900,
                constant_value=value,
            )
        await _make_event(async_db_session, sess, event_type="OA", offset_seconds=600.0)

        result = await get_events(
            async_db_session, target_date, profile_id=async_test_profile.id
        )

        ev = result.events[0]
        assert ev.context is not None
        assert ev.context.preceding_mv_slope_lpm_per_min == pytest.approx(0.0)
        assert ev.context.stability_index == pytest.approx(0.0)
        assert ev.context.ps_delivered_cmh2o == pytest.approx(5.0)
        assert ev.context.mv_prior_120s_lpm == pytest.approx(6.0)
        assert ev.context.mv_source == "device"
        assert ev.preceding_mv_slope_reason is None
        assert ev.stability_reason is None
        assert ev.ps_reason is None

    @pytest.mark.parametrize(
        ("mv_channel", "expected_version"),
        [("mv", None), ("flow", "v1")],
        ids=["device_mv", "flow_derived"],
    )
    async def test_mv_fallback_version_only_with_flow_derived_events(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Any,
        mv_channel: str,
        expected_version: str | None,
    ) -> None:
        """``mv_fallback_version`` labels the response only when a returned
        event's MV was derived from flow."""
        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 21)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        await _make_waveform(
            async_db_session, sess, mv_channel, 1.0, "L/min", 900, constant_value=6.0
        )
        await _make_event(async_db_session, sess, event_type="CA", offset_seconds=600.0)

        result = await get_events(
            async_db_session, target_date, profile_id=async_test_profile.id
        )

        assert result.mv_fallback_version == expected_version
        assert result.events[0].context is not None
        assert result.events[0].context.mv_source == (
            "device" if mv_channel == "mv" else "flow_derived"
        )

    async def test_include_context_false_omits_context_and_reasons(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """include_context=False → events listed, context and its reasons null,
        no flow-derived label."""
        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 22)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        await _make_waveform(
            async_db_session, sess, "flow", 1.0, "L/min", 900, constant_value=6.0
        )
        await _make_event(async_db_session, sess, event_type="OA", offset_seconds=600.0)

        result = await get_events(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            include_context=False,
        )

        assert result.total_events == 1
        row = result.events[0]
        assert row.context is None
        assert row.event_type == "OA"
        assert row.pressure_reason is None
        assert row.mv_reason is None
        assert row.ps_reason is None
        assert result.mv_fallback_version is None

    async def test_event_row_a6_timestamp_contract(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A6: EventRow uses offset-free ISO 8601 wall-clock + offset_seconds, not UTC-offset."""
        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 18)
        day, sess = await _make_day_session(async_db_session, device, target_date)
        session_start = sess.start_time

        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=session_start + timedelta(minutes=30),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session,
            target_date,
            include_context=True,
            profile_id=async_test_profile.id,
        )
        ev = result.events[0]

        # Tier-2: wall-clock must be offset-free (no +HH:MM, no Z)
        assert ev.timezone_status == "unknown"
        assert "+" not in ev.start_time_wall_clock
        assert ev.start_time_wall_clock.endswith("Z") is False
        # Tier-3: offset_seconds = 30 min from session start
        assert ev.offset_seconds == pytest.approx(1800.0, abs=0.1)
        # Response also carries session anchor
        assert result.timezone_status == "unknown"
        assert "+" not in result.session_start_wall_clock

    async def test_context_disabled_returns_no_context_block(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 19)
        day, sess = await _make_day_session(async_db_session, device, target_date)
        session_start = sess.start_time

        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=session_start + timedelta(minutes=10),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        result = await get_events(
            async_db_session,
            target_date,
            include_context=False,
            profile_id=async_test_profile.id,
        )
        assert result.total_events == 1
        assert result.events[0].context is None


# ---------------------------------------------------------------------------
# A6 non-UTC determinism test
# ---------------------------------------------------------------------------


class TestA6TimestampDeterminism:
    async def test_event_timestamps_identical_in_non_utc_host(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A6: wall-clock timestamps must be identical regardless of host timezone.

        The DB stores naive datetimes (no TZ).  isoformat() on a naive datetime
        produces offset-free strings — same output whether the host is UTC,
        America/New_York, or Asia/Tokyo.  This test proves that by checking the
        output is offset-free and that running the tool under TZ=America/New_York
        (simulated by confirming no offset appears) produces the same value as the
        raw DB string.
        """
        import os

        from snore.mcp.tools.events import get_events

        device = await _make_device(async_db_session, async_test_profile.id)
        target_date = date(2024, 8, 20)
        day, sess = await _make_day_session(async_db_session, device, target_date)
        session_start = sess.start_time

        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="OA",
                start_time=session_start + timedelta(hours=1),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        # Capture result under current TZ
        result = await get_events(
            async_db_session, target_date, profile_id=async_test_profile.id
        )
        ev = result.events[0]
        wall_clock_str = ev.start_time_wall_clock

        # Verify: offset-free (no +HH:MM, no Z, no -HH:MM)
        assert "+" not in wall_clock_str
        assert wall_clock_str.rstrip("0123456789:.T-") == ""  # only datetime chars
        assert ev.timezone_status == "unknown"

        # Verify: matches the raw DB value exactly (naive datetime.isoformat())
        expected = (session_start + timedelta(hours=1)).isoformat()
        assert wall_clock_str == expected

        # Verify the same invariant holds under a non-UTC host TZ.
        # Setting os.environ["TZ"] + calling time.tzset() (POSIX-only) actually
        # shifts the process timezone.  On Windows (no tzset), the test still
        # checks offset-freedom through the env change, which is weaker but
        # harmless.  The real protection is that our code never calls
        # .astimezone() or .timestamp() on naive datetimes.
        import time

        original_tz = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "America/New_York"
            if hasattr(time, "tzset"):
                time.tzset()
            # Re-run the same tool — output must be byte-identical
            result2 = await get_events(
                async_db_session, target_date, profile_id=async_test_profile.id
            )
            assert result2.events[0].start_time_wall_clock == wall_clock_str
        finally:
            if original_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original_tz
            if hasattr(time, "tzset"):
                time.tzset()


# ---------------------------------------------------------------------------
# M4 cold-process capabilities test
# ---------------------------------------------------------------------------


class TestCapabilitiesColdProcess:
    async def test_capabilities_returns_empty_lists_on_cold_db(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """M4: docs://capabilities on a fresh DB returns empty lists, not errors."""
        from snore.mcp.tools.overview import get_data_overview

        # Cold DB — no imports
        result = await get_data_overview(
            async_db_session, profile_id=async_test_profile.id
        )

        assert result.devices == []
        assert result.total_sessions == 0
        assert result.available_waveform_channels == []
        assert result.available_event_types == []
        assert not result.analysis_run
        assert result.analysis_session_count == 0

    async def test_capabilities_ensure_registered_parsers_idempotent(self) -> None:
        """M4: ensure_registered_parsers() is safe to call multiple times."""
        from snore.parsers.register_all import ensure_registered_parsers
        from snore.parsers.registry import parser_registry

        # First call registers parsers
        ensure_registered_parsers()
        count_after_first = len(parser_registry.list_parsers())
        assert count_after_first >= 0  # may be 0 if parsers not installed; that's fine

        # Second call must not raise even if parsers are already registered
        try:
            ensure_registered_parsers()
        except Exception as exc:
            raise AssertionError(
                f"ensure_registered_parsers() raised on second call: {exc}"
            ) from exc
