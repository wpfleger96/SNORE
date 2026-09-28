"""FastMCP in-memory roundtrip tests for SNORE MCP tools.

Each test exercises the full server wiring — error boundary, size guard, JSON
serialization — by calling tools through a real ``fastmcp.Client`` connected
in-memory to the server returned by ``make_server()``.

Patching pattern:
  - ``snore.mcp.server._lifespan`` is replaced with a mock asynccontextmanager
    that yields a ``SNORERuntime(scope_provider=..., profile_id=1)`` backed by a
    mock ``AsyncSession``.  The patch must be active while the ``fastmcp.Client``
    context is open because ``_bound_lifespan`` inside ``make_server()`` calls
    ``_lifespan`` by module-level name at connect time.
  - BreathService / DeviceService / RxTracker methods are patched via
    ``unittest.mock.patch`` as in the old suite.
"""

from __future__ import annotations

import json

from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# get_data_overview
# ---------------------------------------------------------------------------


class TestGetDataOverviewRoundtrip:
    async def test_empty_db_returns_empty_overview(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_data_overview on empty DB: no error, devices=[]."""
        with patch(
            "snore.services.device_service.DeviceService.list_devices",
            new_callable=AsyncMock,
            return_value=[],
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool("get_data_overview", {})

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["devices"] == []
        assert payload["total_sessions"] == 0
        assert payload["analysis_run"] is False
        assert payload["timezone_status"] == "unknown"
        assert payload["timezone_name"] is None

    async def test_single_device_appears_in_overview(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_data_overview with one device returns device info.

        Query order in get_data_overview (after list_devices):
          1. analysis-session-count → .scalar_one()
          2. bulk session stats GROUP BY → .all()  rows: (device_id, count, min, max)
          3. bulk therapy modes DISTINCT → .all()  rows: (device_id, therapy_mode)
          4. waveform types → .scalars().all()
          5. event types → .scalars().all()
        """
        mock_device = MagicMock()
        mock_device.id = 1
        mock_device.manufacturer = "ResMed"
        mock_device.model = "AirCurve 11"
        mock_device.serial_number = "SN123"

        # Query 0: BreathService.resolve_timezone() — select(Profile.timezone)
        tz_result = MagicMock()
        tz_result.scalar_one_or_none.return_value = None  # no timezone declared

        # Query 1: analysis session count
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        # Query 2: bulk session stats — rows shaped (device_id, count, min_start, max_start)
        stats_result = MagicMock()
        stats_result.all.return_value = [(1, 2, None, None)]

        # Query 3: bulk therapy modes — rows shaped (device_id, therapy_mode)
        mode_result = MagicMock()
        mode_result.all.return_value = []

        # Query 4: waveform types
        waveform_result = MagicMock()
        waveform_result.scalars.return_value.all.return_value = []

        # Query 5: event types
        event_result = MagicMock()
        event_result.scalars.return_value.all.return_value = []

        mock_db_session.execute = AsyncMock(
            side_effect=[
                tz_result,
                count_result,
                stats_result,
                mode_result,
                waveform_result,
                event_result,
            ]
        )

        with (
            patch(
                "snore.services.device_service.DeviceService.list_devices",
                new_callable=AsyncMock,
                return_value=[mock_device],
            ),
            # Patch build_device_capabilities directly: overview.py lets exceptions
            # propagate, so return None (no capabilities) rather than raising.
            patch(
                "snore.mcp.tools.overview.build_device_capabilities",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool("get_data_overview", {})

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert len(payload["devices"]) == 1
        assert payload["devices"][0]["manufacturer"] == "ResMed"
        assert payload["timezone_status"] == "unknown"
        assert payload["timezone_name"] is None


# ---------------------------------------------------------------------------
# get_settings_timeline
# ---------------------------------------------------------------------------


class TestGetSettingsTimelineRoundtrip:
    async def test_empty_range_returns_no_epochs(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_settings_timeline with no data: no error, epochs=[]."""
        with patch(
            "snore.analysis.rx_tracker.RxTracker.get_history",
            new_callable=AsyncMock,
            return_value=[],
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool(
                    "get_settings_timeline",
                    {"start": "2024-01-01", "end": "2024-12-31"},
                )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["epochs"] == []
        assert payload["total_epochs"] == 0
        assert payload["device_capabilities_by_device"] == {}

    async def test_device_capabilities_at_top_level_not_per_epoch(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """Device capabilities appear in device_capabilities_by_device, not per epoch."""
        from snore.mcp.schemas import DeviceCapabilities  # noqa: PLC0415
        from snore.services.schemas import RxPeriodResponse  # noqa: PLC0415

        period = RxPeriodResponse(
            device_id=3,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            settings={"mode": "CPAP"},
            days_count=90,
        )
        mock_caps = DeviceCapabilities(
            manufacturer="ResMed",
            model="AirSense",
            serial_number="SN001",
            has_flow_waveform=True,
            has_pressure_waveform=True,
            has_leak_waveform=True,
            has_spo2=False,
            has_events=True,
            has_analysis=True,
        )
        with (
            patch(
                "snore.analysis.rx_tracker.RxTracker.get_history",
                new_callable=AsyncMock,
                return_value=[period],
            ),
            patch(
                "snore.mcp.tools.settings.build_device_capabilities",
                new_callable=AsyncMock,
                return_value=mock_caps,
            ),
            patch(
                "snore.mcp.tools.settings._has_analysis",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool(
                    "get_settings_timeline",
                    {"start": "2024-01-01", "end": "2024-12-31"},
                )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert len(payload["epochs"]) == 1
        epoch = payload["epochs"][0]
        assert "device_capabilities" not in epoch
        assert "3" in payload["device_capabilities_by_device"]
        caps = payload["device_capabilities_by_device"]["3"]
        assert caps["manufacturer"] == "ResMed"
        assert caps["has_flow_waveform"] is True

    async def test_invalid_date_range_returns_error(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_settings_timeline with end < start: ToolError raised."""
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "get_settings_timeline",
                    {"start": "2024-12-31", "end": "2024-01-01"},
                )


# ---------------------------------------------------------------------------
# get_settings_changes
# ---------------------------------------------------------------------------


class TestGetSettingsChangesRoundtrip:
    async def test_tool_is_registered(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_settings_changes appears in the server's tool listing."""
        async with mcp_client_factory(mock_db_session) as client:
            tools = await client.list_tools()

        assert "get_settings_changes" in {t.name for t in tools}

    async def test_empty_db_returns_no_changes(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_settings_changes with no data: no error, changes=[]."""
        from snore.services.schemas import RxChangesResponse  # noqa: PLC0415

        with (
            patch(
                "snore.analysis.rx_tracker.RxTracker.get_changes",
                new_callable=AsyncMock,
                return_value=RxChangesResponse(changes=[]),
            ),
            patch(
                "snore.services.mask_log_service.MaskLogService.list_entries",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool(
                    "get_settings_changes",
                    {"start": "2024-01-01", "end": "2024-12-31"},
                )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["changes"] == []
        assert payload["total_changes"] == 0

    async def test_invalid_date_range_returns_error(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_settings_changes with end < start: ToolError raised."""
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "get_settings_changes",
                    {"start": "2024-12-31", "end": "2024-01-01"},
                )


# ---------------------------------------------------------------------------
# get_nightly_summary
# ---------------------------------------------------------------------------


class TestGetNightlySummaryRoundtrip:
    async def test_empty_range_returns_empty_nights(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_nightly_summary with no Day rows: no error, nights=[]."""
        # Return None (no range summary) rather than raising — summary.py leaves
        # bs_range as None and falls through to empty DB result → nights=[].
        with patch(
            "snore.services.breath_service.BreathService.get_nightly_range_summary",
            new_callable=AsyncMock,
            return_value=None,
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool(
                    "get_nightly_summary",
                    {"start": "2024-01-01", "end": "2024-01-31"},
                )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["nights"] == []
        assert payload["total_nights"] == 0

    async def test_invalid_page_returns_error(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_nightly_summary with page=0: ToolError raised."""
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "get_nightly_summary",
                    {"start": "2024-01-01", "end": "2024-01-31", "page": 0},
                )

    async def test_over_90_nights_raises_tool_error_without_db(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_nightly_summary spanning >90 calendar nights raises ToolError before any DB call."""
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError, match="maximum per call is 90"):
                await client.call_tool(
                    "get_nightly_summary",
                    # 2024-01-01 → 2024-04-30 = 121 calendar nights
                    {"start": "2024-01-01", "end": "2024-04-30"},
                )

        # DB was never touched — scope_provider was never called
        mock_db_session.execute.assert_not_called()

    async def test_page_size_91_raises_tool_error(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """page_size=91 is rejected with a clean validation error."""
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError, match="page_size must be between 1 and 90"):
                await client.call_tool(
                    "get_nightly_summary",
                    {"start": "2024-01-01", "end": "2024-01-31", "page_size": 91},
                )


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------


class TestGetEventsRoundtrip:
    async def test_missing_date_raises_tool_error(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_events for a date with no data: ToolError with no-data message.

        The service raises NoSessionsInRangeError (a ValueError subclass) for
        missing dates; MAPPED_SERVICE_ERRORS routes it through raise_mapped_service_error,
        which produces the polished "No therapy data found for date …" message.
        """
        from datetime import date as _date  # noqa: PLC0415

        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        from snore.services.breath_service import (  # noqa: PLC0415
            NoSessionsInRangeError,
        )

        with patch(
            "snore.services.breath_service.BreathService.get_contextual_events",
            new_callable=AsyncMock,
            side_effect=NoSessionsInRangeError(_date(2024, 1, 1), _date(2024, 1, 1)),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                with pytest.raises(ToolError, match="No therapy data"):
                    await client.call_tool("get_events", {"date": "2024-01-01"})

    async def test_events_returned_for_date(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_events with contextual events: no error, events list populated."""
        from snore.services.breath_service import (  # noqa: PLC0415
            ContextualEvent,
            TimezoneStatus,
        )

        ev = ContextualEvent(
            session_id=42,
            session_start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
            event_type="CA",
            event_start_wall_clock=datetime(2024, 1, 1, 22, 30, 0),
            timezone_status=TimezoneStatus.UNKNOWN,
            offset_seconds=1800.0,
            duration_seconds=20.0,
            pressure_at_event_cmh2o=8.2,
            pressure_reason=None,
            leak_at_event_lpm=2.1,
            leak_reason=None,
            mv_prior_120s_lpm=5.8,
            mv_reason=None,
            minutes_since_session_start=30.0,
        )

        with (
            patch(
                "snore.services.breath_service.BreathService.get_contextual_events",
                new_callable=AsyncMock,
                return_value=[ev],
            ),
            patch(
                "snore.mcp.tools._capabilities.build_device_capabilities",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool(
                    "get_events", {"date": "2024-01-01", "include_context": True}
                )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["total_events"] == 1
        assert payload["events"][0]["event_type"] == "CA"
        assert payload["events"][0]["offset_seconds"] == 1800.0
        assert payload["events"][0]["session_id"] == 42
        ctx = payload["events"][0]["context"]
        assert ctx is not None
        assert ctx["pressure_at_event_cmh2o"] == pytest.approx(8.2)
        assert ctx["mv_prior_120s_lpm"] == pytest.approx(5.8)

    async def test_ventilatory_context_serialized_on_wire(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """Ventilatory values land in ``context``; their null reasons and the
        flow-derived MV provenance serialize as strings."""
        from snore.services.breath_service import (  # noqa: PLC0415
            ContextualEvent,
            MvSource,
            NullReason,
        )

        ev = ContextualEvent(
            session_id=42,
            session_start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
            event_type="OA",
            event_start_wall_clock=datetime(2024, 1, 1, 22, 30, 0),
            offset_seconds=1800.0,
            duration_seconds=12.0,
            pressure_at_event_cmh2o=None,
            pressure_reason=NullReason.NOT_AVAILABLE,
            leak_at_event_lpm=None,
            leak_reason=NullReason.NOT_AVAILABLE,
            mv_prior_120s_lpm=6.1,
            mv_reason=None,
            minutes_since_session_start=30.0,
            preceding_mv_slope_lpm_per_min=-1.5,
            preceding_mv_slope_reason=None,
            stability_index=0.12,
            stability_reason=None,
            ps_delivered_cmh2o=None,
            ps_reason=NullReason.NOT_AVAILABLE,
            mv_source=MvSource.FLOW_DERIVED,
        )

        with (
            patch(
                "snore.services.breath_service.BreathService.get_contextual_events",
                new_callable=AsyncMock,
                return_value=[ev],
            ),
            patch(
                "snore.mcp.tools._capabilities.build_device_capabilities",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool("get_events", {"date": "2024-01-01"})

        payload = json.loads(result.content[0].text)
        assert payload["mv_fallback_version"] == "v1"
        row = payload["events"][0]
        assert row["context"]["preceding_mv_slope_lpm_per_min"] == pytest.approx(-1.5)
        assert row["context"]["stability_index"] == pytest.approx(0.12)
        assert row["context"]["ps_delivered_cmh2o"] is None
        assert row["context"]["mv_source"] == "flow_derived"
        assert row["preceding_mv_slope_reason"] is None
        assert row["stability_reason"] is None
        assert row["ps_reason"] == "not_available"

    async def test_invalid_date_format_raises_tool_error(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_events with non-ISO date: ToolError raised."""
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError):
                await client.call_tool("get_events", {"date": "not-a-date"})

    async def test_empty_events_returns_null_response_anchors(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """get_events with no events for a valid date: empty list and null response-level anchors."""
        with (
            patch(
                "snore.services.breath_service.BreathService.get_contextual_events",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "snore.mcp.tools._capabilities.build_device_capabilities",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool("get_events", {"date": "2024-01-01"})

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["events"] == []
        assert payload["total_events"] == 0
        assert payload["session_id"] is None
        assert payload["session_start_wall_clock"] is None

    async def test_multi_session_events_null_response_anchors(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """Events spanning two sessions: response-level anchors null, per-event anchors populated."""
        from snore.services.breath_service import (  # noqa: PLC0415
            ContextualEvent,
            TimezoneStatus,
        )

        # Two events from different sessions
        ev1 = ContextualEvent(
            session_id=10,
            session_start_wall_clock=datetime(2024, 1, 1, 21, 0, 0),
            event_type="OA",
            event_start_wall_clock=datetime(2024, 1, 1, 21, 30, 0),
            timezone_status=TimezoneStatus.UNKNOWN,
            offset_seconds=1800.0,
            duration_seconds=15.0,
            pressure_at_event_cmh2o=9.0,
            pressure_reason=None,
            leak_at_event_lpm=1.0,
            leak_reason=None,
            mv_prior_120s_lpm=6.0,
            mv_reason=None,
            minutes_since_session_start=30.0,
        )
        ev2 = ContextualEvent(
            session_id=11,  # different session
            session_start_wall_clock=datetime(2024, 1, 2, 0, 0, 0),
            event_type="CA",
            event_start_wall_clock=datetime(2024, 1, 2, 1, 0, 0),
            timezone_status=TimezoneStatus.UNKNOWN,
            offset_seconds=3600.0,
            duration_seconds=25.0,
            pressure_at_event_cmh2o=10.0,
            pressure_reason=None,
            leak_at_event_lpm=2.0,
            leak_reason=None,
            mv_prior_120s_lpm=5.5,
            mv_reason=None,
            minutes_since_session_start=60.0,
        )

        with (
            patch(
                "snore.services.breath_service.BreathService.get_contextual_events",
                new_callable=AsyncMock,
                return_value=[ev1, ev2],
            ),
            patch(
                "snore.mcp.tools._capabilities.build_device_capabilities",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool("get_events", {"date": "2024-01-01"})

        assert not result.is_error
        payload = json.loads(result.content[0].text)

        # Response-level anchors null because events span two sessions
        assert payload["session_id"] is None
        assert payload["session_start_wall_clock"] is None
        assert payload["total_events"] == 2

        # Per-event anchors always populated
        events = payload["events"]
        assert events[0]["session_id"] == 10
        assert events[0]["session_start_wall_clock"] == "2024-01-01T21:00:00"
        assert events[1]["session_id"] == 11
        assert events[1]["session_start_wall_clock"] == "2024-01-02T00:00:00"

    async def test_max_events_truncates_list_and_sets_flag(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """max_events < event count: truncated=True, total_events is full count."""
        from snore.services.breath_service import ContextualEvent, TimezoneStatus

        def _make_event() -> ContextualEvent:
            return ContextualEvent(
                session_id=1,
                session_start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
                event_type="OA",
                event_start_wall_clock=datetime(2024, 1, 1, 22, 30, 0),
                timezone_status=TimezoneStatus.UNKNOWN,
                offset_seconds=1800.0,
                duration_seconds=10.0,
                pressure_at_event_cmh2o=None,
                pressure_reason=None,
                leak_at_event_lpm=None,
                leak_reason=None,
                mv_prior_120s_lpm=None,
                mv_reason=None,
                minutes_since_session_start=30.0,
            )

        with (
            patch(
                "snore.services.breath_service.BreathService.get_contextual_events",
                new_callable=AsyncMock,
                return_value=[_make_event() for _ in range(5)],
            ),
            patch(
                "snore.mcp.tools._capabilities.build_device_capabilities",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool(
                    "get_events", {"date": "2024-01-01", "max_events": 3}
                )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["truncated"] is True
        assert payload["total_events"] == 5
        assert len(payload["events"]) == 3

    async def test_default_max_events_produces_no_truncation(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """Default max_events (500) with fewer events: truncated=False."""
        from snore.services.breath_service import ContextualEvent, TimezoneStatus

        ev = ContextualEvent(
            session_id=1,
            session_start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
            event_type="CA",
            event_start_wall_clock=datetime(2024, 1, 1, 22, 30, 0),
            timezone_status=TimezoneStatus.UNKNOWN,
            offset_seconds=1800.0,
            duration_seconds=20.0,
            pressure_at_event_cmh2o=None,
            pressure_reason=None,
            leak_at_event_lpm=None,
            leak_reason=None,
            mv_prior_120s_lpm=None,
            mv_reason=None,
            minutes_since_session_start=30.0,
        )

        with (
            patch(
                "snore.services.breath_service.BreathService.get_contextual_events",
                new_callable=AsyncMock,
                return_value=[ev],
            ),
            patch(
                "snore.mcp.tools._capabilities.build_device_capabilities",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool("get_events", {"date": "2024-01-01"})

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["truncated"] is False
        assert payload["total_events"] == 1

    async def test_max_events_zero_raises_tool_error(
        self, mock_db_session: Any, mcp_client_factory: Any
    ) -> None:
        """max_events=0 is rejected before the DB is touched."""
        from fastmcp.exceptions import ToolError

        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError, match="max_events must be >= 1"):
                await client.call_tool(
                    "get_events", {"date": "2024-01-01", "max_events": 0}
                )

        mock_db_session.execute.assert_not_called()
