"""FastMCP in-memory roundtrip tests for the get_breath_table tool.

Each test exercises the full server wiring — error boundary, size guard, JSON
serialization — by calling the tool through a real fastmcp.Client connected
in-memory to the server.  BreathService.get_breath_table is patched via
extra_patches so no real database is touched.
"""

from __future__ import annotations

import json

from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import pytest

from fastmcp.exceptions import ToolError

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_breath_page(
    *,
    rows: list | None = None,
    bins: list | None = None,
    analysis_status_value: str = "ok",
    null_reason_value: str | None = None,
    is_binned: bool = False,
    total_breaths: int = 1,
    page: int = 1,
    page_size: int = 500,
    session_id: int | None = 42,
) -> object:
    """Build a BreathPage-like mock using the real service DTOs."""
    from snore.services.breath_service import (  # noqa: PLC0415
        AnalysisStatus,
        BreathPage,
        BreathQueryRange,
        NullReason,
    )

    query = BreathQueryRange(
        therapy_date=date(2024, 1, 1),
        offset_start=0.0,
        offset_end=300.0,
    )
    return BreathPage(
        query=query,
        analysis_status=AnalysisStatus(analysis_status_value),
        algo_versions=None,
        null_reason=NullReason(null_reason_value) if null_reason_value else None,
        is_binned=is_binned,
        total_breaths=total_breaths,
        page=page,
        page_size=page_size,
        rows=rows or [],
        bins=bins or [],
        session_id=session_id,
    )


def _make_breath_row() -> object:
    """Build a single BreathRow with representative values."""
    from snore.services.breath_service import (  # noqa: PLC0415
        BreathRow,
        CycleType,
        TimezoneStatus,
        TriggerCycleApplicability,
        TriggerType,
    )

    return BreathRow(
        analysis_result_id=100,
        session_id=42,
        breath_number=1,
        session_start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
        timezone_status=TimezoneStatus.UNKNOWN,
        start_offset_seconds=5.0,
        end_offset_seconds=9.0,
        ti=1.2,
        te=2.4,
        ttot=3.6,
        ie_ratio=0.5,
        duty_cycle=0.33,
        peak_insp_flow=25.0,
        peak_exp_flow=20.0,
        tidal_volume=450.0,
        flatness_index=0.7,
        mid_insp_flattening=0.3,
        flow_class=2,
        flow_class_confidence=0.85,
        is_recovery_breath=False,
        trigger_type=TriggerType.NORMAL,
        cycle_type=CycleType.NORMAL,
        trigger_cycle_confidence=0.9,
        trigger_cycle_applicability=TriggerCycleApplicability.VALIDATED,
        trigger_cycle_reason=None,
        leak_valid=True,
        leak_valid_reason=None,
        ramp_active=False,
        ramp_active_reason=None,
        mask_off=False,
        mask_off_reason=None,
    )


def _make_breath_bin() -> object:
    """Build a single BreathBin with representative values."""
    from snore.services.breath_service import (  # noqa: PLC0415
        AnalysisStatus,
        BreathBin,
    )

    return BreathBin(
        session_start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
        bin_start_offset=0.0,
        bin_end_offset=300.0,
        breath_count=5,
        flatness_index_median=0.65,
        mid_insp_flattening_median=0.25,
        flow_class_mode=2,
        tidal_volume_median=440.0,
        ie_ratio_median=0.48,
        leak_valid_fraction=0.9,
        analysis_status=AnalysisStatus.OK,
    )


# ---------------------------------------------------------------------------
# TestGetBreathTableRoundtrip
# ---------------------------------------------------------------------------


class TestGetBreathTableRoundtrip:
    async def test_raw_happy_path_returns_rows_with_renamed_fields(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """Raw fetch with >=1 BreathRow returns renamed fields and session anchors.

        Verifies:
        - top-level session_id and session_start_wall_clock from first row
        - timezone_status "unknown" on row and response
        - StrEnum values serialised as plain strings (trigger_type, cycle_type)
        - Unit-renamed fields: ti_s, peak_insp_flow_lpm, tidal_volume_ml
        - is_binned: false
        """
        row = _make_breath_row()
        mock_page = _make_mock_breath_page(rows=[row], total_breaths=1)

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    return_value=mock_page,
                ),
            ],
        ) as client:
            result = await client.call_tool(
                "experimental_get_breath_table",
                {
                    "date": "2024-01-01",
                    "offset_start": 0.0,
                    "offset_end": 300.0,
                },
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)

        # Response-level session anchors from the first row
        assert payload["session_id"] == 42
        assert payload["session_start_wall_clock"] == "2024-01-01T22:00:00"
        assert payload["timezone_status"] == "unknown"
        assert payload["is_binned"] is False
        assert payload["total_breaths"] == 1

        row_wire = payload["rows"][0]
        assert row_wire["session_id"] == 42
        assert row_wire["session_start_wall_clock"] == "2024-01-01T22:00:00"
        assert row_wire["timezone_status"] == "unknown"

        # Unit-renamed fields
        assert row_wire["ti_s"] == pytest.approx(1.2)
        assert row_wire["peak_insp_flow_lpm"] == pytest.approx(25.0)
        assert row_wire["tidal_volume_ml"] == pytest.approx(450.0)

        # StrEnum values must be plain strings
        assert row_wire["trigger_type"] == "normal"
        assert row_wire["cycle_type"] == "normal"
        assert row_wire["trigger_cycle_applicability"] == "validated"

        # bins list must be empty for raw mode
        assert payload["bins"] == []

    async def test_binned_path_populates_bins_and_top_level_anchor(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """Binned fetch with bins populated returns is_binned=True and session anchor
        from first bin; rows list must be empty.
        """
        mock_bin = _make_breath_bin()
        mock_page = _make_mock_breath_page(
            bins=[mock_bin],
            is_binned=True,
            total_breaths=5,
            session_id=99,
        )

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    return_value=mock_page,
                ),
            ],
        ) as client:
            result = await client.call_tool(
                "experimental_get_breath_table",
                {
                    "date": "2024-01-01",
                    "offset_start": 0.0,
                    "offset_end": 300.0,
                    "bin_minutes": 5.0,
                },
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)

        assert payload["is_binned"] is True
        assert payload["total_breaths"] == 5
        assert payload["session_start_wall_clock"] == "2024-01-01T22:00:00"
        assert payload["rows"] == []
        # session_id must come from dto.session_id even when only bins are present
        assert payload["session_id"] == 99

        bin_wire = payload["bins"][0]
        assert bin_wire["breath_count"] == 5
        assert bin_wire["tidal_volume_median_ml"] == pytest.approx(440.0)
        assert bin_wire["analysis_status"] == "ok"

    async def test_binned_page_size_echoes_requested_value(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """Binned responses echo the requested page_size, not the bin count."""
        mock_bin = _make_breath_bin()
        mock_page = _make_mock_breath_page(
            bins=[mock_bin],
            is_binned=True,
            total_breaths=5,
            page_size=500,
        )

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    return_value=mock_page,
                ),
            ],
        ) as client:
            result = await client.call_tool(
                "experimental_get_breath_table",
                {
                    "date": "2024-01-01",
                    "offset_start": 0.0,
                    "offset_end": 300.0,
                    "bin_minutes": 5.0,
                    "page_size": 500,
                },
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["is_binned"] is True
        assert len(payload["bins"]) == 1
        assert payload["page_size"] == 500

    async def test_not_run_returns_success_with_zero_breaths(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """analysis_status=not_run yields a successful response with total_breaths=0.

        NOT_RUN is NOT a tool error — it is an informational response indicating
        that no analysis has been run for this date.
        """
        mock_page = _make_mock_breath_page(
            analysis_status_value="not_run",
            null_reason_value="analysis_not_run",
            total_breaths=0,
        )

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    return_value=mock_page,
                ),
            ],
        ) as client:
            result = await client.call_tool(
                "experimental_get_breath_table",
                {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 300.0},
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)

        assert payload["analysis_status"] == "not_run"
        assert payload["null_reason"] == "analysis_not_run"
        assert payload["total_breaths"] == 0

    async def test_multi_session_ambiguity_error_lists_session_ids(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """MultiSessionAmbiguityError → ToolError listing session IDs with disambiguation hint."""
        from snore.services.breath_service import (  # noqa: PLC0415
            MultiSessionAmbiguityError,
            SessionSummary,
        )

        exc = MultiSessionAmbiguityError(
            therapy_date=date(2024, 1, 1),
            device_id=5,
            sessions=[
                SessionSummary(
                    session_id=10,
                    start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
                    duration_seconds=3600.0,
                ),
                SessionSummary(
                    session_id=11,
                    start_wall_clock=datetime(2024, 1, 2, 1, 0, 0),
                    duration_seconds=7200.0,
                ),
            ],
        )

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    side_effect=exc,
                ),
            ],
        ) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "experimental_get_breath_table",
                    {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 300.0},
                )

        message = str(exc_info.value)
        assert "session_id=10" in message
        assert "session_id=11" in message
        assert "pass session_id" in message

    async def test_device_not_owned_error_hides_profile_id(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """DeviceNotOwnedError → ToolError naming device_id but never the profile_id."""
        from snore.services.breath_service import DeviceNotOwnedError  # noqa: PLC0415

        exc = DeviceNotOwnedError(device_id=42, profile_id=999)

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    side_effect=exc,
                ),
            ],
        ) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "experimental_get_breath_table",
                    {
                        "date": "2024-01-01",
                        "offset_start": 0.0,
                        "offset_end": 300.0,
                        "device_id": 42,
                    },
                )

        message = str(exc_info.value)
        assert "device_id=42" in message
        assert "is not available in this session" in message
        assert "profile" not in message.lower()

    async def test_raw_window_over_15min_raises_error_before_service_call(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """Raw window > 900 s without bin_minutes → ToolError mentioning bin_minutes.

        BreathQueryRange validates the window before the service is called, so
        the mock must not be invoked.
        """
        with patch(
            "snore.services.breath_service.BreathService.get_breath_table",
            new_callable=AsyncMock,
        ) as mock_svc:
            async with mcp_client_factory(mock_db_session) as client:
                with pytest.raises(ToolError, match="bin_minutes"):
                    await client.call_tool(
                        "experimental_get_breath_table",
                        {
                            "date": "2024-01-01",
                            "offset_start": 0.0,
                            "offset_end": 1800.0,  # 30 min — exceeds 15-min cap
                        },
                    )

        mock_svc.assert_not_called()

    async def test_no_such_table_error_reports_table_missing(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """OperationalError('no such table: breaths') → ToolError containing 'table_missing'."""
        from sqlalchemy.exc import OperationalError  # noqa: PLC0415

        exc = OperationalError("no such table: breaths", None, None)

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    side_effect=exc,
                ),
            ],
        ) as client:
            with pytest.raises(ToolError, match="table_missing"):
                await client.call_tool(
                    "experimental_get_breath_table",
                    {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 300.0},
                )

    async def test_non_table_operational_error_is_sanitized(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """OperationalError not containing 'no such table' → generic ToolError message.

        The raw SQLite error text (which may include DB paths, column names, or SQL
        fragments) must not appear in the user-facing message.
        """
        from sqlalchemy.exc import OperationalError  # noqa: PLC0415

        exc = OperationalError("attempt to write a readonly database", None, None)

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    side_effect=exc,
                ),
            ],
        ) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "experimental_get_breath_table",
                    {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 300.0},
                )

        message = str(exc_info.value)
        # Must not leak the raw SQLite error text or any path
        assert "readonly database" not in message
        assert "attempt to write" not in message
        # Must contain the generic sanitized message
        assert "database error" in message.lower()

    async def test_no_sessions_in_range_produces_polished_error(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """NoSessionsInRangeError from service → ToolError with polished date message.

        The error must say 'No therapy data found for date <date>' and include
        the get_data_overview hint. The raw internal message must not appear.
        """
        from snore.services.breath_service import (
            NoSessionsInRangeError,  # noqa: PLC0415
        )

        exc = NoSessionsInRangeError(date(2024, 1, 1), date(2024, 1, 1))

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    side_effect=exc,
                ),
            ],
        ) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "experimental_get_breath_table",
                    {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 300.0},
                )

        msg = str(exc_info.value)
        assert "No therapy data found for date 2024-01-01" in msg
        assert "get_data_overview" in msg
        assert "No sessions found in range" not in msg

    async def test_oversize_response_advises_narrowing_query(
        self, mock_db_session: object, mcp_client_factory: object
    ) -> None:
        """When the JSON payload exceeds RESPONSE_SIZE_LIMIT, ToolError advises narrowing."""
        row = _make_breath_row()
        mock_page = _make_mock_breath_page(rows=[row], total_breaths=1)

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.get_breath_table",
                    new_callable=AsyncMock,
                    return_value=mock_page,
                ),
                patch("snore.mcp.tools._scaffold.RESPONSE_SIZE_LIMIT", new=1),
            ],
        ) as client:
            with pytest.raises(ToolError, match="Narrow your query"):
                await client.call_tool(
                    "experimental_get_breath_table",
                    {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 300.0},
                )
