"""Roundtrip unit tests for the find_windows MCP tool.

Uses the conftest ``mcp_client_factory`` / ``mock_db_session`` fixtures with
``snore.services.breath_service.BreathService.find_windows`` patched as an
AsyncMock so no real database or analysis code runs.  Verifies the full
server wiring: criterion validation, n validation, error mapping, and the
wire-JSON mapping rules (str enums, isoformat timestamps, device_id=0 sentinel).
"""

from __future__ import annotations

import json

from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from fastmcp.exceptions import ToolError

# ---------------------------------------------------------------------------
# Helpers — pre-built service return values
# ---------------------------------------------------------------------------


def _algo_versions() -> Any:
    from snore.analysis.shared.versioning import (  # noqa: PLC0415
        AlgorithmIdentity,
        AlgoVersions,
        AnalysisRunMetadata,
    )

    return AlgoVersions(
        identity=AlgorithmIdentity.current(),
        run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
    )


def _make_happy_path_result() -> Any:
    from snore.services.breath_service import (  # noqa: PLC0415
        AnalysisStatus,
        DayAnalysisStatus,
        FindWindowsResult,
        SessionCoverage,
        TimezoneStatus,
        WindowCriterion,
        WindowResult,
    )

    av = _algo_versions()
    session_start = datetime(2025, 7, 14, 22, 0, 0)

    windows = [
        WindowResult(
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            session_id=10,
            session_start_wall_clock=session_start,
            timezone_status=TimezoneStatus.UNKNOWN,
            window_start_offset=5.0,
            window_end_offset=39.0,
            reason_summary="fl_index=0.850, 7 breaths",
            worst_mid_insp_flattening=0.85,
            fl_run_length=None,
            anchor_event_offset=None,
            analysis_result_id=99,
            analysis_status=AnalysisStatus.OK,
            analysis_reason=None,
        ),
        WindowResult(
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            session_id=10,
            session_start_wall_clock=session_start,
            timezone_status=TimezoneStatus.UNKNOWN,
            window_start_offset=30.0,
            window_end_offset=49.0,
            reason_summary="fl_index=0.700, 4 breaths",
            worst_mid_insp_flattening=0.70,
            fl_run_length=None,
            anchor_event_offset=None,
            analysis_result_id=99,
            analysis_status=AnalysisStatus.OK,
            analysis_reason=None,
        ),
    ]

    return FindWindowsResult(
        query_date=date(2025, 7, 14),
        device_id=3,
        criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
        day_status=DayAnalysisStatus.OK,
        session_coverage=[
            SessionCoverage(
                session_id=10,
                analysis_status=AnalysisStatus.OK,
                algo_versions=av,
            )
        ],
        algorithm_identity=av.identity,
        null_reason=None,
        primary_mode=None,
        windows=windows,
    )


def _make_refusal_result() -> Any:
    from snore.services.breath_service import (  # noqa: PLC0415
        AnalysisStatus,
        DayAnalysisStatus,
        FindWindowsResult,
        NullReason,
        SessionCoverage,
        WindowCriterion,
    )

    av = _algo_versions()

    return FindWindowsResult(
        query_date=date(2025, 7, 14),
        device_id=3,
        criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
        day_status=DayAnalysisStatus.MIXED_VERSION,
        session_coverage=[
            SessionCoverage(
                session_id=10,
                analysis_status=AnalysisStatus.OK,
                algo_versions=av,
            ),
            SessionCoverage(
                session_id=11,
                analysis_status=AnalysisStatus.OK,
                algo_versions=av,
            ),
        ],
        algorithm_identity=None,
        null_reason=NullReason.ALGO_VERSION_MISMATCH,
        primary_mode=None,
        windows=[],
    )


def _make_sentinel_result() -> Any:
    """Result with device_id=0 (no-device sentinel from the service)."""
    from snore.services.breath_service import (  # noqa: PLC0415
        DayAnalysisStatus,
        FindWindowsResult,
        NullReason,
        WindowCriterion,
    )

    return FindWindowsResult(
        query_date=date(2025, 7, 14),
        device_id=0,
        criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
        day_status=DayAnalysisStatus.NOT_RUN,
        session_coverage=[],
        algorithm_identity=None,
        null_reason=NullReason.ANALYSIS_NOT_RUN,
        primary_mode=None,
        windows=[],
    )


# ---------------------------------------------------------------------------
# TestFindWindowsRoundtrip
# ---------------------------------------------------------------------------


class TestFindWindowsRoundtrip:
    async def test_happy_path_two_windows_returns_wire_format(
        self, mcp_client_factory: object, mock_db_session: object
    ) -> None:
        """Happy path: 2 windows → wire JSON has str enums, isoformat wall clocks,
        timezone_status 'unknown', and coverage entries with algo_versions dict."""
        mock_fw = AsyncMock(return_value=_make_happy_path_result())
        patch_fw = patch(
            "snore.services.breath_service.BreathService.find_windows",
            mock_fw,
        )
        patch_caps = patch(
            "snore.mcp.tools.windows.build_device_capabilities",
            new_callable=AsyncMock,
            return_value=None,
        )

        async with mcp_client_factory(
            mock_db_session, extra_patches=[patch_fw, patch_caps]
        ) as client:
            result = await client.call_tool(
                "experimental_find_windows",
                {"date": "2025-07-14", "criterion": "worst_flattening_leak_valid"},
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)

        # Top-level string fields
        assert payload["query_date"] == "2025-07-14"
        assert payload["device_id"] == 3
        assert payload["criterion"] == "worst_flattening_leak_valid"
        assert payload["day_status"] == "ok"
        assert payload["null_reason"] is None
        assert payload["primary_mode"] is None

        # Two windows present, ordered worst-first
        assert len(payload["windows"]) == 2
        w0 = payload["windows"][0]
        assert w0["criterion"] == "worst_flattening_leak_valid"
        assert w0["session_id"] == 10
        assert w0["session_start_wall_clock"] == "2025-07-14T22:00:00"
        assert w0["timezone_status"] == "unknown"
        assert w0["window_start_offset"] == pytest.approx(5.0)
        assert w0["analysis_status"] == "ok"
        assert w0["analysis_reason"] is None

        # Coverage entry has algo_versions as a dict (not None)
        assert len(payload["session_coverage"]) == 1
        cov = payload["session_coverage"][0]
        assert cov["session_id"] == 10
        assert cov["analysis_status"] == "ok"
        assert isinstance(cov["algo_versions"], dict)
        assert "identity" in cov["algo_versions"]
        assert "run" in cov["algo_versions"]

        # algorithm_identity present and a dict
        assert isinstance(payload["algorithm_identity"], dict)
        assert "format_version" in payload["algorithm_identity"]

        mock_fw.assert_called_once()

    async def test_refusal_passthrough_algo_version_mismatch(
        self, mcp_client_factory: object, mock_db_session: object
    ) -> None:
        """Mixed algo versions → SUCCESS response with null_reason='algo_version_mismatch'
        and empty windows list — NOT a tool error."""
        mock_fw = AsyncMock(return_value=_make_refusal_result())
        patch_fw = patch(
            "snore.services.breath_service.BreathService.find_windows",
            mock_fw,
        )
        patch_caps = patch(
            "snore.mcp.tools.windows.build_device_capabilities",
            new_callable=AsyncMock,
            return_value=None,
        )

        async with mcp_client_factory(
            mock_db_session, extra_patches=[patch_fw, patch_caps]
        ) as client:
            result = await client.call_tool(
                "experimental_find_windows",
                {"date": "2025-07-14", "criterion": "worst_flattening_leak_valid"},
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)

        assert payload["windows"] == []
        assert payload["null_reason"] == "algo_version_mismatch"
        assert payload["day_status"] == "mixed_version"
        assert len(payload["session_coverage"]) == 2

    async def test_unknown_criterion_raises_tool_error_without_calling_service(
        self, mcp_client_factory: object, mock_db_session: object
    ) -> None:
        """Unknown criterion string → ToolError listing valid criteria; service not called."""
        mock_fw = AsyncMock()
        patch_fw = patch(
            "snore.services.breath_service.BreathService.find_windows",
            mock_fw,
        )

        async with mcp_client_factory(
            mock_db_session, extra_patches=[patch_fw]
        ) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "experimental_find_windows",
                    {"date": "2025-07-14", "criterion": "bad_criterion"},
                )

        err_text = str(exc_info.value)
        assert "bad_criterion" in err_text
        assert "worst_flattening_leak_valid" in err_text
        assert "ca_centered" in err_text
        assert "fl_run_ending_in_recovery" in err_text
        mock_fw.assert_not_called()

    async def test_n_zero_raises_tool_error(
        self, mcp_client_factory: object, mock_db_session: object
    ) -> None:
        """n=0 is rejected by validate_window_count before the impl is invoked."""
        mock_fw = AsyncMock()
        patch_fw = patch(
            "snore.services.breath_service.BreathService.find_windows",
            mock_fw,
        )

        async with mcp_client_factory(
            mock_db_session, extra_patches=[patch_fw]
        ) as client:
            with pytest.raises(ToolError, match="1 and 50"):
                await client.call_tool(
                    "experimental_find_windows",
                    {
                        "date": "2025-07-14",
                        "criterion": "worst_flattening_leak_valid",
                        "n": 0,
                    },
                )

        mock_fw.assert_not_called()

    async def test_n_51_raises_tool_error(
        self, mcp_client_factory: object, mock_db_session: object
    ) -> None:
        """n=51 is rejected by validate_window_count before the impl is invoked."""
        mock_fw = AsyncMock()
        patch_fw = patch(
            "snore.services.breath_service.BreathService.find_windows",
            mock_fw,
        )

        async with mcp_client_factory(
            mock_db_session, extra_patches=[patch_fw]
        ) as client:
            with pytest.raises(ToolError, match="1 and 50"):
                await client.call_tool(
                    "experimental_find_windows",
                    {
                        "date": "2025-07-14",
                        "criterion": "worst_flattening_leak_valid",
                        "n": 51,
                    },
                )

        mock_fw.assert_not_called()

    async def test_device_id_zero_sentinel_emits_null(
        self, mcp_client_factory: object, mock_db_session: object
    ) -> None:
        """Service device_id=0 sentinel → response device_id is null, never 0."""
        mock_fw = AsyncMock(return_value=_make_sentinel_result())
        patch_fw = patch(
            "snore.services.breath_service.BreathService.find_windows",
            mock_fw,
        )

        async with mcp_client_factory(
            mock_db_session, extra_patches=[patch_fw]
        ) as client:
            result = await client.call_tool(
                "experimental_find_windows",
                {"date": "2025-07-14", "criterion": "worst_flattening_leak_valid"},
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["device_id"] is None
        assert payload["windows"] == []
        assert payload["null_reason"] == "analysis_not_run"

    async def test_primary_mode_mismatch_refusal_pass_through_as_success(
        self, mcp_client_factory: object, mock_db_session: object
    ) -> None:
        """Service returning PRIMARY_MODE_MISMATCH for fl_run_ending_in_recovery
        produces a SUCCESS response (not a ToolError) with null_reason=
        'primary_mode_mismatch' and empty windows list."""
        from snore.services.breath_service import (  # noqa: PLC0415
            DayAnalysisStatus,
            FindWindowsResult,
            NullReason,
            WindowCriterion,
        )

        mock_result = FindWindowsResult(
            query_date=date(2025, 7, 14),
            device_id=3,
            criterion=WindowCriterion.FL_RUN_ENDING_IN_RECOVERY,
            day_status=DayAnalysisStatus.MIXED_VERSION,
            session_coverage=[],
            algorithm_identity=None,
            null_reason=NullReason.PRIMARY_MODE_MISMATCH,
            primary_mode=None,
            windows=[],
        )

        mock_fw = AsyncMock(return_value=mock_result)
        patch_fw = patch(
            "snore.services.breath_service.BreathService.find_windows",
            mock_fw,
        )

        async with mcp_client_factory(
            mock_db_session, extra_patches=[patch_fw]
        ) as client:
            result = await client.call_tool(
                "experimental_find_windows",
                {"date": "2025-07-14", "criterion": "fl_run_ending_in_recovery"},
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["windows"] == []
        assert payload["null_reason"] == "primary_mode_mismatch"

    async def test_device_ambiguity_error_raises_tool_error_with_device_list(
        self, mcp_client_factory: object, mock_db_session: object
    ) -> None:
        """DeviceAmbiguityError from service → ToolError listing device ids and serials."""
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        exc = DeviceAmbiguityError(
            therapy_date=date(2025, 7, 14),
            profile_id=1,
            owned_device_ids=[7, 8],
            device_serials={7: "SN-ALPHA", 8: "SN-BETA"},
        )
        mock_fw = AsyncMock(side_effect=exc)
        patch_fw = patch(
            "snore.services.breath_service.BreathService.find_windows",
            mock_fw,
        )

        async with mcp_client_factory(
            mock_db_session, extra_patches=[patch_fw]
        ) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "experimental_find_windows",
                    {"date": "2025-07-14", "criterion": "worst_flattening_leak_valid"},
                )

        err_text = str(exc_info.value)
        assert "device_id=7" in err_text
        assert "SN-ALPHA" in err_text
        assert "device_id=8" in err_text
        assert "SN-BETA" in err_text

    async def test_primary_mode_populated_for_worst_flattening_criterion(
        self, mcp_client_factory: object, mock_db_session: object
    ) -> None:
        """primary_mode is non-null for WORST_FLATTENING_LEAK_VALID when sessions share one mode."""
        from snore.services.breath_service import (  # noqa: PLC0415
            DayAnalysisStatus,
            FindWindowsResult,
            WindowCriterion,
        )

        av = _algo_versions()
        mock_result = FindWindowsResult(
            query_date=date(2025, 7, 14),
            device_id=3,
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            day_status=DayAnalysisStatus.OK,
            session_coverage=[],
            algorithm_identity=av.identity,
            null_reason=None,
            primary_mode="aasm",
            windows=[],
        )

        mock_fw = AsyncMock(return_value=mock_result)
        patch_fw = patch(
            "snore.services.breath_service.BreathService.find_windows",
            mock_fw,
        )
        patch_caps = patch(
            "snore.mcp.tools.windows.build_device_capabilities",
            new_callable=AsyncMock,
            return_value=None,
        )

        async with mcp_client_factory(
            mock_db_session, extra_patches=[patch_fw, patch_caps]
        ) as client:
            result = await client.call_tool(
                "experimental_find_windows",
                {"date": "2025-07-14", "criterion": "worst_flattening_leak_valid"},
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["primary_mode"] == "aasm"


class TestNonFiniteOptions:
    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    @pytest.mark.parametrize("name", ["context_seconds", "flattening_threshold"])
    async def test_non_finite_option_is_refused_before_service_call(
        self, name: str, value: float
    ) -> None:
        from snore.mcp.errors import ValidationError  # noqa: PLC0415
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        mock_fw = AsyncMock()
        with patch("snore.services.breath_service.BreathService.find_windows", mock_fw):
            with pytest.raises(ValidationError, match=f"{name} must be a finite"):
                await find_windows(
                    AsyncMock(),
                    date(2025, 7, 14),
                    profile_id=1,
                    criterion="worst_flattening_leak_valid",
                    n=3,
                    **{name: value},
                )

        mock_fw.assert_not_called()
