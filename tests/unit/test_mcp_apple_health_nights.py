"""get_apple_health_nights through the registered MCP tool."""

from __future__ import annotations

import json

from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from fastmcp.exceptions import ToolError

from snore.services.schemas import HealthNightSummaryRead

_LIST_NIGHTS = "snore.services.health_service.HealthService.list_nights"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"start": "2026-13-01"}, "start"),
        ({"end": "2026-03"}, "end"),
        ({"start": "2026-03-10", "end": "2026-03-01"}, "must not be after"),
        ({"page": 0}, "page must be >= 1"),
        ({"page_size": 91}, "page_size must be between 1 and 90"),
    ],
)
async def test_invalid_arguments_raise_tool_error_without_db(
    mock_db_session: Any, mcp_client_factory: Any, args: dict[str, Any], message: str
) -> None:
    async with mcp_client_factory(mock_db_session) as client:
        with pytest.raises(ToolError, match=message):
            await client.call_tool("get_apple_health_nights", args)

    mock_db_session.execute.assert_not_called()


async def test_page_maps_to_offset_and_rows_serialize(
    mock_db_session: Any, mcp_client_factory: Any
) -> None:
    night = HealthNightSummaryRead(
        night_date=date(2026, 3, 10),
        preferred_source="Watch",
        total_sleep_seconds=25200.0,
        sleep_efficiency_pct=90.0,
        computed_at=datetime(2026, 3, 11, 8, 0),
    )
    list_nights = AsyncMock(return_value=([night], 41))
    with patch(_LIST_NIGHTS, list_nights):
        async with mcp_client_factory(mock_db_session) as client:
            result = await client.call_tool(
                "get_apple_health_nights",
                {"start": "2026-01-01", "page": 3, "page_size": 20},
            )

    payload = json.loads(result.content[0].text)
    list_nights.assert_awaited_once_with(
        from_date=date(2026, 1, 1), to_date=None, limit=20, offset=40
    )
    assert payload["nights"] == [night.model_dump(mode="json")]
    assert (payload["total_nights"], payload["page"], payload["page_size"]) == (
        41,
        3,
        20,
    )
    assert payload["provenance"] == {
        "derived": [
            f"nights[].{field}"
            for field in (
                "time_in_bed_seconds",
                "total_sleep_seconds",
                "core_seconds",
                "deep_seconds",
                "rem_seconds",
                "awake_seconds",
                "unspecified_seconds",
                "sleep_efficiency_pct",
                "stage_coverage_pct",
            )
        ]
    }
