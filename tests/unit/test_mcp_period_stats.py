"""get_period_stats argument validation through the registered MCP tool."""

from __future__ import annotations

import json

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from fastmcp.exceptions import ToolError

_GET_PERIODS = "snore.services.stats_service.StatsService.get_period_statistics"


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({"period_type": "fortnight"}, "period_type"),
        (
            {"start": "2026-01-02", "end": "2026-01-01"},
            "'start' \\(2026-01-02\\) must not be after 'end' \\(2026-01-01\\)",
        ),
        ({"period_type": "day"}, "requires both start and end"),
        ({"period_type": "day", "start": "2026-01-01"}, "requires both"),
        (
            # 2026-01-01 to 2026-04-01 spans 91 nights
            {"period_type": "day", "start": "2026-01-01", "end": "2026-04-01"},
            "spans 91 nights; maximum per call is 90",
        ),
    ],
)
async def test_invalid_args_raise_tool_error_without_db(
    mock_db_session: Any,
    mcp_client_factory: Any,
    args: dict[str, str],
    match: str,
) -> None:
    async with mcp_client_factory(mock_db_session) as client:
        with pytest.raises(ToolError, match=match):
            await client.call_tool("get_period_stats", args)

    mock_db_session.execute.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        {"period_type": "day", "start": "2026-01-01", "end": "2026-03-31"},
        {"period_type": "week"},
        {"period_type": "year", "start": "2020-01-01"},
    ],
)
async def test_capped_day_range_and_open_coarse_range_return_periods(
    mock_db_session: Any, mcp_client_factory: Any, args: dict[str, str]
) -> None:
    with patch(_GET_PERIODS, new_callable=AsyncMock, return_value=[]):
        async with mcp_client_factory(mock_db_session) as client:
            result = await client.call_tool("get_period_stats", args)

    assert not result.is_error
    assert json.loads(result.content[0].text)["periods"] == []
