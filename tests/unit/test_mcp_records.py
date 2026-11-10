"""get_records argument validation and serialization through the registered MCP tool."""

from __future__ import annotations

import json

from datetime import date
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from fastmcp.exceptions import ToolError

from snore.services.schemas import RecordExtremes, RecordsResponse

_GET_RECORDS = "snore.services.stats_service.StatsService.get_records"


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({"top_n": 0}, "top_n"),
        ({"top_n": 51}, "top_n"),
        (
            {"start": "2026-13-01"},
            "Invalid date value '2026-13-01' for parameter 'start'",
        ),
        ({"end": "yesterday"}, "parameter 'end'"),
        (
            {"start": "2026-01-02", "end": "2026-01-01"},
            "'start' \\(2026-01-02\\) must not be after 'end' \\(2026-01-01\\)",
        ),
    ],
)
async def test_invalid_args_raise_tool_error_without_db(
    mock_db_session: Any,
    mcp_client_factory: Any,
    args: dict[str, Any],
    match: str,
) -> None:
    async with mcp_client_factory(mock_db_session) as client:
        with pytest.raises(ToolError, match=match):
            await client.call_tool("get_records", args)

    mock_db_session.execute.assert_not_called()


async def test_open_range_passes_bounds_and_returns_absent_metrics_as_null(
    mock_db_session: Any, mcp_client_factory: Any
) -> None:
    service_result = RecordsResponse(
        ahi=RecordExtremes(
            best=[(date(2026, 3, 2), 0.5)], worst=[(date(2026, 3, 1), 4.0)]
        )
    )
    with patch(
        _GET_RECORDS, new_callable=AsyncMock, return_value=service_result
    ) as get_records:
        async with mcp_client_factory(mock_db_session) as client:
            result = await client.call_tool(
                "get_records", {"top_n": 3, "start": "2026-03-01"}
            )

    get_records.assert_awaited_once_with(
        top_n=3, from_date=date(2026, 3, 1), to_date=None
    )
    payload = json.loads(result.content[0].text)
    assert payload["ahi"] == {
        "best": [["2026-03-02", 0.5]],
        "worst": [["2026-03-01", 4.0]],
    }
    for metric in ("leak", "therapy_hours", "spo2_min", "total_sleep_hours"):
        assert payload[metric] is None
