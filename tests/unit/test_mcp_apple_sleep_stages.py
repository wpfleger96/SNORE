"""get_apple_sleep_stages through the registered MCP tool."""

from __future__ import annotations

import json

from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from fastmcp.exceptions import ToolError

from snore.exceptions import NotFoundError
from snore.services.schemas import HealthSampleRead

_GET_SAMPLES = "snore.services.health_service.HealthService.get_night_samples"
_NIGHT = date(2026, 3, 10)


async def test_invalid_date_raises_tool_error_without_db(
    mock_db_session: Any, mcp_client_factory: Any
) -> None:
    async with mcp_client_factory(mock_db_session) as client:
        with pytest.raises(ToolError, match="date"):
            await client.call_tool("get_apple_sleep_stages", {"date": "2026-13-01"})

    mock_db_session.execute.assert_not_called()


async def test_absent_night_raises_tool_error_naming_the_date(
    mock_db_session: Any, mcp_client_factory: Any
) -> None:
    missing = AsyncMock(side_effect=NotFoundError("internal detail"))
    with patch(_GET_SAMPLES, missing):
        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(
                ToolError, match="^No Apple Health sleep data found for 2026-03-10$"
            ):
                await client.call_tool("get_apple_sleep_stages", {"date": "2026-03-10"})


async def test_stages_serialize_with_wall_clock_times(
    mock_db_session: Any, mcp_client_factory: Any
) -> None:
    stage = HealthSampleRead(
        id=7,
        record_type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Watch",
        start_time=datetime(2026, 3, 10, 23, 15),
        end_time=datetime(2026, 3, 10, 23, 45),
        value_text="AsleepCore",
        night_date=_NIGHT,
    )
    get_samples = AsyncMock(return_value=[stage])
    with patch(_GET_SAMPLES, get_samples):
        async with mcp_client_factory(mock_db_session) as client:
            result = await client.call_tool(
                "get_apple_sleep_stages", {"date": "2026-03-10"}
            )

    payload = json.loads(result.content[0].text)
    get_samples.assert_awaited_once_with(_NIGHT)
    assert payload["date"] == "2026-03-10"
    assert payload["stages"] == [stage.model_dump(mode="json")]
    assert payload["stages"][0]["start_time"] == "2026-03-10T23:15:00"
    assert "provenance" not in payload
