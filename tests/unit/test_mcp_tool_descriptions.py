"""Client-visible MCP tool descriptions keep refusal codes and limits (issue #485).

FastMCP sends a tool's leading docstring prose as its description and ``Args:``
as parameter descriptions; any other section (``Returns:``, a griffe admonition)
is silently dropped, so anything documented there never reaches the client.
"""

from __future__ import annotations

import inspect
import json

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import fastmcp
import pytest

from mcp.types import Tool

from snore.mcp.server import RESPONSE_SIZE_LIMIT, StaticRuntime, make_server
from snore.mcp.tools._scaffold import _DOC_SECTION_HEADER
from snore.mcp.validation import MAX_NIGHTLY_RANGE
from snore.services.breath.dtos import MAX_EPOCH_NIGHTS
from snore.services.schemas import MAX_LARGE_LEAK_SPANS

_RESPONSE_LIMIT = f"{RESPONSE_SIZE_LIMIT:,}"
_NIGHTLY_LIMIT = f"{MAX_NIGHTLY_RANGE} nights"

# Total ``list_tools`` JSON size (~3% above the measured size), paid by every MCP session.
TOOL_LIST_JSON_CHAR_LIMIT = 33_400

_EPOCH_SPAN_LIMIT = f"at most {MAX_EPOCH_NIGHTS} calendar nights"
_EPOCH_DEVICE_RULE = "required when several devices have data in the range"

PINNED_TOKENS: dict[str, list[str]] = {
    "compare_epochs": [
        "no_data_in_range",
        "rx_changed_within_epoch",
        "rx_violations",
        "not_available",
        _EPOCH_SPAN_LIMIT,
        _EPOCH_DEVICE_RULE,
    ],
    "experimental_compare_epochs": [
        "no_data_in_range",
        "rx_changed_within_epoch",
        "rx_violations",
        "analysis_not_run",
        "primary_mode_mismatch",
        "not_available",
        _EPOCH_SPAN_LIMIT,
        _EPOCH_DEVICE_RULE,
    ],
    "experimental_find_windows": [
        "algo_version_mismatch",
        "primary_mode_mismatch",
        "analysis_not_run",
        "For the analysis-based criteria",
        "session_coverage",
    ],
    "experimental_get_breath_table": [
        "analysis_not_run",
        "analysis_stale",
        "stale_version",
        "Raw window must be ≤ 900 s",
    ],
    "get_waveform": [
        "missing_channel_reason",
        "channel_absent",
        "missing_channels",
        "Window cap is 120 s",
        "and no ``session_id``",
    ],
    "render_window": ['"No waveform data" PNG', "and no ``session_id``"],
    "get_events": ['``"not_available"``', _RESPONSE_LIMIT],
    "experimental_get_event_context": ['``"not_available"``', _RESPONSE_LIMIT],
    "get_nightly_summary": [_NIGHTLY_LIMIT],
    "experimental_get_nightly_metrics": [_NIGHTLY_LIMIT],
    "get_night_detail": [f"At most {MAX_LARGE_LEAK_SPANS} spans are listed"],
}


@asynccontextmanager
async def _noop_scope() -> AsyncIterator[MagicMock]:
    yield MagicMock()


@asynccontextmanager
async def _fake_lifespan(*args: Any, **kwargs: Any) -> AsyncIterator[StaticRuntime]:  # noqa: RUF029
    yield StaticRuntime(base_scope_provider=_noop_scope, profile_id=1)


@pytest.fixture
async def server_and_client_tools() -> tuple[fastmcp.FastMCP, dict[str, Tool]]:
    mcp = make_server()
    with patch("snore.mcp.server._lifespan", _fake_lifespan):
        async with fastmcp.Client(mcp) as client:
            tools = await client.list_tools()
    return mcp, {t.name: t for t in tools}


def _client_visible_text(tool: Tool) -> str:
    params = tool.inputSchema.get("properties", {}).values()
    texts = [tool.description or "", *(p.get("description", "") for p in params)]
    return " ".join(" ".join(texts).split())


def _leading_prose(doc: str) -> str:
    """Docstring text before ``Args:``; fails if any other section header is present."""
    lines = doc.splitlines()
    headers = [line for line in lines if _DOC_SECTION_HEADER.match(line)]
    assert headers in ([], ["Args:"]), f"sections FastMCP would drop: {headers}"
    end = lines.index("Args:") if headers else len(lines)
    return "\n".join(lines[:end]).strip()


@pytest.mark.parametrize(("tool_name", "tokens"), PINNED_TOKENS.items())
async def test_pinned_tokens_in_client_visible_text_present(
    server_and_client_tools: tuple[fastmcp.FastMCP, dict[str, Tool]],
    tool_name: str,
    tokens: list[str],
) -> None:
    _, tools = server_and_client_tools
    text = _client_visible_text(tools[tool_name])
    missing = [token for token in tokens if token not in text]
    assert not missing, f"{tool_name} description lacks {missing}"


async def test_every_tool_docstring_reaches_client_intact(
    server_and_client_tools: tuple[fastmcp.FastMCP, dict[str, Tool]],
) -> None:
    """The client description must equal the docstring's full prose before ``Args:``.

    Any extra section header or griffe admonition would make FastMCP keep only part
    of that prose, so a mismatch or a non-``Args:`` header fails the test.
    """
    mcp, client_tools = server_and_client_tools
    for registered in await mcp.list_tools():
        doc = inspect.getdoc(registered.fn)
        assert doc, f"{registered.name} has no docstring"
        expected = " ".join(_leading_prose(doc).split())
        actual = " ".join((client_tools[registered.name].description or "").split())
        assert actual == expected, f"{registered.name}: FastMCP dropped docstring text"


async def test_tool_list_json_stays_under_size_limit(
    server_and_client_tools: tuple[fastmcp.FastMCP, dict[str, Tool]],
) -> None:
    _, tools = server_and_client_tools
    size = sum(
        len(json.dumps(t.model_dump(exclude_none=True, by_alias=True)))
        for t in tools.values()
    )
    assert size < TOOL_LIST_JSON_CHAR_LIMIT, (
        f"list_tools JSON is {size} chars, over {TOOL_LIST_JSON_CHAR_LIMIT}. Every MCP "
        "session pays for this text: cut duplicated or verbose tool text, or raise "
        "TOOL_LIST_JSON_CHAR_LIMIT deliberately if the growth is worth it."
    )
