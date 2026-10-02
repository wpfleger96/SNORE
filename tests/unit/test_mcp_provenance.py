"""Provenance labeling on MCP surfaces.

Every metric is device / derived / experimental.  These tests pin that onto
what MCP clients see: tool descriptions, docs://schemas field descriptions,
server instructions, the per-response ``provenance`` block, and
docs://capabilities.
"""

from __future__ import annotations

import json

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fastmcp
import pytest

from pydantic import BaseModel

from snore.mcp.profiles import get_profile
from snore.mcp.schemas import (
    BreathTableRow,
    DataOverviewResponse,
    EpochStats,
    EventContext,
    NightlyRow,
    NightlySummaryResponse,
)
from snore.mcp.server import StaticRuntime, _build_instructions, make_server
from snore.mcp.tools._scaffold import _scope_and_run
from snore.provenance import PROVENANCE_NOTES, Provenance

EXPERIMENTAL_NOTE = PROVENANCE_NOTES[Provenance.EXPERIMENTAL]
EXPERIMENTAL_TOOLS = [
    "get_nightly_summary",
    "compare_epochs",
    "get_events",
    "get_breath_table",
    "find_windows",
]


@asynccontextmanager
async def _noop_scope() -> AsyncIterator[MagicMock]:
    yield MagicMock()


@asynccontextmanager
async def _fake_lifespan(*args: Any, **kwargs: Any) -> AsyncIterator[StaticRuntime]:  # noqa: RUF029
    yield StaticRuntime(base_scope_provider=_noop_scope, profile_id=1)


async def _tool_descriptions() -> dict[str, str]:
    mcp = make_server()
    with patch("snore.mcp.server._lifespan", _fake_lifespan):
        async with fastmcp.Client(mcp) as client:
            tools = await client.list_tools()
    return {t.name: " ".join((t.description or "").split()) for t in tools}


def _field_description(model: type[BaseModel], field_name: str) -> str:
    return str(model.model_json_schema()["properties"][field_name]["description"])


class TestToolDescriptions:
    @pytest.mark.parametrize("tool_name", EXPERIMENTAL_TOOLS)
    async def test_experimental_tool_description_carries_note(
        self, tool_name: str
    ) -> None:
        descriptions = await _tool_descriptions()

        assert f"Experimental metrics: {EXPERIMENTAL_NOTE}" in descriptions[tool_name]

    async def test_nightly_description_explains_fl_confidence_gate(self) -> None:
        description = (await _tool_descriptions())["get_nightly_summary"]

        assert "RDI here adds the experimental RERA-proxy index" in description
        assert "rule-matched classified breaths" in description
        assert "confidence gate excludes fallback guesses" in description

    async def test_overview_description_omits_note(self) -> None:
        descriptions = await _tool_descriptions()

        assert EXPERIMENTAL_NOTE not in descriptions["get_data_overview"]


class TestSchemaFieldLabels:
    @pytest.mark.parametrize(
        ("model", "field_name"),
        [
            (NightlyRow, "rera_index"),
            (NightlyRow, "rdi"),
            (NightlyRow, "fl_class_ge4_pct"),
            (NightlyRow, "rera_proxy_count"),
            (EpochStats, "rera_proxy_count"),
            (EpochStats, "flow_class_distribution"),
            (EpochStats, "flow_class_distribution_fallback"),
        ],
    )
    def test_fl_rera_proxy_field_is_experimental_with_note(
        self, model: type[BaseModel], field_name: str
    ) -> None:
        description = _field_description(model, field_name)

        assert description.startswith("[EXPERIMENTAL]")
        assert EXPERIMENTAL_NOTE in description

    @pytest.mark.parametrize(
        "field_name", ["flow_class", "trigger_type", "tidal_volume_ml", "leak_valid"]
    )
    def test_breath_table_measurement_is_experimental(self, field_name: str) -> None:
        assert _field_description(BreathTableRow, field_name).startswith(
            "[EXPERIMENTAL]"
        )

    def test_fl_class_ge4_pct_excludes_fallback_guesses(self) -> None:
        description = " ".join(
            _field_description(NightlyRow, "fl_class_ge4_pct").split()
        )

        assert "rule-matched" in description
        assert "excludes fallback guesses" in description

    def test_nightly_ahi_is_derived(self) -> None:
        assert _field_description(NightlyRow, "ahi").startswith("[DERIVED]")

    def test_event_mv_field_names_its_source_field(self) -> None:
        prop = EventContext.model_json_schema()["properties"]["mv_prior_120s_lpm"]

        assert prop["x-provenance-source"] == "mv_source"


class TestInstructions:
    def test_instructions_explain_provenance_tiers(self) -> None:
        instructions = _build_instructions(get_profile("neutral"))

        assert "PROVENANCE:" in instructions
        for marker in ("[DEVICE]", "[DERIVED]", "[EXPERIMENTAL]", "`provenance`"):
            assert marker in instructions

    def test_uars_profile_flags_rera_proxy_as_experimental(self) -> None:
        assert "experimental" in get_profile("uars").priority_hint


class TestResponseProvenanceBlock:
    async def _run(self, result: BaseModel) -> dict[str, Any]:
        runtime = SimpleNamespace(scope_provider=_noop_scope, profile_id=1)
        ctx = MagicMock(lifespan_context=runtime)
        return await _scope_and_run(
            ctx, AsyncMock(return_value=result), tool_name="test_tool"
        )

    async def test_nightly_response_lists_tiers_by_path(self) -> None:
        result = NightlySummaryResponse(
            nights=[NightlyRow(date=date(2026, 1, 1), rera_index=1.0, ahi=2.0)],
            total_nights=1,
            page=1,
            page_size=30,
        )

        payload = await self._run(result)

        assert "nights[].rera_index" in payload["provenance"]["experimental"]
        assert "nights[].ahi" in payload["provenance"]["derived"]
        assert "nights[].rr_mean_bpm" not in json.dumps(payload["provenance"])

    async def test_epoch_distribution_inherits_experimental_parent(self) -> None:
        from snore.mcp.schemas import CompareEpochsResponse  # noqa: PLC0415

        payload = await self._run(CompareEpochsResponse())

        assert (
            "epochs[].mid_insp_flattening.median"
            in payload["provenance"]["experimental"]
        )
        assert "epochs[].device_flg.median" in payload["provenance"]["derived"]
        assert (
            "epochs[].mid_insp_flattening.n_breaths"
            in payload["provenance"]["experimental"]
        )
        assert "epochs[].device_flg.n_breaths" in payload["provenance"]["derived"]

    async def test_device_only_response_has_no_block(self) -> None:
        payload = await self._run(DataOverviewResponse(devices=[]))

        assert "provenance" not in payload


class TestCapabilitiesResource:
    async def test_capabilities_carry_legend_and_experimental_rdi_note(self) -> None:
        overview = DataOverviewResponse(
            devices=[], analysis_run=True, analysis_session_count=1
        )
        mcp = make_server()
        with (
            patch("snore.mcp.server._lifespan", _fake_lifespan),
            patch(
                "snore.mcp.tools.overview.get_data_overview",
                AsyncMock(return_value=overview),
            ),
        ):
            async with fastmcp.Client(mcp) as client:
                contents = await client.read_resource("docs://capabilities")
        caps = json.loads(contents[0].text)

        assert caps["provenance"] == {str(t): n for t, n in PROVENANCE_NOTES.items()}
        assert "experimental" in caps["analysis"]["note"]
