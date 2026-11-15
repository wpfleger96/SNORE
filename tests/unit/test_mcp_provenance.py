"""Provenance labeling on MCP surfaces.

Every metric is device / derived / experimental.  These tests pin that onto
what MCP clients see: tool descriptions, docs://schemas field descriptions,
server instructions, the per-response ``provenance`` block, and
docs://capabilities.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import re

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fastmcp
import pytest

from pydantic import BaseModel

import snore.mcp.tools

from snore.mcp.profiles import ClinicalProfile, get_profile
from snore.mcp.schemas import (
    SCHEMA_MODEL_MAP,
    AppleHealthNightsResponse,
    AppleSleepStagesResponse,
    BreathTableRow,
    CompareEpochsResponse,
    DataOverviewResponse,
    EpochStats,
    EventContext,
    EventsResponse,
    ExperimentalCompareEpochsResponse,
    ExperimentalEventContextRow,
    ExperimentalNightlyResponse,
    ExperimentalNightlyRow,
    NightDetailResponse,
    NightlyRow,
    NightlySummaryResponse,
    NightRecordsResponse,
    PeriodStatsResponse,
    SettingsChangesResponse,
    SettingsTimelineResponse,
    WaveformWindowResponse,
    model_to_schema,
)
from snore.mcp.server import (
    _PROVENANCE_RULE,
    StaticRuntime,
    _build_instructions,
    make_server,
)
from snore.mcp.tools._scaffold import _scope_and_run
from snore.provenance import (
    PROVENANCE_NOTES,
    IndexSource,
    Provenance,
    provenance_field,
    response_provenance,
)
from snore.services.schemas import DayExperimentalMetrics

EXPERIMENTAL_NOTE = PROVENANCE_NOTES[Provenance.EXPERIMENTAL]
EXPERIMENTAL_TOOLS = [
    "experimental_get_nightly_metrics",
    "experimental_get_event_context",
    "experimental_compare_epochs",
    "experimental_get_breath_table",
    "experimental_find_windows",
]
# Tool closures return dict[str, Any]; test_default_tool_model_matches_impl_return
# links this map to the impls.
DEFAULT_TOOLS: dict[str, type[BaseModel] | None] = {
    "get_data_overview": DataOverviewResponse,
    "get_settings_timeline": SettingsTimelineResponse,
    "get_settings_changes": SettingsChangesResponse,
    "get_period_stats": PeriodStatsResponse,
    "get_records": NightRecordsResponse,
    "get_nightly_summary": NightlySummaryResponse,
    "get_night_detail": NightDetailResponse,
    "get_apple_health_nights": AppleHealthNightsResponse,
    "get_apple_sleep_stages": AppleSleepStagesResponse,
    "get_events": EventsResponse,
    "compare_epochs": CompareEpochsResponse,
    "get_waveform": WaveformWindowResponse,
    "render_window": None,  # PNG of device samples
}
# Nightly respiratory values and the source field that sets each one's tier.
NIGHTLY_RESPIRATORY_SOURCES = {
    "rr_mean_bpm": "rr_mean_source",
    "tv_mean_ml": "tv_mean_source",
    "mv_mean_lpm": "mv_mean_source",
    "ti_median_s": "ti_median_source",
    "ie_ratio_median": "ie_ratio_source",
}
# The only sibling fields that may set a default tool's per-value tier: the Day
# headline indices' index_source and the nightly respiratory values' sources.
SOURCE_FIELDS: dict[type[BaseModel], set[str]] = {
    NightlySummaryResponse: {
        "nights[].index_source",
        *(f"nights[].{source}" for source in NIGHTLY_RESPIRATORY_SOURCES.values()),
    },
    NightDetailResponse: {"device.index_source"},
}


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


async def _read_resource(uri: str) -> str:
    mcp = make_server()
    with patch("snore.mcp.server._lifespan", _fake_lifespan):
        async with fastmcp.Client(mcp) as client:
            contents = await client.read_resource(uri)
    return contents[0].text


def _field_description(model: type[BaseModel], field_name: str) -> str:
    return str(model.model_json_schema()["properties"][field_name]["description"])


def _tool_impl(tool_name: str) -> Any:
    """The module-level function behind a default tool's closure."""
    name = {"get_waveform": "waveform_response_from_raw"}.get(tool_name, tool_name)
    for module_info in pkgutil.iter_modules(snore.mcp.tools.__path__):
        module = importlib.import_module(f"snore.mcp.tools.{module_info.name}")
        if inspect.isfunction(impl := getattr(module, name, None)):
            return impl
    raise LookupError(tool_name)


def _has_experimental_tag(schema: object) -> bool:
    """Whether any node of a JSON schema, ``$defs`` included, is tagged experimental."""
    if isinstance(schema, dict):
        return schema.get("x-provenance") == Provenance.EXPERIMENTAL or any(
            _has_experimental_tag(value) for value in schema.values()
        )
    if isinstance(schema, list):
        return any(_has_experimental_tag(value) for value in schema)
    return False


class TestToolDescriptions:
    @pytest.mark.parametrize("tool_name", EXPERIMENTAL_TOOLS)
    async def test_experimental_tool_description_carries_note(
        self, tool_name: str
    ) -> None:
        descriptions = await _tool_descriptions()

        assert f"Experimental metrics: {EXPERIMENTAL_NOTE}" in descriptions[tool_name]

    async def test_nightly_description_explains_fl_confidence_gate(self) -> None:
        description = (await _tool_descriptions())["experimental_get_nightly_metrics"]

        assert "RDI here adds the experimental RERA-proxy index" in description
        assert "rule-matched classified breaths" in description
        assert "confidence gate excludes fallback guesses" in description

    @pytest.mark.parametrize(
        "tool_name", ["experimental_get_nightly_metrics", "experimental_compare_epochs"]
    )
    async def test_rera_proxy_tool_description_names_fallback_inclusion(
        self, tool_name: str
    ) -> None:
        description = " ".join((await _tool_descriptions())[tool_name].split())

        assert "fallback flatness-triage guesses" in description
        assert "sensitivity to waveform shape" in description

    async def test_nightly_description_explains_corrupt_leak_reason(self) -> None:
        description = (await _tool_descriptions())["get_nightly_summary"]

        assert '``leak_above_24_pct_reason`` can also be ``"channel_corrupt"``' in (
            description
        )

    async def test_experimental_epochs_description_names_every_null_reason(
        self,
    ) -> None:
        description = (await _tool_descriptions())["experimental_compare_epochs"]

        assert "no_data_in_range" in description
        assert "analysis_not_run" in description
        assert "not_available" in description

    @pytest.mark.parametrize("tool_name", DEFAULT_TOOLS)
    async def test_default_tool_description_omits_note(self, tool_name: str) -> None:
        descriptions = await _tool_descriptions()

        assert EXPERIMENTAL_NOTE not in descriptions[tool_name]


class TestDefaultToolGate:
    async def test_every_tool_is_default_or_prefixed(self) -> None:
        names = set(await _tool_descriptions())

        assert {n for n in names if not n.startswith("experimental_")} == set(
            DEFAULT_TOOLS
        )

    @pytest.mark.parametrize("model", [m for m in DEFAULT_TOOLS.values() if m])
    def test_default_tool_has_no_experimental_field(
        self, model: type[BaseModel]
    ) -> None:
        block = response_provenance(model)

        assert not block.get("experimental")
        assert not _has_experimental_tag(model.model_json_schema(mode="serialization"))
        assert set(block.get("source_dependent", {}).values()) <= SOURCE_FIELDS.get(
            model, set()
        )

    def test_schema_scan_catches_experimental_parent_of_untagged_leaf(self) -> None:
        """``response_provenance`` lists leaves, so this model's block is empty."""

        class Leaf(BaseModel):
            note: str | None = None

        class Response(BaseModel):
            leaf: Leaf = provenance_field(
                Provenance.EXPERIMENTAL, "A heuristic.", default_factory=Leaf
            )

        assert response_provenance(Response) == {}
        assert _has_experimental_tag(Response.model_json_schema())

    @pytest.mark.parametrize(
        ("tool_name", "model"), [(t, m) for t, m in DEFAULT_TOOLS.items() if m]
    )
    def test_default_tool_model_matches_impl_return(
        self, tool_name: str, model: type[BaseModel]
    ) -> None:
        assert inspect.signature(_tool_impl(tool_name)).return_annotation == (
            model.__name__
        )

    @pytest.mark.parametrize(
        ("schema_type", "model"),
        [
            ("compare_epochs_response", CompareEpochsResponse),
            ("experimental_compare_epochs_response", ExperimentalCompareEpochsResponse),
        ],
    )
    def test_epoch_schema_key_names_its_tool(
        self, schema_type: str, model: type[BaseModel]
    ) -> None:
        assert SCHEMA_MODEL_MAP[schema_type] is model


class TestSchemaFieldLabels:
    @pytest.mark.parametrize(
        ("model", "field_name"),
        [
            (ExperimentalNightlyRow, "rera_index"),
            (ExperimentalNightlyRow, "rdi"),
            (ExperimentalNightlyRow, "fl_class_ge4_pct"),
            (ExperimentalNightlyRow, "rera_proxy_count"),
            (EpochStats, "rera_proxy_count"),
            (EpochStats, "flow_class_distribution"),
            (EpochStats, "flow_class_distribution_fallback"),
        ],
    )
    def test_fl_rera_proxy_field_is_experimental(
        self, model: type[BaseModel], field_name: str
    ) -> None:
        assert _field_description(model, field_name).startswith("[EXPERIMENTAL]")

    @pytest.mark.parametrize(
        "field_name", ["flow_class", "trigger_type", "tidal_volume_ml", "leak_valid"]
    )
    def test_breath_table_measurement_is_experimental(self, field_name: str) -> None:
        assert _field_description(BreathTableRow, field_name).startswith(
            "[EXPERIMENTAL]"
        )

    @pytest.mark.parametrize(
        ("model", "field_name"),
        [
            (ExperimentalNightlyRow, "rera_index"),
            (ExperimentalNightlyRow, "rdi"),
            (ExperimentalNightlyRow, "rera_proxy_count"),
            (EpochStats, "rera_proxy_count"),
            (DayExperimentalMetrics, "rera_index"),
            (DayExperimentalMetrics, "rera_count"),
        ],
    )
    def test_rera_proxy_field_names_fallback_inclusion(
        self, model: type[BaseModel], field_name: str
    ) -> None:
        description = " ".join(_field_description(model, field_name).split())

        assert "fallback" in description
        assert "sensitivity to waveform shape" in description

    def test_fl_class_ge4_pct_excludes_fallback_guesses(self) -> None:
        description = " ".join(
            _field_description(ExperimentalNightlyRow, "fl_class_ge4_pct").split()
        )

        assert "rule-matched" in description
        assert "excludes fallback guesses" in description

    def test_nightly_ahi_headline_is_device_tagged_by_index_source(self) -> None:
        prop = NightlyRow.model_json_schema()["properties"]["ahi"]

        assert _field_description(NightlyRow, "ahi").startswith("[DEVICE]")
        assert prop["x-provenance-source"] == "index_source"

    def test_nightly_ahi_computed_is_derived(self) -> None:
        assert _field_description(NightlyRow, "ahi_computed").startswith("[DERIVED]")

    def test_default_event_mv_field_is_derived_from_the_device_channel(self) -> None:
        prop = EventContext.model_json_schema()["properties"]["mv_prior_120s_lpm"]

        assert prop["description"].startswith("[DERIVED]")
        assert "device" in prop["description"]
        assert "x-provenance-source" not in prop

    def test_experimental_event_mv_field_names_its_flow_fallback(self) -> None:
        description = _field_description(
            ExperimentalEventContextRow, "mv_prior_120s_lpm"
        )

        assert description.startswith("[EXPERIMENTAL]")
        assert "mv_source" in description


class TestInstructions:
    def test_instructions_state_experimental_rule(self) -> None:
        instructions = _build_instructions(get_profile("neutral"))

        for text in (
            EXPERIMENTAL_NOTE,
            "reference standard",
            "experimental_* tools return experimental metrics:",
            "Never present them as device-scored or clinically validated",
            "docs://capabilities",
        ):
            assert text in instructions

    def test_instructions_state_null_convention(self) -> None:
        instructions = _build_instructions(get_profile("neutral"))

        assert "never zero" in instructions
        assert "`*_reason`" in instructions

    @pytest.mark.parametrize("tool", [*DEFAULT_TOOLS, *EXPERIMENTAL_TOOLS])
    def test_instructions_name_no_tool(self, tool: str) -> None:
        """Only profile text may name tools; the server's own text never does."""
        stub = ClinicalProfile(
            name="stub", display_name="Stub", priority_hint="", clinical_context=""
        )
        assert tool not in _build_instructions(stub)

    def test_neutral_instructions_stay_under_1k_chars(self) -> None:
        assert len(_build_instructions(get_profile("neutral"))) < 1000

    def test_uars_profile_flags_rera_proxy_as_experimental(self) -> None:
        hint = get_profile("uars").priority_hint

        assert "RERA proxy" in hint
        assert "not device-scored or clinically validated" in hint

    @pytest.mark.parametrize("field", ["priority_hint", "clinical_context"])
    @pytest.mark.parametrize(
        "metric", ["RDI", "RERA proxy", "flow class", "flattening"]
    )
    def test_neutral_profile_names_no_experimental_metric(
        self, field: str, metric: str
    ) -> None:
        text = getattr(get_profile("neutral"), field)

        assert not re.search(rf"\b{metric}\b", text, re.IGNORECASE)

    @pytest.mark.parametrize(
        ("profile_name", "default_field"),
        [("uars", "device_flg_"), ("csa", "central_apneas")],
    )
    def test_profile_leads_with_default_field_before_experimental_tool(
        self, profile_name: str, default_field: str
    ) -> None:
        hint = get_profile(profile_name).priority_hint

        assert hint.index(default_field) < hint.index(
            "experimental_get_nightly_metrics"
        )
        assert "not device-scored or clinically validated" in hint

    def test_csa_profile_drops_signal_no_field_provides(self) -> None:
        profile = get_profile("csa")

        assert "rolling variance" not in profile.priority_hint
        assert "stability" not in profile.clinical_context


class TestResponseExperimentalFlag:
    async def _run(
        self, result: BaseModel, tool_name: str = "get_nightly_summary"
    ) -> dict[str, Any]:
        runtime = SimpleNamespace(scope_provider=_noop_scope, profile_id=1)
        ctx = MagicMock(lifespan_context=runtime)
        return await _scope_and_run(
            ctx, AsyncMock(return_value=result), tool_name=tool_name
        )

    async def test_default_response_carries_no_provenance_or_flag(self) -> None:
        result = NightlySummaryResponse(
            nights=[
                NightlyRow(
                    date=date(2026, 1, 1), ahi=2.0, index_source=IndexSource.DEVICE
                )
            ],
            total_nights=1,
            page=1,
            page_size=30,
        )

        payload = await self._run(result)

        assert "provenance" not in payload
        assert "experimental" not in payload
        assert payload["nights"][0]["index_source"] == "device"

    async def test_experimental_tool_response_is_flagged(self) -> None:
        result = ExperimentalNightlyResponse(
            nights=[ExperimentalNightlyRow(date=date(2026, 1, 1), rera_index=1.0)],
            total_nights=1,
            page=1,
            page_size=30,
        )

        payload = await self._run(result, "experimental_get_nightly_metrics")

        assert payload["experimental"] is True
        assert "provenance" not in payload
        assert payload["nights"][0]["rera_index"] == 1.0

    def test_nightly_respiratory_values_are_device_tagged_by_their_source(
        self,
    ) -> None:
        block = response_provenance(NightlySummaryResponse)

        for value, source in NIGHTLY_RESPIRATORY_SOURCES.items():
            assert _field_description(NightlyRow, value).startswith("[DEVICE]")
            assert block["source_dependent"][f"nights[].{value}"] == (
                f"nights[].{source}"
            )

    def test_epoch_distribution_inherits_experimental_parent(self) -> None:
        """The gate sees fields nested under an experimental parent."""
        block = response_provenance(ExperimentalCompareEpochsResponse)

        assert "epochs[].mid_insp_flattening.median" in block["experimental"]
        assert "epochs[].mid_insp_flattening.n_breaths" in block["experimental"]


class TestCapabilitiesResource:
    @pytest.mark.parametrize("analysis_run", [True, False])
    async def test_capabilities_carry_legend_and_analysis_dependency(
        self, analysis_run: bool
    ) -> None:
        overview = DataOverviewResponse(
            devices=[],
            analysis_run=analysis_run,
            analysis_session_count=int(analysis_run),
        )
        with patch(
            "snore.mcp.tools.overview.get_data_overview",
            AsyncMock(return_value=overview),
        ):
            caps = json.loads(await _read_resource("docs://capabilities"))

        assert caps["provenance"] == _PROVENANCE_RULE
        assert "Device data needs no analysis" in caps["analysis"]["note"]
        assert "experimental_* tools need analysis" in caps["analysis"]["note"]
        assert "snore analysis run" not in caps["analysis"]["note"]

    async def test_get_schema_docs_list_every_schema_type(self) -> None:
        mcp = make_server()
        with patch("snore.mcp.server._lifespan", _fake_lifespan):
            async with fastmcp.Client(mcp) as client:
                templates = await client.list_resource_templates()
        description = next(
            t.description for t in templates if "schemas" in t.uri_template
        )

        assert set(SCHEMA_MODEL_MAP) <= set(re.findall(r"\w+", description or ""))


class TestSchemasResource:
    @pytest.mark.parametrize("schema_type", sorted(SCHEMA_MODEL_MAP))
    async def test_schema_tags_only_experimental_fields(self, schema_type: str) -> None:
        schema = json.loads(await _read_resource(f"docs://schemas/{schema_type}"))
        body = model_to_schema(SCHEMA_MODEL_MAP[schema_type])
        text = json.dumps(body)

        assert schema == {**body, "x-provenance-legend": _PROVENANCE_RULE}
        assert "[DEVICE]" not in text
        assert "[DERIVED]" not in text
        assert "x-provenance-source" not in text
        assert _has_experimental_tag(body) == ("[EXPERIMENTAL]" in text)

    async def test_experimental_schema_keeps_experimental_tags(self) -> None:
        schema = json.loads(
            await _read_resource("docs://schemas/experimental_nightly_row")
        )

        rdi = schema["properties"]["rdi"]
        assert rdi["x-provenance"] == Provenance.EXPERIMENTAL
        assert rdi["description"].startswith("[EXPERIMENTAL]")

    async def test_default_schema_keeps_source_fields_untagged(self) -> None:
        schema = json.loads(await _read_resource("docs://schemas/nightly_row"))
        props = schema["properties"]

        for field in ("index_source", *NIGHTLY_RESPIRATORY_SOURCES.values()):
            assert field in props
        assert "x-provenance" not in props["ahi"]
        assert not props["ahi"]["description"].startswith("[")

    def test_legend_states_the_experimental_rule(self) -> None:
        assert "[EXPERIMENTAL]" in _PROVENANCE_RULE
        assert EXPERIMENTAL_NOTE in _PROVENANCE_RULE
        assert '`"experimental": true`' in _PROVENANCE_RULE
        for unexpected in ("[DEVICE]", "[DERIVED]", "source_dependent"):
            assert unexpected not in _PROVENANCE_RULE
