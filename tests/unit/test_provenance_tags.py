"""Every metric field in MCP and REST responses declares a tier.

Walks the JSON schema of every MCP response model (``SCHEMA_MODEL_MAP``) and
every OpenAPI schema a REST route can return, and requires an ``x-provenance``
tag (see ``snore.provenance``) on each numeric, boolean, enum or ``Literal``
property, so a new metric or judgement cannot ship without saying whether it is
device, derived, or experimental data.  See AGENTS.md "Provenance tagging".
"""

import json

from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal

import pytest

from pydantic import BaseModel

from snore.api.app import create_app
from snore.api.schemas import WaveformDataResponse
from snore.mcp.schemas import SCHEMA_MODEL_MAP
from snore.provenance import (
    Provenance,
    _weakest,
    description_prefix,
    field_description,
    field_provenance,
    provenance_field,
    response_provenance,
)
from snore.services.schemas import ExperimentalStats, SessionStatistics

pytestmark = pytest.mark.unit

_TIERS = {p.value for p in Provenance}
_SCALAR_TYPES = {"number", "integer", "boolean"}

# Numeric/boolean properties that are never metrics, whatever model holds them.
_NON_METRIC_FIELDS = {
    # Identifiers, pagination and list positions
    "id",
    "offset",
    "limit",
    "total",
    "page",
    "page_size",
    "truncated",
    "breath_number",
    "best_index",
    "worst_index",
    # Time positions (seconds from session start / epoch), not durations
    "start_time",
    "end_time",
    "timestamps",
    "timestamp_start",
    "timestamp_end",
    "start_offset_s",
    "end_offset_s",
    "offset_seconds",
    "offset_start",
    "offset_end",
    "start_offset_seconds",
    "end_offset_seconds",
    "bin_start_offset",
    "bin_end_offset",
    "window_start_offset",
    "window_end_offset",
    "window_start_offset_s",
    "window_end_offset_s",
    "anchor_event_offset",
    "bin_minutes",
    # Data-availability flags
    "has_analysis",
    "has_events",
    "has_event_data",
    "has_statistics",
    "has_flow_waveform",
    "has_leak_waveform",
    "has_pressure_waveform",
    "has_spo2",
    "has_flg_waveform",
    "present_in_dataset",
    "analysis_run",
    # Downsampling/binning bookkeeping
    "total_samples",
    "returned_samples",
    "original_sample_count",
    "downsampled",
    "is_downsampled",
    "is_binned",
    # User toggle: session included in stats
    "enabled",
}
_NON_METRIC_SUFFIXES = ("_id", "_ids")  # identifiers

# Count-like names that are bookkeeping in these models only.  Keyed per model
# because the same name can be a metric elsewhere (a count of SNORE-segmented
# breaths is experimental; a count of rows in a page is not).
_NON_METRIC_MODEL_FIELDS = {
    # Pagination / page-size totals
    ("EventsResponse", "total_events"),
    ("ExperimentalEventContextResponse", "total_events"),
    ("ExperimentalNightlyResponse", "total_nights"),
    ("NightlySummaryResponse", "total_nights"),
    ("SettingsChangesResponse", "total_changes"),
    ("SettingsTimelineResponse", "total_epochs"),
    # Coverage counts of rows/days/sessions, not therapy events
    ("AnalysisJobEnqueued", "session_count"),
    ("DayDetail", "session_count"),
    ("DayListItem", "session_count"),
    ("DeviceInfo", "session_count"),
    ("DeviceUsageSummary", "session_count"),
    ("NightlyRow", "session_count"),
    ("DataOverviewResponse", "analysis_session_count"),
    ("DataOverviewResponse", "total_sessions"),
    ("SessionDetail", "event_count"),
    ("SessionDetail", "waveform_count"),
    ("MaskEpochResponse", "days_count"),
    ("RxPeriodResponse", "days_count"),
    ("PeriodStatistics", "days_in_period"),
    ("EpochDistribution", "n_nights"),
    ("EpochStats", "nights_with_data"),
    ("EpochStats", "nights_missing_analysis"),
    ("DeviceEpochDistribution", "n_nights"),
    ("DeviceEpochStats", "nights_with_data"),
    # Validation-run session/night/sample coverage
    ("AggregateMetrics", "total_sessions"),
    ("AggregateMetrics", "low_sensitivity_sessions"),  # session ids
    ("BreathTrendsAggregateMetrics", "total_sessions"),
    ("BreathTrendsAggregateMetrics", "sessions_compared"),
    ("BreathTrendsAggregateMetrics", "sessions_skipped_no_analysis"),
    ("BreathTrendsAggregateMetrics", "sessions_skipped_no_valid_breaths"),
    ("FlAggregateMetrics", "total_sessions"),
    ("FlAggregateMetrics", "sessions_compared"),
    ("FlAggregateMetrics", "sessions_skipped_no_analysis"),
    ("FlAggregateMetrics", "sessions_skipped_no_flg"),
    ("FlAggregateMetrics", "sessions_skipped_no_valid_breaths"),
    ("ReraAggregateMetrics", "total_sessions"),
    ("ReraAggregateMetrics", "sessions_with_machine_re"),
    ("ReraAggregateMetrics", "sessions_skipped_error"),
    ("ReraAggregateMetrics", "sessions_skipped_no_analysis"),
    ("ReraAggregateMetrics", "sessions_skipped_no_machine_re"),
    ("ReraAggregateMetrics", "sessions_skipped_no_valid_breaths"),
    ("ChannelAggregateMetrics", "sessions_with_data"),
    ("AppleCrossAggregate", "total_nights"),
    ("AppleCrossAggregate", "n_analysis_not_run"),
    ("AppleCrossAggregate", "n_analysis_stale"),
    ("AppleCrossAggregate", "n_device_ambiguous"),
    ("AppleCrossAggregate", "n_skipped_no_apple_bd"),
    ("AppleCrossAggregate", "n_with_apple_bd"),
    ("ChannelComparison", "n_pairs"),
    ("PairCorrelation", "n_paired_nights"),
    # Configured parameters echoed back
    ("ReraAggregateMetrics", "match_tolerance_seconds"),
    ("ComplianceFields", "threshold_hours"),
    # Enum labels of a value's origin, not judgements about it
    ("DayDetail", "index_source"),
    ("DayListItem", "index_source"),
    ("NightlyRow", "index_source"),
    ("EventComparisonDetail", "source"),
}

# Operational responses (auth, admin, jobs, imports, DB maintenance) whose
# numbers describe the system, not therapy data.
_OPERATIONAL_MODELS = {
    "AnalysisDeletePreview",
    "AnalysisJobStatus",
    "AnalysisSessionDetail",
    "AuthStatusResponse",
    "DatabaseStatsPublic",
    "DeleteDataResult",
    "DeletePreview",
    "GoogleBindingItem",
    "HealthImportResultSummary",
    "ImportResultSummary",
    "ImportSourceResultSummary",
    "InviteInfoResponse",
    "LinkedAnalysisSummary",
    "LoginResponse",
    "McpStatus",
    "MeResponse",
    "PipelineJobStatus",
    "ProfileResponse",
    "ResetAllBindingsResponse",
    "ResetResult",
    "TotpStatusResponse",
    "UserInfo",
    "UserItem",
    "UserPreferences",
    "VacuumResult",
    "ValidationRunDetail",
    "ValidationRunStatus",
}


# Untyped REST responses that are not therapy metrics: OAuth redirects, file
# downloads (exports carry their own provenance header), progress streams, and
# deletion counts.
_OPERATIONAL_UNTYPED_ROUTES = {
    "GET /api/v1/auth/google/login",
    "GET /api/v1/auth/google/connect",
    "GET /api/v1/auth/google/callback",
    "POST /api/v1/auth/invites/google",
    "DELETE /api/v1/sessions/",
    "DELETE /api/v1/analysis",
    "GET /api/v1/import/{job_id}/progress",
    "GET /api/v1/export/csv",
    "GET /api/v1/export/json",
    "GET /api/v1/export/raw",
}

# Known gap: metric routes with no JSON schema to tag.  Both return a
# downloadable HTML report (HTMLResponse), not JSON.
_UNTYPED_METRIC_ROUTES = {
    "GET /api/v1/reports/summary",
    "GET /api/v1/reports/comparison",
}

# Every REST operation carrying the ``experimental`` tag.  Pinned so that tagging
# a route cannot quietly exempt it from
# test_default_rest_operations_return_no_experimental_field.  The UI's boundary
# test reads the same file, so a tagged route also needs a wrapper the UI gates.
EXPERIMENTAL_OPERATIONS = set(
    json.loads(
        (
            Path(__file__).parents[2] / "ui/src/tests/experimentalOperations.json"
        ).read_text()
    )
)

_FIX_HINT = (
    "Wrap each metric in provenance_field(Provenance.X, ...); if a field is "
    "bookkeeping (ids, positions), add it to _NON_METRIC_FIELDS; a coverage "
    "count goes in _NON_METRIC_MODEL_FIELDS keyed by model, with a comment. "
    'See AGENTS.md "Provenance tagging".'
)


def _refs(node: Any) -> set[str]:
    """Names of every ``#/.../Name`` ``$ref`` nested anywhere in ``node``."""
    if isinstance(node, dict):
        found = {node["$ref"].rsplit("/", 1)[-1]} if "$ref" in node else set()
        return found.union(*(_refs(v) for v in node.values()))
    if isinstance(node, list):
        return set().union(*(_refs(v) for v in node))
    return set()


@cache
def _openapi() -> dict[str, Any]:
    return create_app().openapi()


def _reachable(node: Any) -> set[str]:
    """Component schemas that ``node``'s ``$ref``s lead to, transitively."""
    components = _openapi()["components"]["schemas"]
    pending, seen = _refs(node), set()
    while pending:
        name = pending.pop()
        seen.add(name)
        pending |= _refs(components[name]) - seen
    return seen


def _rest_object_schemas() -> list[tuple[str, dict[str, Any]]]:
    """OpenAPI component schemas reachable from any route's responses."""
    spec = _openapi()
    components = spec["components"]["schemas"]
    responses = [
        op.get("responses", {})
        for path in spec["paths"].values()
        for op in path.values()
    ]
    return [(name, components[name]) for name in _reachable(responses)]


def _mcp_object_schemas() -> list[tuple[str, dict[str, Any]]]:
    """Every MCP response model schema and its ``$defs``."""
    objects: list[tuple[str, dict[str, Any]]] = []
    for model in SCHEMA_MODEL_MAP.values():
        schema = model.model_json_schema(mode="serialization")
        objects += [(model.__name__, schema), *schema.get("$defs", {}).items()]
    return objects


def _surfaces() -> list[tuple[str, list[tuple[str, dict[str, Any]]]]]:
    return [("MCP", _mcp_object_schemas()), ("REST", _rest_object_schemas())]


def _object_properties() -> list[tuple[str, str, str, dict[str, Any]]]:
    """``(surface, model, field, property schema)`` for every response field."""
    return [
        (surface, model, field, prop)
        for surface, objects in _surfaces()
        for model, schema in objects
        for field, prop in schema.get("properties", {}).items()
    ]


def _model_refs(prop: dict[str, Any]) -> set[str]:
    """Models a property holds directly (not as list items)."""
    variants = [prop, *prop.get("anyOf", []), *prop.get("allOf", [])]
    return {v["$ref"].rsplit("/", 1)[-1] for v in variants if "$ref" in v}


def _untyped_routes() -> set[str]:
    """``METHOD path`` of routes with a 2xx JSON body that references no model."""
    return {
        f"{method.upper()} {path}"
        for path, ops in _openapi()["paths"].items()
        for method, op in ops.items()
        for code, response in op.get("responses", {}).items()
        if code.startswith("2")
        for body in response.get("content", {}).values()
        if not _refs(body.get("schema", {}))
    }


def _is_scalar_metric(prop: dict[str, Any]) -> bool:
    """Numeric/boolean values, or lists of them (e.g. waveform sample arrays)."""
    variants = prop.get("anyOf", [prop])
    return any(
        v.get("type") in _SCALAR_TYPES
        or (
            v.get("type") == "array" and v.get("items", {}).get("type") in _SCALAR_TYPES
        )
        for v in variants
    )


def _is_enum_valued(prop: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> bool:
    """``Literal`` or enum values, inline or by ``$ref``, alone or as list items."""
    variants = prop.get("anyOf", [prop])
    inline = (v for variant in variants for v in (variant, variant.get("items", {})))
    return any("enum" in v or "const" in v for v in inline) or any(
        "enum" in schemas.get(name, {}) for name in _refs(prop)
    )


def _is_exempt(model: str, field: str) -> bool:
    return (
        model in _OPERATIONAL_MODELS
        or field in _NON_METRIC_FIELDS
        or (model, field) in _NON_METRIC_MODEL_FIELDS
        or field.endswith(_NON_METRIC_SUFFIXES)
    )


def _operations(*, experimental: bool) -> dict[str, dict[str, Any]]:
    return {
        f"{method.upper()} {path}": op
        for path, ops in _openapi()["paths"].items()
        for method, op in ops.items()
        if ("experimental" in op.get("tags", [])) is experimental
    }


def test_numeric_and_boolean_fields_have_provenance_tag() -> None:
    offenders = sorted(
        {
            f"{surface} {model}.{field}"
            for surface, model, field, prop in _object_properties()
            if _is_scalar_metric(prop)
            and not _is_exempt(model, field)
            and prop.get("x-provenance") not in _TIERS
        }
    )
    assert not offenders, (
        "Fields missing a valid x-provenance tag:\n"
        + "\n".join(offenders)
        + f"\n{_FIX_HINT}"
    )


def test_enum_and_literal_fields_have_provenance_tag() -> None:
    # A plain ``str`` cannot be told from a label, so a SNORE judgement is typed
    # as a Literal or enum, which puts it under this check.
    offenders: set[str] = set()
    for surface, objects in _surfaces():
        schemas = dict(objects)
        offenders |= {
            f"{surface} {model}.{field}"
            for model, schema in objects
            for field, prop in schema.get("properties", {}).items()
            if _is_enum_valued(prop, schemas)
            and not _is_exempt(model, field)
            and prop.get("x-provenance") not in _TIERS
        }
    assert not offenders, (
        "Enum or Literal fields missing a valid x-provenance tag:\n"
        + "\n".join(sorted(offenders))
        + f"\n{_FIX_HINT}"
    )


class _Band(StrEnum):
    HIGH = "high"


class _Judged(BaseModel):
    literal: Literal["a", "b"]
    optional_literal: Literal["a", "b"] | None = None
    single_literal: Literal["a"]
    enum: _Band
    enum_list: list[_Band]
    plain: str
    number: int


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("literal", True),
        ("optional_literal", True),
        ("single_literal", True),
        ("enum", True),
        ("enum_list", True),
        ("plain", False),
        ("number", False),
    ],
)
def test_enum_check_sees_literals_and_enums_only(field: str, expected: bool) -> None:
    schema = _Judged.model_json_schema()

    assert _is_enum_valued(schema["properties"][field], schema["$defs"]) is expected


def test_model_fields_with_context_dependent_tier_are_tagged() -> None:
    # A model whose tier depends on where it is used (EpochDistribution's fields
    # are tagged derived but it is experimental under mid_insp_flattening) relies
    # on every referencing field carrying the tier for it.
    offenders: set[str] = set()
    for surface, objects in _surfaces():
        props = [
            (model, field, prop)
            for model, schema in objects
            for field, prop in schema.get("properties", {}).items()
        ]
        context_models = {
            ref
            for _, _, prop in props
            if "x-provenance" in prop
            for ref in _model_refs(prop)
        }
        offenders |= {
            f"{surface} {model}.{field}"
            for model, field, prop in props
            if _model_refs(prop) & context_models
            and "x-provenance" not in prop
            and not _is_exempt(model, field)
        }
    assert not offenders, (
        "Fields holding a model that inherits its tier from the parent field, "
        "missing x-provenance:\n" + "\n".join(sorted(offenders)) + f"\n{_FIX_HINT}"
    )


def test_default_rest_operations_return_no_experimental_field() -> None:
    components = _openapi()["components"]["schemas"]
    offenders = sorted(
        {
            f"{route} -> {model}.{field}"
            for route, op in _operations(experimental=False).items()
            for model in _reachable(op.get("responses", {}))
            for field, prop in components[model].get("properties", {}).items()
            if prop.get("x-provenance") == Provenance.EXPERIMENTAL
        }
    )
    assert not offenders, (
        "Default REST operations returning Experimental fields:\n"
        + "\n".join(offenders)
    )


def test_experimental_operation_list_is_pinned() -> None:
    assert set(_operations(experimental=True)) == EXPERIMENTAL_OPERATIONS


def test_untyped_metric_routes_match_known_gap_list() -> None:
    untyped = _untyped_routes() - _OPERATIONAL_UNTYPED_ROUTES

    assert untyped == _UNTYPED_METRIC_ROUTES, (
        "Untyped REST routes changed. A newly typed route: remove it from "
        "_UNTYPED_METRIC_ROUTES. A new untyped route: give it a response model "
        "(preferred) or list it in _UNTYPED_METRIC_ROUTES / "
        "_OPERATIONAL_UNTYPED_ROUTES."
    )


def test_tagged_descriptions_start_with_matching_tier_prefix() -> None:
    mismatched = sorted(
        {
            f"{surface} {model}.{field}"
            for surface, model, field, prop in _object_properties()
            if "x-provenance" in prop
            and not prop.get("description", "").startswith(
                description_prefix(Provenance(prop["x-provenance"]))
            )
        }
    )
    assert not mismatched, "Description prefix != x-provenance tier:\n" + "\n".join(
        mismatched
    )


class _Leaf(BaseModel):
    value: float = provenance_field(Provenance.DERIVED, "Value.")


class _Tree(BaseModel):
    by_name: dict[str, _Leaf] = provenance_field(
        Provenance.EXPERIMENTAL, "Leaves by name.", default_factory=dict
    )
    children: list["_Tree"] = []


def test_response_provenance_walks_dict_values_and_stops_on_self_reference() -> None:
    assert response_provenance(_Tree) == {"experimental": ["by_name{}.value"]}


def test_response_provenance_returns_independent_copies() -> None:
    response_provenance(_Tree)["experimental"].append("mutated")

    assert response_provenance(_Tree)["experimental"] == ["by_name{}.value"]


def test_weakest_picks_least_certain_tier() -> None:
    assert _weakest(Provenance.DEVICE, Provenance.EXPERIMENTAL) == (
        Provenance.EXPERIMENTAL
    )
    assert _weakest(Provenance.DERIVED, Provenance.DEVICE) == Provenance.DERIVED


def test_field_description_strips_tier_prefix() -> None:
    assert _Leaf.model_fields["value"].description == (
        f"{description_prefix(Provenance.DERIVED)}Value."
    )
    assert field_description(_Leaf, "value") == "Value."


@pytest.mark.parametrize(
    "field",
    [
        f"{channel}_{stat}"
        for channel in ("pressure", "epap", "leak")
        for stat in ("max", "median", "95th")
    ],
)
def test_waveform_recomputed_statistic_is_derived(field: str) -> None:
    assert field_provenance(SessionStatistics, field) is Provenance.DERIVED
    assert "recomputed from the waveform" in field_description(SessionStatistics, field)


def test_rest_waveform_values_are_tagged_device() -> None:
    assert field_provenance(WaveformDataResponse, "values") == Provenance.DEVICE


@pytest.mark.parametrize("field", ["effectiveness", "ahi_trend_direction"])
def test_snore_judgements_are_tagged_experimental(field: str) -> None:
    assert field_provenance(ExperimentalStats, field) == Provenance.EXPERIMENTAL
    prop = ExperimentalStats.model_json_schema()["properties"][field]
    assert _is_enum_valued(prop, {})
