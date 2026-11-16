"""Experimental tags are well-formed, and default surfaces carry none.

A field is tagged experimental (see ``snore.provenance``) or it is untagged:
device data or computed from it.  This walks the JSON schema of every MCP
response model (``SCHEMA_MODEL_MAP``) and every OpenAPI schema a REST route can
return, and checks that every ``x-provenance`` value is ``experimental``, that
the tag and the ``[EXPERIMENTAL]`` description prefix always appear together,
and that a model held by a tagged field is never held by an untagged one.  It
also enforces the REST gating rule (no REST operation without the
``experimental`` tag returns an experimental field) and keeps experimental
columns out of exports.  See AGENTS.md "Provenance tagging".
"""

import json

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from pydantic import BaseModel

from snore.analysis.modes.types import ModeResult
from snore.api.app import create_app
from snore.mcp.schemas import SCHEMA_MODEL_MAP
from snore.metrics import EXPORT_STAT_KEYS
from snore.provenance import (
    EXPERIMENTAL_PREFIX,
    experimental_field,
    is_experimental,
)
from snore.services.schemas import ExperimentalStats, SessionDetail, SessionStatistics

pytestmark = pytest.mark.unit

# Untyped REST responses that are not therapy metrics: OAuth redirects, file
# downloads, progress streams, and deletion counts.
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

# Known gap: metric routes with no JSON schema for the gate to inspect.  Both
# return a downloadable HTML report (HTMLResponse), not JSON.
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


def _untagged_holders_of_experimental_models(
    objects: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    """``model.field`` of untagged properties holding a model a tagged property holds.

    A model with no tags of its own is experimental through the field that holds
    it, so every other field holding it must carry the tag too.
    """
    props = [
        (model, field, prop)
        for model, schema in objects
        for field, prop in schema.get("properties", {}).items()
    ]
    held_by_experimental = {
        ref
        for _, _, prop in props
        if prop.get("x-provenance") == "experimental"
        for ref in _refs(prop)
    }
    return sorted(
        f"{model}.{field}"
        for model, field, prop in props
        if "x-provenance" not in prop and _refs(prop) & held_by_experimental
    )


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


def _operations(*, experimental: bool) -> dict[str, dict[str, Any]]:
    return {
        f"{method.upper()} {path}": op
        for path, ops in _openapi()["paths"].items()
        for method, op in ops.items()
        if ("experimental" in op.get("tags", [])) is experimental
    }


def _offenders(failing: Callable[[dict[str, Any]], bool]) -> list[str]:
    """``surface model.field`` of every response property ``failing`` flags."""
    return sorted(
        {
            f"{surface} {model}.{field}"
            for surface, model, field, prop in _object_properties()
            if failing(prop)
        }
    )


def test_default_rest_operations_return_no_experimental_field() -> None:
    components = _openapi()["components"]["schemas"]
    offenders = sorted(
        {
            f"{route} -> {model}.{field}"
            for route, op in _operations(experimental=False).items()
            for model in _reachable(op.get("responses", {}))
            for field, prop in components[model].get("properties", {}).items()
            if prop.get("x-provenance") == "experimental"
        }
    )
    assert not offenders, (
        "Default REST operations returning experimental fields:\n"
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


def test_response_schemas_tag_experimental_only() -> None:
    offenders = _offenders(
        lambda prop: prop.get("x-provenance", "experimental") != "experimental"
    )
    assert not offenders, (
        "Fields with an x-provenance value other than 'experimental':\n"
        + "\n".join(offenders)
    )


def test_tagged_fields_and_experimental_descriptions_match() -> None:
    offenders = _offenders(
        lambda prop: (
            ("x-provenance" in prop)
            != prop.get("description", "").startswith(EXPERIMENTAL_PREFIX)
        )
    )
    assert not offenders, (
        "Fields whose x-provenance tag and [EXPERIMENTAL] description prefix "
        "disagree (use experimental_field):\n" + "\n".join(offenders)
    )


def test_model_held_by_experimental_field_is_never_held_untagged() -> None:
    offenders = [
        f"{surface} {holder}"
        for surface, objects in _surfaces()
        for holder in _untagged_holders_of_experimental_models(objects)
    ]
    assert not offenders, (
        "Untagged fields holding a model that an experimental field also holds "
        "(use experimental_field):\n" + "\n".join(offenders)
    )


def test_untagged_holder_check_sees_list_and_dict_holders() -> None:
    class Leaf(BaseModel):
        value: float = 0.0

    class ExperimentalHolder(BaseModel):
        leaf: Leaf = experimental_field("A heuristic.", default_factory=Leaf)

    class PlainHolder(BaseModel):
        leaves: list[Leaf] = []
        by_name: dict[str, Leaf] = {}

    objects = [
        (model.__name__, model.model_json_schema())
        for model in (Leaf, ExperimentalHolder, PlainHolder)
    ]

    assert _untagged_holders_of_experimental_models(objects) == [
        "PlainHolder.by_name",
        "PlainHolder.leaves",
    ]


@pytest.mark.parametrize(
    "field",
    [
        f"{channel}_{stat}"
        for channel in ("pressure", "epap", "leak")
        for stat in ("max", "median", "95th")
    ],
)
def test_waveform_recomputed_statistic_description_names_its_fallback(
    field: str,
) -> None:
    description = SessionStatistics.model_fields[field].description or ""

    assert (
        "recomputed from the waveform when available, otherwise the device's value"
        in description
    )


class _Tagged(BaseModel):
    guess: float = experimental_field("A guess.")
    reading: float = 0.0


def test_experimental_field_prefixes_description_and_tags_schema() -> None:
    prop = _Tagged.model_json_schema()["properties"]["guess"]

    assert prop["description"] == f"{EXPERIMENTAL_PREFIX}A guess."
    assert prop["x-provenance"] == "experimental"


def test_is_experimental_reads_the_tag() -> None:
    assert is_experimental(_Tagged, "guess") is True
    assert is_experimental(_Tagged, "reading") is False


def test_is_experimental_missing_field_raises_key_error() -> None:
    with pytest.raises(KeyError):
        is_experimental(_Tagged, "missing")


def test_is_experimental_reads_a_computed_field() -> None:
    assert is_experimental(ModeResult, "apnea_count") is True


@pytest.mark.parametrize("field", ["effectiveness", "ahi_trend_direction"])
def test_snore_judgements_are_tagged_experimental(field: str) -> None:
    assert is_experimental(ExperimentalStats, field) is True


def test_export_columns_carry_no_experimental_field() -> None:
    """Exports are a default surface: an experimental column would leak past the gate."""
    assert EXPORT_STAT_KEYS
    columns: list[tuple[type[BaseModel], str]] = [
        (SessionDetail, "duration_hours"),
        *((SessionStatistics, key) for key in EXPORT_STAT_KEYS),
    ]

    experimental = [
        f"{model.__name__}.{field}"
        for model, field in columns
        if is_experimental(model, field)
    ]

    assert experimental == []
