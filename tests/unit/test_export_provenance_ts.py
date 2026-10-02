"""Conflict detection and rendering in the UI provenance-map generator."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.export_provenance_ts import (
    ProvenanceConflictError,
    build_provenance_map,
    render_provenance_ts,
)
from snore.provenance import Provenance


def _prop(tier: str, source: str | None = None) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": "number", "x-provenance": tier}
    if source is not None:
        prop["x-provenance-source"] = source
    return prop


def _spec(**schemas: dict[str, Any]) -> dict[str, Any]:
    return {
        "components": {
            "schemas": {name: {"properties": props} for name, props in schemas.items()}
        }
    }


def test_consistent_field_across_schemas_maps_by_bare_name():
    spec = _spec(
        A={"rdi": _prop("derived"), "id": {"type": "integer"}},
        B={"rdi": _prop("derived"), "mv": _prop("device", "mv_source")},
    )

    result = build_provenance_map(spec, ambiguous={})

    assert result.field_tiers == {"mv": Provenance.DEVICE, "rdi": Provenance.DERIVED}
    assert result.field_sources == {"mv": "mv_source"}
    assert result.schema_field_tiers == {}


def test_nested_inline_and_anyof_properties_are_collected():
    items = {"properties": {"x": _prop("experimental")}}
    inner = {"anyOf": [{"type": "array", "items": items}, {"type": "null"}]}
    spec = _spec(Outer={"inner": inner})

    result = build_provenance_map(spec, ambiguous={})

    assert result.field_tiers == {"x": Provenance.EXPERIMENTAL}


def test_tier_conflict_raises_naming_field_and_schemas():
    spec = _spec(A={"ahi": _prop("device")}, B={"ahi": _prop("derived")})

    with pytest.raises(ProvenanceConflictError, match="ahi: device in A; derived in B"):
        build_provenance_map(spec, ambiguous={})


def test_source_conflict_raises_even_when_tiers_match():
    spec = _spec(
        A={"ahi": _prop("device", "index_source")},
        B={"ahi": _prop("device")},
    )

    with pytest.raises(ProvenanceConflictError, match="source=index_source"):
        build_provenance_map(spec, ambiguous={})


def test_ambiguous_field_is_emitted_per_schema():
    spec = _spec(
        A={"ahi": _prop("device", "index_source")},
        B={"ahi": _prop("derived")},
    )

    result = build_provenance_map(spec, ambiguous={"ahi": "headline vs recount"})

    assert "ahi" not in result.field_tiers
    assert result.schema_field_tiers == {
        "A.ahi": Provenance.DEVICE,
        "B.ahi": Provenance.DERIVED,
    }
    assert result.schema_field_sources == {"A.ahi": "index_source"}


def test_stale_ambiguous_entries_raise():
    spec = _spec(A={"ahi": _prop("derived")}, B={"ahi": _prop("derived")})

    with pytest.raises(ProvenanceConflictError) as exc_info:
        build_provenance_map(spec, ambiguous={"ahi": "x", "gone": "y"})

    message = str(exc_info.value)
    assert "ahi: listed in AMBIGUOUS_FIELDS but no longer conflicts" in message
    assert "gone: listed in AMBIGUOUS_FIELDS but not tagged in any schema" in message


def test_render_is_sorted_and_quotes_qualified_keys():
    spec = _spec(
        A={
            "zeta": _prop("derived"),
            "alpha": _prop("experimental"),
            "ahi": _prop("device"),
        },
        B={"ahi": _prop("derived")},
    )

    rendered = render_provenance_ts(build_provenance_map(spec, ambiguous={"ahi": "x"}))

    assert rendered.startswith("// generated — do not edit.")
    assert rendered.index("alpha: 'experimental'") < rendered.index("zeta: 'derived'")
    assert "'A.ahi': 'device'," in rendered
    assert "FIELD_PROVENANCE_SOURCE: Record<string, string> = {}" in rendered


def test_real_api_schema_has_no_unlisted_conflicts():
    from snore.api.app import create_app

    result = build_provenance_map(create_app().openapi())

    assert result.field_tiers
    assert result.schema_field_sources["DayDetail.ahi"] == "index_source"
