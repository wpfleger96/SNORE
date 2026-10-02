"""Conflict detection and rendering in the UI provenance-map generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.export_provenance_ts import (
    ProvenanceConflictError,
    build_provenance_map,
    render_provenance_ts,
    ts_fingerprint,
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
    assert result.source_fields == {"mv": "mv_source"}
    assert result.schema_field_tiers == {}


def test_nested_inline_object_properties_are_not_api_fields():
    # Only top-level properties are addressable fields; a nested inline object's
    # "x" must not land in the map (nor conflict with a real top-level "x").
    items = {"properties": {"x": _prop("experimental")}}
    inner = {"anyOf": [{"type": "array", "items": items}, {"type": "null"}]}
    spec = _spec(Outer={"inner": inner}, Other={"x": _prop("device")})

    result = build_provenance_map(spec, ambiguous={})

    assert result.field_tiers == {"x": Provenance.DEVICE}


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
    assert result.source_fields == {"A.ahi": "index_source"}


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
            "ahi": _prop("device", "index_source"),
        },
        B={"ahi": _prop("derived")},
    )

    rendered = render_provenance_ts(build_provenance_map(spec, ambiguous={"ahi": "x"}))

    assert rendered.startswith("// generated — do not edit.")
    assert rendered.index('"alpha": "experimental"') < rendered.index(
        '"zeta": "derived"'
    )
    assert '"A.ahi": "device",' in rendered
    assert '"A.ahi": "index_source",' in rendered
    assert (
        'export const PROVENANCE_TIERS = ["device", "derived", "experimental"] as const'
        in rendered
    )


def test_render_escapes_quotes_and_control_characters():
    rendered = render_provenance_ts(
        build_provenance_map(_spec(A={'odd"\n\u0007': _prop("derived")}), ambiguous={})
    )

    assert '"odd\\"\\n\\u0007": "derived"' in rendered


def test_fingerprint_ignores_prettier_reformatting():
    raw = 'export const X: Record<string, string> = {\n  "a": "it\'s",\n}\n'
    formatted = 'export const X: Record<string, string> = {\n    a: "it\'s",\n}\n'

    assert ts_fingerprint(raw) == ts_fingerprint(formatted)
    assert ts_fingerprint(raw) != ts_fingerprint(raw.replace("it's", "its!"))


def test_real_api_schema_has_no_unlisted_conflicts():
    from snore.api.app import create_app

    result = build_provenance_map(create_app().openapi())

    assert result.field_tiers
    assert result.source_fields["DayDetail.ahi"] == "index_source"


def test_committed_provenance_file_matches_real_api_schema():
    """Catches a stale ``provenance.generated.ts`` in ``just test`` (no Prettier needed)."""
    from snore.api.app import create_app

    committed = (
        Path(__file__).resolve().parents[2]
        / "ui"
        / "src"
        / "types"
        / "provenance.generated.ts"
    )
    expected = render_provenance_ts(build_provenance_map(create_app().openapi()))

    assert ts_fingerprint(committed.read_text(encoding="utf-8")) == ts_fingerprint(
        expected
    ), (
        "ui/src/types/provenance.generated.ts is stale; regenerate with `just ui-generate-types`"
    )
