"""Generate ``ui/src/types/provenance.generated.ts`` from the OpenAPI spec.

Usage:
    uv run python scripts/export_provenance_ts.py OPENAPI_JSON OUTPUT_TS

Reads the ``x-provenance`` / ``x-provenance-source`` keys that
``snore.provenance.provenance_field`` writes into every tagged field's JSON
schema and emits field-name-keyed lookup maps for the UI.  The output is
plain TypeScript; ``pnpm run generate:types`` then formats it with Prettier,
so compare a fresh render to the committed file with :func:`ts_fingerprint`.

The maps are keyed by bare field name, which only works while a name means
the same tier everywhere.  A name that carries different tiers (or source
fields) in different schemas is a hard error unless it is listed in
``AMBIGUOUS_FIELDS``; listed names are emitted per schema (``Schema.field``)
instead, and the UI must say which schema the value came from.
"""

from __future__ import annotations

import json
import re
import sys

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from snore.provenance import PROVENANCE_NOTES, Provenance

# Field names that legitimately mean different things in different schemas.
# Each maps to why, so a reviewer can tell a real meaning split from a tagging
# mistake.  An entry that stops conflicting is reported as stale.
AMBIGUOUS_FIELDS: dict[str, str] = {
    "ahi": "Day headline (device, per-value index_source) vs session recount "
    "(derived) vs programmatic detection result (experimental)",
    "oai": "Day headline (device, per-value index_source) vs recount (derived)",
    "cai": "Day headline (device, per-value index_source) vs recount (derived)",
    "hi": "Day headline (device, per-value index_source) vs recount (derived)",
    "duration": "Device/stored event (device, per-value source) vs SNORE-detected "
    "event model (experimental)",
    "duration_hours": "Session length (device) vs flow-waveform coverage of a "
    "validation run (derived)",
}


class ProvenanceConflictError(ValueError):
    """Tagged field names whose tier or source field differs between schemas."""


class ProvenanceMap(BaseModel):
    """Lookup maps emitted into the generated TypeScript module."""

    field_tiers: dict[str, Provenance]
    schema_field_tiers: dict[str, Provenance]
    # Keyed like the tier maps: bare field name, or ``Schema.field`` for the
    # names in ``schema_field_tiers``.
    source_fields: dict[str, str]


def _top_level_properties(schema: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """The ``(name, property schema)`` pairs of a component schema.

    Only top-level properties are API fields; an inline nested object's
    properties are not addressable by name from a response, so they are skipped.
    """
    if not isinstance(schema, dict):
        return
    for name, prop in (schema.get("properties") or {}).items():
        if isinstance(prop, dict):
            yield name, prop


def build_provenance_map(
    openapi: dict[str, Any],
    ambiguous: dict[str, str] | None = None,
) -> ProvenanceMap:
    """Collect field tiers from ``openapi['components']['schemas']``.

    Raises :class:`ProvenanceConflictError` listing every unlisted field name
    whose tier or source differs between schemas, and every ``ambiguous``
    entry that no longer conflicts.
    """
    ambiguous = AMBIGUOUS_FIELDS if ambiguous is None else ambiguous
    # field name -> (tier, source) -> schemas carrying that combination
    seen: dict[str, dict[tuple[Provenance, str | None], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    schemas: dict[str, Any] = openapi.get("components", {}).get("schemas", {})
    for schema_name, schema in schemas.items():
        for name, prop in _top_level_properties(schema):
            if "x-provenance" in prop:
                key = (
                    Provenance(prop["x-provenance"]),
                    prop.get("x-provenance-source"),
                )
                seen[name][key].add(schema_name)

    result = ProvenanceMap(field_tiers={}, schema_field_tiers={}, source_fields={})
    errors: list[str] = []
    for name in sorted(seen):
        variants = seen[name]
        if name not in ambiguous:
            if len(variants) > 1:
                errors.append(f"{name}: {_describe(variants)}")
                continue
            ((tier, source),) = variants
            result.field_tiers[name] = tier
            if source is not None:
                result.source_fields[name] = source
            continue
        if len(variants) == 1:
            errors.append(f"{name}: listed in AMBIGUOUS_FIELDS but no longer conflicts")
            continue
        for (tier, source), schema_names in variants.items():
            for schema_name in schema_names:
                qualified = f"{schema_name}.{name}"
                if qualified in result.schema_field_tiers:
                    errors.append(f"{qualified}: conflicting tags within one schema")
                result.schema_field_tiers[qualified] = tier
                if source is not None:
                    result.source_fields[qualified] = source
    errors += [
        f"{name}: listed in AMBIGUOUS_FIELDS but not tagged in any schema"
        for name in sorted(set(ambiguous) - set(seen))
    ]
    if errors:
        raise ProvenanceConflictError(
            "Provenance tags conflict across schemas (the UI map is keyed by field "
            "name; fix the tag, or add a genuinely different meaning to "
            "AMBIGUOUS_FIELDS in scripts/export_provenance_ts.py):\n  "
            + "\n  ".join(errors)
        )
    result.schema_field_tiers = dict(sorted(result.schema_field_tiers.items()))
    result.source_fields = dict(sorted(result.source_fields.items()))
    return result


def _describe(variants: dict[tuple[Provenance, str | None], set[str]]) -> str:
    parts = []
    order = list(Provenance)
    for (tier, source), schema_names in sorted(
        variants.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1] or "")
    ):
        via = f" (source={source})" if source else ""
        parts.append(f"{tier.value}{via} in {', '.join(sorted(schema_names))}")
    return "; ".join(parts)


def _ts_string(value: str) -> str:
    # JSON string syntax is valid TS and escapes quotes and control characters;
    # Prettier (run by `just ui-generate-types`) rewrites the quoting style.
    return json.dumps(value, ensure_ascii=False)


def _ts_record(doc: str, name: str, record_type: str, entries: dict[str, str]) -> str:
    body = "".join(f"  {_ts_string(k)}: {_ts_string(v)},\n" for k, v in entries.items())
    return f"/** {doc} */\nexport const {name}: {record_type} = {{\n{body}}}\n"


def render_provenance_ts(provenance_map: ProvenanceMap) -> str:
    """The TypeScript module for ``provenance_map`` (valid TS, not yet Prettier-formatted)."""
    tiers = ", ".join(_ts_string(tier.value) for tier in Provenance)
    sections = [
        "// generated — do not edit. Regenerate with `just ui-generate-types`\n"
        "// (scripts/export_provenance_ts.py reads x-provenance from the OpenAPI schemas).\n",
        "/** Every tier, strongest (device) to weakest (experimental). */\n"
        f"export const PROVENANCE_TIERS = [{tiers}] as const\n\n"
        "export type Provenance = (typeof PROVENANCE_TIERS)[number]\n",
        _ts_record(
            "One-line definition of each tier (snore.provenance.PROVENANCE_NOTES).",
            "PROVENANCE_NOTES",
            "Record<Provenance, string>",
            {tier.value: note for tier, note in PROVENANCE_NOTES.items()},
        ),
        _ts_record(
            "Tier of each field name that means the same thing in every schema.",
            "FIELD_PROVENANCE",
            "Record<string, Provenance>",
            {k: v.value for k, v in provenance_map.field_tiers.items()},
        ),
        _ts_record(
            "Tier per `Schema.field` for names whose meaning differs by schema.",
            "SCHEMA_FIELD_PROVENANCE",
            "Record<string, Provenance>",
            {k: v.value for k, v in provenance_map.schema_field_tiers.items()},
        ),
        _ts_record(
            "Sibling field whose per-value content sets the tier, keyed by field name "
            "or `Schema.field`.",
            "PROVENANCE_SOURCE_FIELDS",
            "Record<string, string>",
            provenance_map.source_fields,
        ),
    ]
    return "\n".join(sections)


def ts_fingerprint(text: str) -> str:
    """``text`` minus everything Prettier may change (whitespace, quotes, commas, semicolons).

    Compares a raw render against the committed Prettier-formatted file
    without running Prettier.
    """
    return re.sub(r"[\s'\",;]", "", text)


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__ or "")
        return 2
    openapi = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        provenance_map = build_provenance_map(openapi)
    except ProvenanceConflictError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    Path(sys.argv[2]).write_text(render_provenance_ts(provenance_map), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
