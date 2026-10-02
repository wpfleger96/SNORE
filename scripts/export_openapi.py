"""Export the SNORE FastAPI OpenAPI spec as JSON.

Usage:
    uv run python scripts/export_openapi.py [OUTPUT_PATH [PROVENANCE_PATH]]

Writes the spec to OUTPUT_PATH if given, otherwise to stdout. With
PROVENANCE_PATH, also writes the UI's provenance map (see
:func:`provenance_map`). Building the app does not run its lifespan, so no
database is initialized or touched.
"""

from __future__ import annotations

import json
import sys

from pathlib import Path
from typing import Any

from snore.api.app import create_app
from snore.provenance import PROVENANCE_NOTES


def provenance_map(spec: dict[str, Any]) -> dict[str, Any]:
    """Tier and per-value source sibling of every tagged ``Schema.field``.

    Reads the ``x-provenance`` / ``x-provenance-source`` keys that
    ``provenance_field`` puts on each component schema's top-level properties.
    """
    fields: dict[str, str] = {}
    sources: dict[str, str] = {}
    for schema_name, schema in spec.get("components", {}).get("schemas", {}).items():
        for name, prop in schema.get("properties", {}).items():
            key = f"{schema_name}.{name}"
            if "x-provenance" in prop:
                fields[key] = prop["x-provenance"]
            if "x-provenance-source" in prop:
                sources[key] = prop["x-provenance-source"]
    notes = {tier.value: note for tier, note in PROVENANCE_NOTES.items()}
    return {"notes": notes, "fields": fields, "sources": sources}


def dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def main() -> int:
    spec = create_app().openapi()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(dump_json(spec), encoding="utf-8")
    else:
        sys.stdout.write(dump_json(spec))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(dump_json(provenance_map(spec)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
