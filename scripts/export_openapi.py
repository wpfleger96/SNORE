"""Export the SNORE FastAPI OpenAPI spec as JSON.

Usage:
    uv run python scripts/export_openapi.py [OUTPUT_PATH]

Writes the spec to OUTPUT_PATH if given, otherwise to stdout. Building the
app does not run its lifespan, so no database is initialized or touched.
"""

from __future__ import annotations

import json
import sys

from pathlib import Path

from snore.api.app import create_app


def main() -> int:
    spec = create_app().openapi()
    payload = json.dumps(spec, indent=2, sort_keys=True)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
