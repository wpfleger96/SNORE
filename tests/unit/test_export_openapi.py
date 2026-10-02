"""The UI provenance map that ``scripts/export_openapi.py`` writes."""

from __future__ import annotations

from pathlib import Path

from scripts.export_openapi import dump_json, provenance_map
from snore.api.app import create_app


def test_provenance_map_keys_top_level_tagged_fields_by_schema():
    spec = {
        "components": {
            "schemas": {
                "Day": {
                    "properties": {
                        "ahi": {"x-provenance": "device", "x-provenance-source": "src"},
                        "rdi": {"x-provenance": "derived"},
                        "date": {"type": "string"},
                        "nested": {
                            "type": "object",
                            "properties": {"x": {"x-provenance": "experimental"}},
                        },
                    }
                },
                "Empty": {"type": "string"},
            }
        }
    }

    result = provenance_map(spec)

    assert result["fields"] == {"Day.ahi": "device", "Day.rdi": "derived"}
    assert result["sources"] == {"Day.ahi": "src"}
    assert set(result["notes"]) == {"device", "derived", "experimental"}


def test_committed_provenance_json_matches_api_schema():
    committed = Path(__file__).resolve().parents[2] / "ui/src/types/provenance.json"

    assert committed.read_text(encoding="utf-8") == dump_json(
        provenance_map(create_app().openapi())
    ), "ui/src/types/provenance.json is stale; regenerate with `just ui-generate-types`"
