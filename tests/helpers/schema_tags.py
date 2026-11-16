"""JSON-schema checks for experimental tags."""

from __future__ import annotations


def has_experimental_tag(schema: object) -> bool:
    """Whether any node of a JSON schema, ``$defs`` included, is tagged experimental."""
    if isinstance(schema, dict):
        return schema.get("x-provenance") == "experimental" or any(
            has_experimental_tag(value) for value in schema.values()
        )
    if isinstance(schema, list):
        return any(has_experimental_tag(value) for value in schema)
    return False
