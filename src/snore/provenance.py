"""Provenance tiers for every metric SNORE reports.

Each metric is one of three tiers:

- ``DEVICE``: verbatim data from the recording device (device-scored events,
  waveforms, STR values, oximeter or Apple Health samples).
- ``DERIVED``: deterministic math on device data (index recounts, percentiles,
  usage, roll-ups).
- ``EXPERIMENTAL``: SNORE's own detection and classification heuristics.

Aggregates inherit the weakest input tier: anything built on an experimental
input stays experimental; aggregates of device/derived inputs are derived.
One exception: a sum of device-scored event counts (across sessions, days, or
a validation run) stays device, since it adds no SNORE judgement.  Rates,
means, percentiles and recounts of those events are derived.

The declared tier is the preferred source.  Some fields are recomputed from
the waveform when it is present and fall back to the device's own summary
value otherwise (e.g. ResMed pressure/EPAP/leak percentiles); they are tagged
with the recomputed tier and their description says so.

Response fields declare their tier with :func:`provenance_field`, which both
prefixes the description (visible to LLMs and in generated TS JSDoc) and adds a
machine-readable ``x-provenance`` JSON-schema key (``docs://schemas``, OpenAPI).
Presentation layers read a field's tier back with :func:`field_provenance`, and
:func:`response_provenance` lists a whole response model's non-device paths.
"""

from enum import StrEnum
from functools import cache
from typing import Any

from pydantic import BaseModel, Field


class Provenance(StrEnum):
    """Tiers in order of increasing uncertainty."""

    DEVICE = "device"
    DERIVED = "derived"
    EXPERIMENTAL = "experimental"

    @classmethod
    def weakest(cls, *tiers: "Provenance") -> "Provenance":
        """The least certain of ``tiers`` (what an aggregate of them inherits)."""
        return max(tiers, key=list(cls).index)


PROVENANCE_NOTES: dict[Provenance, str] = {
    Provenance.DEVICE: (
        "Reported by the recording device (CPAP, oximeter, or Apple Health source)."
    ),
    Provenance.DERIVED: "Computed by SNORE from device data.",
    Provenance.EXPERIMENTAL: (
        "SNORE's own heuristic analysis. Experimental trend instrument; not "
        "validated against device-scored events; not clinically validated."
    ),
}


def provenance_field(
    provenance: Provenance,
    description: str,
    *,
    source_field: str | None = None,
    **kwargs: Any,
) -> Any:
    """Pydantic ``Field`` tagged with a provenance tier.

    ``source_field`` names a sibling field whose value says which tier a given
    value actually came from, for fields whose tier varies per value (the
    declared ``provenance`` is then the preferred tier).
    """
    extra: dict[str, str] = {"x-provenance": provenance.value}
    if source_field is not None:
        extra["x-provenance-source"] = source_field
    return Field(
        description=f"[{provenance.value.upper()}] {description}",
        json_schema_extra=extra,  # type: ignore[arg-type]
        **kwargs,
    )


def field_provenance(model: type[BaseModel], name: str) -> Provenance:
    """Declared tier of ``model.name`` (raises ``KeyError`` if untagged)."""
    extra = model.model_fields[name].json_schema_extra
    if not isinstance(extra, dict) or "x-provenance" not in extra:
        raise KeyError(f"{model.__name__}.{name} has no provenance tag")
    return Provenance(str(extra["x-provenance"]))


def response_provenance(model: type[BaseModel]) -> dict[str, Any]:
    """Dotted paths of a response model's non-device fields, by tier.

    Shape: ``{"experimental": [...], "derived": [...], "source_dependent":
    {path: sibling_path}}``; ``[]`` marks list items (``nights[].rdi``), ``{}``
    marks dict values, and empty keys are omitted.  A field nested in a tagged
    parent takes the weaker of the two tiers.  ``source_dependent`` lists
    fields whose tier is set per value by a sibling field (e.g. ``mv_source``).
    Returns a fresh dict on every call; the schema walk itself is cached.
    """
    experimental, derived, source_dependent = _provenance_paths(model)
    block: dict[str, Any] = {}
    if experimental:
        block[Provenance.EXPERIMENTAL.value] = list(experimental)
    if derived:
        block[Provenance.DERIVED.value] = list(derived)
    if source_dependent:
        block["source_dependent"] = dict(source_dependent)
    return block


_Object = tuple[dict[str, Any], str, frozenset[str]]


@cache
def _provenance_paths(
    model: type[BaseModel],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    schema = model.model_json_schema()
    defs: dict[str, Any] = schema.get("$defs", {})
    paths: dict[Provenance, list[str]] = {p: [] for p in Provenance}
    source_dependent: list[tuple[str, str]] = []

    def objects(node: dict[str, Any], path: str, seen: frozenset[str]) -> list[_Object]:
        # ``seen`` holds the $refs on the current path, so a self-referencing
        # model stops instead of recursing forever.
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name in seen:
                return []
            node, seen = defs[name], seen | {name}
        found = [(node, path, seen)] if "properties" in node else []
        for key in ("anyOf", "allOf", "oneOf"):
            for sub in node.get(key, []):
                found += objects(sub, path, seen)
        for key, suffix in (("items", "[]"), ("additionalProperties", "{}")):
            if isinstance(node.get(key), dict):
                found += objects(node[key], f"{path}{suffix}", seen)
        return found

    def walk(
        node: dict[str, Any], path: str, inherited: Provenance, seen: frozenset[str]
    ) -> None:
        own = node.get("x-provenance")
        tier = Provenance.weakest(inherited, Provenance(own or inherited))
        nested = objects(node, path, seen)
        if not nested:
            if own is not None:
                paths[tier].append(path)
            if source := node.get("x-provenance-source"):
                parent = path.rpartition(".")[0]
                source_dependent.append((path, f"{parent}.{source}".lstrip(".")))
            return
        for obj, obj_path, obj_seen in nested:
            for name, prop in obj["properties"].items():
                walk(prop, f"{obj_path}.{name}" if obj_path else name, tier, obj_seen)

    walk(schema, "", Provenance.DEVICE, frozenset())
    return (
        tuple(paths[Provenance.EXPERIMENTAL]),
        tuple(paths[Provenance.DERIVED]),
        tuple(source_dependent),
    )
