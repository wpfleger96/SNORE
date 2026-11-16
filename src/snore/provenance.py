"""Experimental tagging for the metrics SNORE reports.

A field is experimental or it is not.  Untagged fields are device data or
deterministic math on device data.  Experimental fields are SNORE's own
detection/classification heuristics and invented measures or judgements (FL
classes, RERA proxy, breath segmentation features, MV slope and stability,
programmatic event detection, SNORE-vs-device agreement scores, the therapy
effectiveness band, AHI trend direction).  Anything built on an experimental
input is experimental, and a field nested under an experimental field is
experimental too.

Fields whose value can be the device's own number or SNORE's recount say which
through a sibling data field typed :class:`IndexSource` (``index_source``,
``rr_mean_source`` and similar); that is data, not a tag.

Experimental fields are declared with :func:`experimental_field` (computed
fields with :func:`experimental_computed_field`), which prefixes the
description (visible to LLMs and in generated TS JSDoc) and adds a
machine-readable ``x-provenance`` JSON-schema key (``docs://schemas``,
OpenAPI).  Presentation layers read the tag back with :func:`is_experimental`.
"""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field
from pydantic.fields import ComputedFieldInfo, FieldInfo

EXPERIMENTAL_NOTE = (
    "SNORE's own heuristic; an experimental trend instrument, not clinically validated."
)
EXPERIMENTAL_PREFIX = "[EXPERIMENTAL] "
_TAG = {"x-provenance": "experimental"}


class IndexSource(StrEnum):
    """``device``: the device-reported value; ``derived``: SNORE's recount or computed fallback."""

    DEVICE = "device"
    DERIVED = "derived"


def experimental_field(description: str, **kwargs: Any) -> Any:
    """Pydantic ``Field`` tagged experimental."""
    return Field(
        description=f"{EXPERIMENTAL_PREFIX}{description}",
        json_schema_extra=dict(_TAG),
        **kwargs,
    )


def experimental_computed_field[T](description: str) -> Callable[[T], T]:
    """Pydantic ``computed_field`` decorator tagged like :func:`experimental_field`."""
    return computed_field(
        description=f"{EXPERIMENTAL_PREFIX}{description}",
        json_schema_extra=dict(_TAG),
    )


def _field_info(model: type[BaseModel], name: str) -> FieldInfo | ComputedFieldInfo:
    """A stored or computed field's info (raises ``KeyError`` if neither)."""
    if name in model.model_computed_fields:
        return model.model_computed_fields[name]
    return model.model_fields[name]


def is_experimental(model: type[BaseModel], name: str) -> bool:
    """Whether ``model.name`` is tagged experimental (``KeyError`` if no such field)."""
    extra = _field_info(model, name).json_schema_extra
    return isinstance(extra, dict) and extra.get("x-provenance") == "experimental"
