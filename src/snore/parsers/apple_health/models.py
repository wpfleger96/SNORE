"""Canonical record model for Apple Health data from the export.xml reader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class RawHealthRecord:
    """One Apple Health record, normalised from the export.xml reader.

    The downstream importer layer consumes this type directly.

    Canonical sleep stage names (``value_text``):
        ``InBed``, ``AsleepUnspecified``, ``Awake``, ``AsleepCore``,
        ``AsleepDeep``, ``AsleepREM``.
    """

    record_type: str
    """Canonical HK identifier, e.g. ``HKCategoryTypeIdentifierSleepAnalysis``."""

    source_name: str

    source_version: str | None

    device_info: str | None

    start_time: datetime
    """Naive local wall-clock datetime."""

    end_time: datetime
    """Naive; point-samples have ``start_time == end_time``."""

    value_text: str | None
    """Canonical stage name for sleep records; ``None`` for quantity records."""

    value_num: float | None
    """Rounded to 4 decimal places at parse; ``None`` for category records."""

    unit: str | None

    utc_offset_seconds: int | None

    night_date: date
    """``session_night`` of ``start_time`` (noon split), the CPAP ``Day.date`` rule."""

    ingest_channel: str
    """Provenance tag; currently always ``"export_xml"``.  Future channels may add values."""
