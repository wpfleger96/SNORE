"""Type registry and per-record parsing logic for Apple Health data.

To add a future quantity metric: add one entry to ``QUANTITY_TYPES`` (HK
identifier → expected unit, or ``None`` to accept whatever arrives).  No other
changes required.
"""

from __future__ import annotations

from datetime import datetime
from xml.etree.ElementTree import Element

from snore.parsers.apple_health.models import RawHealthRecord
from snore.parsers.base import session_night

SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
SPO2_TYPE = "HKQuantityTypeIdentifierOxygenSaturation"

QUANTITY_TYPES: dict[str, str | None] = {
    SPO2_TYPE: "%",
    "HKQuantityTypeIdentifierRespiratoryRate": "count/min",
    # Real exports write unit "count" (a non-integer nightly value); accept whatever arrives.
    "HKQuantityTypeIdentifierAppleSleepingBreathingDisturbances": None,
}

# Maps all 7 XML sleep value strings to canonical stage names.
# Legacy pre-watchOS-9 "Asleep" treated as AsleepUnspecified per spec.
XML_SLEEP_VALUE_MAP: dict[str, str] = {
    "HKCategoryValueSleepAnalysisInBed": "InBed",
    "HKCategoryValueSleepAnalysisAsleep": "AsleepUnspecified",  # legacy alias
    "HKCategoryValueSleepAnalysisAsleepUnspecified": "AsleepUnspecified",
    "HKCategoryValueSleepAnalysisAwake": "Awake",
    "HKCategoryValueSleepAnalysisAsleepCore": "AsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep": "AsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM": "AsleepREM",
}

_HANDLED_TYPES: frozenset[str] = frozenset({SLEEP_TYPE} | set(QUANTITY_TYPES))


def is_handled_type(hk_type: str) -> bool:
    """Return ``True`` if this HK type identifier is processed by v1 of the parser."""
    return hk_type in _HANDLED_TYPES


def _parse_hk_timestamp(s: str) -> tuple[datetime, int]:
    """Parse ``'YYYY-MM-DD HH:MM:SS ±HHMM'`` → ``(naive wall-clock, utc_offset_seconds)``."""
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")
    offset_secs = int(dt.utcoffset().total_seconds())  # type: ignore[union-attr]
    return dt.replace(tzinfo=None), offset_secs


def _spo2_percent(value: float) -> float:
    """Return an SpO2 value on the percent scale.

    Sources write SpO2 as a fraction (0.95) or a percent (95), both with unit
    ``%``.  The plausible ranges are disjoint, so a value in [0.5, 1.0] is a
    fraction and is scaled.  Anything else is kept as recorded; night aggregates
    drop values outside 50-100 %.  Migration 022 applies the same rule to rows
    imported before this conversion existed.
    """
    return round(value * 100, 4) if 0.5 <= value <= 1.0 else value


def parse_xml_record(elem: Element) -> RawHealthRecord | None:
    """Parse a ``<Record>`` Element into a ``RawHealthRecord``.

    Returns ``None`` for unhandled types or unparseable records.  The caller is
    responsible for counting skips and calling ``elem.clear()``.
    """
    hk_type = elem.get("type", "")
    if not is_handled_type(hk_type):
        return None

    try:
        start_time, utc_offset = _parse_hk_timestamp(elem.get("startDate", ""))
        end_time, _ = _parse_hk_timestamp(elem.get("endDate", ""))
    except ValueError:
        return None

    raw_value = elem.get("value", "")
    value_text: str | None = None
    value_num: float | None = None

    if hk_type == SLEEP_TYPE:
        value_text = XML_SLEEP_VALUE_MAP.get(raw_value)
        if value_text is None:
            return None  # unknown sleep value string — skip
    else:
        try:
            value_num = round(float(raw_value), 4)
        except ValueError:
            return None
        if hk_type == SPO2_TYPE:
            value_num = _spo2_percent(value_num)
        # Unit is stored as-is even if it mismatches the expected unit —
        # preserving data fidelity over silent coercion.

    return RawHealthRecord(
        record_type=hk_type,
        source_name=elem.get("sourceName", ""),
        source_version=elem.get("sourceVersion") or None,
        device_info=elem.get("device") or None,
        start_time=start_time,
        end_time=end_time,
        value_text=value_text,
        value_num=value_num,
        unit=elem.get("unit") or None,
        utc_offset_seconds=utc_offset,
        night_date=session_night(start_time),
        ingest_channel="export_xml",
    )
