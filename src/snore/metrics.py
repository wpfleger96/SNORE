"""Drift-guard registry for the session/day metric set.

The 78-field metric vocabulary is declared in several places (unified parser
model, ORM ``Statistics``, services schema, export lists, day aggregation).
This leaf module (stdlib-only) does not collapse those declarations into one:
adding a session metric still requires parallel edits to the ORM ``Statistics``
model and the parser dataclass.  What it buys is a guard — schema-alignment
tests assert the other declarations stay in sync with this registry, so drift
surfaces as a test failure instead of silent data loss.  Day aggregation
(``DAY_METRIC_STAT_COLUMNS``) and export keys (``EXPORT_STAT_KEYS``) are the
genuine single-source wins: they are declared only here.
"""

from dataclasses import dataclass
from enum import StrEnum


class DayAgg(StrEnum):
    """How a session statistic rolls up into its Day-table column."""

    SUM = "sum"
    MIN = "min"
    MAX = "max"
    USAGE_WEIGHTED_MEAN = "usage_weighted_mean"


@dataclass(frozen=True)
class MetricSpec:
    """One metric column: its name and day-level aggregation.

    ``day_agg`` is None for metrics that have no mirrored Day-table column.
    """

    name: str
    day_agg: DayAgg | None


def _m(name: str) -> MetricSpec:
    """Session-level metric with no Day-table column."""
    return MetricSpec(name=name, day_agg=None)


# All models.Statistics metric columns (every column except session_id),
# ordered as declared in the ORM.
SESSION_METRICS: tuple[MetricSpec, ...] = (
    # Event counts
    _m("obstructive_apneas"),
    _m("central_apneas"),
    _m("mixed_apneas"),
    _m("hypopneas"),
    _m("reras"),
    _m("flow_limitations"),
    # Indices
    _m("ahi"),
    _m("oai"),
    _m("cai"),
    _m("hi"),
    _m("rei"),
    # Device-reported (STR) indices, kept alongside the computed ones above
    _m("ahi_device"),
    _m("oai_device"),
    _m("cai_device"),
    _m("hi_device"),
    # Pressure
    _m("pressure_min"),
    _m("pressure_max"),
    _m("pressure_median"),
    _m("pressure_mean"),
    _m("pressure_95th"),
    # EPAP
    _m("epap_min"),
    _m("epap_max"),
    _m("epap_median"),
    _m("epap_mean"),
    _m("epap_95th"),
    # IPAP
    _m("ipap_median"),
    _m("ipap_95th"),
    _m("ipap_max"),
    # Leak
    _m("leak_min"),
    _m("leak_max"),
    _m("leak_median"),
    _m("leak_mean"),
    _m("leak_95th"),
    _m("leak_percentile_70"),
    # Respiratory rate
    _m("respiratory_rate_min"),
    _m("respiratory_rate_max"),
    _m("respiratory_rate_mean"),
    # Tidal volume
    _m("tidal_volume_min"),
    _m("tidal_volume_max"),
    _m("tidal_volume_mean"),
    # Minute ventilation
    _m("minute_ventilation_min"),
    _m("minute_ventilation_max"),
    _m("minute_ventilation_mean"),
    # SpO2 / pulse oximetry
    _m("spo2_min"),
    _m("spo2_max"),
    _m("spo2_mean"),
    _m("spo2_median"),
    _m("spo2_95th"),
    _m("spo2_time_below_90"),
    _m("pulse_min"),
    _m("pulse_max"),
    _m("pulse_mean"),
    # Usage
    _m("usage_hours"),
    # STR daily summary extras
    _m("uai"),
    _m("ai"),
    _m("rin"),
    _m("csr_pct"),
    _m("spont_cyc_pct"),
    _m("respiratory_rate_95th"),
    _m("tidal_volume_95th"),
    _m("minute_ventilation_95th"),
    _m("ie_ratio_median"),
    _m("ie_ratio_95th"),
    _m("ie_ratio_max"),
    _m("ti_median"),
    _m("ti_95th"),
    _m("ti_max"),
    _m("flow_5th"),
    _m("flow_95th"),
    _m("blow_press_5th"),
    _m("blow_press_95th"),
    _m("blow_flow_median"),
    _m("amb_humidity_median"),
    _m("hum_temp_median"),
    _m("htube_temp_median"),
    _m("htube_pow_median"),
    _m("hum_pow_median"),
    _m("mask_events"),
)


def _day(name: str, day_agg: DayAgg) -> MetricSpec:
    return MetricSpec(name=name, day_agg=day_agg)


# Day-table stat columns mirroring Statistics fields, with their day-level
# aggregation.  Counts, indices (ahi/oai/cai/hi), hours, and bookkeeping
# columns are aggregated separately and intentionally excluded.
DAY_METRIC_STAT_COLUMNS: tuple[MetricSpec, ...] = (
    _day("pressure_min", DayAgg.MIN),
    _day("pressure_max", DayAgg.MAX),
    _day("pressure_median", DayAgg.USAGE_WEIGHTED_MEAN),
    _day("pressure_mean", DayAgg.USAGE_WEIGHTED_MEAN),
    _day("pressure_95th", DayAgg.USAGE_WEIGHTED_MEAN),
    _day("epap_min", DayAgg.MIN),
    _day("epap_max", DayAgg.MAX),
    _day("epap_median", DayAgg.USAGE_WEIGHTED_MEAN),
    _day("epap_mean", DayAgg.USAGE_WEIGHTED_MEAN),
    _day("epap_95th", DayAgg.USAGE_WEIGHTED_MEAN),
    _day("leak_min", DayAgg.MIN),
    _day("leak_max", DayAgg.MAX),
    _day("leak_median", DayAgg.USAGE_WEIGHTED_MEAN),
    _day("leak_mean", DayAgg.USAGE_WEIGHTED_MEAN),
    _day("leak_95th", DayAgg.USAGE_WEIGHTED_MEAN),
    _day("spo2_min", DayAgg.MIN),
    _day("spo2_max", DayAgg.MAX),
    _day("spo2_mean", DayAgg.USAGE_WEIGHTED_MEAN),
)


# Exact CSV/JSON export stat-key subset, in exact output order.  Shared by the
# CSV header, the CSV row builder, the export SELECT, and the statistics-dict
# assembly in export_service.
EXPORT_STAT_KEYS: tuple[str, ...] = (
    "ahi",
    "oai",
    "cai",
    "hi",
    "obstructive_apneas",
    "central_apneas",
    "hypopneas",
    "reras",
    "pressure_mean",
    "pressure_95th",
    "epap_mean",
    "leak_mean",
    "leak_95th",
    "spo2_mean",
    "usage_hours",
    # Appended last so pre-existing sessions.csv columns keep their positions.
    "ahi_device",
    "oai_device",
    "cai_device",
    "hi_device",
)
