"""Drift guards keeping Pydantic statistics schemas aligned with the ORM.

The importer splats ``SessionStatistics.model_dump()`` directly into the
``models.Statistics`` constructor, and the session service hydrates the
services-layer ``SessionStatistics`` from ORM rows via ``model_validate``.
Both shortcuts are only safe while every Pydantic field maps to an ORM
column of the same name; these tests make that invariant executable.
"""

import pytest

from snore.database import models
from snore.metrics import DAY_METRIC_STAT_COLUMNS, EXPORT_STAT_KEYS, SESSION_METRICS
from snore.parsers.unified import SessionStatistics as UnifiedSessionStatistics
from snore.services.schemas import SessionStatistics as ServiceSessionStatistics

pytestmark = pytest.mark.unit

# ORM columns that are intentionally not sourced from the unified statistics
# model. ``session_id`` is the primary/foreign key, set explicitly by the
# importer (``models.Statistics(session_id=..., **stats.model_dump())``) rather
# than carried in the Pydantic payload. Anything else missing from the unified
# model would be silently never imported, so it must be added here deliberately.
_IMPORTER_SUPPLIED_COLUMNS = {"session_id"}


def _statistics_column_names() -> set[str]:
    return {column.name for column in models.Statistics.__table__.columns}


def test_unified_statistics_fields_subset_of_orm_columns() -> None:
    """Every unified-parser statistics field must exist as an ORM column."""
    missing = set(UnifiedSessionStatistics.model_fields) - _statistics_column_names()
    assert not missing, (
        f"parsers.unified.SessionStatistics fields without a "
        f"models.Statistics column: {sorted(missing)}"
    )


def test_service_statistics_fields_subset_of_orm_columns() -> None:
    """Every services-layer statistics field must exist as an ORM column."""
    missing = set(ServiceSessionStatistics.model_fields) - _statistics_column_names()
    assert not missing, (
        f"services.schemas.SessionStatistics fields without a "
        f"models.Statistics column: {sorted(missing)}"
    )


def test_orm_columns_sourced_from_unified_statistics() -> None:
    """Every ORM column must be populated from the unified model on import.

    Reverse of the subset check: an ORM column added without a matching unified
    field would never be written by the importer. Such a column must either get
    a unified field or be added to ``_IMPORTER_SUPPLIED_COLUMNS`` on purpose.
    """
    unsourced = (
        _statistics_column_names()
        - set(UnifiedSessionStatistics.model_fields)
        - _IMPORTER_SUPPLIED_COLUMNS
    )
    assert not unsourced, (
        f"models.Statistics columns not sourced from "
        f"parsers.unified.SessionStatistics (never imported): {sorted(unsourced)}"
    )


# Day-table columns that are not per-metric statistics: bookkeeping, counts,
# indices, and hours are aggregated by dedicated logic in DayManager rather
# than the DAY_METRIC_STAT_COLUMNS registry loop.
_DAY_NON_METRIC_COLUMNS = {
    "id",
    "device_id",
    "date",
    "session_count",
    "total_therapy_hours",
    "obstructive_apneas",
    "central_apneas",
    "hypopneas",
    "reras",
    "ahi",
    "oai",
    "cai",
    "hi",
    # Recount + headline source: dedicated DayManager logic.
    "ahi_computed",
    "oai_computed",
    "cai_computed",
    "hi_computed",
    "index_source",
    "created_at",
    "updated_at",
}


def _day_column_names() -> set[str]:
    return {column.name for column in models.Day.__table__.columns}


def test_session_metrics_registry_matches_orm_columns() -> None:
    """SESSION_METRICS must equal the ORM metric columns exactly."""
    registry = {m.name for m in SESSION_METRICS}
    orm = _statistics_column_names() - _IMPORTER_SUPPLIED_COLUMNS
    assert registry == orm, (
        f"metrics.SESSION_METRICS out of sync with models.Statistics: "
        f"missing={sorted(orm - registry)} extra={sorted(registry - orm)}"
    )


def test_day_metric_stat_columns_match_day_and_statistics() -> None:
    """Every registry Day stat column exists on both Day and Statistics."""
    registry = {m.name for m in DAY_METRIC_STAT_COLUMNS}
    not_day = registry - _day_column_names()
    assert not not_day, f"DAY_METRIC_STAT_COLUMNS not on models.Day: {sorted(not_day)}"
    not_stats = registry - _statistics_column_names()
    assert not not_stats, (
        f"DAY_METRIC_STAT_COLUMNS not on models.Statistics: {sorted(not_stats)}"
    )


def test_day_stat_columns_all_registered() -> None:
    """Every Day metric-stat column must be in DAY_METRIC_STAT_COLUMNS.

    Reverse of the subset check: a Day stat column added without a registry
    entry would never be aggregated by DayManager. Non-metric Day columns are
    listed in ``_DAY_NON_METRIC_COLUMNS`` deliberately.
    """
    registry = {m.name for m in DAY_METRIC_STAT_COLUMNS}
    unregistered = _day_column_names() - _DAY_NON_METRIC_COLUMNS - registry
    assert not unregistered, (
        f"models.Day stat columns missing from metrics.DAY_METRIC_STAT_COLUMNS "
        f"(never aggregated): {sorted(unregistered)}"
    )
    stale = registry & _DAY_NON_METRIC_COLUMNS
    assert not stale, (
        f"DAY_METRIC_STAT_COLUMNS entries also listed as non-metric: {sorted(stale)}"
    )


def test_export_stat_keys_subset_of_session_metrics() -> None:
    """Every export stat key must be a registered session metric."""
    unknown = set(EXPORT_STAT_KEYS) - {m.name for m in SESSION_METRICS}
    assert not unknown, (
        f"metrics.EXPORT_STAT_KEYS not in SESSION_METRICS: {sorted(unknown)}"
    )
