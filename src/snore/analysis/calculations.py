"""Statistical calculations for OSCAR therapy data."""

from __future__ import annotations

from datetime import date, timedelta
from statistics import median
from typing import TYPE_CHECKING, Literal

from snore.database import models
from snore.utils.stats import weighted_mean

if TYPE_CHECKING:
    from snore.services.schemas import PeriodStatistics

PeriodType = Literal["day", "week", "month", "6month", "year"]


def usage_weighted_avg(days: list[models.Day], field: str) -> float | None:
    """Usage-weighted average of a Day field, weighted by ``total_therapy_hours``.

    Nights with more mask-on time count proportionally more, and days without
    the value or without positive therapy hours are excluded.  Returns None
    when no day qualifies.
    """
    return weighted_mean(
        (getattr(day, field), day.total_therapy_hours)
        for day in days
        if getattr(day, field) is not None
        and day.total_therapy_hours is not None
        and day.total_therapy_hours > 0
    )


def calculate_average_ahi(days: list[models.Day]) -> float | None:
    """
    Calculate usage-weighted average AHI across multiple days.

    Each day's AHI is weighted by ``Day.total_therapy_hours`` so nights with
    more mask-on time count proportionally more.  Days without an AHI, or with
    no positive therapy hours, are excluded.

    Args:
        days: List of Day records

    Returns:
        Usage-weighted average AHI or None if no qualifying data
    """
    return usage_weighted_avg(days, "ahi")


def calculate_average_hours_per_day(days: list[models.Day]) -> float:
    """
    Calculate average therapy hours per day.

    Only includes days with actual usage.

    Args:
        days: List of Day records

    Returns:
        Average hours per day
    """
    days_with_usage = [
        day for day in days if day.total_therapy_hours and day.total_therapy_hours > 0
    ]

    if not days_with_usage:
        return 0.0

    total_hours = sum(day.total_therapy_hours for day in days_with_usage)
    return total_hours / len(days_with_usage)


def assess_therapy_effectiveness(avg_ahi: float | None) -> str:
    """
    Assess therapy effectiveness based on AHI.

    Args:
        avg_ahi: Average AHI

    Returns:
        Assessment string: excellent, good, fair, poor
    """
    if avg_ahi is None:
        return "unknown"

    if avg_ahi < 5:
        return "excellent"
    elif avg_ahi < 10:
        return "good"
    elif avg_ahi < 15:
        return "fair"
    else:
        return "poor"


def calculate_median_ahi(days: list[models.Day]) -> float | None:
    """
    Calculate median AHI across multiple days.

    The median is intentionally unweighted: it reports the middle night, giving
    each day equal footing regardless of usage hours (unlike the usage-weighted
    ``calculate_average_ahi``).

    Args:
        days: List of Day records

    Returns:
        Median AHI or None if no data
    """
    ahi_values = [day.ahi for day in days if day.ahi is not None]
    if not ahi_values:
        return None

    return median(ahi_values)


def _get_period_boundaries(
    period_type: PeriodType,
    start_date: date,
    end_date: date,
) -> list[tuple[date, date]]:
    """
    Generate period boundaries for bucketing days.

    Args:
        period_type: One of 'day', 'week', 'month', '6month', 'year'
        start_date: Start of date range
        end_date: End of date range

    Returns:
        List of (period_start, period_end) tuples
    """
    periods = []
    current = start_date

    if period_type == "day":
        while current <= end_date:
            periods.append((current, current))
            current += timedelta(days=1)

    elif period_type == "week":
        current = current - timedelta(days=current.weekday())

        while current <= end_date:
            period_end = current + timedelta(days=6)
            periods.append((current, period_end))
            current = period_end + timedelta(days=1)

    elif period_type == "month":
        current = current.replace(day=1)

        while current <= end_date:
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)

            period_end = next_month - timedelta(days=1)
            periods.append((current, period_end))
            current = next_month

    elif period_type == "6month":
        if current.month <= 6:
            current = current.replace(month=1, day=1)
        else:
            current = current.replace(month=7, day=1)

        while current <= end_date:
            if current.month == 1:
                period_end = current.replace(month=6, day=30)
            else:
                period_end = current.replace(month=12, day=31)

            periods.append((current, period_end))

            if current.month == 1:
                current = current.replace(month=7)
            else:
                current = current.replace(year=current.year + 1, month=1)

    elif period_type == "year":
        current = current.replace(month=1, day=1)

        while current <= end_date:
            period_end = current.replace(month=12, day=31)
            periods.append((current, period_end))
            current = current.replace(year=current.year + 1)
    else:
        raise ValueError(f"Invalid period_type: {period_type}")

    return periods


def calculate_period_statistics(
    day_records: list[models.Day],
    period_type: PeriodType,
) -> list[PeriodStatistics]:
    """
    Calculate statistics grouped by time periods.

    Args:
        day_records: List of Day records
        period_type: One of 'day', 'week', 'month', '6month', 'year'

    Returns:
        List of PeriodStatistics for each period
    """
    if not day_records:
        return []

    from snore.services.schemas import PeriodStatistics  # noqa: PLC0415

    dates = [day.date for day in day_records]
    start_date = min(dates)
    end_date = max(dates)

    periods = _get_period_boundaries(period_type, start_date, end_date)
    results = []

    # For "day" granularity, pre-bucket by date to avoid O(N×P) scan.
    # With multi-year histories, P can exceed 1000 period boundaries.
    _by_date: dict[date, list[models.Day]] = {}
    if period_type == "day":
        for day in day_records:
            _by_date.setdefault(day.date, []).append(day)

    for period_start, period_end in periods:
        if period_type == "day":
            days_in_period = _by_date.get(period_start, [])
        else:
            days_in_period = [
                day for day in day_records if period_start <= day.date <= period_end
            ]

        if not days_in_period:
            continue

        days_used = len(days_in_period)
        days_in_period_count = (period_end - period_start).days + 1

        avg_hours_per_day = calculate_average_hours_per_day(days_in_period)
        avg_ahi = calculate_average_ahi(days_in_period)
        median_ahi = calculate_median_ahi(days_in_period)

        avg_pressure = usage_weighted_avg(days_in_period, "pressure_median")
        avg_leak = usage_weighted_avg(days_in_period, "leak_median")
        avg_spo2 = usage_weighted_avg(days_in_period, "spo2_mean")

        spo2_mins = [day.spo2_min for day in days_in_period if day.spo2_min is not None]
        min_spo2 = min(spo2_mins) if spo2_mins else None

        avg_oai = usage_weighted_avg(days_in_period, "oai")
        avg_cai = usage_weighted_avg(days_in_period, "cai")
        avg_hi = usage_weighted_avg(days_in_period, "hi")

        rera_rates = [
            day.reras / day.total_therapy_hours
            for day in days_in_period
            if day.reras is not None
            and day.total_therapy_hours is not None
            and day.total_therapy_hours > 0
        ]
        avg_rera = sum(rera_rates) / len(rera_rates) if rera_rates else None

        results.append(
            PeriodStatistics(
                period_type=period_type,
                period_start=period_start,
                period_end=period_end,
                days_used=days_used,
                days_in_period=days_in_period_count,
                avg_hours_per_day=avg_hours_per_day if avg_hours_per_day > 0 else None,
                avg_ahi=avg_ahi,
                median_ahi=median_ahi,
                avg_pressure=avg_pressure,
                avg_leak=avg_leak,
                avg_spo2=avg_spo2,
                min_spo2=min_spo2,
                avg_oai=avg_oai,
                avg_cai=avg_cai,
                avg_hi=avg_hi,
                avg_rera=avg_rera,
            )
        )

    return results


_TREND_KEYS = [
    "ahi",
    "usage",
    "spo2",
    "leak",
    "pressure",
    "oai",
    "cai",
    "hi",
    "rera",
    "epap",
    "rr",
    "pulse",
    "mv",
]


def calculate_trends_extended(
    period_stats: list[PeriodStatistics],
    session_extras: dict[date, dict[str, float | None]],
) -> dict[str, list[tuple[date, float | None]]]:
    """
    Extract extended trend data from period statistics and session-level aggregates.

    Args:
        period_stats: List of PeriodStatistics
        session_extras: Per-period session-weighted means keyed by period_start;
            expected sub-keys: epap, rr, pulse, mv (absent → None)

    Returns:
        Dictionary with 13 metric keys, each mapping to a list of (date, value) tuples:
        ahi, usage, spo2, leak, pressure, oai, cai, hi, rera, epap, rr, pulse, mv
    """
    trends: dict[str, list[tuple[date, float | None]]] = {k: [] for k in _TREND_KEYS}

    for stat in period_stats:
        extras = session_extras.get(stat.period_start, {})
        trends["ahi"].append((stat.period_start, stat.avg_ahi))
        trends["usage"].append((stat.period_start, stat.avg_hours_per_day))
        trends["spo2"].append((stat.period_start, stat.avg_spo2))
        trends["leak"].append((stat.period_start, stat.avg_leak))
        trends["pressure"].append((stat.period_start, stat.avg_pressure))
        trends["oai"].append((stat.period_start, stat.avg_oai))
        trends["cai"].append((stat.period_start, stat.avg_cai))
        trends["hi"].append((stat.period_start, stat.avg_hi))
        trends["rera"].append((stat.period_start, stat.avg_rera))
        trends["epap"].append((stat.period_start, extras.get("epap")))
        trends["rr"].append((stat.period_start, extras.get("rr")))
        trends["pulse"].append((stat.period_start, extras.get("pulse")))
        trends["mv"].append((stat.period_start, extras.get("mv")))

    return trends


def calculate_trends(
    period_stats: list[PeriodStatistics],
) -> dict[str, list[tuple[date, float | None]]]:
    """
    Extract trend data from period statistics.

    Delegates to calculate_trends_extended with no session extras.
    Returns all 13 metric keys; callers that only need the original 4
    (ahi, usage, spo2, leak) can index by key as before.

    Args:
        period_stats: List of PeriodStatistics

    Returns:
        Dictionary mapping metric names to lists of (date, value) tuples.
    """
    return calculate_trends_extended(period_stats, {})


def calculate_records(
    days: list[models.Day],
    top_n: int = 5,
) -> dict[str, dict[str, list[tuple[date, float]]]]:
    """
    Calculate top best/worst days for key metrics.

    Args:
        days: List of Day records
        top_n: Number of records to return (default: 5)

    Returns:
        Dictionary mapping metric names to best/worst records.
        Structure: {"ahi": {"best": [(date, value), ...], "worst": [...]}, ...}
    """
    qualified_days = [day for day in days if (day.total_therapy_hours or 0) >= 1.0]

    if not qualified_days:
        return {}

    records: dict[str, dict[str, list[tuple[date, float]]]] = {}

    ahi_days = [(day.date, day.ahi) for day in qualified_days if day.ahi is not None]
    if ahi_days:
        records["ahi"] = {
            "best": sorted(ahi_days, key=lambda x: x[1])[:top_n],
            "worst": sorted(ahi_days, key=lambda x: x[1], reverse=True)[:top_n],
        }

    leak_days = [
        (day.date, day.leak_median)
        for day in qualified_days
        if day.leak_median is not None
    ]
    if leak_days:
        records["leak"] = {
            "best": sorted(leak_days, key=lambda x: x[1])[:top_n],
            "worst": sorted(leak_days, key=lambda x: x[1], reverse=True)[:top_n],
        }

    therapy_days = [
        (day.date, day.total_therapy_hours)
        for day in qualified_days
        if day.total_therapy_hours is not None
    ]
    if therapy_days:
        records["therapy_hours"] = {
            "best": sorted(therapy_days, key=lambda x: x[1], reverse=True)[:top_n],
            "worst": sorted(therapy_days, key=lambda x: x[1])[:top_n],
        }

    spo2_days = [
        (day.date, day.spo2_min) for day in qualified_days if day.spo2_min is not None
    ]
    if spo2_days:
        records["spo2_min"] = {
            "best": sorted(spo2_days, key=lambda x: x[1], reverse=True)[:top_n],
            "worst": sorted(spo2_days, key=lambda x: x[1])[:top_n],
        }

    return records


def calculate_ahi_trend_direction(ahi_values: list[float]) -> str | None:
    """
    Determine AHI trend direction from a time-ordered list of period AHI values.

    Args:
        ahi_values: Chronological list of AHI values

    Returns:
        "improving", "worsening", "stable", or None if insufficient data
    """
    if len(ahi_values) < 2:
        return None
    latest = ahi_values[-1]
    prior = ahi_values[:-1]
    prior_avg = sum(prior) / len(prior)
    if prior_avg == 0:
        return None
    if latest < prior_avg * 0.9:
        return "improving"
    if latest > prior_avg * 1.1:
        return "worsening"
    return "stable"
