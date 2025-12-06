"""Statistical calculations for OSCAR therapy data."""

from datetime import date

from snore.constants import COMPLIANCE_MIN_HOURS
from snore.database import models


def calculate_ahi(
    obstructive: int, hypopnea: int, central: int, duration_hours: float
) -> float:
    """
    Calculate Apnea-Hypopnea Index (AHI).

    AHI = (Obstructive Apneas + Hypopneas + Central Apneas) / Hours

    Args:
        obstructive: Count of obstructive apneas
        hypopnea: Count of hypopneas
        central: Count of central/clear airway apneas
        duration_hours: Session duration in hours

    Returns:
        AHI value (events per hour)
    """
    if duration_hours <= 0:
        return 0.0

    total_events = obstructive + hypopnea + central
    return total_events / duration_hours


def calculate_rdi(
    obstructive: int, hypopnea: int, central: int, rera: int, duration_hours: float
) -> float:
    """
    Calculate Respiratory Disturbance Index (RDI).

    RDI = (Obstructive Apneas + Hypopneas + Central Apneas + RERA) / Hours

    Args:
        obstructive: Count of obstructive apneas
        hypopnea: Count of hypopneas
        central: Count of central/clear airway apneas
        rera: Count of respiratory effort related arousals
        duration_hours: Session duration in hours

    Returns:
        RDI value (events per hour)
    """
    if duration_hours <= 0:
        return 0.0

    total_events = obstructive + hypopnea + central + rera
    return total_events / duration_hours


def is_compliant(hours: float | None) -> bool:
    """
    Check if usage meets compliance requirements.

    Args:
        hours: Hours of therapy usage

    Returns:
        True if compliant (>= 4 hours)
    """
    if hours is None:
        return False
    return hours >= COMPLIANCE_MIN_HOURS


def calculate_compliance_rate(days: list[models.Day]) -> tuple[float, int, int]:
    """
    Calculate compliance rate for a set of days.

    Args:
        days: List of Day records

    Returns:
        Tuple of (compliance_percentage, compliant_days, total_days)
    """
    if not days:
        return 0.0, 0, 0

    total_days = len(days)
    compliant_days = sum(1 for day in days if is_compliant(day.total_therapy_hours))

    compliance_percentage = (compliant_days / total_days) * 100
    return compliance_percentage, compliant_days, total_days


def calculate_average_ahi(days: list[models.Day]) -> float | None:
    """
    Calculate average AHI across multiple days.

    Args:
        days: List of Day records

    Returns:
        Average AHI or None if no data
    """
    ahi_values = [day.ahi for day in days if day.ahi is not None]
    if not ahi_values:
        return None

    return sum(ahi_values) / len(ahi_values)


def calculate_total_hours(days: list[models.Day]) -> float:
    """
    Calculate total therapy hours across multiple days.

    Args:
        days: List of Day records

    Returns:
        Total hours
    """
    return sum(day.total_therapy_hours or 0 for day in days)


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


def get_date_range(days: list[models.Day]) -> tuple[date, date] | None:
    """
    Get date range from list of days.

    Args:
        days: List of Day records

    Returns:
        Tuple of (start_date, end_date) or None if no days
    """
    if not days:
        return None

    dates = [day.date for day in days]
    return min(dates), max(dates)
