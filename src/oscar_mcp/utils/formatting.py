"""Formatting utilities for OSCAR-MCP."""

from datetime import date
from typing import Optional


def format_duration(hours: float) -> str:
    """
    Format duration in hours to human-readable string.

    Args:
        hours: Duration in hours

    Returns:
        Formatted string (e.g., "7h 40m")
    """
    if hours is None:
        return "N/A"

    whole_hours = int(hours)
    minutes = int((hours - whole_hours) * 60)

    if whole_hours > 0 and minutes > 0:
        return f"{whole_hours}h {minutes}m"
    elif whole_hours > 0:
        return f"{whole_hours}h"
    else:
        return f"{minutes}m"


def format_ahi(ahi: Optional[float]) -> str:
    """
    Format AHI value with severity assessment.

    Args:
        ahi: AHI value (events per hour)

    Returns:
        Formatted string with severity
    """
    if ahi is None:
        return "N/A"

    severity = get_ahi_severity(ahi)
    return f"{ahi:.1f} events/hr ({severity})"


def get_ahi_severity(ahi: float) -> str:
    """
    Get AHI severity classification.

    Args:
        ahi: AHI value

    Returns:
        Severity classification
    """
    if ahi < 5:
        return "normal"
    elif ahi < 15:
        return "mild"
    elif ahi < 30:
        return "moderate"
    else:
        return "severe"


def format_pressure(pressure: Optional[float]) -> str:
    """
    Format pressure value with units.

    Args:
        pressure: Pressure in cmH₂O

    Returns:
        Formatted string
    """
    if pressure is None:
        return "N/A"
    return f"{pressure:.1f} cmH₂O"


def format_leak(leak: Optional[float]) -> str:
    """
    Format leak rate with assessment.

    Args:
        leak: Leak rate in L/min

    Returns:
        Formatted string with assessment
    """
    if leak is None:
        return "N/A"

    assessment = "good" if leak < 24 else "high"
    return f"{leak:.1f} L/min ({assessment})"


def format_date_range(start: date, end: date) -> str:
    """
    Format date range.

    Args:
        start: Start date
        end: End date

    Returns:
        Formatted date range string
    """
    if start == end:
        return start.strftime("%B %d, %Y")
    elif start.year == end.year:
        if start.month == end.month:
            return f"{start.strftime('%B %d')} - {end.strftime('%d, %Y')}"
        else:
            return f"{start.strftime('%B %d')} - {end.strftime('%B %d, %Y')}"
    else:
        return f"{start.strftime('%B %d, %Y')} - {end.strftime('%B %d, %Y')}"
