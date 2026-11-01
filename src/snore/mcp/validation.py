"""Input validation helpers for MCP tools."""

from __future__ import annotations

import math
import re

from datetime import date

from snore.mcp.errors import ValidationError

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_date(value: str, param_name: str) -> date:
    """Parse an ISO 8601 date string (YYYY-MM-DD).

    Args:
        value: The date string to parse.
        param_name: Parameter name used in error messages.

    Returns:
        Parsed date object.

    Raises:
        ValidationError: If the string is not a valid YYYY-MM-DD date.
    """
    if not _ISO_DATE_RE.match(value):
        raise ValidationError(
            f"Invalid date value {value!r} for parameter '{param_name}'. "
            "Expected YYYY-MM-DD format, e.g. '2025-08-01'."
        )
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(
            f"Invalid date value {value!r} for parameter '{param_name}': {exc}"
        ) from exc


def parse_date_range(
    start: str,
    end: str,
    *,
    start_param: str = "start",
    end_param: str = "end",
) -> tuple[date, date]:
    """Parse and validate a start/end date range.

    Args:
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).
        start_param: Name of the start parameter for error messages.
        end_param: Name of the end parameter for error messages.

    Returns:
        Tuple of (start_date, end_date).

    Raises:
        ValidationError: If either date is invalid or start > end.
    """
    start_date = parse_date(start, start_param)
    end_date = parse_date(end, end_param)
    if start_date > end_date:
        raise ValidationError(
            f"'{start_param}' ({start}) must not be after '{end_param}' ({end})."
        )
    return start_date, end_date


_PAGE_SIZE_CAP = 90
# Keeps (page - 1) * page_size inside SQLite's signed 64-bit OFFSET.
_MAX_PAGE = 1_000_000


def _require_finite(value: float, param_name: str) -> None:
    """Raise ValidationError for NaN or infinity, which every comparison mishandles."""
    if not math.isfinite(value):
        raise ValidationError(f"{param_name} must be a finite number")


def validate_page_args(page: int, page_size: int) -> int:
    """Validate pagination args; return the validated page_size.

    Raises:
        ValidationError: If ``page`` is outside [1, _MAX_PAGE] or ``page_size``
            is outside [1, _PAGE_SIZE_CAP].
    """
    if page < 1:
        raise ValidationError("page must be >= 1")
    if page > _MAX_PAGE:
        raise ValidationError(f"page must be <= {_MAX_PAGE}")
    if page_size < 1 or page_size > _PAGE_SIZE_CAP:
        raise ValidationError(f"page_size must be between 1 and {_PAGE_SIZE_CAP}")
    return page_size


def validate_compliance_threshold(hours: float) -> None:
    """Raise ValidationError if compliance threshold is negative or non-finite."""
    _require_finite(hours, "compliance_threshold_hours")
    if hours < 0:
        raise ValidationError("compliance_threshold_hours must be >= 0")


def validate_min_duration(min_duration: float | None) -> None:
    """Raise ValidationError if min_duration is negative or non-finite."""
    if min_duration is None:
        return
    _require_finite(min_duration, "min_duration")
    if min_duration < 0:
        raise ValidationError("min_duration must be >= 0")


def validate_max_events(max_events: int) -> None:
    """Raise ValidationError if max_events is less than 1."""
    if max_events < 1:
        raise ValidationError("max_events must be >= 1")


def validate_window_count(n: int) -> None:
    """Raise ValidationError unless 1 <= n <= 50."""
    if n < 1 or n > 50:
        raise ValidationError(f"n must be between 1 and 50 (got {n})")


def validate_epoch_count(count: int) -> None:
    """Raise ValidationError unless 1 <= count <= 6."""
    if count < 1 or count > 6:
        raise ValidationError(
            f"epochs must contain between 1 and 6 entries (got {count})"
        )
