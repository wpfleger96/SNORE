"""Validation utilities for SNORE."""

from datetime import date, datetime

from snore.database import models
from snore.database.session import session_scope


def validate_profile_exists(profile_name: str) -> models.Profile:
    """
    Validate that a profile exists in the database and return it.

    Args:
        profile_name: Profile username

    Returns:
        Profile object if found

    Raises:
        ValueError: If profile does not exist
    """
    with session_scope() as session:
        profile = session.query(models.Profile).filter_by(username=profile_name).first()
        if not profile:
            raise ValueError(
                f"Profile '{profile_name}' not found. "
                f"Use 'oscar-import status' to list available profiles."
            )
        # Detach from session so it can be returned
        session.expunge(profile)
    return profile


def validate_date_format(date_str: str) -> date:
    """
    Validate and parse date string in YYYY-MM-DD format.

    Args:
        date_str: Date string

    Returns:
        Parsed date object

    Raises:
        ValueError: If date format is invalid
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected format: YYYY-MM-DD (e.g., 2024-01-15)"
        ) from None


def validate_date_range(start_date: date, end_date: date) -> bool:
    """
    Validate that date range is logical.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        True if valid

    Raises:
        ValueError: If date range is invalid
    """
    if start_date > end_date:
        raise ValueError(
            f"Invalid date range: start_date ({start_date}) "
            f"must be before or equal to end_date ({end_date})"
        )
    return True


def validate_period_type(period_type: str) -> bool:
    """
    Validate period type.

    Args:
        period_type: Period type string

    Returns:
        True if valid

    Raises:
        ValueError: If period type is invalid
    """
    valid_types = ["daily", "weekly", "monthly", "yearly"]
    if period_type not in valid_types:
        raise ValueError(
            f"Invalid period type: '{period_type}'. Valid types are: {', '.join(valid_types)}"
        )
    return True
