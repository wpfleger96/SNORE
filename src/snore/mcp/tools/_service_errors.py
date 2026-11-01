"""Shared mapping of BreathService exceptions to MCP ValidationError.

OperationalError handling:
- "no such table": maps to a user-visible table_missing message.
- Any other OperationalError: maps to a generic sanitized message so that raw
  SQLite detail (DB file paths, column names, SQL fragments) never surfaces.
"""

from __future__ import annotations

from typing import NoReturn

from sqlalchemy.exc import OperationalError

from snore.mcp.errors import ValidationError
from snore.services.breath_service import (
    DeviceAmbiguityError,
    DeviceNotOwnedError,
    MultiSessionAmbiguityError,
    NoSessionsInRangeError,
)

MAPPED_SERVICE_ERRORS = (
    DeviceNotOwnedError,
    DeviceAmbiguityError,
    MultiSessionAmbiguityError,
    NoSessionsInRangeError,
    OperationalError,
)


def raise_mapped_service_error(exc: Exception) -> NoReturn:
    """Convert a known BreathService exception to ValidationError.

    Message formats are frozen (§3 of the Stage-2 interface contract).
    NEVER include the profile_id in user-facing messages — it is server-internal
    and leaking it aids enumeration.

    Raises:
        ValidationError: for DeviceNotOwnedError, DeviceAmbiguityError,
            MultiSessionAmbiguityError, OperationalError("no such table"), or
            any other OperationalError (sanitized to a generic message).
        The original exception: for any other unexpected type.
    """
    if isinstance(exc, NoSessionsInRangeError):
        if exc.date_start == exc.date_end:
            raise ValidationError(
                f"No therapy data found for date {exc.date_start}. "
                "Use get_data_overview to check which dates have imported data."
            ) from exc
        raise ValidationError(
            f"No therapy data found in range {exc.date_start} to {exc.date_end}. "
            "Use get_data_overview to check which dates have imported data."
        ) from exc

    if isinstance(exc, DeviceNotOwnedError):
        raise ValidationError(
            f"device_id={exc.device_id} is not available in this session"
        ) from exc

    if isinstance(exc, DeviceAmbiguityError):
        device_list = ", ".join(
            f"device_id={d} (serial={exc.device_serials.get(d, '')})"
            for d in exc.owned_device_ids
        )
        raise ValidationError(
            f"Multiple devices have sessions on {exc.therapy_date}: "
            f"pass device_id to disambiguate. Devices: {device_list}"
        ) from exc

    if isinstance(exc, MultiSessionAmbiguityError):
        session_list = "; ".join(
            f"session_id={s.session_id} start={s.start_wall_clock.isoformat()} "
            f"duration_s={round(s.duration_seconds)}"
            for s in exc.sessions
        )
        raise ValidationError(
            f"Multiple sessions on {exc.therapy_date} for device_id={exc.device_id}: "
            f"pass session_id to disambiguate. Sessions: {session_list}"
        ) from exc

    if isinstance(exc, OperationalError):
        if "no such table" in str(exc):
            raise ValidationError(
                "A required table is missing from this database "
                "(reason: table_missing). Run 'snore db init' to create it."
            ) from exc
        # Other OperationalErrors may include DB paths, column names, or SQL
        # fragments — sanitize to a generic message rather than leaking internals.
        raise ValidationError(
            "A database error occurred. "
            "Retry; if the problem persists, check the server logs."
        ) from exc

    # Unknown type — propagate unchanged.
    raise exc
