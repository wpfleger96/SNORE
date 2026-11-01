"""Custom SQLAlchemy column types for SNORE."""

import json

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import DateTime, Text, TypeDecorator


class ValidatedJSON(TypeDecorator[dict[str, Any]]):
    """
    A JSON column type that validates JSON before storing.

    This type ensures that:
    1. Values can be serialized to JSON
    2. Stored values are valid JSON strings
    3. Retrieved values are automatically deserialized to Python objects

    Example:
        class MyModel(Base):
            data = Column(ValidatedJSON, nullable=False)

        obj.data = {"key": "value"}
        print(obj.data)
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        """
        Convert Python object to JSON string before storing.

        Args:
            value: Python object to serialize
            dialect: SQLAlchemy dialect

        Returns:
            JSON string or None

        Raises:
            ValueError: If value cannot be serialized to JSON
        """
        if value is None:
            return None

        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot serialize value to JSON: {e}. Value type: {type(value).__name__}"
            ) from e

    def process_result_value(self, value: str | None, dialect: Any) -> Any:
        """
        Convert JSON string to Python object after retrieval.

        Args:
            value: JSON string from database
            dialect: SQLAlchemy dialect

        Returns:
            Deserialized Python object or None

        Raises:
            ValueError: If stored value is not valid JSON
        """
        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Stored value is not valid JSON: {e}. Value: {value[:100]}..."
            ) from e


class ValidatedJSONWithDefault(ValidatedJSON):
    """
    A JSON column type that provides a default empty dict if value is None.

    Useful for optional JSON columns where you want to avoid None checks.

    Example:
        class MyModel(Base):
            metadata = Column(ValidatedJSONWithDefault)

        obj.metadata = None
        print(obj.metadata)
    """

    cache_ok = True

    def process_result_value(self, value: str | None, dialect: Any) -> Any:
        """
        Convert JSON string to Python object, returning {} if None.

        Args:
            value: JSON string from database
            dialect: SQLAlchemy dialect

        Returns:
            Deserialized Python object or empty dict
        """
        result = super().process_result_value(value, dialect)
        return result if result is not None else {}


class UTCDateTime(TypeDecorator[datetime]):
    """A DateTime column type that preserves UTC timezone info across all dialects.

    SQLite discards tzinfo on storage/retrieval; ``DateTime(timezone=True)``
    is a false promise on that dialect.  This type:

    - **Binds**: normalises any tz-aware datetime to UTC before storage; rejects
      naive datetimes (raise ``ValueError``).
    - **Loads**: restores ``tzinfo=UTC`` on every result regardless of dialect,
      interpreting the stored value as UTC.

    **Dialect behaviour:**

    - SQLite: uses ``DateTime`` (no native TZ); value stored as naive ISO string;
      UTC is re-attached on load.
    - PostgreSQL (and other TIMESTAMP WITH TIME ZONE dialects): uses
      ``DateTime(timezone=True)``; the driver receives and returns offset-aware
      datetimes; we still normalise to UTC on load for consistency.

    Use this for absolute audit instants (created_at, updated_at, last_import …).
    Do NOT use it for device/session wall-clock columns (Session.start_time,
    Event.start_time, etc.) whose source timezone is unknown.

    cache_ok = True: the type carries no instance-specific state.
    """

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        """Return dialect-specific implementation.

        PostgreSQL supports TIMESTAMP WITH TIME ZONE; SQLite does not.
        """
        if dialect.name == "sqlite":
            return dialect.type_descriptor(DateTime(timezone=False))
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: Any, dialect: Any) -> datetime | None:
        """Normalise a tz-aware datetime to UTC before storage.

        - **SQLite**: strips tzinfo so the driver stores a plain ISO string.
        - **Other dialects** (PostgreSQL etc.): passes a UTC-aware datetime so
          the driver / server can use the timezone information (TIMESTAMPTZ).

        Args:
            value: datetime to store; must be tz-aware.
            dialect: SQLAlchemy dialect.

        Returns:
            Naive UTC datetime for SQLite; tz-aware UTC datetime for other dialects.

        Raises:
            ValueError: If value is a naive datetime.
        """
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise ValueError(f"UTCDateTime expects a datetime, got {type(value)!r}")
        if value.tzinfo is None:
            raise ValueError(
                "UTCDateTime requires tz-aware datetimes; "
                "use datetime.now(UTC) or datetime(..., tzinfo=UTC)"
            )
        # Normalise non-UTC offsets to UTC.
        utc_value = value.astimezone(UTC)
        # SQLite stores bare ISO strings; strip tzinfo so the driver doesn't
        # prepend "+00:00".  PostgreSQL (TIMESTAMPTZ) needs the tzinfo present
        # so the driver passes an aware value to the server.
        if dialect.name == "sqlite":
            return utc_value.replace(tzinfo=None)
        return utc_value

    def process_result_value(self, value: Any, dialect: Any) -> datetime | None:
        """Restore UTC tzinfo on every retrieved datetime.

        Args:
            value: Raw value from the database driver.
            dialect: SQLAlchemy dialect (unused).

        Returns:
            tz-aware datetime in UTC, or None.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            # May already carry tzinfo from PostgreSQL driver; normalise to UTC.
            if value.tzinfo is not None:
                return value.astimezone(UTC)
            return value.replace(tzinfo=UTC)
        # SQLite may return an ISO string in some configurations.
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is not None:
                return parsed.astimezone(UTC)
            return parsed.replace(tzinfo=UTC)
        raise ValueError(f"UTCDateTime: unexpected stored value type {type(value)!r}")


# Convenience alias: zero UTC offset as a timedelta, for test assertions.
UTC_ZERO = timedelta(0)

# Re-export so callers can do: from snore.database.types import UTC
_UTC = UTC
__all__ = [
    "ValidatedJSON",
    "ValidatedJSONWithDefault",
    "UTCDateTime",
    "UTC_ZERO",
    "timezone",
]
