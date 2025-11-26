"""Custom SQLAlchemy column types for OSCAR-MCP."""

import json

from typing import Any

from sqlalchemy import Text, TypeDecorator


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

        # Usage
        obj.data = {"key": "value"}  # Automatically validated and serialized
        print(obj.data)  # Automatically deserialized to dict
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
            # Validate that value can be serialized to JSON
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

        # Usage
        obj.metadata = None
        print(obj.metadata)  # Returns {} instead of None
    """

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
