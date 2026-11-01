"""Unit tests for snore.mcp.tools._service_errors.raise_mapped_service_error.

These tests exercise the mapper directly (not through a tool roundtrip) to
verify the message format contract frozen by §3 of the Stage-2 interface.
"""

from __future__ import annotations

from datetime import date

import pytest

from sqlalchemy.exc import OperationalError

from snore.mcp.errors import ValidationError
from snore.mcp.tools._service_errors import raise_mapped_service_error
from snore.services.breath_service import NoSessionsInRangeError


class TestRaiseMappedServiceError:
    def test_single_date_produces_polished_message(self) -> None:
        """NoSessionsInRangeError with date_start==date_end yields per-date message."""
        exc = NoSessionsInRangeError(date(2024, 6, 1), date(2024, 6, 1))
        with pytest.raises(ValidationError) as exc_info:
            raise_mapped_service_error(exc)
        msg = str(exc_info.value)
        assert "No therapy data found for date 2024-06-01" in msg
        assert "get_data_overview" in msg

    def test_range_produces_polished_range_message(self) -> None:
        """NoSessionsInRangeError with date_start!=date_end yields range message."""
        exc = NoSessionsInRangeError(date(2024, 6, 1), date(2024, 6, 7))
        with pytest.raises(ValidationError) as exc_info:
            raise_mapped_service_error(exc)
        msg = str(exc_info.value)
        assert "No therapy data found in range 2024-06-01 to 2024-06-07" in msg
        assert "get_data_overview" in msg

    def test_range_message_does_not_contain_raw_service_message(self) -> None:
        """The raw NoSessionsInRangeError string must not appear in the range error."""
        exc = NoSessionsInRangeError(date(2024, 1, 10), date(2024, 1, 20))
        with pytest.raises(ValidationError) as exc_info:
            raise_mapped_service_error(exc)
        msg = str(exc_info.value)
        # The raw service message is "No sessions found in range …"
        assert "No sessions found" not in msg


class TestOperationalErrors:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("no such table: days", "table_missing"),
            ("disk I/O error at /data/.snore/snore.db", "A database error occurred"),
        ],
    )
    def test_message_suits_tools_that_need_no_analysis(
        self, message: str, expected: str
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            raise_mapped_service_error(OperationalError(message, None, None))

        msg = str(exc_info.value)
        assert expected in msg
        assert "breath" not in msg.lower()
        assert "analysis" not in msg.lower()
        assert "/data" not in msg
