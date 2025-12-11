"""
Tests for validation utilities.

Tests the validation functions that ensure data integrity and provide
helpful error messages for invalid inputs.
"""

from datetime import date

import pytest

from snore.utils.validation import (
    validate_date_format,
    validate_date_range,
    validate_profile_exists,
)


class TestProfileValidation:
    """Test profile existence validation."""

    def test_validate_existing_profile(self, initialized_db):
        """Valid profile should be returned."""
        from snore.database.models import Profile

        profile = Profile(
            username="test_user_validation", first_name="Test", last_name="User"
        )
        initialized_db.add(profile)
        initialized_db.commit()

        result = validate_profile_exists("test_user_validation")

        assert result is not None
        assert result.username == "test_user_validation"
        assert result.first_name == "Test"
        assert result.last_name == "User"

    def test_validate_nonexistent_profile_raises_error(self, initialized_db):
        """Non-existent profile should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_profile_exists("nonexistent_user")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower()
        assert "nonexistent_user" in error_msg
        assert "oscar-import status" in error_msg.lower()

    def test_validate_profile_case_sensitive(self, initialized_db):
        """Profile validation should be case-sensitive."""
        from snore.database.models import Profile

        profile = Profile(username="TestUser")
        initialized_db.add(profile)
        initialized_db.commit()

        result = validate_profile_exists("TestUser")
        assert result.username == "TestUser"

        with pytest.raises(ValueError):
            validate_profile_exists("testuser")


class TestDateValidation:
    """Test date format and range validation."""

    def test_valid_date_format(self):
        """Valid YYYY-MM-DD dates should be parsed correctly."""
        result = validate_date_format("2024-11-05")
        assert result == date(2024, 11, 5)

    def test_valid_date_formats_various(self):
        """Test various valid date formats."""
        assert validate_date_format("2024-01-01") == date(2024, 1, 1)
        assert validate_date_format("2024-12-31") == date(2024, 12, 31)
        assert validate_date_format("2024-02-29") == date(2024, 2, 29)

    def test_invalid_date_format_raises_error(self):
        """Invalid date formats should raise ValueError."""
        invalid_formats = [
            "11/05/2024",
            "05-11-2024",
            "Nov 5 2024",
            "2024/11/05",
            "not-a-date",
        ]

        for invalid_format in invalid_formats:
            with pytest.raises(ValueError) as exc_info:
                validate_date_format(invalid_format)

            error_msg = str(exc_info.value)
            assert "invalid date format" in error_msg.lower()
            assert "yyyy-mm-dd" in error_msg.lower()

    def test_invalid_date_values(self):
        """Invalid date values should raise ValueError."""
        invalid_dates = [
            "2024-02-30",
            "2024-13-01",
            "2024-00-01",
            "2024-01-00",
        ]

        for invalid_date in invalid_dates:
            with pytest.raises(ValueError):
                validate_date_format(invalid_date)

    def test_valid_date_range(self):
        """Valid date ranges should pass validation."""
        start = date(2024, 11, 1)
        end = date(2024, 11, 30)

        result = validate_date_range(start, end)
        assert result is True

    def test_valid_date_range_same_day(self):
        """Same start and end date should be valid."""
        same_date = date(2024, 11, 5)

        result = validate_date_range(same_date, same_date)
        assert result is True

    def test_invalid_date_range_raises_error(self):
        """Invalid date range (start > end) should raise ValueError."""
        start = date(2024, 11, 30)
        end = date(2024, 11, 1)

        with pytest.raises(ValueError) as exc_info:
            validate_date_range(start, end)

        error_msg = str(exc_info.value)
        assert "invalid date range" in error_msg.lower()
        assert str(start) in error_msg
        assert str(end) in error_msg
