"""
Tests for validation utilities.

Tests the validation functions that ensure data integrity and provide
helpful error messages for invalid inputs.
"""

from datetime import date

import pytest

from oscar_mcp.utils.validation import (
    validate_date_format,
    validate_date_range,
    validate_period_type,
    validate_profile_exists,
)


class TestProfileValidation:
    """Test profile existence validation."""

    def test_validate_existing_profile(self, initialized_db):
        """Valid profile should be returned."""
        from oscar_mcp.database.models import Profile

        # Create a test profile
        profile = Profile(
            username="test_user_validation", first_name="Test", last_name="User"
        )
        initialized_db.add(profile)
        initialized_db.commit()

        # Validate should return the profile
        result = validate_profile_exists("test_user_validation")

        assert result is not None
        assert result.username == "test_user_validation"
        assert result.first_name == "Test"
        assert result.last_name == "User"

    def test_validate_nonexistent_profile_raises_error(self, initialized_db):
        """Non-existent profile should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_profile_exists("nonexistent_user")

        # Verify error message is helpful
        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower()
        assert "nonexistent_user" in error_msg
        assert "oscar-import status" in error_msg.lower()

    def test_validate_profile_case_sensitive(self, initialized_db):
        """Profile validation should be case-sensitive."""
        from oscar_mcp.database.models import Profile

        profile = Profile(username="TestUser")
        initialized_db.add(profile)
        initialized_db.commit()

        # Exact match should work
        result = validate_profile_exists("TestUser")
        assert result.username == "TestUser"

        # Different case should fail
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
        assert validate_date_format("2024-02-29") == date(2024, 2, 29)  # Leap year

    def test_invalid_date_format_raises_error(self):
        """Invalid date formats should raise ValueError."""
        invalid_formats = [
            "11/05/2024",  # US format
            "05-11-2024",  # DD-MM-YYYY
            "Nov 5 2024",  # Text month
            "2024/11/05",  # Slashes instead of dashes
            "not-a-date",  # Invalid
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
            "2024-02-30",  # Invalid day for February
            "2024-13-01",  # Invalid month
            "2024-00-01",  # Invalid month (zero)
            "2024-01-00",  # Invalid day (zero)
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


class TestPeriodTypeValidation:
    """Test period type validation."""

    def test_all_valid_period_types(self):
        """All valid period types should pass."""
        valid_types = ["daily", "weekly", "monthly", "yearly"]

        for period_type in valid_types:
            result = validate_period_type(period_type)
            assert result is True

    def test_invalid_period_type_raises_error(self):
        """Invalid period type should raise ValueError."""
        invalid_types = ["hourly", "quarterly", "biweekly", "DAILY", ""]

        for invalid_type in invalid_types:
            with pytest.raises(ValueError) as exc_info:
                validate_period_type(invalid_type)

            error_msg = str(exc_info.value)
            assert "invalid period type" in error_msg.lower()
            # Should list valid types
            assert "daily" in error_msg.lower()
            assert "weekly" in error_msg.lower()
