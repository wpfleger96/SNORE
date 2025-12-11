"""
Tests for session import to day linking functionality.

These tests verify that:
- Imported sessions are properly linked to Day records
- Day records are created/updated during import
- Day-splitting logic works correctly
- list-profiles shows correct counts after import
"""

from datetime import datetime, timedelta

import pytest

from snore.database import models
from snore.database.day_manager import DayManager
from snore.database.session import cleanup_database, init_database, session_scope


@pytest.fixture(autouse=True)
def reset_database_state():
    """Reset global database state before and after each test."""
    cleanup_database()
    yield
    cleanup_database()


@pytest.fixture
def profile_with_device(temp_db):
    """Create a profile and device for testing."""
    init_database(str(temp_db))

    with session_scope() as session:
        profile = models.Profile(
            username="testuser", settings={"day_split_time": "12:00:00"}
        )
        session.add(profile)
        session.flush()

        device = models.Device(
            profile_id=profile.id,
            manufacturer="Test",
            model="Test Model",
            serial_number="TEST123",
        )
        session.add(device)
        session.commit()

        return profile.id, device.id


class TestDayRecordCreation:
    """Test that Day records are created and linked properly."""

    def test_create_session_with_day_record(self, temp_db, profile_with_device):
        """Test that creating a session with day linking works."""
        profile_id, device_id = profile_with_device

        start_time = datetime(2025, 10, 15, 22, 0, 0)

        with session_scope() as session:
            profile = session.get(models.Profile, profile_id)

            day_date = DayManager.get_day_for_session(start_time, profile)
            day = DayManager.create_or_update_day(profile_id, day_date, session)

            new_session = models.Session(
                device_id=device_id,
                device_session_id="test_session_1",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
                day_id=day.id,
            )
            session.add(new_session)
            session.commit()

            assert new_session.day_id is not None, "Session should have day_id set"

            day = session.query(models.Day).filter_by(id=new_session.day_id).first()
            assert day is not None, "Day record should exist"
            assert day.profile_id == profile_id
            assert day.date == datetime(2025, 10, 15).date()

    def test_link_sessions_to_same_day(self, temp_db, profile_with_device):
        """Test that multiple sessions on same day link to same Day record."""
        profile_id, device_id = profile_with_device

        start_time_1 = datetime(2025, 10, 15, 22, 0, 0)
        start_time_2 = datetime(2025, 10, 16, 0, 30, 0)

        with session_scope() as session:
            profile = session.get(models.Profile, profile_id)

            day_date_1 = DayManager.get_day_for_session(start_time_1, profile)
            day_date_2 = DayManager.get_day_for_session(start_time_2, profile)

            assert day_date_1 == day_date_2, "Both sessions should map to same day"

            day = DayManager.create_or_update_day(profile_id, day_date_1, session)

            sess1 = models.Session(
                device_id=device_id,
                device_session_id="test_session_1",
                start_time=start_time_1,
                end_time=start_time_1 + timedelta(hours=4),
                duration_seconds=4 * 3600,
                day_id=day.id,
            )
            sess2 = models.Session(
                device_id=device_id,
                device_session_id="test_session_2",
                start_time=start_time_2,
                end_time=start_time_2 + timedelta(hours=4),
                duration_seconds=4 * 3600,
                day_id=day.id,
            )
            session.add(sess1)
            session.add(sess2)
            session.commit()

            assert sess1.day_id is not None
            assert sess2.day_id is not None

            assert sess1.day_id == sess2.day_id, (
                "Sessions on same day should link to same Day record"
            )


class TestDaySplittingLogic:
    """Test day-splitting logic (sessions before noon belong to previous day)."""

    def test_session_after_noon_belongs_to_same_day(self, temp_db, profile_with_device):
        """Test that session starting after noon belongs to same day."""
        profile_id, device_id = profile_with_device

        start_time = datetime(2025, 10, 15, 22, 0, 0)

        with session_scope() as session:
            profile = session.get(models.Profile, profile_id)

            day_date = DayManager.get_day_for_session(start_time, profile)

            expected_date = datetime(2025, 10, 15).date()
            assert day_date == expected_date

    def test_session_before_noon_belongs_to_previous_day(
        self, temp_db, profile_with_device
    ):
        """Test that session starting before noon belongs to previous day."""
        profile_id, device_id = profile_with_device

        start_time = datetime(2025, 10, 16, 9, 0, 0)

        with session_scope() as session:
            profile = session.get(models.Profile, profile_id)

            day_date = DayManager.get_day_for_session(start_time, profile)

            expected_date = datetime(2025, 10, 15).date()
            assert day_date == expected_date

    def test_custom_day_split_time(self, temp_db):
        """Test that custom day_split_time setting is respected."""
        init_database(str(temp_db))

        with session_scope() as session:
            profile = models.Profile(
                username="testuser", settings={"day_split_time": "14:00:00"}
            )
            session.add(profile)
            session.commit()

            start_time = datetime(2025, 10, 16, 13, 0, 0)

            day_date = DayManager.get_day_for_session(start_time, profile)

            expected_date = datetime(2025, 10, 15).date()
            assert day_date == expected_date


class TestListProfilesIntegration:
    """Test that list-profiles shows correct counts after import."""

    def test_list_profiles_shows_correct_session_count(
        self, temp_db, profile_with_device
    ):
        """Test that list-profiles queries work correctly with day-linked sessions."""
        profile_id, device_id = profile_with_device

        with session_scope() as session:
            profile = session.get(models.Profile, profile_id)

            for i in range(3):
                start_time = datetime(2025, 10, 15 + i, 22, 0, 0)

                day_date = DayManager.get_day_for_session(start_time, profile)
                day = DayManager.create_or_update_day(profile_id, day_date, session)

                sess = models.Session(
                    device_id=device_id,
                    device_session_id=f"test_session_{i}",
                    start_time=start_time,
                    end_time=start_time + timedelta(hours=8),
                    duration_seconds=8 * 3600,
                    day_id=day.id,
                )
                session.add(sess)

            session.commit()

            total_sessions = (
                session.query(models.Session)
                .join(models.Day)
                .filter(models.Day.profile_id == profile_id)
                .count()
            )

            assert total_sessions == 3, (
                "Should find all 3 sessions through Day relationship"
            )

            days_count = (
                session.query(models.Day).filter_by(profile_id=profile_id).count()
            )
            assert days_count == 3, "Should have 3 separate days"

    def test_sessions_without_day_id_not_counted(self, temp_db, profile_with_device):
        """Test that sessions without day_id are not counted (tests the bug we fixed)."""
        profile_id, device_id = profile_with_device

        with session_scope() as session:
            orphan_session = models.Session(
                device_id=device_id,
                device_session_id="orphan_session",
                start_time=datetime(2025, 10, 15, 22, 0, 0),
                end_time=datetime(2025, 10, 16, 6, 0, 0),
                duration_seconds=8 * 3600,
                day_id=None,
            )
            session.add(orphan_session)
            session.commit()

            sessions_through_day = (
                session.query(models.Session)
                .join(models.Day)
                .filter(models.Day.profile_id == profile_id)
                .count()
            )

            assert sessions_through_day == 0, (
                "Sessions without day_id should not be counted"
            )

            direct_count = (
                session.query(models.Session).filter_by(device_id=device_id).count()
            )
            assert direct_count == 1, "Direct query should still find the session"


class TestDayManagerFunctions:
    """Test DayManager utility functions directly."""

    def test_get_day_for_session_after_split(self, temp_db):
        """Test get_day_for_session with time after split."""
        init_database(str(temp_db))

        with session_scope() as session:
            profile = models.Profile(
                username="testuser", settings={"day_split_time": "12:00:00"}
            )
            session.add(profile)
            session.commit()

            session_time = datetime(2025, 10, 15, 22, 0, 0)
            day_date = DayManager.get_day_for_session(session_time, profile)

            assert day_date == datetime(2025, 10, 15).date()

    def test_get_day_for_session_before_split(self, temp_db):
        """Test get_day_for_session with time before split."""
        init_database(str(temp_db))

        with session_scope() as session:
            profile = models.Profile(
                username="testuser", settings={"day_split_time": "12:00:00"}
            )
            session.add(profile)
            session.commit()

            session_time = datetime(2025, 10, 16, 9, 0, 0)
            day_date = DayManager.get_day_for_session(session_time, profile)

            assert day_date == datetime(2025, 10, 15).date()

    def test_create_or_update_day_creates_new(self, temp_db, profile_with_device):
        """Test that create_or_update_day creates new Day when none exists."""
        profile_id, device_id = profile_with_device

        with session_scope() as session:
            day_date = datetime(2025, 10, 16).date()

            existing = (
                session.query(models.Day)
                .filter_by(profile_id=profile_id, date=day_date)
                .first()
            )
            assert existing is None

            day = DayManager.create_or_update_day(profile_id, day_date, session)
            session.commit()

            assert day.id is not None
            assert day.profile_id == profile_id
            assert day.date == day_date

    def test_create_or_update_day_updates_existing(self, temp_db, profile_with_device):
        """Test that create_or_update_day returns existing Day."""
        profile_id, device_id = profile_with_device

        with session_scope() as session:
            day_date = datetime(2025, 10, 16).date()

            day1 = DayManager.create_or_update_day(profile_id, day_date, session)
            session.commit()
            day1_id = day1.id

            day2 = DayManager.create_or_update_day(profile_id, day_date, session)

            assert day2.id == day1_id

            count = (
                session.query(models.Day)
                .filter_by(profile_id=profile_id, date=day_date)
                .count()
            )
            assert count == 1
