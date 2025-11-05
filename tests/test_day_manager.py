"""
Tests for DayManager business logic.

Tests the critical day-splitting algorithm and statistical aggregation logic
that determines which calendar day sessions belong to and how statistics
are aggregated across multiple sessions.
"""

import pytest
from datetime import datetime, date
from oscar_mcp.database.day_manager import DayManager


@pytest.mark.business_logic
class TestDaySplitLogic:
    """Test day-splitting algorithm (OSCAR-compatible noon boundary logic)."""

    def test_default_noon_split_before(self, db_session, test_profile_factory):
        """Session before noon belongs to previous calendar day."""
        profile = test_profile_factory(username="test_user", day_split_time="12:00:00")

        # Session at 11:59 AM on Nov 5
        session_start = datetime(2024, 11, 5, 11, 59, 0)

        result = DayManager.get_day_for_session(session_start, profile)

        # Should belong to Nov 4 (previous day)
        assert result == date(2024, 11, 4)

    def test_default_noon_split_at_boundary(self, db_session, test_profile_factory):
        """Session exactly at noon belongs to same calendar day."""
        profile = test_profile_factory(username="test_user", day_split_time="12:00:00")

        # Session at 12:00 PM on Nov 5
        session_start = datetime(2024, 11, 5, 12, 0, 0)

        result = DayManager.get_day_for_session(session_start, profile)

        # Should belong to Nov 5 (same day)
        assert result == date(2024, 11, 5)

    def test_default_noon_split_after(self, db_session, test_profile_factory):
        """Session after noon belongs to same calendar day."""
        profile = test_profile_factory(username="test_user", day_split_time="12:00:00")

        # Session at 12:01 PM on Nov 5
        session_start = datetime(2024, 11, 5, 12, 1, 0)

        result = DayManager.get_day_for_session(session_start, profile)

        # Should belong to Nov 5 (same day)
        assert result == date(2024, 11, 5)

    def test_custom_split_time_6am(self, db_session, test_profile_factory):
        """Custom 6 AM split time."""
        profile = test_profile_factory(username="test_user", day_split_time="06:00:00")

        # Session at 5:59 AM on Nov 5
        session_before = datetime(2024, 11, 5, 5, 59, 0)
        assert DayManager.get_day_for_session(session_before, profile) == date(2024, 11, 4)

        # Session at 6:00 AM on Nov 5
        session_at = datetime(2024, 11, 5, 6, 0, 0)
        assert DayManager.get_day_for_session(session_at, profile) == date(2024, 11, 5)

    def test_custom_split_time_2pm(self, db_session, test_profile_factory):
        """Custom 2 PM split time."""
        profile = test_profile_factory(username="test_user", day_split_time="14:00:00")

        # Session at 1:59 PM on Nov 5
        session_before = datetime(2024, 11, 5, 13, 59, 0)
        assert DayManager.get_day_for_session(session_before, profile) == date(2024, 11, 4)

        # Session at 2:00 PM on Nov 5
        session_at = datetime(2024, 11, 5, 14, 0, 0)
        assert DayManager.get_day_for_session(session_at, profile) == date(2024, 11, 5)

    def test_midnight_session_with_noon_split(self, db_session, test_profile_factory):
        """Midnight session (00:00) with noon split belongs to previous day."""
        profile = test_profile_factory(username="test_user", day_split_time="12:00:00")

        # Session at midnight (00:00) on Nov 5
        session_start = datetime(2024, 11, 5, 0, 0, 0)

        result = DayManager.get_day_for_session(session_start, profile)

        # Midnight is before noon, so belongs to Nov 4
        assert result == date(2024, 11, 4)

    def test_late_night_session_23_59(self, db_session, test_profile_factory):
        """Late night session (23:59) with noon split belongs to same day."""
        profile = test_profile_factory(username="test_user", day_split_time="12:00:00")

        # Session at 23:59 PM on Nov 5
        session_start = datetime(2024, 11, 5, 23, 59, 0)

        result = DayManager.get_day_for_session(session_start, profile)

        # 23:59 is after noon, so belongs to Nov 5
        assert result == date(2024, 11, 5)

    def test_missing_settings_uses_default(self, db_session, test_profile_factory):
        """Profile without day_split_time setting uses default noon."""
        profile = test_profile_factory(username="test_user")
        profile.settings = {}  # Clear settings

        # Session at 11:59 AM
        session_before = datetime(2024, 11, 5, 11, 59, 0)
        assert DayManager.get_day_for_session(session_before, profile) == date(2024, 11, 4)

        # Session at 12:00 PM
        session_at = datetime(2024, 11, 5, 12, 0, 0)
        assert DayManager.get_day_for_session(session_at, profile) == date(2024, 11, 5)


@pytest.mark.business_logic
class TestStatisticalAggregation:
    """Test statistical aggregation across multiple sessions."""

    def test_single_session_aggregation(self, db_session, test_device, test_session_factory):
        """Single session aggregation should copy statistics directly."""
        device, profile = test_device

        # Create a session with statistics
        session_start = datetime(2024, 11, 5, 12, 0, 0)
        session = test_session_factory(
            device_id=device.id,
            start_time=session_start,
            duration_hours=8.0,
            obstructive_apneas=10,
            central_apneas=5,
            hypopneas=8,
            reras=3,
            ahi=5.0,
            oai=1.25,
            cai=0.625,
            hi=1.0,
            pressure_min=8.0,
            pressure_max=15.0,
            pressure_median=11.0,
            pressure_mean=11.5,
            leak_min=0.0,
            leak_max=24.0,
            leak_median=5.0,
        )

        # Link session to day
        day = DayManager.link_session_to_day(session, profile, db_session)

        # Verify aggregated statistics match single session
        assert day.session_count == 1
        assert day.total_therapy_hours == pytest.approx(8.0, abs=0.01)
        assert day.obstructive_apneas == 10
        assert day.central_apneas == 5
        assert day.hypopneas == 8
        assert day.reras == 3
        assert day.ahi == pytest.approx(5.0, abs=0.01)
        assert day.pressure_min == pytest.approx(8.0, abs=0.01)
        assert day.pressure_max == pytest.approx(15.0, abs=0.01)

    def test_multi_session_event_counts_sum(self, db_session, test_device, test_session_factory):
        """Event counts should sum across sessions."""
        device, profile = test_device

        # Create two sessions
        session1 = test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=4.0,
            obstructive_apneas=10,
            central_apneas=5,
            hypopneas=8,
            reras=3,
        )

        session2 = test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=6.0,
            obstructive_apneas=15,
            central_apneas=3,
            hypopneas=12,
            reras=5,
        )

        # Link both sessions to day
        day = DayManager.link_session_to_day(session1, profile, db_session)
        day = DayManager.link_session_to_day(session2, profile, db_session)

        # Verify sums
        assert day.session_count == 2
        assert day.obstructive_apneas == 25  # 10 + 15
        assert day.central_apneas == 8  # 5 + 3
        assert day.hypopneas == 20  # 8 + 12
        assert day.reras == 8  # 3 + 5

    def test_weighted_average_ahi(self, db_session, test_device, test_session_factory):
        """AHI should be weighted by session duration."""
        device, profile = test_device

        # Session 1: 4 hours with AHI=10
        session1 = test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=4.0,
            ahi=10.0,
        )

        # Session 2: 2 hours with AHI=4
        session2 = test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=2.0,
            ahi=4.0,
        )

        # Link both to day
        day = DayManager.link_session_to_day(session1, profile, db_session)
        day = DayManager.link_session_to_day(session2, profile, db_session)

        # Expected: (10*4 + 4*2) / 6 = 48/6 = 8.0
        assert day.ahi == pytest.approx(8.0, abs=0.01)

    def test_pressure_min_max_across_sessions(self, db_session, test_device, test_session_factory):
        """Pressure min/max should be extremes across all sessions."""
        device, profile = test_device

        # Session 1: pressure range 8-15
        session1 = test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=4.0,
            pressure_min=8.0,
            pressure_max=15.0,
            pressure_median=11.0,
        )

        # Session 2: pressure range 6-12
        session2 = test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=4.0,
            pressure_min=6.0,
            pressure_max=12.0,
            pressure_median=9.0,
        )

        # Link both to day
        day = DayManager.link_session_to_day(session1, profile, db_session)
        day = DayManager.link_session_to_day(session2, profile, db_session)

        # Min should be 6.0, Max should be 15.0
        assert day.pressure_min == pytest.approx(6.0, abs=0.01)
        assert day.pressure_max == pytest.approx(15.0, abs=0.01)

    def test_empty_day_resets_statistics(self, db_session, test_device, test_session_factory):
        """Day with no sessions should have reset statistics."""
        device, profile = test_device

        # Create a day with a session first
        session = test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=8.0,
            obstructive_apneas=10,
            ahi=5.0,
        )

        day = DayManager.link_session_to_day(session, profile, db_session)
        assert day.session_count == 1
        assert day.obstructive_apneas == 10

        # Now remove the session and recalculate
        session.day_id = None
        db_session.flush()
        DayManager._aggregate_day_statistics(day, db_session)

        # Statistics should be reset
        assert day.session_count == 0
        assert day.total_therapy_hours == 0.0
        assert day.obstructive_apneas == 0
        assert day.ahi is None

    def test_partial_data_null_values(self, db_session, test_device, test_session_factory):
        """Sessions with missing statistics should be handled gracefully."""
        device, profile = test_device

        # Session 1: has AHI
        session1 = test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=4.0,
            ahi=8.0,
        )

        # Session 2: no AHI (None)
        session2 = test_session_factory(
            device_id=device.id, start_time=datetime(2024, 11, 5, 22, 0, 0), duration_hours=4.0
        )

        # Link both to day
        day = DayManager.link_session_to_day(session1, profile, db_session)
        day = DayManager.link_session_to_day(session2, profile, db_session)

        # AHI should only consider session1 (not null)
        assert day.ahi == pytest.approx(8.0, abs=0.01)

    def test_zero_duration_session_handling(self, db_session, test_device, test_session_factory):
        """Sessions with zero duration should not cause division by zero."""
        device, profile = test_device

        # Create session with 0 duration (edge case)
        session = test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=0.0,
            ahi=5.0,
        )

        # Link to day - should not crash
        day = DayManager.link_session_to_day(session, profile, db_session)

        # Duration should be 0
        assert day.total_therapy_hours == pytest.approx(0.0, abs=0.01)
        # AHI handling depends on implementation - should not crash

    def test_total_therapy_hours_sums_durations(
        self, db_session, test_device, test_session_factory
    ):
        """Total therapy hours should sum all session durations."""
        device, profile = test_device

        # Create three sessions with different durations
        session1 = test_session_factory(
            device_id=device.id, start_time=datetime(2024, 11, 5, 12, 0, 0), duration_hours=4.0
        )

        session2 = test_session_factory(
            device_id=device.id, start_time=datetime(2024, 11, 5, 18, 0, 0), duration_hours=2.5
        )

        session3 = test_session_factory(
            device_id=device.id, start_time=datetime(2024, 11, 5, 22, 0, 0), duration_hours=1.5
        )

        # Link all to day
        day = DayManager.link_session_to_day(session1, profile, db_session)
        day = DayManager.link_session_to_day(session2, profile, db_session)
        day = DayManager.link_session_to_day(session3, profile, db_session)

        # Total should be 4.0 + 2.5 + 1.5 = 8.0
        assert day.total_therapy_hours == pytest.approx(8.0, abs=0.01)
        assert day.session_count == 3
