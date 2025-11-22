"""
Tests for CLI commands.

These tests verify the command-line interface functionality including:
- delete-sessions command with various input modes
- list-profiles command output
- db stats command
- list-sessions command with limits and truncation
"""

from datetime import datetime, timedelta
import pytest
from click.testing import CliRunner
from sqlalchemy import text

from oscar_mcp.cli import cli
from oscar_mcp.database.session import init_database, session_scope, cleanup_database
from oscar_mcp.database import models
from oscar_mcp.database.day_manager import DayManager


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture(autouse=True)
def reset_database_state():
    """Reset global database state before and after each test."""
    cleanup_database()
    yield
    cleanup_database()


@pytest.fixture
def populated_test_db(temp_db):
    """Create a database populated with realistic test data."""
    init_database(str(temp_db))

    with session_scope() as session:
        profile = models.Profile(username="testuser", settings={"day_split_time": "12:00:00"})
        session.add(profile)
        session.flush()

        device = models.Device(
            profile_id=profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
            serial_number="TEST12345",
        )
        session.add(device)
        session.flush()

        base_time = datetime(2025, 10, 1, 22, 0, 0)
        for i in range(10):
            start_time = base_time + timedelta(days=i)
            end_time = start_time + timedelta(hours=8)

            sess = models.Session(
                device_id=device.id,
                device_session_id=f"test_session_{i}",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=8 * 3600,
                has_statistics=True,
                has_event_data=True,
            )
            session.add(sess)
            session.flush()

            day_date = DayManager.get_day_for_session(start_time, profile)
            day = DayManager.create_or_update_day(profile.id, day_date, session)
            sess.day_id = day.id

            session.add(models.Setting(session_id=sess.id, key="mode", value="CPAP"))

            session.add(
                models.Event(
                    session_id=sess.id,
                    event_type="Apnea",
                    start_time=start_time + timedelta(hours=2),
                    duration_seconds=15.0,
                )
            )

            session.add(models.Statistics(session_id=sess.id, ahi=5.2, usage_hours=7.8))

        session.commit()

    return temp_db


class TestDeleteSessionsCommand:
    """Test delete-sessions command with various scenarios."""

    def test_delete_single_session_by_id(self, cli_runner, populated_test_db):
        """Test deleting a single session by ID."""
        result = cli_runner.invoke(
            cli,
            ["delete-sessions", "--db", str(populated_test_db), "--session-id", "1"],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Successfully deleted 1 session(s)" in result.output

        with session_scope() as session:
            remaining = session.query(models.Session).filter_by(id=1).count()
            assert remaining == 0

    def test_delete_multiple_sessions_by_id(self, cli_runner, populated_test_db):
        """Test deleting multiple sessions by ID (tests SQL IN clause fix)."""
        result = cli_runner.invoke(
            cli,
            ["delete-sessions", "--db", str(populated_test_db), "--session-id", "1,2,3"],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Successfully deleted 3 session(s)" in result.output

        with session_scope() as session:
            remaining = (
                session.query(models.Session).filter(models.Session.id.in_([1, 2, 3])).count()
            )
            assert remaining == 0

    def test_delete_sessions_by_date_range(self, cli_runner, populated_test_db):
        """Test deleting sessions by date range."""
        result = cli_runner.invoke(
            cli,
            [
                "delete-sessions",
                "--db",
                str(populated_test_db),
                "--from-date",
                "2025-10-01",
                "--to-date",
                "2025-10-03",
            ],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "sessions" in result.output.lower()

        with session_scope() as session:
            total_remaining = session.query(models.Session).count()
            assert total_remaining < 10

    def test_delete_cascades_to_child_tables(self, cli_runner, populated_test_db):
        """Test that deleting sessions cascades to events, waveforms, statistics."""
        with session_scope() as session:
            events_before = session.execute(
                text("SELECT COUNT(*) FROM events WHERE session_id = 1")
            ).scalar()
            stats_before = session.execute(
                text("SELECT COUNT(*) FROM statistics WHERE session_id = 1")
            ).scalar()

        assert events_before > 0
        assert stats_before > 0

        result = cli_runner.invoke(
            cli,
            ["delete-sessions", "--db", str(populated_test_db), "--session-id", "1"],
            input="y\n",
        )

        assert result.exit_code == 0

        with session_scope() as session:
            events_after = session.execute(
                text("SELECT COUNT(*) FROM events WHERE session_id = 1")
            ).scalar()
            stats_after = session.execute(
                text("SELECT COUNT(*) FROM statistics WHERE session_id = 1")
            ).scalar()

        assert events_after == 0
        assert stats_after == 0

    def test_delete_session_datetime_formatting(self, cli_runner, populated_test_db):
        """Test that datetime formatting works correctly in delete preview."""
        result = cli_runner.invoke(
            cli,
            ["delete-sessions", "--db", str(populated_test_db), "--session-id", "1"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "2025-10-" in result.output
        assert "Deletion cancelled" in result.output


class TestListProfilesCommand:
    """Test list-profiles command."""

    def test_list_profiles_shows_correct_session_count(self, cli_runner, populated_test_db):
        """Test that list-profiles shows correct session count (tests day linking fix)."""
        result = cli_runner.invoke(cli, ["list-profiles", "--db", str(populated_test_db)])

        assert result.exit_code == 0
        assert "testuser" in result.output
        assert "Sessions: 10" in result.output
        assert "Days with data: 10" in result.output

    def test_list_profiles_empty_database(self, cli_runner, temp_db):
        """Test list-profiles with empty database."""
        init_database(str(temp_db))

        result = cli_runner.invoke(cli, ["list-profiles", "--db", str(temp_db)])

        assert result.exit_code == 0
        assert "No profiles found" in result.output


class TestDbStatsCommand:
    """Test db stats command."""

    def test_db_stats_datetime_formatting(self, cli_runner, populated_test_db):
        """Test that db stats correctly formats datetime values (tests string->datetime fix)."""
        result = cli_runner.invoke(cli, ["db", "stats", "--db", str(populated_test_db)])

        assert result.exit_code == 0
        assert "Database Statistics" in result.output
        assert "Devices: 1" in result.output
        assert "Sessions: 10" in result.output
        assert "Events: 10" in result.output
        assert "Date range:" in result.output
        assert "2025-10-" in result.output

    def test_db_stats_empty_database(self, cli_runner, temp_db):
        """Test db stats with empty database."""
        init_database(str(temp_db))

        result = cli_runner.invoke(cli, ["db", "stats", "--db", str(temp_db)])

        assert result.exit_code == 0
        assert "Devices: 0" in result.output
        assert "Sessions: 0" in result.output


class TestListSessionsCommand:
    """Test list-sessions command."""

    def test_list_sessions_default_limit(self, cli_runner, populated_test_db):
        """Test list-sessions uses default limit of 20."""
        result = cli_runner.invoke(cli, ["list-sessions", "--db", str(populated_test_db)])

        assert result.exit_code == 0
        assert "testuser" in result.output

        session_rows = [
            line for line in result.output.split("\n") if "2025-10-" in line and "testuser" in line
        ]
        assert len(session_rows) == 10

    def test_list_sessions_custom_limit(self, cli_runner, populated_test_db):
        """Test list-sessions with custom limit."""
        result = cli_runner.invoke(
            cli, ["list-sessions", "--db", str(populated_test_db), "--limit", "5"]
        )

        assert result.exit_code == 0

        session_rows = [
            line for line in result.output.split("\n") if "2025-10-" in line and "testuser" in line
        ]
        assert len(session_rows) == 5
        assert "Showing 5 of 10 sessions" in result.output
        assert "Tip:" in result.output

    def test_list_sessions_unlimited(self, cli_runner, populated_test_db):
        """Test list-sessions with --limit 0 shows all sessions."""
        result = cli_runner.invoke(
            cli, ["list-sessions", "--db", str(populated_test_db), "--limit", "0"]
        )

        assert result.exit_code == 0
        session_rows = [
            line for line in result.output.split("\n") if "2025-10-" in line and "testuser" in line
        ]
        assert len(session_rows) == 10

    def test_list_sessions_no_truncation_message(self, cli_runner, temp_db):
        """Test list-sessions doesn't show truncation when all results fit."""
        init_database(str(temp_db))

        with session_scope() as session:
            profile = models.Profile(username="testuser", settings={"day_split_time": "12:00:00"})
            session.add(profile)
            session.flush()

            device = models.Device(
                profile_id=profile.id, manufacturer="Test", model="Test", serial_number="TEST"
            )
            session.add(device)
            session.flush()

            start_time = datetime(2025, 10, 1, 22, 0, 0)
            sess = models.Session(
                device_id=device.id,
                device_session_id="test_session_1",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
            )
            session.add(sess)
            session.commit()

        result = cli_runner.invoke(cli, ["list-sessions", "--db", str(temp_db)])

        assert result.exit_code == 0
        assert "Showing" not in result.output or "Showing all" in result.output
