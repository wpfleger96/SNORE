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

from snore.cli import cli
from snore.database import models
from snore.database.day_manager import DayManager
from snore.database.session import cleanup_database, init_database, session_scope


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
        profile = models.Profile(
            username="testuser", settings={"day_split_time": "12:00:00"}
        )
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
            [
                "delete-sessions",
                "--db",
                str(populated_test_db),
                "--session-id",
                "1,2,3",
            ],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Successfully deleted 3 session(s)" in result.output

        with session_scope() as session:
            remaining = (
                session.query(models.Session)
                .filter(models.Session.id.in_([1, 2, 3]))
                .count()
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

    def test_list_profiles_shows_correct_session_count(
        self, cli_runner, populated_test_db
    ):
        """Test that list-profiles shows correct session count (tests day linking fix)."""
        result = cli_runner.invoke(
            cli, ["list-profiles", "--db", str(populated_test_db)]
        )

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
        result = cli_runner.invoke(
            cli, ["list-sessions", "--db", str(populated_test_db)]
        )

        assert result.exit_code == 0
        assert "testuser" in result.output

        session_rows = [
            line
            for line in result.output.split("\n")
            if "2025-10-" in line and "testuser" in line
        ]
        assert len(session_rows) == 10

    def test_list_sessions_custom_limit(self, cli_runner, populated_test_db):
        """Test list-sessions with custom limit."""
        result = cli_runner.invoke(
            cli, ["list-sessions", "--db", str(populated_test_db), "--limit", "5"]
        )

        assert result.exit_code == 0

        session_rows = [
            line
            for line in result.output.split("\n")
            if "2025-10-" in line and "testuser" in line
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
            line
            for line in result.output.split("\n")
            if "2025-10-" in line and "testuser" in line
        ]
        assert len(session_rows) == 10

    def test_list_sessions_no_truncation_message(self, cli_runner, temp_db):
        """Test list-sessions doesn't show truncation when all results fit."""
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
                model="Test",
                serial_number="TEST",
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


@pytest.fixture
def db_with_analysis(temp_db):
    """Create a database populated with sessions and analysis results."""
    init_database(str(temp_db))

    with session_scope() as session:
        profile = models.Profile(
            username="testuser", settings={"day_split_time": "12:00:00"}
        )
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
        for i in range(5):
            start_time = base_time + timedelta(days=i)
            end_time = start_time + timedelta(hours=8)

            sess = models.Session(
                device_id=device.id,
                device_session_id=f"test_session_{i}",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=8 * 3600,
            )
            session.add(sess)
            session.flush()

            day_date = DayManager.get_day_for_session(start_time, profile)
            day = DayManager.create_or_update_day(profile.id, day_date, session)
            sess.day_id = day.id

            num_analyses = 3 if i < 2 else 1
            for j in range(num_analyses):
                analysis_json = {
                    "session_id": sess.id,
                    "timestamp_start": start_time.timestamp(),
                    "timestamp_end": end_time.timestamp(),
                    "session_duration_hours": 8.0,
                    "total_breaths": 1000,
                    "machine_events": [],
                    "mode_results": {
                        "aasm": {
                            "mode_name": "aasm",
                            "ahi": 5.0,
                            "rdi": 5.0,
                            "apneas": [],
                            "hypopneas": [],
                        }
                    },
                    "flow_analysis": {
                        "total_breaths": 1000,
                        "class_distribution": {
                            1: 500,
                            2: 200,
                            3: 100,
                            4: 100,
                            5: 50,
                            6: 30,
                            7: 20,
                        },
                        "flow_limitation_index": 0.25,
                        "average_confidence": 0.85,
                        "patterns": [],
                    },
                    "csr_detection": None,
                    "periodic_breathing": None,
                }

                analysis = models.AnalysisResult(
                    session_id=sess.id,
                    timestamp_start=start_time,
                    timestamp_end=end_time,
                    programmatic_result_json=analysis_json,
                    processing_time_ms=100,
                    engine_versions_json={"version": "1.0.0"},
                    created_at=start_time + timedelta(minutes=j),
                )
                session.add(analysis)
                session.flush()

                pattern = models.DetectedPattern(
                    analysis_result_id=analysis.id,
                    pattern_id="TEST_PATTERN",
                    start_time=start_time,
                    duration=8 * 3600,
                    confidence=0.95,
                    detected_by="programmatic",
                    metrics_json={"test": "pattern"},
                )
                session.add(pattern)

        session.commit()

    return temp_db


class TestDeleteAnalysisCommand:
    """Test delete-analysis command with various scenarios."""

    def test_delete_analysis_single_session(self, cli_runner, db_with_analysis):
        """Test deleting analysis for a single session (latest only)."""
        with session_scope() as session:
            analysis_before = (
                session.query(models.AnalysisResult).filter_by(session_id=1).count()
            )
            assert analysis_before == 3

        result = cli_runner.invoke(
            cli,
            [
                "delete-analysis",
                "--db",
                str(db_with_analysis),
                "--session-id",
                "1",
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully deleted 1 analysis record(s)" in result.output

        with session_scope() as session:
            analysis_after = (
                session.query(models.AnalysisResult).filter_by(session_id=1).count()
            )
            assert analysis_after == 2

            sess = session.query(models.Session).filter_by(id=1).first()
            assert sess is not None

    def test_delete_analysis_all_versions(self, cli_runner, db_with_analysis):
        """Test deleting all analysis versions for a session."""
        result = cli_runner.invoke(
            cli,
            [
                "delete-analysis",
                "--db",
                str(db_with_analysis),
                "--session-id",
                "1",
                "--all-versions",
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully deleted 3 analysis record(s)" in result.output

        with session_scope() as session:
            analysis_after = (
                session.query(models.AnalysisResult).filter_by(session_id=1).count()
            )
            assert analysis_after == 0

            sess = session.query(models.Session).filter_by(id=1).first()
            assert sess is not None

    def test_delete_analysis_multiple_sessions(self, cli_runner, db_with_analysis):
        """Test deleting analysis for multiple sessions."""
        result = cli_runner.invoke(
            cli,
            [
                "delete-analysis",
                "--db",
                str(db_with_analysis),
                "--session-id",
                "1,2,3",
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully deleted 3 analysis record(s)" in result.output
        assert "3 session(s)" in result.output

        with session_scope() as session:
            analysis_1 = (
                session.query(models.AnalysisResult).filter_by(session_id=1).count()
            )
            analysis_2 = (
                session.query(models.AnalysisResult).filter_by(session_id=2).count()
            )
            analysis_3 = (
                session.query(models.AnalysisResult).filter_by(session_id=3).count()
            )

            assert analysis_1 == 2
            assert analysis_2 == 2
            assert analysis_3 == 0

    def test_delete_analysis_date_range(self, cli_runner, db_with_analysis):
        """Test deleting analysis by date range."""
        result = cli_runner.invoke(
            cli,
            [
                "delete-analysis",
                "--db",
                str(db_with_analysis),
                "--from-date",
                "2025-10-01",
                "--to-date",
                "2025-10-03",
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully deleted" in result.output

        with session_scope() as session:
            total_analysis = session.query(models.AnalysisResult).count()
            assert total_analysis == 7

    def test_delete_analysis_dry_run(self, cli_runner, db_with_analysis):
        """Test dry-run mode doesn't actually delete."""
        with session_scope() as session:
            analysis_before = session.query(models.AnalysisResult).count()

        result = cli_runner.invoke(
            cli,
            [
                "delete-analysis",
                "--db",
                str(db_with_analysis),
                "--session-id",
                "1",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "DRY RUN MODE" in result.output
        assert "Dry run complete" in result.output

        with session_scope() as session:
            analysis_after = session.query(models.AnalysisResult).count()
            assert analysis_after == analysis_before

    def test_delete_analysis_cancellation(self, cli_runner, db_with_analysis):
        """Test that user can cancel deletion."""
        result = cli_runner.invoke(
            cli,
            ["delete-analysis", "--db", str(db_with_analysis), "--session-id", "1"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "Deletion cancelled" in result.output

        with session_scope() as session:
            analysis = (
                session.query(models.AnalysisResult).filter_by(session_id=1).count()
            )
            assert analysis == 3

    def test_delete_analysis_no_filter_error(self, cli_runner, db_with_analysis):
        """Test that command errors when no filter is provided."""
        result = cli_runner.invoke(
            cli, ["delete-analysis", "--db", str(db_with_analysis)]
        )

        assert result.exit_code == 1
        assert "must specify at least one filter" in result.output

    def test_delete_analysis_no_sessions_found(self, cli_runner, temp_db):
        """Test graceful handling when no sessions have analysis."""
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
                model="Test",
                serial_number="TEST",
            )
            session.add(device)
            session.flush()

            sess = models.Session(
                device_id=device.id,
                device_session_id="test_session_1",
                start_time=datetime(2025, 10, 1, 22, 0, 0),
                end_time=datetime(2025, 10, 2, 6, 0, 0),
                duration_seconds=8 * 3600,
            )
            session.add(sess)
            session.commit()

        result = cli_runner.invoke(
            cli,
            ["delete-analysis", "--db", str(temp_db), "--session-id", "1"],
        )

        assert result.exit_code == 0
        assert "No sessions with analysis results found" in result.output

    def test_delete_analysis_cascades_to_patterns(self, cli_runner, db_with_analysis):
        """Test that deleting analysis cascades to detected patterns."""
        with session_scope() as session:
            analysis = (
                session.query(models.AnalysisResult)
                .filter_by(session_id=1)
                .order_by(models.AnalysisResult.created_at.desc())
                .first()
            )
            latest_analysis_id = analysis.id

            patterns_before = (
                session.query(models.DetectedPattern)
                .filter_by(analysis_result_id=latest_analysis_id)
                .count()
            )
            assert patterns_before > 0

        result = cli_runner.invoke(
            cli,
            [
                "delete-analysis",
                "--db",
                str(db_with_analysis),
                "--session-id",
                "1",
                "--force",
            ],
        )

        assert result.exit_code == 0

        with session_scope() as session:
            patterns_after = (
                session.query(models.DetectedPattern)
                .filter_by(analysis_result_id=latest_analysis_id)
                .count()
            )
            assert patterns_after == 0

    def test_delete_analysis_all_flag(self, cli_runner, db_with_analysis):
        """Test deleting all analysis results."""
        result = cli_runner.invoke(
            cli,
            ["delete-analysis", "--db", str(db_with_analysis), "--all", "--force"],
        )

        assert result.exit_code == 0
        assert "Successfully deleted 5 analysis record(s)" in result.output
        assert "5 session(s)" in result.output

        with session_scope() as session:
            total_analysis = session.query(models.AnalysisResult).count()
            assert total_analysis == 4


class TestAnalyzeCommand:
    """Test consolidated analyze command."""

    def test_analyze_missing_selection_flag(self, cli_runner, temp_db):
        """Test that analyze run requires at least one selection flag."""
        init_database(str(temp_db))

        result = cli_runner.invoke(
            cli,
            ["analyze", "run", "--db", str(temp_db), "--profile", "testuser"],
        )

        assert result.exit_code == 1
        assert "Must provide at least one selection flag" in result.output

    def test_analyze_mutually_exclusive_single_flags(self, cli_runner, temp_db):
        """Test that --session-id and --date are mutually exclusive."""
        init_database(str(temp_db))

        result = cli_runner.invoke(
            cli,
            [
                "analyze",
                "run",
                "--db",
                str(temp_db),
                "--profile",
                "testuser",
                "--session-id",
                "1",
                "--date",
                "2025-01-01",
            ],
        )

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_analyze_mutually_exclusive_single_and_batch(self, cli_runner, temp_db):
        """Test that single session flags cannot be used with batch flags."""
        init_database(str(temp_db))

        result = cli_runner.invoke(
            cli,
            [
                "analyze",
                "run",
                "--db",
                str(temp_db),
                "--profile",
                "testuser",
                "--session-id",
                "1",
                "--start",
                "2025-01-01",
            ],
        )

        assert result.exit_code == 1
        assert "cannot be used with batch flags" in result.output

    def test_analyze_profile_not_found(self, cli_runner, temp_db):
        """Test error when profile doesn't exist."""
        init_database(str(temp_db))

        result = cli_runner.invoke(
            cli,
            [
                "analyze",
                "list",
                "--db",
                str(temp_db),
                "--profile",
                "nonexistent",
            ],
        )

        assert result.exit_code == 1
        assert "Profile 'nonexistent' not found" in result.output

    def test_analyze_list_mode(self, cli_runner, db_with_analysis):
        """Test list subcommand shows analysis status."""
        result = cli_runner.invoke(
            cli,
            [
                "analyze",
                "list",
                "--db",
                str(db_with_analysis),
                "--profile",
                "testuser",
            ],
        )

        assert result.exit_code == 0
        assert "Date" in result.output
        assert "Analyzed" in result.output

    def test_analyze_list_with_date_range(self, cli_runner, db_with_analysis):
        """Test list subcommand with date range filtering."""
        result = cli_runner.invoke(
            cli,
            [
                "analyze",
                "list",
                "--db",
                str(db_with_analysis),
                "--profile",
                "testuser",
                "--start",
                "2025-10-01",
                "--end",
                "2025-10-03",
            ],
        )

        assert result.exit_code == 0

    def test_analyze_no_subcommand_shows_help(self, cli_runner, temp_db):
        """Test that running 'analyze' without subcommand shows help."""
        init_database(str(temp_db))

        result = cli_runner.invoke(cli, ["analyze"])

        assert result.exit_code in [0, 2]
        assert "Commands:" in result.output or "show" in result.output

    def test_analyze_show_by_session_id(self, cli_runner, db_with_analysis):
        """Test show subcommand displays stored analysis by session ID."""
        result = cli_runner.invoke(
            cli,
            [
                "analyze",
                "show",
                "--db",
                str(db_with_analysis),
                "--session-id",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "Displaying stored analysis" in result.output
        assert "ANALYSIS SUMMARY" in result.output

    def test_analyze_show_by_date(self, cli_runner, db_with_analysis):
        """Test show subcommand displays stored analysis by date."""
        result = cli_runner.invoke(
            cli,
            [
                "analyze",
                "show",
                "--db",
                str(db_with_analysis),
                "--profile",
                "testuser",
                "--date",
                "2025-10-01",
            ],
        )

        assert result.exit_code == 0
        assert "Displaying stored analysis" in result.output
        assert "ANALYSIS SUMMARY" in result.output

    def test_analyze_show_no_analysis_found(self, cli_runner, temp_db):
        """Test show subcommand gracefully handles missing analysis."""
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
                model="Test",
                serial_number="TEST",
            )
            session.add(device)
            session.flush()

            sess = models.Session(
                device_id=device.id,
                device_session_id="test_session_1",
                start_time=datetime(2025, 10, 1, 22, 0, 0),
                end_time=datetime(2025, 10, 2, 6, 0, 0),
                duration_seconds=8 * 3600,
            )
            session.add(sess)
            session.commit()

        result = cli_runner.invoke(
            cli,
            [
                "analyze",
                "show",
                "--db",
                str(temp_db),
                "--session-id",
                "1",
            ],
        )

        assert result.exit_code == 1
        assert "No analysis found" in result.output
