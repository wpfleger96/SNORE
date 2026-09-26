"""
Tests for CLI commands.

These tests verify the command-line interface functionality including:
- session delete command with various input modes
- profile list command output
- db stats command
- session list command with limits and truncation
"""

import asyncio

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from snore.cli import cli
from snore.database import models
from snore.database.day_manager import DayManager
from snore.database.session import init_database, session_scope


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


async def _create_test_user_and_profile(
    session: AsyncSession, email: str = "cli_test@example.com"
) -> models.Profile:
    """Helper: create a User+Profile in a session and return the Profile.

    Required so Device() rows satisfy the NOT NULL profile_id constraint.
    """
    import uuid

    _user = models.User(canonical_email=f"{uuid.uuid4().hex[:8]}_{email}", role="admin")
    session.add(_user)
    await session.flush()
    _profile = models.Profile(user_id=_user.id, name="Test Profile")
    session.add(_profile)
    await session.flush()
    return _profile


@pytest.fixture
async def populated_test_db(temp_db):
    """Create a database populated with realistic test data."""
    await init_database(str(temp_db))

    async with session_scope() as session:
        _profile = await _create_test_user_and_profile(session)

        device = models.Device(
            profile_id=_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
            serial_number="TEST12345",
        )
        session.add(device)
        await session.flush()

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
            await session.flush()

            day_date = DayManager.get_day_for_session(start_time)
            day = await DayManager.create_or_update_day(device.id, day_date, session)
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

    return temp_db


@pytest.fixture
async def populated_test_db_full(temp_db):
    """Create a database populated with full Statistics and Waveform records."""
    import numpy as np

    await init_database(str(temp_db))

    async with session_scope() as session:
        _profile = await _create_test_user_and_profile(session)

        device = models.Device(
            profile_id=_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
            serial_number="TEST12345",
        )
        session.add(device)
        await session.flush()

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
                has_waveform_data=True,
            )
            session.add(sess)
            await session.flush()

            day_date = DayManager.get_day_for_session(start_time)
            day = await DayManager.create_or_update_day(device.id, day_date, session)
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

            session.add(
                models.Statistics(
                    session_id=sess.id,
                    ahi=5.2 + i * 0.3,
                    usage_hours=7.8,
                    rei=4.5,
                    obstructive_apneas=3,
                    central_apneas=1,
                    hypopneas=5,
                    pressure_mean=10.5,
                    pressure_min=8.0,
                    pressure_max=13.0,
                    pressure_95th=12.5,
                    leak_mean=12.0,
                    leak_percentile_70=15.0,
                    leak_95th=22.0,
                    spo2_mean=95.5,
                    spo2_min=88.0,
                    spo2_time_below_90=120,
                    pulse_mean=72.0,
                    pulse_min=55.0,
                    pulse_max=95.0,
                    respiratory_rate_mean=15.0,
                    tidal_volume_mean=450.0,
                    minute_ventilation_mean=6.8,
                )
            )

            dummy_blob = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32).tobytes()
            for wtype in ["flow", "pressure", "leak"]:
                session.add(
                    models.Waveform(
                        session_id=sess.id,
                        waveform_type=wtype,
                        sample_rate=25.0 if wtype == "flow" else 0.5,
                        unit={"flow": "L/min", "pressure": "cmH2O", "leak": "L/min"}[
                            wtype
                        ],
                        data_blob=dummy_blob,
                        sample_count=2,
                    )
                )

    return temp_db


class TestSessionDeleteCommand:
    """Test session delete command with various scenarios."""

    def test_delete_single_session_by_id(
        self, cli_runner, populated_test_db, db_session
    ):
        """Test deleting a single session by ID."""
        result = cli_runner.invoke(
            cli,
            ["session", "delete", "--db", str(populated_test_db), "--session-id", "1"],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Successfully deleted 1 session(s)" in result.output

        remaining = db_session.query(models.Session).filter_by(id=1).count()
        assert remaining == 0

    def test_delete_multiple_sessions_by_id(
        self, cli_runner, populated_test_db, db_session
    ):
        """Test deleting multiple sessions by ID (tests SQL IN clause fix)."""
        result = cli_runner.invoke(
            cli,
            [
                "session",
                "delete",
                "--db",
                str(populated_test_db),
                "--session-id",
                "1,2,3",
            ],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Successfully deleted 3 session(s)" in result.output

        remaining = (
            db_session.query(models.Session)
            .filter(models.Session.id.in_([1, 2, 3]))
            .count()
        )
        assert remaining == 0

    def test_delete_sessions_by_date_range(
        self, cli_runner, populated_test_db, db_session
    ):
        """Test deleting sessions by date range."""
        result = cli_runner.invoke(
            cli,
            [
                "session",
                "delete",
                "--db",
                str(populated_test_db),
                "--from",
                "2025-10-01",
                "--to",
                "2025-10-03",
            ],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "sessions" in result.output.lower()

        total_remaining = db_session.query(models.Session).count()
        assert total_remaining < 10

    def test_delete_cascades_to_child_tables(
        self, cli_runner, populated_test_db, db_session
    ):
        """Test that deleting sessions cascades to events, waveforms, statistics."""
        events_before = db_session.execute(
            text("SELECT COUNT(*) FROM events WHERE session_id = 1")
        ).scalar()
        stats_before = db_session.execute(
            text("SELECT COUNT(*) FROM statistics WHERE session_id = 1")
        ).scalar()

        assert events_before > 0
        assert stats_before > 0

        result = cli_runner.invoke(
            cli,
            ["session", "delete", "--db", str(populated_test_db), "--session-id", "1"],
            input="y\n",
        )

        assert result.exit_code == 0

        db_session.expire_all()
        events_after = db_session.execute(
            text("SELECT COUNT(*) FROM events WHERE session_id = 1")
        ).scalar()
        stats_after = db_session.execute(
            text("SELECT COUNT(*) FROM statistics WHERE session_id = 1")
        ).scalar()

        assert events_after == 0
        assert stats_after == 0

    def test_delete_session_datetime_formatting(self, cli_runner, populated_test_db):
        """Test that datetime formatting works correctly in delete preview."""
        result = cli_runner.invoke(
            cli,
            ["session", "delete", "--db", str(populated_test_db), "--session-id", "1"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "2025-10-" in result.output
        assert "Deletion cancelled" in result.output


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
        asyncio.run(init_database(str(temp_db)))

        result = cli_runner.invoke(cli, ["db", "stats", "--db", str(temp_db)])

        assert result.exit_code == 0
        assert "Devices: 0" in result.output
        assert "Sessions: 0" in result.output


class TestDbRecomputeDaysCommand:
    """Test db recompute-days command."""

    def test_recompute_days_rederives_ahi(
        self, cli_runner, populated_test_db, db_session
    ):
        """recompute-days re-derives Day.ahi from stored session statistics.

        The fixture adds each session's Statistics AFTER day aggregation runs, so
        every Day.ahi starts NULL; recompute-days must populate it from the now-
        present Statistics without re-importing raw data.
        """
        ahi_before = db_session.execute(
            text("SELECT ahi FROM days WHERE ahi IS NOT NULL")
        ).all()
        assert ahi_before == []

        result = cli_runner.invoke(
            cli,
            ["db", "recompute-days", "--db", str(populated_test_db)],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Recomputed 10 day(s)" in result.output

        db_session.expire_all()
        ahi_values = db_session.execute(text("SELECT ahi FROM days")).scalars().all()
        assert len(ahi_values) == 10
        assert all(v == pytest.approx(5.2) for v in ahi_values)

    def test_recompute_days_prunes_orphans_and_keeps_disabled_only_days(
        self, cli_runner, populated_test_db, db_session
    ):
        """Under the production engine (FK enforcement on), recompute-days
        deletes a Day whose sessions are gone and keeps a Day whose only
        session is disabled, reporting the pruned count."""
        orphan_day_id, disabled_day_id = (
            db_session.execute(
                text(
                    "SELECT day_id FROM sessions WHERE device_session_id IN "
                    "('test_session_0', 'test_session_1') ORDER BY device_session_id"
                )
            )
            .scalars()
            .all()
        )
        db_session.execute(
            text("DELETE FROM sessions WHERE device_session_id = 'test_session_0'")
        )
        db_session.execute(
            text(
                "UPDATE sessions SET enabled = 0 "
                "WHERE device_session_id = 'test_session_1'"
            )
        )
        db_session.commit()

        result = cli_runner.invoke(
            cli,
            ["db", "recompute-days", "--db", str(populated_test_db)],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Recomputed 9 day(s)" in result.output
        assert "pruned 1 orphaned day(s)" in result.output

        db_session.expire_all()
        remaining = db_session.execute(text("SELECT id FROM days")).scalars().all()
        assert orphan_day_id not in remaining
        assert len(remaining) == 9
        disabled_count = db_session.execute(
            text("SELECT session_count FROM days WHERE id = :id"),
            {"id": disabled_day_id},
        ).scalar_one()
        assert disabled_count == 0
        assert (
            db_session.execute(
                text("SELECT COUNT(*) FROM sessions WHERE day_id = :id"),
                {"id": disabled_day_id},
            ).scalar_one()
            == 1
        )

    def test_recompute_days_empty_database(self, cli_runner, temp_db):
        """recompute-days on an empty database reports zero days."""
        asyncio.run(init_database(str(temp_db)))

        result = cli_runner.invoke(
            cli,
            ["db", "recompute-days", "--db", str(temp_db)],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Recomputed 0 day(s)" in result.output


class TestSessionListCommand:
    """Test session list command."""

    def test_list_sessions_default_limit(self, cli_runner, populated_test_db):
        """Test session list uses default limit of 20."""
        result = cli_runner.invoke(
            cli, ["session", "list", "--db", str(populated_test_db)]
        )

        assert result.exit_code == 0

        session_rows = [
            line
            for line in result.output.split("\n")
            if "2025-10-" in line and "TEST12345" in line
        ]
        assert len(session_rows) == 10

    def test_list_sessions_custom_limit(self, cli_runner, populated_test_db):
        """Test session list with custom limit."""
        result = cli_runner.invoke(
            cli, ["session", "list", "--db", str(populated_test_db), "--limit", "5"]
        )

        assert result.exit_code == 0

        session_rows = [
            line
            for line in result.output.split("\n")
            if "2025-10-" in line and "TEST12345" in line
        ]
        assert len(session_rows) == 5
        assert "Showing 5 of 10 sessions" in result.output
        assert "Tip:" in result.output

    def test_list_sessions_unlimited(self, cli_runner, populated_test_db):
        """Test session list with --limit 0 shows all sessions."""
        result = cli_runner.invoke(
            cli, ["session", "list", "--db", str(populated_test_db), "--limit", "0"]
        )

        assert result.exit_code == 0
        session_rows = [
            line
            for line in result.output.split("\n")
            if "2025-10-" in line and "TEST12345" in line
        ]
        assert len(session_rows) == 10

    def test_list_sessions_no_truncation_message(self, cli_runner, temp_db):
        """Test session list doesn't show truncation when all results fit."""
        import asyncio

        asyncio.run(init_database(str(temp_db)))

        async def _setup() -> None:
            async with session_scope() as session:
                _profile = await _create_test_user_and_profile(session)
                device = models.Device(
                    profile_id=_profile.id,
                    manufacturer="Test",
                    model="Test",
                    serial_number="TEST",
                )
                session.add(device)
                await session.flush()

                start_time = datetime(2025, 10, 1, 22, 0, 0)
                sess = models.Session(
                    device_id=device.id,
                    device_session_id="test_session_1",
                    start_time=start_time,
                    end_time=start_time + timedelta(hours=8),
                    duration_seconds=8 * 3600,
                )
                session.add(sess)

        asyncio.run(_setup())

        result = cli_runner.invoke(cli, ["session", "list", "--db", str(temp_db)])

        assert result.exit_code == 0
        assert "Showing" not in result.output or "Showing all" in result.output


@pytest.fixture
async def db_with_analysis(temp_db):
    """Create a database populated with sessions and analysis results."""
    await init_database(str(temp_db))

    async with session_scope() as session:
        _profile = await _create_test_user_and_profile(session)
        device = models.Device(
            profile_id=_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
            serial_number="TEST12345",
        )
        session.add(device)
        await session.flush()

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
            await session.flush()

            day_date = DayManager.get_day_for_session(start_time)
            day = await DayManager.create_or_update_day(device.id, day_date, session)
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
                    created_at=datetime.now(UTC) + timedelta(minutes=j),
                )
                session.add(analysis)
                await session.flush()

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

    return temp_db


class TestAnalysisDeleteCommand:
    """Test analysis delete command with various scenarios."""

    def test_delete_analysis_single_session(
        self, cli_runner, db_with_analysis, db_session
    ):
        """Test deleting analysis for a single session (latest only)."""
        analysis_before = (
            db_session.query(models.AnalysisResult).filter_by(session_id=1).count()
        )
        assert analysis_before == 3

        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "delete",
                "--db",
                str(db_with_analysis),
                "--session-id",
                "1",
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully deleted 1 analysis record(s)" in result.output

        db_session.expire_all()
        analysis_after = (
            db_session.query(models.AnalysisResult).filter_by(session_id=1).count()
        )
        assert analysis_after == 2

        sess = db_session.query(models.Session).filter_by(id=1).first()
        assert sess is not None

    def test_delete_analysis_all_versions(
        self, cli_runner, db_with_analysis, db_session
    ):
        """Test deleting all analysis versions for a session."""
        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "delete",
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

        db_session.expire_all()
        analysis_after = (
            db_session.query(models.AnalysisResult).filter_by(session_id=1).count()
        )
        assert analysis_after == 0

        sess = db_session.query(models.Session).filter_by(id=1).first()
        assert sess is not None

    def test_delete_analysis_multiple_sessions(
        self, cli_runner, db_with_analysis, db_session
    ):
        """Test deleting analysis for multiple sessions."""
        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "delete",
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

        db_session.expire_all()
        analysis_1 = (
            db_session.query(models.AnalysisResult).filter_by(session_id=1).count()
        )
        analysis_2 = (
            db_session.query(models.AnalysisResult).filter_by(session_id=2).count()
        )
        analysis_3 = (
            db_session.query(models.AnalysisResult).filter_by(session_id=3).count()
        )

        assert analysis_1 == 2
        assert analysis_2 == 2
        assert analysis_3 == 0

    def test_delete_analysis_date_range(self, cli_runner, db_with_analysis, db_session):
        """Test deleting analysis by date range."""
        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "delete",
                "--db",
                str(db_with_analysis),
                "--from",
                "2025-10-01",
                "--to",
                "2025-10-03",
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully deleted" in result.output

        db_session.expire_all()
        total_analysis = db_session.query(models.AnalysisResult).count()
        assert total_analysis == 7

    def test_delete_analysis_dry_run(self, cli_runner, db_with_analysis, db_session):
        """Test dry-run mode doesn't actually delete."""
        analysis_before = db_session.query(models.AnalysisResult).count()

        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "delete",
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

        db_session.expire_all()
        analysis_after = db_session.query(models.AnalysisResult).count()
        assert analysis_after == analysis_before

    def test_delete_analysis_cancellation(
        self, cli_runner, db_with_analysis, db_session
    ):
        """Test that user can cancel deletion."""
        result = cli_runner.invoke(
            cli,
            ["analysis", "delete", "--db", str(db_with_analysis), "--session-id", "1"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "Deletion cancelled" in result.output

        analysis = (
            db_session.query(models.AnalysisResult).filter_by(session_id=1).count()
        )
        assert analysis == 3

    def test_delete_analysis_no_filter_error(self, cli_runner, db_with_analysis):
        """Test that command errors when no filter is provided."""
        result = cli_runner.invoke(
            cli, ["analysis", "delete", "--db", str(db_with_analysis)]
        )

        assert result.exit_code == 1
        assert "must specify at least one filter" in result.output

    def test_delete_analysis_no_sessions_found(self, cli_runner, temp_db):
        """Test graceful handling when no sessions have analysis."""
        import asyncio

        asyncio.run(init_database(str(temp_db)))

        async def _setup() -> None:
            async with session_scope() as session:
                _profile = await _create_test_user_and_profile(session)
                device = models.Device(
                    profile_id=_profile.id,
                    manufacturer="Test",
                    model="Test",
                    serial_number="TEST",
                )
                session.add(device)
                await session.flush()

                sess = models.Session(
                    device_id=device.id,
                    device_session_id="test_session_1",
                    start_time=datetime(2025, 10, 1, 22, 0, 0),
                    end_time=datetime(2025, 10, 2, 6, 0, 0),
                    duration_seconds=8 * 3600,
                )
                session.add(sess)

        asyncio.run(_setup())

        result = cli_runner.invoke(
            cli,
            ["analysis", "delete", "--db", str(temp_db), "--session-id", "1"],
        )

        assert result.exit_code == 0
        assert "No sessions with analysis results found" in result.output

    def test_delete_analysis_cascades_to_patterns(
        self, cli_runner, db_with_analysis, db_session
    ):
        """Test that deleting analysis cascades to detected patterns."""
        analysis = (
            db_session.query(models.AnalysisResult)
            .filter_by(session_id=1)
            .order_by(models.AnalysisResult.created_at.desc())
            .first()
        )
        latest_analysis_id = analysis.id

        patterns_before = (
            db_session.query(models.DetectedPattern)
            .filter_by(analysis_result_id=latest_analysis_id)
            .count()
        )
        assert patterns_before > 0

        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "delete",
                "--db",
                str(db_with_analysis),
                "--session-id",
                "1",
                "--force",
            ],
        )

        assert result.exit_code == 0

        db_session.expire_all()
        patterns_after = (
            db_session.query(models.DetectedPattern)
            .filter_by(analysis_result_id=latest_analysis_id)
            .count()
        )
        assert patterns_after == 0

    def test_delete_analysis_all_flag(self, cli_runner, db_with_analysis, db_session):
        """Test deleting all analysis results."""
        result = cli_runner.invoke(
            cli,
            ["analysis", "delete", "--db", str(db_with_analysis), "--all", "--force"],
        )

        assert result.exit_code == 0
        assert "Successfully deleted 5 analysis record(s)" in result.output
        assert "5 session(s)" in result.output

        db_session.expire_all()
        total_analysis = db_session.query(models.AnalysisResult).count()
        assert total_analysis == 4


class TestAnalysisCommand:
    """Test consolidated analysis command."""

    def test_analyze_missing_selection_flag(self, cli_runner, temp_db):
        """Test that analysis run requires at least one selection flag."""
        asyncio.run(init_database(str(temp_db)))

        result = cli_runner.invoke(
            cli,
            ["analysis", "run", "--db", str(temp_db)],
        )

        assert result.exit_code == 1
        assert "Must provide at least one selection flag" in result.output

    def test_analyze_mutually_exclusive_single_flags(self, cli_runner, temp_db):
        """Test that --session-id and --date are mutually exclusive."""
        asyncio.run(init_database(str(temp_db)))

        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "run",
                "--db",
                str(temp_db),
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
        asyncio.run(init_database(str(temp_db)))

        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "run",
                "--db",
                str(temp_db),
                "--session-id",
                "1",
                "--from",
                "2025-01-01",
            ],
        )

        assert result.exit_code == 1
        assert "cannot be used with batch flags" in result.output

    def test_analyze_list_mode(self, cli_runner, db_with_analysis):
        """Test list subcommand shows analysis status."""
        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "list",
                "--db",
                str(db_with_analysis),
            ],
        )

        assert result.exit_code == 0
        assert "Date" in result.output
        assert "Analyzed" in result.output

    def test_analyze_show_by_session_id(self, cli_runner, db_with_analysis):
        """Test show subcommand displays stored analysis by session ID."""
        result = cli_runner.invoke(
            cli,
            [
                "analysis",
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
                "analysis",
                "show",
                "--db",
                str(db_with_analysis),
                "--date",
                "2025-10-01",
            ],
        )

        assert result.exit_code == 0
        assert "Displaying stored analysis" in result.output
        assert "ANALYSIS SUMMARY" in result.output

    def test_analyze_show_no_analysis_found(self, cli_runner, temp_db):
        """Test show subcommand gracefully handles missing analysis."""
        import asyncio

        asyncio.run(init_database(str(temp_db)))

        async def _setup() -> None:
            async with session_scope() as session:
                _profile = await _create_test_user_and_profile(session)
                device = models.Device(
                    profile_id=_profile.id,
                    manufacturer="Test",
                    model="Test",
                    serial_number="TEST",
                )
                session.add(device)
                await session.flush()

                sess = models.Session(
                    device_id=device.id,
                    device_session_id="test_session_1",
                    start_time=datetime(2025, 10, 1, 22, 0, 0),
                    end_time=datetime(2025, 10, 2, 6, 0, 0),
                    duration_seconds=8 * 3600,
                )
                session.add(sess)

        asyncio.run(_setup())

        result = cli_runner.invoke(
            cli,
            [
                "analysis",
                "show",
                "--db",
                str(temp_db),
                "--session-id",
                "1",
            ],
        )

        assert result.exit_code == 1
        assert "No analysis found" in result.output


class TestDbDropCommand:
    """Test db drop command - focused on critical behavior."""

    def test_db_drop_deletes_database(self, cli_runner, populated_test_db):
        """Test that drop command actually deletes the database and associated files."""
        assert populated_test_db.exists()

        result = cli_runner.invoke(
            cli,
            ["db", "drop", "--db", str(populated_test_db)],
            input="y\n",
        )

        assert result.exit_code == 0

        assert not populated_test_db.exists()
        assert not Path(str(populated_test_db) + "-wal").exists()
        assert not Path(str(populated_test_db) + "-shm").exists()

    def test_db_drop_force_flag(self, cli_runner, populated_test_db):
        """Test that --force flag skips confirmation."""
        result = cli_runner.invoke(
            cli,
            ["db", "drop", "--db", str(populated_test_db), "--force"],
        )

        assert result.exit_code == 0
        assert not populated_test_db.exists()


class TestDbInitCommand:
    """Test db init command - focused on critical behavior."""

    def test_db_init_creates_database(self, cli_runner, temp_db):
        """Test that init creates a functional database."""
        result = cli_runner.invoke(
            cli,
            ["db", "init", "--db", str(temp_db)],
        )

        assert result.exit_code == 0
        assert temp_db.exists()

        result2 = cli_runner.invoke(
            cli,
            ["db", "init", "--db", str(temp_db)],
        )

        assert result2.exit_code == 0
        assert temp_db.exists()


class TestSessionShowCommand:
    """Tests for session show command."""

    @pytest.mark.integration
    def test_session_show_settings_flag(
        self, temp_db, resmed_parser, resmed_fixture_path
    ):
        """Test --settings flag displays settings."""
        import asyncio
        import uuid

        from snore.database.importers import import_session

        asyncio.run(init_database(str(temp_db)))

        # Create a User+Profile for the device to satisfy the NOT NULL FK constraint.
        async def _setup_profile() -> int:
            async with session_scope() as db:
                _u = models.User(
                    canonical_email=f"sshow_{uuid.uuid4().hex[:8]}@example.com",
                    role="admin",
                )
                db.add(_u)
                await db.flush()
                _p = models.Profile(user_id=_u.id, name="Test Profile")
                db.add(_p)
                await db.flush()
                return _p.id

        profile_id = asyncio.run(_setup_profile())

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path, limit=1))
        asyncio.run(import_session(sessions[0], profile_id=profile_id))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["session", "show", "1", "--settings", "--db", str(temp_db)]
        )

        assert result.exit_code == 0
        assert "Settings:" in result.output
        assert "mode:" in result.output


class TestWaveformListCommand:
    """Test waveform list command."""

    def test_waveform_list_shows_types(self, cli_runner, populated_test_db_full):
        """Test waveform list displays available types."""
        result = cli_runner.invoke(
            cli,
            [
                "waveform",
                "list",
                "--db",
                str(populated_test_db_full),
                "--session-id",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "flow" in result.output
        assert "pressure" in result.output
        assert "leak" in result.output

    def test_waveform_list_no_session(self, cli_runner, populated_test_db_full):
        """Test waveform list errors when session not specified."""
        result = cli_runner.invoke(
            cli,
            ["waveform", "list", "--db", str(populated_test_db_full)],
        )

        assert result.exit_code != 0


class TestSessionShowExpanded:
    """Test expanded session show output with full statistics."""

    def test_session_show_displays_full_stats(self, cli_runner, populated_test_db_full):
        """Test session show displays comprehensive statistics."""
        result = cli_runner.invoke(
            cli,
            ["session", "show", "1", "--db", str(populated_test_db_full)],
        )

        assert result.exit_code == 0
        assert "AHI:" in result.output
        assert "REI:" in result.output
        assert "Pressure:" in result.output
        assert "Leak:" in result.output
        assert "SpO₂:" in result.output or "SpO2:" in result.output
        assert "Pulse:" in result.output
        assert "Respiratory:" in result.output or "Respiratory Rate:" in result.output

    def test_session_show_displays_waveform_types(
        self, cli_runner, populated_test_db_full
    ):
        """Test session show lists available waveform types."""
        result = cli_runner.invoke(
            cli,
            ["session", "show", "1", "--db", str(populated_test_db_full)],
        )

        assert result.exit_code == 0
        assert "flow" in result.output
        assert "pressure" in result.output
        assert "leak" in result.output


class TestStatsEnhanced:
    """Test enhanced stats command with respiratory, pulse, and SpO2 detail."""

    def test_stats_shows_respiratory(self, cli_runner, populated_test_db_full):
        """Test stats displays respiratory metrics."""
        result = cli_runner.invoke(
            cli,
            ["stats", "--db", str(populated_test_db_full)],
        )

        assert result.exit_code == 0
        assert "Respiratory Rate:" in result.output or "Respiratory" in result.output
        assert "Tidal Volume:" in result.output or "Tidal" in result.output
        assert "Minute Ventilation:" in result.output or "Ventilation" in result.output

    def test_stats_shows_pulse(self, cli_runner, populated_test_db_full):
        """Test stats displays pulse section."""
        result = cli_runner.invoke(
            cli,
            ["stats", "--db", str(populated_test_db_full)],
        )

        assert result.exit_code == 0
        assert "Pulse" in result.output

    def test_stats_shows_spo2_below_90(self, cli_runner, populated_test_db_full):
        """Test stats displays time below 90% SpO2."""
        result = cli_runner.invoke(
            cli,
            ["stats", "--db", str(populated_test_db_full)],
        )

        assert result.exit_code == 0
        assert "Time below 90%:" in result.output or "below 90" in result.output


class TestStatsPeriod:
    """Test stats command with period breakdown."""

    def test_stats_period_month(self, cli_runner, populated_test_db_full):
        """Test stats with monthly period breakdown."""
        result = cli_runner.invoke(
            cli,
            ["stats", "--db", str(populated_test_db_full), "--period", "month"],
        )

        assert result.exit_code == 0
        assert "Oct 2025" in result.output or "2025-10" in result.output


@pytest.fixture
async def db_with_rx_settings_changes(temp_db):
    """DB with three days showing two distinct settings-change dates.

    Day 1 (2025-06-01): mode=CPAP, pressure_fixed=8.0  (baseline, no diff)
    Day 2 (2025-06-02): mode=APAP, pressure_min=6.0    (change date A)
    Day 3 (2025-06-03): mode=APAP, pressure_min=8.0    (change date B, most recent)
    """
    await init_database(str(temp_db))

    async with session_scope() as session:
        _profile = await _create_test_user_and_profile(session)
        device = models.Device(
            profile_id=_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
            serial_number="RX_CHG_TEST",
        )
        session.add(device)
        await session.flush()

        days_settings = [
            (datetime(2025, 6, 1, 22, 0, 0), {"mode": "CPAP", "pressure_fixed": "8.0"}),
            (datetime(2025, 6, 2, 22, 0, 0), {"mode": "APAP", "pressure_min": "6.0"}),
            (datetime(2025, 6, 3, 22, 0, 0), {"mode": "APAP", "pressure_min": "8.0"}),
        ]

        for start_time, settings in days_settings:
            sess = models.Session(
                device_id=device.id,
                device_session_id=f"rx_chg_{start_time.date().isoformat()}",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
                enabled=True,
            )
            session.add(sess)
            await session.flush()

            day = await DayManager.create_or_update_day(
                device.id, DayManager.get_day_for_session(start_time), session
            )
            sess.day_id = day.id

            for key, value in settings.items():
                session.add(models.Setting(session_id=sess.id, key=key, value=value))

    return temp_db


class TestRxChangesCommand:
    """Test rx changes command."""

    def test_changes_renders_most_recent_first_with_old_arrow_new(
        self, cli_runner, db_with_rx_settings_changes
    ):
        """Changes are shown most-recent-first; each row uses 'old → new' format."""
        result = cli_runner.invoke(
            cli, ["rx", "changes", "--db", str(db_with_rx_settings_changes)]
        )

        assert result.exit_code == 0
        assert "RX Settings Changes" in result.output
        assert "→" in result.output

        # Both change dates must appear
        assert "2025-06-03" in result.output
        assert "2025-06-02" in result.output

        # Most-recent-first: 2025-06-03 row must appear before 2025-06-02 row
        idx_recent = result.output.index("2025-06-03")
        idx_older = result.output.index("2025-06-02")
        assert idx_recent < idx_older

        # pressure_min change on day 3 — key rendered as label, values formatted
        assert "Min Pressure" in result.output
        assert "6.0 cmH2O → 8.0 cmH2O" in result.output

        # mode change on day 2 — key rendered as "Mode", values pass through
        assert "Mode" in result.output
        assert "CPAP → APAP" in result.output

    def test_changes_none_value_rendered_as_dash(
        self, cli_runner, db_with_rx_settings_changes
    ):
        """A key absent in the previous day's settings renders old_value as '—'."""
        result = cli_runner.invoke(
            cli, ["rx", "changes", "--db", str(db_with_rx_settings_changes)]
        )

        assert result.exit_code == 0
        # pressure_fixed disappears on day 2 (old=8.0 cmH2O, new=None) → "8.0 cmH2O → —"
        assert "8.0 cmH2O → —" in result.output
        # pressure_min appears on day 2 (old=None, new=6.0 cmH2O) → "— → 6.0 cmH2O"
        assert "— → 6.0 cmH2O" in result.output

    def test_changes_within_date_order_is_ascending_by_key(
        self, cli_runner, db_with_rx_settings_changes
    ):
        """Same-date rows appear in ascending key order (service's within-date sort)."""
        result = cli_runner.invoke(
            cli, ["rx", "changes", "--db", str(db_with_rx_settings_changes)]
        )

        assert result.exit_code == 0
        # Day 2 (2025-06-02) produces three changes with keys: mode, pressure_fixed,
        # pressure_min.  Ascending alphabetical order means mode < pressure_min, so
        # the unique text for the mode row must appear before the pressure_min row.
        idx_mode = result.output.index("CPAP → APAP")
        idx_pressure_min = result.output.index("— → 6.0 cmH2O")
        assert idx_mode < idx_pressure_min

    def test_changes_empty_db_shows_no_changes_message(self, cli_runner, temp_db):
        """Empty database produces a friendly no-changes message, not an error."""
        asyncio.run(init_database(str(temp_db)))

        result = cli_runner.invoke(cli, ["rx", "changes", "--db", str(temp_db)])

        assert result.exit_code == 0
        assert "No RX settings changes found" in result.output


# ---------------------------------------------------------------------------
# Fixtures for rx history / current / compare tests
# ---------------------------------------------------------------------------


async def _make_rx_session(
    session: AsyncSession,
    device_id: int,
    start_time: datetime,
    settings: dict[str, str],
    *,
    ahi: float = 3.0,
    hours: float = 7.5,
    leak: float = 8.0,
    serial_suffix: str = "",
) -> None:
    """Helper: create one session + day + statistics for rx tests."""
    sess = models.Session(
        device_id=device_id,
        device_session_id=f"rx_{start_time.date().isoformat()}{serial_suffix}",
        start_time=start_time,
        end_time=start_time + timedelta(hours=hours),
        duration_seconds=int(hours * 3600),
        enabled=True,
    )
    session.add(sess)
    await session.flush()

    day = await DayManager.create_or_update_day(
        device_id, DayManager.get_day_for_session(start_time), session
    )
    sess.day_id = day.id

    for key, value in settings.items():
        session.add(models.Setting(session_id=sess.id, key=key, value=value))

    session.add(
        models.Statistics(
            session_id=sess.id, ahi=ahi, usage_hours=hours, leak_mean=leak
        )
    )


@pytest.fixture
async def db_with_apap_rx(temp_db):
    """DB with a single APAP period: pressure range and EPR settings present."""
    await init_database(str(temp_db))

    async with session_scope() as session:
        _profile = await _create_test_user_and_profile(session)
        device = models.Device(
            profile_id=_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
            serial_number="APAP_RX_TEST",
        )
        session.add(device)
        await session.flush()

        apap_settings = {
            "mode": "APAP",
            "pressure_min": "6.0",
            "pressure_max": "20.0",
            "epr_level": "2",
            "epr_mode": "Full Time",
        }
        for i in range(3):
            await _make_rx_session(
                session,
                device.id,
                datetime(2025, 7, 1 + i, 22, 0, 0),
                apap_settings,
                serial_suffix=f"_{i}",
            )

    return temp_db


@pytest.fixture
async def db_with_bipap_rx(temp_db):
    """DB with two RX periods: CPAP then BiPAP Auto.

    Period 1 (day 1): CPAP, pressure_fixed=10.0
    Period 2 (days 2-4): BiPAP Auto, epap=6.0, ipap=18.0, ps=4.0, no epr keys
    """
    await init_database(str(temp_db))

    async with session_scope() as session:
        _profile = await _create_test_user_and_profile(session)
        device = models.Device(
            profile_id=_profile.id,
            manufacturer="ResMed",
            model="AirCurve 10 VAuto",
            serial_number="BIPAP_RX_TEST",
        )
        session.add(device)
        await session.flush()

        cpap_settings = {"mode": "CPAP", "pressure_fixed": "10.0"}
        await _make_rx_session(
            session,
            device.id,
            datetime(2025, 7, 1, 22, 0, 0),
            cpap_settings,
            serial_suffix="_cpap",
        )

        bipap_settings = {
            "mode": "BiPAP Auto",
            "epap": "6.0",
            "ipap": "18.0",
            "ps": "4.0",
        }
        for i in range(3):
            await _make_rx_session(
                session,
                device.id,
                datetime(2025, 7, 2 + i, 22, 0, 0),
                bipap_settings,
                serial_suffix=f"_bipap_{i}",
            )

    return temp_db


class TestRxHistoryCommand:
    """Test rx history command with pressure formatting and EPR/PS rendering."""

    def test_history_apap_shows_pressure_range_with_units(
        self, cli_runner, db_with_apap_rx
    ):
        """APAP period renders pressure range with cmH2O units."""
        result = cli_runner.invoke(cli, ["rx", "history", "--db", str(db_with_apap_rx)])

        assert result.exit_code == 0
        assert "6.0-20.0 cmH2O" in result.output

    def test_history_bilevel_shows_epap_ipap_format(self, cli_runner, db_with_bipap_rx):
        """BiPAP Auto period renders pressure as epap-ipap cmH2O (EPAP-IPAP)."""
        result = cli_runner.invoke(
            cli, ["rx", "history", "--db", str(db_with_bipap_rx)]
        )

        assert result.exit_code == 0
        assert "6.0-18.0 cmH2O (EPAP-IPAP)" in result.output

    def test_history_bilevel_shows_ps_not_epr(self, cli_runner, db_with_bipap_rx):
        """BiPAP Auto period shows PS segment, not EPR."""
        result = cli_runner.invoke(
            cli, ["rx", "history", "--db", str(db_with_bipap_rx)]
        )

        assert result.exit_code == 0
        # The BiPAP Auto period must show PS
        assert "PS: 4.0" in result.output
        # And must NOT show EPR for the bilevel period
        lines = result.output.splitlines()
        bipap_lines = [l for l in lines if "6.0-18.0" in l]
        assert bipap_lines, "Expected a BiPAP period line with EPAP-IPAP pressure"
        assert all("EPR:" not in l for l in bipap_lines)


class TestRxCurrentCommand:
    """Test rx current command — EPR vs PS vs omitted segment."""

    def test_current_apap_shows_epr_when_present(self, cli_runner, db_with_apap_rx):
        """APAP current period shows EPR key/value when both epr keys are present."""
        result = cli_runner.invoke(cli, ["rx", "current", "--db", str(db_with_apap_rx)])

        assert result.exit_code == 0
        assert "EPR:" in result.output

    def test_current_bilevel_shows_ps_no_epr(self, cli_runner, db_with_bipap_rx):
        """BiPAP Auto current period shows PS, not EPR."""
        result = cli_runner.invoke(
            cli, ["rx", "current", "--db", str(db_with_bipap_rx)]
        )

        assert result.exit_code == 0
        assert "PS:" in result.output
        assert "EPR:" not in result.output


class TestRxCompareCommand:
    """Test rx compare command — bilevel pressure fits in the pressure column."""

    def test_compare_bilevel_pressure_in_table(self, cli_runner, db_with_bipap_rx):
        """BiPAP Auto period shows short epap-ipap form in the comparison table."""
        result = cli_runner.invoke(
            cli, ["rx", "compare", "--db", str(db_with_bipap_rx)]
        )

        assert result.exit_code == 0
        # Short form for bilevel: "<epap>-<ipap>" without units
        assert "6.0-18.0" in result.output


class TestProfileTimezoneCommand:
    """Test profile set-timezone command and timezone display in profile list."""

    @pytest.fixture
    async def db_with_profile(self, temp_db):
        """Database with a single user + profile; returns (db_path, profile_id)."""
        await init_database(str(temp_db))
        async with session_scope() as session:
            _profile = await _create_test_user_and_profile(session)
            profile_id = _profile.id
        return temp_db, profile_id

    async def _get_timezone(self, db_path: Path, profile_id: int) -> str | None:
        await init_database(str(db_path))
        async with session_scope() as session:
            row = await session.get(models.Profile, profile_id)
            return row.timezone

    def test_set_timezone_valid_zone_persists(self, cli_runner, db_with_profile):
        db_path, profile_id = db_with_profile
        result = cli_runner.invoke(
            cli,
            [
                "profile",
                "set-timezone",
                str(profile_id),
                "America/New_York",
                "--db",
                str(db_path),
            ],
        )

        assert result.exit_code == 0
        assert "America/New_York" in result.output
        assert (
            asyncio.run(self._get_timezone(db_path, profile_id)) == "America/New_York"
        )

    def test_clear_timezone_resets_to_null(self, cli_runner, db_with_profile):
        db_path, profile_id = db_with_profile
        cli_runner.invoke(
            cli,
            [
                "profile",
                "set-timezone",
                str(profile_id),
                "Europe/London",
                "--db",
                str(db_path),
            ],
        )
        result = cli_runner.invoke(
            cli,
            [
                "profile",
                "set-timezone",
                str(profile_id),
                "--clear",
                "--db",
                str(db_path),
            ],
        )

        assert result.exit_code == 0
        assert "Cleared" in result.output
        assert asyncio.run(self._get_timezone(db_path, profile_id)) is None

    def test_invalid_zone_rejected_with_example(self, cli_runner, db_with_profile):
        db_path, profile_id = db_with_profile
        result = cli_runner.invoke(
            cli,
            [
                "profile",
                "set-timezone",
                str(profile_id),
                "Not/AZone",
                "--db",
                str(db_path),
            ],
        )

        assert "America/New_York" in result.output  # friendly example in the error
        assert asyncio.run(self._get_timezone(db_path, profile_id)) is None

    def test_tz_and_clear_together_rejected(self, cli_runner, db_with_profile):
        db_path, profile_id = db_with_profile
        result = cli_runner.invoke(
            cli,
            [
                "profile",
                "set-timezone",
                str(profile_id),
                "Europe/London",
                "--clear",
                "--db",
                str(db_path),
            ],
        )

        assert "exactly one" in result.output
        assert asyncio.run(self._get_timezone(db_path, profile_id)) is None

    def test_profile_list_shows_timezone(self, cli_runner, db_with_profile):
        db_path, profile_id = db_with_profile
        cli_runner.invoke(
            cli,
            [
                "profile",
                "set-timezone",
                str(profile_id),
                "America/New_York",
                "--db",
                str(db_path),
            ],
        )
        result = cli_runner.invoke(cli, ["profile", "list", "--db", str(db_path)])

        assert result.exit_code == 0
        assert "America/New_York" in result.output


class TestImportCorruptTimezone:
    """A corrupt stored Profile.timezone must fail the real import cleanly.

    ZoneInfoNotFoundError is a KeyError subclass; if it escaped to the CLI it
    would bypass the `except RuntimeError` handler and print a raw traceback.
    The service re-raises it as RuntimeError, which the CLI renders as a
    click error with the `snore profile set-timezone` remediation hint.
    """

    @pytest.fixture
    async def db_with_corrupt_timezone(self, temp_db):
        """Database with one user + profile whose timezone is corrupt."""
        await init_database(str(temp_db))
        async with session_scope() as session:
            _profile = await _create_test_user_and_profile(session)
            # Written directly — bypasses `snore profile set-timezone`
            # validation, simulating a corrupted stored value.
            _profile.timezone = "Not/A_Zone"
        return temp_db

    def test_corrupt_timezone_yields_clean_error(
        self, cli_runner, db_with_corrupt_timezone, tmp_path, monkeypatch
    ):
        from unittest.mock import patch

        import snore.logging_config as logging_config

        from snore.services.import_service import ImportService
        from snore.services.schemas import ImportSource

        # Pin the normal (non --verbose) user path: the CLI re-raises import
        # errors instead of rendering them when verbose_mode is set.
        monkeypatch.setattr(logging_config, "verbose_mode", False)

        source = ImportSource(parser_name="oscar_binary", root_path=str(tmp_path))
        with patch.object(ImportService, "detect_sources", return_value=[source]):
            result = cli_runner.invoke(
                cli,
                [
                    "import",
                    str(tmp_path),
                    "--db",
                    str(db_with_corrupt_timezone),
                    "--no-backup",
                ],
            )

        assert result.exit_code != 0
        # Clean, actionable click error — not a raw traceback.
        assert "Not/A_Zone" in result.output
        assert "snore profile set-timezone" in result.output
        assert "Traceback" not in result.output
        assert not isinstance(result.exception, KeyError)

    def test_verbose_mode_reraises_for_debugging(
        self, cli_runner, db_with_corrupt_timezone, tmp_path, monkeypatch
    ):
        from unittest.mock import patch

        import snore.logging_config as logging_config

        from snore.services.import_service import ImportService
        from snore.services.schemas import ImportSource

        monkeypatch.setattr(logging_config, "verbose_mode", True)

        source = ImportSource(parser_name="oscar_binary", root_path=str(tmp_path))
        with patch.object(ImportService, "detect_sources", return_value=[source]):
            result = cli_runner.invoke(
                cli,
                [
                    "import",
                    str(tmp_path),
                    "--db",
                    str(db_with_corrupt_timezone),
                    "--no-backup",
                ],
            )

        # --verbose users get the raw exception for debugging.
        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)


class TestDbCleanupOrphansCommand:
    """Test db cleanup-orphans command."""

    @pytest.fixture
    async def db_with_orphans(self, temp_db):
        """Database with orphaned events and settings (session_id 9999 doesn't exist).

        Orphaned rows are inserted via stdlib sqlite3 which does not set
        PRAGMA foreign_keys=ON by default, bypassing the FK enforcement that
        the aiosqlite engine applies via its "connect" event.
        """
        import sqlite3  # noqa: PLC0415

        await init_database(str(temp_db))

        async with session_scope() as sess:
            _profile = await _create_test_user_and_profile(sess)
            device = models.Device(
                profile_id=_profile.id,
                manufacturer="Test",
                model="Test",
                serial_number="ORPHAN_TEST",
            )
            sess.add(device)
            await sess.flush()

        # Use a stdlib sqlite3 connection (FK enforcement off by default) to
        # insert rows that reference non-existent session_id 9999.
        conn = sqlite3.connect(str(temp_db))
        try:
            conn.execute(
                "INSERT INTO events"
                " (session_id, event_type, start_time, duration_seconds)"
                " VALUES (9999, 'Apnea', '2025-01-01T00:00:00', 15)"
            )
            conn.execute(
                "INSERT INTO settings (session_id, key, value)"
                " VALUES (9999, 'mode', 'CPAP')"
            )
            conn.commit()
        finally:
            conn.close()

        return temp_db

    def test_cleanup_removes_orphaned_rows_and_reports_counts(
        self, cli_runner, db_with_orphans, db_session
    ):
        """Orphaned records are deleted; output shows per-table counts and vacuum tip."""
        events_before = db_session.execute(text("SELECT COUNT(*) FROM events")).scalar()
        settings_before = db_session.execute(
            text("SELECT COUNT(*) FROM settings")
        ).scalar()
        assert events_before == 1
        assert settings_before == 1

        result = cli_runner.invoke(
            cli,
            ["db", "cleanup-orphans", "--db", str(db_with_orphans)],
            input="y\n",
        )

        assert result.exit_code == 0, result.output

        db_session.expire_all()
        events_after = db_session.execute(text("SELECT COUNT(*) FROM events")).scalar()
        settings_after = db_session.execute(
            text("SELECT COUNT(*) FROM settings")
        ).scalar()
        assert events_after == 0
        assert settings_after == 0
        assert "vacuum" in result.output.lower()
        # Per-table counts must appear in output for the seeded tables.
        # print_kv("events", "1") renders as "  events: 1" (Rich strips markup).
        assert "events: 1" in result.output
        assert "settings: 1" in result.output

    def test_cleanup_reports_clean_when_no_orphans(self, cli_runner, temp_db):
        """When no orphaned records exist, command reports a clean database."""
        asyncio.run(init_database(str(temp_db)))

        result = cli_runner.invoke(
            cli,
            ["db", "cleanup-orphans", "--db", str(temp_db)],
            input="y\n",
        )

        assert result.exit_code == 0, result.output
        assert "No orphaned" in result.output

    def test_cleanup_cancellation_aborts(self, cli_runner, db_with_orphans, db_session):
        """Declining the confirmation prompt leaves orphaned records untouched."""
        events_before = db_session.execute(text("SELECT COUNT(*) FROM events")).scalar()
        assert events_before == 1

        result = cli_runner.invoke(
            cli,
            ["db", "cleanup-orphans", "--db", str(db_with_orphans)],
            input="N\n",
        )

        assert result.exit_code != 0

        db_session.expire_all()
        events_after = db_session.execute(text("SELECT COUNT(*) FROM events")).scalar()
        assert events_after == 1
