"""
Integration tests for the complete import pipeline.

Tests the full flow from parsing → database storage → retrieval.
"""

import pytest

from sqlalchemy import text

from snore.database import models
from snore.database.importers import SessionImporter
from snore.database.session import cleanup_database, init_database, session_scope


@pytest.fixture(autouse=True)
def reset_database_state():
    """Reset global database state before and after each test to ensure isolation."""
    cleanup_database()  # Clean before test
    yield
    cleanup_database()  # Clean after test


class TestImportPipeline:
    """Test complete import pipeline."""

    def test_database_auto_creation(self, temp_db):
        """Test that database is auto-created on first use."""
        assert not temp_db.exists()

        init_database(str(temp_db))

        assert temp_db.exists()

        with session_scope() as session:
            result = session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = {row[0] for row in result.fetchall()}

        required_tables = {
            "devices",
            "sessions",
            "waveforms",
            "events",
            "statistics",
            "settings",
        }
        assert required_tables.issubset(tables)

    def test_import_resmed_session(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test importing a complete ResMed session."""
        init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        assert len(sessions) > 0

        session_data = sessions[0]

        importer = SessionImporter()
        result = importer.import_session(session_data)
        assert result is True, "Session should be imported"

        with session_scope() as session:
            device_count = session.query(models.Device).count()
            assert device_count == 1

            db_session = session.query(models.Session).first()
            assert db_session is not None
            assert db_session.device_session_id == session_data.device_session_id

    def test_duplicate_import_prevention(
        self, temp_db, resmed_parser, resmed_fixture_path
    ):
        """Test that duplicate sessions are not re-imported."""
        init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]

        importer = SessionImporter()

        result1 = importer.import_session(session_data)
        assert result1 is True

        result2 = importer.import_session(session_data)
        assert result2 is False

        with session_scope() as session:
            session_count = session.query(models.Session).count()
            assert session_count == 1

    def test_force_reimport(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test force re-import of existing session."""
        init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]

        importer = SessionImporter()

        importer.import_session(session_data)

        result = importer.import_session(session_data, force=True)
        assert result is True

        with session_scope() as session:
            session_count = session.query(models.Session).count()
            assert session_count == 1

    def test_waveform_storage(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test that waveforms are stored correctly."""
        init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]

        importer = SessionImporter()
        importer.import_session(session_data)

        with session_scope() as session:
            waveforms = session.query(models.Waveform).all()

            assert len(waveforms) > 0

            for wf in waveforms:
                assert wf.data_blob is not None
                assert wf.sample_count > 0
                assert wf.sample_rate > 0
                assert len(wf.data_blob) > 0

    def test_event_storage(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test that events are stored correctly."""
        init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]

        if not session_data.has_event_data or len(session_data.events) == 0:
            pytest.skip("Test session has no events")

        importer = SessionImporter()
        importer.import_session(session_data)

        with session_scope() as session:
            event_count = session.query(models.Event).count()

        assert event_count == len(session_data.events)

    def test_statistics_storage(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test that statistics are stored correctly."""
        init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]

        importer = SessionImporter()
        importer.import_session(session_data)

        with session_scope() as session:
            stats = session.query(models.Statistics).first()

        if session_data.has_statistics:
            assert stats is not None

    def test_database_stats(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test database statistics reporting."""
        init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))

        importer = SessionImporter()

        for session_data in sessions:
            importer.import_session(session_data)

        with session_scope() as session:
            device_count = session.query(models.Device).count()
            session_count = session.query(models.Session).count()

            result = session.execute(
                text(
                    "SELECT page_count * page_size / 1024.0 / 1024.0 as size_mb FROM pragma_page_count(), pragma_page_size()"
                )
            )
            size_mb = result.fetchone()[0]

        assert device_count >= 1
        assert session_count == len(sessions)
        assert size_mb > 0

    def test_multiple_devices(self, temp_db):
        """Test handling multiple devices."""
        init_database(str(temp_db))

        with session_scope() as session:
            result = session.execute(
                text(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='devices'"
                )
            )
            schema = result.fetchone()[0]
            assert "UNIQUE" in schema
            assert "serial_number" in schema
