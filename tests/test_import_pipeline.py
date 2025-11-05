"""
Integration tests for the complete import pipeline.

Tests the full flow from parsing → database storage → retrieval.
"""

import pytest

from oscar_mcp.database import DatabaseManager, SessionImporter


class TestImportPipeline:
    """Test complete import pipeline."""

    def test_database_auto_creation(self, temp_db):
        """Test that database is auto-created on first use."""
        # Database shouldn't exist yet
        assert not temp_db.exists()

        # Initialize manager
        with DatabaseManager(db_path=temp_db) as db:
            # Database should now exist
            assert temp_db.exists()

            # Verify tables were created
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {row[0] for row in cursor.fetchall()}

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
        # Parse session
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        assert len(sessions) > 0

        session = sessions[0]

        # Import to database
        with DatabaseManager(db_path=temp_db) as db:
            importer = SessionImporter(db)

            result = importer.import_session(session)
            assert result is True, "Session should be imported"

            # Verify device was created
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM devices")
                assert cursor.fetchone()[0] == 1

            # Verify session was created
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM sessions")
                db_session = cursor.fetchone()
                assert db_session is not None
                assert db_session["device_session_id"] == session.device_session_id

    def test_duplicate_import_prevention(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test that duplicate sessions are not re-imported."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        db = DatabaseManager(db_path=temp_db)
        importer = SessionImporter(db)

        # First import should succeed
        result1 = importer.import_session(session)
        assert result1 is True

        # Second import should be skipped
        result2 = importer.import_session(session)
        assert result2 is False

        # Verify only one session exists
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            assert cursor.fetchone()[0] == 1

    def test_force_reimport(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test force re-import of existing session."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        db = DatabaseManager(db_path=temp_db)
        importer = SessionImporter(db)

        # First import
        importer.import_session(session)

        # Force re-import
        result = importer.import_session(session, force=True)
        assert result is True

        # Should still have only one session
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            assert cursor.fetchone()[0] == 1

    def test_waveform_storage(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test that waveforms are stored correctly."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        db = DatabaseManager(db_path=temp_db)
        importer = SessionImporter(db)
        importer.import_session(session)

        # Verify waveforms were stored
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM waveforms")
            waveforms = cursor.fetchall()

        assert len(waveforms) > 0

        # Check waveform data
        for wf in waveforms:
            assert wf["data_blob"] is not None
            assert wf["sample_count"] > 0
            assert wf["sample_rate"] > 0
            # BLOB should contain data
            assert len(wf["data_blob"]) > 0

    def test_event_storage(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test that events are stored correctly."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        # Only test if session has events
        if not session.has_event_data or len(session.events) == 0:
            pytest.skip("Test session has no events")

        db = DatabaseManager(db_path=temp_db)
        importer = SessionImporter(db)
        importer.import_session(session)

        # Verify events were stored
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            event_count = cursor.fetchone()[0]

        assert event_count == len(session.events)

    def test_statistics_storage(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test that statistics are stored correctly."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        db = DatabaseManager(db_path=temp_db)
        importer = SessionImporter(db)
        importer.import_session(session)

        # Verify statistics were stored
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM statistics")
            stats = cursor.fetchone()

        if session.has_statistics:
            assert stats is not None
        # If no statistics, that's okay too

    def test_database_stats(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test database statistics reporting."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))

        db = DatabaseManager(db_path=temp_db)
        importer = SessionImporter(db)

        # Import all sessions
        for session in sessions:
            importer.import_session(session)

        # Get stats
        stats = db.get_stats()

        assert stats["devices"] >= 1
        assert stats["sessions"] == len(sessions)
        assert stats["size_mb"] > 0

    def test_multiple_devices(self, temp_db):
        """Test handling multiple devices."""
        # This would require fixtures from multiple manufacturers
        # For now, just verify the structure supports it
        db = DatabaseManager(db_path=temp_db)

        # Verify devices table exists and has unique constraint
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='devices'"
            )
            schema = cursor.fetchone()[0]
            assert "UNIQUE" in schema
            assert "serial_number" in schema
