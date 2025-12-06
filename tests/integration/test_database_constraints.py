"""
Tests for database constraints and foreign key behavior.

These tests verify:
- Foreign key constraints are enabled
- CASCADE delete behavior works correctly
- Orphaned record cleanup functionality
- Data integrity is maintained
"""

from datetime import datetime, timedelta

import pytest

from sqlalchemy import text

from snore.database import models
from snore.database.importers import SessionImporter
from snore.database.session import cleanup_database, init_database, session_scope


@pytest.fixture(autouse=True)
def reset_database_state():
    """Reset global database state before and after each test."""
    cleanup_database()
    yield
    cleanup_database()


class TestForeignKeyConstraints:
    """Test that foreign key constraints are properly enabled and enforced."""

    def test_foreign_keys_enabled(self, temp_db):
        """Test that foreign keys are enabled on connection."""
        init_database(str(temp_db))

        with session_scope() as session:
            result = session.execute(text("PRAGMA foreign_keys")).scalar()
            assert result == 1, "Foreign keys should be enabled"

    def test_cascade_delete_session_to_events(self, temp_db):
        """Test that deleting a session cascades to events."""
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
                serial_number="TEST123",
            )
            session.add(device)
            session.flush()

            start_time = datetime(2025, 10, 1, 22, 0, 0)
            test_session = models.Session(
                device_id=device.id,
                device_session_id="test_session",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
            )
            session.add(test_session)
            session.flush()

            event = models.Event(
                session_id=test_session.id,
                event_type="Apnea",
                start_time=start_time + timedelta(hours=2),
                duration_seconds=15.0,
            )
            session.add(event)
            session.commit()

            session_id = test_session.id

        with session_scope() as session:
            events_before = session.execute(
                text("SELECT COUNT(*) FROM events WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
            assert events_before == 1

            sess = session.get(models.Session, session_id)
            session.delete(sess)
            session.commit()

        with session_scope() as session:
            events_after = session.execute(
                text("SELECT COUNT(*) FROM events WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
            assert events_after == 0, "Events should be deleted via CASCADE"

    def test_cascade_delete_session_to_statistics(self, temp_db):
        """Test that deleting a session cascades to statistics."""
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
                serial_number="TEST123",
            )
            session.add(device)
            session.flush()

            start_time = datetime(2025, 10, 1, 22, 0, 0)
            test_session = models.Session(
                device_id=device.id,
                device_session_id="test_session",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
                has_statistics=True,
            )
            session.add(test_session)
            session.flush()

            stats = models.Statistics(session_id=test_session.id, ahi=5.2)
            session.add(stats)
            session.commit()

            session_id = test_session.id

        with session_scope() as session:
            stats_before = session.execute(
                text("SELECT COUNT(*) FROM statistics WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
            assert stats_before == 1

            sess = session.get(models.Session, session_id)
            session.delete(sess)
            session.commit()

        with session_scope() as session:
            stats_after = session.execute(
                text("SELECT COUNT(*) FROM statistics WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
            assert stats_after == 0, "Statistics should be deleted via CASCADE"

    def test_cascade_delete_session_to_settings(self, temp_db):
        """Test that deleting a session cascades to settings."""
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
                serial_number="TEST123",
            )
            session.add(device)
            session.flush()

            start_time = datetime(2025, 10, 1, 22, 0, 0)
            test_session = models.Session(
                device_id=device.id,
                device_session_id="test_session",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
            )
            session.add(test_session)
            session.flush()

            setting = models.Setting(
                session_id=test_session.id, key="mode", value="CPAP"
            )
            session.add(setting)
            session.commit()

            session_id = test_session.id

        with session_scope() as session:
            settings_before = session.execute(
                text("SELECT COUNT(*) FROM settings WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
            assert settings_before == 1

            sess = session.get(models.Session, session_id)
            session.delete(sess)
            session.commit()

        with session_scope() as session:
            settings_after = session.execute(
                text("SELECT COUNT(*) FROM settings WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
            assert settings_after == 0, "Settings should be deleted via CASCADE"

    def test_cascade_delete_session_to_waveforms(self, temp_db):
        """Test that deleting a session cascades to waveforms."""
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
                serial_number="TEST123",
            )
            session.add(device)
            session.flush()

            start_time = datetime(2025, 10, 1, 22, 0, 0)
            test_session = models.Session(
                device_id=device.id,
                device_session_id="test_session",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
                has_waveform_data=True,
            )
            session.add(test_session)
            session.flush()

            waveform = models.Waveform(
                session_id=test_session.id,
                waveform_type="FlowRate",
                sample_rate=25.0,
                data_blob=b"\x00\x01\x02\x03",
            )
            session.add(waveform)
            session.commit()

            session_id = test_session.id

        with session_scope() as session:
            waveforms_before = session.execute(
                text("SELECT COUNT(*) FROM waveforms WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
            assert waveforms_before == 1

            sess = session.get(models.Session, session_id)
            session.delete(sess)
            session.commit()

        with session_scope() as session:
            waveforms_after = session.execute(
                text("SELECT COUNT(*) FROM waveforms WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
            assert waveforms_after == 0, "Waveforms should be deleted via CASCADE"


class TestOrphanedRecordCleanup:
    """Test orphaned record detection and cleanup functionality."""

    def test_cleanup_orphaned_events(self, temp_db):
        """Test cleanup of orphaned event records."""
        init_database(str(temp_db))

        with session_scope() as session:
            # Temporarily disable foreign keys to create orphaned records
            session.execute(text("PRAGMA foreign_keys = OFF"))
            session.execute(
                text(
                    "INSERT INTO events (session_id, event_type, start_time, duration_seconds) VALUES (999, 'Apnea', '2025-10-01 22:00:00', 15.0)"
                )
            )
            session.commit()
            # Re-enable foreign keys
            session.execute(text("PRAGMA foreign_keys = ON"))

        with session_scope() as session:
            orphaned_before = session.execute(
                text("SELECT COUNT(*) FROM events WHERE session_id = 999")
            ).scalar()
            assert orphaned_before == 1

            cleaned = SessionImporter.cleanup_orphaned_records(session)
            assert cleaned >= 1

            orphaned_after = session.execute(
                text("SELECT COUNT(*) FROM events WHERE session_id = 999")
            ).scalar()
            assert orphaned_after == 0

    def test_cleanup_orphaned_statistics(self, temp_db):
        """Test cleanup of orphaned statistics records."""
        init_database(str(temp_db))

        with session_scope() as session:
            # Temporarily disable foreign keys to create orphaned records
            session.execute(text("PRAGMA foreign_keys = OFF"))
            # Use ORM to create orphaned record (defaults applied automatically)
            orphaned_stat = models.Statistics(session_id=999, ahi=5.2)
            session.add(orphaned_stat)
            session.commit()
            # Re-enable foreign keys
            session.execute(text("PRAGMA foreign_keys = ON"))

        with session_scope() as session:
            orphaned_before = session.execute(
                text("SELECT COUNT(*) FROM statistics WHERE session_id = 999")
            ).scalar()
            assert orphaned_before == 1

            cleaned = SessionImporter.cleanup_orphaned_records(session)
            assert cleaned >= 1

            orphaned_after = session.execute(
                text("SELECT COUNT(*) FROM statistics WHERE session_id = 999")
            ).scalar()
            assert orphaned_after == 0

    def test_cleanup_orphaned_settings(self, temp_db):
        """Test cleanup of orphaned settings records."""
        init_database(str(temp_db))

        with session_scope() as session:
            # Temporarily disable foreign keys to create orphaned records
            session.execute(text("PRAGMA foreign_keys = OFF"))
            session.execute(
                text(
                    "INSERT INTO settings (session_id, key, value) VALUES (999, 'mode', 'CPAP')"
                )
            )
            session.commit()
            # Re-enable foreign keys
            session.execute(text("PRAGMA foreign_keys = ON"))

        with session_scope() as session:
            orphaned_before = session.execute(
                text("SELECT COUNT(*) FROM settings WHERE session_id = 999")
            ).scalar()
            assert orphaned_before == 1

            cleaned = SessionImporter.cleanup_orphaned_records(session)
            assert cleaned >= 1

            orphaned_after = session.execute(
                text("SELECT COUNT(*) FROM settings WHERE session_id = 999")
            ).scalar()
            assert orphaned_after == 0

    def test_cleanup_orphaned_waveforms(self, temp_db):
        """Test cleanup of orphaned waveform records."""
        init_database(str(temp_db))

        with session_scope() as session:
            # Temporarily disable foreign keys to create orphaned records
            session.execute(text("PRAGMA foreign_keys = OFF"))
            session.execute(
                text(
                    "INSERT INTO waveforms (session_id, waveform_type, sample_rate, data_blob) VALUES (999, 'FlowRate', 25.0, x'00010203')"
                )
            )
            session.commit()
            # Re-enable foreign keys
            session.execute(text("PRAGMA foreign_keys = ON"))

        with session_scope() as session:
            orphaned_before = session.execute(
                text("SELECT COUNT(*) FROM waveforms WHERE session_id = 999")
            ).scalar()
            assert orphaned_before == 1

            cleaned = SessionImporter.cleanup_orphaned_records(session)
            assert cleaned >= 1

            orphaned_after = session.execute(
                text("SELECT COUNT(*) FROM waveforms WHERE session_id = 999")
            ).scalar()
            assert orphaned_after == 0

    def test_cleanup_multiple_orphaned_records(self, temp_db):
        """Test cleanup of multiple orphaned records across tables."""
        init_database(str(temp_db))

        with session_scope() as session:
            # Temporarily disable foreign keys to create orphaned records
            session.execute(text("PRAGMA foreign_keys = OFF"))
            # Use ORM to create orphaned records (defaults applied automatically)
            from datetime import datetime

            orphaned_event = models.Event(
                session_id=999,
                event_type="Apnea",
                start_time=datetime(2025, 10, 1, 22, 0, 0),
                duration_seconds=15.0,
            )
            orphaned_stat = models.Statistics(session_id=999, ahi=5.2)
            orphaned_setting = models.Setting(session_id=999, key="mode", value="CPAP")
            session.add_all([orphaned_event, orphaned_stat, orphaned_setting])
            session.commit()
            # Re-enable foreign keys
            session.execute(text("PRAGMA foreign_keys = ON"))

        with session_scope() as session:
            cleaned = SessionImporter.cleanup_orphaned_records(session)
            assert cleaned == 3

            total_orphaned = session.execute(
                text("SELECT COUNT(*) FROM events WHERE session_id = 999")
            ).scalar()
            total_orphaned += session.execute(
                text("SELECT COUNT(*) FROM statistics WHERE session_id = 999")
            ).scalar()
            total_orphaned += session.execute(
                text("SELECT COUNT(*) FROM settings WHERE session_id = 999")
            ).scalar()

            assert total_orphaned == 0

    def test_cleanup_no_orphaned_records(self, temp_db):
        """Test cleanup returns 0 when no orphaned records exist."""
        init_database(str(temp_db))

        with session_scope() as session:
            cleaned = SessionImporter.cleanup_orphaned_records(session)
            assert cleaned == 0


class TestDataIntegrity:
    """Test database constraints and data integrity."""

    def test_unique_constraint_device_session(self, temp_db):
        """Test unique constraint on (device_id, device_session_id)."""
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
                serial_number="TEST123",
            )
            session.add(device)
            session.flush()

            start_time = datetime(2025, 10, 1, 22, 0, 0)
            session1 = models.Session(
                device_id=device.id,
                device_session_id="duplicate_id",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
            )
            session.add(session1)
            session.commit()

        with session_scope() as session:
            device = session.query(models.Device).first()
            session2 = models.Session(
                device_id=device.id,
                device_session_id="duplicate_id",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
            )
            session.add(session2)

            with pytest.raises(Exception) as exc_info:
                session.commit()

            session.rollback()
            assert "UNIQUE constraint" in str(exc_info.value)

    def test_unique_constraint_setting_key(self, temp_db):
        """Test unique constraint on (session_id, key) for settings."""
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
                serial_number="TEST123",
            )
            session.add(device)
            session.flush()

            start_time = datetime(2025, 10, 1, 22, 0, 0)
            test_session = models.Session(
                device_id=device.id,
                device_session_id="test_session",
                start_time=start_time,
                end_time=start_time + timedelta(hours=8),
                duration_seconds=8 * 3600,
            )
            session.add(test_session)
            session.flush()

            setting1 = models.Setting(
                session_id=test_session.id, key="mode", value="CPAP"
            )
            session.add(setting1)
            session.commit()

        with session_scope() as session:
            test_session = session.query(models.Session).first()
            setting2 = models.Setting(
                session_id=test_session.id, key="mode", value="APAP"
            )
            session.add(setting2)

            with pytest.raises(Exception) as exc_info:
                session.commit()

            session.rollback()
            assert "UNIQUE constraint" in str(exc_info.value)
