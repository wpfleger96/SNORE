"""
Integration tests for the complete import pipeline.

Tests the full flow from parsing → database storage → retrieval.
"""

import uuid

import pytest

from sqlalchemy import func, select, text

from snore.database import models
from snore.database.importers import import_session
from snore.database.session import init_database, session_scope


async def _create_profile_id() -> int:
    """Create a User + Profile in the current session_scope DB and return the profile id."""
    async with session_scope() as db:
        user = models.User(
            canonical_email=f"pipeline_{uuid.uuid4().hex[:8]}@example.com",
            role="admin",
        )
        db.add(user)
        await db.flush()
        profile = models.Profile(user_id=user.id, name="Test Pipeline Profile")
        db.add(profile)
        await db.flush()
        return profile.id


class TestImportPipeline:
    """Integration tests for the full import pipeline."""

    async def test_database_auto_creation(self, temp_db):
        """Test that database is auto-created on first use."""
        assert not temp_db.exists()

        await init_database(str(temp_db))

        assert temp_db.exists()

        async with session_scope() as session:
            result = await session.execute(
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

    async def test_import_resmed_session(
        self, temp_db, resmed_parser, resmed_fixture_path
    ):
        """Test importing a complete ResMed session."""
        await init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        assert len(sessions) > 0

        session_data = sessions[0]
        profile_id = await _create_profile_id()
        result = await import_session(session_data, profile_id=profile_id)
        assert result is True, "Session should be imported"

        async with session_scope() as session:
            device_count = (
                await session.execute(select(func.count()).select_from(models.Device))
            ).scalar()
            assert device_count == 1

            db_session = (
                (await session.execute(select(models.Session))).scalars().first()
            )
            assert db_session is not None
            assert db_session.device_session_id == session_data.device_session_id

    async def test_duplicate_import_prevention(
        self, temp_db, resmed_parser, resmed_fixture_path
    ):
        """Test that duplicate sessions are not re-imported."""
        await init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]

        profile_id = await _create_profile_id()
        result1 = await import_session(session_data, profile_id=profile_id)
        assert result1 is True

        result2 = await import_session(session_data, profile_id=profile_id)
        assert result2 is False

        async with session_scope() as session:
            session_count = (
                await session.execute(select(func.count()).select_from(models.Session))
            ).scalar()
            assert session_count == 1

    async def test_force_reimport(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test force re-import of existing session."""
        await init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]

        profile_id = await _create_profile_id()
        await import_session(session_data, profile_id=profile_id)

        result = await import_session(session_data, force=True, profile_id=profile_id)
        assert result is True

        async with session_scope() as session:
            session_count = (
                await session.execute(select(func.count()).select_from(models.Session))
            ).scalar()
            assert session_count == 1

    # ``False`` is the #369 regression case; ``True`` guards the positive day totals.
    @pytest.mark.parametrize(
        "enabled", [False, True], ids=["disabled-regression", "enabled-baseline"]
    )
    async def test_force_reimport_preserves_enabled_flag(
        self, temp_db, resmed_parser, resmed_fixture_path, enabled
    ):
        """Force re-import keeps the session's enabled flag and the day honors it."""
        await init_database(str(temp_db))

        session_data = next(iter(resmed_parser.parse_sessions(resmed_fixture_path)))
        profile_id = await _create_profile_id()
        await import_session(session_data, profile_id=profile_id)
        async with session_scope() as session:
            db_session = (await session.execute(select(models.Session))).scalar_one()
            db_session.enabled = enabled

        result = await import_session(session_data, force=True, profile_id=profile_id)
        assert result is True

        async with session_scope() as session:
            db_session = (await session.execute(select(models.Session))).scalar_one()
            day = await session.get(models.Day, db_session.day_id)
            assert db_session.enabled is enabled
            assert day is not None
            assert day.session_count == (1 if enabled else 0)
            assert (day.total_therapy_hours > 0) is enabled
            if not enabled:
                # With its only session disabled the day must not headline the
                # device's daily AHI (or any index).
                assert day.ahi is None
                assert day.index_source is None

    async def test_waveform_storage(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test that waveforms are stored correctly."""
        await init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]
        profile_id = await _create_profile_id()
        await import_session(session_data, profile_id=profile_id)

        async with session_scope() as session:
            waveforms = (await session.execute(select(models.Waveform))).scalars().all()

            assert len(waveforms) > 0

            for wf in waveforms:
                assert wf.data_blob is not None
                assert wf.sample_count > 0
                assert wf.sample_rate > 0
                assert len(wf.data_blob) > 0

    async def test_event_storage(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test that events are stored correctly."""
        await init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]

        if not session_data.has_event_data or len(session_data.events) == 0:
            pytest.skip("Test session has no events")
        profile_id = await _create_profile_id()
        await import_session(session_data, profile_id=profile_id)

        async with session_scope() as session:
            event_count = (
                await session.execute(select(func.count()).select_from(models.Event))
            ).scalar()

        assert event_count == len(session_data.events)

    async def test_statistics_storage(
        self, temp_db, resmed_parser, resmed_fixture_path
    ):
        """Test that statistics are stored correctly."""
        await init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]
        profile_id = await _create_profile_id()
        await import_session(session_data, profile_id=profile_id)

        async with session_scope() as session:
            stats = (await session.execute(select(models.Statistics))).scalars().first()

        if session_data.has_statistics:
            assert stats is not None

    async def test_settings_import_skips_none_values(
        self, temp_db, resmed_parser, resmed_fixture_path
    ):
        """None values in other_settings must not be persisted as the string 'None'."""
        await init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session_data = sessions[0]
        if session_data.settings is None:
            pytest.skip("fixture session has no settings")

        session_data.settings.other_settings["unexpected_none"] = None
        profile_id = await _create_profile_id()
        await import_session(session_data, profile_id=profile_id)

        async with session_scope() as session:
            values = [
                s.value
                for s in (await session.execute(select(models.Setting))).scalars().all()
            ]

        assert "None" not in values

    async def test_database_stats(self, temp_db, resmed_parser, resmed_fixture_path):
        """Test database statistics reporting."""
        await init_database(str(temp_db))

        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))

        profile_id = await _create_profile_id()
        for session_data in sessions:
            await import_session(session_data, profile_id=profile_id)

        async with session_scope() as session:
            device_count = (
                await session.execute(select(func.count()).select_from(models.Device))
            ).scalar()
            session_count = (
                await session.execute(select(func.count()).select_from(models.Session))
            ).scalar()

            result = await session.execute(
                text(
                    "SELECT page_count * page_size / 1024.0 / 1024.0 as size_mb FROM pragma_page_count(), pragma_page_size()"
                )
            )
            size_mb = result.fetchone()[0]

        assert device_count >= 1
        assert session_count == len(sessions)
        assert size_mb > 0

    async def test_multiple_devices(self, temp_db):
        """Test handling multiple devices."""
        await init_database(str(temp_db))

        async with session_scope() as session:
            result = await session.execute(
                text(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='devices'"
                )
            )
            schema = result.fetchone()[0]
            assert "UNIQUE" in schema
            assert "serial_number" in schema
