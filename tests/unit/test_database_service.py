"""Unit tests for DatabaseService."""

from datetime import datetime, timedelta

import pytest

from snore.database.models import (
    AnalysisResult,
    Device,
    Profile,
    Session,
    Statistics,
    User,
    Waveform,
)
from snore.services import waveform_service as waveform_service_module
from snore.services.database_service import DatabaseService

# Import the real _vacuum_background at module level so we bypass the autouse
# _disable_background_vacuum fixture that replaces it with a no-op per test.
from snore.services.database_service import _vacuum_background as _REAL_VACUUM_BG
from snore.services.device_service import DeviceService
from snore.services.waveform_service import WaveformService


class TestDatabaseService:
    """Tests for DatabaseService.get_stats()."""

    async def test_empty_database_stats(self, async_db_session, temp_db):
        """Empty database returns zeros for all counts."""
        service = DatabaseService(async_db_session, profile_id=1)
        stats = await service.get_stats(str(temp_db))

        assert stats.profile_count == 0
        assert stats.device_count == 0
        assert stats.session_count == 0
        assert stats.day_count == 0
        assert stats.event_count == 0
        assert stats.waveform_count == 0
        assert stats.analysis_count == 0
        assert stats.pattern_count == 0
        assert stats.sessions_with_waveforms == 0
        assert stats.sessions_with_events == 0
        assert stats.sessions_with_analysis == 0
        assert stats.analyzable_session_count == 0
        assert stats.waveform_coverage_pct == 0
        assert stats.event_coverage_pct == 0
        assert stats.analysis_coverage_pct == 0
        assert stats.first_session is None
        assert stats.last_session is None
        assert stats.size_mb > 0

    async def test_stats_with_data(self, async_db_session, async_test_device, temp_db):
        """Database with data returns correct counts."""
        now = datetime.now()
        session1 = Session(
            device_id=async_test_device.id,
            device_session_id="test_1",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
            has_waveform_data=True,
            has_event_data=True,
        )
        session2 = Session(
            device_id=async_test_device.id,
            device_session_id="test_2",
            start_time=now + timedelta(days=1),
            end_time=now + timedelta(days=1, hours=7),
            duration_seconds=25200,
            has_waveform_data=False,
            has_event_data=True,
        )
        async_db_session.add(session1)
        async_db_session.add(session2)
        await async_db_session.flush()

        stats1 = Statistics(session_id=session1.id, ahi=2.5, usage_hours=8.0)
        async_db_session.add(stats1)
        await async_db_session.commit()

        service = DatabaseService(async_db_session, profile_id=1)
        stats = await service.get_stats(str(temp_db))

        assert stats.device_count == 1
        assert stats.session_count == 2
        assert stats.sessions_with_waveforms == 1
        assert stats.sessions_with_events == 2
        assert stats.sessions_with_analysis == 0
        assert stats.analyzable_session_count == 0
        assert stats.waveform_coverage_pct == 50.0
        assert stats.event_coverage_pct == 100.0
        assert stats.analysis_coverage_pct == 0.0
        assert stats.first_session is not None
        assert stats.last_session is not None

    async def test_coverage_percentages(
        self, async_db_session, async_test_device, temp_db
    ):
        """Coverage percentages computed correctly."""
        now = datetime.now()
        for i in range(10):
            has_wf = i < 3
            has_ev = i < 7
            session = Session(
                device_id=async_test_device.id,
                device_session_id=f"test_{i}",
                start_time=now + timedelta(days=i),
                end_time=now + timedelta(days=i, hours=8),
                duration_seconds=28800,
                has_waveform_data=has_wf,
                has_event_data=has_ev,
            )
            async_db_session.add(session)
        await async_db_session.commit()

        service = DatabaseService(async_db_session, profile_id=1)
        stats = await service.get_stats(str(temp_db))

        assert stats.session_count == 10
        assert stats.sessions_with_waveforms == 3
        assert stats.sessions_with_events == 7
        assert stats.sessions_with_analysis == 0
        assert stats.analyzable_session_count == 0
        assert stats.waveform_coverage_pct == 30.0
        assert stats.event_coverage_pct == 70.0
        assert stats.analysis_coverage_pct == 0.0

    async def test_file_size_calculation(
        self, async_db_session, async_test_device, temp_db
    ):
        """Database file size is computed correctly."""
        service = DatabaseService(async_db_session, profile_id=1)
        stats = await service.get_stats(str(temp_db))

        assert stats.size_mb > 0
        assert stats.db_path == str(temp_db)

    async def test_nonexistent_file_size_zero(self, async_db_session):
        """Nonexistent database path returns 0 size."""
        service = DatabaseService(async_db_session, profile_id=1)
        fake_path = "/nonexistent/path/database.db"
        stats = await service.get_stats(fake_path)

        assert stats.size_mb == 0
        assert stats.db_path == fake_path

    async def test_analyzed_twice_counts_once_coverage_below_100(
        self, async_db_session, async_test_device, async_test_profile, temp_db
    ):
        """A session re-analyzed twice counts as one analyzed session; coverage stays ≤ 100%."""
        now = datetime.now()
        session = Session(
            device_id=async_test_device.id,
            device_session_id="reanalyzed",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        waveform = Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=25.0,
            data_blob=b"\x00",
        )
        async_db_session.add(waveform)
        await async_db_session.flush()

        # Two analysis results for the same session (re-analysis scenario).
        for _ in range(2):
            result = AnalysisResult(
                session_id=session.id,
                timestamp_start=now,
                timestamp_end=now + timedelta(hours=8),
            )
            async_db_session.add(result)
        await async_db_session.commit()

        service = DatabaseService(async_db_session, profile_id=async_test_profile.id)
        stats = await service.get_stats(str(temp_db))

        assert stats.sessions_with_analysis == 1
        assert stats.analyzable_session_count == 1
        assert stats.analysis_coverage_pct == 100.0

    async def test_non_flow_session_excluded_from_denominator_shows_full_coverage(
        self, async_db_session, async_test_device, async_test_profile, temp_db
    ):
        """Session with no flow waveform is excluded from the analyzable denominator."""
        now = datetime.now()
        # session1 has a flow waveform and an analysis result.
        session1 = Session(
            device_id=async_test_device.id,
            device_session_id="with_flow",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        # session2 has only a pressure waveform — not analyzable.
        session2 = Session(
            device_id=async_test_device.id,
            device_session_id="pressure_only",
            start_time=now + timedelta(days=1),
            end_time=now + timedelta(days=1, hours=8),
            duration_seconds=28800,
        )
        async_db_session.add_all([session1, session2])
        await async_db_session.flush()

        flow_waveform = Waveform(
            session_id=session1.id,
            waveform_type="flow",
            sample_rate=25.0,
            data_blob=b"\x00",
        )
        pressure_waveform = Waveform(
            session_id=session2.id,
            waveform_type="pressure",
            sample_rate=2.0,
            data_blob=b"\x00",
        )
        async_db_session.add_all([flow_waveform, pressure_waveform])
        await async_db_session.flush()

        analysis = AnalysisResult(
            session_id=session1.id,
            timestamp_start=now,
            timestamp_end=now + timedelta(hours=8),
        )
        async_db_session.add(analysis)
        await async_db_session.commit()

        service = DatabaseService(async_db_session, profile_id=async_test_profile.id)
        stats = await service.get_stats(str(temp_db))

        assert stats.session_count == 2
        assert stats.sessions_with_analysis == 1
        # Only session1 is analyzable; session2 (pressure waveform only) is excluded.
        assert stats.analyzable_session_count == 1
        assert stats.analysis_coverage_pct == 100.0

    async def test_zero_analyzable_sessions_coverage_is_zero(
        self, async_db_session, async_test_device, async_test_profile, temp_db
    ):
        """Zero analyzable sessions produces coverage of 0, not a division error."""
        now = datetime.now()
        # Sessions exist but none have a flow waveform.
        session = Session(
            device_id=async_test_device.id,
            device_session_id="no_flow",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.commit()

        service = DatabaseService(async_db_session, profile_id=async_test_profile.id)
        stats = await service.get_stats(str(temp_db))

        assert stats.session_count == 1
        assert stats.analyzable_session_count == 0
        assert stats.sessions_with_analysis == 0
        assert stats.analysis_coverage_pct == 0

    async def test_profile_isolation_new_fields(
        self, async_db_session, async_test_device, async_test_profile, temp_db
    ):
        """sessions_with_analysis and analyzable_session_count are scoped to the active profile."""
        import uuid

        now = datetime.now()

        # Profile 1 session: has flow waveform + analysis result.
        session1 = Session(
            device_id=async_test_device.id,
            device_session_id="profile1_session",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session1)
        await async_db_session.flush()

        waveform1 = Waveform(
            session_id=session1.id,
            waveform_type="flow",
            sample_rate=25.0,
            data_blob=b"\x00",
        )
        async_db_session.add(waveform1)
        await async_db_session.flush()

        analysis1 = AnalysisResult(
            session_id=session1.id,
            timestamp_start=now,
            timestamp_end=now + timedelta(hours=8),
        )
        async_db_session.add(analysis1)
        await async_db_session.flush()

        # Profile 2: separate user, profile, device, session with its own waveform/result.
        user2 = User(
            canonical_email=f"other_{uuid.uuid4().hex[:8]}@example.com",
            role="member",
        )
        async_db_session.add(user2)
        await async_db_session.flush()

        profile2 = Profile(user_id=user2.id, name="Other Profile")
        async_db_session.add(profile2)
        await async_db_session.flush()

        device2 = Device(
            profile_id=profile2.id,
            manufacturer="Other",
            model="Other Model",
            serial_number=f"OTHER_{uuid.uuid4().hex[:8]}",
        )
        async_db_session.add(device2)
        await async_db_session.flush()

        session2 = Session(
            device_id=device2.id,
            device_session_id="profile2_session",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session2)
        await async_db_session.flush()

        waveform2 = Waveform(
            session_id=session2.id,
            waveform_type="flow",
            sample_rate=25.0,
            data_blob=b"\x00",
        )
        async_db_session.add(waveform2)
        await async_db_session.flush()

        analysis2 = AnalysisResult(
            session_id=session2.id,
            timestamp_start=now,
            timestamp_end=now + timedelta(hours=8),
        )
        async_db_session.add(analysis2)
        await async_db_session.commit()

        service = DatabaseService(async_db_session, profile_id=async_test_profile.id)
        stats = await service.get_stats(str(temp_db))

        # Only profile 1's session is counted.
        assert stats.session_count == 1
        assert stats.sessions_with_analysis == 1
        assert stats.analyzable_session_count == 1
        assert stats.analysis_coverage_pct == 100.0


class TestDeviceServiceListDevices:
    """Tests for DeviceService.list_devices() — device listing moved from DatabaseService."""

    async def test_list_devices_empty(self, async_db_session):
        """Empty database returns empty list."""
        service = DeviceService(async_db_session, profile_id=1)
        result = await service.list_devices()

        assert len(result) == 0

    async def test_list_devices_with_data(self, async_db_session, async_test_profile):
        """Returns correct device data."""
        device1 = Device(
            profile_id=async_test_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
            serial_number="TEST001",
        )
        device2 = Device(
            profile_id=async_test_profile.id,
            manufacturer="Philips",
            model="DreamStation",
            serial_number="TEST002",
        )
        device3 = Device(
            profile_id=async_test_profile.id,
            manufacturer="ResMed",
            model="AirCurve 10",
            serial_number="TEST003",
        )
        async_db_session.add_all([device1, device2, device3])
        await async_db_session.commit()

        service = DeviceService(async_db_session, profile_id=1)
        result = await service.list_devices()

        assert len(result) == 3
        assert result[0].manufacturer == "Philips"
        assert result[0].model == "DreamStation"
        assert result[1].manufacturer == "ResMed"
        assert result[1].model == "AirCurve 10"
        assert result[2].manufacturer == "ResMed"
        assert result[2].model == "AirSense 10"

    async def test_list_devices_ordering(self, async_db_session, async_test_profile):
        """Devices are ordered by manufacturer then model."""
        device1 = Device(
            profile_id=async_test_profile.id,
            manufacturer="Philips",
            model="DreamStation 2",
            serial_number="TEST001",
        )
        device2 = Device(
            profile_id=async_test_profile.id,
            manufacturer="ResMed",
            model="AirSense 11",
            serial_number="TEST002",
        )
        device3 = Device(
            profile_id=async_test_profile.id,
            manufacturer="Philips",
            model="DreamStation",
            serial_number="TEST003",
        )
        device4 = Device(
            profile_id=async_test_profile.id,
            manufacturer="ResMed",
            model="AirCurve 10",
            serial_number="TEST004",
        )
        async_db_session.add_all([device1, device2, device3, device4])
        await async_db_session.commit()

        service = DeviceService(async_db_session, profile_id=1)
        result = await service.list_devices()

        assert len(result) == 4
        assert result[0].manufacturer == "Philips"
        assert result[0].model == "DreamStation"
        assert result[1].manufacturer == "Philips"
        assert result[1].model == "DreamStation 2"
        assert result[2].manufacturer == "ResMed"
        assert result[2].model == "AirCurve 10"
        assert result[3].manufacturer == "ResMed"
        assert result[3].model == "AirSense 11"

    async def test_list_devices_includes_all_fields(
        self, async_db_session, async_test_profile
    ):
        """Returns all identity fields including the new firmware/hardware/product_code."""
        device = Device(
            profile_id=async_test_profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
            serial_number="12345ABC",
            firmware_version="SX567-0401",
            hardware_version="R003",
            product_code="37037",
        )
        async_db_session.add(device)
        await async_db_session.commit()

        service = DeviceService(async_db_session, profile_id=1)
        result = await service.list_devices()

        assert len(result) == 1
        assert result[0].id == device.id
        assert result[0].manufacturer == "ResMed"
        assert result[0].model == "AirSense 10"
        assert result[0].serial_number == "12345ABC"
        assert result[0].firmware_version == "SX567-0401"
        assert result[0].hardware_version == "R003"
        assert result[0].product_code == "37037"
        assert result[0].first_seen is not None


class TestVacuum:
    async def test_vacuum_returns_success_status(self, async_db_session, temp_db):
        service = DatabaseService(async_db_session, profile_id=1)
        result = service.vacuum_sqlite(str(temp_db))
        assert result.status == "success"


class TestVacuumBackgroundMarker:
    """_vacuum_background must remove the pending-vacuum marker on success.

    Uses _REAL_VACUUM_BG (module-level import) to bypass the autouse
    _disable_background_vacuum fixture that replaces _vacuum_background with a
    no-op in all tests.
    """

    def test_removes_marker_after_successful_vacuum(self, tmp_path, temp_db):
        """Marker file is deleted after _vacuum_background runs successfully."""
        import sqlite3 as _sqlite3

        from unittest.mock import patch

        # temp_db is created by the fixture; sqlite3 needs the tables to exist
        # but VACUUM works on any valid SQLite file regardless.
        conn = _sqlite3.connect(str(temp_db))
        conn.close()

        marker = tmp_path / "vacuum.pending"
        marker.write_text(str(temp_db))

        with patch("snore.constants.DEFAULT_VACUUM_PENDING_MARKER", marker):
            _REAL_VACUUM_BG(str(temp_db))

        assert not marker.exists(), "Marker must be removed after successful VACUUM"

    def test_marker_survives_vacuum_failure(self, tmp_path):
        """Marker is left in place when VACUUM fails (corrupt SQLite file)."""
        from unittest.mock import patch

        # Create a file with non-SQLite content so sqlite3 raises DatabaseError.
        corrupt_db = tmp_path / "corrupt.db"
        corrupt_db.write_bytes(b"not a sqlite database -- corrupt content")

        marker = tmp_path / "vacuum.pending"
        marker.write_text(str(corrupt_db))

        with patch("snore.constants.DEFAULT_VACUUM_PENDING_MARKER", marker):
            _REAL_VACUUM_BG(str(corrupt_db))

        assert marker.exists(), "Marker must remain when VACUUM fails"

    def test_vacuum_with_no_marker_does_not_raise(self, tmp_path, temp_db):
        """_vacuum_background succeeds even when no marker file is present."""
        import sqlite3 as _sqlite3

        from unittest.mock import patch

        conn = _sqlite3.connect(str(temp_db))
        conn.close()

        absent_marker = tmp_path / "nonexistent.pending"

        with patch("snore.constants.DEFAULT_VACUUM_PENDING_MARKER", absent_marker):
            _REAL_VACUUM_BG(str(temp_db))  # must not raise


class TestDeleteUserDataCommitFailure:
    """delete_user_data must restore quarantined dirs when the commit fails."""

    async def test_commit_failure_restores_dirs(self, async_db_session, tmp_path):
        """Quarantined dirs are renamed back when db.commit() raises."""
        import uuid

        from unittest.mock import AsyncMock

        from snore.database.models import Profile, User
        from snore.services.database_service import DatabaseService

        # Seed a user + profile.
        user = User(
            canonical_email=f"test_{uuid.uuid4().hex[:8]}@example.com",
            role="member",
        )
        async_db_session.add(user)
        await async_db_session.flush()
        profile = Profile(user_id=user.id, name="Test")
        async_db_session.add(profile)
        await async_db_session.flush()

        raw_root = tmp_path / "raw"
        profile_dir = raw_root / str(profile.id)
        profile_dir.mkdir(parents=True)
        (profile_dir / "backup.edf").write_text("precious data")

        # Patch commit to raise.
        orig_commit = async_db_session.commit
        async_db_session.commit = AsyncMock(
            side_effect=RuntimeError("simulated commit failure")
        )
        try:
            with pytest.raises(RuntimeError, match="simulated commit failure"):
                await DatabaseService.delete_user_data(
                    async_db_session, user.id, raw_root
                )
        finally:
            async_db_session.commit = orig_commit

        # Dir must be restored to its original location.
        assert profile_dir.exists(), "Raw dir must be restored after commit failure"
        assert (profile_dir / "backup.edf").read_text() == "precious data"
        quarantine_dst = raw_root / ".quarantine" / str(profile.id)
        assert not quarantine_dst.exists(), (
            "Quarantine dst must be cleaned up on restore"
        )

    async def test_successful_commit_quarantine_removed(
        self, async_db_session, tmp_path
    ):
        """After a successful commit, the quarantined dir is rmtree'd."""
        import uuid

        from snore.database.models import Profile, User
        from snore.services.database_service import DatabaseService

        user = User(
            canonical_email=f"test_{uuid.uuid4().hex[:8]}@example.com",
            role="member",
        )
        async_db_session.add(user)
        await async_db_session.flush()
        profile = Profile(user_id=user.id, name="Test")
        async_db_session.add(profile)
        await async_db_session.flush()

        raw_root = tmp_path / "raw"
        profile_dir = raw_root / str(profile.id)
        profile_dir.mkdir(parents=True)
        (profile_dir / "backup.edf").write_text("data")

        await DatabaseService.delete_user_data(async_db_session, user.id, raw_root)

        # Both src and quarantine dst must be gone.
        assert not profile_dir.exists()
        quarantine_dst = raw_root / ".quarantine" / str(profile.id)
        assert not quarantine_dst.exists()

    async def test_quarantine_survives_post_commit_rmtree_failure(
        self, async_db_session, tmp_path
    ):
        """When the post-commit rmtree raises, the quarantine dir remains for startup recovery."""
        import uuid

        from unittest.mock import patch

        from snore.database.models import Profile, User
        from snore.services.database_service import DatabaseService

        user = User(
            canonical_email=f"test_{uuid.uuid4().hex[:8]}@example.com",
            role="member",
        )
        async_db_session.add(user)
        await async_db_session.flush()
        profile = Profile(user_id=user.id, name="Test")
        async_db_session.add(profile)
        await async_db_session.flush()

        raw_root = tmp_path / "raw"
        profile_dir = raw_root / str(profile.id)
        profile_dir.mkdir(parents=True)
        (profile_dir / "backup.edf").write_text("data")

        # Simulate a crash/error in the post-commit rmtree step.
        with patch(
            "snore.services.database_service.shutil.rmtree",
            side_effect=OSError("disk error"),
        ):
            # Should complete without raising (rmtree failure is best-effort).
            result = await DatabaseService.delete_user_data(
                async_db_session, user.id, raw_root
            )

        assert result[0] >= 0  # devices_deleted (may be 0 if no devices seeded)
        # The quarantine dir must still exist — startup recovery will purge it.
        quarantine_dst = raw_root / ".quarantine" / str(profile.id)
        assert quarantine_dst.exists(), "Quarantine dir must remain when rmtree fails"
        assert not profile_dir.exists(), "Source must be gone (renamed to quarantine)"


class TestDeleteUserDataWaveformCache:
    """delete_user_data must not leave a user's arrays in process memory."""

    async def test_delete_drops_cached_arrays(
        self, async_db_session, async_test_profile, async_test_device, tmp_path
    ):
        """Arrays cached before the delete are gone from the cache afterwards."""
        now = datetime(2025, 1, 1)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="cached",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()
        async_db_session.add(
            Waveform(
                session_id=session.id,
                waveform_type="flow",
                sample_rate=25.0,
                unit="L/min",
                sample_count=10,
                data_blob=bytes(10 * 8),  # 10 (timestamp, value) float32 pairs
            )
        )
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=async_test_profile.id)
        await service.get_waveform_data(session.id, "flow")
        cache = waveform_service_module._waveform_array_cache
        assert len(cache._entries) == 1

        await DatabaseService.delete_user_data(
            async_db_session, async_test_profile.user_id, tmp_path / "raw"
        )

        assert len(cache._entries) == 0


class TestStartupVacuumRecovery:
    """DeletionSaga.recover() case 2 sweeps quarantine dirs left by reset/delete-data."""

    async def test_recover_sweeps_integer_named_quarantine_dirs_without_tombstone(
        self, async_db_session, tmp_path
    ):
        """Case 2: quarantine dirs whose profile_id has no tombstone are purged on startup."""
        import asyncio
        import uuid

        from snore.database.models import Profile, User
        from snore.services.profile_service import DeletionSaga

        # Seed a live profile (no tombstone / deleting_at IS NULL).
        user = User(
            canonical_email=f"test_{uuid.uuid4().hex[:8]}@example.com",
            role="member",
        )
        async_db_session.add(user)
        await async_db_session.flush()
        profile = Profile(user_id=user.id, name="Test")
        async_db_session.add(profile)
        await async_db_session.flush()
        await async_db_session.commit()

        raw_root = tmp_path / "raw"

        # Simulate the "crash between quarantine and commit" state: the quarantine
        # dir exists but the profile row survived (commit didn't happen).
        quarantine_dst = raw_root / ".quarantine" / str(profile.id)
        quarantine_dst.mkdir(parents=True)
        (quarantine_dst / "orphan.edf").write_text("orphaned")

        # Patch session_scope so DeletionSaga uses our test session.
        from contextlib import asynccontextmanager
        from unittest.mock import patch

        @asynccontextmanager
        async def _mock_session_scope():
            yield async_db_session

        with patch("snore.services.profile_service.session_scope", _mock_session_scope):
            saga = DeletionSaga(raw_root=raw_root)
            await asyncio.to_thread(saga.recover)

        # Startup recovery (case 2) must have purged the orphaned quarantine dir.
        assert not quarantine_dst.exists(), (
            "DeletionSaga.recover() case 2 must purge quarantine dir with no tombstone"
        )
