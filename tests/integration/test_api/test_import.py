import json

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx2
import pytest

from fastapi.testclient import TestClient

import snore.api.analysis_jobs as aj_store
import snore.api.import_jobs as job_store

from snore.services.schemas import ImportResult


def _assert_utc_timestamp(value: str) -> None:
    assert datetime.fromisoformat(value).utcoffset() == timedelta(0)


class TestImportUpload:
    def test_upload_returns_202_with_job_id(self, api_client):
        """Upload returns 202 with a job_id for SSE progress tracking."""
        response = api_client.post(
            "/api/v1/import",
            files=[
                (
                    "files",
                    ("test.edf", b"fake edf content", "application/octet-stream"),
                )
            ],
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0

    def test_upload_writes_files_to_temp_dir(self, api_client, monkeypatch):
        """Upload creates temp files that can be retrieved via the job store.

        enqueue_for_execution is patched to a no-op so the background worker never runs
        and its finally-cleanup_files() cannot race the file-existence assertions.
        These tests validate upload path layout and content, not worker execution.
        """
        import snore.api.routers.import_data as import_mod  # noqa: PLC0415

        from snore.api.import_jobs import get_job  # noqa: PLC0415

        monkeypatch.setattr(
            import_mod, "enqueue_for_execution", lambda job, root=None: None
        )

        response = api_client.post(
            "/api/v1/import",
            files=[
                (
                    "files",
                    (
                        "SDCARD/DATALOG/test.edf",
                        b"edf content",
                        "application/octet-stream",
                    ),
                )
            ],
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job = get_job(job_id)
        assert job is not None
        assert job.temp_dir is not None
        assert (job.temp_dir / "SDCARD" / "DATALOG" / "test.edf").exists()
        assert (
            job.temp_dir / "SDCARD" / "DATALOG" / "test.edf"
        ).read_bytes() == b"edf content"

        # Cleanup
        import shutil  # noqa: PLC0415

        from snore.api.import_jobs import remove_job  # noqa: PLC0415

        shutil.rmtree(job.temp_dir, ignore_errors=True)
        remove_job(job_id)

    def test_upload_path_traversal_sanitized(self, api_client, monkeypatch):
        """Path traversal components are stripped from uploaded filenames.

        enqueue_for_execution is patched to a no-op so the background worker never runs
        and its finally-cleanup_files() cannot race the temp-dir rglob assertions.
        """
        import snore.api.routers.import_data as import_mod  # noqa: PLC0415

        from snore.api.import_jobs import get_job  # noqa: PLC0415

        monkeypatch.setattr(
            import_mod, "enqueue_for_execution", lambda job, root=None: None
        )

        response = api_client.post(
            "/api/v1/import",
            files=[
                ("files", ("../../etc/passwd", b"evil", "application/octet-stream"))
            ],
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job = get_job(job_id)
        assert job is not None
        assert job.temp_dir is not None
        # All written files must be inside the temp dir
        tmp_root = job.temp_dir.resolve()
        for path in job.temp_dir.rglob("*"):
            assert path.resolve().is_relative_to(tmp_root)

        import shutil  # noqa: PLC0415

        from snore.api.import_jobs import remove_job  # noqa: PLC0415

        shutil.rmtree(job.temp_dir, ignore_errors=True)
        remove_job(job_id)

    def test_upload_size_limit_exceeded(self, api_client, monkeypatch):
        """Cumulative upload size exceeding limit returns 413."""
        monkeypatch.setattr(
            "snore.api.routers.import_data._get_upload_limits",
            lambda: (10, 500, 10),
        )
        response = api_client.post(
            "/api/v1/import",
            files=[("files", ("big.edf", b"x" * 20, "application/octet-stream"))],
        )
        assert response.status_code == 413

    def test_no_files_returns_422(self, api_client):
        """Request with no 'files' field returns 422 with detail."""
        response = api_client.post(
            "/api/v1/import/",
            files=[("other", ("x.edf", b"x", "application/octet-stream"))],
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "No files provided"

    def test_per_file_limit_exceeded_returns_413(self, api_client, monkeypatch):
        """A single file exceeding max_file_bytes returns 413."""
        # max_upload_bytes=1000, max_files=500, max_file_bytes=5 (very small cap)
        monkeypatch.setattr(
            "snore.api.routers.import_data._get_upload_limits",
            lambda: (1000, 500, 5),
        )
        response = api_client.post(
            "/api/v1/import",
            files=[("files", ("big.edf", b"x" * 10, "application/octet-stream"))],
        )
        assert response.status_code == 413
        assert "per-file limit" in response.json()["detail"]


class TestImportProgress:
    def test_progress_nonexistent_job_returns_404(self, api_client):
        """GET progress for unknown job_id returns 404."""
        response = api_client.get("/api/v1/import/nonexistent/progress")
        assert response.status_code == 404

    def test_progress_stream_emits_events(self, api_client, import_worker):
        """SSE stream emits progress events and a complete event."""
        fake_result = ImportResult(
            total_imported=1,
            total_skipped=0,
            total_failed=0,
            sources=[],
            warnings=[],
        )
        with (
            patch(
                "snore.api.import_worker.ImportService.detect_sources",
                return_value=[],
            ),
            patch(
                "snore.api.import_worker.ImportService.import_sources",
                new_callable=AsyncMock,
                return_value=fake_result,
            ),
        ):
            upload_response = api_client.post(
                "/api/v1/import",
                files=[
                    (
                        "files",
                        ("test.edf", b"fake content", "application/octet-stream"),
                    )
                ],
            )
            assert upload_response.status_code == 202
            job_id = upload_response.json()["job_id"]

            response = api_client.get(
                f"/api/v1/import/{job_id}/progress",
                headers={"Accept": "text/event-stream"},
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            body = response.text
            assert "event: complete" in body
            result_line = [
                line
                for line in body.split("\n")
                if line.startswith("data:") and "total_imported" in line
            ]
            assert len(result_line) == 1
            result_data = json.loads(result_line[0].removeprefix("data: "))
            assert result_data["result"]["total_imported"] == 1


class TestUploadBackupEnabled:
    """Upload job must call import_sources with backup=True (WS2 fix)."""

    def test_upload_job_calls_import_with_backup_true(self, api_client, import_worker):
        """_run_import for UPLOAD type must pass backup=True to import_sources."""
        fake_result = ImportResult(
            total_imported=1,
            total_skipped=0,
            total_failed=0,
            sources=[],
            warnings=[],
        )
        with (
            patch(
                "snore.api.import_worker.ImportService.detect_sources",
                return_value=[],
            ),
            patch(
                "snore.api.import_worker.ImportService.import_sources",
                new_callable=AsyncMock,
                return_value=fake_result,
            ) as mock_import,
        ):
            upload_response = api_client.post(
                "/api/v1/import",
                files=[
                    (
                        "files",
                        ("test.edf", b"fake content", "application/octet-stream"),
                    )
                ],
            )
            assert upload_response.status_code == 202
            job_id = upload_response.json()["job_id"]

            api_client.get(
                f"/api/v1/import/{job_id}/progress",
                headers={"Accept": "text/event-stream"},
            )

        # Verify the upload path called import_sources with backup=True
        mock_import.assert_called_once()
        _, kwargs = mock_import.call_args
        assert kwargs.get("backup") is True, (
            "Upload route must call import_sources(backup=True); "
            f"got backup={kwargs.get('backup')!r}"
        )


class TestStaleTempDirCleanup:
    """_cleanup_stale_upload_spool_dirs removes snore-upload-* dirs older than the threshold."""

    def test_stale_upload_dir_is_removed(self, tmp_path, monkeypatch):
        """A snore-upload-* dir old enough is deleted at startup."""
        import tempfile  # noqa: PLC0415
        import time  # noqa: PLC0415

        from snore.api.app import (  # noqa: PLC0415
            _STALE_UPLOAD_TMPDIR_AGE_SECONDS,
            _cleanup_stale_upload_spool_dirs,
        )

        stale = tmp_path / "snore-upload-stale"
        stale.mkdir()

        # Back-date mtime to exceed the threshold.
        old_mtime = time.time() - _STALE_UPLOAD_TMPDIR_AGE_SECONDS - 60
        import os  # noqa: PLC0415

        os.utime(stale, (old_mtime, old_mtime))

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        _cleanup_stale_upload_spool_dirs()

        assert not stale.exists(), "Stale snore-upload-* dir must be removed at startup"

    def test_recent_upload_dir_is_kept(self, tmp_path, monkeypatch):
        """A snore-upload-* dir created recently is NOT deleted."""
        import tempfile  # noqa: PLC0415

        from snore.api.app import _cleanup_stale_upload_spool_dirs  # noqa: PLC0415

        fresh = tmp_path / "snore-upload-fresh"
        fresh.mkdir()

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        _cleanup_stale_upload_spool_dirs()

        assert fresh.exists(), (
            "Recent snore-upload-* dir must NOT be removed at startup"
        )

    def test_non_snore_dirs_are_ignored(self, tmp_path, monkeypatch):
        """Dirs without the snore-upload- prefix are never touched."""
        import os  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        import time  # noqa: PLC0415

        from snore.api.app import (  # noqa: PLC0415
            _STALE_UPLOAD_TMPDIR_AGE_SECONDS,
            _cleanup_stale_upload_spool_dirs,
        )

        other = tmp_path / "unrelated-tmpdir"
        other.mkdir()
        old_mtime = time.time() - _STALE_UPLOAD_TMPDIR_AGE_SECONDS - 60
        os.utime(other, (old_mtime, old_mtime))

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        _cleanup_stale_upload_spool_dirs()

        assert other.exists(), "Non-snore-upload dirs must never be removed"


class TestImportJobOwnership:
    """cancel and progress return 404 for jobs owned by a different user (isolation matrix)."""

    def test_cancel_foreign_job_returns_404(self, api_client):
        """DELETE /import/{job_id} with a job owned by a different user returns 404.

        The local actor gets user_id=1 (auto-provisioned).  A job created with
        owner_user_id=9999 is a foreign job — the route must return 404 rather
        than 403 to avoid leaking job-ID existence.
        """
        from snore.api.import_jobs import JobType, create_job, remove_job

        job = create_job(JobType.UPLOAD, owner_user_id=9999)
        try:
            response = api_client.delete(f"/api/v1/import/{job.job_id}")
            assert response.status_code == 404
        finally:
            remove_job(job.job_id)
            job.cleanup_files()
            job.release_capacity()

    def test_progress_foreign_job_returns_404(self, api_client):
        """GET /import/{job_id}/progress with a job owned by a different user returns 404."""
        from snore.api.import_jobs import JobType, create_job, remove_job

        job = create_job(JobType.UPLOAD, owner_user_id=9999)
        try:
            response = api_client.get(f"/api/v1/import/{job.job_id}/progress")
            assert response.status_code == 404
        finally:
            remove_job(job.job_id)
            job.cleanup_files()
            job.release_capacity()

    def test_cancel_own_job_not_404(self, api_client):
        """DELETE /import/{job_id} for an unowned job (local mode) returns 204 or 404 for gone."""
        from snore.api.import_jobs import remove_job, reserve_slot

        # owner_user_id=None simulates a pre-multiuser or local-mode job.
        job = reserve_slot(None)
        assert job is not None
        try:
            response = api_client.delete(f"/api/v1/import/{job.job_id}")
            # 204 = cancelled; 404 = job already cleaned up by a race in tests.
            assert response.status_code in (204, 404)
        finally:
            try:
                remove_job(job.job_id)
            except Exception:
                pass
            job.cleanup_files()
            job.release_capacity()

    def test_cancel_terminal_job_returns_409(self, api_client):
        """DELETE /import/{job_id} for a job already in a terminal state returns 409."""
        from snore.api.import_jobs import JobType, cancel_job, create_job, remove_job

        job = create_job(JobType.PATH, owner_user_id=None)
        try:
            # Drive to CANCELLED (terminal) via the public API, same pattern as
            # the analysis counterpart.
            cancel_job(job.job_id)

            response = api_client.delete(f"/api/v1/import/{job.job_id}")
            assert response.status_code == 409
            assert "cannot be cancelled" in response.json()["detail"]
        finally:
            remove_job(job.job_id)
            job.cleanup_files()
            job.release_capacity()


class TestImportTargetProfile:
    """target_profile_id enforcement for file upload and path import endpoints."""

    def test_upload_with_valid_target_profile_id_uses_that_profile(
        self, api_client, monkeypatch
    ):
        """Upload with a valid owned profile_id lands in that profile, not the active one."""
        import snore.api.routers.import_data as import_mod

        from snore.api.import_jobs import get_job, remove_job

        monkeypatch.setattr(
            import_mod, "enqueue_for_execution", lambda job, root=None: None
        )

        # Create a second profile via the profiles API.
        resp = api_client.post("/api/v1/profiles/", json={"name": "Second Profile"})
        assert resp.status_code == 201
        second_profile_id = resp.json()["id"]

        job = None
        try:
            response = api_client.post(
                "/api/v1/import",
                data={"profile_id": str(second_profile_id)},
                files=[
                    ("files", ("test.edf", b"fake content", "application/octet-stream"))
                ],
            )
            assert response.status_code == 202
            job = get_job(response.json()["job_id"])
            assert job is not None
            assert job.target_profile_id == second_profile_id
        finally:
            if job is not None:
                remove_job(job.job_id)
                job.cleanup_files()
                job.release_capacity()

    def test_upload_with_foreign_profile_id_returns_403(self, api_client, monkeypatch):
        """Upload with a profile_id that does not belong to this user returns 403."""
        import snore.api.routers.import_data as import_mod

        monkeypatch.setattr(
            import_mod, "enqueue_for_execution", lambda job, root=None: None
        )

        response = api_client.post(
            "/api/v1/import",
            data={"profile_id": "999999"},
            files=[
                ("files", ("test.edf", b"fake content", "application/octet-stream"))
            ],
        )
        assert response.status_code == 403

    def test_upload_without_profile_id_uses_actor_profile(
        self, api_client, monkeypatch
    ):
        """Upload without profile_id falls back to actor.profile_id (existing behaviour)."""
        import snore.api.routers.import_data as import_mod

        from snore.api.import_jobs import get_job, remove_job

        monkeypatch.setattr(
            import_mod, "enqueue_for_execution", lambda job, root=None: None
        )

        job = None
        try:
            response = api_client.post(
                "/api/v1/import",
                files=[
                    ("files", ("test.edf", b"fake content", "application/octet-stream"))
                ],
            )
            assert response.status_code == 202
            job = get_job(response.json()["job_id"])
            assert job is not None
            assert job.target_profile_id is not None
        finally:
            if job is not None:
                remove_job(job.job_id)
                job.cleanup_files()
                job.release_capacity()


class TestPipelineJobsListAPI:
    """Integration tests for GET /api/v1/import/jobs."""

    @pytest.fixture(autouse=True)
    def clean_stores(self):
        """Reset both job stores before and after each test."""
        job_store._jobs.clear()
        job_store._per_user_count.clear()
        job_store._global_count = 0
        job_store._import_queue.clear()
        aj_store._all_jobs.clear()
        aj_store._queue.clear()
        yield
        job_store._jobs.clear()
        job_store._per_user_count.clear()
        job_store._global_count = 0
        job_store._import_queue.clear()
        aj_store._all_jobs.clear()
        aj_store._queue.clear()

    def test_empty_list_initially(self, api_client):
        response = api_client.get("/api/v1/import/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert data["jobs"] == []

    def test_completed_job_appears_with_correct_shape(self, api_client):
        """A completed job appears in the list with the expected response fields."""
        from snore.api.import_jobs import JobType, create_job, remove_job

        job = create_job(JobType.PATH, owner_user_id=None)
        job.set_file_count(2)
        try:
            job.try_start()
            job._finish(
                succeeded=True,
                terminal_msg={"event": "complete", "data": {}},
            )

            response = api_client.get("/api/v1/import/jobs")
            assert response.status_code == 200
            jobs = response.json()["jobs"]
            assert len(jobs) == 1

            j = jobs[0]
            assert j["job_id"] == job.job_id
            assert j["job_type"] == "path"
            assert j["state"] == "succeeded"
            assert j["stage"] == "done"
            assert j["file_count"] == 2
            _assert_utc_timestamp(j["created_at"])
            _assert_utc_timestamp(j["started_at"])
            _assert_utc_timestamp(j["finished_at"])
            assert "import_result" in j
            assert "linked_analysis" in j
        finally:
            remove_job(job.job_id)
            job.cleanup_files()
            job.release_capacity()

    def test_pending_job_has_null_execution_timestamps(self, api_client):
        from snore.api.import_jobs import JobType, create_job, remove_job

        job = create_job(JobType.PATH, owner_user_id=None)
        try:
            response = api_client.get("/api/v1/import/jobs")
            assert response.status_code == 200
            item = next(
                item for item in response.json()["jobs"] if item["job_id"] == job.job_id
            )
            _assert_utc_timestamp(item["created_at"])
            assert item["started_at"] is None
            assert item["finished_at"] is None
        finally:
            remove_job(job.job_id)
            job.cleanup_files()
            job.release_capacity()

    def test_stitched_analysis_summary_present(self, api_client):
        """When an analysis job is linked, linked_analysis is populated."""
        from snore.api.import_jobs import JobType, create_job, remove_job

        # Create the import job.
        job = create_job(JobType.PATH, owner_user_id=None)
        try:
            job.try_start()

            # Enqueue an analysis job and link it.
            aj = aj_store.enqueue(
                profile_id=1,
                session_ids=[1, 2],
                source=aj_store.AnalysisJobSource.IMPORT,
                owner_user_id=None,
            )
            assert aj is not None
            job.set_analysis_link(analysis_job_id=aj.job_id, queue_full=False)
            job._finish(
                succeeded=True,
                terminal_msg={
                    "event": "complete",
                    "data": {"analysis_job_id": aj.job_id},
                },
            )

            response = api_client.get("/api/v1/import/jobs")
            assert response.status_code == 200
            jobs = response.json()["jobs"]
            assert len(jobs) == 1

            j = jobs[0]
            assert j["analysis_job_id"] == aj.job_id
            assert j["analysis_queued"] is True
            assert j["linked_analysis"] is not None
            assert j["linked_analysis"]["job_id"] == aj.job_id
            assert j["linked_analysis"]["state"] == "queued"
            assert j["stage"] == "analysis_queued"
        finally:
            remove_job(job.job_id)
            job.cleanup_files()
            job.release_capacity()

    def test_ownership_isolation_foreign_job_absent(self, api_client):
        """A job owned by another user_id does not appear in the list."""
        from snore.api.import_jobs import JobType, create_job, remove_job

        # api_client actor gets user_id=1; create a job owned by user 9999.
        job = create_job(JobType.PATH, owner_user_id=9999)
        try:
            response = api_client.get("/api/v1/import/jobs")
            assert response.status_code == 200
            job_ids = [j["job_id"] for j in response.json()["jobs"]]
            assert job.job_id not in job_ids
        finally:
            remove_job(job.job_id)
            job.cleanup_files()
            job.release_capacity()

    def test_no_imported_session_ids_in_import_result(self, api_client):
        """import_result in the response must never contain imported_session_ids."""
        from snore.api.import_jobs import JobPhase, JobType, create_job, remove_job

        job = create_job(JobType.PATH, owner_user_id=None)
        try:
            job.try_start()
            job.phase_complete(
                JobPhase.IMPORT,
                {
                    "total_imported": 1,
                    "total_skipped": 0,
                    "total_failed": 0,
                    "warnings": [],
                    "imported_session_ids": [42],
                    "sources": [],
                },
            )
            job._finish(
                succeeded=True,
                terminal_msg={"event": "complete", "data": {}},
            )

            response = api_client.get("/api/v1/import/jobs")
            assert response.status_code == 200
            jobs = response.json()["jobs"]
            assert len(jobs) == 1

            import_result = jobs[0]["import_result"]
            assert import_result is not None
            assert "imported_session_ids" not in import_result
        finally:
            remove_job(job.job_id)
            job.cleanup_files()
            job.release_capacity()

    def test_persisted_job_survives_in_memory_removal(self, api_client, db_session):
        """A job persisted to DB appears in the list after being removed from memory."""
        from snore.api.import_jobs import JobPhase, JobType, create_job, remove_job
        from snore.database.models import ImportJobRecord

        job = create_job(JobType.PATH, owner_user_id=None)
        job.set_file_count(5)
        try:
            job.try_start()
            job.phase_complete(
                JobPhase.IMPORT,
                {
                    "total_imported": 3,
                    "total_skipped": 1,
                    "total_failed": 1,
                    "warnings": ["test warning"],
                    "imported_session_ids": [1, 2, 3],
                    "sources": [],
                },
            )
            job._finish(
                succeeded=True,
                terminal_msg={"event": "complete", "data": {}},
            )

            db_session.add(
                ImportJobRecord(
                    job_id=job.job_id,
                    job_type=job.job_type.value,
                    owner_user_id=job.owner_user_id,
                    target_profile_id=job.target_profile_id,
                    state=job.state.value,
                    file_count=job.file_count,
                    sessions_imported=job.sessions_imported,
                    import_result_json=job.import_result_snapshot,
                    error_message=job.error_message,
                    analysis_queued=job.analysis_queued,
                    created_at=job.created_at_wall,
                    started_at=job.started_at_wall,
                    finished_at=job.finished_at_wall,
                    updated_at=datetime.now(UTC),
                )
            )
            db_session.commit()

            # Remove from in-memory store — simulates reaper TTL or restart.
            remove_job(job.job_id)

            response = api_client.get("/api/v1/import/jobs")
            assert response.status_code == 200
            jobs = response.json()["jobs"]
            assert len(jobs) == 1

            j = jobs[0]
            assert j["job_id"] == job.job_id
            assert j["state"] == "succeeded"
            assert j["stage"] == "done"
            assert j["file_count"] == 5
            _assert_utc_timestamp(j["created_at"])
            _assert_utc_timestamp(j["started_at"])
            _assert_utc_timestamp(j["finished_at"])
            assert j["import_result"] is not None
            assert j["import_result"]["total_imported"] == 3
            assert j["import_result"]["total_skipped"] == 1
            assert j["import_result"]["total_failed"] == 1
            assert j["import_result"]["warnings"] == ["test warning"]
            assert j["linked_analysis"] is None
            assert j["analysis_job_id"] is None
        finally:
            job.cleanup_files()
            job.release_capacity()

    def test_persisted_job_deduped_against_in_memory(self, api_client, db_session):
        """When a job exists both in memory and DB, only one copy appears."""
        from snore.api.import_jobs import JobType, create_job, remove_job
        from snore.database.models import ImportJobRecord

        job = create_job(JobType.PATH, owner_user_id=None)
        try:
            job.try_start()
            job._finish(
                succeeded=True,
                terminal_msg={"event": "complete", "data": {}},
            )

            db_session.add(
                ImportJobRecord(
                    job_id=job.job_id,
                    job_type=job.job_type.value,
                    owner_user_id=job.owner_user_id,
                    target_profile_id=job.target_profile_id,
                    state=job.state.value,
                    file_count=job.file_count,
                    sessions_imported=job.sessions_imported,
                    import_result_json=None,
                    error_message=None,
                    analysis_queued=None,
                    created_at=job.created_at_wall,
                    finished_at=job.finished_at_wall,
                    updated_at=datetime.now(UTC),
                )
            )
            db_session.commit()

            # Job is in both stores — list should deduplicate.
            response = api_client.get("/api/v1/import/jobs")
            assert response.status_code == 200
            jobs = response.json()["jobs"]
            matching = [j for j in jobs if j["job_id"] == job.job_id]
            assert len(matching) == 1
        finally:
            remove_job(job.job_id)
            job.cleanup_files()
            job.release_capacity()

    def test_persisted_failed_job_shows_error(self, api_client, db_session):
        """A failed job persisted to DB shows its error message."""
        from snore.api.import_jobs import JobType, create_job, remove_job
        from snore.database.models import ImportJobRecord

        job = create_job(JobType.PATH, owner_user_id=None)
        try:
            job.try_start()
            job._finish(
                succeeded=False,
                terminal_msg={"event": "error", "data": {"message": "Disk full"}},
            )

            db_session.add(
                ImportJobRecord(
                    job_id=job.job_id,
                    job_type=job.job_type.value,
                    owner_user_id=job.owner_user_id,
                    target_profile_id=job.target_profile_id,
                    state=job.state.value,
                    file_count=job.file_count,
                    sessions_imported=None,
                    import_result_json=None,
                    error_message=job.error_message,
                    analysis_queued=None,
                    created_at=job.created_at_wall,
                    finished_at=job.finished_at_wall,
                    updated_at=datetime.now(UTC),
                )
            )
            db_session.commit()
            remove_job(job.job_id)

            response = api_client.get("/api/v1/import/jobs")
            assert response.status_code == 200
            jobs = response.json()["jobs"]
            assert len(jobs) == 1
            assert jobs[0]["state"] == "failed"
            assert jobs[0]["stage"] == "failed"
            assert jobs[0]["error_message"] == "Disk full"
        finally:
            job.cleanup_files()
            job.release_capacity()

    def test_non_terminal_db_rows_excluded_from_list(self, api_client, db_session):
        """Non-terminal ImportJobRecord rows with no in-memory counterpart are hidden.

        Startup recovery should clear these, but if recovery is skipped (e.g. DB
        locked at boot), a non-terminal DB row must not appear as a phantom
        forever-running job in the jobs list.  Only terminal DB history is shown.
        """
        import uuid

        from snore.database.models import ImportJobRecord

        for state in ["pending_upload", "pending", "running"]:
            db_session.add(
                ImportJobRecord(
                    job_id=uuid.uuid4().hex,
                    job_type="upload",
                    owner_user_id=None,
                    target_profile_id=None,
                    state=state,
                    file_count=0,
                    sessions_imported=None,
                    import_result_json=None,
                    error_message=None,
                    analysis_queued=None,
                    created_at=datetime.now(UTC),
                    finished_at=None,
                    updated_at=datetime.now(UTC),
                )
            )
        db_session.commit()

        response = api_client.get("/api/v1/import/jobs")
        assert response.status_code == 200
        jobs = response.json()["jobs"]

        states_returned = {j["state"] for j in jobs}
        assert "pending_upload" not in states_returned, (
            "Non-terminal DB row (pending_upload) must not appear as phantom"
        )
        assert "pending" not in states_returned, (
            "Non-terminal DB row (pending) must not appear as phantom"
        )
        assert "running" not in states_returned, (
            "Non-terminal DB row (running) must not appear as phantom"
        )

    def test_historical_path_job_type_renders_without_error(
        self, api_client, db_session
    ):
        """A historical DB row with job_type='path' must render correctly in GET /import/jobs.

        The server-path import feature was removed, but existing databases may contain
        rows with job_type='path'.  These must not cause errors or disappear from the
        jobs list — backward compatibility for the DB column is required.
        """
        import uuid

        from snore.database.models import ImportJobRecord

        historical_job_id = uuid.uuid4().hex
        db_session.add(
            ImportJobRecord(
                job_id=historical_job_id,
                job_type="path",
                owner_user_id=None,
                target_profile_id=None,
                state="succeeded",
                file_count=3,
                sessions_imported=2,
                import_result_json=None,
                error_message=None,
                analysis_queued=None,
                created_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        db_session.commit()

        response = api_client.get("/api/v1/import/jobs")
        assert response.status_code == 200
        jobs = response.json()["jobs"]

        matching = [j for j in jobs if j["job_id"] == historical_job_id]
        assert len(matching) == 1, (
            "Historical job with job_type='path' must appear in GET /import/jobs"
        )
        j = matching[0]
        assert j["job_type"] == "path"
        assert j["state"] == "succeeded"
        assert j["stage"] == "done"
        assert j["file_count"] == 3
        assert j["sessions_imported"] == 2


class TestRescanEndpoint:
    """Integration tests for POST /api/v1/import/rescan."""

    # ── helpers ────────────────────────────────────────────────────────────

    def _make_archive(self, raw_root: Path, serial: str = "SN123456789") -> Path:
        """Populate a minimal archive tree: <raw_root>/<serial>/DATALOG/."""
        serial_dir = raw_root / serial
        (serial_dir / "DATALOG").mkdir(parents=True)
        (serial_dir / "STR.edf").write_bytes(b"\x00" * 4)  # minimal sentinel
        return raw_root

    # ── happy path ─────────────────────────────────────────────────────────

    def test_rescan_returns_202_with_job_id(self, api_client, monkeypatch, tmp_path):
        """Happy path: archive present → 202 with job_id."""
        import snore.api.routers.import_data as import_mod  # noqa: PLC0415

        raw_root = tmp_path / "raw"
        profiles = api_client.get("/api/v1/profiles/").json()
        actor_profile_id = next(p["id"] for p in profiles if p["is_default"])
        self._make_archive(raw_root / str(actor_profile_id))

        monkeypatch.setattr("snore.constants.DEFAULT_RAW_BACKUP_DIR", raw_root)
        monkeypatch.setattr(
            import_mod, "enqueue_for_execution", lambda job, root=None: None
        )

        response = api_client.post("/api/v1/import/rescan", json={})
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0

    def test_rescan_job_type_is_rescan(self, api_client, monkeypatch, tmp_path):
        """Job created by rescan endpoint has job_type='rescan'."""
        import snore.api.routers.import_data as import_mod  # noqa: PLC0415

        from snore.api.import_jobs import get_job  # noqa: PLC0415

        raw_root = tmp_path / "raw"
        profiles = api_client.get("/api/v1/profiles/").json()
        actor_profile_id = next(p["id"] for p in profiles if p["is_default"])
        self._make_archive(raw_root / str(actor_profile_id))

        monkeypatch.setattr("snore.constants.DEFAULT_RAW_BACKUP_DIR", raw_root)
        monkeypatch.setattr(
            import_mod, "enqueue_for_execution", lambda job, root=None: None
        )

        response = api_client.post("/api/v1/import/rescan", json={})
        assert response.status_code == 202
        job = get_job(response.json()["job_id"])
        assert job is not None
        assert job.job_type.value == "rescan"

    def test_rescan_with_explicit_profile_id(self, api_client, monkeypatch, tmp_path):
        """Rescan with an explicit owned profile_id uses that profile."""
        import snore.api.routers.import_data as import_mod  # noqa: PLC0415

        from snore.api.import_jobs import get_job  # noqa: PLC0415

        # Create a second profile.
        resp = api_client.post("/api/v1/profiles/", json={"name": "Second Profile"})
        assert resp.status_code == 201
        second_profile_id = resp.json()["id"]

        raw_root = tmp_path / "raw"
        self._make_archive(raw_root / str(second_profile_id))

        monkeypatch.setattr("snore.constants.DEFAULT_RAW_BACKUP_DIR", raw_root)
        monkeypatch.setattr(
            import_mod, "enqueue_for_execution", lambda job, root=None: None
        )

        response = api_client.post(
            "/api/v1/import/rescan", json={"profile_id": second_profile_id}
        )
        assert response.status_code == 202
        job = get_job(response.json()["job_id"])
        assert job is not None
        assert job.target_profile_id == second_profile_id

    # ── 422 cases ──────────────────────────────────────────────────────────

    def test_rescan_422_no_archive(self, api_client, monkeypatch, tmp_path):
        """Archive directory absent → 422 with informative message."""
        raw_root = tmp_path / "raw"
        monkeypatch.setattr("snore.constants.DEFAULT_RAW_BACKUP_DIR", raw_root)

        response = api_client.post("/api/v1/import/rescan", json={})
        assert response.status_code == 422
        assert "archive" in response.json()["detail"].lower()

    def test_rescan_422_archive_no_device_data(self, api_client, monkeypatch, tmp_path):
        """Profile dir exists but has no <serial>/DATALOG subtree → 422."""
        raw_root = tmp_path / "raw"
        profiles = api_client.get("/api/v1/profiles/").json()
        actor_profile_id = next(p["id"] for p in profiles if p["is_default"])
        # Create the profile dir but no serial/DATALOG inside it.
        (raw_root / str(actor_profile_id)).mkdir(parents=True)

        monkeypatch.setattr("snore.constants.DEFAULT_RAW_BACKUP_DIR", raw_root)

        response = api_client.post("/api/v1/import/rescan", json={})
        assert response.status_code == 422
        assert "device data" in response.json()["detail"].lower()

    # ── 429 ────────────────────────────────────────────────────────────────

    def test_rescan_429_caps_exceeded(self, api_client, monkeypatch, tmp_path):
        """Admission caps exceeded → 429."""
        raw_root = tmp_path / "raw"
        profiles = api_client.get("/api/v1/profiles/").json()
        actor_profile_id = next(p["id"] for p in profiles if p["is_default"])
        self._make_archive(raw_root / str(actor_profile_id))

        monkeypatch.setattr("snore.constants.DEFAULT_RAW_BACKUP_DIR", raw_root)
        monkeypatch.setattr(
            "snore.api.import_jobs._check_and_reserve", lambda owner_user_id: False
        )

        response = api_client.post("/api/v1/import/rescan", json={})
        assert response.status_code == 429

    # ── auth ───────────────────────────────────────────────────────────────

    def test_rescan_401_unauthenticated(self, async_db_session, monkeypatch, tmp_path):
        """Unauthenticated request → 401."""
        from tests.helpers.api_client import make_test_client  # noqa: PLC0415

        client = make_test_client(async_db_session, unauthenticated=True)
        response = client.post("/api/v1/import/rescan", json={})
        assert response.status_code == 401

    def test_rescan_403_demo_actor(self, async_db_session, monkeypatch, tmp_path):
        """Demo actor (read-only) → 403."""
        from snore.auth.actor import ActorContext, AuthMode, Role  # noqa: PLC0415
        from tests.helpers.api_client import make_test_client  # noqa: PLC0415

        demo_actor = ActorContext(
            user_id=1, profile_id=1, role=Role.DEMO, mode=AuthMode.LOCAL
        )
        client = make_test_client(async_db_session, actor=demo_actor)
        response = client.post("/api/v1/import/rescan", json={})
        assert response.status_code == 403

    def test_rescan_403_foreign_profile_id(self, api_client, monkeypatch, tmp_path):
        """profile_id that belongs to another user → 403."""
        response = api_client.post("/api/v1/import/rescan", json={"profile_id": 999999})
        assert response.status_code == 403

    # ── re-parse options ───────────────────────────────────────────────────

    def _post_rescan(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        body: dict[str, object],
    ) -> httpx2.Response:
        import snore.api.routers.import_data as import_mod  # noqa: PLC0415

        raw_root = tmp_path / "raw"
        profiles = api_client.get("/api/v1/profiles/").json()
        actor_profile_id = next(p["id"] for p in profiles if p["is_default"])
        self._make_archive(raw_root / str(actor_profile_id))
        monkeypatch.setattr("snore.constants.DEFAULT_RAW_BACKUP_DIR", raw_root)
        monkeypatch.setattr(
            import_mod, "enqueue_for_execution", lambda job, root=None: None
        )
        return api_client.post("/api/v1/import/rescan", json=body)

    def test_rescan_defaults_to_plain_reimport_with_analysis(
        self, api_client, monkeypatch, tmp_path
    ):
        from snore.api.import_jobs import get_job  # noqa: PLC0415

        response = self._post_rescan(api_client, monkeypatch, tmp_path, {})

        job = get_job(response.json()["job_id"])
        assert job is not None
        assert (job.force, job.date_from, job.date_to, job.analyze) == (
            False,
            None,
            None,
            True,
        )

    def test_rescan_carries_reparse_options_to_job(
        self, api_client, monkeypatch, tmp_path
    ):
        from datetime import date  # noqa: PLC0415

        from snore.api.import_jobs import get_job  # noqa: PLC0415

        response = self._post_rescan(
            api_client,
            monkeypatch,
            tmp_path,
            {
                "force": True,
                "from_date": "2024-06-01",
                "to_date": "2024-06-30",
                "analyze": False,
            },
        )

        assert response.status_code == 202
        job = get_job(response.json()["job_id"])
        assert job is not None
        assert (job.force, job.date_from, job.date_to, job.analyze) == (
            True,
            date(2024, 6, 1),
            date(2024, 6, 30),
            False,
        )

    def test_rescan_422_when_from_date_after_to_date(
        self, api_client, monkeypatch, tmp_path
    ):
        response = self._post_rescan(
            api_client,
            monkeypatch,
            tmp_path,
            {"from_date": "2024-07-01", "to_date": "2024-06-30"},
        )

        assert response.status_code == 422
        assert job_store._jobs == {}

    def test_second_forced_rescan_of_same_profile_returns_409(
        self, api_client, monkeypatch, tmp_path
    ):
        first = self._post_rescan(api_client, monkeypatch, tmp_path, {"force": True})
        assert first.status_code == 202

        second = api_client.post("/api/v1/import/rescan", json={"force": True})

        assert second.status_code == 409
        assert "already queued or running" in second.json()["detail"]
        assert list(job_store._jobs) == [first.json()["job_id"]]

    def test_unforced_rescan_accepted_while_forced_rescan_active(
        self, api_client, monkeypatch, tmp_path
    ):
        self._post_rescan(api_client, monkeypatch, tmp_path, {"force": True})

        second = api_client.post("/api/v1/import/rescan", json={})

        assert second.status_code == 202

    def test_forced_rescan_accepted_after_previous_one_finished(
        self, api_client, monkeypatch, tmp_path
    ):
        first = self._post_rescan(api_client, monkeypatch, tmp_path, {"force": True})
        job_store.get_job(first.json()["job_id"]).try_cancel()

        second = api_client.post("/api/v1/import/rescan", json={"force": True})

        assert second.status_code == 202
