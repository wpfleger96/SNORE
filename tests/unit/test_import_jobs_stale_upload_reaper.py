"""Unit tests for the stale PENDING_UPLOAD reaper and periodic spool sweep.

Verifies that abandoned multi-chunk uploads are reclaimed after the idle timeout
expires, that active uploads are left alone, that capacity counters are correctly
released, that get_live_spool_dirs tracks in-flight jobs, and that the reaper
thread invokes the optional spool_sweep_fn each iteration.
"""

from __future__ import annotations

import os
import threading
import time

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import snore.api.import_jobs as job_store

from snore.api.app import (
    _STALE_UPLOAD_TMPDIR_AGE_SECONDS,
    _cleanup_stale_upload_spool_dirs,
)
from snore.api.import_jobs import (
    PENDING_UPLOAD_TIMEOUT_SECONDS,
    JobState,
    JobType,
    _reap_stale_pending_uploads,
    create_job,
    get_job,
    get_live_spool_dirs,
    reserve_slot,
    start_reaper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expired_wall() -> datetime:
    """Return a wall-clock timestamp just past the stale-upload timeout."""
    return datetime.now(UTC) - timedelta(seconds=PENDING_UPLOAD_TIMEOUT_SECONDS + 1)


def _fresh_wall() -> datetime:
    """Return a wall-clock timestamp safely within the stale-upload timeout."""
    return datetime.now(UTC) - timedelta(seconds=PENDING_UPLOAD_TIMEOUT_SECONDS - 60)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStaleUploadReaper:
    """Stale PENDING_UPLOAD jobs are reaped; active and non-PENDING_UPLOAD jobs are not."""

    def test_reap_after_timeout(self):
        """A PENDING_UPLOAD job past the idle timeout is cancelled and removed."""
        job = reserve_slot(None)
        assert job is not None
        assert job.state == JobState.PENDING_UPLOAD

        job._last_activity_wall = _expired_wall()
        _reap_stale_pending_uploads()

        assert get_job(job.job_id) is None
        assert job.is_terminal  # try_cancel() was called

    def test_not_reaped_before_timeout(self):
        """A PENDING_UPLOAD job within the idle window is left alone."""
        job = reserve_slot(None)
        assert job is not None
        # _last_activity_wall is initialised to now — well within the timeout.
        _reap_stale_pending_uploads()
        assert get_job(job.job_id) is not None

    def test_touch_resets_clock_prevents_reap(self):
        """touch() updates _last_activity_wall so the job is not reaped."""
        job = reserve_slot(None)
        assert job is not None

        # Backdate past timeout, then call touch() to reset.
        job._last_activity_wall = _expired_wall()
        job.touch()

        _reap_stale_pending_uploads()

        # The job should still be in the store because touch() refreshed the clock.
        assert get_job(job.job_id) is not None

    def test_pending_job_not_reaped(self, tmp_path):
        """A PENDING (not PENDING_UPLOAD) job is not reaped even when backdated."""
        d = tmp_path / "pending"
        d.mkdir()
        pending_job = create_job(JobType.UPLOAD, temp_dir=d)
        assert pending_job.state == JobState.PENDING

        # Backdate far past the timeout.
        pending_job._last_activity_wall = _expired_wall()

        _reap_stale_pending_uploads()

        assert get_job(pending_job.job_id) is not None
        assert pending_job.state == JobState.PENDING

    def test_running_job_not_reaped(self, tmp_path):
        """A RUNNING job is not reaped by the stale-upload reaper."""
        d = tmp_path / "running"
        d.mkdir()
        running_job = create_job(JobType.UPLOAD, temp_dir=d)
        running_job.try_start()
        assert running_job.state == JobState.RUNNING

        running_job._last_activity_wall = _expired_wall()

        _reap_stale_pending_uploads()

        assert get_job(running_job.job_id) is not None
        assert running_job.state == JobState.RUNNING

    def test_capacity_released_after_reap(self):
        """After reaping, the global admission counter is decremented."""
        with job_store._counts_lock:
            count_before = job_store._global_count

        job = reserve_slot(None)
        assert job is not None

        with job_store._counts_lock:
            count_after_reserve = job_store._global_count

        assert count_after_reserve == count_before + 1

        job._last_activity_wall = _expired_wall()
        _reap_stale_pending_uploads()

        with job_store._counts_lock:
            count_after_reap = job_store._global_count

        assert count_after_reap == count_before

    def test_multiple_stale_jobs_all_reaped(self):
        """Multiple stale PENDING_UPLOAD jobs are all reclaimed in one pass."""
        jobs = [reserve_slot(None) for _ in range(3)]
        for job in jobs:
            assert job is not None
            job._last_activity_wall = _expired_wall()

        _reap_stale_pending_uploads()

        for job in jobs:
            assert get_job(job.job_id) is None
            assert job.is_terminal

    def test_mix_of_stale_and_fresh_jobs(self):
        """Only stale jobs are reaped; fresh jobs in the same store are untouched."""
        stale = reserve_slot(None)
        assert stale is not None
        stale._last_activity_wall = _expired_wall()

        fresh = reserve_slot(None)
        assert fresh is not None
        # fresh._last_activity_wall is already recent (set by default_factory)

        _reap_stale_pending_uploads()

        assert get_job(stale.job_id) is None
        assert get_job(fresh.job_id) is not None

    def test_convert_to_pending_before_reap_is_not_cancelled(self):
        """A job that converts from PENDING_UPLOAD to PENDING before the reaper
        processes it is NOT cancelled, even if its activity wall is backdated.

        This covers the race: final chunk arrives, convert_to_pending() succeeds
        (client already received HTTP 200), then the reaper runs.  The atomic
        try_cancel_if_stale_upload checks state under the job lock, sees PENDING,
        and returns None without touching the job.
        """
        job = reserve_slot(None)
        assert job is not None
        assert job.state == JobState.PENDING_UPLOAD

        # Backdate so the job would be stale if still in PENDING_UPLOAD.
        job._last_activity_wall = _expired_wall()

        # Simulate the final chunk arriving: job transitions to PENDING.
        converted = job.convert_to_pending()
        assert converted is True
        state_after_convert: JobState = job.state
        assert state_after_convert == JobState.PENDING

        _reap_stale_pending_uploads()

        # Must NOT be reaped — it's now PENDING, not PENDING_UPLOAD.
        assert get_job(job.job_id) is not None
        state_after_reap: JobState = job.state
        assert state_after_reap == JobState.PENDING
        assert not job.is_terminal


@pytest.mark.unit
class TestTryCancelIfStaleUploadReturnType:
    """try_cancel_if_stale_upload returns float (idle seconds) on cancel, None otherwise."""

    def test_returns_idle_seconds_when_stale(self):
        """Returns a positive float equal to idle duration when the job is cancelled."""
        job = reserve_slot(None)
        assert job is not None
        job._last_activity_wall = _expired_wall()

        result = job.try_cancel_if_stale_upload(PENDING_UPLOAD_TIMEOUT_SECONDS)

        assert result is not None
        assert isinstance(result, float)
        assert result > PENDING_UPLOAD_TIMEOUT_SECONDS
        assert job.is_terminal

    def test_returns_none_when_fresh(self):
        """Returns None when the job is within the timeout window."""
        job = reserve_slot(None)
        assert job is not None
        # _last_activity_wall is recent by default.

        result = job.try_cancel_if_stale_upload(PENDING_UPLOAD_TIMEOUT_SECONDS)

        assert result is None
        assert not job.is_terminal

    def test_returns_none_for_pending_state(self):
        """Returns None when the job has already converted to PENDING."""
        job = reserve_slot(None)
        assert job is not None
        job._last_activity_wall = _expired_wall()

        job.convert_to_pending()
        assert job.state == JobState.PENDING

        result = job.try_cancel_if_stale_upload(PENDING_UPLOAD_TIMEOUT_SECONDS)

        assert result is None
        assert job.state == JobState.PENDING

    def test_returned_idle_matches_actual_age(self):
        """The returned idle seconds is close to the actual age of _last_activity_wall."""
        idle_age = timedelta(seconds=PENDING_UPLOAD_TIMEOUT_SECONDS + 120)
        job = reserve_slot(None)
        assert job is not None
        job._last_activity_wall = datetime.now(UTC) - idle_age

        result = job.try_cancel_if_stale_upload(PENDING_UPLOAD_TIMEOUT_SECONDS)

        assert result is not None
        # Should be approximately idle_age.total_seconds() — allow 5s of test slop.
        assert abs(result - idle_age.total_seconds()) < 5.0


@pytest.mark.unit
class TestGetLiveSpoolDirs:
    """get_live_spool_dirs returns the spool paths of in-flight jobs only."""

    def test_empty_when_no_jobs(self):
        """Returns an empty frozenset when the job store is empty."""
        result = get_live_spool_dirs()
        assert isinstance(result, frozenset)
        assert len(result) == 0

    def test_includes_live_job_temp_dir(self, tmp_path):
        """A job's temp_dir appears in the result while the job is in the store."""
        spool = tmp_path / "spool"
        spool.mkdir()
        create_job(JobType.UPLOAD, temp_dir=spool)

        result = get_live_spool_dirs()

        assert isinstance(result, frozenset)
        assert spool in result

    def test_excludes_job_without_temp_dir(self):
        """Jobs with temp_dir=None do not contribute a path to the result."""
        job = reserve_slot(None)
        assert job is not None
        assert job.temp_dir is None

        result = get_live_spool_dirs()

        assert len(result) == 0

    def test_excludes_removed_job_temp_dir(self, tmp_path):
        """A job removed from the store no longer appears in the result."""
        spool = tmp_path / "spool"
        spool.mkdir()
        job = create_job(JobType.UPLOAD, temp_dir=spool)
        assert spool in get_live_spool_dirs()

        job_store._jobs.pop(job.job_id, None)

        assert spool not in get_live_spool_dirs()

    def test_multiple_jobs_all_included(self, tmp_path):
        """All in-flight jobs with a temp_dir are returned."""
        spools = [tmp_path / f"spool{i}" for i in range(3)]
        for s in spools:
            s.mkdir()
        for s in spools:
            create_job(JobType.UPLOAD, temp_dir=s)

        result = get_live_spool_dirs()

        for s in spools:
            assert s in result
        assert len(result) == len(spools)

    def test_returns_frozenset(self, tmp_path):
        """Result is always a frozenset, never a mutable set."""
        spool = tmp_path / "spool"
        spool.mkdir()
        create_job(JobType.UPLOAD, temp_dir=spool)

        result = get_live_spool_dirs()

        assert isinstance(result, frozenset)


@pytest.mark.unit
class TestStartReaperSpoolSweepFn:
    """start_reaper calls spool_sweep_fn each iteration; None is a safe default."""

    def test_sweep_fn_invoked_each_iteration(self):
        """spool_sweep_fn is called on every reaper loop iteration."""
        call_count = 0
        called = threading.Event()

        def _sweep() -> None:
            nonlocal call_count
            call_count += 1
            called.set()

        thread, stop = start_reaper(interval=0.01, spool_sweep_fn=_sweep)
        try:
            fired = called.wait(timeout=2.0)
        finally:
            stop.set()
            thread.join(timeout=1.0)

        assert fired, "spool_sweep_fn was never called within 2 s"
        assert call_count >= 1

    def test_no_sweep_fn_does_not_raise(self):
        """start_reaper(spool_sweep_fn=None) runs without error."""
        thread, stop = start_reaper(interval=0.01, spool_sweep_fn=None)
        # Give the loop one tick to confirm it doesn't crash.
        time.sleep(0.05)
        assert thread.is_alive(), "Reaper thread must still be running after one tick"
        stop.set()
        thread.join(timeout=1.0)

    def test_sweep_fn_exception_does_not_kill_reaper(self):
        """A sweep fn that raises does not crash the reaper thread."""
        calls: list[int] = []
        reached_second = threading.Event()

        def _flaky() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("first call fails")
            reached_second.set()

        thread, stop = start_reaper(interval=0.01, spool_sweep_fn=_flaky)
        try:
            survived = reached_second.wait(timeout=2.0)
        finally:
            stop.set()
            thread.join(timeout=1.0)

        assert survived, "reaper thread died after sweep fn raised"

    def test_sweep_fn_evaluated_per_tick_not_captured(self, tmp_path):
        """The sweep lambda re-evaluates get_live_spool_dirs() on each tick.

        A job created AFTER the reaper starts must appear in a subsequent tick's
        skip set, proving the lambda does not capture a snapshot at startup.
        The sweep signals only once it actually observes the late job, so a tick
        whose snapshot predates create_job cannot satisfy the wait.
        """
        spool = tmp_path / "late_spool"
        spool.mkdir()
        first_tick = threading.Event()
        saw_late_job = threading.Event()

        def _recording_sweep() -> None:
            if spool in get_live_spool_dirs():
                saw_late_job.set()
            first_tick.set()

        thread, stop = start_reaper(interval=0.01, spool_sweep_fn=_recording_sweep)
        try:
            assert first_tick.wait(timeout=2.0), "Reaper never ticked"
            create_job(JobType.UPLOAD, temp_dir=spool)
            fired = saw_late_job.wait(timeout=2.0)
        finally:
            stop.set()
            thread.join(timeout=1.0)

        assert fired, (
            "get_live_spool_dirs() must be re-evaluated per tick, not captured at start"
        )


@pytest.mark.unit
class TestCleanupStaleSpoolDirsSkipPaths:
    """_cleanup_stale_upload_spool_dirs honours skip_paths for active uploads."""

    def _make_stale_dir(self, parent: Path, name: str) -> Path:
        """Create a subdirectory whose mtime is old enough to be swept."""
        d = parent / name
        d.mkdir()
        old_time = time.time() - _STALE_UPLOAD_TMPDIR_AGE_SECONDS - 1
        os.utime(d, (old_time, old_time))
        return d

    def test_skip_paths_spares_active_upload_dir(self, tmp_path):
        """A stale dir in skip_paths is preserved; its non-skipped sibling is removed.

        This is the core guard preventing deletion of in-progress chunked uploads:
        mtime is unreliable for active uploads because chunk writes land in
        subdirectories and don't bump the top-level spool dir mtime.
        """
        kept = self._make_stale_dir(tmp_path, "active_upload")
        removed = self._make_stale_dir(tmp_path, "orphaned_upload")
        mock_cfg = MagicMock()
        mock_cfg.upload_spool_dir = tmp_path

        with patch("snore.api.config.get_config", return_value=mock_cfg):
            _cleanup_stale_upload_spool_dirs(skip_paths=frozenset([kept]))

        assert kept.exists(), "Directory in skip_paths must not be removed"
        assert not removed.exists(), "Non-skipped stale sibling must be removed"
