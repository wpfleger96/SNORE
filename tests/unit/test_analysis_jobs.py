"""Unit tests for the analysis job queue state machine.

Tests cover the module-level store functions and the AnalysisJob state machine
without starting real worker threads or touching a database.
"""

from __future__ import annotations

import time

from datetime import UTC, datetime

import pytest

import snore.api.analysis_jobs as jobs

from snore.api.analysis_jobs import (
    _DEFAULT_ANALYSIS_JOB_CONCURRENCY,
    _DEFAULT_ANALYSIS_MAX_WORKERS,
    JOB_TTL_SECONDS,
    MAX_QUEUED,
    AnalysisJob,
    AnalysisJobSource,
    AnalysisJobState,
    _get_analysis_workers,
    _get_job_concurrency,
    _reap_terminal,
    cancel_job,
    enqueue,
    get_job,
    in_flight_session_ids,
    list_jobs,
    shutdown,
    start_worker,
)

# ---------------------------------------------------------------------------
# Fixture: clean module state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_analysis_jobs():
    """Stop leaked workers and reset global analysis job state around each test.

    shutdown() stops every worker generation registered in the module —
    including workers leaked by other test files sharing this process (e.g.
    an API lifespan that never finished its teardown) — and prunes exited
    threads itself. Do NOT rebind jobs._worker_threads here: dropping the
    registry would orphan any thread that outlived the join timeout, and an
    orphan with an unset stop event steals jobs from the shared queue.
    """
    jobs.shutdown(timeout=5.0)
    jobs._all_jobs.clear()
    jobs._queue.clear()
    yield
    jobs.shutdown(timeout=5.0)
    jobs._all_jobs.clear()
    jobs._queue.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enqueue_one(session_ids: list[int] | None = None) -> AnalysisJob:
    job = enqueue(
        profile_id=1,
        session_ids=session_ids or [1, 2, 3],
        source=AnalysisJobSource.BATCH,
    )
    assert job is not None
    return job


# ---------------------------------------------------------------------------
# 1. Enqueue happy path
# ---------------------------------------------------------------------------


def test_enqueue_happy_path_appears_in_store():
    job = enqueue(
        profile_id=1,
        session_ids=[10, 20],
        source=AnalysisJobSource.BATCH,
        owner_user_id=42,
    )
    assert job is not None
    assert job.state == AnalysisJobState.QUEUED
    assert get_job(job.job_id) is job
    assert any(j is job for j in list_jobs())


def test_enqueue_owner_scoping():
    job = enqueue(
        profile_id=1,
        session_ids=[1],
        source=AnalysisJobSource.IMPORT,
        owner_user_id=7,
    )
    assert job is not None
    assert list_jobs(owner_user_id=7) == [job]
    assert list_jobs(owner_user_id=99) == []


def test_list_jobs_none_owner_visible_to_any_caller():
    """A job with owner_user_id=None is visible to any authenticated caller."""
    job = enqueue(
        profile_id=1,
        session_ids=[1],
        source=AnalysisJobSource.IMPORT,
        owner_user_id=None,
    )
    assert job is not None
    assert job in list_jobs(owner_user_id=42)
    assert job in list_jobs(owner_user_id=99)


def test_list_jobs_foreign_owner_not_visible():
    """A job owned by user A must not appear in user B's list."""
    job = enqueue(
        profile_id=1,
        session_ids=[1],
        source=AnalysisJobSource.IMPORT,
        owner_user_id=7,
    )
    assert job is not None
    assert job not in list_jobs(owner_user_id=99)


def test_list_jobs_none_parameter_returns_all():
    """Passing owner_user_id=None to list_jobs returns all jobs."""
    job_a = enqueue(
        profile_id=1, session_ids=[1], source=AnalysisJobSource.BATCH, owner_user_id=1
    )
    job_b = enqueue(
        profile_id=1, session_ids=[2], source=AnalysisJobSource.BATCH, owner_user_id=2
    )
    assert job_a is not None
    assert job_b is not None
    all_jobs = list_jobs(owner_user_id=None)
    assert job_a in all_jobs
    assert job_b in all_jobs


def test_enqueue_stores_modes_and_primary_mode():
    job = enqueue(
        profile_id=1,
        session_ids=[1],
        source=AnalysisJobSource.BATCH,
        modes=["aasm", "resmed"],
        primary_mode="resmed",
        store_results=False,
    )
    assert job is not None
    assert job.modes == ["aasm", "resmed"]
    assert job.primary_mode == "resmed"
    assert job.store_results is False


# ---------------------------------------------------------------------------
# 2. Enqueue returns None at MAX_QUEUED
# ---------------------------------------------------------------------------


def test_enqueue_returns_none_when_queue_full():
    for _ in range(MAX_QUEUED):
        j = enqueue(profile_id=1, session_ids=[1], source=AnalysisJobSource.BATCH)
        assert j is not None
    assert (
        enqueue(profile_id=1, session_ids=[1], source=AnalysisJobSource.BATCH) is None
    )


# ---------------------------------------------------------------------------
# 3. Cancel of QUEUED job → CANCELLED, removed from queue
# ---------------------------------------------------------------------------


def test_cancel_queued_job_transitions_and_removes():
    job = _enqueue_one()
    # Read via locals: asserting on job.state directly makes mypy narrow the
    # property to a literal and flag the post-mutation comparison below.
    state_before = job.state
    assert state_before == AnalysisJobState.QUEUED
    assert len(jobs._queue) == 1

    result = cancel_job(job.job_id)

    assert result is True
    state_after = job.state
    assert state_after == AnalysisJobState.CANCELLED
    assert len(jobs._queue) == 0


def test_cancel_unknown_job_returns_false():
    assert cancel_job("nonexistent") is False


def test_cancel_already_terminal_job_returns_false():
    job = _enqueue_one()
    cancel_job(job.job_id)
    # Already CANCELLED; second call should return False.
    assert cancel_job(job.job_id) is False


# ---------------------------------------------------------------------------
# 4. try_start after cancel_flag set returns False → TOCTOU guard
# ---------------------------------------------------------------------------


def test_try_start_after_cancel_flag_returns_false_and_is_cancelled():
    job = _enqueue_one()
    # Simulate cancel flag set before the worker dequeues the job.
    with job._lock:
        job._cancel_flag = True
        job._state = AnalysisJobState.QUEUED  # reset to QUEUED
        job._finished_at = None

    result = job.try_start()

    assert result is False
    assert job.state == AnalysisJobState.CANCELLED
    assert job.finished_at is not None


def test_try_start_without_cancel_transitions_to_running():
    job = _enqueue_one()
    result = job.try_start()
    assert result is True
    assert job.state == AnalysisJobState.RUNNING
    assert job.started_at is not None


# ---------------------------------------------------------------------------
# 5. finish(succeeded=True) after cancel_flag set → CANCELLED not SUCCEEDED
# ---------------------------------------------------------------------------


def test_finish_with_cancel_flag_yields_cancelled():
    job = _enqueue_one()
    job.try_start()
    # Cancel arrives while the job is RUNNING.
    with job._lock:
        job._cancel_flag = True

    job.finish(succeeded=True)

    assert job.state == AnalysisJobState.CANCELLED
    assert job.finished_at is not None


def test_finish_succeeded_without_cancel():
    job = _enqueue_one()
    job.try_start()
    job.finish(succeeded=True)
    assert job.state == AnalysisJobState.SUCCEEDED


def test_finish_failed_sets_error_message():
    job = _enqueue_one()
    job.try_start()
    job.finish(succeeded=False, error_message="boom")
    assert job.state == AnalysisJobState.FAILED
    assert job.error_message == "boom"


def test_finish_is_idempotent_after_terminal():
    job = _enqueue_one()
    job.try_start()
    job.finish(succeeded=True)
    job.finish(succeeded=False, error_message="should be ignored")
    # First finish wins.
    assert job.state == AnalysisJobState.SUCCEEDED
    assert job.error_message is None


# ---------------------------------------------------------------------------
# 6. _reap_terminal removes expired jobs; retains fresh and non-terminal ones
# ---------------------------------------------------------------------------


def test_reap_terminal_removes_expired_and_retains_others():
    # Expired terminal job
    job_expired = _enqueue_one()
    cancel_job(job_expired.job_id)
    with job_expired._lock:
        job_expired._finished_at = time.monotonic() - JOB_TTL_SECONDS - 1

    # Non-terminal job
    job_running = enqueue(
        profile_id=1, session_ids=[5], source=AnalysisJobSource.IMPORT
    )
    assert job_running is not None
    with job_running._lock:
        job_running._state = AnalysisJobState.RUNNING

    # Recent terminal job (within TTL)
    job_recent = enqueue(profile_id=1, session_ids=[6], source=AnalysisJobSource.BATCH)
    assert job_recent is not None
    cancel_job(job_recent.job_id)  # finished_at = now

    _reap_terminal()

    assert get_job(job_expired.job_id) is None
    assert get_job(job_running.job_id) is job_running
    assert get_job(job_recent.job_id) is job_recent


# ---------------------------------------------------------------------------
# 6b. in_flight_session_ids covers live jobs of one profile
# ---------------------------------------------------------------------------


def test_in_flight_session_ids_counts_queued_and_running_jobs_of_profile():
    queued = enqueue(
        profile_id=1,
        session_ids=[1, 2],
        source=AnalysisJobSource.BATCH,
        owner_user_id=7,
    )
    running = enqueue(profile_id=1, session_ids=[3], source=AnalysisJobSource.IMPORT)
    other_profile = enqueue(
        profile_id=2, session_ids=[4], source=AnalysisJobSource.BATCH
    )
    finished = enqueue(profile_id=1, session_ids=[5], source=AnalysisJobSource.BATCH)
    cancelling = enqueue(profile_id=1, session_ids=[6], source=AnalysisJobSource.BATCH)
    assert queued and running and other_profile and finished and cancelling
    assert running.try_start()
    assert finished.try_start()
    finished.finish(succeeded=True)
    assert cancelling.try_start()
    assert cancel_job(cancelling.job_id)

    assert in_flight_session_ids(1) == {1, 2, 3}
    assert in_flight_session_ids(2) == {4}
    assert in_flight_session_ids(3) == set()


# ---------------------------------------------------------------------------
# 7. shutdown marks QUEUED jobs CANCELLED
# ---------------------------------------------------------------------------


def test_shutdown_cancels_queued_jobs():
    job1 = enqueue(profile_id=1, session_ids=[1], source=AnalysisJobSource.BATCH)
    job2 = enqueue(profile_id=1, session_ids=[2], source=AnalysisJobSource.BATCH)
    assert job1 is not None
    assert job2 is not None

    shutdown(timeout=0.1)

    assert job1.state == AnalysisJobState.CANCELLED
    assert job2.state == AnalysisJobState.CANCELLED
    assert len(jobs._queue) == 0


def test_shutdown_sets_cancel_flag_on_running_job():
    job = _enqueue_one()
    with job._lock:
        job._state = AnalysisJobState.RUNNING  # Simulate worker picked it up.
        jobs._queue.clear()

    shutdown(timeout=0.1)

    assert job.cancel_requested is True


# ---------------------------------------------------------------------------
# 8. to_dict shape matches AnalysisJobStatus (no profile_id)
# ---------------------------------------------------------------------------


def test_to_dict_fields_and_no_profile_id():
    job = enqueue(
        profile_id=99,
        session_ids=[1, 2],
        source=AnalysisJobSource.IMPORT,
        owner_user_id=5,
    )
    assert job is not None
    d = job.to_dict()

    assert "profile_id" not in d
    assert d["job_id"] == job.job_id
    assert d["session_count"] == 2
    assert d["source"] == "import"
    assert d["owner_user_id"] == 5
    assert d["state"] == "queued"
    assert "created_at" in d
    assert d["started_at"] is None
    assert d["finished_at"] is None
    assert d["eta_seconds"] is None


def test_to_dict_timestamps_are_wall_clock_datetimes():
    """created_at/started_at/finished_at must be wall-clock ``datetime`` values,
    not time.monotonic() — a monotonic value (seconds since boot) fed to the
    client would render as a 1970-era date (see PR #290)."""
    job = _enqueue_one()
    job.try_start()
    job.finish(succeeded=True)

    d = job.to_dict()
    now = datetime.now(UTC)
    # A just-created/started/finished job's timestamps are within seconds of now.
    assert isinstance(d["created_at"], datetime)
    assert isinstance(d["started_at"], datetime)
    assert isinstance(d["finished_at"], datetime)
    assert abs((d["created_at"] - now).total_seconds()) < 60
    assert abs((d["started_at"] - now).total_seconds()) < 60
    assert abs((d["finished_at"] - now).total_seconds()) < 60


# ---------------------------------------------------------------------------
# 9. _get_analysis_workers reads config and falls back gracefully
# ---------------------------------------------------------------------------


def test_get_analysis_workers_returns_configured_value():
    from snore.api.config import AppConfig, reset_config, set_config
    from snore.auth.actor import AuthMode

    cfg = AppConfig(
        auth_mode=AuthMode.LOCAL,
        session_secret="",
        public_base_url="",
        public_origin=None,
        bind_host="127.0.0.1",
        trusted_proxies=frozenset(),
        dev_origins=frozenset(),
        cors_origins=["http://localhost:5173"],
        google_client_id="",
        google_client_secret="",
        oauth_attempt_ttl_seconds=600,
        pre_auth_cookie_ttl_seconds=600,
        max_upload_bytes=512 * 1024 * 1024,
        max_file_bytes=256 * 1024 * 1024,
        max_upload_files=10_000,
        max_jobs_per_user=3,
        max_jobs_global=10,
        analysis_max_workers=6,
        analysis_job_concurrency=2,
    )
    set_config(cfg)
    try:
        assert _get_analysis_workers() == 6
    finally:
        reset_config()


def test_get_analysis_workers_falls_back_to_default_when_config_raises(monkeypatch):
    import snore.api.config as _config_mod

    monkeypatch.setattr(_config_mod, "_config", None)
    # Patch load_config to raise so get_config() propagates the error.
    monkeypatch.setattr(
        _config_mod,
        "load_config",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("config unavailable")),
    )
    assert _get_analysis_workers() == _DEFAULT_ANALYSIS_MAX_WORKERS


# ---------------------------------------------------------------------------
# 10. _get_job_concurrency reads config and falls back gracefully
# ---------------------------------------------------------------------------


def test_get_job_concurrency_returns_configured_value():
    from snore.api.config import AppConfig, reset_config, set_config
    from snore.auth.actor import AuthMode

    cfg = AppConfig(
        auth_mode=AuthMode.LOCAL,
        session_secret="",
        public_base_url="",
        public_origin=None,
        bind_host="127.0.0.1",
        trusted_proxies=frozenset(),
        dev_origins=frozenset(),
        cors_origins=["http://localhost:5173"],
        google_client_id="",
        google_client_secret="",
        oauth_attempt_ttl_seconds=600,
        pre_auth_cookie_ttl_seconds=600,
        max_upload_bytes=512 * 1024 * 1024,
        max_file_bytes=256 * 1024 * 1024,
        max_upload_files=10_000,
        max_jobs_per_user=3,
        max_jobs_global=10,
        analysis_max_workers=4,
        analysis_job_concurrency=3,
    )
    set_config(cfg)
    try:
        assert _get_job_concurrency() == 3
    finally:
        reset_config()


def test_default_job_concurrency_resolves_to_four_when_config_value_is_none():
    """analysis_job_concurrency=None (unset env var) resolves to the module default (4)."""
    from snore.api.config import AppConfig, reset_config, set_config
    from snore.auth.actor import AuthMode

    cfg = AppConfig(
        auth_mode=AuthMode.LOCAL,
        session_secret="",
        public_base_url="",
        public_origin=None,
        bind_host="127.0.0.1",
        trusted_proxies=frozenset(),
        dev_origins=frozenset(),
        cors_origins=["http://localhost:5173"],
        google_client_id="",
        google_client_secret="",
        oauth_attempt_ttl_seconds=600,
        pre_auth_cookie_ttl_seconds=600,
        max_upload_bytes=512 * 1024 * 1024,
        max_file_bytes=256 * 1024 * 1024,
        max_upload_files=10_000,
        max_jobs_per_user=3,
        max_jobs_global=10,
        analysis_max_workers=4,
        analysis_job_concurrency=None,
    )
    set_config(cfg)
    try:
        assert _get_job_concurrency() == 4
        assert _get_job_concurrency() == _DEFAULT_ANALYSIS_JOB_CONCURRENCY
    finally:
        reset_config()


def test_explicit_job_concurrency_clamps_to_max_queued():
    """An explicit analysis_job_concurrency value above MAX_QUEUED is clamped."""
    from snore.api.config import AppConfig, reset_config, set_config
    from snore.auth.actor import AuthMode

    cfg = AppConfig(
        auth_mode=AuthMode.LOCAL,
        session_secret="",
        public_base_url="",
        public_origin=None,
        bind_host="127.0.0.1",
        trusted_proxies=frozenset(),
        dev_origins=frozenset(),
        cors_origins=["http://localhost:5173"],
        google_client_id="",
        google_client_secret="",
        oauth_attempt_ttl_seconds=600,
        pre_auth_cookie_ttl_seconds=600,
        max_upload_bytes=512 * 1024 * 1024,
        max_file_bytes=256 * 1024 * 1024,
        max_upload_files=10_000,
        max_jobs_per_user=3,
        max_jobs_global=10,
        analysis_max_workers=4,
        analysis_job_concurrency=MAX_QUEUED + 5,
    )
    set_config(cfg)
    try:
        assert _get_job_concurrency() == MAX_QUEUED
    finally:
        reset_config()


def test_get_job_concurrency_falls_back_to_default_when_config_raises(monkeypatch):
    import snore.api.config as _config_mod

    monkeypatch.setattr(_config_mod, "_config", None)
    monkeypatch.setattr(
        _config_mod,
        "load_config",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("config unavailable")),
    )
    assert _get_job_concurrency() == _DEFAULT_ANALYSIS_JOB_CONCURRENCY


# ---------------------------------------------------------------------------
# 11. Worker concurrency: two jobs run simultaneously with concurrency=2
# ---------------------------------------------------------------------------


def test_two_jobs_run_concurrently():
    """Under the default concurrency (4 workers >= 2 jobs), both jobs reach RUNNING before either finishes."""
    import threading

    from unittest.mock import patch

    job1_running = threading.Event()
    job2_running = threading.Event()
    release = threading.Event()

    # job1_ref holds the first job so fake_execute can tell them apart.
    job1_ref: list[AnalysisJob | None] = [None]

    def fake_execute(job: AnalysisJob) -> None:
        if not job.try_start():
            return
        if job is job1_ref[0]:
            job1_running.set()
        else:
            job2_running.set()
        release.wait(timeout=5.0)
        job.finish(succeeded=True)

    # No _get_job_concurrency patch: the default (4 workers) is >= 2 jobs, so
    # both run concurrently without waiting for one another.
    with patch("snore.api.analysis_jobs._execute_job", side_effect=fake_execute):
        job1 = enqueue(profile_id=1, session_ids=[1], source=AnalysisJobSource.BATCH)
        job2 = enqueue(profile_id=1, session_ids=[2], source=AnalysisJobSource.BATCH)
        assert job1 is not None and job2 is not None
        job1_ref[0] = job1

        threads, stop_event = start_worker()
        try:
            assert job1_running.wait(timeout=3.0), "job1 never reached RUNNING"
            assert job2_running.wait(timeout=3.0), (
                "job2 never reached RUNNING concurrently with job1"
            )
        finally:
            release.set()  # unblock any blocked fake_execute before stopping
            stop_event.set()
            with jobs._condition:
                jobs._condition.notify_all()
            for t in threads:
                t.join(timeout=5.0)


# ---------------------------------------------------------------------------
# 12. Worker serialization: with concurrency=1, jobs run one at a time
# ---------------------------------------------------------------------------


def test_serial_when_concurrency_one():
    """With concurrency=1, job2 does not start until job1 finishes."""
    import threading

    from unittest.mock import patch

    job1_running = threading.Event()
    job1_done = threading.Event()
    job2_started = threading.Event()
    release_job1 = threading.Event()

    job1_ref: list[AnalysisJob | None] = [None]

    def fake_execute(job: AnalysisJob) -> None:
        if not job.try_start():
            return
        if job is job1_ref[0]:
            job1_running.set()
            release_job1.wait(timeout=5.0)
            job.finish(succeeded=True)
            job1_done.set()
        else:
            job2_started.set()
            job.finish(succeeded=True)

    with (
        patch("snore.api.analysis_jobs._execute_job", side_effect=fake_execute),
        patch("snore.api.analysis_jobs._get_job_concurrency", return_value=1),
    ):
        job1 = enqueue(profile_id=1, session_ids=[1], source=AnalysisJobSource.BATCH)
        job2 = enqueue(profile_id=1, session_ids=[2], source=AnalysisJobSource.BATCH)
        assert job1 is not None and job2 is not None
        job1_ref[0] = job1

        threads, stop_event = start_worker()
        try:
            assert job1_running.wait(timeout=3.0), "job1 never started"
            # job2 must NOT start while job1 is still blocked.
            assert not job2_started.wait(timeout=0.3), (
                "job2 started while job1 was still running"
            )
            release_job1.set()
            assert job1_done.wait(timeout=3.0), "job1 never finished"
            assert job2_started.wait(timeout=3.0), (
                "job2 never started after job1 finished"
            )
        finally:
            release_job1.set()  # unblock fake_execute if the test body failed early
            stop_event.set()
            with jobs._condition:
                jobs._condition.notify_all()
            for t in threads:
                t.join(timeout=5.0)


# ---------------------------------------------------------------------------
# 13. shutdown() joins all N worker threads; none survive
# ---------------------------------------------------------------------------


def test_shutdown_joins_all_worker_threads():
    """shutdown() waits for all analysis-job-worker-* threads to exit."""
    from unittest.mock import patch

    with patch("snore.api.analysis_jobs._get_job_concurrency", return_value=3):
        threads, _stop = start_worker()

    assert len(threads) == 3
    assert all(t.is_alive() for t in threads), "Workers should be alive before shutdown"

    shutdown(timeout=5.0)

    # Assert that the threads this test started are all dead — not a global scan,
    # which would be flaky if a prior test leaked a thread.
    assert not any(t.is_alive() for t in threads), (
        f"Worker threads still alive after shutdown: {[t.name for t in threads if t.is_alive()]}"
    )


# ---------------------------------------------------------------------------
# 14. to_dict() eta_seconds: None when not started or no progress; positive mid-run
# ---------------------------------------------------------------------------


def test_to_dict_eta_seconds_none_when_not_started():
    """eta_seconds is None for a job that has not started yet."""
    job = _enqueue_one()
    d = job.to_dict()
    assert d["eta_seconds"] is None


def test_to_dict_eta_seconds_none_when_no_progress():
    """eta_seconds is None while RUNNING but progress_completed is still zero."""
    job = _enqueue_one()
    job.try_start()
    # No update_progress call — completed stays at 0.
    d = job.to_dict()
    assert d["state"] == "running"
    assert d["eta_seconds"] is None


def test_to_dict_eta_seconds_positive_mid_run():
    """eta_seconds is a positive integer when the job is running with measurable progress."""
    job = _enqueue_one(session_ids=list(range(10)))
    job.try_start()
    # Backdate started_at by 5 seconds so the rate estimate is non-trivial.
    with job._lock:
        job._started_at = time.monotonic() - 5.0
    job.update_progress(done=5, total=10)

    d = job.to_dict()
    assert d["state"] == "running"
    eta = d["eta_seconds"]
    assert isinstance(eta, int), f"expected int, got {type(eta)}"
    assert eta > 0, f"expected positive ETA, got {eta}"


def test_to_dict_eta_seconds_none_after_completion():
    """eta_seconds is None once the job transitions out of RUNNING."""
    job = _enqueue_one(session_ids=list(range(10)))
    job.try_start()
    job.update_progress(done=5, total=10)
    job.finish(succeeded=True)

    d = job.to_dict()
    assert d["state"] == "succeeded"
    assert d["eta_seconds"] is None


def test_to_dict_eta_seconds_zero_when_overcompleted():
    """eta_seconds is 0, not negative, when progress_completed exceeds progress_total."""
    job = _enqueue_one(session_ids=list(range(10)))
    job.try_start()
    # Simulate a race where completed overshoots total.
    with job._lock:
        job._started_at = time.monotonic() - 5.0
    job.update_progress(done=12, total=10)

    d = job.to_dict()
    assert d["state"] == "running"
    assert d["eta_seconds"] == 0
