"""Drive the admin day recompute job from tests.

``make_test_client`` skips the app lifespan, so no worker thread runs: a POST
leaves its job queued until a test executes it here, on the test's own thread.
"""

from __future__ import annotations

from snore.api import recompute_jobs

RECOMPUTE_PATH = "/api/v1/db/recompute-days"


def clear_recompute_jobs() -> None:
    """Stop any worker and empty the module-global job store."""
    recompute_jobs.shutdown(timeout=5.0)
    recompute_jobs._all_jobs.clear()
    recompute_jobs._queue.clear()


def run_queued_recompute_job() -> None:
    """Execute the queued job to completion, as the worker thread would."""
    recompute_jobs._execute_job(recompute_jobs._queue.popleft())
