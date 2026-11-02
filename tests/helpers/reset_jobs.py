"""Drive database reset and data deletion jobs from tests.

``make_test_client`` skips the app lifespan, so no worker thread runs: a POST
leaves its job queued until a test executes it here, on the test's own thread.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from snore.api import reset_jobs

RESET_PATH = "/api/v1/db/reset"
DELETE_DATA_PATH = "/api/v1/auth/me/delete-data"
STATUS_PATH = "/api/v1/db/reset-jobs/status"


def clear_reset_jobs() -> None:
    """Stop any worker and empty the module-global job store."""
    reset_jobs.shutdown(timeout=5.0)
    reset_jobs._all_jobs.clear()
    reset_jobs._queue.clear()


def token_header(job_id: str) -> dict[str, str]:
    return {"X-Reset-Job-Token": job_id}


def read_status(client: TestClient, job_id: str) -> Any:
    return client.get(STATUS_PATH, headers=token_header(job_id))


def run_queued_job() -> None:
    """Execute the queued job to completion, as the worker thread would."""
    reset_jobs._execute_job(reset_jobs._queue.popleft())


def post_and_run_job(
    client: TestClient, path: str, **post_kwargs: Any
) -> dict[str, Any]:
    """POST *path*, run the job it queues, and return the status body."""
    response = client.post(path, **post_kwargs)
    assert response.status_code == 202, response.text
    run_queued_job()
    status = read_status(client, response.json()["job_id"])
    assert status.status_code == 200, status.text
    return status.json()


def run_reset(client: TestClient, **post_kwargs: Any) -> dict[str, Any]:
    """Run ``POST /db/reset`` to success and return its ``ResetResult`` body."""
    status = post_and_run_job(client, RESET_PATH, **post_kwargs)
    assert status["state"] == "succeeded", status
    return status["reset_result"]


def run_delete_data(client: TestClient) -> dict[str, Any]:
    """Run ``POST /auth/me/delete-data`` to success; return its result body."""
    status = post_and_run_job(client, DELETE_DATA_PATH)
    assert status["state"] == "succeeded", status
    return status["delete_data_result"]
