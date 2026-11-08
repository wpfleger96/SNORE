import threading

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from snore.api import recompute_jobs, reset_jobs
from snore.api.constants import NO_STORE
from snore.api.deps import service_dep
from snore.api.guards import RequireAdmin
from snore.api.jobs.core import (
    MAINTENANCE_CONFLICT_DETAIL,
    admit_maintenance,
    register_maintenance_job,
)
from snore.api.schemas import (
    RecomputeDaysJobStatus,
    ResetJobEnqueued,
    ResetJobStatus,
)
from snore.database.target import DatabaseTarget
from snore.services.database_service import DatabaseService
from snore.services.schemas import DatabaseStats, VacuumResult

router = APIRouter()

DatabaseServiceDep = Annotated[DatabaseService, Depends(service_dep(DatabaseService))]

# Set for the whole of a POST /vacuum, so no reset or recompute starts under it.
_vacuum_running = threading.Event()
register_maintenance_job("vacuum", _vacuum_running.is_set)


class DatabaseStatsPublic(DatabaseStats):
    # Exclude server filesystem path from API responses.
    db_path: str = Field(default="", exclude=True)


class ResetRequest(BaseModel):
    include_accounts: bool = False


def _get_target() -> DatabaseTarget:
    """Resolve the current database target from the environment/session."""
    return DatabaseTarget.from_env_and_flags(db_flag=None, warn_ignored=False)


@router.get("/stats", response_model=DatabaseStatsPublic)
async def get_stats(
    service: DatabaseServiceDep,
    target: Annotated[DatabaseTarget, Depends(_get_target)],
    _actor: RequireAdmin,
) -> DatabaseStats:
    db_path = target.sqlite_path if target.dialect == "sqlite" else ""
    return await service.get_stats(db_path)


@router.post("/vacuum", response_model=VacuumResult)
def vacuum_db(
    service: DatabaseServiceDep,
    target: Annotated[DatabaseTarget, Depends(_get_target)],
    _actor: RequireAdmin,
) -> VacuumResult:
    """Vacuum the SQLite database to reclaim space after deletions.

    Requires a SQLite file target; raises 422 for non-SQLite databases.
    Returns 409 while another vacuum or any other database maintenance task (a
    reset, data deletion or day recompute) is queued or running.
    """
    if (
        target.dialect != "sqlite"
        or not target.location
        or target.location == ":memory:"
    ):
        raise HTTPException(
            status_code=422,
            detail="VACUUM is only available for SQLite file databases.",
        )
    with admit_maintenance("vacuum") as admitted:
        if not admitted or _vacuum_running.is_set():
            raise HTTPException(status_code=409, detail=MAINTENANCE_CONFLICT_DETAIL)
        _vacuum_running.set()
    try:
        return service.vacuum_sqlite(target.sqlite_path)
    finally:
        _vacuum_running.clear()


@router.post("/reset", status_code=202, response_model=ResetJobEnqueued)
async def reset_db(
    target: Annotated[DatabaseTarget, Depends(_get_target)],
    actor: RequireAdmin,
    body: Annotated[ResetRequest | None, Body()] = None,
) -> JSONResponse:
    """Queue a background job that deletes database rows and vacuums.

    Two modes selected via the request body:

    ``include_accounts=false`` (default):
        Delete all sleep data (devices, sessions, waveforms, events, statistics,
        settings, analysis results, breaths, detected patterns, days, Apple Health
        samples and summaries, mask log, validation runs, analysis and import job
        records) and purge every ``raw/<profile_id>/`` backup directory.
        User accounts, auth identities, 2FA recovery codes, invites, and profile
        containers are preserved, so users keep their accounts and can re-import
        whatever is still on their SD cards.

    ``include_accounts=true``:
        Full factory reset (every row in every table) plus raw-dir purge and
        vacuum.  A fresh bootstrap admin invite is created for the calling
        admin's email.  The caller's session dies mid-job; they must redeem that
        invite to regain access.

    The work outlasts the proxy's request timeout on a large database, so this
    returns 202 at once with a ``job_id`` to poll with
    ``GET /db/reset-jobs/status``, which carries the result, including a factory
    reset's one-time ``bootstrap_invite_url``.  Returns 409 while any database
    maintenance task (another reset, a data deletion, a vacuum or a day
    recompute) is queued or running.

    Note: the job holds the SQLite write lock for the entire delete+commit, so
    on large databases concurrent import/analysis writers may hit their busy
    timeout — operators should quiesce imports before a factory reset.
    """
    req = body or ResetRequest()
    return reset_jobs.accepted_or_409(
        reset_jobs.enqueue(
            kind=reset_jobs.ResetJobKind.RESET,
            user_id=actor.user_id,
            include_accounts=req.include_accounts,
            db_path=reset_jobs.sqlite_file_path(target),
        )
    )


@router.get("/reset-jobs/status", response_model=ResetJobStatus)
async def get_reset_job_status(
    token: Annotated[str, Header(alias="X-Reset-Job-Token")],
) -> JSONResponse:
    """Report a reset or data deletion job to the holder of its token."""
    # Deliberately no actor dependency: a factory reset deletes the caller's
    # session mid-job, and this is how they get the invite that restores access.
    # The token is the credential, and it travels in a header because URLs land
    # in uvicorn and Cloudflare logs, and this response can hold the invite URL.
    job = reset_jobs.get_job(token)
    if job is None:
        raise HTTPException(status_code=404, detail="Reset job not found or expired")
    status = ResetJobStatus.model_validate(job.to_dict())
    return JSONResponse(content=status.model_dump(mode="json"), headers=NO_STORE)


@router.post("/recompute-days", status_code=202, response_model=RecomputeDaysJobStatus)
async def recompute_days(actor: RequireAdmin) -> RecomputeDaysJobStatus:
    """Queue a background job that re-derives every Day row.

    The API counterpart of ``snore db recompute-days``: no re-parse.  Every Day
    across all users, profiles and devices is re-aggregated from the session
    statistics and waveforms already stored, and Day rows no session references
    are deleted.  Use it after a change to day aggregation.

    Returns 202 at once; poll ``GET /db/recompute-days`` for progress and the
    result.  Returns 409 while any database maintenance task (another
    recompute, a reset, a data deletion or a vacuum) is queued or running.
    """
    job = recompute_jobs.enqueue(owner_user_id=actor.user_id)
    if job is None:
        raise HTTPException(status_code=409, detail=MAINTENANCE_CONFLICT_DETAIL)
    return job.to_status()


@router.get("/recompute-days", response_model=RecomputeDaysJobStatus)
async def get_recompute_days_status(_actor: RequireAdmin) -> JSONResponse:
    """Report the most recent day recompute job.

    Returns 404 when no job has run since the server started, or the last one
    ended long enough ago to have expired.
    """
    job = recompute_jobs.latest_job()
    if job is None:
        raise HTTPException(status_code=404, detail="No day recompute job found")
    return JSONResponse(
        content=job.to_status().model_dump(mode="json"), headers=NO_STORE
    )
