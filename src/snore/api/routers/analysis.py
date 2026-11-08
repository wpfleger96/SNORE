from datetime import datetime, time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.types import AnalysisResult
from snore.api import analysis_jobs
from snore.api.deps import (
    DateRangeParams,
    PaginationParams,
    get_db,
    service_dep,
    service_dep_immediate,
)
from snore.api.errors import NotFoundError
from snore.api.guards import RequireAuth, RequireWritable
from snore.api.schemas import (
    AnalysisDeleteRequest,
    AnalysisJobEnqueued,
    AnalysisJobsListResponse,
    AnalysisRunRequest,
    BatchAnalysisRequest,
    PaginatedResponse,
)
from snore.services import AnalysisFacade
from snore.services.schemas import (
    EMPTY_SELECTION_MESSAGES,
    AnalysisDeletePreview,
    AnalysisListItem,
)

router = APIRouter()

AnalysisFacadeDep = Annotated[AnalysisFacade, Depends(service_dep(AnalysisFacade))]
AnalysisImmediateFacadeDep = Annotated[
    AnalysisFacade, Depends(service_dep_immediate(AnalysisFacade))
]


@router.get("/analysis/sessions", response_model=PaginatedResponse[AnalysisListItem])
async def list_analysis_sessions(
    facade: AnalysisFacadeDep,
    pagination: PaginationParams = Depends(),
    dates: DateRangeParams = Depends(),
    analyzed_only: bool = Query(default=False),
    sort_by: str = Query(default="date-desc"),
) -> PaginatedResponse[AnalysisListItem]:
    total = await facade.count_sessions_with_status(
        start=dates.start_datetime, end=dates.end_datetime, analyzed_only=analyzed_only
    )
    items = await facade.list_sessions_with_status(
        start=dates.start_datetime,
        end=dates.end_datetime,
        limit=pagination.limit,
        offset=pagination.offset,
        analyzed_only=analyzed_only,
        sort_by=sort_by,
    )
    return PaginatedResponse(
        items=items, total=total, limit=pagination.limit, offset=pagination.offset
    )


@router.get("/sessions/{session_id}/analysis", response_model=AnalysisResult)
async def get_analysis(session_id: int, facade: AnalysisFacadeDep) -> AnalysisResult:
    result = await facade.get_analysis_result(session_id)
    if result is None:
        raise NotFoundError(f"No analysis found for session {session_id}")
    return result


@router.post(
    "/sessions/{session_id}/analysis", status_code=201, response_model=AnalysisResult
)
async def run_analysis(
    session_id: int,
    body: AnalysisRunRequest,
    facade: AnalysisFacadeDep,
    _actor: RequireWritable,
) -> AnalysisResult:
    try:
        return await facade.run_analysis(
            session_id,
            modes=body.modes,
            primary_mode=body.primary_mode,
            store_results=body.store_results,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/analysis")
async def delete_analysis(
    body: AnalysisDeleteRequest,
    _actor: RequireWritable,
    facade: AnalysisImmediateFacadeDep,
) -> dict[str, int]:
    if body.session_ids:
        owned = await facade.get_owned_session_ids(body.session_ids)
        if owned != set(body.session_ids):
            raise HTTPException(
                status_code=404, detail="One or more sessions not found"
            )
    deleted_count = await facade.delete_analysis(
        body.session_ids, all_versions=body.all_versions
    )
    return {"deleted_count": deleted_count}


@router.get("/analysis/delete-preview", response_model=AnalysisDeletePreview)
async def get_analysis_delete_preview(
    facade: AnalysisFacadeDep,
    session_ids: list[int] = Query(default=[]),
    all_versions: bool = Query(default=False),
) -> AnalysisDeletePreview:
    if not session_ids:
        return AnalysisDeletePreview(
            sessions_with_analysis=0,
            total_analysis_records=0,
            records_to_delete=0,
            patterns_count=0,
        )
    return await facade.get_delete_preview(session_ids, all_versions=all_versions)


@router.post("/analysis/batch", status_code=202, response_model=AnalysisJobEnqueued)
async def run_batch_analysis(
    body: BatchAnalysisRequest,
    facade: AnalysisFacadeDep,
    actor: RequireWritable,
) -> AnalysisJobEnqueued:
    from snore.analysis.modes.config import (  # noqa: PLC0415
        AVAILABLE_CONFIGS,
        DEFAULT_MODE,
    )

    if body.from_date is None and body.to_date is None and body.only is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of from_date, to_date, or only is required",
        )

    # Validate modes and primary_mode at the endpoint so invalid input fails fast.
    invalid_modes = [m for m in body.modes if m not in AVAILABLE_CONFIGS]
    if invalid_modes:
        raise HTTPException(status_code=422, detail=f"Unknown mode(s): {invalid_modes}")
    if body.primary_mode is not None and body.primary_mode not in body.modes:
        raise HTTPException(
            status_code=422,
            detail="primary_mode must be a member of modes",
        )
    if body.primary_mode is None and DEFAULT_MODE not in body.modes:
        raise HTTPException(
            status_code=422,
            detail="primary_mode must be supplied explicitly when aasm is not in modes",
        )

    from_dt = datetime.combine(body.from_date, time.min) if body.from_date else None
    to_dt = datetime.combine(body.to_date, time.max) if body.to_date else None

    session_ids = await facade.list_session_ids(
        from_date=from_dt, to_date=to_dt, only=body.only
    )
    if not session_ids:
        if body.only is None:
            detail = "No sessions found for the specified date range"
        else:
            detail = EMPTY_SELECTION_MESSAGES[body.only]
        raise HTTPException(status_code=422, detail=detail)

    if body.only is not None:
        # A filtered batch re-selects sessions that a queued job has not
        # written yet; enqueueing them again would store duplicate runs.
        in_flight = analysis_jobs.in_flight_session_ids(facade.profile_id)
        session_ids = [sid for sid in session_ids if sid not in in_flight]
        if not session_ids:
            raise HTTPException(
                status_code=422,
                detail="Every selected session is already queued for analysis",
            )

    aj = analysis_jobs.enqueue(
        profile_id=facade.profile_id,
        session_ids=session_ids,
        source=analysis_jobs.AnalysisJobSource.BATCH,
        owner_user_id=actor.user_id,
        modes=list(body.modes),
        primary_mode=body.primary_mode,
        store_results=body.store_results,
    )
    if aj is None:
        raise HTTPException(
            status_code=429, detail="Analysis queue is full; try again later"
        )

    return AnalysisJobEnqueued(job_id=aj.job_id, session_count=len(session_ids))


@router.get("/analysis/jobs", response_model=AnalysisJobsListResponse)
async def list_analysis_jobs(
    actor: RequireAuth,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AnalysisJobsListResponse:
    from snore.api.jobs import merge_job_lists, terminal_records_query  # noqa: PLC0415
    from snore.api.schemas import AnalysisJobStatus  # noqa: PLC0415
    from snore.database import models  # noqa: PLC0415

    in_memory: list[AnalysisJobStatus] = []
    in_memory_ids: set[str] = set()

    for aj in analysis_jobs.list_jobs(owner_user_id=actor.user_id):
        in_memory_ids.add(aj.job_id)
        in_memory.append(AnalysisJobStatus.model_validate(aj.to_dict()))

    def _db_row_to_status(rec: models.AnalysisJobRecord) -> AnalysisJobStatus:
        return AnalysisJobStatus(
            job_id=rec.job_id,
            state=analysis_jobs.AnalysisJobState(rec.state).value,
            source=rec.source,
            session_count=len(rec.session_ids_json) if rec.session_ids_json else 0,
            progress_completed=rec.progress_completed,
            progress_total=rec.progress_total,
            error_message=rec.error_message,
            created_at=rec.created_at,
            started_at=rec.started_at,
            finished_at=rec.finished_at,
            owner_user_id=rec.owner_user_id,
        )

    stmt = terminal_records_query(
        models.AnalysisJobRecord,
        actor.user_id,
        [s.value for s in analysis_jobs.TERMINAL_STATES],
    )
    db_rows = (await db.execute(stmt)).scalars().all()

    merged = merge_job_lists(
        in_memory,
        in_memory_ids,
        db_rows,
        to_status=_db_row_to_status,
        sort_key=lambda j: j.created_at,
    )
    return AnalysisJobsListResponse(jobs=merged)


@router.delete("/analysis/jobs/{job_id}", status_code=204)
async def cancel_analysis_job(job_id: str, actor: RequireWritable) -> None:
    from snore.api.jobs import cancel_or_409, owned_or_404  # noqa: PLC0415

    owned_or_404(
        analysis_jobs.get_job(job_id), actor.user_id, not_found_detail="Job not found"
    )
    cancel_or_409(
        analysis_jobs.cancel_job,
        job_id,
        already_detail="Analysis job is already finished and cannot be cancelled",
    )
