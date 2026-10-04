from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from snore.api.deps import service_dep
from snore.api.errors import NotFoundError
from snore.api.schemas import EventItem
from snore.database import models
from snore.services import AnalysisFacade, EventService
from snore.services.schemas import EventMatchResult

router = APIRouter()

EventServiceDep = Annotated[EventService, Depends(service_dep(EventService))]
AnalysisFacadeDep = Annotated[AnalysisFacade, Depends(service_dep(AnalysisFacade))]


def _event_to_item(event: models.Event, session_start: datetime) -> EventItem:
    return EventItem(
        id=event.id,
        event_type=event.event_type,
        start_time=event.start_time.timestamp(),
        duration_seconds=event.duration_seconds or 0.0,
        offset_seconds=(event.start_time - session_start).total_seconds(),
        spo2_drop=event.spo2_drop,
        peak_flow_limitation=event.peak_flow_limitation,
    )


@router.get("/{session_id}/events", response_model=list[EventItem])
async def list_events(
    session_id: int,
    svc: EventServiceDep,
    event_type: str | None = Query(default=None),
) -> list[EventItem]:
    events, session_start = await svc.list_session_events(session_id, event_type)
    return [_event_to_item(e, session_start) for e in events]


@router.get("/{session_id}/events/match", response_model=EventMatchResult)
async def match_events(
    session_id: int,
    svc: EventServiceDep,
    facade: AnalysisFacadeDep,
    mode: str = Query(default="aasm"),
) -> EventMatchResult:
    machine_times = await svc.get_machine_apnea_hypopnea_times(session_id)

    analysis = await facade.get_analysis_result(session_id)
    if not analysis:
        raise NotFoundError(f"No analysis results found for session {session_id}")

    programmatic_times: list[float] = []
    mode_result = analysis.mode_results.get(mode)
    if mode_result is not None:
        for apnea in mode_result.apneas:
            programmatic_times.append(analysis.timestamp_start + apnea.start_time)
        for hyp in mode_result.hypopneas:
            programmatic_times.append(analysis.timestamp_start + hyp.start_time)
        programmatic_times.sort()

    return svc.match_events(machine_times, programmatic_times)
