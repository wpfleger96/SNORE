from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from snore.api.deps import service_dep
from snore.api.schemas import EventItem
from snore.database import models
from snore.services import EventService

router = APIRouter()

EventServiceDep = Annotated[EventService, Depends(service_dep(EventService))]


def _event_to_item(event: models.Event, session_start: datetime) -> EventItem:
    return EventItem(
        id=event.id,
        event_type=event.event_type,
        start_time=event.start_time.timestamp(),
        duration_seconds=event.duration_seconds,
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
