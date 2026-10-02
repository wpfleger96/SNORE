"""Event matching service for comparing machine vs programmatic detections."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select

from snore.analysis.modes.postprocess import (
    EVENT_MATCH_TOLERANCE_SECONDS,
    match_events_by_start_time,
)
from snore.database import models
from snore.exceptions import NotFoundError
from snore.services._base import ProfileScopedService, session_device_join
from snore.services.schemas import EventMatchResult

__all__ = ["EVENT_MATCH_TOLERANCE_SECONDS", "EventService"]


class _TimedEvent(BaseModel, frozen=True):
    """A bare timestamp shaped for ``match_events_by_start_time``."""

    start_time: float


class EventService(ProfileScopedService):
    """Service for event matching and comparison."""

    async def _get_session_owned(self, session_id: int) -> Any:
        """Return the session if it belongs to this profile, else raise NotFoundError."""
        session = (
            (
                await self.db_session.execute(
                    session_device_join(select(models.Session)).where(
                        models.Session.id == session_id,
                        self._profile_filter(),
                    )
                )
            )
            .scalars()
            .first()
        )
        if session is None:
            raise NotFoundError(f"Session {session_id} not found")
        return session

    async def list_session_events(
        self,
        session_id: int,
        event_type: str | None = None,
    ) -> tuple[list[Any], datetime]:
        """Return (events, session_start) for a session."""

        session = await self._get_session_owned(session_id)

        stmt = select(models.Event).where(models.Event.session_id == session_id)
        if event_type:
            stmt = stmt.where(models.Event.event_type == event_type)
        stmt = stmt.order_by(models.Event.start_time)
        events = list((await self.db_session.execute(stmt)).scalars().all())
        return events, session.start_time

    async def get_machine_event_times(self, session_id: int) -> list[float]:
        """Return sorted machine event timestamps for a session."""

        await self._get_session_owned(session_id)

        events = (
            (
                await self.db_session.execute(
                    select(models.Event).where(models.Event.session_id == session_id)
                )
            )
            .scalars()
            .all()
        )
        return sorted(e.start_time.timestamp() for e in events)

    @staticmethod
    def match_events(
        machine_times: list[float],
        programmatic_times: list[float],
        tolerance: float = EVENT_MATCH_TOLERANCE_SECONDS,
    ) -> EventMatchResult:
        """Match machine vs programmatic events one-to-one, like batch validation."""
        result = match_events_by_start_time(
            [_TimedEvent(start_time=t) for t in sorted(programmatic_times)],
            [_TimedEvent(start_time=t) for t in sorted(machine_times)],
            tolerance,
        )
        return EventMatchResult(
            machine_count=len(machine_times),
            programmatic_count=len(programmatic_times),
            matched=len(result.matched),
            false_positives=len(result.false_positives),
            false_negatives=len(result.false_negatives),
        )
