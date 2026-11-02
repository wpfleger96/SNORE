"""Event query service for device-scored session events."""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database import models
from snore.exceptions import NotFoundError
from snore.services._base import ProfileScopedService, session_device_join

__all__ = ["EventService", "device_event_types", "reported_count"]


async def device_event_types(
    db_session: AsyncSession, device_ids: Collection[int]
) -> dict[int, set[str]]:
    """Return the event types each device has ever scored.

    Covers all of a device's sessions, including disabled ones: one scored event
    anywhere makes that device's zeros real.  Every requested id gets an entry.
    """
    reported: dict[int, set[str]] = {device_id: set() for device_id in device_ids}
    rows = await db_session.execute(
        select(models.Session.device_id, models.Event.event_type)
        .join(models.Event, models.Event.session_id == models.Session.id)
        .where(models.Session.device_id.in_(reported))
        .distinct()
    )
    for device_id, event_type in rows:
        reported[device_id].add(event_type)
    return reported


def reported_count(
    value: int | None, event_type: str, reported: Collection[str]
) -> int | None:
    """Return a device event count, or None for a zero that isn't a measurement.

    A zero only means "none happened" on a device that has scored ``event_type``
    at least once; otherwise the device never reports that type.
    """
    return value if value or event_type in reported else None


class EventService(ProfileScopedService):
    """Service for querying a session's device-scored events."""

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
