"""Session service for session listing, detail, deletion, and management operations."""

from datetime import datetime
from typing import Any

from sqlalchemy import ColumnElement, UnaryExpression, delete, func, select
from sqlalchemy.engine import CursorResult

from snore.constants import DEFAULT_LIST_SESSIONS_LIMIT
from snore.database import models
from snore.database.day_manager import DayManager
from snore.exceptions import NotFoundError
from snore.services._base import (
    ProfileScopedService,
    get_owned_session_ids,
    paginate,
    session_device_join,
)
from snore.services.mask_log_service import MaskLogService
from snore.services.schemas import (
    DeletePreview,
    SessionDetail,
    SessionListItem,
    SessionListResult,
    SessionSetting,
    SessionStatistics,
)
from snore.services.waveform_service import clear_waveform_array_cache
from snore.utils.db_chunk import iter_id_chunks

__all__ = ["SessionService"]


class SessionService(ProfileScopedService):
    """Service for session listing, detail, deletion, and management."""

    @staticmethod
    def _session_filters(
        device: str | None,
        from_date: datetime | None,
        to_date: datetime | None,
        include_disabled: bool,
    ) -> list[ColumnElement[bool]]:
        """Build shared WHERE conditions for session list and count queries."""
        filters: list[ColumnElement[bool]] = []

        if not include_disabled:
            filters.append(models.Session.enabled.is_(True))

        if device:
            filters.append(models.Device.serial_number == device)

        if from_date:
            filters.append(models.Session.start_time >= from_date)

        if to_date:
            filters.append(models.Session.start_time <= to_date)

        return filters

    async def list_sessions(
        self,
        device: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        limit: int = DEFAULT_LIST_SESSIONS_LIMIT,
        offset: int = 0,
        sort_by: str = "date-desc",
        include_disabled: bool = False,
    ) -> SessionListResult:
        """List sessions with filters, sorting, and pagination."""
        filters = self._session_filters(device, from_date, to_date, include_disabled)

        sort_map: dict[str, UnaryExpression[Any]] = {
            "date-asc": models.Session.start_time.asc(),
            "date-desc": models.Session.start_time.desc(),
            "session-id": models.Session.id.asc(),
            "duration": models.Session.duration_seconds.desc(),
        }

        list_query = (
            session_device_join(
                select(models.Session, models.Device, models.Statistics.ahi)
            )
            .outerjoin(
                models.Statistics, models.Session.id == models.Statistics.session_id
            )
            .where(self._profile_filter(), *filters)
        )

        result, total_count = await paginate(
            self.db_session,
            list_query,
            order_by=sort_map.get(sort_by, models.Session.start_time.desc()),
            limit=limit,
            offset=offset,
        )

        sessions = []
        for session, dev, ahi in result:
            sessions.append(
                SessionListItem(
                    id=session.id,
                    therapy_day=DayManager.get_day_for_session(session.start_time),
                    start_time=session.start_time,
                    duration_hours=(session.duration_seconds or 0.0) / 3600,
                    enabled=bool(session.enabled),
                    manufacturer=dev.manufacturer,
                    model=dev.model,
                    serial_number=dev.serial_number,
                    ahi=ahi,
                )
            )

        return SessionListResult(
            sessions=sessions, total_count=total_count, limit=limit
        )

    async def get_session_detail(
        self, session_id: int, include_settings: bool = False
    ) -> SessionDetail:
        """Get detailed information for a single session.

        Raises NotFoundError if the session doesn't exist or belongs to a
        different profile (foreign ID → 404, not 403, to avoid oracle attacks).
        """
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
        if not session:
            raise NotFoundError(f"Session {session_id} not found")

        device = (
            (
                await self.db_session.execute(
                    select(models.Device).where(models.Device.id == session.device_id)
                )
            )
            .scalars()
            .first()
        )

        stats_record = (
            (
                await self.db_session.execute(
                    select(models.Statistics).where(
                        models.Statistics.session_id == session.id
                    )
                )
            )
            .scalars()
            .first()
        )

        event_count = (
            await self.db_session.execute(
                select(func.count())
                .select_from(models.Event)
                .where(models.Event.session_id == session.id)
            )
        ).scalar() or 0

        waveform_count = (
            await self.db_session.execute(
                select(func.count())
                .select_from(models.Waveform)
                .where(models.Waveform.session_id == session.id)
            )
        ).scalar() or 0

        waveform_types = (
            (
                await self.db_session.execute(
                    select(models.Waveform.waveform_type)
                    .where(models.Waveform.session_id == session.id)
                    .distinct()
                )
            )
            .scalars()
            .all()
        )
        waveform_type_list = list(waveform_types)

        statistics = None
        if stats_record:
            statistics = SessionStatistics.model_validate(stats_record)

        settings_list = None
        if include_settings:
            settings_records = (
                (
                    await self.db_session.execute(
                        select(models.Setting)
                        .where(models.Setting.session_id == session.id)
                        .order_by(models.Setting.key)
                    )
                )
                .scalars()
                .all()
            )
            settings_list = [
                SessionSetting(key=s.key, value=s.value) for s in settings_records
            ]

        mask_service = MaskLogService(self.db_session, self.profile_id)
        active_mask = await mask_service.get_active_entry_for_date(
            session.start_time.date()
        )

        duration_seconds = session.duration_seconds or 0.0
        return SessionDetail(
            id=session.id,
            device_session_id=session.device_session_id,
            device_manufacturer=device.manufacturer if device else None,
            device_model=device.model if device else None,
            device_serial=device.serial_number if device else None,
            therapy_day=DayManager.get_day_for_session(session.start_time),
            start_time=session.start_time,
            end_time=session.end_time,
            duration_hours=duration_seconds / 3600,
            duration_seconds=duration_seconds,
            therapy_mode=session.therapy_mode,
            enabled=session.enabled,
            event_count=event_count,
            waveform_count=waveform_count,
            waveform_types=waveform_type_list,
            has_statistics=session.has_statistics,
            has_event_data=session.has_event_data,
            import_source=session.import_source,
            parser_version=session.parser_version,
            data_quality_notes=(
                session.data_quality_notes
                if isinstance(session.data_quality_notes, list)
                else []
            ),
            statistics=statistics,
            settings=settings_list,
            active_mask=active_mask,
        )

    async def get_delete_preview(
        self,
        device: str | None = None,
        session_ids: list[int] | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        delete_all: bool = False,
    ) -> DeletePreview:
        """Preview sessions and related data that would be deleted."""
        if not any([device, session_ids, from_date, to_date, delete_all]):
            raise ValueError(
                "At least one filter must be specified: "
                "--device, --session-id, --from, --to, or --all"
            )

        if session_ids:
            # Dedupe: chunked IN-binds don't implicitly de-duplicate like a single IN.
            session_ids = list(dict.fromkeys(session_ids))

        filters: list[ColumnElement[bool]] = []

        if device:
            filters.append(models.Device.serial_number == device)

        if from_date:
            filters.append(models.Session.start_time >= from_date)

        if to_date:
            filters.append(models.Session.start_time <= to_date)

        base_query = (
            session_device_join(select(models.Session, models.Device))
            .where(self._profile_filter(), *filters)
            .order_by(models.Session.start_time.desc())
        )

        # An explicit ID list is the only unbounded bind; chunk it and re-sort
        # after concatenation.  The filter-only branches stay a single execute.
        queries = (
            [
                base_query.where(models.Session.id.in_(chunk))
                for chunk in iter_id_chunks(session_ids)
            ]
            if session_ids
            else [base_query]
        )

        sessions = []
        session_ids_to_delete: list[int] = []
        for query in queries:
            for session, dev in await self.db_session.execute(query):
                sessions.append(
                    SessionListItem(
                        id=session.id,
                        therapy_day=DayManager.get_day_for_session(session.start_time),
                        start_time=session.start_time,
                        duration_hours=(session.duration_seconds or 0.0) / 3600,
                        enabled=True,
                        manufacturer=dev.manufacturer,
                        model=dev.model,
                        serial_number=dev.serial_number,
                        ahi=None,
                    )
                )
                session_ids_to_delete.append(session.id)

        if session_ids:
            sessions.sort(key=lambda s: s.start_time, reverse=True)

        if not session_ids_to_delete:
            return DeletePreview(
                sessions=[], event_count=0, waveform_count=0, stats_count=0
            )

        # Split into three single-IN COUNTs summed over chunks: packing all
        # three IN-lists into one statement would bind 3x the chunk size and
        # overrun SQLite's parameter cap.
        async def _count_related(model: type[Any]) -> int:
            total = 0
            for chunk in iter_id_chunks(session_ids_to_delete):
                total += (
                    await self.db_session.execute(
                        select(func.count())
                        .select_from(model)
                        .where(model.session_id.in_(chunk))
                    )
                ).scalar() or 0
            return total

        event_count = await _count_related(models.Event)
        waveform_count = await _count_related(models.Waveform)
        stats_count = await _count_related(models.Statistics)

        return DeletePreview(
            sessions=sessions,
            event_count=event_count,
            waveform_count=waveform_count,
            stats_count=stats_count,
        )

    async def delete_sessions(self, session_ids: list[int]) -> int:
        """Delete sessions by ID list.

        All requested IDs must belong to this profile.  If any are foreign the
        caller should have validated first (via ``get_owned_ids``) and returned
        404 before calling this method.  The DELETE itself carries the predicate
        as a defence-in-depth measure.

        Day aggregates for the affected days are recalculated after the DELETE
        (mirrors ``set_session_enabled``), so a day left with fewer sessions is
        re-aggregated and a day left with no sessions at all is pruned (see
        ``DayManager.recalculate_day``).
        """
        # Dedupe: chunked IN-binds don't implicitly de-duplicate like a single IN.
        session_ids = list(dict.fromkeys(session_ids))
        if not session_ids:
            return 0

        # Collect affected day IDs before the rows disappear.  Chunks are
        # disjoint but a day can recur across them, so union into a set.
        day_ids: set[int] = set()
        for chunk in iter_id_chunks(session_ids):
            rows = (
                (
                    await self.db_session.execute(
                        session_device_join(select(models.Session.day_id))
                        .where(
                            models.Session.id.in_(chunk),
                            models.Session.day_id.is_not(None),
                            self._profile_filter(),
                        )
                        .distinct()
                    )
                )
                .scalars()
                .all()
            )
            day_ids.update(d for d in rows if d is not None)

        # Ownership predicate inside the DELETE — foreign IDs cannot be deleted
        # even if the caller skips the pre-validation.
        deleted = 0
        for chunk in iter_id_chunks(session_ids):
            cursor: CursorResult[Any] = await self.db_session.execute(  # type: ignore[assignment]
                delete(models.Session)
                .where(models.Session.id.in_(chunk))
                .where(
                    models.Session.device_id.in_(
                        select(models.Device.id).where(self._profile_filter())
                    )
                )
            )
            deleted += cursor.rowcount or 0

        # Deleting sessions cascades to their waveform rows, freeing SQLite
        # rowids that a later import can reuse.  Drop the id-keyed array cache so
        # a reused id never serves a deleted row's arrays.
        clear_waveform_array_cache()

        # day_ids were collected under the profile filter; the join repeats it
        # as defence-in-depth because recalculate_day can delete the row.
        for chunk in iter_id_chunks(list(day_ids)):
            days = (
                (
                    await self.db_session.execute(
                        select(models.Day)
                        .join(models.Device, models.Day.device_id == models.Device.id)
                        .where(models.Day.id.in_(chunk), self._profile_filter())
                    )
                )
                .scalars()
                .all()
            )
            for day in days:
                await DayManager.recalculate_day(day, self.db_session)

        return deleted

    async def get_owned_ids(self, session_ids: list[int]) -> set[int]:
        """Return the subset of session_ids that belong to this profile.

        Used by routes to validate ownership before mutation: any ID absent from
        the returned set is either missing or owned by a different profile.
        """
        return await get_owned_session_ids(
            self.db_session, self.profile_id, session_ids
        )

    async def set_session_enabled(self, session_id: int, enabled: bool) -> None:
        """Toggle session enabled/disabled status.

        Raises NotFoundError if the session doesn't exist or belongs to a
        different profile (foreign ID → 404 to avoid oracle attacks).
        """
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
        if not session:
            raise NotFoundError(f"Session {session_id} not found")

        if session.enabled == enabled:
            return

        session.enabled = enabled
        await self.db_session.flush()

        if session.day_id:
            day = (
                (
                    await self.db_session.execute(
                        select(models.Day).where(models.Day.id == session.day_id)
                    )
                )
                .scalars()
                .first()
            )
            if day:
                await DayManager.recalculate_day(day, self.db_session)

    async def resolve_session_id(
        self, session_id: int | None, date: datetime | None
    ) -> int:
        """Resolve session ID from either explicit ID or date.

        When ``session_id`` is supplied, it is validated against the profile
        predicate so a foreign ID is treated as not found rather than echoed
        back to the caller.
        """
        if session_id is not None:
            # Validate ownership — foreign session → not found.
            owned = (
                await self.db_session.execute(
                    session_device_join(select(models.Session.id)).where(
                        models.Session.id == session_id,
                        self._profile_filter(),
                    )
                )
            ).scalar_one_or_none()
            if owned is None:
                raise ValueError(f"Session {session_id} not found")
            return session_id

        if date is None:
            raise ValueError("Either session_id or date must be provided")

        session = (
            (
                await self.db_session.execute(
                    session_device_join(select(models.Session))
                    .join(models.Day, models.Session.day_id == models.Day.id)
                    .where(
                        models.Day.date == date.date(),
                        self._profile_filter(),
                    )
                )
            )
            .scalars()
            .first()
        )

        if not session:
            raise ValueError(f"No session found for date {date.date()}")

        return int(session.id)
