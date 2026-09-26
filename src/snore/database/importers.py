"""
Session import functionality for converting UnifiedSession to database records.

Handles the complete import process including waveforms, events, and statistics.

Transaction ownership (§6)
---------------------------
- ``import_session``: delegates to ``import_sessions_batch`` with a single-element
  list and an explicit ``session_scope()`` — same UoW pattern as the batch path.
- ``import_sessions_batch``: requires a caller-provided ``db`` session; uses only
  ``begin_nested()`` savepoints so one failed session cannot poison the batch.
  The caller (``ImportService``) opens one ``session_scope()`` per bounded chunk
  and injects it here so the UoW boundary is owned at the service layer.

Bulk strategy (frozen in §90/PR-2)
------------------------------------
Waveforms, events, and settings use ``await db.execute(insert(Model), mappings)``
inside the per-session ``begin_nested()`` savepoint.  Core INSERT rows (Device,
Session, Day, Statistics) use ``add()`` / ``add_all()`` because they are small in
number, require identity-map access after flush, and benefit from ORM-level
change tracking.
"""

import itertools
import json
import logging
import re

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import NamedTuple

import numpy as np

from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database import models
from snore.database.day_manager import DayManager
from snore.database.session import session_scope
from snore.parsers.unified import UnifiedSession, WaveformData

logger = logging.getLogger(__name__)


class SessionImportResult(NamedTuple):
    """Result of a single-session import attempt.

    Invariant: ``extra_day_ids`` and ``deleted_session_ids`` can be non-empty
    even when ``imported`` is ``False``.  The legacy-format purge step runs
    unconditionally and is durable on the skip path by design — callers MUST
    drain both fields regardless of the value of ``imported``.  Treating
    ``imported=False`` as "nothing happened" will silently orphan day rows
    that need re-aggregation and leave stale PKs in accumulated ID lists.
    """

    imported: bool
    day_id: int | None
    session_id: int | None
    extra_day_ids: set[int]
    deleted_session_ids: set[int]


# Matches only the legacy noon-bucket ID format (e.g. "20260130_merged").
# New proximity-chained IDs (e.g. "20260130_225000_merged") must NOT match so
# they continue to flow through the normal all_covered replace/skip logic.
_OLD_MERGED_PATTERN = re.compile(r"^\d{8}_merged$")


async def _find_overlapping(
    db: AsyncSession,
    device_id: int,
    start_time: datetime,
    end_time: datetime,
    exclude_ids: frozenset[int] = frozenset(),
) -> list[models.Session]:
    """Return all sessions for device_id that strictly overlap [start_time, end_time).

    Strict inequality: adjacent segments where a.end == b.start do NOT overlap.
    Results are ordered by start_time so callers can reason about segment order.

    Callers MUST pass naive (tz-stripped) datetimes; Session.start_time and
    Session.end_time are device wall-clock columns stored without timezone info.

    Args:
        exclude_ids: Session PKs to exclude from the query.  Used by the force
            path to prevent the existing same-ID row from self-matching.
    """
    stmt = (
        select(models.Session)
        .where(
            models.Session.device_id == device_id,
            models.Session.start_time < end_time,
            models.Session.end_time > start_time,
        )
        .order_by(models.Session.start_time)
    )
    if exclude_ids:
        stmt = stmt.where(models.Session.id.not_in(exclude_ids))
    return list((await db.execute(stmt)).scalars().all())


def serialize_waveform(waveform: WaveformData) -> bytes:
    """
    Serialize waveform data to bytes for database storage.

    Stores timestamps and values as float32 numpy arrays.
    No compression - SQLite and filesystem handle that efficiently.

    Args:
        waveform: WaveformData object

    Returns:
        Serialized bytes
    """
    if isinstance(waveform.timestamps, list):
        timestamps = np.array(waveform.timestamps, dtype=np.float32)
    else:
        timestamps = waveform.timestamps.astype(np.float32)

    if isinstance(waveform.values, list):
        values = np.array(waveform.values, dtype=np.float32)
    else:
        values = waveform.values.astype(np.float32)

    data = np.column_stack([timestamps, values])
    return data.tobytes()


class SessionImporter:
    """Handles importing UnifiedSession objects to database using SQLAlchemy."""

    def __init__(self, profile_id: int) -> None:
        """Initialize importer.

        Args:
            profile_id: Profile that owns the devices and sessions created by this
                importer — required.  All ``Device`` rows are created with this
                ``profile_id``; device lookups are scoped to it so foreign devices
                are never matched.
        """
        self.profile_id = profile_id

    @staticmethod
    async def cleanup_orphaned_records(db: AsyncSession) -> dict[str, int]:
        """
        Remove orphaned records from child tables that reference non-existent sessions.

        This can happen if CASCADE delete is not enabled or if a database was corrupted.

        Uses typed ``delete()`` statements keyed on ``session_id`` FK — no f-string
        SQL, no internal commit (the caller's transaction owns commit/rollback).

        Args:
            db: SQLAlchemy async session (caller owns the transaction)

        Returns:
            Mapping of table name to the number of orphaned records removed from it.
            All four table keys (``settings``, ``events``, ``waveforms``,
            ``statistics``) are always present; values are 0 for tables with no
            orphaned rows.
        """
        orphan_tables = [
            models.Setting,
            models.Event,
            models.Waveform,
            models.Statistics,
        ]
        counts: dict[str, int] = {}

        for model_cls in orphan_tables:
            stmt = delete(model_cls).where(
                model_cls.session_id.notin_(  # type: ignore[attr-defined]
                    select(models.Session.id)
                )
            )
            result = await db.execute(stmt)
            count = max(result.rowcount, 0)  # type: ignore[attr-defined]
            if count > 0:
                logger.debug(
                    f"Cleaned {count} orphaned records from {model_cls.__tablename__}"
                )
            counts[model_cls.__tablename__] = count

        # Removing orphaned waveform rows frees SQLite rowids that a later import
        # can reuse; drop the id-keyed array cache so a reused id never serves a
        # deleted row's arrays.
        if counts[models.Waveform.__tablename__]:
            from snore.services.waveform_service import (  # noqa: PLC0415
                clear_waveform_array_cache,
            )

            clear_waveform_array_cache()

        # No db.commit() — caller owns the transaction boundary.
        return counts

    async def _import_single_session(
        self,
        db: AsyncSession,
        session_data: UnifiedSession,
        force: bool = False,
    ) -> SessionImportResult:
        """Import a single session and return a :class:`SessionImportResult`.

        This method NEVER aggregates — aggregation is handled by the caller.
        It does NOT open its own savepoint; callers that want per-session
        isolation must wrap each call in ``db.begin_nested()``.

        See :class:`SessionImportResult` for the caller contract, in particular
        the invariant that ``extra_day_ids`` and ``deleted_session_ids`` must be
        drained regardless of ``imported``.

        Args:
            db: SQLAlchemy async database session
            session_data: UnifiedSession to import
            force: If True, re-import existing sessions
        """
        profile_id = self.profile_id
        stmt = select(models.Device).where(
            models.Device.serial_number == session_data.device_info.serial_number,
            models.Device.profile_id == profile_id,
        )
        device = (await db.execute(stmt)).scalars().first()

        if device:
            device.manufacturer = session_data.device_info.manufacturer
            device.model = session_data.device_info.model
            device.firmware_version = session_data.device_info.firmware_version
            device.hardware_version = session_data.device_info.hardware_version
            device.product_code = session_data.device_info.product_code
            device.last_import = datetime.now(UTC)
        else:
            device = models.Device(
                profile_id=profile_id,
                manufacturer=session_data.device_info.manufacturer,
                model=session_data.device_info.model,
                serial_number=session_data.device_info.serial_number,
                firmware_version=session_data.device_info.firmware_version,
                hardware_version=session_data.device_info.hardware_version,
                product_code=session_data.device_info.product_code,
            )
            db.add(device)
            await db.flush()

        existing_stmt = select(models.Session).filter_by(
            device_id=device.id,
            device_session_id=session_data.device_session_id,
        )
        existing = (await db.execute(existing_stmt)).scalars().first()

        if existing and not force:
            logger.debug(
                f"Session {session_data.device_session_id} already exists, skipping"
            )
            return SessionImportResult(False, None, None, set(), set())

        # Strip tz info once here: Session.start_time / end_time are device
        # wall-clock columns stored without timezone info.
        naive_start = session_data.start_time.replace(tzinfo=None)
        naive_end = session_data.end_time.replace(tzinfo=None)

        # Overlap guard: run BEFORE any deletions so we never delete rows when
        # the final decision is "skip".  When force=True, exclude the existing
        # same-ID row from the overlap query so it doesn't self-match.
        exclude_ids = frozenset({existing.id}) if existing else frozenset()
        overlapping = await _find_overlapping(
            db,
            device_id=device.id,
            start_time=naive_start,
            end_time=naive_end,
            exclude_ids=exclude_ids,
        )

        replaced_day_ids: set[int] = set()
        deleted_session_ids: set[int] = set()
        if overlapping:
            # Partition: stale legacy noon-bucket rows (e.g. "20260130_merged") are
            # purged unconditionally so a plain re-import can replace them with the
            # new proximity-chained format.  The remaining rows are evaluated with
            # the existing all_covered replace/skip logic.
            old_format_rows = [
                row
                for row in overlapping
                if _OLD_MERGED_PATTERN.match(row.device_session_id)
            ]
            remaining_overlapping = [
                row
                for row in overlapping
                if not _OLD_MERGED_PATTERN.match(row.device_session_id)
            ]
            if old_format_rows:
                old_ids = [row.device_session_id for row in old_format_rows]
                replaced_day_ids.update(
                    row.day_id for row in old_format_rows if row.day_id is not None
                )
                deleted_session_ids.update(row.id for row in old_format_rows)
                for row in old_format_rows:
                    await db.delete(row)
                await db.flush()
                logger.info(
                    f"Overlap guard: purged stale old-format row(s) {old_ids} to "
                    f"allow re-import as {session_data.device_session_id}"
                )
            if remaining_overlapping:
                all_covered = all(
                    naive_start <= row.start_time and naive_end >= row.end_time
                    for row in remaining_overlapping
                )
                if all_covered:
                    # Incoming session fully covers every overlapping row: replace them.
                    replaced_ids = [
                        row.device_session_id for row in remaining_overlapping
                    ]
                    replaced_day_ids.update(
                        row.day_id
                        for row in remaining_overlapping
                        if row.day_id is not None
                    )
                    deleted_session_ids.update(row.id for row in remaining_overlapping)
                    for row in remaining_overlapping:
                        await db.delete(row)
                    await db.flush()
                    logger.info(
                        f"Overlap guard: replaced {replaced_ids} with incoming "
                        f"{session_data.device_session_id} "
                        f"({naive_start} – {naive_end})"
                    )
                else:
                    # Incoming session only partially overlaps with a session it does
                    # NOT fully cover: skip incoming without touching anything.
                    # Any old-format rows already deleted above are included in the
                    # return values so the batch can re-aggregate their days and prune
                    # their IDs.
                    overlap_ids = [
                        row.device_session_id for row in remaining_overlapping
                    ]
                    logger.warning(
                        f"Overlap guard: skipping {session_data.device_session_id} "
                        f"({naive_start} – {naive_end}) — "
                        f"partial overlap with existing {overlap_ids}"
                    )
                    return SessionImportResult(
                        False, None, None, replaced_day_ids, deleted_session_ids
                    )

        # Now that the import decision is final ("proceed"), delete the existing
        # same-ID row for force re-imports.  Doing this after the overlap check
        # ensures no row is deleted when the guard decides to skip.
        if existing:
            logger.debug(f"Force re-importing session {session_data.device_session_id}")
            if existing.day_id is not None:
                replaced_day_ids.add(existing.day_id)
            deleted_session_ids.add(existing.id)
            await db.delete(existing)
            await db.flush()

        notes_json = (
            json.dumps(session_data.data_quality_notes)
            if session_data.data_quality_notes
            else None
        )

        new_session = models.Session(
            device_id=device.id,
            device_session_id=session_data.device_session_id,
            start_time=naive_start,
            end_time=naive_end,
            duration_seconds=session_data.duration_seconds,
            therapy_mode=session_data.settings.mode.value
            if session_data.settings
            else None,
            import_source=session_data.import_source,
            parser_version=session_data.parser_version,
            data_quality_notes=notes_json,
            mask_on_segments=(
                [list(seg) for seg in session_data.mask_on_segments]
                if session_data.mask_on_segments is not None
                else None
            ),
            has_waveform_data=session_data.has_waveform_data,
            has_event_data=session_data.has_event_data,
            has_statistics=session_data.has_statistics,
        )
        db.add(new_session)
        await db.flush()

        day_date = DayManager.get_day_for_session(session_data.start_time)
        day = await DayManager.get_or_create_day(device.id, day_date, db)
        new_session.day_id = day.id
        day_id = day.id

        if session_data.has_waveform_data:
            await self._import_waveforms(db, new_session.id, session_data)

        if session_data.has_event_data:
            await self._import_events(db, new_session.id, session_data)

        if session_data.has_statistics:
            self._import_statistics(db, new_session.id, session_data)

        if session_data.settings:
            await self._import_settings(db, new_session.id, session_data)

        logger.debug(
            f"Imported session {session_data.device_session_id} from {session_data.start_time}"
        )
        # Compute which replaced day IDs need independent re-aggregation: those that
        # differ from the new session's day (a segment that crossed noon can produce a
        # replaced day_id distinct from day_id).
        extra_day_ids = replaced_day_ids - {day_id}
        return SessionImportResult(
            True, day_id, new_session.id, extra_day_ids, deleted_session_ids
        )

    async def import_session(
        self,
        session_data: UnifiedSession,
        force: bool = False,
        *,
        db: AsyncSession,
    ) -> bool:
        """Import a complete session using a caller-provided async session.

        The caller owns the UoW — this method uses only ``begin_nested()``
        savepoints via ``import_sessions_batch``.  No ``session_scope()`` is
        opened here; use the module-level ``import_session`` function for
        standalone imports with automatic scope ownership.

        Args:
            session_data: UnifiedSession to import
            force: If True, re-import existing sessions
            db: Required caller-provided async session.

        Returns:
            True if imported, False if skipped (already exists)
        """
        imported, _skipped, _failed, _ids = await self.import_sessions_batch(
            [session_data],
            force=force,
            batch_size=1,
            db=db,
        )
        return imported > 0

    async def import_sessions_batch(
        self,
        sessions: Iterable[UnifiedSession],
        force: bool = False,
        batch_size: int = 50,
        progress_callback: Callable[[str], None] | None = None,
        cancel_predicate: Callable[[], bool] | None = None,
        *,
        db: AsyncSession,
    ) -> tuple[int, int, int, list[int]]:
        """
        Import multiple sessions in batched transactions with per-session savepoints.

        Each session import is wrapped in ``begin_nested()`` so a single failed session
        releases its savepoint and increments ``failed`` without poisoning the rest of
        the batch.  Day statistics are aggregated at the end of each batch.

        Checks ``cancel_predicate()`` between batches.  If cancellation is requested
        the loop exits early; partial results accumulated so far are returned.

        ``sessions`` may be any iterable (list or lazy iterator); the method consumes
        it in bounded chunks of ``batch_size`` so no full-batch prefetch is required.

        Transaction ownership: the CALLER owns the transaction.  This method uses
        ``begin_nested()`` savepoints only.  The production caller is ``ImportService``,
        which opens one ``session_scope()`` per bounded chunk and injects the session
        here so the UoW boundary is explicit and owned at the service layer.

        For standalone one-session imports, use ``import_session()`` which opens its
        own explicit service-owned scope.

        Args:
            sessions: Iterable of UnifiedSession objects to import.  Need not be
                a list; a lazy generator is consumed in bounded ``batch_size`` chunks.
            force: If True, re-import existing sessions
            batch_size: Number of sessions per transaction (default: 50)
            progress_callback: Optional callback for progress messages
            cancel_predicate: Optional callable; returns True when cancellation
                is requested.  Checked between batches (not mid-batch).
            db: Required injected SQLAlchemy session.  The caller owns
                commit/rollback of the outer transaction.

        Returns:
            Tuple of (imported_count, skipped_count, failed_count, imported_session_ids).
            imported_session_ids are DB Session.id values for each successfully imported row.
        """
        imported = 0
        skipped = 0
        failed = 0
        all_imported_ids: list[int] = []

        session_iter = iter(sessions)

        for batch_num in itertools.count(1):
            # Check cancellation between batches.
            if cancel_predicate is not None and cancel_predicate():
                break

            batch = list(itertools.islice(session_iter, batch_size))
            if not batch:
                break

            batch_day_ids: set[int] = set()

            logger.debug(f"Importing batch {batch_num} ({len(batch)} sessions)")

            # Caller-owned transaction: use savepoints only.
            (
                imported,
                skipped,
                failed,
                batch_day_ids,
                batch_ids,
            ) = await self._import_batch_with_session(
                db,
                batch,
                force,
                imported,
                skipped,
                failed,
                batch_day_ids,
                progress_callback,
                all_imported_ids,
            )
            all_imported_ids = batch_ids  # batch mutates the list in-place; keep ref
            if batch_day_ids:
                for day_id in batch_day_ids:
                    day_record = await db.get(models.Day, day_id)
                    if day_record:
                        # A replaced session may have been the sole occupant of
                        # its Day row; recalculate_day prunes the orphaned row.
                        await DayManager.recalculate_day(day_record, db)

        # Force and overlap-replace imports delete existing sessions (cascading
        # to their waveforms) before inserting replacements, so SQLite may reuse
        # the freed rowids.  Clear unconditionally — a reused id serving a deleted
        # row's arrays is a correctness bug, while an over-clear only costs the
        # next reader one re-deserialize.
        from snore.services.waveform_service import (  # noqa: PLC0415
            clear_waveform_array_cache,
        )

        clear_waveform_array_cache()

        return imported, skipped, failed, all_imported_ids

    async def _import_batch_with_session(
        self,
        db: AsyncSession,
        batch: list[UnifiedSession],
        force: bool,
        imported: int,
        skipped: int,
        failed: int,
        batch_day_ids: set[int],
        progress_callback: Callable[[str], None] | None,
        batch_session_ids: list[int] | None = None,
    ) -> tuple[int, int, int, set[int], list[int]]:
        """Import one batch of sessions within a caller-provided session.

        Returns updated (imported, skipped, failed, batch_day_ids, imported_ids).
        imported_ids are the DB Session.id values for successfully imported sessions.
        """
        if batch_session_ids is None:
            batch_session_ids = []

        for session_data in batch:
            try:
                async with db.begin_nested():
                    result = await self._import_single_session(db, session_data, force)
                if result.imported:
                    imported += 1
                    if result.day_id:
                        batch_day_ids.add(result.day_id)
                    batch_day_ids.update(result.extra_day_ids)
                    # Prune any earlier-imported IDs that were deleted by the
                    # overlap-replace step so downstream consumers don't receive
                    # references to rows that no longer exist.
                    if result.deleted_session_ids:
                        batch_session_ids[:] = [
                            i
                            for i in batch_session_ids
                            if i not in result.deleted_session_ids
                        ]
                    if result.session_id is not None:
                        batch_session_ids.append(result.session_id)
                else:
                    skipped += 1
                    # Drain both sets unconditionally — old-format purges are durable
                    # on the skip path (see SessionImportResult invariant).
                    if result.extra_day_ids:
                        batch_day_ids.update(result.extra_day_ids)
                    if result.deleted_session_ids:
                        batch_session_ids[:] = [
                            i
                            for i in batch_session_ids
                            if i not in result.deleted_session_ids
                        ]
            except Exception as e:
                from snore.database.txn import _is_sqlite_contention  # noqa: PLC0415

                if _is_sqlite_contention(e):
                    # Re-raise so run_txn's retry loop can attempt a fresh session.
                    raise
                logger.error(
                    f"Failed to import session {session_data.device_session_id}: {e}"
                )
                failed += 1

            if progress_callback:
                sessions_done = imported + skipped + failed
                progress_callback(f"Importing session {sessions_done}...")

        return imported, skipped, failed, batch_day_ids, batch_session_ids

    async def _import_waveforms(
        self, db: AsyncSession, session_id: int, session_data: UnifiedSession
    ) -> None:
        """Import all waveforms for session using typed bulk INSERT."""
        if not session_data.waveforms:
            return

        mappings = []
        for waveform_type, waveform in session_data.waveforms.items():
            data_blob = serialize_waveform(waveform)
            sample_count = (
                len(waveform.values)
                if isinstance(waveform.values, list)
                else len(waveform.values)
            )
            mappings.append(
                {
                    "session_id": session_id,
                    "waveform_type": waveform_type.value,
                    "sample_rate": waveform.sample_rate,
                    "unit": waveform.unit,
                    "min_value": waveform.min_value,
                    "max_value": waveform.max_value,
                    "mean_value": waveform.mean_value,
                    "data_blob": data_blob,
                    "sample_count": sample_count,
                }
            )

        if mappings:
            await db.execute(insert(models.Waveform), mappings)
        logger.debug(f"Bulk imported {len(mappings)} waveforms")

    async def _import_events(
        self, db: AsyncSession, session_id: int, session_data: UnifiedSession
    ) -> None:
        """Import all respiratory events for session using typed bulk INSERT."""
        if not session_data.events:
            return

        mappings = [
            {
                "session_id": session_id,
                "event_type": event.event_type.value,
                "start_time": event.start_time,
                "duration_seconds": event.duration_seconds,
                "spo2_drop": event.spo2_drop,
                "peak_flow_limitation": event.peak_flow_limitation,
            }
            for event in session_data.events
        ]
        if mappings:
            await db.execute(insert(models.Event), mappings)

        logger.debug(f"Bulk imported {len(mappings)} events")

    def _import_statistics(
        self, db: AsyncSession, session_id: int, session_data: UnifiedSession
    ) -> None:
        """Import session statistics."""
        stats = session_data.statistics

        stats_record = models.Statistics(session_id=session_id, **stats.model_dump())
        db.add(stats_record)

        logger.debug("Imported session statistics")

    async def _import_settings(
        self, db: AsyncSession, session_id: int, session_data: UnifiedSession
    ) -> None:
        """Import session settings using typed bulk INSERT."""
        settings = session_data.settings

        if not settings:
            return

        settings_dict: dict[str, object] = settings.model_dump(
            mode="json", exclude={"other_settings"}, exclude_none=True
        )

        if settings.other_settings:
            settings_dict.update(settings.other_settings)

        # exclude_none only covers the main model fields; other_settings is
        # merged in afterward, so guard against None here to avoid persisting
        # the literal string "None". Floats are rounded to strip EDF
        # gain-arithmetic noise (28.900000000000002 → 28.9).
        mappings = [
            {
                "session_id": session_id,
                "key": key,
                "value": str(round(value, 2))
                if isinstance(value, float)
                else str(value),
            }
            for key, value in settings_dict.items()
            if value is not None
        ]

        if mappings:
            await db.execute(insert(models.Setting), mappings)

        logger.debug(f"Imported {len(mappings)} settings")


async def import_session(
    session_data: UnifiedSession, force: bool = False, *, profile_id: int
) -> bool:
    """
    Convenience function to import a session.

    Opens an explicit ``session_scope()`` (service-layer UoW ownership) and
    delegates to ``SessionImporter.import_session``.  The importer itself
    does not open any scopes.

    Args:
        session_data: UnifiedSession to import
        force: Force re-import if exists
        profile_id: Profile that owns the device — required.

    Returns:
        True if imported, False if skipped
    """
    importer = SessionImporter(profile_id)
    async with session_scope() as db:
        return await importer.import_session(session_data, force=force, db=db)
