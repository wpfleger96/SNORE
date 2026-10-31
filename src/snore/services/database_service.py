"""Database service for database statistics and metadata operations."""

import logging
import os
import shutil

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import ColumnElement, create_engine, delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database import models
from snore.services._base import ProfileScopedService, session_device_join
from snore.services.schemas import DatabaseStats, VacuumResult

__all__ = ["DatabaseService", "_vacuum_background", "file_size_mb"]

logger = logging.getLogger(__name__)


def file_size_mb(path: str) -> float:
    """Size of *path* in MiB; 0.0 when *path* is empty or missing."""
    if not path or not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def _vacuum_background(db_path: str) -> None:
    """Run VACUUM on a SQLite file path; intended for BackgroundTasks scheduling.

    FastAPI runs sync background-task functions in a thread pool, so this never
    blocks the event loop. No-op when db_path is not a valid SQLite file target.

    Uses the Python ``sqlite3`` module directly (no SQLAlchemy pool) with
    ``isolation_level=None`` (autocommit) so each statement runs outside any
    transaction.  VACUUM requires exclusive access; if any reader holds a lock,
    the VACUUM fails immediately and the failure is logged as a warning.  This
    is non-fatal — the DB is simply not compacted until the next scheduled run
    (e.g. after the next reset, or on the next server start with a VACUUM cron).
    """
    import sqlite3  # noqa: PLC0415

    if not DatabaseService.is_sqlite_target(db_path):
        logger.debug("_vacuum_background: skipping non-SQLite target %r", db_path)
        return
    conn: sqlite3.Connection | None = None
    try:
        # isolation_level=None → SQLite autocommit; VACUUM must run outside
        # an explicit transaction.  No SQLAlchemy pool, so the connection is
        # opened, used, and closed in a single call path.
        conn = sqlite3.connect(db_path, isolation_level=None)
        conn.execute("VACUUM")
        logger.info("Background VACUUM completed for %s", db_path)
        # Remove the pending-vacuum marker so a subsequent restart does not
        # schedule another VACUUM unnecessarily.
        try:
            from snore.constants import DEFAULT_VACUUM_PENDING_MARKER  # noqa: PLC0415

            DEFAULT_VACUUM_PENDING_MARKER.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not clear vacuum pending marker: %s", exc)
    except Exception as exc:
        logger.warning("Background VACUUM failed for %s (non-fatal): %s", db_path, exc)
    finally:
        if conn is not None:
            conn.close()


class DatabaseService(ProfileScopedService):
    """Service for database statistics and metadata operations."""

    async def _scoped_count(
        self,
        model: type[Any],
        *joins: tuple[type[Any], ColumnElement[bool]],
        where: Sequence[ColumnElement[bool]] = (),
        expr: ColumnElement[int] | None = None,
    ) -> int:
        """Profile-scoped COUNT over *model*, joined through *joins* in order.

        Every join chain must reach ``models.Device`` so ``_profile_filter()``
        applies.  ``expr`` overrides the default ``COUNT(*)`` (e.g. for
        ``COUNT(DISTINCT ...)``).
        """
        stmt = select(expr if expr is not None else func.count()).select_from(model)
        for target, onclause in joins:
            stmt = stmt.join(target, onclause)
        stmt = stmt.where(self._profile_filter(), *where)
        return (await self.db_session.execute(stmt)).scalar() or 0

    async def get_stats(self, db_path: str) -> DatabaseStats:
        """
        Query database statistics including table counts and coverage metrics.

        Args:
            db_path: Path to the database file (for file size calculation)

        Returns:
            DatabaseStats with all counts, percentages, and date range
        """
        session_to_device = (
            models.Device,
            models.Session.device_id == models.Device.id,
        )

        # Profile table is the one count not scoped to this profile.
        profile_count = (
            await self.db_session.execute(
                select(func.count()).select_from(models.Profile)
            )
        ).scalar() or 0
        device_count = await self._scoped_count(models.Device)
        session_count = await self._scoped_count(models.Session, session_to_device)
        day_count = await self._scoped_count(
            models.Day, (models.Device, models.Day.device_id == models.Device.id)
        )
        event_count = await self._scoped_count(
            models.Event,
            (models.Session, models.Event.session_id == models.Session.id),
            session_to_device,
        )
        waveform_count = await self._scoped_count(
            models.Waveform,
            (models.Session, models.Waveform.session_id == models.Session.id),
            session_to_device,
        )
        analysis_count = await self._scoped_count(
            models.AnalysisResult,
            (models.Session, models.AnalysisResult.session_id == models.Session.id),
            session_to_device,
        )
        pattern_count = await self._scoped_count(
            models.DetectedPattern,
            (
                models.AnalysisResult,
                models.DetectedPattern.analysis_result_id == models.AnalysisResult.id,
            ),
            (models.Session, models.AnalysisResult.session_id == models.Session.id),
            session_to_device,
        )

        sessions_with_waveforms = await self._scoped_count(
            models.Session,
            session_to_device,
            where=[models.Session.has_waveform_data.is_(True)],
        )
        sessions_with_events = await self._scoped_count(
            models.Session,
            session_to_device,
            where=[models.Session.has_event_data.is_(True)],
        )
        # Distinct sessions that have at least one analysis result (re-analysis of the
        # same session produces multiple rows; COUNT(DISTINCT) prevents over-counting).
        sessions_with_analysis = await self._scoped_count(
            models.AnalysisResult,
            (models.Session, models.AnalysisResult.session_id == models.Session.id),
            session_to_device,
            expr=func.count(func.distinct(models.AnalysisResult.session_id)),
        )
        # Sessions that have a flow waveform — the only sessions analysis can run on.
        # Uses Session.has_waveform_data=True only flags *any* waveform type, so we
        # join Waveform directly and filter on waveform_type == "flow".
        analyzable_session_count = await self._scoped_count(
            models.Session,
            session_to_device,
            (
                models.Waveform,
                (models.Waveform.session_id == models.Session.id)
                & (models.Waveform.waveform_type == "flow"),
            ),
        )

        first_session_raw = (
            await self.db_session.execute(
                session_device_join(
                    select(func.min(models.Session.start_time)).select_from(
                        models.Session
                    )
                ).where(self._profile_filter())
            )
        ).scalar()

        last_session_raw = (
            await self.db_session.execute(
                session_device_join(
                    select(func.max(models.Session.start_time)).select_from(
                        models.Session
                    )
                ).where(self._profile_filter())
            )
        ).scalar()

        first_session = None
        last_session = None
        if first_session_raw is not None:
            first_session = (
                datetime.fromisoformat(first_session_raw)
                if isinstance(first_session_raw, str)
                else first_session_raw
            )
        if last_session_raw is not None:
            last_session = (
                datetime.fromisoformat(last_session_raw)
                if isinstance(last_session_raw, str)
                else last_session_raw
            )

        size_bytes = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        size_mb = size_bytes / (1024 * 1024)

        waveform_coverage_pct = (
            (sessions_with_waveforms / session_count * 100) if session_count > 0 else 0
        )
        event_coverage_pct = (
            (sessions_with_events / session_count * 100) if session_count > 0 else 0
        )
        # Coverage = analyzed / analyzable (sessions with a flow waveform), not
        # total analysis rows / total sessions.  This stays in [0, 100] after
        # re-analysis and correctly excludes summary-only imports from the denominator.
        analysis_coverage_pct = (
            (sessions_with_analysis / analyzable_session_count * 100)
            if analyzable_session_count > 0
            else 0
        )

        return DatabaseStats(
            db_path=db_path,
            size_mb=size_mb,
            profile_count=profile_count,
            device_count=device_count,
            session_count=session_count,
            day_count=day_count,
            event_count=event_count,
            waveform_count=waveform_count,
            analysis_count=analysis_count,
            pattern_count=pattern_count,
            sessions_with_waveforms=sessions_with_waveforms,
            sessions_with_events=sessions_with_events,
            sessions_with_analysis=sessions_with_analysis,
            analyzable_session_count=analyzable_session_count,
            waveform_coverage_pct=waveform_coverage_pct,
            event_coverage_pct=event_coverage_pct,
            analysis_coverage_pct=analysis_coverage_pct,
            first_session=first_session,
            last_session=last_session,
        )

    def vacuum_sqlite(self, db_path: str) -> VacuumResult:
        """Run VACUUM on a SQLite file-backed database.

        Dispatched separately from row-reset so callers can vacuum without
        deleting, or skip vacuum on non-SQLite targets.

        Args:
            db_path: Path to the SQLite file.

        Raises:
            RuntimeError: If db_path is not a valid SQLite file target.
        """
        if not self.is_sqlite_target(db_path):
            raise RuntimeError(
                f"vacuum_sqlite() requires a SQLite file target; got {db_path!r}"
            )
        vacuum_engine = create_engine(
            f"sqlite:///{db_path}",
            isolation_level="AUTOCOMMIT",
        )
        try:
            with vacuum_engine.connect() as conn:
                conn.execute(text("VACUUM"))
        finally:
            vacuum_engine.dispose()

        size = (
            os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0.0
        )
        return VacuumResult(status="success", size_before_mb=size, size_after_mb=size)

    @staticmethod
    async def delete_user_data(
        db: AsyncSession,
        user_id: int,
        raw_root: Path,
    ) -> tuple[int, int, int]:
        """Delete all sleep data owned by *user_id*; commit; purge raw dirs.

        Deletes Device rows for every live profile owned by the user — the DB
        cascades (Device → Session/Day/Waveform/Event/Statistics/Setting/
        AnalysisResult/Breath/DetectedPattern) handle the rest.  Session has two
        FK paths to Day (via sessions.device_id and sessions.day_id); both
        cascades fire but are benign — by the time SQLite processes the Day-FK
        cascade, the Session row is already gone via the Device-FK cascade.

        Import job records (no FK, explicit delete) are also removed.

        Crash-safe cleanup (quarantine-before-commit pattern):

        Raw backup directories are quarantine-renamed into ``.quarantine/``
        BEFORE the commit so that a container restart between quarantine and
        commit leaves the dirs visible to ``DeletionSaga.recover()`` case 2 on
        next boot.  On commit failure the dirs are renamed back; on success
        they are rmtree'd best-effort.  ``CancelledError`` may skip the rmtree
        — startup recovery cleans up any ``.quarantine/`` leftovers.

        Trade-off: a crash between quarantine and commit means DB rows survive
        but raw backups are swept by startup recovery.  New imports proceed
        normally, but the raw archive is lost and cannot be rebuilt from the
        database.

        NOTE: Concurrent in-flight imports for this user's devices will fail if
        their device rows are deleted mid-import; this is accepted behavior.

        Commits the transaction.  Caller is responsible for writing the vacuum
        marker and scheduling vacuum.

        Returns:
            (devices_deleted, import_jobs_deleted, profiles_processed) counts.
        """
        from snore.services.profile_service import (
            quarantine_profile_raw_dir,  # noqa: PLC0415
        )

        profile_ids = list(
            (
                await db.execute(
                    select(models.Profile.id).where(
                        models.Profile.user_id == user_id,
                        models.Profile.deleting_at.is_(None),
                    )
                )
            ).scalars()
        )

        # Device rows per profile — RETURNING is bounded by a small device count.
        devices_deleted = 0
        for profile_id in profile_ids:
            result = await db.execute(
                delete(models.Device)
                .where(models.Device.profile_id == profile_id)
                .returning(models.Device.id)
            )
            devices_deleted += len(result.scalars().all())

        # Import records are potentially unbounded — use rowcount.
        import_cursor = await db.execute(
            delete(models.ImportJobRecord).where(
                models.ImportJobRecord.owner_user_id == user_id
            )
        )
        import_jobs_deleted = import_cursor.rowcount or 0  # type: ignore[attr-defined]

        # Quarantine raw dirs BEFORE committing.  A crash between quarantine
        # and commit leaves the dirs in .quarantine/ for startup recovery to
        # sweep; the DB rows survive so the state is consistent.
        quarantined: list[tuple[int, Path]] = []
        for pid in profile_ids:
            dst = quarantine_profile_raw_dir(pid, raw_root)
            if dst is not None:
                quarantined.append((pid, dst))

        try:
            await db.commit()
        except Exception:
            # Commit failed — restore quarantined dirs so raw backups are not
            # orphaned alongside the uncommitted DB rows.
            for pid, dst in quarantined:
                src = raw_root / str(pid)
                try:
                    dst.rename(src)
                except Exception:
                    logger.warning(
                        "Could not restore quarantined raw dir for profile %d after "
                        "commit failure; dir remains in .quarantine/ for startup recovery",
                        pid,
                    )
            raise

        # Commit succeeded — best-effort rmtree.  CancelledError may skip this;
        # startup recovery (DeletionSaga case 2) purges .quarantine/ leftovers.
        for pid, dst in quarantined:
            try:
                shutil.rmtree(dst, ignore_errors=True)
                logger.info("Purged raw backup for profile %d (delete-data)", pid)
            except Exception:
                logger.warning(
                    "Post-commit rmtree for profile %d failed; dir remains in "
                    ".quarantine/ for startup recovery",
                    pid,
                )

        return devices_deleted, import_jobs_deleted, len(profile_ids)

    @staticmethod
    def is_sqlite_target(db_path: str) -> bool:
        """Return True if *db_path* identifies a SQLite file target.

        Routes through ``DatabaseTarget`` to determine the dialect rather than
        relying on path-string heuristics.  A bare file path (no ``://``) is
        treated as a SQLite path by ``DatabaseTarget.from_url()``.  An explicit
        URL is parsed to extract the dialect; only ``"sqlite"`` with a non-memory
        location returns True.

        Examples:
            >>> DatabaseService.is_sqlite_target("/home/user/snore.db")
            True
            >>> DatabaseService.is_sqlite_target(":memory:")
            False
            >>> DatabaseService.is_sqlite_target("postgresql://user@host/db")
            False
            >>> DatabaseService.is_sqlite_target("sqlite:////abs/path.db")
            True
        """
        if not db_path or db_path == ":memory:":
            return False
        from snore.database.target import DatabaseTarget  # noqa: PLC0415

        try:
            target = DatabaseTarget.from_url(db_path)
        except (ValueError, Exception):
            return False  # Unrecognised — treat as non-SQLite to be safe
        return target.dialect == "sqlite" and target.location not in ("", ":memory:")
