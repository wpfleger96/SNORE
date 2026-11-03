"""ImportService — orchestrates the CPAP data import pipeline."""

from __future__ import annotations

import logging

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.importers import SessionImporter
from snore.database.session import session_scope
from snore.database.txn import run_txn
from snore.database.write_gate import write_gate
from snore.parsers.register_all import ensure_registered_parsers
from snore.parsers.registry import parser_registry
from snore.parsers.unified import UnifiedSession
from snore.services.schemas import ImportResult, ImportSource, ImportSourceResult

logger = logging.getLogger(__name__)

__all__ = ["ImportService", "normalize_datalog_suffix", "safe_relative_path"]


def safe_relative_path(filename: str) -> str | None:
    """Sanitize a multipart filename, preserving relative directory structure."""
    parts = PurePosixPath(filename.replace("\\", "/")).parts
    safe = [
        clean
        for p in parts
        if (clean := p.replace("\x00", ""))
        and clean not in (".", "..")
        and not clean.startswith("/")
        and not (len(clean) == 2 and clean[1] == ":")
    ]
    return "/".join(safe) or None


def normalize_datalog_suffix(path: str) -> str | None:
    """Return the DATALOG-relative suffix of *path*, lowercased, or None.

    Client paths carry an arbitrary selection-root prefix (e.g. "SDCARD/")
    that varies by OS and SD-card mount point. FAT filesystems are
    case-insensitive, so the suffix is lowercased for reliable comparison.

    Example:
        "SDCARD/DATALOG/2024/20240101_010000_BRP.edf"
        → "datalog/2024/20240101_010000_brp.edf"
    """
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part.lower() == "datalog":
            return "/".join(parts[i:]).lower()
    return None


class ImportService:
    """Orchestrates the CPAP data import pipeline."""

    def detect_sources(self, path: Path) -> list[ImportSource]:
        """Detect all importable data sources under path."""
        ensure_registered_parsers()
        results = parser_registry.detect_all_parsers(path)

        sources: list[ImportSource] = []
        for parser, detection in results:
            meta = detection.metadata or {}
            all_roots = meta.get("all_roots", [])

            if not all_roots:
                sources.append(
                    ImportSource(
                        parser_name=parser.parser_id,
                        device_serial=meta.get("device_serial"),
                        profile_name=meta.get("profile_name"),
                        structure_type=meta.get("structure_type"),
                        root_path=str(meta.get("data_root") or path),
                        data_root=meta.get("data_root"),
                    )
                )
            else:
                root_metadata = meta.get("root_metadata", {})
                for root_path in all_roots:
                    root_info = root_metadata.get(root_path, {})
                    sources.append(
                        ImportSource(
                            parser_name=parser.parser_id,
                            device_serial=root_info.get(
                                "device_serial", meta.get("device_serial")
                            ),
                            profile_name=root_info.get(
                                "profile_name", meta.get("profile_name")
                            ),
                            structure_type=root_info.get(
                                "structure_type", meta.get("structure_type")
                            ),
                            root_path=root_path,
                            data_root=root_info.get("data_root", meta.get("data_root")),
                        )
                    )

        return sources

    async def import_sources(
        self,
        sources: list[ImportSource],
        *,
        force: bool = False,
        batch_size: int = 50,
        backup: bool = True,
        backup_root: Path | None = None,
        sort_by: str | None = None,
        limit: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        parallel: bool = True,
        dry_run: bool = False,
        force_cleanup: bool = False,
        progress_callback: Callable[[str], None] | None = None,
        cancel_predicate: Callable[[], bool] | None = None,
        profile_id: int,
    ) -> ImportResult:
        """Run the import pipeline for the selected sources.

        Backup (optional) → parse → import (or dry-run count). Returns aggregate
        counts and per-source breakdown.

        Args:
            cancel_predicate: Optional callable that returns True when the caller
                has requested cancellation.  Checked between sources and at each
                batch boundary inside ``SessionImporter``.
            force_cleanup: If True (and not dry_run), remove orphaned child-table
                records before importing.  Defaults to False; on-demand cleanup is
                available via ``snore db cleanup-orphans``.
            profile_id:  Resolved profile ID — required.  All devices and sessions
                         created during this import are owned by this profile.
        """

        def emit(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        def is_cancelled() -> bool:
            return cancel_predicate is not None and cancel_predicate()

        source_results: list[ImportSourceResult] = []
        total_imported = 0
        total_skipped = 0
        total_failed = 0
        all_imported_ids: list[int] = []

        # Default backup root is namespaced by profile so raw files from
        # different profiles never share a directory — mirrors ExportService.
        if backup_root is None:
            from snore.constants import DEFAULT_RAW_BACKUP_DIR  # noqa: PLC0415

            backup_root = DEFAULT_RAW_BACKUP_DIR / str(profile_id)

        # Profile-declared IANA timezone: parsers whose source data encodes
        # absolute instants (OSCAR epoch-ms) use it to emit device-local
        # wall-clock times matching the ResMed contract; None keeps legacy
        # UTC wall-clock (A6).
        from snore.database.models import Profile  # noqa: PLC0415

        async with session_scope() as db_session:
            profile = await db_session.get(Profile, profile_id)
            timezone_name = profile.timezone if profile else None

        # Validate the stored name eagerly.  ZoneInfoNotFoundError is a
        # KeyError subclass, so if it escaped from the lazy parse generators
        # it would bypass the CLI's RuntimeError handling and surface as a raw
        # traceback.  Re-raise as RuntimeError — the error type callers of
        # import_sources already render cleanly.
        if timezone_name is not None:
            try:
                ZoneInfo(timezone_name)
            except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Stored profile timezone {timezone_name!r} is not a valid "
                    "IANA timezone. Fix it with 'snore profile set-timezone "
                    "<ZONE>' and re-run the import."
                ) from exc

        # force_cleanup is retained for programmatic callers and tests; the
        # intended user-facing path is `snore db cleanup-orphans`.
        if not dry_run and force_cleanup:
            async with write_gate():
                async with session_scope(immediate=True) as db_session:
                    cleanup_counts = await SessionImporter.cleanup_orphaned_records(
                        db_session
                    )
                    orphaned = sum(cleanup_counts.values())
                    if orphaned > 0:
                        emit(f"Cleaned up {orphaned} orphaned records from database")

        parser_map = {p.parser_id: p for p in parser_registry.list_parsers()}

        for source in sources:
            if is_cancelled():
                break

            parser = parser_map.get(source.parser_name)
            if parser is None:
                logger.warning(
                    "Parser %r not found — skipping source", source.parser_name
                )
                source_results.append(
                    ImportSourceResult(
                        source=source,
                        imported=0,
                        skipped=0,
                        failed=0,
                        warnings=[f"Parser {source.parser_name!r} not found"],
                    )
                )
                continue

            source_root = Path(source.root_path)
            warnings: list[str] = []

            # Back up raw files, then parse the source, not the backup copy: the
            # backup is a cumulative per-device archive, so parsing the source
            # keeps each import to the files it brought. Rescan is the path
            # that parses the archive.
            if backup and not dry_run and parser.supports_raw_backup:
                device_serial = source.device_serial or ""
                if device_serial:
                    from snore.services.backup_service import BackupService

                    backup_svc = BackupService(backup_root)
                    try:
                        emit("Backing up raw files...")
                        backup_result = backup_svc.backup_via_parser(
                            parser,
                            source_root,
                            device_serial,
                            progress_callback=emit,
                        )
                        if backup_result.was_skipped:
                            emit(f"Backup skipped: {backup_result.skipped_reason}")
                        else:
                            emit(f"Backed up to {backup_result.backup_root}")
                            if backup_result.files_failed:
                                msg = (
                                    f"Backup partial: {backup_result.files_failed} "
                                    "file(s) could not be copied"
                                )
                                warnings.append(msg)
                                emit(msg)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Backup failed: {exc}\nUse backup=False to skip backup."
                        ) from exc
                else:
                    warnings.append("No device serial — backup skipped")
                    emit("No device serial found — skipping backup")

            if is_cancelled():
                break

            # Parse sessions lazily — no full-batch prefetch.
            # parse_sessions() returns a generator; import_sessions_batch consumes
            # it in bounded batch_size chunks so memory is bounded per-batch.
            session_iter = parser.parse_sessions(
                source_root,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
                sort_by=sort_by,
                parallel=parallel,
                progress_callback=emit,
                timezone_name=timezone_name,
            )
            emit("Detected sessions — starting import")

            if dry_run:
                # Count what would be imported without writing to DB.
                # Consume the iterator to get the count (dry_run materializes anyway).
                sessions_list = list(session_iter)
                source_results.append(
                    ImportSourceResult(
                        source=source,
                        imported=len(sessions_list),
                        skipped=0,
                        failed=0,
                        warnings=warnings,
                    )
                )
                total_imported += len(sessions_list)
                continue

            # Import — ImportService opens ONE scope per bounded batch chunk.
            # Each chunk is committed separately; a failed chunk does not poison
            # subsequent chunks.
            importer = SessionImporter(profile_id)
            emit("Importing sessions...")
            imported = 0
            skipped = 0
            failed = 0
            source_imported_ids: list[int] = []
            session_iter_internal = iter(session_iter)
            import itertools as _itertools  # noqa: PLC0415

            for _chunk_num in _itertools.count(1):
                if cancel_predicate is not None and cancel_predicate():
                    break
                chunk = list(_itertools.islice(session_iter_internal, batch_size))
                if not chunk:
                    break

                # run_txn opens a fresh session per attempt and retries on
                # SQLite contention.  UNIQUE(device_id, device_session_id)
                # makes this idempotent: a replay of the same chunk produces
                # the same rows and skips duplicates.
                async def _import_chunk(
                    db: AsyncSession,
                    *,
                    _chunk: list[UnifiedSession] = chunk,
                    _importer: SessionImporter = importer,
                ) -> tuple[int, int, int, list[int]]:
                    return await _importer.import_sessions_batch(
                        iter(_chunk),
                        force=force,
                        batch_size=batch_size,
                        progress_callback=progress_callback,
                        cancel_predicate=cancel_predicate,
                        db=db,
                    )

                async with write_gate():
                    ci, cs, cf, chunk_ids = await run_txn(_import_chunk)
                imported += ci
                skipped += cs
                failed += cf
                source_imported_ids.extend(chunk_ids)

            total_imported += imported
            total_skipped += skipped
            total_failed += failed
            all_imported_ids.extend(source_imported_ids)

            source_results.append(
                ImportSourceResult(
                    source=source,
                    imported=imported,
                    skipped=skipped,
                    failed=failed,
                    warnings=warnings,
                    imported_session_ids=source_imported_ids,
                )
            )

        return ImportResult(
            total_imported=total_imported,
            total_skipped=total_skipped,
            total_failed=total_failed,
            sources=source_results,
            warnings=[],
            imported_session_ids=all_imported_ids,
        )
