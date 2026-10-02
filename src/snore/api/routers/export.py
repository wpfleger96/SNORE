from __future__ import annotations

import asyncio
import shutil
import tempfile

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import Receive, Scope, Send

from snore.api.deps import ActorDep, get_db
from snore.constants import DEFAULT_RAW_BACKUP_DIR
from snore.services.export_service import ExportService

router = APIRouter()


class _TempExportResponse(FileResponse):
    """Serve an export file, then remove its temp dir however the response ends.

    Cleanup runs in ``__call__``'s ``finally`` rather than a ``BackgroundTask``
    because Starlette skips background tasks when the client disconnects
    mid-stream.
    """

    def __init__(
        self, tmpdir: Path, path: Path, media_type: str, filename: str
    ) -> None:
        super().__init__(path, media_type=media_type, filename=filename)
        self.tmpdir = tmpdir

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.to_thread(shutil.rmtree, self.tmpdir, ignore_errors=True)


@contextmanager
def _export_tmpdir() -> Iterator[Path]:
    """Temp dir for one export; removed if building the export fails.

    On success, ownership passes to the ``_TempExportResponse`` that streams it.
    """
    tmpdir = Path(tempfile.mkdtemp())
    try:
        yield tmpdir
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


@router.get("/csv")
async def export_csv(
    actor: ActorDep,
    db: AsyncSession = Depends(get_db),
    from_date: date | None = Query(default=None),
    to_date: date | None = Query(default=None),
    device: str | None = Query(default=None),
    include_waveforms: bool = Query(default=False),
) -> FileResponse:
    svc = ExportService(actor.profile_id)
    with _export_tmpdir() as tmpdir:
        # The service writes several CSV files into a directory; ship them as one zip.
        output_dir = tmpdir / "export"
        await svc.export_csv(
            db,
            output_dir,
            date_from=from_date,
            date_to=to_date,
            device_serial=device,
            include_waveforms=include_waveforms,
        )
        archive = await asyncio.to_thread(
            shutil.make_archive,
            str(tmpdir / "snore_export_csv"),
            "zip",
            root_dir=output_dir,
        )
        # Only the zip is needed while streaming; don't hold both copies on disk.
        await asyncio.to_thread(shutil.rmtree, output_dir)
        return _TempExportResponse(
            tmpdir, Path(archive), "application/zip", "snore_export_csv.zip"
        )


@router.get("/json")
async def export_json(
    actor: ActorDep,
    db: AsyncSession = Depends(get_db),
    from_date: date | None = Query(default=None),
    to_date: date | None = Query(default=None),
    device: str | None = Query(default=None),
) -> FileResponse:
    svc = ExportService(actor.profile_id)
    with _export_tmpdir() as tmpdir:
        output = tmpdir / "export.json"
        await svc.export_json(
            db,
            output,
            date_from=from_date,
            date_to=to_date,
            device_serial=device,
        )
        return _TempExportResponse(
            tmpdir, output, "application/json", "snore_export.json"
        )


@router.get("/raw")
def export_raw(
    actor: ActorDep,
    from_date: date | None = Query(default=None),
    to_date: date | None = Query(default=None),
    device: str | None = Query(default=None),
    trim_str: bool = Query(default=False),
    as_zip: bool = Query(default=True),
) -> FileResponse:
    # Backup root is always the actor's profile-scoped directory — never
    # client-supplied, to prevent cross-profile file access.
    backup_root = DEFAULT_RAW_BACKUP_DIR / str(actor.profile_id)
    svc = ExportService(actor.profile_id, backup_root=backup_root)
    with _export_tmpdir() as tmpdir:
        result = svc.export_raw(
            tmpdir / "snore_export_raw.zip",
            date_from=from_date,
            date_to=to_date,
            device_serial=device,
            trim_str=trim_str,
            as_zip=as_zip,
        )
        return _TempExportResponse(
            tmpdir, result.output_path, "application/zip", "snore_export_raw.zip"
        )
