from __future__ import annotations

import asyncio
import shutil
import tempfile

from collections.abc import Iterator
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from snore.api.deps import ActorDep, get_db
from snore.constants import DEFAULT_RAW_BACKUP_DIR
from snore.services.export_service import ExportService

router = APIRouter()


def _stream_file(path: Path, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk


def _streaming_export(
    tmpdir: str,
    output_path: Path,
    media_type: str,
    filename: str,
) -> StreamingResponse:
    return StreamingResponse(
        _stream_file(output_path),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
        background=BackgroundTask(shutil.rmtree, tmpdir),
    )


@router.get("/csv")
async def export_csv(
    actor: ActorDep,
    db: AsyncSession = Depends(get_db),
    from_date: date | None = Query(default=None),
    to_date: date | None = Query(default=None),
    device: str | None = Query(default=None),
    include_waveforms: bool = Query(default=False),
) -> StreamingResponse:
    svc = ExportService(actor.profile_id)
    tmpdir = tempfile.mkdtemp()
    # The service writes several CSV files into a directory; ship them as one zip.
    output_dir = Path(tmpdir) / "export"
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
        str(Path(tmpdir) / "snore_export"),
        "zip",
        root_dir=output_dir,
    )
    return _streaming_export(
        tmpdir, Path(archive), "application/zip", "snore_export.zip"
    )


@router.get("/json")
async def export_json(
    actor: ActorDep,
    db: AsyncSession = Depends(get_db),
    from_date: date | None = Query(default=None),
    to_date: date | None = Query(default=None),
    device: str | None = Query(default=None),
) -> StreamingResponse:
    svc = ExportService(actor.profile_id)
    tmpdir = tempfile.mkdtemp()
    output = Path(tmpdir) / "export.json"
    await svc.export_json(
        db,
        output,
        date_from=from_date,
        date_to=to_date,
        device_serial=device,
    )
    return _streaming_export(tmpdir, output, "application/json", "snore_export.json")


@router.get("/raw")
def export_raw(
    actor: ActorDep,
    from_date: date | None = Query(default=None),
    to_date: date | None = Query(default=None),
    device: str | None = Query(default=None),
    trim_str: bool = Query(default=False),
    as_zip: bool = Query(default=True),
) -> StreamingResponse:
    # Backup root is always the actor's profile-scoped directory — never
    # client-supplied, to prevent cross-profile file access.
    backup_root = DEFAULT_RAW_BACKUP_DIR / str(actor.profile_id)
    svc = ExportService(actor.profile_id, backup_root=backup_root)
    tmpdir = tempfile.mkdtemp()
    output = Path(tmpdir) / "snore_export_raw.zip"
    result = svc.export_raw(
        output,
        date_from=from_date,
        date_to=to_date,
        device_serial=device,
        trim_str=trim_str,
        as_zip=as_zip,
    )
    return _streaming_export(
        tmpdir, result.output_path, "application/zip", "snore_export_raw.zip"
    )
