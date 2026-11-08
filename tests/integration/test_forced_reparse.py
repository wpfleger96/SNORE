"""Forced re-parse and the other import options against a real database.

Runs the import worker on the recorded ResMed SD card, copied into a profile's
raw archive for rescans or into an upload spool for uploads, to check what
``force``, the date range and ``analyze`` do to sessions already imported.
"""

from __future__ import annotations

import asyncio
import shutil

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sqlalchemy import select, update

from snore.api.import_jobs import ImportJob, JobState, JobType, create_job
from snore.api.import_worker import _run_import, _run_rescan
from snore.database import models
from snore.database.session import cleanup_database, init_database, session_scope
from tests.integration.conftest import _make_profile

pytestmark = pytest.mark.integration

# The recorded card holds one session, starting 2024-06-21 01:34.  The parser
# keys it to the night of 06-20 and then checks its start date, so a range must
# span both days to include it.
_SESSION_NIGHT = date(2024, 6, 20)
_SESSION_START = date(2024, 6, 21)
_STALE_PARSER_VERSION = "0.0.0"


@pytest.fixture
def archive(temp_db: Path, tmp_path: Path, resmed_fixture_path: Path) -> Iterator[Any]:
    """Yield ``(profile_id, archive_root)`` with the recorded card archived."""

    async def _seed() -> int:
        await init_database(str(temp_db))
        async with session_scope() as db:
            return (await _make_profile(db)).id

    profile_id = asyncio.run(_seed())
    archive_root = tmp_path / "raw" / str(profile_id)
    shutil.copytree(resmed_fixture_path, archive_root / "22231974465")
    yield profile_id, archive_root
    asyncio.run(cleanup_database())


def _rescan(archive: tuple[int, Path], **options: Any) -> tuple[ImportJob, MagicMock]:
    """Run one rescan job to completion; return it and the analysis enqueue mock."""
    profile_id, archive_root = archive
    job = create_job(JobType.RESCAN, owner_user_id=None, **options)
    job.target_profile_id = profile_id
    enqueued = MagicMock()
    enqueued.job_id = "analysis-1"
    with patch("snore.api.analysis_jobs.enqueue", return_value=enqueued) as enqueue:
        _run_rescan(job, archive_root)
    assert job.state == JobState.SUCCEEDED, job.error_message
    return job, enqueue


def _upload(
    archive: tuple[int, Path], card: Path, spool: Path, *, force: bool = False
) -> ImportJob:
    """Run one upload job of *card* to completion, as the final chunk would."""
    profile_id, archive_root = archive
    shutil.copytree(card, spool)
    job = create_job(JobType.UPLOAD, owner_user_id=None, temp_dir=spool, force=force)
    job.target_profile_id = profile_id
    with patch("snore.api.analysis_jobs.enqueue", return_value=None):
        _run_import(job, archive_root)
    assert job.state == JobState.SUCCEEDED, job.error_message
    return job


def _session_ids() -> list[int]:
    async def _query() -> list[int]:
        async with session_scope() as db:
            return list((await db.scalars(select(models.Session.id))).all())

    return asyncio.run(_query())


def _mark_sessions_stale() -> None:
    """Give every stored session an old parser version, as before a parser fix."""

    async def _update() -> None:
        async with session_scope() as db:
            await db.execute(
                update(models.Session).values(parser_version=_STALE_PARSER_VERSION)
            )

    asyncio.run(_update())


def _stale_session_count() -> int:
    async def _query() -> int:
        async with session_scope() as db:
            rows = await db.scalars(
                select(models.Session.id).where(
                    models.Session.parser_version == _STALE_PARSER_VERSION
                )
            )
            return len(rows.all())

    return asyncio.run(_query())


def test_rescan_without_force_skips_existing_sessions(archive):
    _rescan(archive)
    _mark_sessions_stale()

    job, _ = _rescan(archive)

    assert job.import_result_snapshot["total_imported"] == 0
    assert job.import_result_snapshot["total_skipped"] == 1
    assert _stale_session_count() == 1


def test_rescan_with_force_replaces_existing_sessions(archive):
    _rescan(archive)
    _mark_sessions_stale()

    job, _ = _rescan(archive, force=True)

    assert job.import_result_snapshot["total_imported"] == 1
    assert len(_session_ids()) == 1
    assert _stale_session_count() == 0


def test_forced_rescan_outside_date_range_leaves_sessions_alone(archive):
    _rescan(archive)
    _mark_sessions_stale()

    job, _ = _rescan(
        archive,
        force=True,
        date_from=date(2024, 7, 1),
        date_to=date(2024, 7, 31),
    )

    assert job.import_result_snapshot["total_imported"] == 0
    assert _stale_session_count() == 1


def test_forced_rescan_inside_date_range_replaces_sessions(archive):
    _rescan(archive)
    _mark_sessions_stale()

    job, _ = _rescan(
        archive, force=True, date_from=_SESSION_NIGHT, date_to=_SESSION_START
    )

    assert job.import_result_snapshot["total_imported"] == 1
    assert _stale_session_count() == 0


def test_rescan_with_analyze_off_queues_no_analysis(archive):
    job, enqueue = _rescan(archive, analyze=False)

    enqueue.assert_not_called()
    assert job.analysis_job_id is None
    assert job.analysis_queued is None
    terminal = job.attach_observer().get(timeout=0)
    assert terminal is not None
    assert terminal["event"] == "complete"
    assert terminal["data"]["import_committed"] is True
    assert terminal["data"]["result"]["total_imported"] == 1
    assert "analysis_job_id" not in terminal["data"]


def test_rescan_with_analyze_on_queues_analysis_for_imported_sessions(archive):
    job, enqueue = _rescan(archive)

    enqueue.assert_called_once()
    assert enqueue.call_args.kwargs["session_ids"] == _session_ids()
    assert job.analysis_job_id == "analysis-1"


def test_upload_without_force_skips_existing_sessions(
    archive, resmed_fixture_path, tmp_path
):
    _upload(archive, resmed_fixture_path, tmp_path / "spool-1")
    _mark_sessions_stale()

    job = _upload(archive, resmed_fixture_path, tmp_path / "spool-2")

    assert job.import_result_snapshot["total_skipped"] == 1
    assert _stale_session_count() == 1


def test_upload_with_force_replaces_existing_sessions(
    archive, resmed_fixture_path, tmp_path
):
    _upload(archive, resmed_fixture_path, tmp_path / "spool-1")
    _mark_sessions_stale()

    job = _upload(archive, resmed_fixture_path, tmp_path / "spool-2", force=True)

    assert job.import_result_snapshot["total_imported"] == 1
    assert len(_session_ids()) == 1
    assert _stale_session_count() == 0
