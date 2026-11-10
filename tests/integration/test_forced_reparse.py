"""Forced re-parse and the other import options against a real database.

Runs the rescan worker on the recorded ResMed SD card, copied into a profile's
raw archive, to check what ``force``, the date range and ``analyze`` do to
sessions already imported.
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
from snore.api.import_worker import _run_rescan
from snore.database import models
from snore.database.session import cleanup_database, init_database, session_scope
from tests.integration.conftest import _make_profile

pytestmark = pytest.mark.integration

# The recorded card holds one session, starting 2024-06-21 01:34.  Import date
# ranges select by night (noon split), so it belongs to the night of 06-20: a
# range of 06-20 alone includes it, and a range of 06-21 alone does not.
_SESSION_NIGHT = date(2024, 6, 20)
_DAY_AFTER_NIGHT = date(2024, 6, 21)
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


def test_forced_rescan_of_calendar_start_day_skips_session(archive):
    _rescan(archive)
    _mark_sessions_stale()

    job, _ = _rescan(
        archive, force=True, date_from=_DAY_AFTER_NIGHT, date_to=_DAY_AFTER_NIGHT
    )

    assert job.import_result_snapshot["total_imported"] == 0
    assert _stale_session_count() == 1


def test_forced_rescan_inside_date_range_replaces_sessions(archive):
    _rescan(archive)
    _mark_sessions_stale()

    job, _ = _rescan(
        archive, force=True, date_from=_SESSION_NIGHT, date_to=_SESSION_NIGHT
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


def test_forced_rescan_keeps_disabled_session_disabled(archive):
    _rescan(archive)

    async def _disable_all() -> None:
        async with session_scope() as db:
            await db.execute(update(models.Session).values(enabled=False))

    asyncio.run(_disable_all())
    _mark_sessions_stale()

    job, _ = _rescan(archive, force=True)

    async def _enabled_flags() -> list[bool]:
        async with session_scope() as db:
            return list((await db.scalars(select(models.Session.enabled))).all())

    assert job.import_result_snapshot["total_imported"] == 1
    assert _stale_session_count() == 0
    assert asyncio.run(_enabled_flags()) == [False]
