"""Integration tests for orphaned import-job recovery on server restart.

On startup, ``_recover_orphaned_import_jobs`` marks any ImportJobRecord rows
with a non-terminal state (pending_upload, pending, running) as failed.
Terminal rows (succeeded, failed, cancelled) are left untouched.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sqlalchemy import select

from snore.api.app import _recover_orphaned_import_jobs
from snore.database.models import ImportJobRecord
from snore.database.session import cleanup_database, init_database, session_scope

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _make_record(
    job_id: str,
    state: str,
    finished: bool = False,
) -> ImportJobRecord:
    fin = _now() if finished else None
    return ImportJobRecord(
        job_id=job_id,
        job_type="upload",
        state=state,
        file_count=0,
        created_at=_now(),
        finished_at=fin,
        updated_at=_now(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_recovery_marks_non_terminal_rows_failed(tmp_path):
    """Non-terminal rows are set to failed=True with the restart error message."""
    db_path = str(tmp_path / "recovery.db")
    await init_database(db_path)

    try:
        # Seed one row in each non-terminal state.
        non_terminal = ["pending_upload", "pending", "running"]
        async with session_scope(immediate=True) as db:
            for state in non_terminal:
                db.add(_make_record(f"job_{state}", state))

        await _recover_orphaned_import_jobs()

        async with session_scope() as db:
            rows = (
                (
                    await db.execute(
                        select(ImportJobRecord).where(
                            ImportJobRecord.job_id.in_(
                                [f"job_{s}" for s in non_terminal]
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )

        records = {r.job_id: r for r in rows}
        for state in non_terminal:
            rec = records[f"job_{state}"]
            assert rec.state == "failed", (
                f"Expected 'failed' for job_{state}, got {rec.state!r}"
            )
            assert rec.error_message == "Server restarted while job was in progress"
            assert rec.finished_at is not None
            assert rec.updated_at is not None

    finally:
        await cleanup_database()


async def test_recovery_leaves_terminal_rows_unchanged(tmp_path):
    """Terminal rows (succeeded, failed, cancelled) are not modified by recovery."""
    db_path = str(tmp_path / "terminal.db")
    await init_database(db_path)

    try:
        terminal_states = ["succeeded", "failed", "cancelled"]

        async with session_scope(immediate=True) as db:
            for state in terminal_states:
                rec = _make_record(f"job_{state}", state, finished=True)
                if state == "failed":
                    rec.error_message = "original error"
                db.add(rec)

        await _recover_orphaned_import_jobs()

        async with session_scope() as db:
            rows = (
                (
                    await db.execute(
                        select(ImportJobRecord).where(
                            ImportJobRecord.job_id.in_(
                                [f"job_{s}" for s in terminal_states]
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )

        records = {r.job_id: r for r in rows}
        for state in terminal_states:
            rec = records[f"job_{state}"]
            assert rec.state == state, (
                f"Terminal state should be unchanged: expected {state!r}, got {rec.state!r}"
            )
        # The existing 'failed' row must keep its original error message.
        assert records["job_failed"].error_message == "original error"

    finally:
        await cleanup_database()


async def test_recovery_with_no_orphans_is_silent(tmp_path):
    """When there are no orphaned rows, recovery completes without logging or errors."""
    db_path = str(tmp_path / "no_orphans.db")
    await init_database(db_path)

    try:
        async with session_scope(immediate=True) as db:
            db.add(_make_record("job_done", "succeeded", finished=True))

        # Must not raise.
        await _recover_orphaned_import_jobs()

        async with session_scope() as db:
            rec = (
                (
                    await db.execute(
                        select(ImportJobRecord).where(
                            ImportJobRecord.job_id == "job_done"
                        )
                    )
                )
                .scalars()
                .first()
            )

        assert rec is not None
        assert rec.state == "succeeded"

    finally:
        await cleanup_database()


async def test_recovery_with_empty_table_is_noop(tmp_path):
    """Recovery on an empty table completes successfully."""
    db_path = str(tmp_path / "empty.db")
    await init_database(db_path)

    try:
        await _recover_orphaned_import_jobs()

        async with session_scope() as db:
            count = len((await db.execute(select(ImportJobRecord))).scalars().all())
        assert count == 0

    finally:
        await cleanup_database()
