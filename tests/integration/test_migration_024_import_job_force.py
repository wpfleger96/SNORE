"""Migration 024 adds ``import_job_records.force``, unforced for existing rows."""

import sqlite3

from pathlib import Path

from alembic import command as alembic_command
from alembic.config import Config

from snore.database.session import (
    _build_alembic_config,
    cleanup_database,
    init_database,
)

_PRE_024 = "023_analysis_results_covering_index"


async def _db_at_023(tmp_path: Path) -> tuple[str, Config]:
    db_path = str(tmp_path / "pre024.db")
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    await init_database(db_path)
    await cleanup_database()
    alembic_command.downgrade(cfg, _PRE_024)
    return db_path, cfg


def _columns(db_path: str) -> set[str]:
    con = sqlite3.connect(db_path)
    try:
        return {row[1] for row in con.execute("PRAGMA table_info(import_job_records)")}
    finally:
        con.close()


async def test_upgrade_marks_existing_jobs_unforced(tmp_path):
    db_path, cfg = await _db_at_023(tmp_path)
    assert "force" not in _columns(db_path)
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO import_job_records (job_id, job_type, state, file_count, "
        "created_at, updated_at) VALUES ('j1', 'upload', 'pending', 1, "
        "'2026-01-01 00:00:00.000000', '2026-01-01 00:00:00.000000')"
    )
    con.commit()
    con.close()

    alembic_command.upgrade(cfg, "head")

    con = sqlite3.connect(db_path)
    try:
        assert con.execute("SELECT force FROM import_job_records").fetchall() == [(0,)]
    finally:
        con.close()


async def test_downgrade_removes_column(tmp_path):
    db_path, cfg = await _db_at_023(tmp_path)
    alembic_command.upgrade(cfg, "head")
    assert "force" in _columns(db_path)

    alembic_command.downgrade(cfg, _PRE_024)

    assert "force" not in _columns(db_path)
