"""Migration 020 indexes analysis_results for latest-analysis lookups."""

import sqlite3

from pathlib import Path

from alembic import command as alembic_command
from alembic.config import Config

from snore.database.session import (
    _build_alembic_config,
    cleanup_database,
    init_database,
)

_INDEX = "ix_analysis_results_session_id_created_at"


def _index_columns(db_path: str) -> list[str] | None:
    con = sqlite3.connect(db_path)
    try:
        indexes = {row[1] for row in con.execute("PRAGMA index_list(analysis_results)")}
        if _INDEX not in indexes:
            return None
        return [row[2] for row in con.execute(f"PRAGMA index_info({_INDEX})")]
    finally:
        con.close()


async def _db_at_019(tmp_path: Path) -> tuple[str, Config]:
    db_path = str(tmp_path / "pre020.db")
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    await init_database(db_path)
    await cleanup_database()
    alembic_command.downgrade(cfg, "019_oscar_event_parity")
    return db_path, cfg


async def test_upgrade_from_019_creates_index(tmp_path):
    db_path, cfg = await _db_at_019(tmp_path)
    assert _index_columns(db_path) is None

    alembic_command.upgrade(cfg, "head")

    assert _index_columns(db_path) == ["session_id", "created_at", "id"]


async def test_downgrade_removes_index(tmp_path):
    db_path, cfg = await _db_at_019(tmp_path)
    alembic_command.upgrade(cfg, "head")
    assert _index_columns(db_path) is not None

    alembic_command.downgrade(cfg, "019_oscar_event_parity")

    assert _index_columns(db_path) is None


async def test_upgrade_is_idempotent_when_index_exists(tmp_path):
    db_path, cfg = await _db_at_019(tmp_path)
    con = sqlite3.connect(db_path)
    con.execute(
        f"CREATE INDEX {_INDEX} ON analysis_results (session_id, created_at, id)"
    )
    con.commit()
    con.close()

    alembic_command.upgrade(cfg, "head")

    assert _index_columns(db_path) == ["session_id", "created_at", "id"]
