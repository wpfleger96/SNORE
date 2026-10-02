"""Migration 018 adds the device-index headline columns and backfills days as derived."""

import sqlite3

from pathlib import Path

import pytest

from alembic import command as alembic_command

from snore.database.session import (
    _build_alembic_config,
    cleanup_database,
    init_database,
)

_NEW_COLUMNS = {
    "ahi_computed",
    "oai_computed",
    "cai_computed",
    "hi_computed",
    "index_source",
}


def _statistics_columns(con: sqlite3.Connection) -> set[str]:
    return {row[1] for row in con.execute("PRAGMA table_info(statistics)")}


def _insert_day(
    con: sqlite3.Connection,
    day_id: int,
    indices: tuple,
    extra: dict[str, str] | None = None,
) -> None:
    extra = extra or {}
    columns = "".join(f", {name}" for name in extra)
    con.execute(
        "INSERT INTO days (id, device_id, date, session_count, "
        "total_therapy_hours, obstructive_apneas, central_apneas, hypopneas, "
        f"reras, ahi, oai, cai, hi, created_at, updated_at{columns}) "
        "VALUES (?, 1, ?, 1, 6.0, 0, 0, 0, 0, ?, ?, ?, ?, "
        f"'2025-01-01 00:00:00', '2025-01-01 00:00:00'{', ?' * len(extra)})",
        (day_id, f"2025-01-{day_id:02d}", *indices, *extra.values()),
    )


async def test_upgrade_from_017_backfills_recount_as_derived(tmp_path):
    db_path = str(tmp_path / "pre018.db")
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    await init_database(db_path)
    await cleanup_database()
    alembic_command.downgrade(cfg, "017_sessions_day_id_index")

    con = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(days)")}
        assert not columns & _NEW_COLUMNS
        assert "usage_hours_device" not in _statistics_columns(con)
        _insert_day(con, 1, (5.0, 2.5, 1.25, 1.25))
        _insert_day(con, 2, (None, 2.0, None, None))
        _insert_day(con, 3, (None, None, None, None))
        con.commit()
    finally:
        con.close()

    alembic_command.upgrade(cfg, "head")

    con = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(days)")}
        assert _NEW_COLUMNS <= columns
        assert "usage_hours_device" in _statistics_columns(con)
        rows = con.execute(
            "SELECT id, ahi_computed, oai_computed, cai_computed, hi_computed, "
            "index_source FROM days ORDER BY id"
        ).fetchall()
    finally:
        con.close()

    assert rows == [
        (1, 5.0, 2.5, 1.25, 1.25, "derived"),
        (2, None, 2.0, None, None, "derived"),
        (3, None, None, None, None, None),
    ]


def _insert_device_day(con: sqlite3.Connection) -> None:
    """A day headlining device indices (4.0/...) over a 6.0/3.0/1.5/1.5 recount."""
    _insert_day(con, 1, (4.0, 2.0, 1.0, 1.0))
    con.execute(
        "UPDATE days SET ahi_computed = 6.0, oai_computed = 3.0, "
        "cai_computed = 1.5, hi_computed = 1.5, index_source = 'device' "
        "WHERE id = 1"
    )


def _day_indices(db_path: str) -> tuple:
    con = sqlite3.connect(db_path)
    try:
        return con.execute(
            "SELECT ahi, oai, cai, hi, ahi_computed, oai_computed, cai_computed, "
            "hi_computed, index_source FROM days WHERE id = 1"
        ).fetchone()
    finally:
        con.close()


_RECOUNT_AS_DERIVED = (6.0, 3.0, 1.5, 1.5, 6.0, 3.0, 1.5, 1.5, "derived")


async def test_downgrade_restores_recount_and_reupgrade_labels_it_derived(tmp_path):
    db_path = str(tmp_path / "device_day.db")
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    await init_database(db_path)
    await cleanup_database()
    con = sqlite3.connect(db_path)
    try:
        _insert_device_day(con)
        con.commit()
    finally:
        con.close()

    alembic_command.downgrade(cfg, "017_sessions_day_id_index")

    con = sqlite3.connect(db_path)
    try:
        assert con.execute(
            "SELECT ahi, oai, cai, hi FROM days WHERE id = 1"
        ).fetchone() == (6.0, 3.0, 1.5, 1.5)
    finally:
        con.close()

    alembic_command.upgrade(cfg, "head")

    assert _day_indices(db_path) == _RECOUNT_AS_DERIVED


async def test_checksum_replay_of_018_keeps_recount_on_device_days(tmp_path):
    """Startup replay of an edited 018 runs the on-disk downgrade, then upgrade."""
    db_path = str(tmp_path / "replay.db")
    await init_database(db_path)
    await cleanup_database()
    con = sqlite3.connect(db_path)
    try:
        _insert_device_day(con)
        con.execute(
            "UPDATE snore_migration_checksums SET checksum = 'cafebabe' "
            "WHERE revision = '018_day_device_indices'"
        )
        con.commit()
    finally:
        con.close()

    await init_database(db_path)
    await cleanup_database()

    assert _day_indices(db_path) == _RECOUNT_AS_DERIVED


async def _fresh_db(tmp_path: Path) -> str:
    """A new DB built by ``create_all`` and stamped at head."""
    db_path = str(tmp_path / "fresh.db")
    await init_database(db_path)
    await cleanup_database()
    return db_path


async def _migrated_db(tmp_path: Path) -> str:
    """A DB taken back to 017 and upgraded through 018."""
    db_path = await _fresh_db(tmp_path)
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    alembic_command.downgrade(cfg, "017_sessions_day_id_index")
    alembic_command.upgrade(cfg, "head")
    return db_path


@pytest.mark.parametrize("make_db", [_fresh_db, _migrated_db])
async def test_index_source_rejects_unknown_values(tmp_path, make_db):
    db_path = await make_db(tmp_path)
    con = sqlite3.connect(db_path)
    try:
        _insert_day(con, 1, (5.0, 2.5, 1.25, 1.25))
        for value in ("device", "derived", None):
            con.execute("UPDATE days SET index_source = ? WHERE id = 1", (value,))
        with pytest.raises(sqlite3.IntegrityError, match="invalid days.index_source"):
            con.execute("UPDATE days SET index_source = 'Device' WHERE id = 1")
        with pytest.raises(sqlite3.IntegrityError, match="invalid days.index_source"):
            _insert_day(con, 2, (None,) * 4, {"index_source": "bogus"})
        _insert_day(con, 3, (None,) * 4, {"index_source": "derived"})
    finally:
        con.close()
