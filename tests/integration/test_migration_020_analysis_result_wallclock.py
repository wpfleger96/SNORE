"""Migration 020 rebases epoch-offset analysis windows onto the session start."""

import sqlite3

import pytest

from alembic import command as alembic_command

from snore.database.session import (
    _build_alembic_config,
    cleanup_database,
    init_database,
)

_PRE_020 = "019_oscar_event_parity"
_SESSION_START = "2025-01-10 22:00:00.000000"

_EPOCH_ROW = 1
_EPOCH_WINDOW = ("1970-01-01 00:00:00.000000", "1970-01-01 07:30:12.500000")
_EPOCH_ROW_WALLCLOCK = ("2025-01-10 22:00:00.000000", "2025-01-11 05:30:12.500000")

_WALLCLOCK_ROW = 2
_WALLCLOCK_WINDOW = ("2025-01-10 22:00:00.000000", "2025-01-11 02:00:00.000000")


def _seed(con: sqlite3.Connection) -> None:
    con.execute(
        "INSERT INTO sessions (id, device_id, device_session_id, start_time, "
        "end_time, duration_seconds, import_date, import_source, "
        "data_quality_notes, has_waveform_data, has_event_data, has_statistics, "
        "enabled) VALUES (1, 1, '1', ?, '2025-01-11 06:00:00.000000', 28800, "
        "'2025-01-11 07:00:00.000000', 'resmed_edf', '{}', 0, 1, 1, 1)",
        (_SESSION_START,),
    )
    con.executemany(
        "INSERT INTO analysis_results (id, session_id, timestamp_start, "
        "timestamp_end, programmatic_result_json, engine_versions_json, created_at) "
        "VALUES (?, 1, ?, ?, '{}', '{}', '2025-01-11 08:00:00.000000')",
        [(_EPOCH_ROW, *_EPOCH_WINDOW), (_WALLCLOCK_ROW, *_WALLCLOCK_WINDOW)],
    )


def _windows(db_path: str) -> dict[int, tuple[str, str]]:
    con = sqlite3.connect(db_path)
    try:
        return {
            row[0]: (row[1], row[2])
            for row in con.execute(
                "SELECT id, timestamp_start, timestamp_end FROM analysis_results"
            )
        }
    finally:
        con.close()


@pytest.fixture
async def pre_020(tmp_path):
    """A seeded DB at 019, plus its Alembic config."""
    db_path = str(tmp_path / "pre020.db")
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    await init_database(db_path)
    await cleanup_database()
    alembic_command.downgrade(cfg, _PRE_020)
    con = sqlite3.connect(db_path)
    try:
        _seed(con)
        con.commit()
    finally:
        con.close()
    return db_path, cfg


async def test_upgrade_epoch_offsets_become_session_wallclock(pre_020):
    db_path, cfg = pre_020

    alembic_command.upgrade(cfg, "head")

    assert _windows(db_path)[_EPOCH_ROW] == _EPOCH_ROW_WALLCLOCK


async def test_upgrade_wallclock_rows_unchanged(pre_020):
    db_path, cfg = pre_020

    alembic_command.upgrade(cfg, "head")

    assert _windows(db_path)[_WALLCLOCK_ROW] == _WALLCLOCK_WINDOW


async def test_downgrade_wallclock_rows_return_to_epoch_offsets(pre_020):
    """Downgrade can't tell migrated rows from native ones, so it reverses both."""
    db_path, cfg = pre_020
    alembic_command.upgrade(cfg, "head")

    alembic_command.downgrade(cfg, _PRE_020)

    assert _windows(db_path) == {
        _EPOCH_ROW: _EPOCH_WINDOW,
        _WALLCLOCK_ROW: ("1970-01-01 00:00:00.000000", "1970-01-01 04:00:00.000000"),
    }


async def test_reupgrade_after_downgrade_reaches_the_same_state(pre_020):
    db_path, cfg = pre_020
    alembic_command.upgrade(cfg, "head")
    windows = _windows(db_path)

    alembic_command.downgrade(cfg, _PRE_020)
    alembic_command.upgrade(cfg, "head")

    assert _windows(db_path) == windows
