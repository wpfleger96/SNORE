"""Migration 020 rebases epoch-offset analysis windows onto the session start.

Every window carries sub-second parts, so each expected value also pins that
microseconds survive the rewrite.
"""

import sqlite3

import pytest

from alembic import command as alembic_command

from snore.database.session import (
    _build_alembic_config,
    cleanup_database,
    init_database,
)

_PRE_020 = "019_oscar_event_parity"

# id -> (start_time, end_time, import_source)
_SESSIONS = {
    1: ("2025-01-10 22:00:00.000000", "2025-01-11 06:00:00.000000", "resmed_edf"),
    2: ("2025-03-01 23:15:00.000000", "2025-03-02 07:00:00.000000", "resmed_edf"),
    # Demo copies of session 1 from `snore db scrub-demo`, shifted +400 and -30 days.
    3: ("2026-02-14 22:00:00.000000", "2026-02-15 06:00:00.000000", "demo"),
    4: ("2024-12-11 22:00:00.000000", "2024-12-12 06:00:00.000000", "demo"),
}

_EPOCH_ROW = 1
_WALLCLOCK_ROW = 2
_SECOND_SESSION_ROW = 3
_DEMO_LATER_ROW = 4
_DEMO_EARLIER_ROW = 5

# id -> (session_id, window as seeded at 019)
_SEEDED = {
    _EPOCH_ROW: (1, ("1970-01-01 00:00:00.040000", "1970-01-01 07:30:12.500000")),
    _WALLCLOCK_ROW: (
        1,
        ("2025-01-10 22:00:00.250000", "2025-01-11 02:00:00.250000"),
    ),
    _SECOND_SESSION_ROW: (
        2,
        ("1970-01-01 00:00:01.000000", "1970-01-01 06:00:00.750000"),
    ),
    _DEMO_LATER_ROW: (
        3,
        ("1971-02-05 00:00:00.040000", "1971-02-05 07:30:12.500000"),
    ),
    _DEMO_EARLIER_ROW: (
        4,
        ("1969-12-02 00:00:00.040000", "1969-12-02 07:30:12.500000"),
    ),
}


def _seed(con: sqlite3.Connection) -> None:
    con.executemany(
        "INSERT INTO sessions (id, device_id, device_session_id, start_time, "
        "end_time, duration_seconds, import_date, import_source, "
        "data_quality_notes, has_waveform_data, has_event_data, has_statistics, "
        "enabled) VALUES (?, 1, ?, ?, ?, 28800, '2025-01-11 07:00:00.000000', ?, "
        "'{}', 0, 1, 1, 1)",
        [
            (session_id, str(session_id), start, end, source)
            for session_id, (start, end, source) in _SESSIONS.items()
        ],
    )
    con.executemany(
        "INSERT INTO analysis_results (id, session_id, timestamp_start, "
        "timestamp_end, programmatic_result_json, engine_versions_json, created_at) "
        "VALUES (?, ?, ?, ?, '{}', '{}', '2025-01-11 08:00:00.000000')",
        [
            (row_id, session_id, *window)
            for row_id, (session_id, window) in _SEEDED.items()
        ],
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


async def test_upgrade_epoch_offsets_become_own_session_wallclock(pre_020):
    db_path, cfg = pre_020

    alembic_command.upgrade(cfg, "head")

    windows = _windows(db_path)
    assert windows[_EPOCH_ROW] == (
        "2025-01-10 22:00:00.040000",
        "2025-01-11 05:30:12.500000",
    )
    assert windows[_SECOND_SESSION_ROW] == (
        "2025-03-01 23:15:01.000000",
        "2025-03-02 05:15:00.750000",
    )


async def test_upgrade_demo_shifted_offsets_become_demo_session_wallclock(pre_020):
    """Rows shifted by whole days into 1971 or back into 1969 land on their session."""
    db_path, cfg = pre_020

    alembic_command.upgrade(cfg, "head")

    windows = _windows(db_path)
    assert windows[_DEMO_LATER_ROW] == (
        "2026-02-14 22:00:00.040000",
        "2026-02-15 05:30:12.500000",
    )
    assert windows[_DEMO_EARLIER_ROW] == (
        "2024-12-11 22:00:00.040000",
        "2024-12-12 05:30:12.500000",
    )


async def test_upgrade_wallclock_rows_unchanged(pre_020):
    db_path, cfg = pre_020

    alembic_command.upgrade(cfg, "head")

    assert _windows(db_path)[_WALLCLOCK_ROW] == _SEEDED[_WALLCLOCK_ROW][1]


async def test_downgrade_wallclock_rows_return_to_unshifted_epoch_offsets(pre_020):
    """Downgrade is lossy: it reverts every row at or after its session's start.

    Plain rows get their seeded values back. It can't tell migrated rows from
    native ones, so the native wall-clock row becomes an epoch offset too, and
    demo rows come back without their day shift.
    """
    db_path, cfg = pre_020
    alembic_command.upgrade(cfg, "head")

    alembic_command.downgrade(cfg, _PRE_020)

    assert _windows(db_path) == {
        _EPOCH_ROW: _SEEDED[_EPOCH_ROW][1],
        _WALLCLOCK_ROW: ("1970-01-01 00:00:00.250000", "1970-01-01 04:00:00.250000"),
        _SECOND_SESSION_ROW: _SEEDED[_SECOND_SESSION_ROW][1],
        _DEMO_LATER_ROW: _SEEDED[_EPOCH_ROW][1],
        _DEMO_EARLIER_ROW: _SEEDED[_EPOCH_ROW][1],
    }


async def test_reupgrade_after_downgrade_reaches_the_same_state(pre_020):
    db_path, cfg = pre_020
    alembic_command.upgrade(cfg, "head")
    windows = _windows(db_path)

    alembic_command.downgrade(cfg, _PRE_020)
    alembic_command.upgrade(cfg, "head")

    assert _windows(db_path) == windows
