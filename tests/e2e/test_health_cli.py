"""End-to-end tests for the `snore health` command group.

Tests drive the real binary (no in-process imports) and exercise:
- `health import` (dry-run and real), idempotency, invalid path
- `health list` and `health show` after a real import
"""

from __future__ import annotations

import re
import sqlite3

from pathlib import Path

import pytest

HEALTH_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "health_data"
# Date present in the fixture's export.xml that has multiple sleep stages.
FIXTURE_NIGHT = "2024-01-15"


@pytest.fixture
def health_db(tmp_path, e2e_home):
    """Fresh database with Apple Health fixture imported once."""
    from tests.e2e import helpers

    db = tmp_path / "health.db"
    result = helpers.run_snore(
        "health", "import", str(HEALTH_FIXTURE), db=db, home=e2e_home
    )
    assert result.returncode == 0, (
        f"fixture import failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return db


def test_health_import_dry_run_writes_nothing(snore, fresh_db_path):
    """`--dry-run` previews records without persisting any rows."""
    result = snore(
        "health", "import", str(HEALTH_FIXTURE), "--dry-run", db=fresh_db_path
    )
    assert result.returncode == 0, result.stderr or result.stdout
    combined = result.stdout + result.stderr
    assert "dry run" in combined.lower() or "would insert" in combined.lower()

    if fresh_db_path.exists():
        con = sqlite3.connect(str(fresh_db_path))
        count = con.execute("SELECT COUNT(*) FROM health_samples").fetchone()[0]
        con.close()
        assert count == 0, f"Expected 0 rows after dry-run; got {count}"


def test_health_import_real_inserts_rows(snore, fresh_db_path):
    """A real import exits 0 and reports inserted > 0."""
    result = snore("health", "import", str(HEALTH_FIXTURE), db=fresh_db_path)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Inserted" in result.stdout

    con = sqlite3.connect(str(fresh_db_path))
    count = con.execute("SELECT COUNT(*) FROM health_samples").fetchone()[0]
    con.close()
    assert count > 0, "Expected health_samples rows after real import"


def test_health_import_idempotent(snore, tmp_path):
    """A second identical import reports 0 new records."""
    db = tmp_path / "health.db"
    first = snore("health", "import", str(HEALTH_FIXTURE), db=db)
    assert first.returncode == 0, first.stderr or first.stdout

    second = snore("health", "import", str(HEALTH_FIXTURE), db=db)
    assert second.returncode == 0, second.stderr or second.stdout
    assert "0 new records" in second.stdout


def test_health_import_invalid_path_exits_nonzero(snore, fresh_db_path):
    """A path that does not exist produces a nonzero exit and a clear message."""
    result = snore("health", "import", "/nonexistent/path/export.zip", db=fresh_db_path)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "not exist" in combined.lower() or "error" in combined.lower()


def test_health_list_shows_fixture_nights(snore, health_db):
    """`health list` shows the nights from the fixture (newest first)."""
    result = snore("health", "list", db=health_db)
    assert result.returncode == 0, result.stderr or result.stdout
    assert FIXTURE_NIGHT in result.stdout


def test_health_list_respects_limit(snore, health_db):
    """`--limit 1` returns at most 1 data row."""
    result = snore("health", "list", "--limit", "1", db=health_db)
    assert result.returncode == 0, result.stderr or result.stdout
    # Count date-like lines (YYYY-MM-DD) in stdout. The header row says "Date",
    # not a real date, so only actual data rows contribute matches.
    date_rows = re.findall(r"\d{4}-\d{2}-\d{2}", result.stdout)
    assert len(date_rows) <= 1, f"Expected ≤1 date row with --limit 1; got: {date_rows}"


def test_health_show_displays_intervals_and_totals(snore, health_db):
    """`health show <date>` renders sleep intervals and summary totals."""
    result = snore("health", "show", FIXTURE_NIGHT, db=health_db)
    assert result.returncode == 0, result.stderr or result.stdout
    # The totals block labels are always printed.
    assert "Total sleep" in result.stdout
    assert "Time in bed" in result.stdout
    assert "Efficiency" in result.stdout


def test_health_show_unknown_date_exits_nonzero(snore, health_db):
    """`health show` with no data for the date exits nonzero."""
    result = snore("health", "show", "2000-01-01", db=health_db)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "no health data" in combined.lower()


def test_health_show_spo2_samples_on_percent_scale(snore, tmp_path):
    """SpO2 stored as a fraction prints as percent; percent SpO2 and RR print as stored."""
    # The fixture already holds a fraction SpO2 (0.962) and an RR (14.5) sample
    # for FIXTURE_NIGHT; add a percent-scale SpO2 sample from another source.
    percent_record = (
        '<Record type="HKQuantityTypeIdentifierOxygenSaturation" '
        'sourceName="Third-party Oximeter" sourceVersion="1.0" device="Oximeter" '
        'creationDate="2024-01-16 04:00:00 -0500" startDate="2024-01-16 04:00:00 -0500" '
        'endDate="2024-01-16 04:00:00 -0500" value="94.25" unit="%"/>\n'
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    xml = (HEALTH_FIXTURE / "export.xml").read_text()
    (export_dir / "export.xml").write_text(
        xml.replace("</HealthData>", percent_record + "</HealthData>")
    )
    db = tmp_path / "health.db"
    imported = snore("health", "import", str(export_dir), db=db)
    assert imported.returncode == 0, imported.stderr or imported.stdout

    result = snore("health", "show", FIXTURE_NIGHT, db=db)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "96.2 %" in result.stdout
    assert "0.962 %" not in result.stdout
    assert "94.25 %" in result.stdout
    assert "14.5 count/min" in result.stdout
