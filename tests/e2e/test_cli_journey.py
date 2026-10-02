"""Full realistic CLI user journey, driven through the real ``snore`` binary.

This is the test that mirrors how someone actually uses SNORE: import a night
off the SD card, inspect it, analyze it, look at stats, export it, then clean
up. Assertions are tolerant invariants (exit codes, row counts, presence of
key fields) rather than byte-exact output, so harmless formatting changes don't
cause false failures while real behavioral regressions still surface.
"""

from __future__ import annotations

from snore.provenance import PROVENANCE_NOTES, Provenance


def test_import_creates_populated_database(snore, fresh_db_path, resmed_sd):
    """`snore import` on the SD fixture creates a DB with the expected entities."""
    result = snore("import", str(resmed_sd), "--no-backup", "--all", db=fresh_db_path)
    assert result.returncode == 0, result.stderr or result.stdout
    assert fresh_db_path.exists()
    assert "Imported: 1 sessions" in result.stdout

    stats = snore("db", "stats", db=fresh_db_path)
    assert stats.returncode == 0
    assert "Devices: 1" in stats.stdout
    assert "Sessions: 1" in stats.stdout
    # The fixture night carries machine events and waveforms.
    assert "Events: 13" in stats.stdout
    assert "Waveforms: 11" in stats.stdout


def test_reimport_is_idempotent(snore, imported_db, resmed_sd):
    """Re-importing the same source must not duplicate sessions."""
    again = snore("import", str(resmed_sd), "--no-backup", "--all", db=imported_db)
    assert again.returncode == 0
    stats = snore("db", "stats", db=imported_db)
    assert "Sessions: 1" in stats.stdout


def test_session_list_and_show(snore, imported_db):
    """`session list` and `session show --settings` surface the imported night."""
    listing = snore("session", "list", db=imported_db)
    assert listing.returncode == 0
    assert "2024-06-20" in listing.stdout
    assert "22231974465" in listing.stdout  # device serial
    assert "Showing all 1 sessions" in listing.stdout

    show = snore("session", "show", "1", "--settings", db=imported_db)
    assert show.returncode == 0
    assert "Session ID: 1" in show.stdout
    # Deterministic values for this fixed real night — exact regression guards.
    assert "Therapy Mode: APAP" in show.stdout
    # Indices are SNORE recounts over mask-on hours, so they carry the derived
    # marker and the command ends with its legend.
    assert "AHI†: 17.6" in show.stdout
    assert "OAI†: 7.1" in show.stdout
    assert "CAI†: 3.5" in show.stdout
    assert "HI†: 7.1" in show.stdout
    assert show.stdout.strip().splitlines()[-1] == (
        f"† derived: {PROVENANCE_NOTES[Provenance.DERIVED]}"
    )
    # All five real waveform channels parsed from the EDF files.
    for channel in ("epap", "flow", "leak", "pressure", "therapy_pressure"):
        assert channel in show.stdout


def test_full_journey_import_analyze_stats_export_delete(snore, imported_db, tmp_path):
    """End-to-end: analyze → show → stats → export → delete → vacuum on one DB."""
    # Analyze the single session (default AASM mode), storing results.
    analyze = snore("analysis", "run", "--session-id", "1", db=imported_db)
    assert analyze.returncode == 0, analyze.stderr or analyze.stdout

    # `analysis list` should now report the session as analyzed.
    listed = snore("analysis", "list", db=imported_db)
    assert listed.returncode == 0
    assert "✓" in listed.stdout

    # `analysis show` renders the stored summary.
    shown = snore("analysis", "show", "--session-id", "1", db=imported_db)
    assert shown.returncode == 0
    assert "ANALYSIS SUMMARY" in shown.stdout

    # Therapy statistics summary.
    stats = snore("stats", db=imported_db)
    assert stats.returncode == 0
    assert "Events" in stats.stdout

    # CSV export produces the three expected files with data rows.
    csv_dir = tmp_path / "csv_out"
    csv = snore("export", "csv", "--output", str(csv_dir), db=imported_db)
    assert csv.returncode == 0
    sessions_csv = csv_dir / "sessions.csv"
    assert sessions_csv.exists()
    csv_lines = sessions_csv.read_text().strip().splitlines()
    assert len(csv_lines) >= 2  # header + at least one session
    assert "device_session_id" in csv_lines[0]
    assert (csv_dir / "events.csv").exists()
    assert (csv_dir / "settings.csv").exists()

    # Delete the session (confirm via stdin) and confirm it cascades away.
    deleted = snore(
        "session", "delete", "--session-id", "1", db=imported_db, stdin="y\n"
    )
    assert deleted.returncode == 0
    assert "Successfully deleted 1 session(s)" in deleted.stdout

    after = snore("db", "stats", db=imported_db)
    assert "Sessions: 0" in after.stdout

    # Maintenance after deletion should succeed on the now-empty DB.
    vacuum = snore("db", "vacuum", "--yes", db=imported_db)
    assert vacuum.returncode == 0


def test_top_level_help_lists_core_commands(snore):
    """`snore --help` advertises the user-facing command surface."""
    result = snore("--help")
    assert result.returncode == 0
    for command in ("import", "analysis", "session", "stats", "export", "serve"):
        assert command in result.stdout
