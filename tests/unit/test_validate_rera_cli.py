"""CLI display/export tests for ``snore validate-rera``.

The whole DB layer is patched out (session context, profile resolution, and
``ReraValidator.validate_date_range``) so these tests exercise only the
terminal rendering — top-N truncation of the scored table and ``--export``
dispatch — against a synthetic report.
"""

from __future__ import annotations

import json

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner, Result

from snore.cli.commands.validate_rera import validate_rera
from snore.validation.rera_report import (
    ReraSessionValidation,
    ReraValidationReport,
)
from snore.validation.rera_validator import ReraValidator


def _scored_session(session_id: int, machine_re: int) -> ReraSessionValidation:
    return ReraSessionValidation(
        session_id=session_id,
        date="2025-06-01",
        duration_hours=8.0,
        skipped_reason=None,
        machine_re_count=machine_re,
        amplitude_rera_count=3,
        proxy_rera_count=2,
        amplitude_matched=1,
        proxy_matched=1,
        amplitude_sensitivity=0.5,
        amplitude_precision=0.33,
        proxy_sensitivity=0.5,
        proxy_precision=0.5,
    )


def _report(num_scored: int) -> ReraValidationReport:
    # Descending machine RE so the ordering the CLI applies is observable.
    sessions = [
        _scored_session(i, machine_re=num_scored - i) for i in range(num_scored)
    ]
    aggregate = ReraValidator._calculate_aggregate(sessions)
    return ReraValidationReport(
        report_date="2025-06-02 00:00:00",
        date_range_start="2025-06-01",
        date_range_end="2025-06-30",
        aggregate=aggregate,
        sessions=sessions,
    )


@asynccontextmanager
async def _fake_session_scope(*args, **kwargs):
    yield MagicMock()


def _invoke(
    report: ReraValidationReport, extra_args: list[str] | None = None
) -> Result:
    runner = CliRunner()
    with (
        patch("snore.database.session.init_database", new_callable=AsyncMock),
        patch("snore.database.session.session_scope", _fake_session_scope),
        patch(
            "snore.auth.factory.resolve_cli_profile_id",
            AsyncMock(return_value=1),
        ),
        patch.object(
            ReraValidator,
            "validate_date_range",
            AsyncMock(return_value=report),
        ),
    ):
        return runner.invoke(
            validate_rera,
            ["--from", "2025-06-01", "--to", "2025-06-30", *(extra_args or [])],
        )


def test_scored_table_truncates_beyond_top_20():
    result = _invoke(_report(25))
    assert result.exit_code == 0, result.output
    # Header announces the truncation, footer counts the hidden rows.
    assert "top 20 of 25" in result.output
    assert "5 more scored sessions not shown" in result.output


def test_scored_table_shows_all_when_within_cap():
    result = _invoke(_report(3))
    assert result.exit_code == 0, result.output
    assert "top 20 of" not in result.output
    assert "more scored sessions not shown" not in result.output
    assert "Scored sessions* (3;" in result.output


def test_report_marks_snore_scores_and_prints_legend_once():
    result = _invoke(_report(3))
    assert result.exit_code == 0, result.output
    assert "Amplitude*:" in result.output
    # Device-scored RE count is unmarked; only its per-hour density is derived.
    machine_line = next(
        line for line in result.output.splitlines() if "Machine RE" in line
    )
    assert "Machine RE:" in machine_line
    assert machine_line.rstrip().endswith("/h†)")
    legend = result.output.splitlines()[-1]
    assert legend == (
        "* experimental (SNORE heuristic)  "
        "† derived (computed by SNORE from device data)"
    )
    assert result.output.count("* experimental") == 1


def test_export_json_dispatch_writes_file(tmp_path):
    out = tmp_path / "report.json"
    result = _invoke(_report(3), ["--export", str(out)])
    assert result.exit_code == 0, result.output
    assert "Report exported to" in result.output
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["aggregate"]["sessions_with_machine_re"] == 3


def test_export_unknown_suffix_errors(tmp_path):
    out = tmp_path / "report.txt"
    result = _invoke(_report(3), ["--export", str(out)])
    assert result.exit_code != 0
    assert "Unknown export format" in result.output
