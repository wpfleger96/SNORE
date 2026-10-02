"""CLI display tests for the low-sample ``!`` flag in validate-fl / validate-breaths.

The DB layer and each validator's ``validate_date_range`` are patched out so
only the terminal rendering of a synthetic report is exercised.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import click

from click.testing import CliRunner, Result

from snore.cli.commands.validate_breaths import validate_breaths
from snore.cli.commands.validate_fl import validate_fl
from snore.validation.breath_trends_report import (
    BreathTrendsSessionValidation,
    BreathTrendsValidationReport,
    ChannelComparison,
)
from snore.validation.breath_trends_validator import BreathTrendsValidator
from snore.validation.fl_report import FlSessionValidation, FlValidationReport
from snore.validation.fl_validator import LOW_SAMPLE_BREATHS, FlowLimitationValidator

_BREATHS_FOOTNOTE = "! at least one channel has < 20 pairs"
_FL_FOOTNOTE = f"! fewer than {LOW_SAMPLE_BREATHS} breaths compared"


@asynccontextmanager
async def _fake_session_scope(*args: Any, **kwargs: Any) -> AsyncIterator[MagicMock]:
    yield MagicMock()


def _invoke(command: click.Command, validator: type, report: Any) -> Result:
    with (
        patch("snore.database.session.init_database", new_callable=AsyncMock),
        patch("snore.database.session.session_scope", _fake_session_scope),
        patch(
            "snore.auth.factory.resolve_cli_profile_id",
            AsyncMock(return_value=1),
        ),
        patch.object(validator, "validate_date_range", AsyncMock(return_value=report)),
    ):
        return CliRunner().invoke(
            command, ["--from", "2025-06-01", "--to", "2025-06-30"]
        )


def _session_rows(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("2025-06-01")]


def _fl_report(n_breaths: int) -> FlValidationReport:
    session = FlSessionValidation(
        session_id=1,
        date="2025-06-01",
        duration_hours=8.0,
        parser_version="v1",
        has_flg_waveform=True,
        n_breaths_compared=n_breaths,
        low_sample_warning=n_breaths < LOW_SAMPLE_BREATHS,
        spearman_flattening_r=0.4,
    )
    return FlValidationReport(
        report_date="2025-06-02 00:00:00",
        date_range_start="2025-06-01",
        date_range_end="2025-06-30",
        aggregate=FlowLimitationValidator._calculate_aggregate([session]),
        sessions=[session],
    )


def _breaths_report(n_pairs: int) -> BreathTrendsValidationReport:
    session = BreathTrendsSessionValidation(
        session_id=1,
        date="2025-06-01",
        duration_hours=8.0,
        parser_version="v1",
        n_breaths=500,
        channels={
            "rr": ChannelComparison(n_pairs=n_pairs, median_abs_error=0.5),
        },
    )
    return BreathTrendsValidationReport(
        report_date="2025-06-02 00:00:00",
        date_range_start="2025-06-01",
        date_range_end="2025-06-30",
        aggregate=BreathTrendsValidator._calculate_aggregate([session]),
        sessions=[session],
    )


def test_validate_fl_low_sample_row_flagged_with_footnote():
    result = _invoke(validate_fl, FlowLimitationValidator, _fl_report(5))
    assert result.exit_code == 0, result.output
    assert "5    !" in _session_rows(result.output)[0]
    assert _FL_FOOTNOTE in result.output


def test_validate_fl_no_footnote_without_low_sample_rows():
    result = _invoke(validate_fl, FlowLimitationValidator, _fl_report(200))
    assert result.exit_code == 0, result.output
    assert "!" not in _session_rows(result.output)[0]
    assert _FL_FOOTNOTE not in result.output


def test_validate_breaths_low_sample_row_flagged_with_footnote():
    result = _invoke(validate_breaths, BreathTrendsValidator, _breaths_report(5))
    assert result.exit_code == 0, result.output
    assert _session_rows(result.output)[0].endswith("!")
    assert _BREATHS_FOOTNOTE in result.output


def test_validate_breaths_no_footnote_without_low_sample_rows():
    result = _invoke(validate_breaths, BreathTrendsValidator, _breaths_report(200))
    assert result.exit_code == 0, result.output
    assert not _session_rows(result.output)[0].endswith("!")
    assert _BREATHS_FOOTNOTE not in result.output
