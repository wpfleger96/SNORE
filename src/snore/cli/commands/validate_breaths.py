"""validate-breaths command — signal-level validation against device 0.5 Hz trend signals."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from snore.cli.decorators import (
    CliCtx,
    cli_error_boundary,
    date_range_options_required,
    profile_scoped_command,
)
from snore.cli.display import console, mark_provenance, print_footer, print_header
from snore.provenance import Provenance


@click.command()
@date_range_options_required
@click.option(
    "--export",
    type=click.Path(),
    help="Export report to file (.json or .csv)",
)
@profile_scoped_command
async def validate_breaths(
    ctx: CliCtx,
    date_from: datetime,
    date_to: datetime,
    export: str | None,
) -> None:
    """
    Validate SNORE's per-breath segmentation against device 0.5 Hz trend signals.

    Compares SNORE's per-breath RR, TV, Ti, and I:E ratio against the device's
    independent rr, tv, ti, and ie_ratio trend channels using Spearman correlation,
    median absolute error, and mean bias.  Sessions without a completed analysis or
    without leak-valid breaths are reported as skipped.  Individual channels absent
    from the device recording (e.g. ti/ie_ratio on APAP) are reported as
    channel_not_recorded.
    """
    if date_from > date_to:
        raise click.ClickException("--from date must be before or equal to --to date")

    from snore.validation import (  # noqa: PLC0415
        BreathTrendsValidator,
        export_breath_trends_report_csv,
        export_breath_trends_report_json,
    )

    async with cli_error_boundary("Breath-trends validation error"):
        validator = BreathTrendsValidator(ctx.db, ctx.profile_id)

        console.print(
            f"Running breath-trends validation from {date_from.date()} to {date_to.date()}..."
        )

        report = await validator.validate_date_range(
            date_from.strftime("%Y-%m-%d"),
            date_to.strftime("%Y-%m-%d"),
        )

        agg = report.aggregate

        print_footer()
        print_header("BREATH-TRENDS VALIDATION REPORT")
        console.print(
            f"Date Range: {report.date_range_start} to {report.date_range_end}"
        )
        console.print(f"Total Sessions:    {agg.total_sessions}")
        console.print(f"Sessions Compared: {agg.sessions_compared}")
        console.print(f"Skipped (no analysis):      {agg.sessions_skipped_no_analysis}")
        console.print(
            f"Skipped (no valid breaths): {agg.sessions_skipped_no_valid_breaths}"
        )

        if agg.sessions_compared > 0:
            console.print(
                f"\n{mark_provenance('Aggregate Metrics', Provenance.EXPERIMENTAL)} (over sessions with channel data):"
            )

            def _fmt(v: float | None, precision: int = 3) -> str:
                return f"{v:.{precision}f}" if v is not None else "N/A"

            console.print(
                f"  {'Channel':<10} {'N sessions':<12} {'Mean Spearman r':<17} "
                f"{'Mean MAE':<12} {'Mean Bias'}"
            )
            print_footer()

            ch_aggs = [
                ("rr", agg.rr, "bpm"),
                ("tv", agg.tv, "mL"),
                ("ti", agg.ti, "s"),
                ("ie_ratio", agg.ie_ratio, "pp"),
            ]
            for ch_name, ca, unit in ch_aggs:
                console.print(
                    f"  {ch_name:<10} {ca.sessions_with_data:<12} "
                    f"{_fmt(ca.mean_spearman_r):<17} "
                    f"{_fmt(ca.mean_median_abs_error, 2) + ' ' + unit:<12} "
                    f"{_fmt(ca.mean_bias, 2) + ' ' + unit}"
                )

        compared_sessions = [s for s in report.sessions if s.skipped_reason is None]
        skipped_sessions = [s for s in report.sessions if s.skipped_reason is not None]

        if compared_sessions:
            console.print(
                f"\n{mark_provenance('Per-Session Results', Provenance.EXPERIMENTAL)} (MAE per channel):"
            )
            console.print(
                f"{'Date':<12} {'ID':<6} {'N':<6} "
                f"{'RR MAE':>8} {'TV MAE':>8} {'Ti MAE':>8} {'IE MAE':>8}"
            )
            print_footer()

            _LOW_SAMPLE_THRESHOLD = 20

            def _mae(session_result: object, ch: str) -> str:
                cc = session_result.channels.get(ch)  # type: ignore[attr-defined]
                if cc is None or cc.skipped_reason is not None:
                    return "  N/R"
                if cc.n_pairs == 0 or cc.median_abs_error is None:
                    return "  N/A"
                return f"{cc.median_abs_error:>8.2f}"

            def _low_sample_flag(session_result: object) -> str:
                """Return '!' if any compared channel has n_pairs < threshold."""
                for ch in ("rr", "tv", "ti", "ie_ratio"):
                    cc = session_result.channels.get(ch)  # type: ignore[attr-defined]
                    if (
                        cc is not None
                        and cc.skipped_reason is None
                        and cc.n_pairs < _LOW_SAMPLE_THRESHOLD
                        and cc.n_pairs > 0
                    ):
                        return "!"
                return " "

            low_sample_shown = False
            for s in compared_sessions[:20]:
                flag = _low_sample_flag(s)
                if flag == "!":
                    low_sample_shown = True
                console.print(
                    f"{s.date:<12} "
                    f"{s.session_id:<6} "
                    f"{s.n_breaths:<6} "
                    f"{_mae(s, 'rr'):>8} "
                    f"{_mae(s, 'tv'):>8} "
                    f"{_mae(s, 'ti'):>8} "
                    f"{_mae(s, 'ie_ratio'):>8}"
                    f"{flag}"
                )

            if len(compared_sessions) > 20:
                console.print(f"... and {len(compared_sessions) - 20} more sessions")

            if low_sample_shown:
                console.print(
                    f"! at least one channel has < {_LOW_SAMPLE_THRESHOLD} pairs"
                )

        if skipped_sessions:
            console.print(f"\nSkipped sessions: {len(skipped_sessions)}")
            for s in skipped_sessions[:5]:
                console.print(f"  {s.date} (id={s.session_id}): {s.skipped_reason}")
            if len(skipped_sessions) > 5:
                console.print(f"  ... and {len(skipped_sessions) - 5} more")

        if export:
            export_path = Path(export)
            if export_path.suffix == ".json":
                export_breath_trends_report_json(report, export_path)
                console.print(f"\nReport exported to {export_path}")
            elif export_path.suffix == ".csv":
                export_breath_trends_report_csv(report, export_path)
                console.print(f"\nReport exported to {export_path}")
            else:
                raise click.ClickException(
                    f"Unknown export format '{export_path.suffix}'. Use .json or .csv"
                )
