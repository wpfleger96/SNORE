"""validate-fl command — signal-level FL validation against device FLG waveform."""

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
async def validate_fl(
    ctx: CliCtx,
    date_from: datetime,
    date_to: datetime,
    export: str | None,
) -> None:
    """
    Validate SNORE's per-breath FL metrics against the device's FLG waveform signal.

    Compares flattening_severity (1 - mid_insp_flattening) and flatness_index
    against the device's 0.5 Hz FlowLim.2s signal using Spearman correlation
    and AUC metrics.  Sessions without a device FLG waveform or without a
    completed analysis are reported as skipped.
    """
    if date_from > date_to:
        raise click.ClickException("--from date must be before or equal to --to date")

    from snore.validation import (  # noqa: PLC0415
        FlowLimitationValidator,
        export_fl_report_csv,
        export_fl_report_json,
    )

    async with cli_error_boundary("FL validation error"):
        validator = FlowLimitationValidator(ctx.db, ctx.profile_id)

        console.print(
            f"Running FL validation from {date_from.date()} to {date_to.date()}..."
        )

        report = await validator.validate_date_range(
            date_from.strftime("%Y-%m-%d"),
            date_to.strftime("%Y-%m-%d"),
        )

        agg = report.aggregate

        print_footer()
        print_header("FL SIGNAL VALIDATION REPORT")
        console.print(
            f"Date Range: {report.date_range_start} to {report.date_range_end}"
        )
        console.print(f"Total Sessions:    {agg.total_sessions}")
        console.print(f"Sessions Compared: {agg.sessions_compared}")
        console.print(f"Skipped (no FLG waveform):   {agg.sessions_skipped_no_flg}")
        console.print(
            f"Skipped (no analysis):        {agg.sessions_skipped_no_analysis}"
        )
        console.print(
            f"Skipped (no valid breaths):   {agg.sessions_skipped_no_valid_breaths}"
        )

        if agg.sessions_compared > 0:
            console.print(
                f"\n{mark_provenance('Aggregate Metrics', Provenance.EXPERIMENTAL)} (over compared sessions):"
            )

            def _fmt_r(v: float | None) -> str:
                return f"{v:.3f}" if v is not None else "N/A"

            console.print(
                f"  Mean Spearman r (flattening_severity): {_fmt_r(agg.mean_spearman_flattening_r)}"
            )
            console.print(
                f"  Mean Spearman r (flatness_index):      {_fmt_r(agg.mean_spearman_flatness_r)}"
            )
            console.print(
                f"  Mean AUC @ FLG >= 0.25:                {_fmt_r(agg.mean_auc_t25)}"
            )
            console.print(
                f"  Mean AUC @ FLG >= 0.50:                {_fmt_r(agg.mean_auc_t50)}"
            )
            console.print(
                f"  Mean Spearman r (flow_class weight):   {_fmt_r(agg.mean_spearman_class_weight_r)}"
            )
            console.print(
                f"  Mean class-weight AUC @ FLG >= 0.25:  {_fmt_r(agg.mean_auc_class_t25)}"
            )
            console.print(
                f"  Mean class-weight AUC @ FLG >= 0.50:  {_fmt_r(agg.mean_auc_class_t50)}"
            )
            if agg.cross_night_spearman_r is not None:
                console.print(
                    f"  Cross-night Spearman r (95th pct):     {agg.cross_night_spearman_r:.3f}"
                    f"  (p={agg.cross_night_spearman_p:.3f})"
                )

        compared_sessions = [s for s in report.sessions if s.skipped_reason is None]
        skipped_sessions = [s for s in report.sessions if s.skipped_reason is not None]

        if compared_sessions:
            console.print(
                f"\n{mark_provenance('Per-Session Results', Provenance.EXPERIMENTAL)}:"
            )
            console.print(
                f"{'Date':<12} {'ID':<6} {'N':<6} {'Spear-flat':<12} {'Spear-fi':<10} {'AUC25':<8} {'AUC50':<8} {'Nc':<6} {'Spear-cw':<10} {'cwAUC25':<9} {'cwAUC50':<9}"
            )
            print_footer()

            for s in compared_sessions[:10]:

                def _fv(v: float | None) -> str:
                    return f"{v:.3f}" if v is not None else " N/A"

                warn = "!" if s.low_sample_warning else " "
                console.print(
                    f"{s.date:<12} "
                    f"{s.session_id:<6} "
                    f"{s.n_breaths_compared:<5}{warn} "
                    f"{_fv(s.spearman_flattening_r):<12} "
                    f"{_fv(s.spearman_flatness_r):<10} "
                    f"{_fv(s.auc_t25):<8} "
                    f"{_fv(s.auc_t50):<8} "
                    f"{s.n_class_breaths_compared:<6} "
                    f"{_fv(s.spearman_class_weight_r):<10} "
                    f"{_fv(s.auc_class_t25):<9} "
                    f"{_fv(s.auc_class_t50):<9}"
                )

            if len(compared_sessions) > 10:
                console.print(f"... and {len(compared_sessions) - 10} more sessions")

        if skipped_sessions:
            console.print(f"\nSkipped sessions: {len(skipped_sessions)}")
            for s in skipped_sessions[:5]:
                console.print(f"  {s.date} (id={s.session_id}): {s.skipped_reason}")
            if len(skipped_sessions) > 5:
                console.print(f"  ... and {len(skipped_sessions) - 5} more")

        if export:
            export_path = Path(export)
            if export_path.suffix == ".json":
                export_fl_report_json(report, export_path)
                console.print(f"\nReport exported to {export_path}")
            elif export_path.suffix == ".csv":
                export_fl_report_csv(report, export_path)
                console.print(f"\nReport exported to {export_path}")
            else:
                raise click.ClickException(
                    f"Unknown export format '{export_path.suffix}'. Use .json or .csv"
                )
