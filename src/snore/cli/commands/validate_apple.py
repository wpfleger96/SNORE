"""validate-apple command — cross-source FL/RERA validation against Apple Health."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from rich.table import Table

from snore.cli.decorators import (
    CliCtx,
    cli_error_boundary,
    date_range_options_required,
    profile_scoped_command,
)
from snore.cli.display import console, mark_provenance, print_header
from snore.provenance import Provenance


def _fmt_r(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


@click.command()
@date_range_options_required
@click.option(
    "--device-id",
    type=int,
    default=None,
    help="Pin SNORE device id (disambiguates multi-device nights)",
)
@click.option(
    "--export",
    type=click.Path(),
    help="Export report to file (.json or .csv)",
)
@profile_scoped_command
async def validate_apple(
    ctx: CliCtx,
    date_from: datetime,
    date_to: datetime,
    device_id: int | None,
    export: str | None,
) -> None:
    """
    Validate SNORE FL/RERA nightly indices against independent Apple Health signals.

    Correlates rera_index and fl_class_ge4_pct against Apple's sleeping-breathing-
    disturbance metric and watch-derived fragmentation (awake_seconds,
    sleep_efficiency_pct) using per-pair Spearman correlations.  Nights whose
    SNORE analysis has not run or is stale, and nights lacking Apple data, are
    reported as skipped.
    """
    if date_from > date_to:
        raise click.ClickException("--from date must be before or equal to --to date")

    # Reject an unsupported --export suffix up front, before running the
    # validation, so a bad path never wastes a full range scan.
    if export is not None and Path(export).suffix not in (".json", ".csv"):
        raise click.ClickException(
            f"Unknown export format '{Path(export).suffix}'. Use .json or .csv"
        )

    from snore.validation import (  # noqa: PLC0415
        AppleCrossValidator,
        export_apple_cross_report_csv,
        export_apple_cross_report_json,
    )

    async with cli_error_boundary("Apple cross-source validation error"):
        validator = AppleCrossValidator(ctx.db, ctx.profile_id)

        console.print(
            f"Running Apple cross-source validation from "
            f"{date_from.date()} to {date_to.date()}..."
        )

        report = await validator.validate_date_range(
            date_from.strftime("%Y-%m-%d"),
            date_to.strftime("%Y-%m-%d"),
            device_id=device_id,
        )

        agg = report.aggregate

        print_header("APPLE CROSS-SOURCE VALIDATION REPORT")
        console.print(
            f"Date Range: {report.date_range_start} to {report.date_range_end}"
        )
        console.print(f"Total Nights:               {agg.total_nights}")
        console.print(f"Nights w/ Apple BD:         {agg.n_with_apple_bd}")
        console.print(f"Skipped (no Apple BD):      {agg.n_skipped_no_apple_bd}")
        console.print(f"Skipped (analysis not run): {agg.n_analysis_not_run}")
        console.print(f"Skipped (analysis stale):   {agg.n_analysis_stale}")
        console.print(f"Skipped (device ambiguous): {agg.n_device_ambiguous}")
        console.print(
            "[dim](skip counters are independent axes over the same "
            "nights; do not sum them)[/dim]"
        )

        corr_table = Table(
            title=mark_provenance(
                "Cross-source Spearman correlations", Provenance.EXPERIMENTAL
            )
        )
        corr_table.add_column("Metric pair")
        corr_table.add_column("rho", justify="right")
        corr_table.add_column("p", justify="right")
        corr_table.add_column("n", justify="right")
        corr_table.add_column("reason")
        for label, pair in (
            ("rera_index vs apple_bd", agg.rera_vs_apple_bd),
            ("fl_ge4_pct vs apple_bd", agg.fl_vs_apple_bd),
            ("rera_index vs awake_s", agg.rera_vs_awake_seconds),
            ("fl_ge4_pct vs sleep_eff", agg.fl_vs_sleep_efficiency),
        ):
            corr_table.add_row(
                label,
                _fmt_r(pair.rho),
                _fmt_r(pair.p_value),
                str(pair.n_paired_nights),
                pair.reason or "",
            )
        console.print(corr_table)

        if export:
            export_path = Path(export)
            # Suffix already validated up front to be .json or .csv.
            if export_path.suffix == ".json":
                export_apple_cross_report_json(report, export_path)
            else:
                export_apple_cross_report_csv(report, export_path)
            console.print(f"\nReport exported to {export_path}")
