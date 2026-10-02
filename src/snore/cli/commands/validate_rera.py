"""validate-rera command — score both RERA definitions against machine RE."""

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
from snore.cli.display import (
    console,
    fmt_sig,
    mark_provenance,
    print_footer,
    print_header,
)
from snore.provenance import Provenance, field_provenance
from snore.validation.rera_report import ReraAggregateMetrics


@click.command()
@date_range_options_required
@click.option(
    "--export",
    type=click.Path(),
    help="Export report to file (.json or .csv)",
)
@profile_scoped_command
async def validate_rera(
    ctx: CliCtx,
    date_from: datetime,
    date_to: datetime,
    export: str | None,
) -> None:
    """
    Validate SNORE's two RERA definitions against machine-flagged RE events.

    Scores the amplitude-crescendo detector (mode_result.reras) and the
    query-time FL-run proxy (recomputed from stored breaths) independently
    against the device's RE events.  ResMed flags RE conservatively, so most
    sessions have zero machine RE and are reported as skipped; near-zero
    precision on the rest is expected — read it against the aggregate's
    chance-precision floor.
    """
    if date_from > date_to:
        raise click.ClickException("--from date must be before or equal to --to date")

    from snore.validation import (  # noqa: PLC0415
        ReraValidator,
        export_rera_report_csv,
        export_rera_report_json,
    )

    async with cli_error_boundary("RERA validation error"):
        validator = ReraValidator(ctx.db, ctx.profile_id)

        console.print(
            f"Running RERA validation from {date_from.date()} to {date_to.date()}..."
        )

        report = await validator.validate_date_range(
            date_from.strftime("%Y-%m-%d"),
            date_to.strftime("%Y-%m-%d"),
        )

        agg = report.aggregate

        print_footer()
        print_header("RERA VALIDATION REPORT")
        console.print(
            f"Date Range: {report.date_range_start} to {report.date_range_end}"
        )
        console.print(f"Total Sessions:          {agg.total_sessions}")
        console.print(f"Sessions with machine RE: {agg.sessions_with_machine_re}")
        console.print(f"Skipped (no machine RE):  {agg.sessions_skipped_no_machine_re}")
        console.print(f"Skipped (no analysis):    {agg.sessions_skipped_no_analysis}")
        console.print(
            f"Skipped (no breaths):     {agg.sessions_skipped_no_valid_breaths}"
        )
        console.print(f"Skipped (error):          {agg.sessions_skipped_error}")

        console.print("\nEvent counts / densities (per therapy hour):")
        machine_re = mark_provenance(
            "Machine RE", field_provenance(ReraAggregateMetrics, "total_machine_re")
        )
        machine_re_density = mark_provenance(
            f"{fmt_sig(agg.machine_re_density)}/h",
            field_provenance(ReraAggregateMetrics, "machine_re_density"),
        )
        console.print(
            f"  {machine_re}:  {agg.total_machine_re:<6} ({machine_re_density})"
        )
        amplitude = mark_provenance(
            "Amplitude", field_provenance(ReraAggregateMetrics, "total_amplitude_reras")
        )
        console.print(
            f"  {amplitude}:  {agg.total_amplitude_reras:<6} "
            f"({fmt_sig(agg.amplitude_density)}/h)"
        )
        proxy = mark_provenance(
            "FL-run proxy", field_provenance(ReraAggregateMetrics, "total_proxy_reras")
        )
        console.print(
            f"  {proxy}:{agg.total_proxy_reras:<6} ({fmt_sig(agg.proxy_density)}/h)"
        )
        console.print(
            "  Chance-precision floor (whole-dataset, "
            f"tol={agg.match_tolerance_seconds}s): "
            f"{fmt_sig(agg.chance_precision_floor)}"
        )

        if agg.sessions_with_machine_re > 0:
            console.print(
                f"\n{mark_provenance('Scores', Provenance.EXPERIMENTAL)} over sessions with machine RE (amplitude | proxy):"
            )
            console.print(
                f"  Sensitivity (mean):   "
                f"{fmt_sig(agg.mean_amplitude_sensitivity)} | "
                f"{fmt_sig(agg.mean_proxy_sensitivity)}"
            )
            console.print(
                f"  Sensitivity (pooled): "
                f"{fmt_sig(agg.pooled_amplitude_sensitivity)} | "
                f"{fmt_sig(agg.pooled_proxy_sensitivity)}"
            )
            console.print(
                f"  Precision (mean):     "
                f"{fmt_sig(agg.mean_amplitude_precision)} | "
                f"{fmt_sig(agg.mean_proxy_precision)}"
            )
            console.print(
                f"  Precision (pooled):   "
                f"{fmt_sig(agg.pooled_amplitude_precision)} | "
                f"{fmt_sig(agg.pooled_proxy_precision)}"
            )
            console.print(
                f"  F1 (mean):            {fmt_sig(agg.mean_amplitude_f1)} | "
                f"{fmt_sig(agg.mean_proxy_f1)}"
            )
            console.print(
                "  Chance-precision floor (scored hours): "
                f"{fmt_sig(agg.scored_chance_precision_floor)}"
            )

        # Most-informative rows lead: most machine RE first.
        scored = sorted(
            (s for s in report.sessions if s.skipped_reason is None),
            key=lambda s: s.machine_re_count,
            reverse=True,
        )
        _table_cap = 20
        if scored:
            shown = scored[:_table_cap]
            if len(scored) > _table_cap:
                header = (
                    f"\n{mark_provenance('Scored sessions', Provenance.EXPERIMENTAL)} (top {_table_cap} of {len(scored)} "
                    "by machine RE; amplitude / proxy):"
                )
            else:
                header = f"\n{mark_provenance('Scored sessions', Provenance.EXPERIMENTAL)} ({len(scored)}; amplitude / proxy):"
            console.print(header)
            console.print(
                f"{'Date':<12} {'ID':<6} {'RE':<4} "
                f"{'aSens':<9} {'aPrec':<9} {'pSens':<9} {'pPrec':<9}"
            )
            print_footer()
            for s in shown:
                console.print(
                    f"{s.date:<12} "
                    f"{s.session_id:<6} "
                    f"{s.machine_re_count:<4} "
                    f"{fmt_sig(s.amplitude_sensitivity):<9} "
                    f"{fmt_sig(s.amplitude_precision):<9} "
                    f"{fmt_sig(s.proxy_sensitivity):<9} "
                    f"{fmt_sig(s.proxy_precision):<9}"
                )
            if len(scored) > _table_cap:
                console.print(
                    f"... {len(scored) - _table_cap} more scored sessions "
                    "not shown (use --export for the full set)"
                )

        if export:
            export_path = Path(export)
            if export_path.suffix == ".json":
                export_rera_report_json(report, export_path)
                console.print(f"\nReport exported to {export_path}")
            elif export_path.suffix == ".csv":
                export_rera_report_csv(report, export_path)
                console.print(f"\nReport exported to {export_path}")
            else:
                raise click.ClickException(
                    f"Unknown export format '{export_path.suffix}'. Use .json or .csv"
                )
