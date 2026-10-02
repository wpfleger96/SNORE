"""validate command — batch session validation against machine events."""

from __future__ import annotations

import logging
import sys

from datetime import datetime
from pathlib import Path

import click

from snore.cli.decorators import (
    CliCtx,
    cli_error_boundary,
    date_range_options,
    profile_scoped_command,
)
from snore.cli.display import (
    ICON_CHECK,
    ICON_ERROR,
    ICON_WARN,
    Column,
    console,
    err_console,
    mark_provenance,
    print_footer,
    print_header,
    print_kv,
    print_table,
)
from snore.provenance import Provenance, field_provenance
from snore.validation.report import AggregateMetrics


@click.command()
@date_range_options
@click.option(
    "--mode",
    "-m",
    default="aasm",
    type=str,
    help="Detection mode to validate (default: aasm)",
)
@click.option(
    "--export",
    type=click.Path(),
    help="Export report to file (.json or .csv)",
)
@click.option(
    "--integrity",
    is_flag=True,
    default=False,
    help="Run data-integrity check instead of detection validation",
)
@click.option(
    "--device-id",
    type=int,
    default=None,
    help="Restrict integrity check to a specific device ID",
)
@profile_scoped_command
async def validate(
    ctx: CliCtx,
    date_from: datetime | None,
    date_to: datetime | None,
    mode: str,
    export: str | None,
    integrity: bool,
    device_id: int | None,
) -> None:
    """
    Run batch validation across multiple sessions.

    Validates SNORE's detection against machine events for sessions in the specified
    date range, and displays aggregate metrics.

    Use --integrity to run a structural data-integrity check instead (dates not required).
    """
    if device_id is not None and not integrity:
        raise click.UsageError("--device-id requires --integrity")

    if integrity:
        await _run_integrity(ctx, device_id)
        return

    if date_from is None or date_to is None:
        raise click.UsageError("--from and --to are required unless --integrity is set")

    if date_from > date_to:
        raise click.ClickException("--from date must be before or equal to --to date")

    from snore.validation import (  # noqa: PLC0415
        BatchValidator,
        export_report_csv,
        export_report_json,
    )

    async with cli_error_boundary("Validation error"):
        validator = BatchValidator(ctx.db, ctx.profile_id)

        console.print(
            f"Running validation from {date_from.date()} to {date_to.date()}..."
        )
        console.print(f"Mode: {mode}\n")

        report = await validator.validate_date_range(
            date_from.strftime("%Y-%m-%d"),
            date_to.strftime("%Y-%m-%d"),
            mode=mode,
        )

        print_footer()
        print_header("VALIDATION REPORT")
        console.print(
            f"Date Range: {report.date_range_start} to {report.date_range_end}"
        )
        console.print(f"Sessions Analyzed: {report.aggregate.total_sessions}")
        console.print(f"Total Machine Events: {report.aggregate.total_machine_events}")
        programmatic = mark_provenance(
            "Total Programmatic Events",
            field_provenance(AggregateMetrics, "total_programmatic_events"),
        )
        console.print(f"{programmatic}: {report.aggregate.total_programmatic_events}")

        console.print(
            f"\n{mark_provenance('Aggregate Metrics', Provenance.EXPERIMENTAL)}:"
        )
        console.print(
            f"  Apneas:     "
            f"Avg Sens: {report.aggregate.avg_apnea_sensitivity * 100:.0f}%  "
            f"Avg Prec: {report.aggregate.avg_apnea_precision * 100:.0f}%  "
            f"Avg F1: {report.aggregate.avg_apnea_f1:.2f}"
        )
        console.print(
            f"  Hypopneas:  "
            f"Avg Sens: {report.aggregate.avg_hypopnea_sensitivity * 100:.0f}%  "
            f"Avg Prec: {report.aggregate.avg_hypopnea_precision * 100:.0f}%  "
            f"Avg F1: {report.aggregate.avg_hypopnea_f1:.2f}"
        )

        if report.aggregate.low_sensitivity_sessions:
            console.print(
                f"\nSessions with Low Sensitivity (<60%): "
                f"{len(report.aggregate.low_sensitivity_sessions)}"
            )
            console.print(
                f"  Session IDs: {report.aggregate.low_sensitivity_sessions[:10]}"
            )
            if len(report.aggregate.low_sensitivity_sessions) > 10:
                console.print(
                    f"  ... and {len(report.aggregate.low_sensitivity_sessions) - 10} more"
                )

        console.print(
            f"\n{mark_provenance('Per-Session Results', Provenance.EXPERIMENTAL)}:"
        )
        console.print(
            f"{'Date':<12} {'ID':<6} {'Machine':<8} {'Prog':<8} {'Apnea Sens':<11} {'Hypopnea Sens':<13}"
        )
        print_footer()

        for session in report.sessions[:10]:
            console.print(
                f"{session.date:<12} "
                f"{session.session_id:<6} "
                f"{session.machine_event_count:<8} "
                f"{session.programmatic_event_count:<8} "
                f"{session.apnea_sensitivity * 100:>6.0f}%     "
                f"{session.hypopnea_sensitivity * 100:>6.0f}%"
            )

        if len(report.sessions) > 10:
            console.print(f"... and {len(report.sessions) - 10} more sessions")

        if export:
            export_path = Path(export)
            if export_path.suffix == ".json":
                export_report_json(report, export_path)
                console.print(f"\nReport exported to {export_path}")
            elif export_path.suffix == ".csv":
                export_report_csv(report, export_path)
                console.print(f"\nReport exported to {export_path}")
            else:
                raise click.ClickException(
                    f"Unknown export format '{export_path.suffix}'. Use .json or .csv"
                )


async def _run_integrity(ctx: CliCtx, device_id: int | None) -> None:
    """Run the data-integrity check and print results; exits nonzero on issues."""
    from snore.validation.batch import BatchValidator  # noqa: PLC0415

    try:
        validator = BatchValidator(ctx.db, ctx.profile_id)
        report = await validator.check_data_integrity(device_id=device_id)
    except Exception as e:
        import traceback  # noqa: PLC0415

        err_console.print(f"Integrity check error: {e}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        raise click.ClickException(str(e)) from e

    print_header("DATA INTEGRITY REPORT")
    filter_label = str(device_id) if device_id is not None else "all devices"
    print_kv("Checked at", report.checked_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
    print_kv("Device filter", filter_label)
    print_kv("Total issues", str(report.total_issues))
    print_footer()

    # Null day_id sessions
    console.print(
        f"\n{ICON_CHECK if not report.null_day_id_sessions else ICON_ERROR} "
        f"Sessions with null day_id: {len(report.null_day_id_sessions)}"
    )
    if report.null_day_id_sessions:
        print_table(
            columns=[Column("Session ID", 12)],
            rows=[[str(sid)] for sid in report.null_day_id_sessions[:20]],
        )
        if len(report.null_day_id_sessions) > 20:
            console.print(f"  ... and {len(report.null_day_id_sessions) - 20} more")

    # Overlapping session pairs
    console.print(
        f"\n{ICON_CHECK if not report.overlapping_session_pairs else ICON_ERROR} "
        f"Overlapping session pairs: {len(report.overlapping_session_pairs)}"
    )
    if report.overlapping_session_pairs:
        print_table(
            columns=[
                Column("Device", 8),
                Column("Session A", 24),
                Column("Session B", 24),
            ],
            rows=[
                [
                    str(p.device_id),
                    p.session_a_device_session_id,
                    p.session_b_device_session_id,
                ]
                for p in report.overlapping_session_pairs[:20]
            ],
        )
        if len(report.overlapping_session_pairs) > 20:
            console.print(
                f"  ... and {len(report.overlapping_session_pairs) - 20} more"
            )

    # Cross-parser same-day
    console.print(
        f"\n{ICON_CHECK if not report.cross_parser_same_day else ICON_WARN} "
        f"Cross-parser same-day entries: {len(report.cross_parser_same_day)}"
    )
    if report.cross_parser_same_day:
        print_table(
            columns=[Column("Device", 8), Column("Date", 12), Column("Sources", 0)],
            rows=[
                [
                    str(cp.device_id),
                    str(cp.day_date),
                    ", ".join(cp.import_sources),
                ]
                for cp in report.cross_parser_same_day[:20]
            ],
        )
        if len(report.cross_parser_same_day) > 20:
            console.print(f"  ... and {len(report.cross_parser_same_day) - 20} more")

    print_footer()

    # Cross-parser same-day is surfaced as a warning, not an error: having two
    # parsers record the same night is unusual but not corrupt.  Only hard
    # structural issues (NULL day_id, overlapping session pairs) exit nonzero.
    hard_issue_count = len(report.null_day_id_sessions) + len(
        report.overlapping_session_pairs
    )
    if hard_issue_count > 0:
        sys.exit(1)
