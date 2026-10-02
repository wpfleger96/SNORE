"""Health command group — Apple Health import and summaries."""

from __future__ import annotations

import asyncio
import re

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

from snore.cli.decorators import (
    CliCtx,
    actor_options,
    date_range_options,
    db_option,
    profile_scoped_command,
    resolve_profile_id_once,
)
from snore.cli.display import (
    ICON_CHART,
    ICON_SCAN,
    ICON_STATS,
    Column,
    console,
    mark_field,
    print_dry_run_complete,
    print_dry_run_header,
    print_footer,
    print_header,
    print_info,
    print_skip,
    print_success,
    print_table,
    print_warning,
)
from snore.provenance import field_provenance
from snore.services.schemas import HealthNightDetailRead, HealthNightSummaryRead

if TYPE_CHECKING:
    from snore.database.models import HealthSample


def _fmt_hours(secs: float | None) -> str:
    if secs is None:
        return "—"
    return f"{secs / 3600:.1f}h"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "—"
    return f"{pct:.0f}%"


def _fmt_duration_label(start: datetime, end: datetime) -> str:
    delta_minutes = int((end - start).total_seconds() / 60)
    if delta_minutes < 60:
        return f"{delta_minutes}m"
    hours, mins = divmod(delta_minutes, 60)
    return f"{hours}h {mins}m" if mins else f"{hours}h"


_STAGE_LABELS: dict[str, str] = {
    "InBed": "In Bed",
    "AsleepUnspecified": "Asleep (unspecified)",
    "Awake": "Awake",
    "AsleepCore": "Core",
    "AsleepDeep": "Deep",
    "AsleepREM": "REM",
}

_METRIC_LABELS: dict[str, str] = {
    "HKQuantityTypeIdentifierOxygenSaturation": "SpO2",
    "HKQuantityTypeIdentifierRespiratoryRate": "Resp. Rate",
    "HKQuantityTypeIdentifierAppleSleepingBreathingDisturbances": "Breathing Disturbances",
    "HKQuantityTypeIdentifierHeartRate": "Heart Rate",
}

# Matches "HKQuantityTypeIdentifier" or "HKCategoryTypeIdentifier" prefix.
_HK_PREFIX = re.compile(r"^HK(?:Quantity|Category)TypeIdentifier")


def _stage_label(value_text: str | None) -> str:
    if value_text is None:
        return "—"
    return _STAGE_LABELS.get(value_text, value_text)


def _metric_label(record_type: str) -> str:
    if record_type in _METRIC_LABELS:
        return _METRIC_LABELS[record_type]
    return _HK_PREFIX.sub("", record_type)


def _fmt_sample_value(sample: HealthSample) -> str:
    from snore.services.health_service import (  # noqa: PLC0415
        SPO2_RECORD_TYPE,
        spo2_display_pct,
    )

    if sample.value_num is None:
        return str(sample.value_text or "")
    if sample.record_type == SPO2_RECORD_TYPE:
        pct = spo2_display_pct(sample.value_num)
        # Round only values scaled from a fraction; others print as stored.
        shown = f"{pct:.1f}" if pct != sample.value_num else f"{pct}"
        return f"{shown} %"
    return f"{sample.value_num} {sample.unit}"


@click.group()
def health() -> None:
    """Apple Health data commands."""
    pass


@health.command("import")
@click.argument("path", type=click.Path())
@click.option("--dry-run", is_flag=True, help="Preview without writing to the database")
@date_range_options
@click.option("--limit", "-n", type=int, default=None, help="Stop after N records")
@click.option(
    "--batch-size",
    type=int,
    default=500,
    show_default=True,
    help="Records per database transaction",
)
# Not @profile_scoped_command: resolves the profile briefly, then manages its own per-batch write sessions.
@db_option
@actor_options
def health_import(
    path: str,
    dry_run: bool,
    date_from: datetime | None,
    date_to: datetime | None,
    limit: int | None,
    batch_size: int,
    db: str | None,
    actor_user: str | None,
    actor_profile: str | None,
) -> None:
    """Import an Apple Health export (zip file or extracted directory)."""
    export_path = Path(path)
    if not export_path.exists():
        raise click.ClickException(f"Path does not exist: {export_path}")

    processed_ref: list[int] = [0]

    def _progress(total: int) -> None:
        processed_ref[0] = total
        console.print(
            f"\r{ICON_SCAN} Processing... {total} records", end="", highlight=False
        )

    async def _run() -> None:
        from snore.services.health_import_service import (  # noqa: PLC0415
            HealthImportService,
        )

        profile_id = await resolve_profile_id_once(db, actor_user, actor_profile)

        if dry_run:
            print_dry_run_header("imported")

        try:
            result = await HealthImportService().import_file(
                export_path,
                profile_id,
                date_from=date_from.date() if date_from else None,
                date_to=date_to.date() if date_to else None,
                limit=limit,
                batch_size=batch_size,
                dry_run=dry_run,
                progress_callback=_progress,
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e

        if processed_ref[0] > 0:
            console.print()  # end the carriage-return progress line

        print_header("Import Summary", ICON_STATS)
        if dry_run:
            print_info(f"Would insert:     {result.inserted} new records")
            print_skip(f"Already present: {result.skipped} records")
        else:
            print_success(f"Inserted:        {result.inserted} new records")
            print_skip(f"Skipped:         {result.skipped} duplicate records")
            print_info(f"Nights updated:  {result.nights_recomputed}")

        if result.unknown_metrics:
            total_unknown = sum(result.unknown_metrics.values())
            top = sorted(result.unknown_metrics.items(), key=lambda kv: -kv[1])[:3]
            top_str = ", ".join(f"{k}: {v}" for k, v in top)
            suffix = "..." if len(result.unknown_metrics) > 3 else ""
            print_warning(
                f"Unhandled types: {total_unknown} records ({top_str}{suffix})"
            )

        print_footer()

        if dry_run:
            print_dry_run_complete("import")

    asyncio.run(_run())


@health.command("list")
@date_range_options
@click.option(
    "--limit",
    "-n",
    type=int,
    default=30,
    show_default=True,
    help="Max nights to show",
)
@profile_scoped_command
async def health_list(
    ctx: CliCtx,
    date_from: datetime | None,
    date_to: datetime | None,
    limit: int,
) -> None:
    """List nightly sleep summaries (newest first)."""
    from snore.services.health_service import HealthService  # noqa: PLC0415

    svc = HealthService(ctx.db, ctx.profile_id)
    rows, _ = await svc.list_nights(
        from_date=date_from.date() if date_from else None,
        to_date=date_to.date() if date_to else None,
        limit=limit,
        offset=0,
    )

    if not rows:
        console.print("No sleep data found")
        return

    print_table(
        [
            Column("Date", 12),
            Column(
                "Sleep",
                8,
                field_provenance(HealthNightSummaryRead, "total_sleep_seconds"),
            ),
            Column(
                "Eff%",
                6,
                field_provenance(HealthNightSummaryRead, "sleep_efficiency_pct"),
            ),
            Column("Core", 7, field_provenance(HealthNightSummaryRead, "core_seconds")),
            Column("Deep", 7, field_provenance(HealthNightSummaryRead, "deep_seconds")),
            Column("REM", 7, field_provenance(HealthNightSummaryRead, "rem_seconds")),
            Column("Source", 0),
        ],
        (
            (
                str(row.night_date),
                _fmt_hours(row.total_sleep_seconds),
                _fmt_pct(row.sleep_efficiency_pct),
                _fmt_hours(row.core_seconds),
                _fmt_hours(row.deep_seconds),
                _fmt_hours(row.rem_seconds),
                row.preferred_source or "—",
            )
            for row in rows
        ),
        wide=False,
    )


@health.command("show")
@click.argument("night_date", metavar="DATE", type=click.DateTime(formats=["%Y-%m-%d"]))
@profile_scoped_command
async def health_show(ctx: CliCtx, night_date: datetime) -> None:
    """Show sleep detail for a single night (YYYY-MM-DD)."""
    from sqlalchemy import select as sa_select  # noqa: PLC0415

    from snore.database.models import HealthSample  # noqa: PLC0415
    from snore.exceptions import NotFoundError  # noqa: PLC0415
    from snore.parsers.apple_health.type_handlers import SLEEP_TYPE  # noqa: PLC0415
    from snore.services.health_service import HealthService  # noqa: PLC0415

    night = night_date.date()
    svc = HealthService(ctx.db, ctx.profile_id)

    try:
        detail = await svc.get_night_detail(night)
    except NotFoundError:
        raise click.ClickException(f"No health data for {night}") from None

    sleep_samples = await svc.get_night_samples(night)

    quantity_samples = list(
        (
            await ctx.db.execute(
                sa_select(HealthSample)
                .where(
                    HealthSample.profile_id == ctx.profile_id,
                    HealthSample.night_date == night,
                    HealthSample.record_type != SLEEP_TYPE,
                )
                .order_by(HealthSample.start_time)
            )
        )
        .scalars()
        .all()
    )

    # Display sleep stage intervals from the preferred source.
    if sleep_samples:
        source_display = detail.preferred_source or "unknown"
        print_header(f"Sleep Intervals — {source_display}", ICON_SCAN)
        print_table(
            [
                Column("Start", 8),
                Column("End", 8),
                Column("Stage", 22),
                Column("Duration", 0),
            ],
            (
                (
                    f"{s.start_time:%H:%M}",
                    f"{s.end_time:%H:%M}",
                    _stage_label(s.value_text),
                    _fmt_duration_label(s.start_time, s.end_time),
                )
                for s in sleep_samples
            ),
            wide=False,
        )

    # Summary totals block.
    print_header("Totals", ICON_STATS)
    for label, field, value in [
        ("Total sleep", "total_sleep_seconds", _fmt_hours(detail.total_sleep_seconds)),
        ("Time in bed", "time_in_bed_seconds", _fmt_hours(detail.time_in_bed_seconds)),
        ("Efficiency", "sleep_efficiency_pct", _fmt_pct(detail.sleep_efficiency_pct)),
        ("Core", "core_seconds", _fmt_hours(detail.core_seconds)),
        ("Deep", "deep_seconds", _fmt_hours(detail.deep_seconds)),
        ("REM", "rem_seconds", _fmt_hours(detail.rem_seconds)),
        ("Awake", "awake_seconds", _fmt_hours(detail.awake_seconds)),
        ("Stage coverage", "stage_coverage_pct", _fmt_pct(detail.stage_coverage_pct)),
    ]:
        label = mark_field(label, HealthNightDetailRead, field)
        console.print(f"  {label:<18} {value}", markup=False, highlight=False)
    print_footer()

    # Quantity samples (SpO2, heart rate, etc.) for that night.
    if quantity_samples:
        print_header("Health Samples", ICON_CHART)
        print_table(
            [Column("Type", 30), Column("Value", 15), Column("Timestamp", 0)],
            (
                (
                    _metric_label(s.record_type),
                    _fmt_sample_value(s),
                    f"{s.start_time:%Y-%m-%d %H:%M}",
                )
                for s in quantity_samples
            ),
            wide=False,
        )
