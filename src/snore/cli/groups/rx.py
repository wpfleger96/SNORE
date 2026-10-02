"""RX (prescription) settings tracking and analysis commands."""

from __future__ import annotations

import click

from snore.cli.decorators import CliCtx, profile_scoped_command
from snore.cli.display import (
    Column,
    console,
    mark_provenance,
    print_footer,
    print_header,
    print_kv,
    print_subsection,
    print_table,
)
from snore.cli.display.settings import format_setting_key, format_setting_value
from snore.provenance import field_provenance
from snore.services.schemas import RxPeriodResponse


def _outcome(label: str, field: str) -> str:
    """Mark a period-outcome label; settings stay unmarked (device values)."""
    return mark_provenance(label, field_provenance(RxPeriodResponse, field))


def _format_change_value(key: str, val: str | None) -> str:
    if val is None:
        return "—"
    return format_setting_value(key, val)


def _format_pressure(settings: dict[str, str], *, short: bool = False) -> str:
    if "epap" in settings and "ipap" in settings:
        epap = settings["epap"]
        ipap = settings["ipap"]
        p = f"{epap}-{ipap}"
        return p if short else f"{p} cmH2O (EPAP-IPAP)"
    if "pressure_min" in settings and "pressure_max" in settings:
        p = f"{settings['pressure_min']}-{settings['pressure_max']}"
        return p if short else f"{p} cmH2O"
    if "pressure_fixed" in settings:
        p = settings["pressure_fixed"]
        return f"{p} (F)" if short else f"{p} cmH2O (Fixed)"
    return "?"


@click.group()
def rx() -> None:
    """RX (prescription) settings tracking and analysis."""
    pass


@rx.command("history")
@profile_scoped_command
async def rx_history(ctx: CliCtx) -> None:
    """
    Show RX settings history with average outcomes.

    Displays all prescription periods in chronological order with settings
    and key metrics like average AHI and therapy hours.

    Example:
        snore rx history
    """
    from snore.analysis.rx_tracker import RxTracker  # noqa: PLC0415

    stats_periods = await RxTracker(ctx.profile_id).get_history(ctx.db)

    if not stats_periods:
        console.print("No RX periods found")
        return

    print_header("RX Settings History", wide=True)

    for i, period in enumerate(stats_periods, 1):
        days_count = period.days_count
        end_str = (
            period.end_date.strftime("%Y-%m-%d")
            if i < len(stats_periods)
            else "present"
        )

        console.print(
            f"\nPeriod {i}: {period.start_date.strftime('%Y-%m-%d')} to {end_str} ({days_count} days)"
        )

        mode = period.settings.get("mode", "?")
        pressure_str = _format_pressure(period.settings)

        summary_parts = [
            f"Mode: {mode}",
            f"Pressure: {pressure_str}",
        ]
        if "epr_level" in period.settings and "epr_mode" in period.settings:
            summary_parts.append(
                f"EPR: {period.settings['epr_level']} {period.settings['epr_mode']}"
            )
        elif "ps" in period.settings:
            summary_parts.append(f"PS: {period.settings['ps']}")

        console.print("  " + " | ".join(summary_parts))

        ahi_val = f"{period.avg_ahi:.1f}" if period.avg_ahi is not None else "N/A"
        parts = [f"  {_outcome('Avg AHI', 'avg_ahi')}: {ahi_val}"]
        if period.avg_hours is not None:
            parts.append(
                f"{_outcome('Avg Hours', 'avg_hours')}: {period.avg_hours:.1f}"
            )
        if period.avg_leak is not None:
            parts.append(f"{_outcome('Avg Leak', 'avg_leak')}: {period.avg_leak:.1f}")
        console.print(" | ".join(parts))

    console.print()
    print_footer(wide=True)


@rx.command("current")
@profile_scoped_command
async def rx_current(ctx: CliCtx) -> None:
    """
    Show current RX settings period.

    Displays the most recent prescription settings along with outcomes.

    Example:
        snore rx current
    """
    from snore.analysis.rx_tracker import RxTracker  # noqa: PLC0415

    current = await RxTracker(ctx.profile_id).get_current(ctx.db)

    if current is None:
        console.print("No RX periods found")
        return

    days_count = current.days_count

    print_header("Current RX Settings", wide=True)
    console.print(
        f"Period: {current.start_date.strftime('%Y-%m-%d')} to present ({days_count} days)"
    )

    mode = current.settings.get("mode", "?")
    pressure_str = _format_pressure(current.settings)

    print_kv("Mode", str(mode), indent=0)
    print_kv("Pressure", str(pressure_str), indent=0)
    if "epr_level" in current.settings and "epr_mode" in current.settings:
        print_kv(
            "EPR",
            f"{current.settings['epr_level']} {current.settings['epr_mode']}",
            indent=0,
        )
    elif "ps" in current.settings:
        print_kv("PS", current.settings["ps"], indent=0)

    print_subsection("Outcomes")
    print_kv(
        "Avg AHI",
        f"{current.avg_ahi:.1f}" if current.avg_ahi is not None else "N/A",
        provenance=field_provenance(RxPeriodResponse, "avg_ahi"),
    )

    if current.median_ahi is not None:
        print_kv(
            "Median AHI",
            f"{current.median_ahi:.1f}",
            provenance=field_provenance(RxPeriodResponse, "median_ahi"),
        )

    if current.avg_hours is not None:
        print_kv(
            "Avg Hours",
            f"{current.avg_hours:.1f}",
            provenance=field_provenance(RxPeriodResponse, "avg_hours"),
        )

    if current.avg_leak is not None:
        print_kv(
            "Avg Leak",
            f"{current.avg_leak:.1f}",
            provenance=field_provenance(RxPeriodResponse, "avg_leak"),
        )

    print_footer(wide=True)


@rx.command("compare")
@click.option(
    "--min-days",
    type=int,
    default=7,
    help="Minimum days for period to be included (default: 7)",
)
@profile_scoped_command
async def rx_compare(ctx: CliCtx, min_days: int) -> None:
    """
    Compare RX periods and identify best/worst settings.

    Shows a table of all prescription periods with statistics side-by-side
    and highlights the best and worst periods based on average AHI.

    Example:
        snore rx compare
        snore rx compare --min-days 14
    """
    from snore.analysis.rx_tracker import RxTracker  # noqa: PLC0415

    comparison = await RxTracker(ctx.profile_id).get_comparison(
        ctx.db, min_days=min_days
    )
    stats_periods = comparison.periods

    if not stats_periods:
        console.print("No RX periods found")
        return

    if len(stats_periods) < 2:
        console.print(
            "At least 2 periods are needed for comparison. Use 'snore rx history' to view the single period."
        )
        return

    best = (
        stats_periods[comparison.best_index]
        if comparison.best_index is not None
        else None
    )
    worst = (
        stats_periods[comparison.worst_index]
        if comparison.worst_index is not None
        else None
    )

    print_header("RX Period Comparison", wide=True)
    console.print(
        f"{'Dates':<25} {'Days':<6} {_outcome('Avg AHI', 'avg_ahi'):<10} {_outcome('Avg Leak', 'avg_leak'):<10} {'Mode':<8} {'Pressure':<15} {'EPR':<10}"
    )
    print_footer(wide=True)

    for idx, period in enumerate(stats_periods):
        days_count = period.days_count
        start_str = period.start_date.strftime("%Y-%m-%d")
        end_str = (
            period.end_date.strftime("%Y-%m-%d")
            if idx < len(stats_periods) - 1
            else "present"
        )
        date_range = f"{start_str}..{end_str}"

        mode = period.settings.get("mode", "?")[:7]
        if "epr_level" in period.settings and "epr_mode" in period.settings:
            epr = f"{period.settings['epr_level']} {period.settings['epr_mode'][:2]}"
        elif "ps" in period.settings:
            epr = f"PS:{period.settings['ps']}"
        else:
            epr = "?"

        pressure_str = _format_pressure(period.settings, short=True)

        ahi_str = f"{period.avg_ahi:.1f}" if period.avg_ahi is not None else "N/A"
        leak_str = f"{period.avg_leak:.1f}" if period.avg_leak is not None else "N/A"

        marker = ""
        if idx == comparison.best_index:
            marker = "  <- Best"
        elif idx == comparison.worst_index:
            marker = "  <- Worst"

        console.print(
            f"{date_range:<25} {days_count:<6} {ahi_str:<10} {leak_str:<10} {mode:<8} {pressure_str:<15} {epr:<10}{marker}"
        )

    print_footer(wide=True)

    if best:
        console.print(
            f"\nBest Period ({_outcome('Avg AHI', 'avg_ahi')}: {best.avg_ahi:.1f}):"
        )
        console.print(
            f"  {best.start_date.strftime('%Y-%m-%d')} to {best.end_date.strftime('%Y-%m-%d')} ({best.days_count} days)"
        )
        console.print(f"  Settings: {best.settings}")

    if worst:
        console.print(
            f"\nWorst Period ({_outcome('Avg AHI', 'avg_ahi')}: {worst.avg_ahi:.1f}):"
        )
        console.print(
            f"  {worst.start_date.strftime('%Y-%m-%d')} to {worst.end_date.strftime('%Y-%m-%d')} ({worst.days_count} days)"
        )
        console.print(f"  Settings: {worst.settings}")


@rx.command("changes")
@profile_scoped_command
async def rx_changes(ctx: CliCtx) -> None:
    """
    Show day-level prescription settings changes across all devices.

    Displays each per-key settings change with old and new values,
    sorted most-recent-first. Useful for reviewing the titration trail
    without opening the web UI.

    Example:
        snore rx changes
    """
    from snore.analysis.rx_tracker import RxTracker  # noqa: PLC0415

    response = await RxTracker(ctx.profile_id).get_changes(ctx.db)
    # Stable sort preserves the service's within-date ordering (ascending by device_id, key).
    changes = sorted(response.changes, key=lambda c: c.date, reverse=True)

    if not changes:
        console.print("No RX settings changes found")
        return

    print_header("RX Settings Changes", wide=True)
    print_table(
        columns=[
            Column("Date", 12),
            Column("Device", 24),
            Column("Setting", 16),
            Column("Change", 0),
        ],
        rows=[
            (
                c.date.strftime("%Y-%m-%d"),
                c.device_name,
                format_setting_key(c.key),
                f"{_format_change_value(c.key, c.old_value)} → {_format_change_value(c.key, c.new_value)}",
            )
            for c in changes
        ],
        wide=True,
    )
    print_footer(wide=True)
