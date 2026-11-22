"""stats command — therapy usage and clinical statistics."""

from __future__ import annotations

from typing import cast

import click

from snore.cli.decorators import CliCtx, profile_scoped_command
from snore.cli.display import (
    ICON_CHART,
    Column,
    console,
    mark_experimental,
    print_footer,
    print_header,
    print_kv,
    print_subsection,
    print_table,
)
from snore.constants import COMPLIANCE_THRESHOLD_HOURS, LEAK_REDLINE_LPM
from snore.parsers.base import night_clock_time
from snore.services.schemas import PeriodStatistics


@click.command()
@click.option("--days", type=int, help="Limit to last N days")
@click.option(
    "--period",
    type=click.Choice(["day", "week", "month", "6month", "year"]),
    help="Show statistics broken down by period",
)
@click.option("--trend", is_flag=True, help="Show trend analysis chart")
@click.option(
    "--records", is_flag=True, help="Show top 5 best/worst days for key metrics"
)
@profile_scoped_command
async def stats(
    ctx: CliCtx,
    days: int | None,
    period: str | None,
    trend: bool,
    records: bool,
) -> None:
    """Show therapy usage and clinical statistics."""
    from snore.services.stats_service import StatsService  # noqa: PLC0415

    if trend and not period:
        period = "week"

    service = StatsService(ctx.db, ctx.profile_id)
    summary = await service.get_summary(days)

    if not summary:
        print_header("Therapy Statistics", ICON_CHART)
        console.print("\nNo therapy data found.")
        print_footer()
        console.print()
        return

    print_header("Therapy Statistics", ICON_CHART)

    print_subsection("Date Range")
    print_kv("First session", str(summary.first_date))
    print_kv("Last session", str(summary.last_date))
    print_kv("Days since last use", str(summary.days_since_last))

    print_subsection("Usage")
    print_kv("Total therapy hours", f"{summary.total_hours:,.1f} hrs")
    print_kv("Average per night", f"{summary.avg_hours:.1f} hrs")
    print_kv("Days with data", str(summary.days_with_data))
    print_kv(
        f"Days >= {COMPLIANCE_THRESHOLD_HOURS:g} hrs",
        str(summary.days_compliant),
    )
    print_kv("Compliance (all days)", f"{summary.compliance_pct:.1f}%")

    print_subsection("Clinical")
    print_kv(
        "Average AHI",
        f"{summary.avg_ahi:.1f}" if summary.avg_ahi is not None else "N/A",
    )
    if summary.avg_rei is not None:
        print_kv("Average REI", f"{summary.avg_rei:.1f}")

    if summary.avg_pressure is not None:
        print_subsection("Pressure")
        print_kv("Average", f"{summary.avg_pressure:.1f} cmH₂O")
        if summary.min_pressure is not None and summary.max_pressure is not None:
            print_kv(
                "Range",
                f"{summary.min_pressure:.1f} - {summary.max_pressure:.1f} cmH₂O",
            )

    if summary.avg_epap is not None:
        print_subsection("EPAP")
        print_kv("Average", f"{summary.avg_epap:.1f} cmH₂O")

    if summary.avg_leak is not None:
        print_subsection("Leak")
        print_kv("Average", f"{summary.avg_leak:.1f} L/min")
        # A SNORE judgement with no schema field behind it, so it is marked here.
        leak_assessment = (
            "well controlled" if summary.avg_leak < LEAK_REDLINE_LPM else "elevated"
        )
        print_kv(mark_experimental("Assessment"), leak_assessment)

    if summary.avg_spo2 is not None:
        print_subsection("SpO₂")
        print_kv("Average", f"{summary.avg_spo2:.1f}%")
        if summary.min_spo2 is not None:
            print_kv("Minimum recorded", f"{summary.min_spo2:.0f}%")

    if summary.total_spo2_time_below_90 > 0:
        minutes_below_90 = summary.total_spo2_time_below_90 / 60
        print_kv(
            "Time below 90%",
            f"{minutes_below_90:.1f} minutes",
        )

    if summary.avg_pulse is not None:
        print_subsection("Pulse")
        print_kv("Average", f"{summary.avg_pulse:.1f} BPM")

    respiratory = [
        (
            "Respiratory Rate",
            summary.avg_respiratory_rate,
            "{:.1f} breaths/min",
        ),
        ("Tidal Volume", summary.avg_tidal_volume, "{:.0f} mL"),
        (
            "Minute Ventilation",
            summary.avg_minute_ventilation,
            "{:.1f} L/min",
        ),
    ]
    if any(value is not None for _, value, _ in respiratory):
        print_subsection("Respiratory")
        for label, value, fmt in respiratory:
            if value is not None:
                print_kv(label, fmt.format(value))

    if summary.event_counts:
        print_subsection("Events")
        for ec in summary.event_counts:
            print_kv(ec.event_type, f"{ec.count:,} ({ec.percentage:.1f}%)")

    if period:
        from snore.analysis.calculations import PeriodType  # noqa: PLC0415

        period_literal = cast(PeriodType, period)
        period_stats: list[PeriodStatistics] = await service.get_period_statistics(
            period_literal, days
        )

        if period_stats:
            period_names = {
                "day": "Daily",
                "week": "Weekly",
                "month": "Monthly",
                "6month": "6-Month",
                "year": "Yearly",
            }

            print_header(f"Therapy Statistics ({period_names[period]})", wide=True)

            period_rows = []
            breakdown_rows = []
            for period_stat in period_stats:  # type: PeriodStatistics
                if period == "day":
                    period_label = str(period_stat.period_start)
                elif period == "week":
                    period_label = f"{period_stat.period_start.strftime('%Y-W%U')}"
                elif period == "month":
                    period_label = period_stat.period_start.strftime("%b %Y")
                elif period == "6month":
                    half = "H1" if period_stat.period_start.month == 1 else "H2"
                    period_label = f"{period_stat.period_start.year} {half}"
                else:
                    period_label = str(period_stat.period_start.year)

                breakdown_rows.append(
                    (
                        period_label,
                        *(
                            _number_cell(value, "{:.1f}")
                            for value in (
                                period_stat.avg_oai,
                                period_stat.avg_cai,
                                period_stat.avg_hi,
                                period_stat.avg_uai,
                            )
                        ),
                        _number_cell(period_stat.avg_time_in_apnea_minutes, "{:.1f}m"),
                        _number_cell(period_stat.avg_tidal_volume_ml, "{:.0f} mL"),
                        _clock_cell(period_stat.avg_bedtime_hours),
                        _clock_cell(period_stat.avg_wake_time_hours),
                    )
                )

                days_str = f"{period_stat.days_used}/{period_stat.days_in_period}"

                hours_str = (
                    f"{period_stat.avg_hours_per_day:.1f}h"
                    if period_stat.avg_hours_per_day is not None
                    else "N/A"
                )

                avg_ahi_str = (
                    f"{period_stat.avg_ahi:.1f}"
                    if period_stat.avg_ahi is not None
                    else "N/A"
                )

                med_ahi_str = (
                    f"{period_stat.median_ahi:.1f}"
                    if period_stat.median_ahi is not None
                    else "N/A"
                )

                period_rows.append(
                    (
                        period_label,
                        days_str,
                        hours_str,
                        _number_cell(period_stat.compliance_pct, "{:.0f}%"),
                        avg_ahi_str,
                        med_ahi_str,
                        _number_cell(period_stat.device_flg_95th, "{:.2f}"),
                        _number_cell(period_stat.snore_95th, "{:.2f}"),
                    )
                )

            print_table(
                [
                    Column("Period", 12),
                    Column("Days", 6),
                    Column("Avg Hours", 11),
                    Column(f">={COMPLIANCE_THRESHOLD_HOURS:g}h", 6),
                    Column("Avg AHI", 9),
                    Column("Med AHI", 9),
                    Column("FL 95th", 9),
                    Column("Snore 95th", 11),
                ],
                period_rows,
            )

            print_subsection("AHI Breakdown and Sessions")
            print_table(
                [
                    Column("Period", 12),
                    Column("OAI", 6),
                    Column("CAI", 6),
                    Column("HI", 6),
                    Column("UAI", 6),
                    Column("In Apnea", 10),
                    Column("Avg TV", 9),
                    Column("Bedtime", 9),
                    Column("Wake", 0),
                ],
                breakdown_rows,
            )

            print_footer(wide=True)

            if trend:
                import plotext as plt  # noqa: PLC0415

                trends = await service.get_trends(period_literal, days)
                ahi_trend = trends.ahi

                ahi_values = [v for _, v in ahi_trend if v is not None]
                if ahi_values:
                    dates_for_plot = [d for d, v in ahi_trend if v is not None]
                    date_labels = [d.strftime("%Y-%m-%d") for d in dates_for_plot]
                    x_indices = list(range(len(ahi_values)))

                    print_header("AHI Trend", wide=True)

                    fig = plt.figure
                    fig.clear()
                    fig.draw(
                        fig.signal(x_indices, ahi_values, marker="braille").lines()
                    )
                    fig.ruler("x").ticks(x_indices, date_labels)
                    fig.title("AHI Over Time")
                    fig.label("Period", axis="x")
                    fig.label("AHI (events/hour)", axis="y")
                    # fig.show() writes to fd 1 from native code, bypassing sys.stdout
                    # and the console; build() + console.out keeps ordering/capture.
                    console.out(fig.build().string(colorless=True), highlight=False)

                    print_footer(wide=True)

    if records:
        records_data = await service.get_records(days, top_n=5)

        # An unset metric had no qualifying day; nothing set means no records.
        if records_data.model_fields_set:
            print_header("Records (Top 5)", wide=True)

            metric_rows = [
                ("ahi", records_data.ahi, "Best AHI", "Worst AHI"),
                ("leak", records_data.leak, "Best Leak", "Worst Leak"),
                (
                    "therapy_hours",
                    records_data.therapy_hours,
                    "Longest Sessions",
                    "Shortest Sessions",
                ),
                ("spo2_min", records_data.spo2_min, "Best SpO2 Min", "Worst SpO2 Min"),
            ]

            for metric, extremes, best_label, worst_label in metric_rows:
                if extremes is None:
                    continue
                best_records = extremes.best
                worst_records = extremes.worst

                record_rows = []
                max_rows = max(len(best_records), len(worst_records))
                for i in range(max_rows):
                    best_str = ""
                    worst_str = ""

                    if i < len(best_records):
                        dt, val = best_records[i]
                        if metric == "therapy_hours":
                            best_str = f"  {dt}: {val:.1f}h"
                        else:
                            best_str = f"  {dt}: {val:.1f}"

                    if i < len(worst_records):
                        dt, val = worst_records[i]
                        if metric == "therapy_hours":
                            worst_str = f"{dt}: {val:.1f}h"
                        else:
                            worst_str = f"{dt}: {val:.1f}"

                    record_rows.append((best_str, worst_str))

                console.print()
                print_table(
                    [Column(best_label, 35), Column(worst_label, 0)],
                    record_rows,
                )

            print_footer(wide=True)

    console.print()
    print_footer()
    console.print()


def _number_cell(value: float | None, fmt: str) -> str:
    """Format ``value`` with ``fmt``, ``N/A`` when there is no value."""
    return fmt.format(value) if value is not None else "N/A"


def _clock_cell(hours_after_noon: float | None) -> str:
    """Format hours after noon of a night's date as a 24-hour clock time."""
    if hours_after_noon is None:
        return "N/A"
    return night_clock_time(hours_after_noon).strftime("%H:%M")
