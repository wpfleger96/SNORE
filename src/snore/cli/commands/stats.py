"""stats command — therapy usage and clinical statistics."""

from __future__ import annotations

from typing import cast

import click

from snore.cli.decorators import CliCtx, profile_scoped_command
from snore.cli.display import (
    ICON_CHART,
    Column,
    console,
    mark_provenance,
    print_footer,
    print_header,
    print_kv,
    print_subsection,
    print_table,
)
from snore.provenance import field_provenance
from snore.services.schemas import (
    PeriodStatistics,
    RecordsResponse,
    TherapySummary,
    TrendsResponse,
)


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
    print_kv(
        "Days since last use",
        str(summary.days_since_last),
        provenance=field_provenance(TherapySummary, "days_since_last"),
    )

    print_subsection("Usage")
    print_kv(
        "Total therapy hours",
        f"{summary.total_hours:,.1f} hrs",
        provenance=field_provenance(TherapySummary, "total_hours"),
    )
    print_kv(
        "Average per night",
        f"{summary.avg_hours:.1f} hrs",
        provenance=field_provenance(TherapySummary, "avg_hours"),
    )
    print_kv(
        "Days with data",
        str(summary.days_with_data),
        provenance=field_provenance(TherapySummary, "days_with_data"),
    )

    print_subsection("Clinical")
    print_kv(
        "Average AHI",
        f"{summary.avg_ahi:.1f}" if summary.avg_ahi is not None else "N/A",
        provenance=field_provenance(TherapySummary, "avg_ahi"),
    )
    print_kv(
        "Effectiveness",
        str(summary.effectiveness),
        provenance=field_provenance(TherapySummary, "effectiveness"),
    )

    if summary.avg_rei is not None:
        print_kv(
            "Average REI",
            f"{summary.avg_rei:.1f}",
            provenance=field_provenance(TherapySummary, "avg_rei"),
        )

    if summary.avg_pressure is not None:
        print_subsection("Pressure")
        print_kv(
            "Average",
            f"{summary.avg_pressure:.1f} cmH₂O",
            provenance=field_provenance(TherapySummary, "avg_pressure"),
        )
        if summary.min_pressure is not None and summary.max_pressure is not None:
            print_kv(
                "Range",
                f"{summary.min_pressure:.1f} - {summary.max_pressure:.1f} cmH₂O",
                provenance=field_provenance(TherapySummary, "min_pressure"),
            )

    if summary.avg_epap is not None:
        print_subsection("EPAP")
        print_kv(
            "Average",
            f"{summary.avg_epap:.1f} cmH₂O",
            provenance=field_provenance(TherapySummary, "avg_epap"),
        )

    if summary.avg_leak is not None:
        leak_provenance = field_provenance(TherapySummary, "avg_leak")
        print_subsection("Leak")
        print_kv("Average", f"{summary.avg_leak:.1f} L/min", provenance=leak_provenance)
        leak_assessment = "well controlled" if summary.avg_leak < 24 else "elevated"
        print_kv("Assessment", leak_assessment, provenance=leak_provenance)

    if summary.avg_spo2 is not None:
        print_subsection("SpO₂")
        print_kv(
            "Average",
            f"{summary.avg_spo2:.1f}%",
            provenance=field_provenance(TherapySummary, "avg_spo2"),
        )
        if summary.min_spo2 is not None:
            print_kv(
                "Minimum recorded",
                f"{summary.min_spo2:.0f}%",
                provenance=field_provenance(TherapySummary, "min_spo2"),
            )

    if summary.total_spo2_time_below_90 > 0:
        minutes_below_90 = summary.total_spo2_time_below_90 / 60
        print_kv(
            "Time below 90%",
            f"{minutes_below_90:.1f} minutes",
            provenance=field_provenance(TherapySummary, "total_spo2_time_below_90"),
        )

    if summary.avg_pulse is not None:
        print_subsection("Pulse")
        print_kv(
            "Average",
            f"{summary.avg_pulse:.1f} BPM",
            provenance=field_provenance(TherapySummary, "avg_pulse"),
        )

    if (
        summary.avg_respiratory_rate is not None
        or summary.avg_tidal_volume is not None
        or summary.avg_minute_ventilation is not None
    ):
        print_subsection("Respiratory")
        if summary.avg_respiratory_rate is not None:
            print_kv(
                "Respiratory Rate",
                f"{summary.avg_respiratory_rate:.1f} breaths/min",
                provenance=field_provenance(TherapySummary, "avg_respiratory_rate"),
            )
        if summary.avg_tidal_volume is not None:
            print_kv(
                "Tidal Volume",
                f"{summary.avg_tidal_volume:.0f} mL",
                provenance=field_provenance(TherapySummary, "avg_tidal_volume"),
            )
        if summary.avg_minute_ventilation is not None:
            print_kv(
                "Minute Ventilation",
                f"{summary.avg_minute_ventilation:.1f} L/min",
                provenance=field_provenance(TherapySummary, "avg_minute_ventilation"),
            )

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
                        avg_ahi_str,
                        med_ahi_str,
                    )
                )

            print_table(
                [
                    Column("Period", 20),
                    Column(
                        "Days",
                        6,
                        field_provenance(PeriodStatistics, "days_used"),
                    ),
                    Column(
                        "Avg Hours",
                        11,
                        field_provenance(PeriodStatistics, "avg_hours_per_day"),
                    ),
                    Column("Avg AHI", 9, field_provenance(PeriodStatistics, "avg_ahi")),
                    Column(
                        "Med AHI", 9, field_provenance(PeriodStatistics, "median_ahi")
                    ),
                ],
                period_rows,
            )

            print_footer(wide=True)

            if trend:
                import plotext as plt  # noqa: PLC0415

                trends = await service.get_trends(period_literal, days)
                ahi_trend = trends["ahi"]

                ahi_values = [v for _, v in ahi_trend if v is not None]
                if ahi_values:
                    dates_for_plot = [d for d, v in ahi_trend if v is not None]
                    date_labels = [d.strftime("%Y-%m-%d") for d in dates_for_plot]
                    x_indices = list(range(len(ahi_values)))

                    from snore.analysis.calculations import (  # noqa: PLC0415
                        calculate_ahi_trend_direction,
                    )

                    direction_str = calculate_ahi_trend_direction(ahi_values)
                    direction = f"({direction_str})" if direction_str else ""

                    print_header(
                        mark_provenance(
                            "AHI Trend", field_provenance(TrendsResponse, "ahi")
                        ),
                        wide=True,
                    )

                    plt.clf()
                    plt.plot(x_indices, ahi_values, marker="braille")
                    plt.xticks(x_indices, date_labels)
                    plt.title(f"AHI Over Time {direction}")
                    plt.xlabel("Period")
                    plt.ylabel("AHI (events/hour)")
                    plt.show()

                    print_footer(wide=True)

    if records:
        records_data = await service.get_records(days, top_n=5)

        if records_data:
            print_header("Records (Top 5)", wide=True)

            metric_labels = {
                "ahi": ("Best AHI", "Worst AHI"),
                "leak": ("Best Leak", "Worst Leak"),
                "therapy_hours": ("Longest Sessions", "Shortest Sessions"),
                "spo2_min": ("Best SpO2 Min", "Worst SpO2 Min"),
            }

            for metric, (best_label, worst_label) in metric_labels.items():
                if metric not in records_data:
                    continue
                metric_provenance = field_provenance(RecordsResponse, metric)

                best_records = records_data[metric]["best"]
                worst_records = records_data[metric]["worst"]

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
                    [
                        Column(best_label, 35, metric_provenance),
                        Column(worst_label, 0, metric_provenance),
                    ],
                    record_rows,
                )

            print_footer(wide=True)

    console.print()
    print_footer()
    console.print()
