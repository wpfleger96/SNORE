"""Text summary generation for OSCAR therapy data."""

from datetime import date

from snore.analysis.calculations import (
    assess_therapy_effectiveness,
    calculate_average_ahi,
    calculate_average_hours_per_day,
    calculate_compliance_rate,
    calculate_total_hours,
    is_compliant,
)
from snore.database import models
from snore.utils.formatting import (
    format_date_range,
    format_duration,
    format_leak,
    format_pressure,
    get_ahi_severity,
)


def generate_day_summary(day: models.Day) -> str:
    """
    Generate human-readable summary for a single day.

    Args:
        day: Day record

    Returns:
        Summary text
    """
    date_str = day.date.strftime("%B %d, %Y")
    summary_parts = [f"On {date_str}"]

    # Usage
    if day.total_therapy_hours:
        compliant = is_compliant(day.total_therapy_hours)
        compliance_text = (
            "with good compliance" if compliant else "(below 4-hour minimum)"
        )
        summary_parts.append(
            f", therapy was used for {format_duration(day.total_therapy_hours)} {compliance_text}."
        )
    else:
        summary_parts.append(", no therapy data was recorded.")
        return "".join(summary_parts)

    # AHI
    if day.ahi is not None:
        severity = get_ahi_severity(day.ahi)
        ahi_desc = {
            "normal": "normal range",
            "mild": "mild sleep apnea range",
            "moderate": "moderate sleep apnea range",
            "severe": "severe sleep apnea range",
        }.get(severity, "unknown range")

        summary_parts.append(
            f" The AHI was {day.ahi:.1f} events per hour ({ahi_desc})."
        )

        # Event details
        events = []
        if day.obstructive_apneas > 0:
            events.append(
                f"{day.obstructive_apneas} obstructive apnea{'s' if day.obstructive_apneas != 1 else ''}"
            )
        if day.hypopneas > 0:
            events.append(
                f"{day.hypopneas} hypopnea{'s' if day.hypopneas != 1 else ''}"
            )
        if day.central_apneas > 0:
            events.append(
                f"{day.central_apneas} central apnea{'s' if day.central_apneas != 1 else ''}"
            )

        if events:
            events_text = ", ".join(events)
            summary_parts.append(f" There were {events_text}.")

    # Pressure
    if day.pressure_median is not None:
        summary_parts.append(
            f" Median pressure was {format_pressure(day.pressure_median)}"
        )
        if day.pressure_95th is not None:
            summary_parts.append(
                f" with a 95th percentile of {format_pressure(day.pressure_95th)}."
            )
        else:
            summary_parts.append(".")

    # Leak
    if day.leak_median is not None:
        leak_assessment = "well controlled" if day.leak_median < 24 else "elevated"
        summary_parts.append(
            f" Leak rates were {leak_assessment} with a median of {format_leak(day.leak_median)}."
        )

    # SpO2
    if day.spo2_avg is not None:
        spo2_assessment = "healthy" if day.spo2_avg >= 95 else "concerning"
        summary_parts.append(
            f" Average SpO₂ was {day.spo2_avg:.1f}% ({spo2_assessment})"
        )
        if day.spo2_min is not None:
            summary_parts.append(f" with a minimum of {day.spo2_min:.0f}%.")
        else:
            summary_parts.append(".")

    # Overall assessment
    if day.ahi is not None:
        if day.ahi < 5 and is_compliant(day.total_therapy_hours):
            summary_parts.append(
                " Overall, this represents effective therapy with good adherence."
            )
        elif day.ahi < 5:
            summary_parts.append(
                " Therapy was effective but usage was below recommendations."
            )
        elif day.ahi < 15:
            summary_parts.append(
                " Therapy shows room for improvement; consider consulting with your sleep specialist."
            )
        else:
            summary_parts.append(
                " High AHI suggests therapy may need adjustment; consult with your sleep specialist."
            )

    return "".join(summary_parts)


def generate_period_summary(
    profile_name: str, days: list[models.Day], period_start: date, period_end: date
) -> str:
    """
    Generate human-readable summary for a time period.

    Args:
        profile_name: Name of the profile
        days: List of Day records in the period
        period_start: Start date
        period_end: End date

    Returns:
        Summary text
    """
    if not days:
        return f"No therapy data found for {profile_name} in the period {format_date_range(period_start, period_end)}."

    summary_parts = []

    # Header
    date_range = format_date_range(period_start, period_end)
    days_in_period = (period_end - period_start).days + 1
    summary_parts.append(
        f"{profile_name} used CPAP therapy for {len(days)} out of {days_in_period} days "
        f"in the period {date_range}"
    )

    # Compliance
    compliance_pct, compliant_days, total_days = calculate_compliance_rate(days)
    compliance_assessment = (
        "excellent"
        if compliance_pct >= 90
        else "good"
        if compliance_pct >= 70
        else "fair"
        if compliance_pct >= 50
        else "poor"
    )
    summary_parts.append(
        f", achieving {compliance_pct:.0f}% compliance "
        f"({compliant_days} days with >= 4 hours usage, {compliance_assessment})."
    )

    # Usage statistics
    total_hours = calculate_total_hours(days)
    avg_hours = calculate_average_hours_per_day(days)
    summary_parts.append(
        f" Average nightly usage was {format_duration(avg_hours)} "
        f"with a total of {format_duration(total_hours)}."
    )

    # AHI and effectiveness
    avg_ahi = calculate_average_ahi(days)
    if avg_ahi is not None:
        effectiveness = assess_therapy_effectiveness(avg_ahi)
        effectiveness_desc = {
            "excellent": "excellent therapy control",
            "good": "good therapy control",
            "fair": "fair therapy control",
            "poor": "suboptimal therapy control",
        }.get(effectiveness, "therapy control")

        summary_parts.append(
            f" The average AHI was {avg_ahi:.1f} events per hour, indicating {effectiveness_desc}."
        )

    # Pressure statistics (if available)
    pressure_values = [
        day.pressure_median for day in days if day.pressure_median is not None
    ]
    if pressure_values:
        avg_pressure = sum(pressure_values) / len(pressure_values)
        summary_parts.append(f" Pressure averaged {format_pressure(avg_pressure)}")

        # Leak statistics
        leak_values = [day.leak_median for day in days if day.leak_median is not None]
        if leak_values:
            avg_leak = sum(leak_values) / len(leak_values)
            leak_assessment = (
                "good mask seal" if avg_leak < 24 else "elevated leak rates"
            )
            summary_parts.append(
                f" with {leak_assessment} (average leak {avg_leak:.1f} L/min)."
            )
        else:
            summary_parts.append(".")

    # SpO2 statistics (if available)
    spo2_values = [day.spo2_avg for day in days if day.spo2_avg is not None]
    if spo2_values:
        avg_spo2 = sum(spo2_values) / len(spo2_values)
        spo2_mins = [day.spo2_min for day in days if day.spo2_min is not None]
        min_spo2 = min(spo2_mins) if spo2_mins else None

        spo2_assessment = (
            "healthy"
            if avg_spo2 >= 95
            else "borderline"
            if avg_spo2 >= 92
            else "concerning"
        )
        summary_parts.append(
            f" SpO₂ levels were {spo2_assessment} with an average of {avg_spo2:.1f}%"
        )

        if min_spo2 is not None:
            summary_parts.append(f" and a minimum of {min_spo2:.0f}%.")
        else:
            summary_parts.append(".")

    # Overall assessment
    if avg_ahi is not None and compliance_pct >= 70:
        if avg_ahi < 5:
            summary_parts.append(
                " This represents highly effective therapy with strong adherence."
            )
        elif avg_ahi < 10:
            summary_parts.append(
                " This represents effective therapy with good adherence. "
                "Minor optimization may further improve results."
            )
        else:
            summary_parts.append(
                " While adherence is good, therapy effectiveness could be improved. "
                "Consider discussing adjustments with your sleep specialist."
            )
    elif compliance_pct < 70:
        summary_parts.append(
            " Improving nightly usage consistency may enhance therapy benefits. "
            "Aim for at least 4 hours per night, every night."
        )

    return "".join(summary_parts)
