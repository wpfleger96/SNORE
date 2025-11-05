"""
OSCAR-MCP Server

MCP server providing tools for analyzing and inspecting OSCAR CPAP/APAP therapy data.
"""

import json
import logging
from datetime import date, timedelta
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from oscar_mcp.constants import (
    CHANNEL_DEFINITIONS,
    COMPLIANCE_MIN_HOURS,
)
from oscar_mcp.database.session import session_scope
from oscar_mcp.database import models
from oscar_mcp.models.profile import ProfileSummary
from oscar_mcp.models.machine import MachineSummary
from oscar_mcp.models.day import DayTextReport
from oscar_mcp.models.statistics import TherapySummary, ComplianceReport
from oscar_mcp.utils.validation import (
    validate_profile_exists,
    validate_date_format,
    validate_date_range,
)
from oscar_mcp.analysis.summaries import generate_day_summary, generate_period_summary
from oscar_mcp.analysis.calculations import (
    calculate_compliance_rate,
    calculate_average_ahi,
    assess_therapy_effectiveness,
    calculate_total_hours,
    calculate_average_hours_per_day,
    get_date_range,
)

logger = logging.getLogger(__name__)

INSTRUCTIONS = """
OSCAR-MCP v{version('oscar-mcp')}

You are the OSCAR-MCP server. You provide access to CPAP/BiPAP therapy data from the OSCAR
(Open Source CPAP Analysis Reporter) application.

IMPORTANT NOTES:
- Before querying data, use list_profiles to see available profiles
- All dates should be in YYYY-MM-DD format
- The server provides human-readable text summaries optimized for LLM analysis
- Data must first be imported using the 'oscar-import' CLI tool

AVAILABLE TOOLS:
- list_profiles: List all available profiles in the database
- get_therapy_summary: Get comprehensive therapy summary (main tool for reports)
- get_day_report: Get detailed report for a specific day
- get_compliance: Get compliance report for a date range
- list_machines: List devices for a profile

WORKFLOW:
1. Use list_profiles to find available profiles
2. Use get_therapy_summary for overall analysis
3. Use get_day_report for specific date analysis
4. Use get_compliance for compliance reports

CHANNEL INFORMATION:
OSCAR tracks various data channels including:
- Pressure (CPAP/IPAP/EPAP)
- Events (Apneas, Hypopneas, RERAs, Flow Limitations)
- Waveforms (Flow Rate, Leak Rate, Respiratory Rate, Tidal Volume)
- Statistics (AHI, RDI)
- Oximetry (SpO₂, Pulse Rate) if available

CLINICAL CONTEXT:
- AHI (Apnea-Hypopnea Index): <5 normal, 5-15 mild, 15-30 moderate, >30 severe
- Compliance: >= 4 hours per night
- Leak rates: <24 L/min acceptable
- SpO₂: >= 95% normal
"""

server = FastMCP(name="oscar-mcp", instructions=INSTRUCTIONS)


# ============================================================================
# Resources (Documentation)
# ============================================================================


@server.resource("docs://channels")
def get_channels_documentation() -> str:
    """Documentation of available OSCAR data channels."""
    channels_info = {"description": "OSCAR data channels and their properties", "channels": []}

    for channel_id, definition in CHANNEL_DEFINITIONS.items():
        channels_info["channels"].append(
            {
                "id": channel_id,
                "code": definition.code,
                "name": definition.name,
                "description": definition.description,
                "type": definition.channel_type.value,
                "unit": definition.unit,
                "calculations": [calc.value for calc in definition.calculations],
            }
        )

    return json.dumps(channels_info, indent=2)


@server.resource("docs://ahi_severity")
def get_ahi_severity_info() -> str:
    """Documentation of AHI severity classifications."""
    return json.dumps(
        {
            "description": "AHI (Apnea-Hypopnea Index) severity classifications",
            "unit": "events per hour",
            "classifications": {
                "normal": "< 5 events/hr",
                "mild": "5-15 events/hr",
                "moderate": "15-30 events/hr",
                "severe": "> 30 events/hr",
            },
            "note": "AHI is the primary metric for assessing sleep apnea severity",
        },
        indent=2,
    )


@server.resource("docs://compliance")
def get_compliance_info() -> str:
    """Documentation of compliance requirements."""
    return json.dumps(
        {
            "description": "CPAP therapy compliance requirements",
            "minimum_hours": COMPLIANCE_MIN_HOURS,
            "insurance_requirement": "Typically >= 4 hours per night for >= 70% of nights",
            "recommendation": "Use therapy every night for maximum benefit",
            "note": "Compliance requirements may vary by insurance provider and region",
        },
        indent=2,
    )


# ============================================================================
# Tools (Actions)
# ============================================================================


@server.tool("list_profiles")
def list_profiles() -> List[ProfileSummary]:
    """
    List all available profiles in the database.

    Returns:
        List of profile summaries
    """
    try:
        with session_scope() as session:
            profiles = session.query(models.Profile).all()

            result = []
            for profile in profiles:
                # Calculate summary information
                machine_count = len(profile.machines)
                total_days = len(profile.days)

                date_range = get_date_range(profile.days)
                date_range_start = date_range[0] if date_range else None
                date_range_end = date_range[1] if date_range else None

                result.append(
                    ProfileSummary(
                        id=profile.id,
                        name=profile.username,  # Map username to name for API compatibility
                        first_name=profile.first_name,
                        last_name=profile.last_name,
                        date_of_birth=profile.date_of_birth,
                        height_cm=profile.height_cm,
                        created_at=profile.created_at,
                        updated_at=profile.updated_at,
                        machine_count=machine_count,
                        total_days=total_days,
                        date_range_start=date_range_start,
                        date_range_end=date_range_end,
                    )
                )

            return result

    except Exception as e:
        logger.error(f"Error listing profiles: {e}", exc_info=True)
        raise ValueError(f"Error listing profiles: {e}")


@server.tool("get_therapy_summary")
def get_therapy_summary(
    *, profile_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> TherapySummary:
    """
    Get comprehensive therapy summary with human-readable text.

    This is the main tool for generating therapy reports.

    Args:
        profile_name: Name of the profile
        start_date: Start date (YYYY-MM-DD), defaults to 30 days ago
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        Therapy summary with detailed text report
    """
    try:
        # Validate profile
        validate_profile_exists(profile_name)

        # Parse dates
        if end_date is None:
            end_dt = date.today()
        else:
            end_dt = validate_date_format(end_date)

        if start_date is None:
            start_dt = end_dt - timedelta(days=30)
        else:
            start_dt = validate_date_format(start_date)

        validate_date_range(start_dt, end_dt)

        # Query days in range
        with session_scope() as session:
            profile = session.query(models.Profile).filter_by(username=profile_name).first()

            days = (
                session.query(models.Day)
                .filter(
                    models.Day.profile_id == profile.id,
                    models.Day.date >= start_dt,
                    models.Day.date <= end_dt,
                )
                .all()
            )

            # Generate summary
            summary_text = generate_period_summary(profile_name, days, start_dt, end_dt)

            return TherapySummary(
                profile_name=profile_name,
                period_start=start_dt,
                period_end=end_dt,
                summary=summary_text,
            )

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating therapy summary: {e}", exc_info=True)
        raise ValueError(f"Error generating therapy summary: {e}")


@server.tool("get_day_report")
def get_day_report(*, profile_name: str, date_str: str) -> DayTextReport:
    """
    Get detailed report for a specific day.

    Args:
        profile_name: Name of the profile
        date_str: Date (YYYY-MM-DD)

    Returns:
        Day report with human-readable summary
    """
    try:
        # Validate
        validate_profile_exists(profile_name)
        query_date = validate_date_format(date_str)

        # Query day
        with session_scope() as session:
            profile = session.query(models.Profile).filter_by(username=profile_name).first()

            day = (
                session.query(models.Day)
                .filter(models.Day.profile_id == profile.id, models.Day.date == query_date)
                .first()
            )

            if not day:
                return DayTextReport(
                    date=query_date,
                    summary=f"No therapy data found for {profile_name} on {query_date.strftime('%B %d, %Y')}.",
                )

            # Generate summary
            summary_text = generate_day_summary(day)

            return DayTextReport(date=query_date, summary=summary_text)

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating day report: {e}", exc_info=True)
        raise ValueError(f"Error generating day report: {e}")


@server.tool("get_compliance")
def get_compliance_report(*, profile_name: str, start_date: str, end_date: str) -> ComplianceReport:
    """
    Get compliance report for a date range.

    Args:
        profile_name: Name of the profile
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Compliance report with statistics
    """
    try:
        # Validate
        validate_profile_exists(profile_name)
        start_dt = validate_date_format(start_date)
        end_dt = validate_date_format(end_date)
        validate_date_range(start_dt, end_dt)

        # Query days
        with session_scope() as session:
            profile = session.query(models.Profile).filter_by(username=profile_name).first()

            days = (
                session.query(models.Day)
                .filter(
                    models.Day.profile_id == profile.id,
                    models.Day.date >= start_dt,
                    models.Day.date <= end_dt,
                )
                .all()
            )

            # Calculate compliance
            days_in_period = (end_dt - start_dt).days + 1
            days_used = len(days)

            compliance_pct, days_compliant, _ = calculate_compliance_rate(days)
            total_hours = calculate_total_hours(days)
            avg_hours = calculate_average_hours_per_day(days) if days_used > 0 else 0.0
            avg_ahi = calculate_average_ahi(days)
            effectiveness = assess_therapy_effectiveness(avg_ahi)

            return ComplianceReport(
                period_start=start_dt,
                period_end=end_dt,
                days_in_period=days_in_period,
                days_used=days_used,
                days_compliant=days_compliant,
                compliance_percentage=compliance_pct,
                total_hours=total_hours,
                avg_hours_per_night=avg_hours,
                avg_ahi=avg_ahi,
                therapy_effectiveness=effectiveness,
            )

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}", exc_info=True)
        raise ValueError(f"Error generating compliance report: {e}")


@server.tool("list_machines")
def list_machines(*, profile_name: str) -> List[MachineSummary]:
    """
    List devices for a profile.

    Args:
        profile_name: Name of the profile

    Returns:
        List of machine summaries
    """
    try:
        # Validate
        validate_profile_exists(profile_name)

        with session_scope() as session:
            profile = session.query(models.Profile).filter_by(username=profile_name).first()

            devices = session.query(models.Device).filter_by(profile_id=profile.id).all()

            result = []
            for device in devices:
                session_count = len(device.sessions)
                total_hours = sum(
                    s.duration_seconds / 3600 for s in device.sessions if s.duration_seconds
                )

                # Construct machine_id from device info
                machine_id = f"{device.manufacturer}_{device.model}_{device.serial_number}"

                # Determine machine type (basic heuristic, can be enhanced)
                machine_type = "CPAP"  # Default
                if device.model:
                    model_lower = device.model.lower()
                    if "auto" in model_lower or "apap" in model_lower:
                        machine_type = "AutoCPAP"
                    elif "bipap" in model_lower or "vpap" in model_lower:
                        machine_type = "BiPAP"
                    elif "oximeter" in model_lower or "pulse" in model_lower:
                        machine_type = "Oximeter"

                result.append(
                    MachineSummary(
                        id=device.id,
                        machine_id=machine_id,
                        serial_number=device.serial_number,
                        brand=device.manufacturer,
                        model=device.model,
                        machine_type=machine_type,
                        created_at=device.first_seen,
                        last_import=device.last_import,
                        session_count=session_count,
                        total_hours=total_hours,
                    )
                )

            return result

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"Error listing machines: {e}", exc_info=True)
        raise ValueError(f"Error listing machines: {e}")
