"""
SNORE Server

MCP server providing tools for analyzing and inspecting OSCAR CPAP/APAP therapy data.
"""

import json
import logging

from datetime import date, datetime, timedelta
from typing import Any

from mcp.server.fastmcp import FastMCP

from snore.analysis.calculations import (
    assess_therapy_effectiveness,
    calculate_average_ahi,
    calculate_average_hours_per_day,
    calculate_compliance_rate,
    calculate_total_hours,
    get_date_range,
)
from snore.analysis.service import AnalysisService
from snore.analysis.summaries import generate_day_summary, generate_period_summary
from snore.constants import (
    CHANNEL_DEFINITIONS,
    COMPLIANCE_MIN_HOURS,
)
from snore.database import models
from snore.database.session import session_scope
from snore.models.analysis import (
    AnalysisSummary,
    DetailedAnalysisResult,
    FlowLimitationSummary,
    SessionAnalysisStatus,
)
from snore.models.day import DayTextReport
from snore.models.machine import MachineSummary
from snore.models.profile import ProfileSummary
from snore.models.statistics import ComplianceReport, TherapySummary
from snore.utils.validation import (
    validate_date_format,
    validate_date_range,
    validate_profile_exists,
)

logger = logging.getLogger(__name__)

INSTRUCTIONS = """
SNORE v{version('snore')}
Sleep eNvironment Observation & Respiratory Evaluation

You are the SNORE server. You provide access to CPAP/BiPAP therapy data from the OSCAR
(Open Source CPAP Analysis Reporter) application.

IMPORTANT NOTES:
- Before querying data, use list_profiles to see available profiles
- All dates should be in YYYY-MM-DD format
- The server provides human-readable text summaries optimized for LLM analysis
- Data must first be imported using the 'snore-import' CLI tool

AVAILABLE TOOLS:
- list_profiles: List all available profiles in the database
- get_therapy_summary: Get comprehensive therapy summary (main tool for reports)
- get_day_report: Get detailed report for a specific day
- get_compliance: Get compliance report for a date range
- list_machines: List devices for a profile
- analyze_session: Run comprehensive programmatic analysis on a session
- get_analysis_results: Retrieve detailed analysis results
- list_analysis_sessions: List sessions with their analysis status

WORKFLOW:
1. Use list_profiles to find available profiles
2. Use get_therapy_summary for overall analysis
3. Use get_day_report for specific date analysis
4. Use analyze_session to detect apneas, hypopneas, flow limitation, and patterns
5. Use get_analysis_results to view detailed analysis findings

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

server = FastMCP(name="snore", instructions=INSTRUCTIONS)


# ============================================================================
# Resources (Documentation)
# ============================================================================


@server.resource("docs://channels")
def get_channels_documentation() -> str:
    """Documentation of available OSCAR data channels."""
    channels_info: dict[str, Any] = {
        "description": "OSCAR data channels and their properties",
        "channels": [],
    }

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
def list_profiles() -> list[ProfileSummary]:
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
                machine_count = len(profile.devices)
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
        raise ValueError(f"Error listing profiles: {e}") from e


@server.tool("get_therapy_summary")
def get_therapy_summary(
    *, profile_name: str, start_date: str | None = None, end_date: str | None = None
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
            profile = (
                session.query(models.Profile).filter_by(username=profile_name).first()
            )
            if profile is None:
                raise ValueError(f"Profile '{profile_name}' not found")

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
        raise ValueError(f"Error generating therapy summary: {e}") from e


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
            profile = (
                session.query(models.Profile).filter_by(username=profile_name).first()
            )
            if profile is None:
                raise ValueError(f"Profile '{profile_name}' not found")

            day = (
                session.query(models.Day)
                .filter(
                    models.Day.profile_id == profile.id, models.Day.date == query_date
                )
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
        raise ValueError(f"Error generating day report: {e}") from e


@server.tool("get_compliance")
def get_compliance_report(
    *, profile_name: str, start_date: str, end_date: str
) -> ComplianceReport:
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
            profile = (
                session.query(models.Profile).filter_by(username=profile_name).first()
            )
            if profile is None:
                raise ValueError(f"Profile '{profile_name}' not found")

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
        raise ValueError(f"Error generating compliance report: {e}") from e


@server.tool("list_machines")
def list_machines(*, profile_name: str) -> list[MachineSummary]:
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
            profile = (
                session.query(models.Profile).filter_by(username=profile_name).first()
            )
            if profile is None:
                raise ValueError(f"Profile '{profile_name}' not found")

            devices = (
                session.query(models.Device).filter_by(profile_id=profile.id).all()
            )

            result = []
            for device in devices:
                session_count = len(device.sessions)
                total_hours = sum(
                    s.duration_seconds / 3600
                    for s in device.sessions
                    if s.duration_seconds
                )

                # Construct machine_id from device info
                machine_id = (
                    f"{device.manufacturer}_{device.model}_{device.serial_number}"
                )

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
        raise ValueError(f"Error listing machines: {e}") from e


@server.tool("analyze_session")
def analyze_session(
    profile_name: str, session_date: str | None = None, session_id: int | None = None
) -> AnalysisSummary:
    """
    Run comprehensive programmatic analysis on a CPAP session.

    This tool performs breath-by-breath analysis to detect:
    - Flow limitation patterns (7-class system)
    - Respiratory events (apneas, hypopneas, RERAs)
    - Complex patterns (Cheyne-Stokes Respiration, periodic breathing)
    - Clinical indices (AHI, RDI, Flow Limitation Index)

    Args:
        profile_name: Profile username
        session_date: Session date in YYYY-MM-DD format (use this OR session_id)
        session_id: Database session ID (use this OR session_date)

    Returns:
        AnalysisSummary with key metrics and findings
    """
    try:
        validate_profile_exists(profile_name)

        if session_date:
            validate_date_format(session_date)

        if not session_date and not session_id:
            raise ValueError("Must provide either session_date or session_id")

        with session_scope() as session:
            profile = (
                session.query(models.Profile).filter_by(username=profile_name).first()
            )
            if profile is None:
                raise ValueError(f"Profile '{profile_name}' not found")

            if session_date:
                parsed_date = date.fromisoformat(session_date)
                db_session = (
                    session.query(models.Session)
                    .join(models.Day)
                    .filter(
                        models.Day.profile_id == profile.id,
                        models.Day.date == parsed_date,
                    )
                    .first()
                )

                if not db_session:
                    raise ValueError(f"No session found for {session_date}")

                session_id = db_session.id

            # session_id is guaranteed to be set (either from date lookup or parameter)
            assert session_id is not None, "session_id should not be None"

            analysis_service = AnalysisService(session)
            result = analysis_service.analyze_session(
                session_id=session_id, store_results=True
            )

            flow_analysis = result.flow_analysis
            event_timeline = result.event_timeline
            fli = flow_analysis["fl_index"]

            if fli < 0.2:
                severity = "minimal"
            elif fli < 0.4:
                severity = "mild"
            elif fli < 0.6:
                severity = "moderate"
            else:
                severity = "severe"

            if event_timeline["ahi"] < 5:
                if severity == "minimal":
                    overall_severity = "normal"
                else:
                    overall_severity = severity
            elif event_timeline["ahi"] < 15:
                overall_severity = "mild"
            elif event_timeline["ahi"] < 30:
                overall_severity = "moderate"
            else:
                overall_severity = "severe"

            stored_result = analysis_service.get_analysis_result(session_id)
            analysis_id = stored_result["analysis_id"] if stored_result else None

            summary = AnalysisSummary(
                session_id=result.session_id,
                analysis_id=analysis_id,
                timestamp_start=datetime.fromtimestamp(result.timestamp_start),
                timestamp_end=datetime.fromtimestamp(result.timestamp_end),
                duration_hours=result.duration_hours,
                ahi=event_timeline["ahi"],
                rdi=event_timeline["rdi"],
                flow_limitation_index=fli,
                total_breaths=result.total_breaths,
                total_events=event_timeline["total_events"],
                apnea_count=len(event_timeline["apneas"]),
                hypopnea_count=len(event_timeline["hypopneas"]),
                rera_count=len(event_timeline["reras"]),
                csr_detected=result.csr_detection is not None,
                periodic_breathing_detected=result.periodic_breathing is not None,
                positional_events_detected=result.positional_analysis is not None,
                severity_assessment=overall_severity,
                processing_time_ms=int(result.processing_time_ms),
            )

            return summary

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"Error analyzing session: {e}", exc_info=True)
        raise ValueError(f"Error analyzing session: {e}") from e


@server.tool("get_analysis_results")
def get_analysis_results(session_id: int) -> DetailedAnalysisResult:
    """
    Retrieve detailed analysis results for a session.

    Returns comprehensive analysis including all detected events, patterns,
    breath-by-breath classifications, and clinical summaries.

    Args:
        session_id: Database session ID

    Returns:
        DetailedAnalysisResult with complete analysis data
    """
    try:
        with session_scope() as session:
            analysis_service = AnalysisService(session)
            stored = analysis_service.get_analysis_result(session_id)

            if not stored:
                raise ValueError(
                    f"No analysis found for session {session_id}. Run analyze_session first."
                )

            prog_result = stored["programmatic_result"]

            return DetailedAnalysisResult(
                summary=AnalysisSummary(
                    session_id=session_id,
                    analysis_id=stored["analysis_id"],
                    timestamp_start=datetime.fromisoformat(stored["timestamp_start"]),
                    timestamp_end=datetime.fromisoformat(stored["timestamp_end"]),
                    duration_hours=(
                        datetime.fromisoformat(stored["timestamp_end"])
                        - datetime.fromisoformat(stored["timestamp_start"])
                    ).total_seconds()
                    / 3600,
                    ahi=prog_result["event_timeline"]["ahi"],
                    rdi=prog_result["event_timeline"]["rdi"],
                    flow_limitation_index=prog_result["flow_analysis"]["fl_index"],
                    total_breaths=prog_result["total_breaths"],
                    total_events=prog_result["event_timeline"]["total_events"],
                    apnea_count=len(prog_result["event_timeline"]["apneas"]),
                    hypopnea_count=len(prog_result["event_timeline"]["hypopneas"]),
                    rera_count=len(prog_result["event_timeline"]["reras"]),
                    csr_detected=prog_result["csr_detection"] is not None,
                    periodic_breathing_detected=prog_result["periodic_breathing"]
                    is not None,
                    positional_events_detected=prog_result["positional_analysis"]
                    is not None,
                    severity_assessment="",
                    processing_time_ms=stored["processing_time_ms"],
                ),
                event_timeline=prog_result["event_timeline"],
                flow_limitation=FlowLimitationSummary(
                    flow_limitation_index=prog_result["flow_analysis"]["fl_index"],
                    total_breaths=prog_result["flow_analysis"]["total_breaths"],
                    class_distribution=prog_result["flow_analysis"][
                        "class_distribution"
                    ],
                    average_confidence=prog_result["flow_analysis"][
                        "average_confidence"
                    ],
                    severity="",
                ),
                csr_detection=prog_result["csr_detection"],
                periodic_breathing=prog_result["periodic_breathing"],
                positional_analysis=prog_result["positional_analysis"],
                confidence_scores=prog_result["confidence_summary"],
                clinical_summary=prog_result["clinical_summary"],
            )

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving analysis results: {e}", exc_info=True)
        raise ValueError(f"Error retrieving analysis results: {e}") from e


@server.tool("list_analysis_sessions")
def list_analysis_sessions(
    profile_name: str, start_date: str | None = None, end_date: str | None = None
) -> list[SessionAnalysisStatus]:
    """
    List sessions with their analysis status.

    Shows which sessions have been analyzed and which are available for analysis.

    Args:
        profile_name: Profile username
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        List of sessions with analysis status
    """
    try:
        validate_profile_exists(profile_name)

        if start_date:
            validate_date_format(start_date)
        if end_date:
            validate_date_format(end_date)

        with session_scope() as session:
            profile = (
                session.query(models.Profile).filter_by(username=profile_name).first()
            )
            if profile is None:
                raise ValueError(f"Profile '{profile_name}' not found")

            query = (
                session.query(models.Session)
                .join(models.Day)
                .filter(models.Day.profile_id == profile.id)
            )

            if start_date:
                query = query.filter(models.Day.date >= date.fromisoformat(start_date))
            if end_date:
                query = query.filter(models.Day.date <= date.fromisoformat(end_date))

            sessions = query.order_by(models.Day.date.desc()).all()

            result = []
            for db_session in sessions:
                analysis = (
                    session.query(models.AnalysisResult)
                    .filter_by(session_id=db_session.id)
                    .order_by(models.AnalysisResult.created_at.desc())
                    .first()
                )

                duration_hours = (
                    db_session.duration_seconds / 3600
                    if db_session.duration_seconds
                    else 0.0
                )

                result.append(
                    SessionAnalysisStatus(
                        session_id=db_session.id,
                        session_date=db_session.start_time,
                        duration_hours=duration_hours,
                        has_analysis=analysis is not None,
                        analysis_id=analysis.id if analysis else None,
                        analyzed_at=analysis.created_at if analysis else None,
                    )
                )

            return result

    except ValueError as e:
        raise e
    except Exception as e:
        logger.error(f"Error listing analysis sessions: {e}", exc_info=True)
        raise ValueError(f"Error listing analysis sessions: {e}") from e
