"""
Validation report generation and export.

Provides functionality to generate and export validation reports in JSON and CSV formats.
"""

import csv
import json

from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from snore.provenance import Provenance, provenance_field


class OverlappingSessionPair(BaseModel):
    """A pair of sessions on the same device whose time ranges strictly overlap."""

    session_a_id: int = Field(description="Database ID of the first session")
    session_b_id: int = Field(description="Database ID of the second session")
    device_id: int = Field(description="Device that owns both sessions")
    session_a_device_session_id: str = Field(
        description="device_session_id of session A"
    )
    session_b_device_session_id: str = Field(
        description="device_session_id of session B"
    )
    session_a_start: datetime = Field(description="Start time of session A")
    session_a_end: datetime = Field(description="End time of session A")
    session_b_start: datetime = Field(description="Start time of session B")
    session_b_end: datetime = Field(description="End time of session B")


class CrossParserSameDay(BaseModel):
    """A device+date combination with sessions from more than one import source."""

    device_id: int = Field(description="Device ID")
    day_date: date = Field(description="Calendar date of the Day row")
    import_sources: list[str] = Field(
        description="Distinct import_source values for sessions on this day"
    )


class IntegrityReport(BaseModel):
    """Data-integrity check results for the SNORE session database."""

    checked_at: datetime = Field(description="Timestamp when the check ran")
    device_id_filter: int | None = Field(
        default=None, description="Device ID filter applied, or null for all devices"
    )
    null_day_id_sessions: list[int] = Field(
        description="Session IDs where day_id IS NULL (not yet linked to a day)"
    )
    overlapping_session_pairs: list[OverlappingSessionPair] = Field(
        description="Pairs of same-device sessions whose time ranges strictly overlap"
    )
    cross_parser_same_day: list[CrossParserSameDay] = Field(
        description="Device+date combinations with sessions from multiple import sources"
    )
    total_issues: int = Field(description="Total count of detected integrity issues")


class SessionValidation(BaseModel):
    """Validation results for a single session."""

    session_id: int = Field(description="Database session ID")
    date: str = Field(description="Session date (YYYY-MM-DD)")
    duration_hours: float = provenance_field(
        Provenance.DERIVED, "Flow-waveform coverage (hours)"
    )
    machine_event_count: int = provenance_field(
        Provenance.DEVICE, "Total machine events"
    )
    programmatic_event_count: int = provenance_field(
        Provenance.EXPERIMENTAL, "Total programmatic events"
    )
    apnea_sensitivity: float = provenance_field(
        Provenance.EXPERIMENTAL, "Apnea sensitivity (0-1)"
    )
    apnea_precision: float = provenance_field(
        Provenance.EXPERIMENTAL, "Apnea precision (0-1)"
    )
    apnea_f1: float = provenance_field(Provenance.EXPERIMENTAL, "Apnea F1 score (0-1)")
    hypopnea_sensitivity: float = provenance_field(
        Provenance.EXPERIMENTAL, "Hypopnea sensitivity (0-1)"
    )
    hypopnea_precision: float = provenance_field(
        Provenance.EXPERIMENTAL, "Hypopnea precision (0-1)"
    )
    hypopnea_f1: float = provenance_field(
        Provenance.EXPERIMENTAL, "Hypopnea F1 score (0-1)"
    )
    notes: str | None = Field(default=None, description="Additional notes")

    # Device-reported (STR) nightly indices from the Statistics ``*_device``
    # columns and ``uai``.  Null when the Statistics row is absent or the device
    # did not record the index (e.g. APAP records ahi/oai/cai/hi but not uai;
    # vAuto records all five).
    device_ahi: float | None = provenance_field(
        Provenance.DEVICE, "Device AHI (events/hr)", default=None
    )
    device_oai: float | None = provenance_field(
        Provenance.DEVICE,
        "Device OAI (obstructive apnea index, events/hr)",
        default=None,
    )
    device_cai: float | None = provenance_field(
        Provenance.DEVICE, "Device CAI (central apnea index, events/hr)", default=None
    )
    device_hi: float | None = provenance_field(
        Provenance.DEVICE, "Device HI (hypopnea index, events/hr)", default=None
    )
    device_uai: float | None = provenance_field(
        Provenance.DEVICE,
        "Device UAI (upper-airway/unclassified apnea index, events/hr; vAuto/bilevel only)",
        default=None,
    )


class AggregateMetrics(BaseModel):
    """Aggregate validation metrics across multiple sessions."""

    total_sessions: int = Field(description="Total sessions analyzed")
    total_machine_events: int = provenance_field(
        Provenance.DEVICE, "Total machine events"
    )
    total_programmatic_events: int = provenance_field(
        Provenance.EXPERIMENTAL, "Total programmatic events"
    )
    avg_apnea_sensitivity: float = provenance_field(
        Provenance.EXPERIMENTAL, "Average apnea sensitivity"
    )
    avg_apnea_precision: float = provenance_field(
        Provenance.EXPERIMENTAL, "Average apnea precision"
    )
    avg_apnea_f1: float = provenance_field(Provenance.EXPERIMENTAL, "Average apnea F1")
    avg_hypopnea_sensitivity: float = provenance_field(
        Provenance.EXPERIMENTAL, "Average hypopnea sensitivity"
    )
    avg_hypopnea_precision: float = provenance_field(
        Provenance.EXPERIMENTAL, "Average hypopnea precision"
    )
    avg_hypopnea_f1: float = provenance_field(
        Provenance.EXPERIMENTAL, "Average hypopnea F1"
    )
    low_sensitivity_sessions: list[int] = Field(
        description="Session IDs with <60% sensitivity"
    )


class ValidationReport(BaseModel):
    """Complete validation report."""

    report_date: str = Field(description="Report generation date")
    date_range_start: str = Field(description="Start date of analyzed sessions")
    date_range_end: str = Field(description="End date of analyzed sessions")
    aggregate: AggregateMetrics = Field(description="Aggregate metrics")
    sessions: list[SessionValidation] = Field(description="Per-session results")


def export_report_json(report: ValidationReport, output_path: Path) -> None:
    """
    Export validation report as JSON.

    Args:
        report: Validation report to export
        output_path: Path to output JSON file
    """
    with open(output_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)


def export_report_csv(report: ValidationReport, output_path: Path) -> None:
    """
    Export validation report as CSV.

    Args:
        report: Validation report to export
        output_path: Path to output CSV file
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "date",
                "duration_hours",
                "machine_events",
                "programmatic_events",
                "apnea_sens",
                "apnea_prec",
                "apnea_f1",
                "hypopnea_sens",
                "hypopnea_prec",
                "hypopnea_f1",
                "device_ahi",
                "device_oai",
                "device_cai",
                "device_hi",
                "device_uai",
                "notes",
            ],
        )
        writer.writeheader()

        for session in report.sessions:
            writer.writerow(
                {
                    "session_id": session.session_id,
                    "date": session.date,
                    "duration_hours": f"{session.duration_hours:.1f}",
                    "machine_events": session.machine_event_count,
                    "programmatic_events": session.programmatic_event_count,
                    "apnea_sens": f"{session.apnea_sensitivity * 100:.0f}%",
                    "apnea_prec": f"{session.apnea_precision * 100:.0f}%",
                    "apnea_f1": f"{session.apnea_f1:.2f}",
                    "hypopnea_sens": f"{session.hypopnea_sensitivity * 100:.0f}%",
                    "hypopnea_prec": f"{session.hypopnea_precision * 100:.0f}%",
                    "hypopnea_f1": f"{session.hypopnea_f1:.2f}",
                    "device_ahi": f"{session.device_ahi:.2f}"
                    if session.device_ahi is not None
                    else "",
                    "device_oai": f"{session.device_oai:.2f}"
                    if session.device_oai is not None
                    else "",
                    "device_cai": f"{session.device_cai:.2f}"
                    if session.device_cai is not None
                    else "",
                    "device_hi": f"{session.device_hi:.2f}"
                    if session.device_hi is not None
                    else "",
                    "device_uai": f"{session.device_uai:.2f}"
                    if session.device_uai is not None
                    else "",
                    "notes": session.notes or "",
                }
            )
