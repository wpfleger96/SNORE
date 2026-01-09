"""
Validation report generation and export.

Provides functionality to generate and export validation reports in JSON and CSV formats.
"""

import csv
import json

from pathlib import Path

from pydantic import BaseModel, Field


class SessionValidation(BaseModel):
    """Validation results for a single session."""

    session_id: int = Field(description="Database session ID")
    date: str = Field(description="Session date (YYYY-MM-DD)")
    duration_hours: float = Field(description="Session duration in hours")
    machine_event_count: int = Field(description="Total machine events")
    programmatic_event_count: int = Field(description="Total programmatic events")
    apnea_sensitivity: float = Field(description="Apnea sensitivity (0-1)")
    apnea_precision: float = Field(description="Apnea precision (0-1)")
    apnea_f1: float = Field(description="Apnea F1 score (0-1)")
    hypopnea_sensitivity: float = Field(description="Hypopnea sensitivity (0-1)")
    hypopnea_precision: float = Field(description="Hypopnea precision (0-1)")
    hypopnea_f1: float = Field(description="Hypopnea F1 score (0-1)")
    notes: str | None = Field(default=None, description="Additional notes")


class AggregateMetrics(BaseModel):
    """Aggregate validation metrics across multiple sessions."""

    total_sessions: int = Field(description="Total sessions analyzed")
    total_machine_events: int = Field(description="Total machine events")
    total_programmatic_events: int = Field(description="Total programmatic events")
    avg_apnea_sensitivity: float = Field(description="Average apnea sensitivity")
    avg_apnea_precision: float = Field(description="Average apnea precision")
    avg_apnea_f1: float = Field(description="Average apnea F1")
    avg_hypopnea_sensitivity: float = Field(description="Average hypopnea sensitivity")
    avg_hypopnea_precision: float = Field(description="Average hypopnea precision")
    avg_hypopnea_f1: float = Field(description="Average hypopnea F1")
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
                    "notes": session.notes or "",
                }
            )
