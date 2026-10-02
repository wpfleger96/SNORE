"""
RERA validation report models and exporters.

Event-level comparison of SNORE's two *programmatic* RERA definitions against
the ResMed device's machine-flagged RE events (``models.Event`` of type RERA,
surfaced session-relative via ``convert_machine_reras``):

1. **amplitude** — the analysis-time amplitude-crescendo detector
   (``detector.py::_detect_reras``), read from stored analysis JSON
   (``mode_result.reras``); feeds ``ModeResult.rdi``.
2. **proxy** — the query-time FL-run proxy: runs of >= 2 consecutive breaths
   with ``flow_class >= 4`` ending in a recovery breath, recomputed from stored
   breath rows; feeds the nightly ``rera_count``/``rera_index``.

The two definitions disagree by construction.  ResMed flags RE very
conservatively (essentially only in the APAP era), so most sessions carry zero
machine RE and are reported as skipped (``no_machine_re_events``) — excluded
from the sensitivity/precision aggregates but still counted.  Near-zero
precision is the *expected* result: the aggregate carries a chance-precision
floor (pooled machine-RE rate x 2 x tolerance) so those scores read as context,
not breakage.
"""

from __future__ import annotations

import csv
import json

from pathlib import Path

from pydantic import BaseModel, Field

from snore.provenance import Provenance, provenance_field


class ReraSessionValidation(BaseModel):
    """RERA validation results for a single session (possibly skipped)."""

    session_id: int = Field(description="Database session ID")
    date: str = Field(description="Session date (YYYY-MM-DD)")
    duration_hours: float = provenance_field(
        Provenance.DEVICE, "Session duration in hours"
    )
    skipped_reason: str | None = Field(
        default=None,
        description=(
            "Why this session was excluded from the sensitivity/precision "
            "aggregates. Possible values: "
            "'no_machine_re_events' — the device flagged zero RE events (the "
            "dominant case; counts/densities are still reported); "
            "'no_analysis' — no completed (OK-status) analysis result; "
            "'no_valid_breaths' — analysis present but no stored breath rows; "
            "'error' — unhandled exception during session validation; "
            "None — session was fully scored against machine RE."
        ),
    )
    machine_re_count: int = provenance_field(
        Provenance.DEVICE,
        "Machine-flagged RE (RERA) events for this session",
        default=0,
    )
    amplitude_rera_count: int = provenance_field(
        Provenance.EXPERIMENTAL,
        "Amplitude-detector RERAs (mode_result.reras)",
        default=0,
    )
    proxy_rera_count: int = provenance_field(
        Provenance.EXPERIMENTAL,
        "FL-run proxy RERAs recomputed from stored breaths",
        default=0,
    )
    amplitude_matched: int | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Amplitude RERAs matched to a machine RE event (scored sessions "
        "only; null when the session was skipped)",
        default=None,
    )
    proxy_matched: int | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "FL-run-proxy RERAs matched to a machine RE event (scored sessions "
        "only; null when the session was skipped)",
        default=None,
    )
    machine_re_density: float | None = provenance_field(
        Provenance.DERIVED, "Machine RE events per therapy hour", default=None
    )
    machine_re_density_reason: str | None = Field(
        default=None, description="Why machine_re_density is null"
    )
    amplitude_sensitivity: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Amplitude-RERA recall vs machine RE (matched / machine RE)",
        default=None,
    )
    amplitude_sensitivity_reason: str | None = Field(
        default=None, description="Why amplitude_sensitivity is null"
    )
    amplitude_precision: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Amplitude-RERA precision vs machine RE (matched / amplitude RERAs)",
        default=None,
    )
    amplitude_precision_reason: str | None = Field(
        default=None, description="Why amplitude_precision is null"
    )
    amplitude_f1: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Amplitude-RERA F1 vs machine RE", default=None
    )
    amplitude_f1_reason: str | None = Field(
        default=None, description="Why amplitude_f1 is null"
    )
    amplitude_density: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Amplitude RERAs per therapy hour", default=None
    )
    amplitude_density_reason: str | None = Field(
        default=None, description="Why amplitude_density is null"
    )
    proxy_sensitivity: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "FL-run-proxy recall vs machine RE (matched / machine RE)",
        default=None,
    )
    proxy_sensitivity_reason: str | None = Field(
        default=None, description="Why proxy_sensitivity is null"
    )
    proxy_precision: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "FL-run-proxy precision vs machine RE (matched / proxy RERAs)",
        default=None,
    )
    proxy_precision_reason: str | None = Field(
        default=None, description="Why proxy_precision is null"
    )
    proxy_f1: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "FL-run-proxy F1 vs machine RE", default=None
    )
    proxy_f1_reason: str | None = Field(
        default=None, description="Why proxy_f1 is null"
    )
    proxy_density: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "FL-run-proxy RERAs per therapy hour", default=None
    )
    proxy_density_reason: str | None = Field(
        default=None, description="Why proxy_density is null"
    )


class ReraAggregateMetrics(BaseModel):
    """Aggregate RERA validation metrics across a date range."""

    total_sessions: int = Field(description="Sessions in the requested range")
    sessions_with_machine_re: int = Field(
        description="Sessions the device flagged >= 1 RE — the scored population"
    )
    sessions_skipped_no_machine_re: int = Field(
        description="Sessions skipped: device flagged zero RE (dominant case)"
    )
    sessions_skipped_no_analysis: int = Field(
        description="Sessions skipped: no completed analysis result"
    )
    sessions_skipped_no_valid_breaths: int = Field(
        description="Sessions skipped: analysis present but no stored breaths"
    )
    sessions_skipped_error: int = Field(
        description="Sessions skipped: unhandled error during validation"
    )
    total_machine_re: int = provenance_field(
        Provenance.DEVICE, "Total machine RE across all sessions"
    )
    total_amplitude_reras: int = provenance_field(
        Provenance.EXPERIMENTAL, "Total amplitude-detector RERAs"
    )
    total_proxy_reras: int = provenance_field(
        Provenance.EXPERIMENTAL, "Total FL-run-proxy RERAs"
    )
    machine_re_density: float | None = provenance_field(
        Provenance.DERIVED, "Pooled machine RE per therapy hour", default=None
    )
    amplitude_density: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Pooled amplitude RERAs per therapy hour", default=None
    )
    proxy_density: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Pooled FL-run-proxy RERAs per therapy hour",
        default=None,
    )
    match_tolerance_seconds: float = Field(
        description="Start-time tolerance used for machine-RE matching"
    )
    chance_precision_floor: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Whole-dataset chance-precision floor (density context): machine RE "
        "per SECOND over ALL evaluated therapy hours (scored + no-machine-RE "
        "sessions) x (2 x match_tolerance_seconds). Most hours carry zero RE, "
        "so this reads against the pooled densities. For the scored-session "
        "scores below use `scored_chance_precision_floor` instead. Null when "
        "no evaluated therapy hours exist.",
        default=None,
    )
    scored_chance_precision_floor: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Scored-population chance-precision floor: machine RE per SECOND over "
        "scored-session hours only x (2 x match_tolerance_seconds). Scored "
        "sessions carry far more RE than the dataset average, so this is the "
        "honest baseline to compare the precision/sensitivity below against. "
        "Null when no scored therapy hours exist.",
        default=None,
    )
    mean_amplitude_sensitivity: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean amplitude sensitivity over scored sessions",
        default=None,
    )
    mean_amplitude_precision: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean amplitude precision over scored sessions",
        default=None,
    )
    mean_amplitude_f1: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Mean amplitude F1 over scored sessions", default=None
    )
    mean_proxy_sensitivity: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean FL-run-proxy sensitivity over scored sessions",
        default=None,
    )
    mean_proxy_precision: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean FL-run-proxy precision over scored sessions",
        default=None,
    )
    mean_proxy_f1: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean FL-run-proxy F1 over scored sessions",
        default=None,
    )
    pooled_amplitude_sensitivity: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Pooled amplitude sensitivity over scored sessions "
        "(total matched / total machine RE), not a per-session mean",
        default=None,
    )
    pooled_amplitude_precision: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Pooled amplitude precision over scored sessions "
        "(total matched / total amplitude RERAs), not a per-session mean",
        default=None,
    )
    pooled_proxy_sensitivity: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Pooled FL-run-proxy sensitivity over scored sessions "
        "(total matched / total machine RE), not a per-session mean",
        default=None,
    )
    pooled_proxy_precision: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Pooled FL-run-proxy precision over scored sessions "
        "(total matched / total proxy RERAs), not a per-session mean",
        default=None,
    )


class ReraValidationReport(BaseModel):
    """Complete RERA validation report."""

    report_date: str = Field(
        description="Report generation timestamp (YYYY-MM-DD HH:MM:SS)"
    )
    date_range_start: str = Field(description="Start date of the requested range")
    date_range_end: str = Field(description="End date of the requested range")
    aggregate: ReraAggregateMetrics = Field(description="Aggregate metrics")
    sessions: list[ReraSessionValidation] = Field(description="Per-session results")


def export_rera_report_json(report: ReraValidationReport, output_path: Path) -> None:
    """Export RERA validation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)


def export_rera_report_csv(report: ReraValidationReport, output_path: Path) -> None:
    """Export RERA validation report as CSV (one row per session)."""
    fieldnames = [
        "session_id",
        "date",
        "duration_hours",
        "skipped_reason",
        "machine_re_count",
        "amplitude_rera_count",
        "proxy_rera_count",
        "amplitude_matched",
        "proxy_matched",
        "machine_re_density",
        "amplitude_sensitivity",
        "amplitude_precision",
        "amplitude_f1",
        "amplitude_density",
        "proxy_sensitivity",
        "proxy_precision",
        "proxy_f1",
        "proxy_density",
    ]

    def _fmt(v: float | None) -> str:
        # Full round-trippable precision — tiny magnitudes (proxy precision ~1e-3)
        # must not be rounded away in the data export.
        return "" if v is None else repr(v)

    def _fmt_int(v: int | None) -> str:
        # Matched counts are null for skipped sessions; keep the cell empty.
        return "" if v is None else str(v)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in report.sessions:
            writer.writerow(
                {
                    "session_id": s.session_id,
                    "date": s.date,
                    "duration_hours": f"{s.duration_hours:.1f}",
                    "skipped_reason": s.skipped_reason or "",
                    "machine_re_count": s.machine_re_count,
                    "amplitude_rera_count": s.amplitude_rera_count,
                    "proxy_rera_count": s.proxy_rera_count,
                    "amplitude_matched": _fmt_int(s.amplitude_matched),
                    "proxy_matched": _fmt_int(s.proxy_matched),
                    "machine_re_density": _fmt(s.machine_re_density),
                    "amplitude_sensitivity": _fmt(s.amplitude_sensitivity),
                    "amplitude_precision": _fmt(s.amplitude_precision),
                    "amplitude_f1": _fmt(s.amplitude_f1),
                    "amplitude_density": _fmt(s.amplitude_density),
                    "proxy_sensitivity": _fmt(s.proxy_sensitivity),
                    "proxy_precision": _fmt(s.proxy_precision),
                    "proxy_f1": _fmt(s.proxy_f1),
                    "proxy_density": _fmt(s.proxy_density),
                }
            )
