"""
Flow-limitation validation report models and exporters.

Signal-level comparison of SNORE's per-breath flow-limitation metrics against
the ResMed device's continuous FlowLim.2s (FLG) signal.

Severity orientation
--------------------
Both SNORE comparators use **direct severity** (higher = more flow-limited):

- ``flattening_severity``:  ``1 − mid_insp_flattening``
  ``mid_insp_flattening`` is an *inverse* severity measure (~1.0 = unimpeded,
  <0.7 = flow-limited).  The nightly ``fl_median``/``fl_95th`` aggregates in
  BreathService store raw ``mid_insp_flattening`` values (inverse severity).
  ``snore_fl_95th`` here reports the 95th percentile of ``flattening_severity``
  (i.e., direct severity, = 1 − raw mid_insp_flattening) for comparability
  with ``device_flg_95th``.

- ``flatness_index``: direct severity as-is (higher = more flat = more limited).
"""

import csv
import json

from pathlib import Path

from pydantic import BaseModel, Field

from snore.provenance import Provenance, provenance_field


class FlSessionValidation(BaseModel):
    """Validation results for a single session's FL signal comparison."""

    session_id: int = Field(description="Database session ID")
    date: str = Field(description="Session date (YYYY-MM-DD)")
    duration_hours: float = provenance_field(
        Provenance.DEVICE, "Session duration in hours"
    )
    parser_version: str = Field(description="Waveform parser/import version tag")
    has_flg_waveform: bool = Field(
        description="Whether a device FLG waveform row exists for this session"
    )
    skipped_reason: str | None = Field(
        default=None,
        description=(
            "Why this session was excluded from comparison. Possible values: "
            "'no_flg_waveform' — no device FLG waveform row; "
            "'no_analysis' — no completed analysis result; "
            "'no_valid_breaths' — no leak-valid breaths with required fields; "
            "'no_flg_samples' — waveform row exists but data_blob is None or sample_count is 0; "
            "'no_aligned_pairs' — all breath windows had zero FLG samples after alignment; "
            "'error' — unhandled exception during session validation; "
            "None — session was fully compared."
        ),
    )
    n_breaths_compared: int = provenance_field(
        Provenance.EXPERIMENTAL,
        "Number of (SNORE-segmented breath, FLG) pairs actually compared after "
        "dropping NaN (zero-sample) alignment windows",
        default=0,
    )
    low_sample_warning: bool = provenance_field(
        Provenance.EXPERIMENTAL, "True when n_breaths_compared < 20", default=False
    )
    n_class_breaths_compared: int = provenance_field(
        Provenance.EXPERIMENTAL,
        "Number of breaths entering the flow_class-weight metrics "
        "(spearman_class_weight_r, auc_class_t25/t50): the subset of "
        "n_breaths_compared that is also rule-matched with a known class. "
        "Can be far smaller than n_breaths_compared",
        default=0,
    )
    spearman_flattening_r: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Spearman r between flattening_severity (1 − mid_insp_flattening) "
        "and breath-averaged device FLG; None if n < 3 or either side constant",
        default=None,
    )
    spearman_flattening_p: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "p-value for spearman_flattening_r", default=None
    )
    spearman_flatness_r: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Spearman r between flatness_index (direct severity) "
        "and breath-averaged device FLG; None if n < 3 or either side constant",
        default=None,
    )
    spearman_flatness_p: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "p-value for spearman_flatness_r", default=None
    )
    auc_t25: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "AUC (Mann-Whitney U / n_pos*n_neg) discriminating device FLG >= 0.25 "
        "using flattening_severity as score; None if either class empty",
        default=None,
    )
    auc_t50: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "AUC discriminating device FLG >= 0.50 using flattening_severity; "
        "None if either class empty",
        default=None,
    )
    spearman_class_weight_r: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Spearman r between 7-class flow_class severity weight and "
        "breath-averaged device FLG, over rule-matched breaths only "
        "(flow_confidence > 0.5; fallback-confidence guesses excluded); "
        "None if fewer than 3 such breaths or either side constant",
        default=None,
    )
    spearman_class_weight_p: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "p-value for spearman_class_weight_r", default=None
    )
    auc_class_t25: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "AUC discriminating device FLG >= 0.25 using flow_class severity "
        "weight as score, over rule-matched breaths; None if either class empty",
        default=None,
    )
    auc_class_t50: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "AUC discriminating device FLG >= 0.50 using flow_class severity "
        "weight, over rule-matched breaths; None if either class empty",
        default=None,
    )
    snore_fl_95th: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "95th percentile of flattening_severity (1 − mid_insp_flattening) "
        "over leak-valid breaths; direct severity orientation",
        default=None,
    )
    device_flg_95th: float | None = provenance_field(
        Provenance.DERIVED,
        "95th percentile of masked FLG samples (values in [0, 1]) "
        "over the full session",
        default=None,
    )


class FlAggregateMetrics(BaseModel):
    """Aggregate FL validation metrics across multiple sessions."""

    total_sessions: int = Field(description="Sessions in the requested date range")
    sessions_compared: int = Field(description="Sessions with metrics computed")
    sessions_skipped_no_flg: int = Field(
        description="Sessions skipped: no FLG waveform row"
    )
    sessions_skipped_no_analysis: int = Field(
        description="Sessions skipped: no completed analysis result"
    )
    sessions_skipped_no_valid_breaths: int = Field(
        description="Sessions skipped: no leak-valid breaths with required fields"
    )
    mean_spearman_flattening_r: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean Spearman r (flattening_severity) over compared sessions",
        default=None,
    )
    mean_spearman_flatness_r: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean Spearman r (flatness_index) over compared sessions",
        default=None,
    )
    mean_auc_t25: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Mean AUC at FLG threshold 0.25", default=None
    )
    mean_auc_t50: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Mean AUC at FLG threshold 0.50", default=None
    )
    mean_spearman_class_weight_r: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean Spearman r (flow_class severity weight) over compared sessions",
        default=None,
    )
    mean_auc_class_t25: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean class-weight AUC at FLG threshold 0.25",
        default=None,
    )
    mean_auc_class_t50: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean class-weight AUC at FLG threshold 0.50",
        default=None,
    )
    cross_night_spearman_r: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Cross-night Spearman r of (snore_fl_95th, device_flg_95th) pairs; "
        "None if fewer than 3 paired nights",
        default=None,
    )
    cross_night_spearman_p: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "p-value for cross_night_spearman_r", default=None
    )


class FlValidationReport(BaseModel):
    """Complete FL signal validation report."""

    report_date: str = Field(
        description="Report generation timestamp (YYYY-MM-DD HH:MM:SS)"
    )
    date_range_start: str = Field(description="Start date of the requested range")
    date_range_end: str = Field(description="End date of the requested range")
    aggregate: FlAggregateMetrics = Field(description="Aggregate metrics")
    sessions: list[FlSessionValidation] = Field(description="Per-session results")


def export_fl_report_json(report: FlValidationReport, output_path: Path) -> None:
    """Export FL validation report as JSON.

    Args:
        report: FL validation report to export
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)


def export_fl_report_csv(report: FlValidationReport, output_path: Path) -> None:
    """Export FL validation report as CSV (per-session rows).

    Args:
        report: FL validation report to export
        output_path: Path to output CSV file
    """
    fieldnames = [
        "session_id",
        "date",
        "duration_hours",
        "parser_version",
        "has_flg_waveform",
        "skipped_reason",
        "n_breaths_compared",
        "low_sample_warning",
        "n_class_breaths_compared",
        "spearman_flattening_r",
        "spearman_flattening_p",
        "spearman_flatness_r",
        "spearman_flatness_p",
        "auc_t25",
        "auc_t50",
        "spearman_class_weight_r",
        "spearman_class_weight_p",
        "auc_class_t25",
        "auc_class_t50",
        "snore_fl_95th",
        "device_flg_95th",
    ]

    def _fmt(v: float | None) -> str:
        if v is None:
            return ""
        return f"{v:.4f}"

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
                    "parser_version": s.parser_version,
                    "has_flg_waveform": s.has_flg_waveform,
                    "skipped_reason": s.skipped_reason or "",
                    "n_breaths_compared": s.n_breaths_compared,
                    "low_sample_warning": s.low_sample_warning,
                    "n_class_breaths_compared": s.n_class_breaths_compared,
                    "spearman_flattening_r": _fmt(s.spearman_flattening_r),
                    "spearman_flattening_p": _fmt(s.spearman_flattening_p),
                    "spearman_flatness_r": _fmt(s.spearman_flatness_r),
                    "spearman_flatness_p": _fmt(s.spearman_flatness_p),
                    "auc_t25": _fmt(s.auc_t25),
                    "auc_t50": _fmt(s.auc_t50),
                    "spearman_class_weight_r": _fmt(s.spearman_class_weight_r),
                    "spearman_class_weight_p": _fmt(s.spearman_class_weight_p),
                    "auc_class_t25": _fmt(s.auc_class_t25),
                    "auc_class_t50": _fmt(s.auc_class_t50),
                    "snore_fl_95th": _fmt(s.snore_fl_95th),
                    "device_flg_95th": _fmt(s.device_flg_95th),
                }
            )
