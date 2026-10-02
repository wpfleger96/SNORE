"""
Breath-trends validation report models and exporters.

Cross-validates SNORE's per-breath respiratory metrics against the device's
independent 0.5 Hz trend signals (rr, tv, ti, ie_ratio).

Channel semantics
-----------------
Four device channels are compared against SNORE's per-breath segmentation:

``rr`` (bpm)
    Device: respiratory rate at 0.5 Hz, range [0, 90].
    SNORE:  instantaneous RR = 60 / (end_offset_s − start_offset_s).
    Device values are windowed/smoothed; SNORE values are instantaneous —
    MAE captures the expected smoothing divergence.

``tv`` (mL)
    Device: tidal volume at 0.5 Hz, stored in mL (converted from L at import),
    range [0, 4000].
    SNORE:  ``tidal_volume_ml`` column (mL).

``ti`` (s)
    Device: inspiratory time at 0.5 Hz, range [0, 10].  VAuto (bilevel) only;
    APAP sessions will report this channel as ``channel_not_recorded``.
    SNORE:  ``inspiration_time_s`` column (s).

``ie_ratio`` (percentage points)
    Device: I:E ratio assumed to be expressed as Ti/Te × 100 (e.g., a 1:2
    inspiration-to-expiration ratio → ie_ratio = 50).  VAuto only.
    SNORE:  100 × inspiration_time_s / expiration_time_s.
    Note: observed device data shows a +40 pp mean bias relative to SNORE's
    computation, suggesting the device may report the unitless Ti/Te ratio
    directly rather than Ti/Te × 100.  The exact device convention is
    unconfirmed; verification is future work.

Zero-average masking
--------------------
After aligning the device trend over each breath window, pairs where the
device-side average equals exactly 0.0 are dropped before metric computation.
Rationale: a real breath cannot produce a zero-average device sample (zeros in
the device signal occur at mask-off/no-breathing regions); any breath window
whose device average rounds to zero is dominated by trailing mask-off fill
values and is noise, not signal.  NaN (windows with no device samples at all)
are dropped by the same pass.
"""

import csv
import json

from pathlib import Path

from pydantic import BaseModel, Field

from snore.provenance import Provenance, provenance_field


class ChannelComparison(BaseModel):
    """Comparison metrics for one device trend channel in one session."""

    n_pairs: int = Field(
        default=0,
        description="Number of (SNORE, device) pairs after dropping NaN and zero-device windows",
    )
    spearman_r: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Spearman r between SNORE per-breath value and device breath-window average; "
        "None if n < 3 or either side is constant",
        default=None,
    )
    spearman_p: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "p-value for spearman_r", default=None
    )
    median_abs_error: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Median |SNORE − device| in native units (bpm / mL / s / pp); "
        "None if n_pairs == 0",
        default=None,
    )
    mean_bias: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean (SNORE − device) in native units; None if n_pairs == 0",
        default=None,
    )
    skipped_reason: str | None = Field(
        default=None,
        description=(
            "Why this channel was excluded: 'channel_not_recorded' | None. "
            "'channel_not_recorded' means no device waveform row exists for this "
            "channel in this session (normal for ti/ie_ratio on APAP sessions)."
        ),
    )


class ChannelAggregateMetrics(BaseModel):
    """Per-channel aggregate metrics across the validated date range."""

    sessions_with_data: int = Field(
        description="Sessions where this channel has at least one aligned pair"
    )
    mean_spearman_r: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean Spearman r over sessions with a non-None spearman_r",
        default=None,
    )
    mean_median_abs_error: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean of per-session median_abs_error over sessions with data",
        default=None,
    )
    mean_bias: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Mean of per-session mean_bias over sessions with data",
        default=None,
    )


class BreathTrendsAggregateMetrics(BaseModel):
    """Aggregate breath-trends validation metrics across multiple sessions."""

    total_sessions: int = Field(description="Sessions in the requested date range")
    sessions_compared: int = Field(
        description="Sessions with analysis and valid breaths (skipped_reason is None)"
    )
    sessions_skipped_no_analysis: int = Field(
        description="Sessions skipped: no completed analysis result"
    )
    sessions_skipped_no_valid_breaths: int = Field(
        description="Sessions skipped: no leak-valid breaths with timing columns"
    )
    rr: ChannelAggregateMetrics = Field(description="Aggregate for the RR channel")
    tv: ChannelAggregateMetrics = Field(description="Aggregate for the TV channel")
    ti: ChannelAggregateMetrics = Field(description="Aggregate for the Ti channel")
    ie_ratio: ChannelAggregateMetrics = Field(
        description="Aggregate for the I:E ratio channel"
    )


class BreathTrendsSessionValidation(BaseModel):
    """Breath-trends validation results for a single session."""

    session_id: int = Field(description="Database session ID")
    date: str = Field(description="Session date (YYYY-MM-DD)")
    duration_hours: float = provenance_field(
        Provenance.DEVICE, "Session duration in hours"
    )
    parser_version: str = Field(description="Waveform parser/import version tag")
    skipped_reason: str | None = Field(
        default=None,
        description=(
            "Why the whole session was excluded: "
            "'no_analysis' | 'no_valid_breaths' | 'error' | None.  "
            "'no_analysis' — no completed analysis result for this session.  "
            "'no_valid_breaths' — analysis exists but no leak-valid breaths with "
            "timing columns.  "
            "'error' — unhandled exception during session validation; details in logs."
        ),
    )
    n_breaths: int = provenance_field(
        Provenance.EXPERIMENTAL,
        (
            "Count of SNORE-segmented leak-valid breaths with timing columns "
            "fetched for this session.  This is the count BEFORE per-channel alignment filtering, so it is an upper "
            "bound across all channels; the authoritative per-channel count is each "
            "channel's `n_pairs`.  Intentionally differs from FL's `n_breaths_compared`, "
            "which counts breaths used in the FL comparison."
        ),
        default=0,
    )
    channels: dict[str, ChannelComparison] = Field(
        default_factory=dict,
        description=(
            "Per-channel comparison results keyed by channel name "
            "('rr', 'tv', 'ti', 'ie_ratio').  "
            "Empty when skipped_reason is set."
        ),
    )


class BreathTrendsValidationReport(BaseModel):
    """Complete breath-trends validation report."""

    report_date: str = Field(
        description="Report generation timestamp (YYYY-MM-DD HH:MM:SS)"
    )
    date_range_start: str = Field(description="Start date of the requested range")
    date_range_end: str = Field(description="End date of the requested range")
    aggregate: BreathTrendsAggregateMetrics = Field(description="Aggregate metrics")
    sessions: list[BreathTrendsSessionValidation] = Field(
        description="Per-session results"
    )


def export_breath_trends_report_json(
    report: BreathTrendsValidationReport, output_path: Path
) -> None:
    """Export breath-trends validation report as JSON.

    Args:
        report: Report to export.
        output_path: Path to the output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)


_CHANNEL_NAMES = ("rr", "tv", "ti", "ie_ratio")


def export_breath_trends_report_csv(
    report: BreathTrendsValidationReport, output_path: Path
) -> None:
    """Export breath-trends validation report as CSV (one row per session-channel).

    Args:
        report: Report to export.
        output_path: Path to the output CSV file.
    """
    fieldnames = [
        "session_id",
        "date",
        "duration_hours",
        "parser_version",
        "skipped_reason",
        "n_breaths",
        "channel",
        "n_pairs",
        "spearman_r",
        "spearman_p",
        "median_abs_error",
        "mean_bias",
        "channel_skipped_reason",
    ]

    def _fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in report.sessions:
            if s.skipped_reason is not None:
                # Skipped session — emit one row per channel with nulls
                for ch in _CHANNEL_NAMES:
                    writer.writerow(
                        {
                            "session_id": s.session_id,
                            "date": s.date,
                            "duration_hours": f"{s.duration_hours:.1f}",
                            "parser_version": s.parser_version,
                            "skipped_reason": s.skipped_reason,
                            "n_breaths": s.n_breaths,
                            "channel": ch,
                            "n_pairs": "",
                            "spearman_r": "",
                            "spearman_p": "",
                            "median_abs_error": "",
                            "mean_bias": "",
                            "channel_skipped_reason": "",
                        }
                    )
            else:
                for ch in _CHANNEL_NAMES:
                    cc = s.channels[ch]
                    writer.writerow(
                        {
                            "session_id": s.session_id,
                            "date": s.date,
                            "duration_hours": f"{s.duration_hours:.1f}",
                            "parser_version": s.parser_version,
                            "skipped_reason": "",
                            "n_breaths": s.n_breaths,
                            "channel": ch,
                            "n_pairs": cc.n_pairs,
                            "spearman_r": _fmt(cc.spearman_r),
                            "spearman_p": _fmt(cc.spearman_p),
                            "median_abs_error": _fmt(cc.median_abs_error),
                            "mean_bias": _fmt(cc.mean_bias),
                            "channel_skipped_reason": cc.skipped_reason or "",
                        }
                    )
