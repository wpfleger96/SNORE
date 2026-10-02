"""Apple Health cross-source night-level validation: models, seam, exporters.

SNORE's experimental FL/RERA nightly metrics carry a single reference axis
(the ResMed device's own signals).  Apple Watch data provides a *genuinely
independent* second axis: RERAs end in cortical arousal, so watch-detected
awakenings / sleep fragmentation and Apple's own sleeping-breathing-disturbance
metric are noisy-but-independent validity checks on the SNORE indices.

This is measurement infrastructure only — it correlates existing nightly
outputs, and changes no algorithm or threshold.

Metric orientation
------------------
- ``rera_index`` / ``fl_class_ge4_pct``: higher = more respiratory disturbance.
- ``apple_breathing_disturbances``: higher = more disturbed breathing.
- ``awake_seconds``: higher = more fragmentation.
- ``sleep_efficiency_pct``: higher = *less* fragmentation (expected negative rho
  against the SNORE disturbance indices).

Spearman rho captures the direction, so no sign-flipping is applied here.
"""

from __future__ import annotations

import csv
import json

from datetime import date
from pathlib import Path

import numpy as np

from pydantic import BaseModel, Field

from snore.provenance import Provenance, provenance_field
from snore.validation.stats import spearman_or_none

# Minimum paired nights before Spearman is attempted.  This is a floor for
# *computability*, not statistical power: near it, the p-value is coarse and
# floored high (3 pairs → smallest possible two-sided p ≈ 0.33) and rho is only
# directional.  Treat rho below ~n=5 as a sign, not an effect size.
_MIN_PAIRS = 3


class PairCorrelation(BaseModel):
    """Spearman correlation for one SNORE↔Apple metric pair over paired nights."""

    rho: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Spearman rho over nights present in both series; None when fewer "
        "than 3 pairs or a side is constant (see reason)",
        default=None,
    )
    p_value: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "p-value for rho; None whenever rho is None",
        default=None,
    )
    n_paired_nights: int = Field(
        default=0,
        description="Nights contributing a value to both series",
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Why rho is null: 'insufficient_pairs' (< 3 paired nights); "
            "'degenerate' (>= 3 pairs but a side is constant / scipy returned "
            "NaN); None when rho was computed"
        ),
    )


def correlate_night_pairs(
    snore_values_by_night: dict[date, float],
    apple_values_by_night: dict[date, float],
) -> PairCorrelation:
    """Pure Spearman over the night-date intersection of two series.

    The mandatory design seam: a module-level, array-in function with no DB or
    session dependency, so an offline sweep harness can drive it directly.  The
    validator is a thin wrapper that assembles the two dicts and calls this.

    Nights present in both mappings are paired in ascending date order; the rest
    are dropped.  Fewer than 3 pairs yields ``reason='insufficient_pairs'``; a
    constant side (or NaN from scipy) yields ``reason='degenerate'``.
    """
    shared = sorted(snore_values_by_night.keys() & apple_values_by_night.keys())
    n = len(shared)
    if n < _MIN_PAIRS:
        return PairCorrelation(n_paired_nights=n, reason="insufficient_pairs")

    snore_arr = np.array([snore_values_by_night[d] for d in shared], dtype=np.float64)
    apple_arr = np.array([apple_values_by_night[d] for d in shared], dtype=np.float64)
    rho, p = spearman_or_none(snore_arr, apple_arr)
    if rho is None:
        return PairCorrelation(n_paired_nights=n, reason="degenerate")
    return PairCorrelation(rho=rho, p_value=p, n_paired_nights=n, reason=None)


class AppleCrossNightRecord(BaseModel):
    """One night's SNORE indices, independent Apple signals, and skip status."""

    night_date: str = Field(description="Therapy night (YYYY-MM-DD, noon-split)")

    rera_index: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "SNORE nightly RERA index (RERAs / therapy hour)",
        default=None,
    )
    rera_index_reason: str | None = Field(
        default=None, description="NullReason code when rera_index is null"
    )
    fl_class_ge4_pct: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "SNORE percent of leak-valid classified breaths at flow_class >= 4",
        default=None,
    )
    fl_class_ge4_pct_reason: str | None = Field(
        default=None, description="NullReason code when fl_class_ge4_pct is null"
    )

    apple_breathing_disturbances: float | None = provenance_field(
        Provenance.DEVICE,
        "Mean Apple sleeping-breathing-disturbance value for the night",
        default=None,
    )
    apple_bd_reason: str | None = Field(
        default=None,
        description="'no_apple_bd' when Apple recorded no disturbance value; else None",
    )
    awake_seconds: float | None = provenance_field(
        Provenance.DERIVED,
        "Apple-derived awake time in seconds (fragmentation)",
        default=None,
    )
    sleep_efficiency_pct: float | None = provenance_field(
        Provenance.DERIVED, "Apple-derived sleep efficiency percent", default=None
    )

    skip_reason: str | None = Field(
        default=None,
        description=(
            "Why the night contributes no SNORE side to any correlation: "
            "'analysis_not_run' — SNORE analysis never ran for the night; "
            "'analysis_stale' — SNORE analysis is stale / version-mismatched; "
            "'device_ambiguous' — the night has sessions from more than one "
            "device and no device_id was pinned to disambiguate; "
            "None — the night carries usable SNORE indices"
        ),
    )


class AppleCrossAggregate(BaseModel):
    """Aggregate coverage counters and the four cross-source correlations.

    The skip counters below are independent axes over the same nights, not a
    partition: the SNORE-side counters (``n_analysis_not_run``,
    ``n_analysis_stale``, ``n_device_ambiguous``) classify why a night's SNORE
    indices are unusable, while the Apple-side ``n_skipped_no_apple_bd`` counts
    nights lacking an Apple breathing-disturbance value.  A single night can be
    counted on both axes, so the counters must not be summed.
    """

    total_nights: int = Field(
        description=(
            "Nights carrying SNORE sessions in range (resolved + device-ambiguous)"
        )
    )
    n_analysis_not_run: int = Field(
        description="Nights skipped: SNORE analysis never ran"
    )
    n_analysis_stale: int = Field(
        description="Nights skipped: SNORE analysis stale / version-mismatched"
    )
    n_device_ambiguous: int = Field(
        description="Nights skipped: multiple devices, no device_id pinned"
    )
    n_skipped_no_apple_bd: int = Field(
        description=(
            "Nights with no Apple breathing-disturbance value (an independent "
            "axis from the SNORE-side skip counters; do not sum)"
        )
    )
    n_with_apple_bd: int = Field(
        description="Nights carrying an Apple breathing-disturbance value"
    )

    rera_vs_apple_bd: PairCorrelation = Field(
        description="rera_index vs Apple breathing disturbances"
    )
    fl_vs_apple_bd: PairCorrelation = Field(
        description="fl_class_ge4_pct vs Apple breathing disturbances"
    )
    rera_vs_awake_seconds: PairCorrelation = Field(
        description="rera_index vs Apple awake_seconds (fragmentation)"
    )
    fl_vs_sleep_efficiency: PairCorrelation = Field(
        description="fl_class_ge4_pct vs Apple sleep_efficiency_pct"
    )


class AppleCrossValidationReport(BaseModel):
    """Complete Apple Health cross-source night-level validation report."""

    report_date: str = Field(
        description="Report generation timestamp (YYYY-MM-DD HH:MM:SS)"
    )
    date_range_start: str = Field(description="Start date of the requested range")
    date_range_end: str = Field(description="End date of the requested range")
    aggregate: AppleCrossAggregate = Field(description="Coverage + correlations")
    nights: list[AppleCrossNightRecord] = Field(description="Per-night records")


def export_apple_cross_report_json(
    report: AppleCrossValidationReport, output_path: Path
) -> None:
    """Export the report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)


def export_apple_cross_report_csv(
    report: AppleCrossValidationReport, output_path: Path
) -> None:
    """Export the per-night records as CSV."""
    fieldnames = [
        "night_date",
        "rera_index",
        "rera_index_reason",
        "fl_class_ge4_pct",
        "fl_class_ge4_pct_reason",
        "apple_breathing_disturbances",
        "apple_bd_reason",
        "awake_seconds",
        "sleep_efficiency_pct",
        "skip_reason",
    ]

    def _fmt(v: float | None) -> str:
        return "" if v is None else f"{v:.4f}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.nights:
            writer.writerow(
                {
                    "night_date": r.night_date,
                    "rera_index": _fmt(r.rera_index),
                    "rera_index_reason": r.rera_index_reason or "",
                    "fl_class_ge4_pct": _fmt(r.fl_class_ge4_pct),
                    "fl_class_ge4_pct_reason": r.fl_class_ge4_pct_reason or "",
                    "apple_breathing_disturbances": _fmt(
                        r.apple_breathing_disturbances
                    ),
                    "apple_bd_reason": r.apple_bd_reason or "",
                    "awake_seconds": _fmt(r.awake_seconds),
                    "sleep_efficiency_pct": _fmt(r.sleep_efficiency_pct),
                    "skip_reason": r.skip_reason or "",
                }
            )
