"""
Flow-limitation signal validator.

Compares SNORE's per-breath FL metrics against the ResMed device's continuous
FlowLim.2s (FLG) signal stored in the waveforms table.

Unlike the event-matching BatchValidator, this module is signal-level because
ResMed emits FL only as a 0.5 Hz continuous signal (FlowLim.2s, 0–1) and never
as discrete FL events.  The validator aligns each breath window against the FLG
signal and computes Spearman correlations and AUC metrics.

The primary SNORE comparator is ``flattening_severity = 1 − mid_insp_flattening``
(direct severity: higher = more flow-limited), because ``mid_insp_flattening``
is an inverse measure (~1.0 = unimpeded).  The secondary comparator is
``flatness_index`` (already direct severity).
"""

from __future__ import annotations

import logging
import warnings

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from scipy.stats import mannwhitneyu
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.data.waveform_loader import deserialize_waveform_blob
from snore.analysis.shared.versioning import AnalysisStatus
from snore.constants import FLOW_LIMITATION_CLASSES
from snore.constants import FlowLimitationConstants as FLC
from snore.database import models
from snore.services.breath_service import BreathService
from snore.validation.alignment import average_waveform_over_breaths
from snore.validation.fl_report import (
    FlAggregateMetrics,
    FlSessionValidation,
    FlValidationReport,
)
from snore.validation.session_scoping import work_session
from snore.validation.stats import mean_or_none, spearman_or_none

logger = logging.getLogger(__name__)

_FLG_WAVEFORM_TYPE = "fl"
# Sessions comparing fewer breaths than this set ``low_sample_warning``.
LOW_SAMPLE_BREATHS = 20


def _auc_mwu(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """AUC via Mann-Whitney U: U / (n_pos * n_neg).  None if either class empty."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mannwhitneyu(pos, neg, alternative="greater")
    return float(result.statistic) / (len(pos) * len(neg))


def auc_severity_vs_flg(
    severity_valid: np.ndarray, flg_valid: np.ndarray, threshold: float
) -> float | None:
    """AUC of a severity score separating breaths with ``FLG >= threshold``.

    The single FLG-breakpoint labeling seam: ``score_fl_arrays`` computes both
    threshold AUCs through it, and the offline sweep's FLG grid drives it
    directly over cached per-session arrays so the two paths stay identical.
    """
    return _auc_mwu(severity_valid, flg_valid >= threshold)


def _percentile95(arr: np.ndarray) -> float | None:
    """95th percentile; None for empty arrays."""
    if len(arr) == 0:
        return None
    idx = min(int(len(arr) * 0.95), len(arr) - 1)
    return float(np.sort(arr)[idx])


# Default FLG breakpoints for the AUC labels — the query-time tunables the
# offline sweep harness re-scores around.  They live here (not the sweep module)
# because ``score_fl_arrays`` is the single implementation of FL array scoring.
FLG_AUC_LOW_THRESHOLD_DEFAULT = 0.25
FLG_AUC_HIGH_THRESHOLD_DEFAULT = 0.50


@dataclass(frozen=True)
class FlArrayScores:
    """Per-session FL scoring outputs from the pure ``score_fl_arrays`` core."""

    n_breaths_compared: int
    n_class_breaths_compared: int
    spearman_flattening_r: float | None
    spearman_flattening_p: float | None
    spearman_flatness_r: float | None
    spearman_flatness_p: float | None
    auc_low: float | None
    auc_high: float | None
    spearman_class_weight_r: float | None
    spearman_class_weight_p: float | None
    auc_class_low: float | None
    auc_class_high: float | None
    snore_fl_95th: float | None
    device_flg_95th: float | None


def score_fl_arrays(
    mid_insp_flattening: np.ndarray,
    flatness_index: np.ndarray,
    class_weight: np.ndarray,
    rule_matched: np.ndarray,
    breath_flg: np.ndarray,
    session_flg_values: np.ndarray,
    *,
    flg_low_threshold: float = FLG_AUC_LOW_THRESHOLD_DEFAULT,
    flg_high_threshold: float = FLG_AUC_HIGH_THRESHOLD_DEFAULT,
) -> FlArrayScores | None:
    """Score aligned per-breath FL arrays against the device FLG signal.

    The pure scoring core shared by ``FlowLimitationValidator`` (via a thin
    wrapper) and the offline threshold-sweep harness.  No DB or session
    dependency: all inputs are parallel per-breath arrays.

    Args:
        mid_insp_flattening: per-breath ``mid_insp_flattening`` (inverse severity;
            ~1.0 = unimpeded).  ``flattening_severity = 1 − mid_insp`` internally.
        flatness_index: per-breath ``flatness_index`` (direct severity).
        class_weight: per-breath 7-class severity weight, ``NaN`` where unknown.
        rule_matched: per-breath bool; only rule-matched breaths enter the
            class-weight correlation (fallback guesses are excluded).
        breath_flg: per-breath FLG averaged over the breath window, ``NaN`` where
            no FLG samples fell in the window.
        session_flg_values: full-session clipped FLG samples, for the device 95th
            percentile (intentionally asymmetric with the breath-aligned subset).
        flg_low_threshold: FLG breakpoint for the low-threshold AUC labels.
        flg_high_threshold: FLG breakpoint for the high-threshold AUC labels.

    Returns:
        ``FlArrayScores``, or ``None`` when no breath has an aligned FLG value
        (the caller renders this as ``no_aligned_pairs``).
    """
    flattening_severity = 1.0 - mid_insp_flattening

    valid = ~np.isnan(breath_flg)
    if valid.sum() == 0:
        return None

    flg_valid = breath_flg[valid]
    flat_sev_valid = flattening_severity[valid]
    flatness_valid = flatness_index[valid]
    n_compared = int(valid.sum())

    spr_flat_r, spr_flat_p = spearman_or_none(flat_sev_valid, flg_valid)
    spr_fi_r, spr_fi_p = spearman_or_none(flatness_valid, flg_valid)

    auc_low = auc_severity_vs_flg(flat_sev_valid, flg_valid, flg_low_threshold)
    auc_high = auc_severity_vs_flg(flat_sev_valid, flg_valid, flg_high_threshold)

    # flow_class weight vs FLG — over rule-matched breaths with a known class.
    class_mask = valid & rule_matched & ~np.isnan(class_weight)
    n_class_compared = int(class_mask.sum())
    weights_c = class_weight[class_mask]
    flg_c = breath_flg[class_mask]
    spr_cw_r, spr_cw_p = spearman_or_none(weights_c, flg_c)
    auc_class_low = auc_severity_vs_flg(weights_c, flg_c, flg_low_threshold)
    auc_class_high = auc_severity_vs_flg(weights_c, flg_c, flg_high_threshold)

    # device_flg_95th uses all session FLG samples (full-session population);
    # snore_fl_95th uses only the breath-aligned subset — intentionally asymmetric.
    snore_95th = _percentile95(flat_sev_valid)
    device_95th = _percentile95(session_flg_values)

    return FlArrayScores(
        n_breaths_compared=n_compared,
        n_class_breaths_compared=n_class_compared,
        spearman_flattening_r=spr_flat_r,
        spearman_flattening_p=spr_flat_p,
        spearman_flatness_r=spr_fi_r,
        spearman_flatness_p=spr_fi_p,
        auc_low=auc_low,
        auc_high=auc_high,
        spearman_class_weight_r=spr_cw_r,
        spearman_class_weight_p=spr_cw_p,
        auc_class_low=auc_class_low,
        auc_class_high=auc_class_high,
        snore_fl_95th=snore_95th,
        device_flg_95th=device_95th,
    )


class FlowLimitationValidator:
    """Validates SNORE's FL metrics against the device's FLG waveform signal."""

    def __init__(self, db_session: AsyncSession | None, profile_id: int) -> None:
        # None → JOB mode: work_session opens a fresh short scope per session so
        # the WAL read snapshot is released between sessions.  A real session →
        # shared mode: every unit of work runs on it, one transaction, as before.
        self._injected = db_session
        self._db: AsyncSession = db_session  # type: ignore[assignment]
        self._profile_id = profile_id

    async def validate_date_range(
        self,
        date_from: str,
        date_to: str,
    ) -> FlValidationReport:
        """Run FL signal validation across a date range.

        Args:
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)

        Returns:
            FlValidationReport with aggregate and per-session metrics
        """
        stmt = (
            select(models.Session)
            .join(models.Device, models.Session.device_id == models.Device.id)
            .where(
                models.Device.profile_id == self._profile_id,
                models.Session.start_time >= datetime.fromisoformat(date_from),
                models.Session.start_time
                <= datetime.fromisoformat(f"{date_to} 23:59:59"),
            )
            .order_by(models.Session.start_time)
        )

        async with work_session(self._injected) as db:
            self._db = db
            sessions = (await db.execute(stmt)).scalars().all()
        logger.info(f"Found {len(sessions)} sessions between {date_from} and {date_to}")

        session_results: list[FlSessionValidation] = []
        for session in sessions:
            try:
                async with work_session(self._injected) as db:
                    self._db = db
                    result = await self._validate_session(session)
                session_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to validate session {session.id}: {e}")
                session_results.append(
                    FlSessionValidation(
                        session_id=session.id,
                        date=session.start_time.strftime("%Y-%m-%d"),
                        duration_hours=(session.duration_seconds or 0) / 3600.0,
                        parser_version=session.parser_version or "unknown",
                        has_flg_waveform=False,
                        skipped_reason="error",
                    )
                )

        aggregate = self._calculate_aggregate(session_results)

        return FlValidationReport(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_range_start=date_from,
            date_range_end=date_to,
            aggregate=aggregate,
            sessions=session_results,
        )

    async def _validate_session(self, session: models.Session) -> FlSessionValidation:
        """Validate a single session, returning a FlSessionValidation (possibly skipped)."""
        duration_hours = (session.duration_seconds or 0) / 3600.0
        date_str = session.start_time.strftime("%Y-%m-%d")
        parser_version = session.parser_version or "unknown"

        def _skip(reason: str, has_flg: bool) -> FlSessionValidation:
            return FlSessionValidation(
                session_id=session.id,
                date=date_str,
                duration_hours=duration_hours,
                parser_version=parser_version,
                has_flg_waveform=has_flg,
                skipped_reason=reason,
            )

        # 1. Fetch FLG waveform row
        waveform_row = (
            (
                await self._db.execute(
                    select(models.Waveform).filter_by(
                        session_id=session.id, waveform_type=_FLG_WAVEFORM_TYPE
                    )
                )
            )
            .scalars()
            .first()
        )
        if waveform_row is None:
            return _skip("no_flg_waveform", has_flg=False)

        # 2. Look up latest OK analysis via BreathService public helper
        breath_svc = BreathService(self._db, self._profile_id)
        status, _algo, ar_id = await breath_svc.latest_analysis_for_session(session.id)
        if status != AnalysisStatus.OK or ar_id is None:
            return _skip("no_analysis", has_flg=True)

        # 3. Fetch leak-valid breaths with all required fields
        breath_stmt = (
            select(models.Breath)
            .where(
                models.Breath.analysis_result_id == ar_id,
                models.Breath.leak_valid.is_(True),
                models.Breath.mid_insp_flattening.is_not(None),
                models.Breath.flatness_index.is_not(None),
                models.Breath.start_offset_s.is_not(None),
                models.Breath.end_offset_s.is_not(None),
            )
            .order_by(models.Breath.breath_number)
        )
        breaths = (await self._db.execute(breath_stmt)).scalars().all()
        if not breaths:
            return _skip("no_valid_breaths", has_flg=True)

        # 4. Guard NULL data_blob before deserialization
        if waveform_row.data_blob is None:
            return _skip("no_flg_samples", has_flg=True)

        # 5. Check sample count before deserialization
        sample_count = waveform_row.sample_count or 0
        if sample_count == 0:
            return _skip("no_flg_samples", has_flg=True)

        flg_timestamps, flg_values = deserialize_waveform_blob(
            waveform_row.data_blob, sample_count
        )
        # Mask out the rare −0.01 mask-off sentinel and any non-finite values;
        # clamp remaining to [0, 1].
        valid_mask = (flg_values >= 0.0) & np.isfinite(flg_values)
        flg_timestamps = flg_timestamps[valid_mask]
        flg_values = np.clip(flg_values[valid_mask], 0.0, 1.0)

        # 6. Align breaths to FLG
        starts = np.array([b.start_offset_s for b in breaths], dtype=np.float64)
        ends = np.array([b.end_offset_s for b in breaths], dtype=np.float64)
        mid_insp = np.array([b.mid_insp_flattening for b in breaths], dtype=np.float64)
        flatness = np.array([b.flatness_index for b in breaths], dtype=np.float64)

        # Per-breath 7-class severity weight (NaN where the class is unknown).
        # flow_confidence == FL_DEFAULT_CONFIDENCE marks a fallback guess; only
        # rule-matched breaths (above it) enter the class-weight correlation so
        # fallback guesses do not pollute it.
        class_weight = np.array(
            [
                FLOW_LIMITATION_CLASSES[b.flow_class]["weight"]
                if b.flow_class in FLOW_LIMITATION_CLASSES
                else np.nan
                for b in breaths
            ],
            dtype=np.float64,
        )
        rule_matched = np.array(
            [
                b.flow_confidence is not None
                and b.flow_confidence > FLC.FL_DEFAULT_CONFIDENCE
                for b in breaths
            ],
            dtype=bool,
        )

        breath_flg = average_waveform_over_breaths(
            starts,
            ends,
            flg_timestamps.astype(np.float64),
            flg_values.astype(np.float64),
        )

        # 7. Metrics — delegate to the pure array-in scoring core.
        scores = score_fl_arrays(
            mid_insp,
            flatness,
            class_weight,
            rule_matched,
            breath_flg,
            flg_values,
        )
        if scores is None:
            return _skip("no_aligned_pairs", has_flg=True)

        return FlSessionValidation(
            session_id=session.id,
            date=date_str,
            duration_hours=duration_hours,
            parser_version=parser_version,
            has_flg_waveform=True,
            skipped_reason=None,
            n_breaths_compared=scores.n_breaths_compared,
            low_sample_warning=scores.n_breaths_compared < LOW_SAMPLE_BREATHS,
            n_class_breaths_compared=scores.n_class_breaths_compared,
            spearman_flattening_r=scores.spearman_flattening_r,
            spearman_flattening_p=scores.spearman_flattening_p,
            spearman_flatness_r=scores.spearman_flatness_r,
            spearman_flatness_p=scores.spearman_flatness_p,
            auc_t25=scores.auc_low,
            auc_t50=scores.auc_high,
            spearman_class_weight_r=scores.spearman_class_weight_r,
            spearman_class_weight_p=scores.spearman_class_weight_p,
            auc_class_t25=scores.auc_class_low,
            auc_class_t50=scores.auc_class_high,
            snore_fl_95th=scores.snore_fl_95th,
            device_flg_95th=scores.device_flg_95th,
        )

    @staticmethod
    def _calculate_aggregate(
        sessions: list[FlSessionValidation],
    ) -> FlAggregateMetrics:
        compared = [s for s in sessions if s.skipped_reason is None]
        skipped_no_flg = sum(
            1 for s in sessions if s.skipped_reason == "no_flg_waveform"
        )
        skipped_no_analysis = sum(
            1 for s in sessions if s.skipped_reason == "no_analysis"
        )
        skipped_no_breaths = sum(
            1 for s in sessions if s.skipped_reason == "no_valid_breaths"
        )

        flat_rs = [
            s.spearman_flattening_r
            for s in compared
            if s.spearman_flattening_r is not None
        ]
        fi_rs = [
            s.spearman_flatness_r for s in compared if s.spearman_flatness_r is not None
        ]
        auc25s = [s.auc_t25 for s in compared if s.auc_t25 is not None]
        auc50s = [s.auc_t50 for s in compared if s.auc_t50 is not None]

        cw_rs = [
            s.spearman_class_weight_r
            for s in compared
            if s.spearman_class_weight_r is not None
        ]
        cw_auc25s = [s.auc_class_t25 for s in compared if s.auc_class_t25 is not None]
        cw_auc50s = [s.auc_class_t50 for s in compared if s.auc_class_t50 is not None]

        # Cross-night Spearman on (snore_fl_95th, device_flg_95th) pairs
        pairs = [
            (s.snore_fl_95th, s.device_flg_95th)
            for s in compared
            if s.snore_fl_95th is not None and s.device_flg_95th is not None
        ]
        cross_r: float | None = None
        cross_p: float | None = None
        if len(pairs) >= 3:
            snore_95ths = np.array([p[0] for p in pairs])
            device_95ths = np.array([p[1] for p in pairs])
            cross_r, cross_p = spearman_or_none(snore_95ths, device_95ths)

        return FlAggregateMetrics(
            total_sessions=len(sessions),
            sessions_compared=len(compared),
            sessions_skipped_no_flg=skipped_no_flg,
            sessions_skipped_no_analysis=skipped_no_analysis,
            sessions_skipped_no_valid_breaths=skipped_no_breaths,
            mean_spearman_flattening_r=mean_or_none(flat_rs),
            mean_spearman_flatness_r=mean_or_none(fi_rs),
            mean_auc_t25=mean_or_none(auc25s),
            mean_auc_t50=mean_or_none(auc50s),
            mean_spearman_class_weight_r=mean_or_none(cw_rs),
            mean_auc_class_t25=mean_or_none(cw_auc25s),
            mean_auc_class_t50=mean_or_none(cw_auc50s),
            cross_night_spearman_r=cross_r,
            cross_night_spearman_p=cross_p,
        )
