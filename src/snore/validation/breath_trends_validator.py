"""
Breath-trends validator.

Cross-validates SNORE's per-breath respiratory segmentation against the device's
independent 0.5 Hz trend signals.  See ``breath_trends_report`` for the full
semantic description of each channel and the zero-average masking rule.
"""

from __future__ import annotations

import logging

from datetime import datetime

import numpy as np

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.shared.versioning import AnalysisStatus
from snore.database import models
from snore.database.waveform_blobs import deserialize_waveform_blob
from snore.services.breath_service import BreathService
from snore.validation.alignment import average_waveform_over_breaths
from snore.validation.breath_trends_report import (
    _CHANNEL_NAMES,
    BreathTrendsAggregateMetrics,
    BreathTrendsSessionValidation,
    BreathTrendsValidationReport,
    ChannelAggregateMetrics,
    ChannelComparison,
)
from snore.validation.session_scoping import work_session
from snore.validation.stats import mean_or_none, spearman_or_none

logger = logging.getLogger(__name__)


def _channel_not_recorded() -> ChannelComparison:
    return ChannelComparison(skipped_reason="channel_not_recorded")


def _snore_rr(breaths: list[models.Breath]) -> np.ndarray:
    """Instantaneous RR in bpm from breath timing; NaN when duration ≤ 0."""
    result = np.full(len(breaths), np.nan, dtype=np.float64)
    for i, b in enumerate(breaths):
        dur = b.end_offset_s - b.start_offset_s
        if dur > 0:
            result[i] = 60.0 / dur
    return result


def _snore_tv(breaths: list[models.Breath]) -> np.ndarray:
    """Tidal volume in mL; NaN when column is None."""
    return np.array(
        [
            b.tidal_volume_ml if b.tidal_volume_ml is not None else np.nan
            for b in breaths
        ],
        dtype=np.float64,
    )


def _snore_ti(breaths: list[models.Breath]) -> np.ndarray:
    """Inspiratory time in seconds; NaN when column is None."""
    return np.array(
        [
            b.inspiration_time_s if b.inspiration_time_s is not None else np.nan
            for b in breaths
        ],
        dtype=np.float64,
    )


def _snore_ie_ratio(breaths: list[models.Breath]) -> np.ndarray:
    """I:E ratio as 100 × Ti / Te; NaN when Ti/Te missing or Te ≤ 0."""
    result = np.full(len(breaths), np.nan, dtype=np.float64)
    for i, b in enumerate(breaths):
        ti = b.inspiration_time_s
        te = b.expiration_time_s
        if ti is not None and te is not None and te > 0:
            result[i] = 100.0 * ti / te
    return result


_SNORE_VALUE_FUNCS = {
    "rr": _snore_rr,
    "tv": _snore_tv,
    "ti": _snore_ti,
    "ie_ratio": _snore_ie_ratio,
}


def _compute_channel_metrics(
    snore_vals: np.ndarray,
    device_avgs: np.ndarray,
) -> ChannelComparison:
    """Compute ChannelComparison metrics from aligned SNORE/device arrays.

    Drops pairs where either side is NaN, or where the device average equals
    exactly 0.0 (mask-off zero-fill noise; see module docstring).
    """
    valid = np.isfinite(snore_vals) & np.isfinite(device_avgs) & (device_avgs != 0.0)
    s = snore_vals[valid]
    d = device_avgs[valid]
    n = int(valid.sum())

    if n == 0:
        return ChannelComparison(n_pairs=0)

    r, p = spearman_or_none(s, d)
    diff = s - d
    mae = float(np.median(np.abs(diff)))
    bias = float(np.mean(diff))

    return ChannelComparison(
        n_pairs=n,
        spearman_r=r,
        spearman_p=p,
        median_abs_error=mae,
        mean_bias=bias,
    )


class BreathTrendsValidator:
    """Validates SNORE's breath segmentation against device 0.5 Hz trend signals."""

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
    ) -> BreathTrendsValidationReport:
        """Run breath-trends validation across a date range.

        Args:
            date_from: Start date (YYYY-MM-DD).
            date_to: End date (YYYY-MM-DD).

        Returns:
            BreathTrendsValidationReport with aggregate and per-session metrics.
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

        session_results: list[BreathTrendsSessionValidation] = []
        for session in sessions:
            try:
                async with work_session(self._injected) as db:
                    self._db = db
                    result = await self._validate_session(session)
                session_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to validate session {session.id}: {e}")
                session_results.append(
                    BreathTrendsSessionValidation(
                        session_id=session.id,
                        date=session.start_time.strftime("%Y-%m-%d"),
                        duration_hours=(session.duration_seconds or 0) / 3600.0,
                        parser_version=session.parser_version or "unknown",
                        skipped_reason="error",
                    )
                )

        aggregate = self._calculate_aggregate(session_results)

        return BreathTrendsValidationReport(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_range_start=date_from,
            date_range_end=date_to,
            aggregate=aggregate,
            sessions=session_results,
        )

    async def _validate_session(
        self, session: models.Session
    ) -> BreathTrendsSessionValidation:
        """Validate a single session against all four device trend channels."""
        duration_hours = (session.duration_seconds or 0) / 3600.0
        date_str = session.start_time.strftime("%Y-%m-%d")
        parser_version = session.parser_version or "unknown"

        def _skip(reason: str) -> BreathTrendsSessionValidation:
            return BreathTrendsSessionValidation(
                session_id=session.id,
                date=date_str,
                duration_hours=duration_hours,
                parser_version=parser_version,
                skipped_reason=reason,
            )

        # 1. Look up latest OK analysis
        breath_svc = BreathService(self._db, self._profile_id)
        status, _algo, ar_id = await breath_svc.latest_analysis_for_session(session.id)
        if status != AnalysisStatus.OK or ar_id is None:
            return _skip("no_analysis")

        # 2. Fetch all leak-valid breaths that have timing offsets
        breath_stmt = (
            select(models.Breath)
            .where(
                models.Breath.analysis_result_id == ar_id,
                models.Breath.leak_valid.is_(True),
                models.Breath.start_offset_s.is_not(None),
                models.Breath.end_offset_s.is_not(None),
            )
            .order_by(models.Breath.breath_number)
        )
        breaths = (await self._db.execute(breath_stmt)).scalars().all()
        if not breaths:
            return _skip("no_valid_breaths")

        starts = np.array([b.start_offset_s for b in breaths], dtype=np.float64)
        ends = np.array([b.end_offset_s for b in breaths], dtype=np.float64)

        # 3. Fetch all channel waveforms in one query, then compare per channel
        waveform_stmt = select(models.Waveform).where(
            models.Waveform.session_id == session.id,
            models.Waveform.waveform_type.in_(_CHANNEL_NAMES),
        )
        waveform_rows = (await self._db.execute(waveform_stmt)).scalars().all()
        waveform_by_type = {row.waveform_type: row for row in waveform_rows}

        channels: dict[str, ChannelComparison] = {}
        for ch in _CHANNEL_NAMES:
            waveform_row = waveform_by_type.get(ch)

            if waveform_row is None:
                channels[ch] = _channel_not_recorded()
                continue

            sample_count = waveform_row.sample_count or 0
            if sample_count == 0:
                channels[ch] = _channel_not_recorded()
                continue

            if waveform_row.data_blob is None:
                channels[ch] = _channel_not_recorded()
                continue

            ts, vals = deserialize_waveform_blob(waveform_row.data_blob, sample_count)

            device_avgs = average_waveform_over_breaths(
                starts,
                ends,
                ts.astype(np.float64),
                vals.astype(np.float64),
            )

            snore_vals = _SNORE_VALUE_FUNCS[ch](list(breaths))
            channels[ch] = _compute_channel_metrics(snore_vals, device_avgs)

        return BreathTrendsSessionValidation(
            session_id=session.id,
            date=date_str,
            duration_hours=duration_hours,
            parser_version=parser_version,
            skipped_reason=None,
            n_breaths=len(breaths),
            channels=channels,
        )

    @staticmethod
    def _calculate_aggregate(
        sessions: list[BreathTrendsSessionValidation],
    ) -> BreathTrendsAggregateMetrics:
        compared = [s for s in sessions if s.skipped_reason is None]
        skipped_no_analysis = sum(
            1 for s in sessions if s.skipped_reason == "no_analysis"
        )
        skipped_no_breaths = sum(
            1 for s in sessions if s.skipped_reason == "no_valid_breaths"
        )

        def _channel_agg(ch: str) -> ChannelAggregateMetrics:
            with_data: list[ChannelComparison] = []
            for s in compared:
                cc = s.channels.get(ch)
                if cc is not None and cc.skipped_reason is None and cc.n_pairs > 0:
                    with_data.append(cc)
            rs = [c.spearman_r for c in with_data if c.spearman_r is not None]
            maes = [
                c.median_abs_error for c in with_data if c.median_abs_error is not None
            ]
            biases = [c.mean_bias for c in with_data if c.mean_bias is not None]
            return ChannelAggregateMetrics(
                sessions_with_data=len(with_data),
                mean_spearman_r=mean_or_none(rs),
                mean_median_abs_error=mean_or_none(maes),
                mean_bias=mean_or_none(biases),
            )

        return BreathTrendsAggregateMetrics(
            total_sessions=len(sessions),
            sessions_compared=len(compared),
            sessions_skipped_no_analysis=skipped_no_analysis,
            sessions_skipped_no_valid_breaths=skipped_no_breaths,
            rr=_channel_agg("rr"),
            tv=_channel_agg("tv"),
            ti=_channel_agg("ti"),
            ie_ratio=_channel_agg("ie_ratio"),
        )
