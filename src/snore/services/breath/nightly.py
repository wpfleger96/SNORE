"""Nightly summaries — per-night and range aggregation."""

from __future__ import annotations

import asyncio
import logging

from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from snore.analysis.queries import latest_analysis_row
from snore.analysis.shared.versioning import (
    RERA_PROXY_ALGO_VERSION,
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisStatus,
    DayAnalysisStatus,
    NullReason,
)
from snore.constants import LEAK_REDLINE_LPM
from snore.constants import FlowLimitationConstants as FLC
from snore.database import models
from snore.services.channel_stats import (
    ChannelHistogram,
    ChannelHistograms,
    PooledHistogram,
    channel_histogram,
    decode_channel_blobs,
    pool_histograms,
    pooled_distribution,
    pooled_share_above,
)
from snore.therapy_hours import effective_session_hours
from snore.utils.db_chunk import iter_id_chunks
from snore.utils.stats import nearest_rank_index

from ._core import SessionClassification, _BreathServiceCore
from .algorithms import count_fl_run_reras_arrays, periodic_breathing_seconds
from .breath_stream import stream_run_reductions
from .dtos import (
    DeviceAmbiguityError,
    DeviceNotOwnedError,
    ExperimentalNightlyRange,
    NightlyAnalysisSummary,
    NightlyDeviceRange,
    NightlyDeviceStats,
    NoSessionsInRangeError,
    SessionCoverage,
)

logger = logging.getLogger(__name__)

# Maximum calendar nights a single nightly range call may span.
# Longer ranges must be paged by the caller (e.g. AppleCrossValidator).
MAX_NIGHTS_PER_CALL = 90

# Device channels the nightly summary decodes; ``channel_histogram`` applies
# each one's floor.
_NIGHT_CHANNELS = ("fl", "snore", "leak")

# Every breath column the experimental night reads, in the row order of the
# arrays ``_reduce_night_run`` reduces.
_NIGHT_BREATH_COLUMNS = (
    models.Breath.leak_valid,
    models.Breath.mid_insp_flattening,
    models.Breath.inspiration_time_s,
    models.Breath.i_e_ratio,
    models.Breath.flow_class,
    models.Breath.flow_confidence,
    models.Breath.is_recovery_breath,
    models.Breath.peak_flow_lpm,
)
(
    _LEAK_VALID,
    _MID_INSP_FLATTENING,
    _INSPIRATION_TIME,
    _IE_RATIO,
    _FLOW_CLASS,
    _FLOW_CONFIDENCE,
    _IS_RECOVERY,
    _PEAK_FLOW,
) = range(8)


def _stat_pair(
    value: float | None, reason: NullReason
) -> tuple[float | None, NullReason | None]:
    """``(value, None)`` when the value was computable, else ``(None, reason)``."""
    return (value, None) if value is not None else (None, reason)


def _high_tail_distribution(
    pooled: PooledHistogram, reason: NullReason
) -> tuple[float | None, float | None, float | None, NullReason | None]:
    """``(median, p95, max, None)`` of pooled samples, or all-``None`` + reason when empty.

    For direct-severity metrics where higher = worse (``pooled_distribution``).
    """
    distribution = pooled_distribution(pooled)
    if distribution is None:
        return None, None, None, reason
    median, p95, maximum = distribution
    return median, p95, maximum, None


def _median(ordered: np.ndarray) -> float:
    """``statistics.median`` of non-empty ascending values, rounding as it does."""
    mid = ordered.size // 2
    if ordered.size % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2


def _low_tail_distribution(
    vals: np.ndarray, reason: NullReason
) -> tuple[float | None, float | None, float | None, NullReason | None]:
    """``(median, p5, min, None)`` over ``vals``, or all-``None`` + reason when empty.

    For inverse-severity metrics where lower = worse.
    """
    if not vals.size:
        return None, None, None, reason
    ordered = np.sort(vals)
    return (
        _median(ordered),
        float(ordered[nearest_rank_index(ordered.size, 0.05)]),
        float(ordered[0]),
        None,
    )


class _NightRunBreaths(BaseModel):
    """One analysis run's breaths, reduced to what its night's stats need."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Leak-valid non-null values, in breath order.
    mid_insp_flattening: np.ndarray
    inspiration_time_s: np.ndarray
    i_e_ratio: np.ndarray
    # Leak-valid rule-matched FL classifications, and those of class >= 4.
    fl_classified: int
    fl_class_ge4: int
    # RERA-proxy events over every breath of the run, leak-valid or not.
    reras: int


def _non_null(values: np.ndarray) -> np.ndarray:
    return values[~np.isnan(values)]


def _concat(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """``arrays`` end to end; empty when there are none."""
    return np.concatenate([np.empty(0), *arrays])


def _reduce_night_run(breaths: np.ndarray) -> _NightRunBreaths:
    """Reduce one run's breaths (``_NIGHT_BREATH_COLUMNS`` rows, in breath order).

    Runs on a worker thread.
    """
    leak_valid = breaths[:, breaths[_LEAK_VALID] == 1.0]
    flow_class = leak_valid[_FLOW_CLASS]
    # Count only rule-matched classifications.  Fallback guesses sit at exactly
    # FL_DEFAULT_CONFIDENCE and would otherwise inflate the FL rate with
    # class-4 guesses.
    rule_matched = flow_class[
        ~np.isnan(flow_class)
        & (leak_valid[_FLOW_CONFIDENCE] > FLC.FL_DEFAULT_CONFIDENCE)
    ]
    return _NightRunBreaths(
        mid_insp_flattening=_non_null(leak_valid[_MID_INSP_FLATTENING]),
        inspiration_time_s=_non_null(leak_valid[_INSPIRATION_TIME]),
        i_e_ratio=_non_null(leak_valid[_IE_RATIO]),
        fl_classified=rule_matched.size,
        fl_class_ge4=int((rule_matched >= 4).sum()),
        # RERA proxy: FL runs ending in a recovery breath.  Scans every breath,
        # not just leak-valid ones, because runs need sequence contiguity.
        reras=count_fl_run_reras_arrays(
            breaths[_FLOW_CLASS], breaths[_IS_RECOVERY] == 1.0, breaths[_PEAK_FLOW]
        ),
    )


def _periodic_breathing_pct(
    ok_day_sessions: Sequence[models.Session],
    pb_seconds_by_session: Mapping[int, float | None],
    *,
    mixed_version: bool,
) -> tuple[float | None, NullReason | None]:
    """Percent of analyzed (OK-session) time spent in periodic breathing.

    Numerator: persisted PB episode seconds; denominator: the durations of
    ``ok_day_sessions`` (the same OK gate the rest of the night uses).
    PB detection having run on >= 1 OK session with zero episodes is a
    genuine 0.0 %.  Mixed algorithm versions → ``ALGO_VERSION_MISMATCH``; no
    parseable PB result or zero eligible duration → ``NOT_AVAILABLE``.
    """
    if mixed_version:
        return None, NullReason.ALGO_VERSION_MISMATCH
    pb_seconds = 0.0
    eligible_seconds = 0.0
    pb_ran = False
    for s in ok_day_sessions:
        eligible_seconds += s.duration_seconds or 0.0
        session_pb = pb_seconds_by_session.get(s.id)
        if session_pb is not None:
            pb_ran = True
            pb_seconds += session_pb
    if not pb_ran or eligible_seconds <= 0:
        return None, NullReason.NOT_AVAILABLE
    return pb_seconds / eligible_seconds * 100.0, None


def _session_pb_seconds(session_id: int, episodes: object) -> float | None:
    """``periodic_breathing_seconds`` for one session; a malformed episode
    list degrades to None (PB not available) instead of failing the night."""
    try:
        return periodic_breathing_seconds(episodes)
    except ValueError:
        logger.warning(
            "Malformed periodic-breathing episodes for session %d; "
            "treating PB as not available",
            session_id,
        )
        return None


def _total_therapy_hours(
    day_sessions: Sequence[models.Session], day_row: Any | None
) -> float:
    """``Day.total_therapy_hours``, falling back to the sessions' summed durations."""
    if day_row is not None and day_row.total_therapy_hours is not None:
        return float(day_row.total_therapy_hours)
    return sum(s.duration_seconds or 0.0 for s in day_sessions) / 3600.0


def _is_compliant(
    day_sessions: Sequence[models.Session], day_row: Any | None, threshold: float
) -> bool:
    """Whether the night's therapy hours reach the compliance ``threshold``."""
    return _total_therapy_hours(day_sessions, day_row) >= threshold


def _pool_night(
    day_sessions: Sequence[models.Session],
    by_session: Mapping[int, ChannelHistogram],
) -> PooledHistogram:
    """Pooled usable samples across all of a night's sessions."""
    return pool_histograms(
        [by_session[sid] for s in day_sessions if (sid := int(s.id)) in by_session]
    )


def _device_night_stats(
    *,
    therapy_date: date,
    device_id: int,
    day_sessions: Sequence[models.Session],
    day_row: Any | None,
    compliance_threshold_hours: float,
    channel_histograms: ChannelHistograms,
    corrupt_leak_sessions: Collection[int] = frozenset(),
) -> NightlyDeviceStats:
    """Build a NightlyDeviceStats from per-session channel histograms. No I/O.

    ``channel_histograms`` holds ``channel_histogram`` results, so the
    ``fl`` and ``leak`` negative sentinels (e.g. −0.01 from digital −1 at
    mask-off) are already dropped.  The stored leak waveform is already L/min.
    A session in ``corrupt_leak_sessions`` nulls the night's leak share: the
    surviving sessions' samples would be a partial value.
    """
    fl = _pool_night(day_sessions, channel_histograms["fl"])
    snore = _pool_night(day_sessions, channel_histograms["snore"])
    leak = _pool_night(day_sessions, channel_histograms["leak"])

    device_flg_median, device_flg_95th, device_flg_max, device_flg_reason = (
        _high_tail_distribution(fl, NullReason.CHANNEL_ABSENT)
    )
    snore_median, snore_95th, _, snore_reason = _high_tail_distribution(
        snore, NullReason.CHANNEL_ABSENT
    )
    leak_corrupt = any(int(s.id) in corrupt_leak_sessions for s in day_sessions)
    leak_fraction = (
        None
        if leak_corrupt
        else pooled_share_above(leak, LEAK_REDLINE_LPM, inclusive=True)
    )
    leak_above_24_pct, leak_above_24_pct_reason = _stat_pair(
        round(100.0 * leak_fraction, 1) if leak_fraction is not None else None,
        NullReason.CHANNEL_CORRUPT if leak_corrupt else NullReason.CHANNEL_ABSENT,
    )

    return NightlyDeviceStats(
        therapy_date=therapy_date,
        device_id=device_id,
        total_therapy_hours=_total_therapy_hours(day_sessions, day_row),
        is_compliant=_is_compliant(day_sessions, day_row, compliance_threshold_hours),
        device_flg_median=device_flg_median,
        device_flg_95th=device_flg_95th,
        device_flg_max=device_flg_max,
        device_flg_reason=device_flg_reason,
        snore_median=snore_median,
        snore_95th=snore_95th,
        snore_pct_time=pooled_share_above(snore, 0.5, inclusive=False),
        snore_reason=snore_reason,
        leak_above_24_pct=leak_above_24_pct,
        leak_above_24_pct_reason=leak_above_24_pct_reason,
    )


class NightlyMixin(_BreathServiceCore):
    """Nightly-summary methods."""

    async def get_analysis_status(
        self,
        session_id: int,
    ) -> tuple[AnalysisStatus, AlgoVersions | None]:
        """(status, versions) for a session's latest AnalysisResult.

        Returns (NOT_RUN, None) if the session is not owned by this profile.
        """
        # Verify profile ownership before querying analysis result
        owned = (
            await self._db.execute(
                select(models.Session.id)
                .join(models.Device, models.Session.device_id == models.Device.id)
                .where(
                    models.Session.id == session_id,
                    models.Device.profile_id == self._profile_id,
                )
            )
        ).scalar_one_or_none()
        if owned is None:
            return AnalysisStatus.NOT_RUN, None

        row = await latest_analysis_row(self._db, session_id)
        if row is None:
            return AnalysisStatus.NOT_RUN, None
        return self._classify_analysis_row(row)

    @staticmethod
    def _build_nightly_summary(
        *,
        therapy_date: date,
        device_id: int,
        day_sessions: Sequence[models.Session],
        day_row: Any | None,
        ar_classification: Mapping[int, SessionClassification],
        runs: Mapping[int, _NightRunBreaths],
        pb_seconds_by_session: Mapping[int, float | None] | None = None,
        usage_hours_by_session: Mapping[int, float | None] | None = None,
    ) -> NightlyAnalysisSummary:
        """Build a NightlyAnalysisSummary from pre-fetched data. No I/O.

        ``runs`` maps an analysis run id to its ``_reduce_night_run`` result;
        a run without breaths is absent and contributes nothing.
        ``pb_seconds_by_session`` maps a session id to its latest run's
        persisted periodic-breathing seconds (None when PB detection never ran
        or its episodes were malformed); only OK sessions are consulted.
        ``usage_hours_by_session`` maps a session id to its
        ``Statistics.usage_hours`` (absent: no Statistics row); it feeds the
        ``rera_index`` mask-on denominator.
        """
        from snore.services.breath_service import BreathService  # noqa: PLC0415

        session_coverages: list[SessionCoverage] = []
        ok_sessions: list[tuple[int, AlgoVersions]] = []
        identities_for_reduce: list[AlgorithmIdentity] = []
        missing_or_stale: list[int] = []
        algo_identity: AlgorithmIdentity | None = None

        for s in day_sessions:
            sid = s.id
            status, algo, _ar_id = ar_classification.get(
                sid, (AnalysisStatus.NOT_RUN, None, None)
            )
            session_coverages.append(
                SessionCoverage(
                    session_id=sid, analysis_status=status, algo_versions=algo
                )
            )
            if status == AnalysisStatus.OK and algo is not None:
                ok_sessions.append((sid, algo))
                identities_for_reduce.append(algo.identity)
                algo_identity = algo.identity
            else:
                missing_or_stale.append(sid)

        eligible = len(day_sessions)
        analyzed = len(ok_sessions)

        day_status = BreathService._reduce_day_status(
            session_coverages, identities_for_reduce
        )
        # Recount, not the device headline: RDI adds a RERA index over SNORE
        # mask-on hours, so both terms must share that denominator.
        day_ahi = day_row.ahi_computed if day_row is not None else None

        ok_session_ids = {sid for sid, _algo in ok_sessions}
        ok_day_sessions = [s for s in day_sessions if s.id in ok_session_ids]
        # PB is a share of recorded time, so it divides by session span; the
        # RERA index below is a rate per mask-on hour like day_ahi.
        periodic_breathing_pct, pb_reason = _periodic_breathing_pct(
            ok_day_sessions,
            pb_seconds_by_session or {},
            mixed_version=day_status == DayAnalysisStatus.MIXED_VERSION,
        )

        if not ok_sessions:
            return NightlyAnalysisSummary(
                therapy_date=therapy_date,
                device_id=device_id,
                day_status=day_status,
                session_coverage=session_coverages,
                eligible_session_count=eligible,
                analyzed_session_count=0,
                missing_or_stale_session_ids=missing_or_stale,
                algorithm_identity=None,
                rera_count=None,
                rera_reason=NullReason.NOT_AVAILABLE,
                primary_mode=None,
                fl_median=None,
                fl_5th=None,
                fl_min=None,
                fl_reason=NullReason.NOT_AVAILABLE,
                fl_class_ge4_pct=None,
                fl_class_ge4_pct_reason=NullReason.NOT_AVAILABLE,
                ti_median_s=None,
                ti_median_reason=NullReason.NOT_AVAILABLE,
                ie_ratio_median=None,
                ie_ratio_reason=NullReason.NOT_AVAILABLE,
                rera_index=None,
                rera_index_reason=NullReason.NOT_AVAILABLE,
                rdi=None,
                rdi_reason=NullReason.NOT_AVAILABLE,
                periodic_breathing_pct=periodic_breathing_pct,
                pb_reason=pb_reason,
            )

        # MIXED_VERSION within a day is handled by _reduce_day_status; under current
        # _latest_analysis_for_session semantics all OK sessions share the current identity.

        modes_seen = {algo.run.primary_mode for _, algo in ok_sessions}
        uniform_primary_mode = next(iter(modes_seen)) if len(modes_seen) == 1 else None

        # The OK sessions' runs, in session order.
        ok_runs = [
            runs[ar_id]
            for sid, _algo in ok_sessions
            if (ar_id := ar_classification[sid][2]) is not None and ar_id in runs
        ]
        fl_vals = _concat(r.mid_insp_flattening for r in ok_runs)
        ti_vals = _concat(r.inspiration_time_s for r in ok_runs)
        ie_vals = _concat(r.i_e_ratio for r in ok_runs)
        fl_class_den = sum(r.fl_classified for r in ok_runs)
        fl_class_ge4_num = sum(r.fl_class_ge4 for r in ok_runs)
        rera_count = sum(r.reras for r in ok_runs)

        fl_median, fl_5th, fl_min, fl_reason = _low_tail_distribution(
            fl_vals, NullReason.NOT_AVAILABLE
        )

        fl_class_ge4_pct, fl_class_ge4_pct_reason = _stat_pair(
            100.0 * fl_class_ge4_num / fl_class_den if fl_class_den > 0 else None,
            NullReason.NOT_AVAILABLE,
        )

        ti_median_s, ti_median_reason = _stat_pair(
            _median(np.sort(ti_vals)) if ti_vals.size else None,
            NullReason.NOT_AVAILABLE,
        )

        ie_ratio_median, ie_ratio_reason = _stat_pair(
            _median(np.sort(ie_vals)) if ie_vals.size else None,
            NullReason.NOT_AVAILABLE,
        )

        # rera_index / rdi arithmetic (finding 7 — moves into service)
        rera_reason: NullReason | None = (
            None if uniform_primary_mode is not None else NullReason.NOT_AVAILABLE
        )
        final_rera_count = rera_count if uniform_primary_mode is not None else None

        rera_index: float | None
        rera_index_reason: NullReason | None
        rdi: float | None
        rdi_reason: NullReason | None

        # RERAs are only counted on OK sessions, so divide by their mask-on
        # hours alone (the Day.total_therapy_hours rule, restricted to analyzed
        # sessions): the whole night's hours would understate a partially
        # analyzed night, and day_ahi shares this per-session basis in rdi.
        usage_hours = usage_hours_by_session or {}
        analyzed_hours = sum(
            effective_session_hours(usage_hours.get(s.id), s.duration_seconds)
            for s in ok_day_sessions
        )
        if final_rera_count is not None:
            if analyzed_hours > 0:
                rera_index = round(final_rera_count / analyzed_hours, 2)
                rera_index_reason = None
            else:
                rera_index = None
                rera_index_reason = NullReason.DURATION_ZERO
        elif rera_reason is not None:
            rera_index = None
            rera_index_reason = rera_reason
        else:
            rera_index = None
            rera_index_reason = None

        if day_ahi is not None and rera_index is not None:
            rdi = round(day_ahi + rera_index, 2)
            rdi_reason = None
        elif rera_index is None and rera_index_reason is not None:
            rdi = None
            rdi_reason = rera_index_reason
        else:
            rdi = None
            rdi_reason = NullReason.NOT_AVAILABLE

        return NightlyAnalysisSummary(
            therapy_date=therapy_date,
            device_id=device_id,
            day_status=day_status,
            session_coverage=session_coverages,
            eligible_session_count=eligible,
            analyzed_session_count=analyzed,
            missing_or_stale_session_ids=missing_or_stale,
            algorithm_identity=algo_identity,
            rera_count=final_rera_count,
            rera_reason=rera_reason,
            rera_proxy_version=(
                RERA_PROXY_ALGO_VERSION if final_rera_count is not None else None
            ),
            primary_mode=uniform_primary_mode,
            fl_median=fl_median,
            fl_5th=fl_5th,
            fl_min=fl_min,
            fl_reason=fl_reason,
            fl_class_ge4_pct=fl_class_ge4_pct,
            fl_class_ge4_pct_reason=fl_class_ge4_pct_reason,
            ti_median_s=ti_median_s,
            ti_median_reason=ti_median_reason,
            ie_ratio_median=ie_ratio_median,
            ie_ratio_reason=ie_ratio_reason,
            rera_index=rera_index,
            rera_index_reason=rera_index_reason,
            rdi=rdi,
            rdi_reason=rdi_reason,
            periodic_breathing_pct=periodic_breathing_pct,
            pb_reason=pb_reason,
        )

    async def get_experimental_nightly_summary(
        self,
        therapy_date: date,
        device_id: int | None = None,
    ) -> NightlyAnalysisSummary:
        """Latest-run analysis fields aggregated across all OK sessions of a day.

        Thin wrapper over ``get_experimental_nightly_range`` for the single-night
        range.  Raises ``NoSessionsInRangeError`` when no owned sessions exist
        and no device_id was given, ``ValueError`` when an explicit owned
        device has no sessions on the date; ``DeviceAmbiguityError`` and
        ``DeviceNotOwnedError`` propagate from device resolution.
        """
        range_summary = await self.get_experimental_nightly_range(
            therapy_date, therapy_date, device_id=device_id
        )
        if not range_summary.nights:
            if device_id is None:
                raise NoSessionsInRangeError(therapy_date, therapy_date)
            raise ValueError(f"No sessions found for date {therapy_date}")
        return range_summary.nights[0]

    async def get_experimental_day_summary(
        self, day: models.Day
    ) -> NightlyAnalysisSummary | None:
        """``get_experimental_nightly_summary`` for a Day row the caller already loaded.

        Reads the Day's sessions by ``day_id`` in one query instead of re-reading
        the Day by date, and builds the night from the enabled ones.  ``None``
        when every session is disabled, a normal user choice; raises
        ``ValueError`` when the Day has no session on a device of this profile.
        """
        # The night reads only ids and durations, so whole Session rows (and
        # their JSON columns) are never loaded.
        rows = (
            await self._db.execute(
                select(
                    models.Session.id,
                    models.Session.enabled,
                    models.Session.duration_seconds,
                )
                .join(models.Device, models.Session.device_id == models.Device.id)
                .where(
                    models.Session.day_id == day.id,
                    models.Device.profile_id == self._profile_id,
                )
                .order_by(models.Session.start_time)
            )
        ).all()
        if not rows:
            raise ValueError(f"No sessions found for date {day.date}")
        sessions = [row for row in rows if row.enabled]
        if not sessions:
            return None
        [night] = await self._experimental_nights(
            day.device_id, {day.date: sessions}, {day.date: day}, [day.date]
        )
        return night

    async def _range_context(
        self,
        date_start: date,
        date_end: date,
        device_id: int | None,
    ) -> tuple[int, dict[date, list[Any]], dict[date, models.Day]]:
        """``(device_id, sessions_by_date, day_by_date)`` for a bounded date range.

        Raises ``ValueError`` for a reversed or over-long range.
        ``DeviceAmbiguityError`` and ``DeviceNotOwnedError`` propagate from device
        resolution; a range with no sessions to auto-select from yields empty
        maps, which the callers turn into an empty summary.
        """
        if date_end < date_start:
            raise ValueError(
                f"date_end ({date_end}) must be >= date_start ({date_start})"
            )

        n_calendar = (date_end - date_start).days + 1

        # Enforce pagination cap (plan Phase 1: "paginated ~30 nights/call").
        if n_calendar > MAX_NIGHTS_PER_CALL:
            raise ValueError(
                f"Date range spans {n_calendar} nights; maximum per call is "
                f"{MAX_NIGHTS_PER_CALL}. Use multiple calls to page over longer ranges."
            )

        # Resolve device ONCE across the full range.
        try:
            return await self._resolve_range_with_days(date_start, date_end, device_id)
        except (DeviceAmbiguityError, DeviceNotOwnedError):
            raise
        except ValueError:
            return device_id or 0, {}, {}

    async def get_nightly_device_summary(
        self,
        date_start: date,
        date_end: date,
        device_id: int | None = None,
        compliance_threshold_hours: float = 4.0,
        night_dates: Collection[date] | None = None,
    ) -> NightlyDeviceRange:
        """Per-night device-channel stats + aggregate compliance.

        ``nights`` holds the nights with sessions whose date is in
        ``night_dates`` (default: every night in the range), and only their
        waveforms are decoded, so a caller showing one page of the range pays
        for that page alone.  ``n_nights`` and the compliance fields always
        cover the whole range: they need Day rows and sessions, not samples.

        The page is the caller's, given as dates, unlike
        ``get_experimental_nightly_range``, which pages its own nights: the
        ``get_nightly_summary`` MCP tool pages ``Day`` rows, which this service
        does not own.  That page can name a date with no night here (another
        device's Day, or a Day whose sessions are all disabled); such a date
        simply builds no night.

        Reads Day rows and waveform channels only, so it answers identically
        whether or not analysis has run.
        """
        resolved_device_id, sessions_by_date, day_by_date = await self._range_context(
            date_start, date_end, device_id
        )
        built_dates = [
            d
            for d in sorted(sessions_by_date)
            if night_dates is None or d in night_dates
        ]
        channel_histograms, corrupt_sessions = await decode_channel_blobs(
            self._db,
            [s.id for d in built_dates for s in sessions_by_date[d]],
            _NIGHT_CHANNELS,
            channel_histogram,
        )
        nights = [
            _device_night_stats(
                therapy_date=therapy_date,
                device_id=resolved_device_id,
                day_sessions=sessions_by_date[therapy_date],
                day_row=day_by_date.get(therapy_date),
                compliance_threshold_hours=compliance_threshold_hours,
                channel_histograms=channel_histograms,
                corrupt_leak_sessions=corrupt_sessions["leak"],
            )
            for therapy_date in built_dates
        ]
        n_calendar = (date_end - date_start).days + 1
        days_compliant = sum(
            1
            for therapy_date, day_sessions in sessions_by_date.items()
            if _is_compliant(
                day_sessions, day_by_date.get(therapy_date), compliance_threshold_hours
            )
        )
        return NightlyDeviceRange(
            date_start=date_start,
            date_end=date_end,
            device_id=resolved_device_id,
            n_nights=len(sessions_by_date),
            compliance_threshold_hours=compliance_threshold_hours,
            n_calendar_nights=n_calendar,
            days_compliant=days_compliant,
            compliance_pct=days_compliant / n_calendar * 100.0,
            nights=nights,
        )

    async def get_experimental_nightly_range(
        self,
        date_start: date,
        date_end: date,
        device_id: int | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> ExperimentalNightlyRange:
        """Per-night analysis summaries (bulk-query path).

        With ``page`` and ``page_size`` the range is paged newest night first
        and only that page's nights are loaded and built; ``n_nights`` still
        counts every night with sessions in the range.  Without them every
        night is returned, oldest first.

        The page is a slice of this service's own nights: the resolved device's
        dates with enabled sessions.  ``get_nightly_device_summary`` instead
        takes the page as dates, because its MCP caller pages ``Day`` rows.
        """
        resolved_device_id, sessions_by_date, day_by_date = await self._range_context(
            date_start, date_end, device_id
        )

        if page is not None and page_size is not None:
            offset = (page - 1) * page_size
            night_dates = sorted(sessions_by_date, reverse=True)[
                offset : offset + page_size
            ]
        else:
            night_dates = sorted(sessions_by_date)

        return ExperimentalNightlyRange(
            date_start=date_start,
            date_end=date_end,
            device_id=resolved_device_id,
            n_nights=len(sessions_by_date),
            nights=await self._experimental_nights(
                resolved_device_id, sessions_by_date, day_by_date, night_dates
            ),
        )

    async def _experimental_nights(
        self,
        resolved_device_id: int,
        sessions_by_date: Mapping[date, Sequence[Any]],
        day_by_date: Mapping[date, models.Day],
        night_dates: Sequence[date],
    ) -> list[NightlyAnalysisSummary]:
        """Analysis summaries of ``night_dates``, in that order (bulk queries)."""
        # Collect the session IDs of the nights being built
        all_sessions: list[Any] = []
        for therapy_date in night_dates:
            all_sessions.extend(sessions_by_date[therapy_date])
        all_session_ids = [s.id for s in all_sessions]

        # Bulk latest-AnalysisResult classification + persisted PB episodes
        # (column-limited; the result JSON is never loaded; no N+1).
        ar_classification, pb_episodes_by_session = await self._latest_analysis_bulk(
            all_session_ids
        )
        # PB seconds are only consumed for OK sessions — skip parsing the rest.
        pb_seconds_by_session = {
            sid: _session_pb_seconds(sid, episodes)
            for sid, episodes in pb_episodes_by_session.items()
            if ar_classification[sid][0] == AnalysisStatus.OK
        }

        # Statistics.usage_hours of OK sessions: the rera_index denominator.
        ok_session_ids = [
            sid
            for sid, (status, _algo, _ar_id) in ar_classification.items()
            if status == AnalysisStatus.OK
        ]
        usage_hours_by_session: dict[int, float | None] = {}
        for sid_chunk in iter_id_chunks(ok_session_ids):
            usage_rows = await self._db.execute(
                select(
                    models.Statistics.session_id, models.Statistics.usage_hours
                ).where(models.Statistics.session_id.in_(sid_chunk))
            )
            usage_hours_by_session.update(
                {row.session_id: row.usage_hours for row in usage_rows}
            )

        ok_ar_ids = [
            ar_id
            for (_status, _algo, ar_id) in ar_classification.values()
            if _status == AnalysisStatus.OK and ar_id is not None
        ]
        # Resolved per run, so a patched ``_reduce_night_run`` takes effect.
        runs = await stream_run_reductions(
            self._db,
            ok_ar_ids,
            _NIGHT_BREATH_COLUMNS,
            lambda breaths: _reduce_night_run(breaths),
        )

        def build_nights() -> list[NightlyAnalysisSummary]:
            return [
                self._build_nightly_summary(
                    therapy_date=therapy_date,
                    device_id=resolved_device_id,
                    day_sessions=sessions_by_date[therapy_date],
                    day_row=day_by_date.get(therapy_date),
                    ar_classification=ar_classification,
                    runs=runs,
                    pb_seconds_by_session=pb_seconds_by_session,
                    usage_hours_by_session=usage_hours_by_session,
                )
                for therapy_date in night_dates
            ]

        return await asyncio.to_thread(build_nights)
