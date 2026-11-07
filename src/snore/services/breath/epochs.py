"""Cross-epoch distributions.

``compare_device_epochs`` reports device-channel samples over every enabled
session and needs no analysis.  ``compare_experimental_epochs`` reports experimental
breath-feature distributions over analyzed sessions.  The two share helpers,
not a code path, because they gate sessions differently.
"""

from __future__ import annotations

import asyncio

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Sequence
from datetime import date
from itertools import pairwise
from typing import Any

import numpy as np

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from snore.analysis.rx_tracker import RX_KEYS, changed_setting_keys
from snore.analysis.shared.versioning import (
    CROSS_VERSION_REFUSAL_KEYS,
    RERA_PROXY_ALGO_VERSION,
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisStatus,
    NullReason,
)
from snore.constants import FlowLimitationConstants as FLC
from snore.database import models
from snore.services.channel_stats import (
    ChannelHistogram,
    channel_histogram,
    decode_channel_blobs,
    pool_histograms,
    pooled_count,
    value_at_rank,
)
from snore.utils.db_chunk import iter_id_chunks
from snore.utils.stats import nearest_rank_index

from ._core import _BreathServiceCore
from .algorithms import count_fl_run_reras_arrays
from .breath_stream import stream_run_reductions
from .dtos import (
    MAX_EPOCH_NIGHTS,
    DeviceCompareEpochsResult,
    DeviceDistributionStats,
    DeviceEpochStats,
    DeviceNotOwnedError,
    DistributionMetric,
    DistributionStats,
    EpochBreathStats,
    EpochRequest,
    EpochRxViolation,
    ExperimentalCompareEpochsResult,
    NoSessionsInRangeError,
)

# Breath column holding each distribution metric.
_METRIC_COLUMNS = {
    DistributionMetric.MID_INSP_FLATTENING: models.Breath.mid_insp_flattening,
    DistributionMetric.FLATNESS_INDEX: models.Breath.flatness_index,
    DistributionMetric.TIDAL_VOLUME_ML: models.Breath.tidal_volume_ml,
    DistributionMetric.IE_RATIO: models.Breath.i_e_ratio,
}
# Every breath column the experimental epoch stats read, in the row order of
# the arrays ``_reduce_run_breaths`` reduces.
_BREATH_COLUMNS = (
    models.Breath.leak_valid,
    models.Breath.flow_class,
    models.Breath.flow_confidence,
    models.Breath.is_recovery_breath,
    models.Breath.peak_flow_lpm,
    *_METRIC_COLUMNS.values(),
)
_LEAK_VALID, _FLOW_CLASS, _FLOW_CONFIDENCE, _IS_RECOVERY, _PEAK_FLOW = range(5)
_METRIC_INDEX = {metric: i for i, metric in enumerate(_METRIC_COLUMNS, start=5)}


def _distrib(n: int, at_rank: Callable[[int], float]) -> dict[str, float | None]:
    """Median, IQR and P95 of ``n`` values read by ascending 0-based rank.

    Exactly what sorting the values gives: ``median`` is ``statistics.median``,
    ``iqr`` is ``sorted[3n // 4] - sorted[n // 4]`` (upper rank clamped) and
    ``p95`` is the nearest rank.  All None when ``n`` is 0.
    """
    if n == 0:
        return {"median": None, "iqr": None, "p95": None}
    return {
        "median": (at_rank((n - 1) // 2) + at_rank(n // 2)) / 2,
        "iqr": at_rank(min(n * 3 // 4, n - 1)) - at_rank(n // 4),
        "p95": at_rank(nearest_rank_index(n, 0.95)),
    }


def _breath_distrib(
    arrays: Sequence[np.ndarray], n_breaths: int, n_nights: int
) -> DistributionStats:
    """Stats over the concatenated ``arrays``."""
    ordered = np.sort(np.concatenate(arrays) if arrays else np.empty(0))
    return DistributionStats(
        **_distrib(ordered.size, lambda rank: float(ordered[rank])),
        n_breaths=n_breaths,
        n_nights=n_nights,
    )


def _device_distrib(
    histograms: dict[int, ChannelHistogram], night_by_session: dict[int, date]
) -> DeviceDistributionStats:
    """Stats over the usable samples of ``night_by_session``'s sessions.

    ``histograms`` maps a session id to its ``channel_histogram``.  Two sessions
    on one therapy date count as one night.
    """
    with_samples = [
        sid for sid in night_by_session if sid in histograms and histograms[sid][0].size
    ]
    pooled = pool_histograms([histograms[sid] for sid in with_samples])
    n = pooled_count(pooled)
    return DeviceDistributionStats(
        **_distrib(n, lambda rank: value_at_rank(pooled, rank)),
        n_samples=n,
        n_nights=len({night_by_session[sid] for sid in with_samples}),
    )


class _RunBreaths(BaseModel):
    """One analysis run's breaths, reduced to what its epoch's stats need."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Leak-valid non-null values of each requested metric, in breath order.
    features: dict[DistributionMetric, np.ndarray]
    # flow_class -> leak-valid breath count, keyed in first-seen order.
    flow_classes: dict[int, int]
    fallback_flow_classes: dict[int, int]
    leak_valid_breaths: int
    # RERA-proxy events over every breath of the run, leak-valid or not.
    reras: int


def _first_seen_counts(values: np.ndarray) -> dict[int, int]:
    """Count of each value, keyed in the order the values first appear."""
    keys, first_index, counts = np.unique(values, return_index=True, return_counts=True)
    return {int(keys[i]): int(counts[i]) for i in np.argsort(first_index)}


def _reduce_run_breaths(
    breaths: np.ndarray, metrics: Collection[DistributionMetric]
) -> _RunBreaths:
    """Reduce one run's breaths (``_BREATH_COLUMNS`` rows, in breath order).

    Runs on a worker thread.
    """
    leak_valid = breaths[:, breaths[_LEAK_VALID] == 1.0]
    flow_class = leak_valid[_FLOW_CLASS]
    confidence = leak_valid[_FLOW_CONFIDENCE]
    classified = ~np.isnan(flow_class)
    # Split FL classifications by the same confidence gate nightly uses:
    # rule-matched (flow_confidence > FL_DEFAULT_CONFIDENCE) go in flow_classes
    # so the class>=4 fraction reconciles with nightly fl_class_ge4_pct; low-
    # confidence fallback flatness-triage guesses (confidence exactly at the
    # default) go in fallback_flow_classes so they don't inflate FL rates.  Rows
    # with missing (NaN) or below-default confidence are excluded from both.
    rule_matched = classified & (confidence > FLC.FL_DEFAULT_CONFIDENCE)
    fallback = classified & (confidence == FLC.FL_DEFAULT_CONFIDENCE)
    features: dict[DistributionMetric, np.ndarray] = {}
    for metric in _METRIC_COLUMNS:
        if metric in metrics:
            values = leak_valid[_METRIC_INDEX[metric]]
            features[metric] = values[~np.isnan(values)]
    return _RunBreaths(
        features=features,
        flow_classes=_first_seen_counts(flow_class[rule_matched]),
        fallback_flow_classes=_first_seen_counts(flow_class[fallback]),
        leak_valid_breaths=leak_valid.shape[1],
        # RERA proxy: FL runs ending in a recovery breath.  Scans every breath,
        # not just leak-valid ones, because runs need sequence contiguity.
        reras=count_fl_run_reras_arrays(
            breaths[_FLOW_CLASS], breaths[_IS_RECOVERY] == 1.0, breaths[_PEAK_FLOW]
        ),
    )


class _EpochBreaths(BaseModel):
    """One epoch's breath distributions and flow-class counts, pooled over runs."""

    distributions: dict[DistributionMetric, DistributionStats]
    flow_classes: dict[int, int]
    fallback_flow_classes: dict[int, int]


def _pool_epoch_runs(
    runs: Sequence[_RunBreaths],
    requested: Collection[DistributionMetric],
    n_nights: int,
) -> _EpochBreaths:
    """Pool an epoch's runs, given in contributing order so the flow-class keys
    keep their first-seen order.  Runs on a worker thread: the sorts are long.
    """
    n_leak_valid = sum(run.leak_valid_breaths for run in runs)
    flow_classes: Counter[int] = Counter()
    fallback: Counter[int] = Counter()
    for run in runs:
        flow_classes.update(run.flow_classes)
        fallback.update(run.fallback_flow_classes)
    return _EpochBreaths(
        distributions={
            metric: _breath_distrib(
                [run.features[metric] for run in runs], n_leak_valid, n_nights
            )
            if metric in requested
            else DistributionStats(
                median=None, iqr=None, p95=None, n_breaths=0, n_nights=0
            )
            for metric in DistributionMetric
        },
        flow_classes=dict(flow_classes),
        fallback_flow_classes=dict(fallback),
    )


def _epoch_rx_violation(
    label: str, dated_snapshots: list[tuple[date, dict[str, str]]]
) -> EpochRxViolation | None:
    """The RX changes between consecutive per-session snapshots, or None if homogeneous.

    An empty snapshot means the settings are unknown, which is not a change: it is
    skipped, so a change is only reported between sessions that both have settings.
    """
    known_snapshots = [(night, rx) for night, rx in dated_snapshots if rx]
    changed_keys: set[str] = set()
    change_dates: list[date] = []
    for (_, prev_rx), (snap_date, snap_rx) in pairwise(known_snapshots):
        diffs = changed_setting_keys(prev_rx, snap_rx)
        if diffs:
            changed_keys |= diffs
            change_dates.append(snap_date)
    if not changed_keys:
        return None
    return EpochRxViolation(
        epoch_label=label,
        changed_keys=sorted(changed_keys),
        change_dates=change_dates,
    )


def _first_known_rx(snapshots: Iterable[dict[str, str]]) -> dict[str, str]:
    """The epoch's reported settings: its first snapshot with any settings known."""
    return next((rx for rx in snapshots if rx), {})


def _null_epoch_stats(
    *,
    label: str,
    date_start: date,
    date_end: date,
    null_reason: NullReason,
    rera_reason: NullReason,
    nights_with_data: int = 0,
    nights_missing_analysis: int = 0,
    rx_settings: dict[str, str] | None = None,
) -> EpochBreathStats:
    """EpochBreathStats with all distributions nulled (refusal/no-data epochs)."""
    null_dist = DistributionStats(
        median=None, iqr=None, p95=None, n_breaths=0, n_nights=0
    )
    return EpochBreathStats(
        label=label,
        date_start=date_start,
        date_end=date_end,
        nights_with_data=nights_with_data,
        nights_missing_analysis=nights_missing_analysis,
        algorithm_identity=None,
        null_reason=null_reason,
        primary_mode=None,
        mid_insp_flattening=null_dist,
        flatness_index=null_dist,
        flow_class_distribution={},
        flow_class_distribution_fallback={},
        tidal_volume_ml=null_dist,
        ie_ratio=null_dist,
        rera_proxy_count=None,
        rera_reason=rera_reason,
        rx_settings=rx_settings or {},
    )


class EpochsMixin(_BreathServiceCore):
    """Epoch-comparison methods."""

    async def _union_device(self, epochs: list[EpochRequest]) -> int | NullReason:
        """Validate the epochs and resolve the one device they all target.

        Returns the device id, or the reason the whole comparison has nothing to
        report: NOT_AVAILABLE for an explicit device this profile does not own,
        NO_DATA_IN_RANGE when no session falls in the union of the epoch ranges.
        Raises ValueError for a reversed epoch, one spanning more than
        MAX_EPOCH_NIGHTS nights, or conflicting device_ids; DeviceAmbiguityError
        (several owned devices, no device_id) propagates.
        """
        for epoch in epochs:
            if epoch.date_start > epoch.date_end:
                raise ValueError(
                    f"Epoch '{epoch.label}': date_start ({epoch.date_start})"
                    f" must be <= date_end ({epoch.date_end})"
                )
            nights = (epoch.date_end - epoch.date_start).days + 1
            if nights > MAX_EPOCH_NIGHTS:
                raise ValueError(
                    f"Epoch '{epoch.label}' spans {nights} nights; an epoch may"
                    f" span at most {MAX_EPOCH_NIGHTS}"
                )

        explicit_device_ids = {e.device_id for e in epochs if e.device_id is not None}
        if len(explicit_device_ids) > 1:
            raise ValueError(
                "All epochs in a comparison must target the same device_id"
            )
        try:
            device_id, sessions_by_date = await self._resolve_range(
                min(e.date_start for e in epochs),
                max(e.date_end for e in epochs),
                explicit_device_ids.pop() if explicit_device_ids else None,
            )
        except DeviceNotOwnedError:
            return NullReason.NOT_AVAILABLE
        except NoSessionsInRangeError:
            return NullReason.NO_DATA_IN_RANGE
        return device_id if sessions_by_date else NullReason.NO_DATA_IN_RANGE

    async def _dated_rx_snapshots(
        self, night_by_session: dict[int, date]
    ) -> list[tuple[date, dict[str, str]]]:
        """``(night, rx)`` per session, in ``night_by_session`` order.

        Each rx holds that session's own prescription settings (not merged per
        day).
        """
        rx_by_session: dict[int, dict[str, str]] = {sid: {} for sid in night_by_session}
        # Chunk only the unbounded session_id list; the RX_KEYS filter is a small
        # constant IN-list kept in every chunk.
        for chunk in iter_id_chunks(list(night_by_session)):
            setting_rows = await self._db.execute(
                select(
                    models.Setting.session_id, models.Setting.key, models.Setting.value
                ).where(
                    models.Setting.session_id.in_(chunk),
                    models.Setting.key.in_(RX_KEYS),
                    models.Setting.value.is_not(None),
                )
            )
            for row in setting_rows:
                if row.value:
                    rx_by_session[row.session_id][row.key] = row.value
        return [(night, rx_by_session[sid]) for sid, night in night_by_session.items()]

    async def compare_device_epochs(
        self, epochs: list[EpochRequest]
    ) -> DeviceCompareEpochsResult:
        """Device flow-limitation and snore distributions across epochs.

        Covers every enabled session and reads no analysis state.  A mid-epoch RX
        change nulls only that epoch (RX_CHANGED_WITHIN_EPOCH), before its waveforms
        are loaded; the top-level null_reason carries it only when EVERY epoch was
        nulled for it.  An epoch without sessions is NO_DATA_IN_RANGE.  A channel
        absent from an epoch is not an epoch failure: that channel's stats are null.
        """
        union_device = await self._union_device(epochs)
        if isinstance(union_device, NullReason):
            return DeviceCompareEpochsResult(
                epochs=[
                    DeviceEpochStats(
                        label=e.label,
                        date_start=e.date_start,
                        date_end=e.date_end,
                        null_reason=union_device,
                    )
                    for e in epochs
                ],
                null_reason=union_device,
            )

        # Resolve every epoch first, so the waveforms are decoded once for all of
        # them, overlapping epochs included.
        epoch_stats: list[DeviceEpochStats] = []
        rx_violations: list[EpochRxViolation] = []
        to_compute: list[tuple[DeviceEpochStats, dict[int, date]]] = []
        for epoch in epochs:
            _, sessions_by_date = await self._resolve_range(
                epoch.date_start, epoch.date_end, union_device
            )
            night_by_session = {
                sess.id: night
                for night, sessions in sessions_by_date.items()
                for sess in sessions
            }
            dated_rx = await self._dated_rx_snapshots(night_by_session)
            stats = DeviceEpochStats(
                label=epoch.label,
                date_start=epoch.date_start,
                date_end=epoch.date_end,
                nights_with_data=len(sessions_by_date),
                rx_settings=_first_known_rx(rx for _, rx in dated_rx),
            )

            rx_violation = _epoch_rx_violation(epoch.label, dated_rx)
            if rx_violation is not None:
                rx_violations.append(rx_violation)
                stats.null_reason = NullReason.RX_CHANGED_WITHIN_EPOCH
            elif not night_by_session:
                stats.null_reason = NullReason.NO_DATA_IN_RANGE
            else:
                to_compute.append((stats, night_by_session))
            epoch_stats.append(stats)

        # channel_histogram drops FLG's negative sentinels; a snore value of 0 is
        # real data.  A corrupt blob counts as an absent channel.
        histograms, _ = await decode_channel_blobs(
            self._db,
            list(dict.fromkeys(sid for _, nights in to_compute for sid in nights)),
            ["fl", "snore"],
            channel_histogram,
        )
        for stats, night_by_session in to_compute:
            stats.device_flg = _device_distrib(histograms["fl"], night_by_session)
            stats.snore_dist = _device_distrib(histograms["snore"], night_by_session)

        all_rx_changed = all(
            e.null_reason == NullReason.RX_CHANGED_WITHIN_EPOCH for e in epoch_stats
        )
        return DeviceCompareEpochsResult(
            epochs=epoch_stats,
            null_reason=NullReason.RX_CHANGED_WITHIN_EPOCH if all_rx_changed else None,
            rx_violations=rx_violations,
        )

    async def compare_experimental_epochs(
        self,
        epochs: list[EpochRequest],
        metrics: list[DistributionMetric] | None = None,
    ) -> ExperimentalCompareEpochsResult:
        """Experimental breath-feature distributions across RxTracker epochs.

        Two-phase design: metadata checks (RX + identity) run across ALL epochs
        BEFORE any breath queries.

        Per-epoch nulling: a mid-epoch RX change nulls only the epoch it occurred
        in (RX_CHANGED_WITHIN_EPOCH); clean epochs in the same comparison still
        compute normally.  The top-level null_reason is RX_CHANGED_WITHIN_EPOCH
        only when EVERY epoch was nulled for an RX change; otherwise None.
        rx_violations is fully populated either way.
        Warning (non-blocking): CROSS_VERSION_REFUSAL_KEYS differ across epochs —
        distributions are still computed; callers should inspect version_warnings.
        Mixed primary modes degrade RERA fields only (PRIMARY_MODE_MISMATCH).

        flow_class_distribution counts rule-matched classifications only (the same
        confidence gate as nightly fl_class_ge4_pct); low-confidence fallback
        guesses stamped at the default confidence are reported separately in
        flow_class_distribution_fallback.  Rows with missing or below-default
        confidence values are excluded from both distributions.
        """
        # -----------------------------------------------------------------------
        # Phase 1: Resolve ONE device for the whole comparison, then per-epoch sessions.
        # -----------------------------------------------------------------------

        union_resolved_device_id = await self._union_device(epochs)
        if isinstance(union_resolved_device_id, NullReason):
            return ExperimentalCompareEpochsResult(
                null_reason=union_resolved_device_id,
                epochs=[
                    _null_epoch_stats(
                        label=e.label,
                        date_start=e.date_start,
                        date_end=e.date_end,
                        null_reason=union_resolved_device_id,
                        rera_reason=union_resolved_device_id,
                    )
                    for e in epochs
                ],
                version_warnings=[],
            )

        sessions_by_epoch: list[dict[date, list[Any]]] = []
        for epoch in epochs:
            _, sessions_by_date = await self._resolve_range(
                epoch.date_start, epoch.date_end, union_resolved_device_id
            )
            sessions_by_epoch.append(sessions_by_date)
        # Classify every session once, overlapping epochs included.
        classification = await self._latest_analysis_classification(
            list(
                dict.fromkeys(
                    sess.id
                    for sessions_by_date in sessions_by_epoch
                    for sessions in sessions_by_date.values()
                    for sess in sessions
                )
            )
        )

        # Each entry: dict with epoch metadata + resolved sessions + RX data
        epoch_resolved: list[dict[str, Any]] = []
        rx_violations: list[EpochRxViolation] = []

        for epoch, sessions_by_date in zip(epochs, sessions_by_epoch, strict=True):
            # Collect contributing sessions (OK only) and their therapy dates
            contributing_sessions: list[tuple[int, AlgoVersions]] = []
            # Their analysis run ids, in the same order.
            contributing_ar_ids: list[int] = []
            night_by_ok_session: dict[int, date] = {}
            nights_with_data = 0
            nights_missing_analysis = 0

            for therapy_date, sessions in sessions_by_date.items():
                ok_on_date: list[tuple[int, AlgoVersions]] = []
                for sess in sessions:
                    status, algo, ar_id = classification[sess.id]
                    if status == AnalysisStatus.OK and algo is not None:
                        ok_on_date.append((sess.id, algo))
                        if ar_id is not None:
                            contributing_ar_ids.append(ar_id)
                        night_by_ok_session[sess.id] = therapy_date

                if ok_on_date:
                    nights_with_data += 1
                    contributing_sessions.extend(ok_on_date)
                else:
                    nights_missing_analysis += 1

            # Within-epoch RX homogeneity check (per-session, not per-day merged);
            # the dates are for change-date tracking.
            session_rx_dated = await self._dated_rx_snapshots(night_by_ok_session)
            rx_violation = _epoch_rx_violation(epoch.label, session_rx_dated)
            if rx_violation is not None:
                rx_violations.append(rx_violation)

            # An epoch's own mid-epoch RX change nulls only that epoch; clean
            # epochs in the same comparison still compute normally (per-epoch
            # nulling, not a batch refusal).
            if rx_violation is not None:
                epoch_null_reason: NullReason | None = (
                    NullReason.RX_CHANGED_WITHIN_EPOCH
                )
            elif contributing_sessions:
                epoch_null_reason = None
            else:
                epoch_null_reason = NullReason.NO_DATA_IN_RANGE

            epoch_resolved.append(
                {
                    "epoch": epoch,
                    "null_reason": epoch_null_reason,
                    "contributing_sessions": contributing_sessions,
                    "all_rx": [rx for _, rx in session_rx_dated],
                    "nights_with_data": nights_with_data,
                    "nights_missing_analysis": nights_missing_analysis,
                    "rx_violation": rx_violation,
                    "contributing_ar_ids": contributing_ar_ids,
                }
            )

        # -----------------------------------------------------------------------
        # Phase 2: Cross-epoch identity check (BEFORE any breath queries)
        # -----------------------------------------------------------------------

        # Gather all identities from ALL contributing sessions across ALL epochs
        all_identities_combined: list[AlgorithmIdentity] = [
            algo.identity
            for ed in epoch_resolved
            for _, algo in ed["contributing_sessions"]
        ]

        # Collect version warnings for CROSS_VERSION_REFUSAL_KEYS fields that differ
        # across contributing sessions.  This is non-blocking: distributions are still
        # computed; callers should inspect version_warnings for compatibility context.
        version_warnings: list[str] = []
        if len(all_identities_combined) > 1:
            cross_keys = CROSS_VERSION_REFUSAL_KEYS
            all_dumps = [id_.model_dump() for id_ in all_identities_combined]
            for k in sorted(cross_keys):
                all_vals_for_key = sorted({d[k] for d in all_dumps})
                if len(all_vals_for_key) > 1:
                    vals_str = ", ".join(f"'{v}'" for v in all_vals_for_key)
                    version_warnings.append(
                        f"algorithm_identity.{k} differs across epochs: {vals_str}"
                    )

        # RX change within an epoch nulls that epoch only (its distributions are
        # meaningless once therapy settings changed mid-range); clean epochs still
        # compute below.  The top-level null_reason reflects a whole-comparison
        # refusal only when EVERY epoch was nulled for the RX change.
        all_epochs_rx_changed = all(
            ed["null_reason"] == NullReason.RX_CHANGED_WITHIN_EPOCH
            for ed in epoch_resolved
        )
        top_null_reason = (
            NullReason.RX_CHANGED_WITHIN_EPOCH if all_epochs_rx_changed else None
        )

        # -----------------------------------------------------------------------
        # Phase 3: Compute distributions (only if all checks passed)
        # -----------------------------------------------------------------------

        requested = set(metrics) if metrics is not None else set(DistributionMetric)
        # One pass ordered by breath number over the breaths of every computable
        # epoch, overlapping ones included, serves both the leak-valid
        # distributions and the RERA proxy.
        runs = await stream_run_reductions(
            self._db,
            list(
                dict.fromkeys(
                    ar_id
                    for ed in epoch_resolved
                    if ed["null_reason"] is None
                    for ar_id in ed["contributing_ar_ids"]
                )
            ),
            _BREATH_COLUMNS,
            lambda breaths: _reduce_run_breaths(breaths, requested),
        )
        epoch_stats: list[EpochBreathStats] = []

        for ed in epoch_resolved:
            epoch = ed["epoch"]
            contributing_sessions = ed["contributing_sessions"]
            nights_with_data = ed["nights_with_data"]
            nights_missing_analysis = ed["nights_missing_analysis"]
            all_rx = ed["all_rx"]
            null_reason_ed: NullReason | None = ed["null_reason"]

            if null_reason_ed is not None or not contributing_sessions:
                resolved_null_reason = null_reason_ed or NullReason.NO_DATA_IN_RANGE
                # rera_reason mirrors the epoch's null cause, except that an
                # epoch whose sessions all lack a current analysis says so.
                epoch_rera_reason = (
                    NullReason.ANALYSIS_NOT_RUN
                    if resolved_null_reason == NullReason.NO_DATA_IN_RANGE
                    and nights_missing_analysis
                    else resolved_null_reason
                )
                epoch_stats.append(
                    _null_epoch_stats(
                        label=epoch.label,
                        date_start=epoch.date_start,
                        date_end=epoch.date_end,
                        nights_with_data=nights_with_data,
                        nights_missing_analysis=nights_missing_analysis,
                        null_reason=resolved_null_reason,
                        rera_reason=epoch_rera_reason,
                        rx_settings=_first_known_rx(all_rx),
                    )
                )
                continue

            all_identities = [algo.identity for _, algo in contributing_sessions]

            # Check primary_mode uniformity for RERA
            all_modes_str = [algo.run.primary_mode for _, algo in contributing_sessions]
            if len(set(all_modes_str)) == 1:
                uniform_primary_mode: str | None = all_modes_str[0]
                rera_reason: NullReason | None = None
            else:
                uniform_primary_mode = None
                rera_reason = NullReason.PRIMARY_MODE_MISMATCH

            # Runs in contributing order, so the flow-class keys keep their
            # first-seen order.
            epoch_runs = [
                runs[ar_id]
                for ar_id in ed["contributing_ar_ids"]
                if ar_id in runs  # a run without breaths contributes nothing
            ]
            pooled = await asyncio.to_thread(
                _pool_epoch_runs, epoch_runs, requested, nights_with_data
            )
            distributions = pooled.distributions
            rera_count = (
                sum(run.reras for run in epoch_runs)
                if uniform_primary_mode is not None
                else None
            )

            epoch_stats.append(
                EpochBreathStats(
                    label=epoch.label,
                    date_start=epoch.date_start,
                    date_end=epoch.date_end,
                    nights_with_data=nights_with_data,
                    nights_missing_analysis=nights_missing_analysis,
                    algorithm_identity=all_identities[0],
                    null_reason=None,
                    primary_mode=uniform_primary_mode,
                    mid_insp_flattening=distributions[
                        DistributionMetric.MID_INSP_FLATTENING
                    ],
                    flatness_index=distributions[DistributionMetric.FLATNESS_INDEX],
                    flow_class_distribution=pooled.flow_classes,
                    flow_class_distribution_fallback=pooled.fallback_flow_classes,
                    tidal_volume_ml=distributions[DistributionMetric.TIDAL_VOLUME_ML],
                    ie_ratio=distributions[DistributionMetric.IE_RATIO],
                    rera_proxy_count=rera_count,
                    rera_reason=rera_reason,
                    rera_proxy_version=(
                        RERA_PROXY_ALGO_VERSION if rera_count is not None else None
                    ),
                    rx_settings=_first_known_rx(all_rx),
                )
            )

        # Future-proofing: warn when rera_proxy_version differs across epochs.
        # Currently impossible (single constant), but guards against future bumps.
        rera_versions = {
            es.rera_proxy_version
            for es in epoch_stats
            if es.rera_proxy_version is not None
        }
        if len(rera_versions) > 1:
            sorted_vers = sorted(rera_versions)
            vals_str = ", ".join(f"'{v}'" for v in sorted_vers)
            version_warnings.append(
                f"rera_proxy_version differs across epochs: {vals_str}"
            )

        return ExperimentalCompareEpochsResult(
            epochs=epoch_stats,
            null_reason=top_null_reason,
            rx_violations=rx_violations,
            version_warnings=version_warnings,
        )
