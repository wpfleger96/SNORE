"""Cross-epoch distributions.

``compare_device_epochs`` reports device-channel samples over every enabled
session and needs no analysis.  ``compare_experimental_epochs`` reports experimental
breath-feature distributions over analyzed sessions.  The two share helpers,
not a code path, because they gate sessions differently.
"""

from __future__ import annotations

import asyncio

from collections.abc import Callable, Collection, Iterable, Sequence
from datetime import date
from itertools import pairwise
from typing import Any

import numpy as np

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Row, select

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
from .algorithms import _count_fl_run_reras
from .dtos import (
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
# Every breath column the experimental epoch stats read, including the ones the
# RERA proxy (``iter_fl_run_recoveries``) reads by attribute.
_BREATH_COLUMNS = (
    models.Breath.analysis_result_id,
    models.Breath.leak_valid,
    models.Breath.flow_class,
    models.Breath.flow_confidence,
    models.Breath.is_recovery_breath,
    models.Breath.peak_flow_lpm,
    *_METRIC_COLUMNS.values(),
)
# Breath rows fetched from the cursor at a time.  Only one analysis run's rows
# are held at once, so this bounds the fetch buffer, not the run.
_BREATH_FETCH_ROWS = 4096


def _distrib(n: int, at_rank: Callable[[int], float]) -> dict[str, float | None]:
    """Median, IQR and P95 of ``n`` values read by ascending 0-based rank.

    Exactly what sorting the values gives: ``median`` is ``statistics.median``,
    ``iqr`` is ``sorted[3n // 4] - sorted[n // 4]`` (upper rank clamped) and
    ``p95`` is the nearest rank.  All None when ``n`` is 0.
    """
    if n == 0:
        return {"median": None, "iqr": None, "p95": None}
    mid = n // 2
    median = at_rank(mid) if n % 2 else (at_rank(mid - 1) + at_rank(mid)) / 2
    return {
        "median": median,
        "iqr": at_rank(min(n * 3 // 4, n - 1)) - at_rank(n // 4),
        "p95": at_rank(nearest_rank_index(n, 0.95)),
    }


def _breath_distrib(
    arrays: Sequence[np.ndarray], n_breaths: int, n_nights: int
) -> DistributionStats:
    """Stats over the concatenated ``arrays``; the stable sort matches ``sorted``."""
    ordered = np.sort(np.concatenate(arrays) if arrays else np.empty(0), kind="stable")
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


def _reduce_run_breaths(
    rows: Sequence[_BreathRow], metrics: Collection[DistributionMetric]
) -> _RunBreaths:
    """Reduce one run's breath rows, ordered by breath_number.

    Runs on a worker thread, so ``rows`` are plain rows, never ORM objects.
    """
    leak_valid = [b for b in rows if b.leak_valid is True]
    # Split FL classifications by the same confidence gate nightly uses:
    # rule-matched (flow_confidence > FL_DEFAULT_CONFIDENCE) go in flow_classes
    # so the class>=4 fraction reconciles with nightly fl_class_ge4_pct; low-
    # confidence fallback flatness-triage guesses (confidence exactly at the
    # default) go in fallback_flow_classes so they don't inflate FL rates.  Rows
    # with missing or below-default confidence are excluded from both.
    flow_classes: dict[int, int] = {}
    fallback: dict[int, int] = {}
    for b in leak_valid:
        if b.flow_class is None:
            continue
        if (
            b.flow_confidence is not None
            and b.flow_confidence > FLC.FL_DEFAULT_CONFIDENCE
        ):
            flow_classes[b.flow_class] = flow_classes.get(b.flow_class, 0) + 1
        elif b.flow_confidence == FLC.FL_DEFAULT_CONFIDENCE:
            fallback[b.flow_class] = fallback.get(b.flow_class, 0) + 1
    return _RunBreaths(
        features={
            metric: np.array(
                [v for b in leak_valid if (v := getattr(b, column.key)) is not None],
                dtype=np.float64,
            )
            for metric, column in _METRIC_COLUMNS.items()
            if metric in metrics
        },
        flow_classes=flow_classes,
        fallback_flow_classes=fallback,
        leak_valid_breaths=len(leak_valid),
        # RERA proxy: FL runs ending in a recovery breath.  Scans every breath,
        # not just leak-valid ones, because runs need sequence contiguity.
        reras=_count_fl_run_reras(rows),
    )


def _merge_counts(into: dict[int, int], counts: dict[int, int]) -> None:
    for key, count in counts.items():
        into[key] = into.get(key, 0) + count


def _contributing_ar_ids(resolved_epoch: dict[str, Any]) -> list[int]:
    """Analysis run ids of a resolved epoch's contributing sessions, in order."""
    ar_ids = resolved_epoch["ar_ids"]
    return [
        ar_ids[sid]
        for sid, _ in resolved_epoch["contributing_sessions"]
        if ar_ids.get(sid) is not None
    ]


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
        Raises ValueError for a reversed epoch or conflicting device_ids;
        DeviceAmbiguityError (several owned devices, no device_id) propagates.
        """
        for epoch in epochs:
            if epoch.date_start > epoch.date_end:
                raise ValueError(
                    f"Epoch '{epoch.label}': date_start ({epoch.date_start})"
                    f" must be <= date_end ({epoch.date_end})"
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

    async def _load_run_breaths(
        self, ar_ids: list[int], metrics: Collection[DistributionMetric]
    ) -> dict[int, _RunBreaths]:
        """Each run's reduced breaths by analysis run id; runs without breaths are absent.

        One pass ordered by breath number serves both the leak-valid
        distributions and the RERA proxy.  Rows stream from the cursor and only
        one run's rows are held at a time, each reduced on a worker thread.
        """
        runs: dict[int, _RunBreaths] = {}

        async def reduce_run(rows: list[_BreathRow]) -> None:
            runs[rows[0].analysis_result_id] = await asyncio.to_thread(
                _reduce_run_breaths, rows, metrics
            )

        breath = models.Breath
        for chunk in iter_id_chunks(ar_ids):
            result = await self._db.stream(
                select(*_BREATH_COLUMNS)
                .where(breath.analysis_result_id.in_(chunk))
                .order_by(breath.analysis_result_id, breath.breath_number)
            )
            run_rows: list[_BreathRow] = []
            async for fetched in result.partitions(_BREATH_FETCH_ROWS):
                for row in fetched:
                    if (
                        run_rows
                        and row.analysis_result_id != run_rows[0].analysis_result_id
                    ):
                        await reduce_run(run_rows)
                        run_rows = []
                    run_rows.append(row)
            if run_rows:
                await reduce_run(run_rows)
        return runs

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
        _null_dist = DistributionStats(
            median=None, iqr=None, p95=None, n_breaths=0, n_nights=0
        )

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

        # Each entry: dict with epoch metadata + resolved sessions + RX data
        epoch_resolved: list[dict[str, Any]] = []
        rx_violations: list[EpochRxViolation] = []

        for epoch in epochs:
            _, sessions_by_date = await self._resolve_range(
                epoch.date_start, epoch.date_end, union_resolved_device_id
            )

            # Collect contributing sessions (OK only) and their therapy dates
            contributing_sessions: list[tuple[int, AlgoVersions]] = []
            night_by_ok_session: dict[int, date] = {}
            nights_with_data = 0
            nights_missing_analysis = 0
            ar_ids_for_epoch: dict[int, int | None] = {}
            classification, _ = await self._latest_analysis_bulk(
                [sess.id for sessions in sessions_by_date.values() for sess in sessions]
            )

            for therapy_date, sessions in sessions_by_date.items():
                ok_on_date: list[tuple[int, AlgoVersions]] = []
                for sess in sessions:
                    status, algo, ar_id = classification[sess.id]
                    if status == AnalysisStatus.OK and algo is not None:
                        ok_on_date.append((sess.id, algo))
                        ar_ids_for_epoch[sess.id] = ar_id
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
                    "ar_ids": ar_ids_for_epoch,
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
        # One pass over the breaths of every computable epoch, overlapping ones
        # included.
        runs = await self._load_run_breaths(
            list(
                dict.fromkeys(
                    ar_id
                    for ed in epoch_resolved
                    if ed["null_reason"] is None
                    for ar_id in _contributing_ar_ids(ed)
                )
            ),
            requested,
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
                for ar_id in _contributing_ar_ids(ed)
                if ar_id in runs  # a run without breaths contributes nothing
            ]
            n_lv = sum(run.leak_valid_breaths for run in epoch_runs)
            distributions = {
                metric: _breath_distrib(
                    [run.features[metric] for run in epoch_runs],
                    n_lv,
                    nights_with_data,
                )
                if metric in requested
                else _null_dist
                for metric in DistributionMetric
            }
            fc_dist: dict[int, int] = {}
            fc_dist_fallback: dict[int, int] = {}
            for run in epoch_runs:
                _merge_counts(fc_dist, run.flow_classes)
                _merge_counts(fc_dist_fallback, run.fallback_flow_classes)
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
                    flow_class_distribution=fc_dist,
                    flow_class_distribution_fallback=fc_dist_fallback,
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
