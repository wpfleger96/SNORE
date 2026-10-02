"""find_windows — criterion-based window discovery."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from datetime import date, datetime
from typing import Any, cast

from sqlalchemy import select

from snore.analysis.shared.versioning import (
    CROSS_VERSION_REFUSAL_KEYS,
    AlgorithmIdentity,
    AnalysisStatus,
    DayAnalysisStatus,
    NullReason,
)
from snore.database import models

from ._core import _BreathServiceCore
from .algorithms import iter_fl_run_recoveries
from .dtos import (
    FindWindowsResult,
    SessionCoverage,
    WindowCriterion,
    WindowCriterionOptions,
    WindowResult,
)


class WindowsMixin(_BreathServiceCore):
    """Window-discovery methods."""

    async def find_windows(
        self,
        therapy_date: date,
        criterion: WindowCriterion,
        n: int,
        options: WindowCriterionOptions | None = None,
        device_id: int | None = None,
    ) -> FindWindowsResult:
        """N windows matching criterion, worst first.

        Windows are built per criterion (worst flattening over leak-valid
        breaths, CA-centered, or FL run ending in recovery), severity-ranked
        worst-first, and deduplicated by overlap so the same span is never
        reported twice.  Options irrelevant to the chosen criterion are
        rejected with ValueError rather than silently ignored.
        """
        opts = options or WindowCriterionOptions()

        # Validate criterion-irrelevant options (see docstring)
        defaults = WindowCriterionOptions()
        if criterion == WindowCriterion.WORST_FLATTENING_LEAK_VALID:
            bad = [
                f
                for f in (
                    "context_seconds",
                    "min_fl_run_length",
                    "fl_class_threshold",
                    "recovery_amplitude_margin",
                )
                if getattr(opts, f) != getattr(defaults, f)
            ]
            if bad:
                raise ValueError(
                    f"Options irrelevant to WORST_FLATTENING_LEAK_VALID: {bad}"
                )
        elif criterion == WindowCriterion.CA_CENTERED:
            bad = [
                f
                for f in (
                    "include_unknown_leak",
                    "flattening_threshold",
                    "min_window_breaths",
                    "context_breaths_before",
                    "context_breaths_after",
                    "min_fl_run_length",
                    "fl_class_threshold",
                    "recovery_amplitude_margin",
                )
                if getattr(opts, f) != getattr(defaults, f)
            ]
            if bad:
                raise ValueError(f"Options irrelevant to CA_CENTERED: {bad}")
        elif criterion == WindowCriterion.FL_RUN_ENDING_IN_RECOVERY:
            bad = [
                f
                for f in (
                    "include_unknown_leak",
                    "flattening_threshold",
                    "min_window_breaths",
                    "context_breaths_before",
                    "context_breaths_after",
                    "context_seconds",
                )
                if getattr(opts, f) != getattr(defaults, f)
            ]
            if bad:
                raise ValueError(
                    f"Options irrelevant to FL_RUN_ENDING_IN_RECOVERY: {bad}"
                )
        elif criterion == WindowCriterion.RERA_PROXY_CENTERED:
            bad = [
                f
                for f in (
                    "include_unknown_leak",
                    "flattening_threshold",
                    "min_window_breaths",
                    "context_breaths_before",
                    "context_breaths_after",
                )
                if getattr(opts, f) != getattr(defaults, f)
            ]
            if bad:
                raise ValueError(f"Options irrelevant to RERA_PROXY_CENTERED: {bad}")

        # Resolve device (raises DeviceAmbiguityError for ≥2 devices, ValueError for 0)
        try:
            resolved_device_id, sessions_by_date = await self._resolve_range(
                therapy_date, therapy_date, device_id
            )
        except ValueError:
            return FindWindowsResult(
                query_date=therapy_date,
                device_id=device_id or 0,
                criterion=criterion,
                day_status=DayAnalysisStatus.NOT_RUN,
                session_coverage=[],
                algorithm_identity=None,
                null_reason=NullReason.ANALYSIS_NOT_RUN,
                primary_mode=None,
                windows=[],
            )
        # DeviceAmbiguityError propagates to caller

        day_sessions = sessions_by_date.get(therapy_date, [])
        if not day_sessions:
            return FindWindowsResult(
                query_date=therapy_date,
                device_id=resolved_device_id,
                criterion=criterion,
                day_status=DayAnalysisStatus.NOT_RUN,
                session_coverage=[],
                algorithm_identity=None,
                null_reason=NullReason.ANALYSIS_NOT_RUN,
                primary_mode=None,
                windows=[],
            )

        # Build per-session analysis status
        session_ids = [s.id for s in day_sessions]
        session_starts = {s.id: s.start_time for s in day_sessions}

        coverage: list[SessionCoverage] = []
        identities: list[AlgorithmIdentity] = []
        primary_modes: list[str] = []
        ar_by_session: dict[int, int | None] = {}
        for sid in session_ids:
            status, algo, ar_id = await self._latest_analysis_for_session(sid)
            ar_by_session[sid] = ar_id
            coverage.append(
                SessionCoverage(
                    session_id=sid, analysis_status=status, algo_versions=algo
                )
            )
            if status == AnalysisStatus.OK and algo is not None:
                identities.append(algo.identity)
                primary_modes.append(algo.run.primary_mode)

        # Determine day_status via the centralized reducer (_reduce_day_status)
        day_status = self._reduce_day_status(coverage, identities)

        # Check identity uniformity for CROSS_VERSION_REFUSAL_KEYS
        uniform_identity: AlgorithmIdentity | None = None
        if identities:
            first_id = identities[0].model_dump()
            cross_keys = CROSS_VERSION_REFUSAL_KEYS
            all_same = all(
                {k: id_.model_dump()[k] for k in cross_keys}
                == {k: first_id[k] for k in cross_keys}
                for id_ in identities[1:]
            )
            if all_same:
                uniform_identity = identities[0]
            else:
                # MIXED_VERSION for FL-ranked criteria
                if criterion != WindowCriterion.CA_CENTERED:
                    return FindWindowsResult(
                        query_date=therapy_date,
                        device_id=resolved_device_id,
                        criterion=criterion,
                        day_status=DayAnalysisStatus.MIXED_VERSION,
                        session_coverage=coverage,
                        algorithm_identity=None,
                        null_reason=NullReason.ALGO_VERSION_MISMATCH,
                        primary_mode=None,
                        windows=[],
                    )

        # FL_RUN_ENDING_IN_RECOVERY: also requires uniform primary_mode
        uniform_primary_mode: str | None = None
        if primary_modes:
            if len(set(primary_modes)) == 1:
                uniform_primary_mode = primary_modes[0]
            elif criterion in (
                WindowCriterion.FL_RUN_ENDING_IN_RECOVERY,
                WindowCriterion.RERA_PROXY_CENTERED,
            ):
                return FindWindowsResult(
                    query_date=therapy_date,
                    device_id=resolved_device_id,
                    criterion=criterion,
                    day_status=day_status,
                    session_coverage=coverage,
                    algorithm_identity=uniform_identity,
                    null_reason=NullReason.PRIMARY_MODE_MISMATCH,
                    primary_mode=None,
                    windows=[],
                )

        result_primary_mode = uniform_primary_mode

        # ar_by_session populated during the coverage loop above
        ar_status_by_session: dict[int, AnalysisStatus] = {
            c.session_id: c.analysis_status for c in coverage
        }

        # Build windows per criterion
        windows: list[WindowResult] = []

        if criterion == WindowCriterion.WORST_FLATTENING_LEAK_VALID:
            windows = await self._find_worst_flattening_windows(
                session_ids=session_ids,
                session_starts=session_starts,
                ar_by_session=ar_by_session,
                ar_status_by_session=ar_status_by_session,
                n=n,
                opts=opts,
            )

        elif criterion == WindowCriterion.CA_CENTERED:
            windows = await self._find_ca_centered_windows(
                therapy_date=therapy_date,
                device_id=resolved_device_id,
                session_starts=session_starts,
                ar_by_session=ar_by_session,
                ar_status_by_session=ar_status_by_session,
                n=n,
                opts=opts,
            )

        elif criterion == WindowCriterion.FL_RUN_ENDING_IN_RECOVERY:
            windows = await self._find_fl_run_windows(
                session_ids=session_ids,
                session_starts=session_starts,
                ar_by_session=ar_by_session,
                ar_status_by_session=ar_status_by_session,
                n=n,
                opts=opts,
            )

        elif criterion == WindowCriterion.RERA_PROXY_CENTERED:
            windows = await self._find_rera_proxy_centered_windows(
                session_ids=session_ids,
                session_starts=session_starts,
                ar_by_session=ar_by_session,
                ar_status_by_session=ar_status_by_session,
                n=n,
                opts=opts,
            )

        return FindWindowsResult(
            query_date=therapy_date,
            device_id=resolved_device_id,
            criterion=criterion,
            day_status=day_status,
            session_coverage=coverage,
            algorithm_identity=uniform_identity,
            null_reason=None,
            primary_mode=result_primary_mode,
            windows=windows,
        )

    async def _iter_session_breaths(
        self,
        session_ids: list[int],
        session_starts: dict[int, datetime],
        ar_by_session: dict[int, int | None],
        ar_status_by_session: dict[int, AnalysisStatus],
    ) -> AsyncIterator[tuple[int, int, AnalysisStatus, Sequence[Any], datetime]]:
        """Yield (sid, ar_id, ar_status, breath_rows, session_start) per OK session.

        Skips sessions without an OK analysis result and sessions with no breath rows.
        """
        for sid in session_ids:
            ar_id = ar_by_session.get(sid)
            ar_status = ar_status_by_session.get(sid, AnalysisStatus.NOT_RUN)
            if ar_id is None or ar_status != AnalysisStatus.OK:
                continue
            breath_rows = (
                (
                    await self._db.execute(
                        select(models.Breath)
                        .where(models.Breath.analysis_result_id == ar_id)
                        .order_by(models.Breath.breath_number)
                    )
                )
                .scalars()
                .all()
            )
            if not breath_rows:
                continue
            yield sid, ar_id, ar_status, breath_rows, session_starts[sid]

    async def _find_worst_flattening_windows(
        self,
        session_ids: list[int],
        session_starts: dict[int, datetime],
        ar_by_session: dict[int, int | None],
        ar_status_by_session: dict[int, AnalysisStatus],
        n: int,
        opts: WindowCriterionOptions,
    ) -> list[WindowResult]:
        """Build WORST_FLATTENING_LEAK_VALID windows per §6 construction rule."""
        tz_status, tz_name = await self.resolve_timezone()
        candidates: list[WindowResult] = []
        async for (
            sid,
            ar_id,
            ar_status,
            breath_rows,
            session_start,
        ) in self._iter_session_breaths(
            session_ids, session_starts, ar_by_session, ar_status_by_session
        ):
            # Filter eligible anchors per §6 step 1
            eligible_indices: list[int] = []
            for i, b in enumerate(breath_rows):
                if b.mid_insp_flattening is None:
                    continue
                if b.leak_valid is True or (
                    opts.include_unknown_leak and b.leak_valid is None
                ):
                    if (
                        opts.flattening_threshold is None
                        or b.mid_insp_flattening <= opts.flattening_threshold
                    ):
                        eligible_indices.append(i)

            # Lower flattening = more flow-limited, so sort ascending (§6 step 2)
            eligible_indices.sort(
                key=lambda i: cast(float, breath_rows[i].mid_insp_flattening)
            )

            for anchor_idx in eligible_indices:
                # §6 step 3: form candidate window
                start_idx = max(0, anchor_idx - opts.context_breaths_before)
                end_idx = min(
                    len(breath_rows) - 1, anchor_idx + opts.context_breaths_after
                )
                window_breaths = breath_rows[start_idx : end_idx + 1]

                if len(window_breaths) < opts.min_window_breaths:
                    continue

                anchor_b = breath_rows[anchor_idx]
                candidates.append(
                    WindowResult(
                        criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
                        session_id=sid,
                        session_start_wall_clock=session_start,
                        timezone_status=tz_status,
                        timezone_name=tz_name,
                        window_start_offset=window_breaths[0].start_offset_s,
                        window_end_offset=window_breaths[-1].end_offset_s,
                        reason_summary=(
                            f"fl_index={anchor_b.mid_insp_flattening:.3f}, "
                            f"{len(window_breaths)} breaths"
                        ),
                        worst_mid_insp_flattening=anchor_b.mid_insp_flattening,
                        fl_run_length=None,
                        anchor_event_offset=None,
                        analysis_result_id=ar_id,
                        analysis_status=ar_status,
                        analysis_reason=None,
                    )
                )

        return self._dedup_and_top_n(
            candidates, n, key=lambda w: -cast(float, w.worst_mid_insp_flattening)
        )

    async def _find_ca_centered_windows(
        self,
        therapy_date: date,
        device_id: int,
        session_starts: dict[int, datetime],
        ar_by_session: dict[int, int | None],
        ar_status_by_session: dict[int, AnalysisStatus],
        n: int,
        opts: WindowCriterionOptions,
    ) -> list[WindowResult]:
        """Build CA_CENTERED windows — anchored on Event rows (CA_CENTERED proceeds
        on any day_status including NOT_RUN, per §6 pass-3 IMPORTANT-5)."""
        stmt = (
            select(models.Event, models.Session)
            .join(models.Session, models.Event.session_id == models.Session.id)
            .join(models.Day, models.Session.day_id == models.Day.id)
            .where(
                models.Day.date == therapy_date,
                models.Session.device_id == device_id,
                models.Event.event_type == "CA",
            )
            .order_by(models.Event.start_time)
        )
        event_rows = (await self._db.execute(stmt)).all()

        tz_status, tz_name = await self.resolve_timezone()
        candidates: list[WindowResult] = []
        for ev_row in event_rows:
            ev = ev_row.Event
            sess = ev_row.Session
            sid = sess.id
            session_start = session_starts.get(sid, sess.start_time)
            # offset from session start
            ev_offset = (ev.start_time - session_start).total_seconds()
            win_start = max(0.0, ev_offset - opts.context_seconds)
            win_end = ev_offset + opts.context_seconds

            ar_id = ar_by_session.get(sid)
            ar_status = ar_status_by_session.get(sid, AnalysisStatus.NOT_RUN)

            candidates.append(
                WindowResult(
                    criterion=WindowCriterion.CA_CENTERED,
                    session_id=sid,
                    session_start_wall_clock=session_start,
                    timezone_status=tz_status,
                    timezone_name=tz_name,
                    window_start_offset=win_start,
                    window_end_offset=win_end,
                    reason_summary=f"CA event at offset {ev_offset:.1f}s",
                    worst_mid_insp_flattening=None,
                    fl_run_length=None,
                    anchor_event_offset=ev_offset,
                    analysis_result_id=ar_id,
                    analysis_status=ar_status,
                    analysis_reason=(
                        NullReason.ANALYSIS_NOT_RUN if ar_id is None else None
                    ),
                )
            )

        return self._dedup_and_top_n(
            candidates, n, key=lambda w: -(w.anchor_event_offset or 0.0)
        )

    async def _find_fl_recovery_windows(
        self,
        criterion: WindowCriterion,
        session_ids: list[int],
        session_starts: dict[int, datetime],
        ar_by_session: dict[int, int | None],
        ar_status_by_session: dict[int, AnalysisStatus],
        n: int,
        opts: WindowCriterionOptions,
        window_for: Callable[
            [Sequence[Any], int, int, int],
            tuple[float, float, str, float | None],
        ],
    ) -> list[WindowResult]:
        """Shared builder for the two FL-run/recovery criteria.

        Scans for FL runs ending in a recovery breath — the same iterator
        backs _count_fl_run_reras, so windows and counts identify exactly
        the same events.  ``window_for(breath_rows, run_start, run_last,
        recovery_idx)`` returns the per-criterion ``(window_start_offset,
        window_end_offset, reason_summary, anchor_event_offset)``.
        """
        tz_status, tz_name = await self.resolve_timezone()
        candidates: list[WindowResult] = []
        async for (
            sid,
            ar_id,
            ar_status,
            breath_rows,
            session_start,
        ) in self._iter_session_breaths(
            session_ids, session_starts, ar_by_session, ar_status_by_session
        ):
            for run_start, run_last, recovery_idx in iter_fl_run_recoveries(
                breath_rows,
                fl_class_threshold=opts.fl_class_threshold,
                min_fl_run_length=opts.min_fl_run_length,
                recovery_amplitude_margin=opts.recovery_amplitude_margin,
            ):
                fl_run = breath_rows[run_start : run_last + 1]
                win_start, win_end, reason, anchor = window_for(
                    breath_rows, run_start, run_last, recovery_idx
                )
                candidates.append(
                    WindowResult(
                        criterion=criterion,
                        session_id=sid,
                        session_start_wall_clock=session_start,
                        timezone_status=tz_status,
                        timezone_name=tz_name,
                        window_start_offset=win_start,
                        window_end_offset=win_end,
                        reason_summary=reason,
                        worst_mid_insp_flattening=min(
                            (
                                b.mid_insp_flattening
                                for b in fl_run
                                if b.mid_insp_flattening is not None
                            ),
                            default=None,
                        ),
                        fl_run_length=len(fl_run),
                        anchor_event_offset=anchor,
                        analysis_result_id=ar_id,
                        analysis_status=ar_status,
                        analysis_reason=None,
                    )
                )

        return self._dedup_and_top_n(candidates, n, key=lambda w: w.fl_run_length or 0)

    async def _find_fl_run_windows(
        self,
        session_ids: list[int],
        session_starts: dict[int, datetime],
        ar_by_session: dict[int, int | None],
        ar_status_by_session: dict[int, AnalysisStatus],
        n: int,
        opts: WindowCriterionOptions,
    ) -> list[WindowResult]:
        """Build FL_RUN_ENDING_IN_RECOVERY windows — RERA-proxy: runs of ≥N consecutive
        FL breaths ending in a recovery breath (analysis-time flag OR the
        self-contained v2 criterion; see iter_fl_run_recoveries)."""

        def window_for(
            breath_rows: Sequence[Any], run_start: int, run_last: int, recovery_idx: int
        ) -> tuple[float, float, str, float | None]:
            full_run = breath_rows[run_start : recovery_idx + 1]
            fl_length = run_last - run_start + 1
            return (
                full_run[0].start_offset_s,
                full_run[-1].end_offset_s,
                f"fl_run={fl_length} breaths, ends in recovery",
                None,
            )

        return await self._find_fl_recovery_windows(
            WindowCriterion.FL_RUN_ENDING_IN_RECOVERY,
            session_ids,
            session_starts,
            ar_by_session,
            ar_status_by_session,
            n,
            opts,
            window_for,
        )

    async def _find_rera_proxy_centered_windows(
        self,
        session_ids: list[int],
        session_starts: dict[int, datetime],
        ar_by_session: dict[int, int | None],
        ar_status_by_session: dict[int, AnalysisStatus],
        n: int,
        opts: WindowCriterionOptions,
    ) -> list[WindowResult]:
        """Build RERA_PROXY_CENTERED windows — context window of ±context_seconds
        centered on the recovery breath of each RERA-proxy event, ranked by run length."""

        def window_for(
            breath_rows: Sequence[Any], run_start: int, run_last: int, recovery_idx: int
        ) -> tuple[float, float, str, float | None]:
            rec_offset = breath_rows[recovery_idx].start_offset_s
            last_end = breath_rows[-1].end_offset_s
            run_start_offset = breath_rows[run_start].start_offset_s
            run_end_offset = breath_rows[run_last].end_offset_s
            return (
                max(0.0, rec_offset - opts.context_seconds),
                min(last_end, rec_offset + opts.context_seconds),
                (
                    f"rera_proxy: fl_run [{run_start_offset:.1f}-{run_end_offset:.1f}]s,"
                    f" recovery at {rec_offset:.1f}s"
                ),
                rec_offset,
            )

        return await self._find_fl_recovery_windows(
            WindowCriterion.RERA_PROXY_CENTERED,
            session_ids,
            session_starts,
            ar_by_session,
            ar_status_by_session,
            n,
            opts,
            window_for,
        )

    @staticmethod
    def _dedup_and_top_n(
        candidates: list[WindowResult],
        n: int,
        key: Callable[[WindowResult], Any],
    ) -> list[WindowResult]:
        """Deduplicate overlapping windows (>50% of shorter), keep worst; return top-N."""
        # Sort by severity descending (largest key first)
        sorted_cands = sorted(candidates, key=key, reverse=True)
        kept: list[WindowResult] = []
        for cand in sorted_cands:
            overlaps = False
            for existing in kept:
                if existing.session_id != cand.session_id:
                    continue
                overlap_start = max(
                    existing.window_start_offset, cand.window_start_offset
                )
                overlap_end = min(existing.window_end_offset, cand.window_end_offset)
                if overlap_end <= overlap_start:
                    continue
                overlap_len = overlap_end - overlap_start
                shorter = min(
                    existing.window_end_offset - existing.window_start_offset,
                    cand.window_end_offset - cand.window_start_offset,
                )
                if shorter > 0 and overlap_len / shorter > 0.5:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(cand)
            if len(kept) >= n:
                break
        return kept
