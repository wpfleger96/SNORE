"""get_breath_table — raw/binned breath fetch."""

from __future__ import annotations

import statistics

from datetime import datetime

from sqlalchemy import func, select

from snore.analysis.shared.versioning import AnalysisStatus, NullReason
from snore.database import models

from ._core import _BreathServiceCore
from .dtos import (
    BreathBin,
    BreathPage,
    BreathQueryRange,
    BreathRow,
    CycleType,
    MultiSessionAmbiguityError,
    SessionSummary,
    TriggerCycleApplicability,
    TriggerType,
)


class TableMixin(_BreathServiceCore):
    """Breath-table query methods."""

    # ------------------------------------------------------------------
    # §13 — Public seam methods
    # ------------------------------------------------------------------

    async def get_breath_table(self, query: BreathQueryRange) -> BreathPage:
        """Raw or binned breath fetch.

        Latest analysis run per session selected by (created_at DESC, id DESC).
        analysis_status=NOT_RUN when no AnalysisResult exists;
        STALE_VERSION when engine_versions_json differs from current identity.
        """
        tz_status, tz_name = await self.resolve_timezone()

        # Resolve session_id
        if query.session_id is not None:
            session_id = query.session_id
            # Verify ownership: session must belong to this profile and date,
            # and be enabled (disabled sessions are excluded like _resolve_range)
            session_stmt = (
                select(models.Session)
                .join(models.Device, models.Session.device_id == models.Device.id)
                .where(
                    models.Session.id == session_id,
                    models.Device.profile_id == self._profile_id,
                    models.Session.enabled.is_(True),
                    models.Session.day_id.in_(
                        select(models.Day.id).where(
                            models.Day.date == query.therapy_date
                        )
                    ),
                )
            )
            session_row = (await self._db.execute(session_stmt)).scalars().first()
            if session_row is None:
                raise ValueError(
                    f"session_id {session_id} does not belong to date {query.therapy_date},"
                    " is disabled, or is not owned by this profile"
                )
            device_id = session_row.device_id
            # If caller also specified device_id, verify the session belongs to it
            if query.device_id is not None and device_id != query.device_id:
                raise ValueError(
                    f"session_id {session_id} belongs to device {device_id},"
                    f" not requested device {query.device_id}"
                )
        else:
            # Use _resolve_range for point query; require exactly one session on the date
            resolved_device_id, sessions_by_date = await self._resolve_range(
                query.therapy_date, query.therapy_date, query.device_id
            )
            day_sessions = sessions_by_date.get(query.therapy_date, [])
            if not day_sessions:
                raise ValueError(f"No sessions found for date {query.therapy_date}")
            if len(day_sessions) > 1:
                sessions_list = [
                    SessionSummary(
                        session_id=s.id,
                        start_wall_clock=s.start_time,
                        timezone_status=tz_status,
                        timezone_name=tz_name,
                        duration_seconds=s.duration_seconds or 0.0,
                    )
                    for s in day_sessions
                ]
                raise MultiSessionAmbiguityError(
                    therapy_date=query.therapy_date,
                    device_id=resolved_device_id,
                    sessions=sessions_list,
                )
            session_id = day_sessions[0].id
            device_id = resolved_device_id

        (
            analysis_status,
            algo_versions,
            analysis_result_id,
        ) = await self._latest_analysis_for_session(session_id)

        # Get session start for wall-clock anchoring
        sess_row = (
            (
                await self._db.execute(
                    select(models.Session).where(models.Session.id == session_id)
                )
            )
            .scalars()
            .first()
        )
        session_start: datetime = sess_row.start_time if sess_row else datetime.min

        # No analysis run
        if analysis_result_id is None or analysis_status == AnalysisStatus.NOT_RUN:
            return BreathPage(
                query=query,
                analysis_status=AnalysisStatus.NOT_RUN,
                algo_versions=None,
                null_reason=NullReason.ANALYSIS_NOT_RUN,
                is_binned=query.bin_minutes is not None,
                total_breaths=0,
                page=query.page,
                page_size=query.page_size,
                session_id=session_id,
            )

        # Stale version — return empty page with status
        if analysis_status == AnalysisStatus.STALE_VERSION:
            return BreathPage(
                query=query,
                analysis_status=AnalysisStatus.STALE_VERSION,
                algo_versions=algo_versions,
                null_reason=NullReason.ANALYSIS_STALE,
                is_binned=query.bin_minutes is not None,
                total_breaths=0,
                page=query.page,
                page_size=query.page_size,
                session_id=session_id,
            )

        # Fetch matching breaths
        base_stmt = (
            select(models.Breath)
            .where(
                models.Breath.analysis_result_id == analysis_result_id,
                models.Breath.start_offset_s >= query.offset_start,
                models.Breath.end_offset_s <= query.offset_end,
            )
            .order_by(models.Breath.session_id, models.Breath.breath_number)
        )

        total_result = await self._db.execute(
            select(func.count()).select_from(base_stmt.subquery())
        )
        total_breaths = total_result.scalar_one()

        if query.bin_minutes is None:
            # Raw fetch with pagination
            offset_rows = (query.page - 1) * query.page_size
            paginated = base_stmt.offset(offset_rows).limit(query.page_size)
            breath_rows = (await self._db.execute(paginated)).scalars().all()

            rows = [
                BreathRow(
                    analysis_result_id=b.analysis_result_id,
                    session_id=b.session_id,
                    breath_number=b.breath_number,
                    session_start_wall_clock=session_start,
                    timezone_status=tz_status,
                    timezone_name=tz_name,
                    start_offset_seconds=b.start_offset_s,
                    end_offset_seconds=b.end_offset_s,
                    ti=b.inspiration_time_s,
                    te=b.expiration_time_s,
                    ttot=b.total_time_s,
                    ie_ratio=b.i_e_ratio,
                    duty_cycle=b.duty_cycle,
                    peak_insp_flow=b.peak_flow_lpm,
                    peak_exp_flow=b.peak_exp_flow_lpm,
                    tidal_volume=b.tidal_volume_ml,
                    flatness_index=b.flatness_index,
                    mid_insp_flattening=b.mid_insp_flattening,
                    flow_class=b.flow_class,
                    flow_class_confidence=b.flow_confidence,
                    is_recovery_breath=b.is_recovery_breath,
                    trigger_type=(
                        TriggerType(b.inferred_trigger_type)
                        if b.inferred_trigger_type
                        else None
                    ),
                    cycle_type=(
                        CycleType(b.inferred_cycle_type)
                        if b.inferred_cycle_type
                        else None
                    ),
                    trigger_cycle_confidence=b.trigger_confidence,
                    trigger_cycle_applicability=(
                        TriggerCycleApplicability.VALIDATED
                        if b.trigger_cycle_applicable is True
                        else (
                            TriggerCycleApplicability.UNVALIDATED_DEVICE
                            if b.trigger_cycle_applicable is False
                            else None
                        )
                    ),
                    trigger_cycle_reason=(
                        NullReason(b.trigger_cycle_reason)
                        if b.trigger_cycle_reason
                        else None
                    ),
                    leak_valid=b.leak_valid,
                    leak_valid_reason=(
                        NullReason(b.leak_valid_reason) if b.leak_valid_reason else None
                    ),
                    ramp_active=b.ramp_active,
                    ramp_active_reason=(
                        NullReason(b.ramp_active_reason)
                        if b.ramp_active_reason
                        else None
                    ),
                    mask_off=b.mask_off,
                    mask_off_reason=(
                        NullReason(b.mask_off_reason) if b.mask_off_reason else None
                    ),
                )
                for b in breath_rows
            ]

            return BreathPage(
                query=query,
                analysis_status=analysis_status,
                algo_versions=algo_versions,
                null_reason=None,
                is_binned=False,
                total_breaths=total_breaths,
                page=query.page,
                page_size=query.page_size,
                rows=rows,
                session_id=session_id,
            )
        else:
            # Binned fetch — load all matching breaths then aggregate
            all_breaths = (await self._db.execute(base_stmt)).scalars().all()
            bin_secs = query.bin_minutes * 60.0
            bins: list[BreathBin] = []

            # Group breaths into time bins
            bin_start = query.offset_start
            while bin_start < query.offset_end:
                bin_end = min(bin_start + bin_secs, query.offset_end)
                bin_breaths = [
                    b
                    for b in all_breaths
                    if b.start_offset_s >= bin_start and b.start_offset_s < bin_end
                ]
                if bin_breaths:
                    fi_vals = [
                        b.flatness_index
                        for b in bin_breaths
                        if b.flatness_index is not None
                    ]
                    mif_vals = [
                        b.mid_insp_flattening
                        for b in bin_breaths
                        if b.mid_insp_flattening is not None
                    ]
                    tv_vals = [
                        b.tidal_volume_ml
                        for b in bin_breaths
                        if b.tidal_volume_ml is not None
                    ]
                    ie_vals = [
                        b.i_e_ratio for b in bin_breaths if b.i_e_ratio is not None
                    ]
                    fc_vals = [
                        b.flow_class for b in bin_breaths if b.flow_class is not None
                    ]
                    lv_count = sum(1 for b in bin_breaths if b.leak_valid is True)
                    lv_eligible = sum(
                        1 for b in bin_breaths if b.leak_valid is not None
                    )

                    bins.append(
                        BreathBin(
                            session_start_wall_clock=session_start,
                            timezone_status=tz_status,
                            timezone_name=tz_name,
                            bin_start_offset=bin_start,
                            bin_end_offset=bin_end,
                            breath_count=len(bin_breaths),
                            flatness_index_median=(
                                statistics.median(fi_vals) if fi_vals else None
                            ),
                            mid_insp_flattening_median=(
                                statistics.median(mif_vals) if mif_vals else None
                            ),
                            flow_class_mode=(
                                max(set(fc_vals), key=fc_vals.count)
                                if fc_vals
                                else None
                            ),
                            tidal_volume_median=(
                                statistics.median(tv_vals) if tv_vals else None
                            ),
                            ie_ratio_median=(
                                statistics.median(ie_vals) if ie_vals else None
                            ),
                            leak_valid_fraction=(
                                lv_count / lv_eligible if lv_eligible > 0 else None
                            ),
                            analysis_status=analysis_status,
                        )
                    )
                bin_start = bin_end

            return BreathPage(
                query=query,
                analysis_status=analysis_status,
                algo_versions=algo_versions,
                null_reason=None,
                is_binned=True,
                total_breaths=total_breaths,
                page=1,
                page_size=query.page_size,
                bins=bins,
                session_id=session_id,
            )
