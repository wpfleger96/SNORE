"""Nightly tools: the device summary and its experimental counterpart.

Timestamp contract (A6): date fields are Python ``date`` objects serialized as
``YYYY-MM-DD`` — no timezone issue since dates have no time component.

``get_nightly_summary`` is device-only: Day and Statistics rows plus
BreathService.get_nightly_device_summary(), so it answers identically whether or
not analysis has run.  ``experimental_get_nightly_metrics`` serves SNORE's own
heuristics from BreathService.get_experimental_nightly_range().  Compliance
denominator is ``n_calendar_nights`` (the full calendar span, not just nights
with data).  All Day and Session queries are profile-scoped through
Device.profile_id to prevent cross-profile data leaks.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.shared.versioning import NullReason
from snore.constants import EVENT_TYPE_RERA, EVENT_TYPE_UNCLASSIFIED_APNEA
from snore.database import models
from snore.services.event_service import device_event_types, reported_count
from snore.services.tidal_volume import tidal_volume_ml

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from snore.services.breath_service import (
        NightlyAnalysisSummary,
        NightlyDeviceRange,
        NightlyDeviceStats,
    )

from snore.mcp.errors import ValidationError
from snore.mcp.schemas import (
    ComplianceFields,
    DeviceCapabilities,
    ExperimentalNightlyResponse,
    ExperimentalNightlyRow,
    NightlyChannelFields,
    NightlyRow,
    NightlySummaryResponse,
)
from snore.mcp.tools._capabilities import _has_analysis, build_device_capabilities
from snore.mcp.tools._helpers import str_or_none
from snore.mcp.tools._scaffold import (
    _scope_and_run,
    _with_experimental_disclaimer,
    tool_error_boundary,
)
from snore.mcp.tools._service_errors import (
    MAPPED_SERVICE_ERRORS,
    raise_mapped_service_error,
)
from snore.metrics import DAY_INDEX_FIELDS

_DEFAULT_PAGE_SIZE = 30
_DEFAULT_COMPLIANCE_THRESHOLD_HOURS = 4.0
MAX_NIGHTLY_RANGE = 90
# Day columns copied onto NightlyRow: the headline indices and their recounts.
_NIGHTLY_INDEX_FIELDS = tuple(
    f"{index}{suffix}" for index in DAY_INDEX_FIELDS for suffix in ("", "_computed")
)


async def get_nightly_summary(
    db_session: AsyncSession,
    start: date,
    end: date,
    profile_id: int,
    device_id: int | None = None,
    page: int = 1,
    page_size: int = _DEFAULT_PAGE_SIZE,
    compliance_threshold_hours: float = _DEFAULT_COMPLIANCE_THRESHOLD_HOURS,
) -> NightlySummaryResponse:
    """Return per-night device summary for a date range.

    ahi/oai/cai/hi are the Day headline: the device-reported daily value when
    trusted, otherwise SNORE's recount; index_source says which.  The recount,
    when computable, is in *_computed.  Channel statistics come from
    BreathService.get_nightly_device_summary(); absent entries are null with
    reason (A2).  Compliance uses n_calendar_nights as denominator.  No analysis
    data is read, except _has_analysis for capability bookkeeping.

    Raises ValidationError when BreathService reports device ownership problems
    (DeviceAmbiguityError, DeviceNotOwnedError).  The server boundary converts
    ValidationError to a ToolError before it reaches the client.
    """
    from snore.services.breath_service import BreathService  # noqa: PLC0415

    # ValueError (end < start or > 90 nights) is pre-validated/rejected upstream.
    try:
        bs_range = await BreathService(
            db_session, profile_id
        ).get_nightly_device_summary(
            start,
            end,
            device_id=device_id,
            compliance_threshold_hours=compliance_threshold_hours,
        )
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)

    # Keyed by device too: without a device_id the Day rows can include another
    # device, whose sessions on that date are all disabled.
    bs_by_night = {
        (night.device_id, night.therapy_date): night for night in bs_range.nights
    }

    # Count total matching days for pagination — scoped via Device.profile_id
    count_q = (
        select(func.count(models.Day.id))
        .join(models.Device, models.Day.device_id == models.Device.id)
        .where(
            models.Day.date >= start,
            models.Day.date <= end,
            models.Device.profile_id == profile_id,
        )
    )
    if device_id is not None:
        count_q = count_q.where(models.Day.device_id == device_id)
    total = (await db_session.execute(count_q)).scalar_one()

    # Fetch the page of Day rows — scoped via Device.profile_id
    day_q = (
        select(models.Day)
        .join(models.Device, models.Day.device_id == models.Device.id)
        .where(
            models.Day.date >= start,
            models.Day.date <= end,
            models.Device.profile_id == profile_id,
        )
        .order_by(models.Day.date.desc())
        .limit(page_size)
        .offset((page - 1) * page_size)
    )
    if device_id is not None:
        day_q = day_q.where(models.Day.device_id == device_id)
    day_rows = (await db_session.execute(day_q)).scalars().all()

    if not day_rows:
        return NightlySummaryResponse(
            nights=[],
            total_nights=total,
            page=page,
            page_size=page_size,
            compliance=_compliance_fields(bs_range) if start != end else None,
        )

    day_ids = [int(d.id) for d in day_rows]

    # For each day get the earliest enabled session (representative for stats).
    # Defense-in-depth: join through Device.profile_id even though day_ids are
    # already profile-scoped by the Day query above.
    session_rows = (
        await db_session.execute(
            select(
                models.Session.id, models.Session.day_id, models.Session.import_source
            )
            .join(models.Device, models.Session.device_id == models.Device.id)
            .where(
                models.Session.day_id.in_(day_ids),
                models.Session.enabled.is_(True),
                models.Device.profile_id == profile_id,
            )
            .order_by(models.Session.day_id, models.Session.start_time)
        )
    ).all()

    day_to_session: dict[int, int] = {}
    import_source_by_session: dict[int, str | None] = {}
    for row_session_id, s_day_id, import_source in session_rows:
        if s_day_id is None:  # excluded by the day_id IN filter
            continue
        day_to_session.setdefault(int(s_day_id), int(row_session_id))
        import_source_by_session[int(row_session_id)] = import_source

    session_ids = list(day_to_session.values())

    # Statistics rows for MV/RR/TV/Ti/I:E — defense-in-depth join through Device.profile_id.
    stats_by_session: dict[int, models.Statistics] = {}
    if session_ids:
        stat_rows = (
            (
                await db_session.execute(
                    select(models.Statistics)
                    .join(
                        models.Session,
                        models.Statistics.session_id == models.Session.id,
                    )
                    .join(
                        models.Device,
                        models.Session.device_id == models.Device.id,
                    )
                    .where(
                        models.Statistics.session_id.in_(session_ids),
                        models.Device.profile_id == profile_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        for stat in stat_rows:
            stats_by_session[int(stat.session_id)] = stat

    reported_by_device = await device_event_types(
        db_session, {d.device_id for d in day_rows}
    )
    nights: list[NightlyRow] = []

    for day in day_rows:
        session_id = day_to_session.get(int(day.id))
        reported = reported_by_device[day.device_id]
        stats = stats_by_session.get(session_id) if session_id else None
        ti_median_s = stats.ti_median if stats else None
        ie_ratio_median = stats.ie_ratio_median if stats else None
        usage_h = day.total_therapy_hours

        nights.append(
            NightlyRow(
                date=day.date,
                usage_hours=round(usage_h, 2) if usage_h is not None else None,
                session_count=day.session_count or 0,
                **_day_indices(day),
                index_source=day.index_source,
                obstructive_apneas=day.obstructive_apneas,
                central_apneas=day.central_apneas,
                hypopneas=day.hypopneas,
                reras=reported_count(day.reras, EVENT_TYPE_RERA, reported),
                unclassified_apneas=reported_count(
                    day.unclassified_apneas, EVENT_TYPE_UNCLASSIFIED_APNEA, reported
                ),
                pressure_median_cmh2o=day.pressure_median,
                pressure_95th_cmh2o=day.pressure_95th,
                epap_median_cmh2o=day.epap_median,
                leak_median_lpm=day.leak_median,
                leak_95th_lpm=day.leak_95th,
                rr_mean_bpm=stats.respiratory_rate_mean if stats else None,
                tv_mean_ml=(
                    tidal_volume_ml(
                        stats.tidal_volume_mean,
                        import_source_by_session.get(int(stats.session_id)),
                    )
                    if stats
                    else None
                ),
                mv_mean_lpm=stats.minute_ventilation_mean if stats else None,
                spo2_mean_pct=day.spo2_mean,
                ti_median_s=ti_median_s,
                ti_median_reason=_absent_reason(ti_median_s, session_id),
                ie_ratio_median=ie_ratio_median,
                ie_ratio_reason=_absent_reason(ie_ratio_median, session_id),
                device_id=day.device_id,
                **_device_stat_fields(
                    bs_by_night.get((day.device_id, day.date))
                ).model_dump(),
            )
        )

    compliance: ComplianceFields | None = None
    if len(day_rows) > 1 or (start != end):
        compliance = _compliance_fields(bs_range)

    return NightlySummaryResponse(
        nights=nights,
        total_nights=total,
        page=page,
        page_size=page_size,
        compliance=compliance,
        device_capabilities=await _range_capabilities(
            db_session, profile_id, bs_range.device_id, start, end
        ),
    )


async def experimental_get_nightly_metrics(
    db_session: AsyncSession,
    start: date,
    end: date,
    profile_id: int,
    device_id: int | None = None,
    page: int = 1,
    page_size: int = _DEFAULT_PAGE_SIZE,
) -> ExperimentalNightlyResponse:
    """Return SNORE's per-night experimental metrics, newest night first.

    Rows are the nights with enabled sessions; a night without a current (OK)
    analysis keeps its row, with null values and the service's reasons.

    Raises ValidationError when BreathService reports device ownership problems
    (DeviceAmbiguityError, DeviceNotOwnedError).  The server boundary converts
    ValidationError to a ToolError before it reaches the client.
    """
    from snore.services.breath_service import BreathService  # noqa: PLC0415

    # ValueError (end < start or > 90 nights) is pre-validated/rejected upstream.
    try:
        bs_range = await BreathService(
            db_session, profile_id
        ).get_experimental_nightly_range(
            start, end, device_id=device_id, page=page, page_size=page_size
        )
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)

    return ExperimentalNightlyResponse(
        nights=[_experimental_row(night) for night in bs_range.nights],
        total_nights=bs_range.n_nights,
        page=page,
        page_size=page_size,
        device_capabilities=await _range_capabilities(
            db_session, profile_id, bs_range.device_id, start, end
        ),
    )


def _rounded(value: float | None, ndigits: int) -> float | None:
    return None if value is None else round(value, ndigits)


def _absent_reason(value: float | None, session_id: int | None) -> str | None:
    """Why a device summary value is null; ``session_id`` is the night's first enabled session."""
    if value is not None:
        return None
    reason = NullReason.NO_SESSIONS if session_id is None else NullReason.CHANNEL_ABSENT
    return reason.value


def _compliance_fields(bs_range: NightlyDeviceRange) -> ComplianceFields:
    return ComplianceFields(
        threshold_hours=bs_range.compliance_threshold_hours,
        days_compliant=bs_range.days_compliant,
        days_total=bs_range.n_calendar_nights,
        compliance_pct=round(bs_range.compliance_pct, 1),
    )


async def _range_capabilities(
    db_session: AsyncSession,
    profile_id: int,
    resolved_device_id: int,
    start: date,
    end: date,
) -> DeviceCapabilities | None:
    """Capabilities of the device BreathService resolved; None when it resolved none."""
    if resolved_device_id <= 0:
        return None
    return await build_device_capabilities(
        db_session,
        profile_id,
        resolved_device_id,
        date_start=start,
        date_end=end,
        analysis_run=await _has_analysis(db_session, profile_id),
    )


def _device_stat_fields(night: NightlyDeviceStats | None) -> NightlyChannelFields:
    """NightlyRow channel fields; ``night`` is None for a Day with no enabled sessions."""
    if night is None:
        no_sessions = NullReason.NO_SESSIONS.value
        return NightlyChannelFields(
            leak_above_24_pct_reason=no_sessions,
            device_flg_reason=no_sessions,
            snore_reason=no_sessions,
        )
    return NightlyChannelFields(
        leak_above_24_pct=night.leak_above_24_pct,
        leak_above_24_pct_reason=str_or_none(night.leak_above_24_pct_reason),
        device_flg_median=_rounded(night.device_flg_median, 4),
        device_flg_95th=_rounded(night.device_flg_95th, 4),
        device_flg_max=_rounded(night.device_flg_max, 4),
        device_flg_reason=str_or_none(night.device_flg_reason),
        snore_median=_rounded(night.snore_median, 4),
        snore_95th=_rounded(night.snore_95th, 4),
        snore_pct_time=_rounded(night.snore_pct_time, 4),
        snore_reason=str_or_none(night.snore_reason),
    )


def _experimental_row(night: NightlyAnalysisSummary) -> ExperimentalNightlyRow:
    fl_reason = str_or_none(night.fl_reason)
    return ExperimentalNightlyRow(
        date=night.therapy_date,
        device_id=night.device_id,
        rera_index=night.rera_index,
        rera_index_reason=str_or_none(night.rera_index_reason),
        rdi=night.rdi,
        rdi_reason=str_or_none(night.rdi_reason),
        fl_median=_rounded(night.fl_median, 4),
        fl_median_reason=fl_reason,
        fl_5th=_rounded(night.fl_5th, 4),
        fl_5th_reason=fl_reason,
        fl_min=_rounded(night.fl_min, 4),
        fl_min_reason=fl_reason,
        fl_class_ge4_pct=_rounded(night.fl_class_ge4_pct, 1),
        fl_class_ge4_pct_reason=str_or_none(night.fl_class_ge4_pct_reason),
        rera_proxy_count=night.rera_count,
        rera_proxy_reason=str_or_none(night.rera_reason),
        rera_proxy_version=night.rera_proxy_version,
        segmented_ti_median_s=_rounded(night.ti_median_s, 3),
        segmented_ti_median_reason=str_or_none(night.ti_median_reason),
        segmented_ie_ratio=_rounded(night.ie_ratio_median, 3),
        segmented_ie_ratio_reason=str_or_none(night.ie_ratio_reason),
        periodic_breathing_pct=_rounded(night.periodic_breathing_pct, 1),
        pb_reason=str_or_none(night.pb_reason),
    )


def _day_indices(day: models.Day) -> dict[str, Any]:
    """Headline + recount indices from a Day row, rounded to 2 dp."""
    return {
        name: None if (value := getattr(day, name)) is None else round(value, 2)
        for name in _NIGHTLY_INDEX_FIELDS
    }


def register(mcp: FastMCP) -> None:
    from snore.mcp.validation import (  # noqa: PLC0415
        parse_date_range,
        validate_compliance_threshold,
        validate_page_args,
    )

    def _parse_nightly_range(start: str, end: str) -> tuple[date, date]:
        start_d, end_d = parse_date_range(start, end)
        n_calendar = (end_d - start_d).days + 1
        if n_calendar > MAX_NIGHTLY_RANGE:
            raise ValidationError(
                f"Date range spans {n_calendar} nights; maximum per call is {MAX_NIGHTLY_RANGE}. "
                "Use multiple calls to page over longer ranges."
            )
        return start_d, end_d

    @mcp.tool()
    @tool_error_boundary
    async def get_nightly_summary(
        ctx: Context,
        start: str,
        end: str,
        device_id: int | None = None,
        page: int = 1,
        page_size: int = 30,
        compliance_threshold_hours: float = _DEFAULT_COMPLIANCE_THRESHOLD_HOURS,
    ) -> dict[str, Any]:
        """Return per-night device data and compliance for a date range.

        Paginated at 30 nights/call (adjustable). Every value comes from
        device-scored events, device summaries and device waveforms, so the
        response is the same whether or not analysis has run.
        ``ahi``/``oai``/``cai``/``hi`` are the device-reported daily value when
        trusted, otherwise SNORE's recount; ``index_source`` (``"device"`` or
        ``"derived"``) says which. The recount, when computable, is in
        ``*_computed``. ``obstructive_apneas``, ``central_apneas``,
        ``hypopneas``, ``reras`` and ``unclassified_apneas`` are device-scored
        event counts. ``reras`` and ``unclassified_apneas`` are null, not zero,
        when the device has never scored one.

        ``leak_above_24_pct`` is the percent of leak samples at or above
        24 L/min, OSCAR's time over the default redline. ``device_flg_*`` and
        ``snore_*`` summarize the device flow-limitation and snore channels.
        When one of these channels is null, its reason is
        ``"channel_absent"`` (the channel was not recorded) or
        ``"no_sessions"`` (the night has no enabled sessions).
        ``leak_above_24_pct_reason`` can also be ``"channel_corrupt"``: a
        session's stored leak data could not be decoded, so the night reports
        no partial value.

        ``rr_mean_bpm``, ``tv_mean_ml``, ``mv_mean_lpm``, ``ti_median_s`` and
        ``ie_ratio_median`` are device summary values of the night's first
        session. ``ti_median_reason`` and ``ie_ratio_reason`` are
        ``"channel_absent"`` when the device did not report the value and
        ``"no_sessions"`` when the night has no enabled sessions. The device
        I:E scale is unconfirmed, so ``ie_ratio_median`` is not comparable to
        ``segmented_ie_ratio``.

        The ``compliance`` block is present whenever ``start != end`` (range
        mode), even when the range contains no night data rows; it is ``null``
        only for single-date requests. ``days_total`` counts CALENDAR nights in
        the requested range — nights without data count as non-compliant.

        SNORE's own flow-limitation, RERA-proxy, segmented-breath and
        periodic-breathing metrics are in ``experimental_get_nightly_metrics``.

        Args:
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.
            device_id: Optional device ID filter.
            page: Page number (1-based). Default 1.
            page_size: Nights per page (must be between 1 and 90). Default 30.
            compliance_threshold_hours: Hours to count as compliant (default 4.0).

        Returns:
            NightlySummaryResponse with nights list, pagination, and compliance block.
        """
        from snore.mcp.tools.summary import (
            get_nightly_summary as _impl,  # noqa: PLC0415
        )

        start_d, end_d = _parse_nightly_range(start, end)
        capped_page_size = validate_page_args(page, page_size)
        validate_compliance_threshold(compliance_threshold_hours)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_nightly_summary",
            start=start_d,
            end=end_d,
            device_id=device_id,
            page=page,
            page_size=capped_page_size,
            compliance_threshold_hours=compliance_threshold_hours,
        )

    @mcp.tool()
    @_with_experimental_disclaimer
    @tool_error_boundary
    async def experimental_get_nightly_metrics(
        ctx: Context,
        start: str,
        end: str,
        device_id: int | None = None,
        page: int = 1,
        page_size: int = 30,
    ) -> dict[str, Any]:
        """Return SNORE's per-night experimental metrics, newest night first.

        Paginated at 30 nights/call (adjustable). Rows are the nights with
        enabled sessions. A night without a current (OK) analysis keeps its row
        with null values and a ``"not_available"`` reason.

        RDI here adds the experimental RERA-proxy index to the night's
        ``ahi_computed``; compare ``rdi`` against ``ahi_computed``, not ``ahi``
        — on device-headline nights ``rdi`` can be lower than ``ahi``.
        ``rera_index`` is RERA-proxy events per mask-on hour of analyzed
        sessions; ``rera_index_reason`` is ``"duration_zero"`` when a RERA
        count exists but those analyzed sessions have zero mask-on hours,
        making the per-hour rate undefined.

        ``fl_class_ge4_pct`` is the percent of leak-valid, rule-matched
        classified breaths with ``flow_class >= 4``; the confidence gate
        excludes fallback guesses. ``fl_median``/``fl_5th``/``fl_min`` are
        per-breath mid-inspiratory flattening where lower = more flow-limited,
        the opposite direction of ``device_flg_*``.

        ``segmented_ti_median_s`` and ``segmented_ie_ratio`` come from SNORE's
        breath segmentation. ``segmented_ie_ratio`` is unitless Ti/Te and is not
        comparable to the device ``ie_ratio_median``.

        ``periodic_breathing_pct`` is the percent of analyzed-session time
        spent in periodic-breathing episodes found by analysis (0.0 when
        detection ran and found none). When null, ``pb_reason`` is one of:
        ``"not_available"`` (no OK session with a PB result, the persisted
        episodes were malformed, or the OK sessions have zero duration);
        ``"algo_version_mismatch"`` (the night's sessions were analyzed with
        different algorithm versions).

        Args:
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.
            device_id: Optional device ID filter.
            page: Page number (1-based). Default 1.
            page_size: Nights per page (must be between 1 and 90). Default 30.

        Returns:
            ExperimentalNightlyResponse with nights list and pagination.
        """
        from snore.mcp.tools.summary import (
            experimental_get_nightly_metrics as _impl,  # noqa: PLC0415
        )

        start_d, end_d = _parse_nightly_range(start, end)
        capped_page_size = validate_page_args(page, page_size)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="experimental_get_nightly_metrics",
            start=start_d,
            end=end_d,
            device_id=device_id,
            page=page,
            page_size=capped_page_size,
        )
