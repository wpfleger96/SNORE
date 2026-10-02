"""find_windows tool — BreathService.find_windows() adapter.

Timestamp contract (A6):
- Session wall-clock anchors are stored as naive device datetimes.
- Output uses tier-2 (offset-free ISO 8601 + timezone_status
  "unknown" | "user_declared", with timezone_name carrying the profile's
  declared IANA zone) for session wall-clock anchors and tier-3 (float offsets
  from session start) for in-session window positions.  No UTC offsets are
  fabricated.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from snore.constants import RERAProxyConstants
from snore.mcp.errors import ValidationError
from snore.mcp.schemas import (
    FindWindowsResponse,
    WindowRow,
    localize_wall_clock,
    tz_fields,
)
from snore.mcp.tools._capabilities import build_device_capabilities
from snore.mcp.tools._coverage import map_session_coverage
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


async def find_windows(
    db_session: AsyncSession,
    therapy_date: date,
    profile_id: int,
    criterion: str,
    n: int = 5,
    device_id: int | None = None,
    include_unknown_leak: bool = False,
    flattening_threshold: float | None = None,
    min_window_breaths: int = 3,
    context_breaths_before: int = 3,
    context_breaths_after: int = 3,
    context_seconds: float = 120.0,
    min_fl_run_length: int = RERAProxyConstants.MIN_FL_RUN_LENGTH,
    fl_class_threshold: int = RERAProxyConstants.FL_CLASS_THRESHOLD,
    recovery_amplitude_margin: float = RERAProxyConstants.RECOVERY_AMPLITUDE_MARGIN,
) -> FindWindowsResponse:
    """Return N worst breath windows matching a flow-limitation criterion.

    Validates criterion, delegates all ranking/dedup/window construction
    to BreathService.find_windows, then maps the typed DTO to FindWindowsResponse.

    Args:
        db_session: Async database session.
        therapy_date: The therapy date to query.
        profile_id: Profile scope — all ownership checks are in the service.
        criterion: Window selection criterion (one of the WindowCriterion values).
        n: Number of windows to return (1–50; pre-validated by server wrapper).
        device_id: Optional device filter; required when multiple devices share a date.
        include_unknown_leak: Include breaths with unknown leak validity (leak_valid=None).
        flattening_threshold: Maximum mid-insp flattening to anchor a window.
        min_window_breaths: Minimum breaths per formed window.
        context_breaths_before: Context breaths before the anchor breath.
        context_breaths_after: Context breaths after the anchor breath.
        context_seconds: Window half-width in seconds for ca_centered and
            rera_proxy_centered — both extend ±context_seconds from the anchor.
        min_fl_run_length: Minimum FL-class run length (fl_run_ending_in_recovery
            and rera_proxy_centered).
        fl_class_threshold: Minimum flow_class value to count as flow-limited.
        recovery_amplitude_margin: Peak-flow margin over the FL-run mean for the
            self-contained recovery criterion (fl_run_ending_in_recovery and
            rera_proxy_centered).
    """
    from snore.services.breath_service import (  # noqa: PLC0415
        BreathService,
        WindowCriterion,
        WindowCriterionOptions,
    )

    # Validate criterion before touching the DB; service mock must not be called on error.
    try:
        crit = WindowCriterion(criterion)
    except ValueError:
        raise ValidationError(
            f"Unknown criterion {criterion!r}. Valid criteria: "
            "worst_flattening_leak_valid, ca_centered, fl_run_ending_in_recovery, "
            "rera_proxy_centered"
        ) from None

    opts = WindowCriterionOptions(
        include_unknown_leak=include_unknown_leak,
        flattening_threshold=flattening_threshold,
        min_window_breaths=min_window_breaths,
        context_breaths_before=context_breaths_before,
        context_breaths_after=context_breaths_after,
        context_seconds=context_seconds,
        min_fl_run_length=min_fl_run_length,
        fl_class_threshold=fl_class_threshold,
        recovery_amplitude_margin=recovery_amplitude_margin,
    )

    try:
        result = await BreathService(db_session, profile_id).find_windows(
            therapy_date=therapy_date,
            criterion=crit,
            n=n,
            options=opts,
            device_id=device_id,
        )
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)
    # ValueError for criterion-irrelevant options propagates unchanged;
    # the server tool_error_boundary maps it to ToolError.

    # Service uses device_id=0 as a sentinel for "no device resolved" (never emit it).
    resolved_device_id = result.device_id if result.device_id > 0 else None

    coverage = map_session_coverage(result.session_coverage)

    windows = [
        WindowRow(
            criterion=str(w.criterion),
            session_id=w.session_id,
            session_start_wall_clock=localize_wall_clock(
                w.session_start_wall_clock, w.timezone_status, w.timezone_name
            ),
            **tz_fields(w),
            window_start_offset=w.window_start_offset,
            window_end_offset=w.window_end_offset,
            reason_summary=w.reason_summary,
            worst_mid_insp_flattening=w.worst_mid_insp_flattening,
            fl_run_length=w.fl_run_length,
            anchor_event_offset=w.anchor_event_offset,
            analysis_result_id=w.analysis_result_id,
            analysis_status=str(w.analysis_status),
            analysis_reason=str_or_none(w.analysis_reason),
        )
        for w in result.windows
    ]

    caps = None
    if resolved_device_id is not None:
        caps = await build_device_capabilities(
            db_session,
            profile_id,
            resolved_device_id,
            date_start=therapy_date,
            date_end=therapy_date,
        )

    return FindWindowsResponse(
        query_date=therapy_date.isoformat(),
        device_id=resolved_device_id,
        criterion=str(result.criterion),
        day_status=str(result.day_status),
        session_coverage=coverage,
        algorithm_identity=result.algorithm_identity.model_dump(mode="json")
        if result.algorithm_identity is not None
        else None,
        null_reason=str_or_none(result.null_reason),
        primary_mode=result.primary_mode,
        windows=windows,
        device_capabilities=caps,
    )


def register(mcp: FastMCP) -> None:
    from snore.mcp.validation import parse_date, validate_window_count  # noqa: PLC0415

    @mcp.tool()
    @_with_experimental_disclaimer
    @tool_error_boundary
    async def find_windows(
        ctx: Context,
        date: str,
        criterion: str,
        n: int = 5,
        device_id: int | None = None,
        include_unknown_leak: bool = False,
        flattening_threshold: float | None = None,
        min_window_breaths: int = 3,
        context_breaths_before: int = 3,
        context_breaths_after: int = 3,
        context_seconds: float = 120.0,
        min_fl_run_length: int = RERAProxyConstants.MIN_FL_RUN_LENGTH,
        fl_class_threshold: int = RERAProxyConstants.FL_CLASS_THRESHOLD,
        recovery_amplitude_margin: float = RERAProxyConstants.RECOVERY_AMPLITUDE_MARGIN,
    ) -> dict[str, Any]:
        """Find the N worst breath windows matching a flow-limitation criterion for a night.

        Use this tool to locate specific regions in a therapy session worth reviewing in
        detail (e.g. in ``get_breath_table`` or ``render_window``).  Each window
        is a contiguous breath sequence ranked by severity; windows with >50% overlap
        (relative to the shorter) are deduped, keeping the higher-ranked.  Results are
        ordered worst-first, except ``ca_centered``, which is ordered by the CA event's
        offset from its own session's start (smallest first; windows from different
        sessions interleave by offset, not by wall-clock time).

        Requires breath-level analysis results (``get_data_overview`` → ``analysis_run``
        must be true).

        Args:
            date: Session date in YYYY-MM-DD format.
            criterion: Window selection criterion.  One of:
                ``"worst_flattening_leak_valid"`` — windows anchored on the leak-valid
                    breaths with the lowest (most flow-limited) mid-inspiratory flattening,
                    ranked by that single anchor breath's score; use to find FL hotspots.
                ``"ca_centered"`` — context window of ±``context_seconds`` around each
                    CA event, ordered by offset within its session (see above); ``n``
                    keeps the first N in that order.  Works even when the day mixes
                    algorithm versions.
                ``"fl_run_ending_in_recovery"`` — FL runs immediately followed by a
                    recovery breath (the analysis-time recovery flag OR the
                    self-contained v2 criterion: flow class drops to <=2 with peak
                    flow >= ``(1 + recovery_amplitude_margin)`` x the run mean);
                    requires uniform primary_mode across sessions.
                ``"rera_proxy_centered"`` — context window of ±``context_seconds``
                    centered on the recovery breath of each detected RERA-proxy event
                    (FL run → recovery breath); ranked by FL run length descending.
                    Accepts ``context_seconds``, ``min_fl_run_length``,
                    ``fl_class_threshold``, ``recovery_amplitude_margin``.
                    Pass ``session_id`` + ``window_start_offset`` / ``window_end_offset``
                    to ``render_window`` for visual inspection.
                    Requires uniform primary_mode across sessions.
            n: Number of windows to return (1–50, default 5).
            device_id: Filter to a specific device.  Required when multiple devices
                       have data for the same date.
            include_unknown_leak: Include breaths where leak validity is unknown
                (default false).  Only relevant for ``worst_flattening_leak_valid``.
            flattening_threshold: Maximum mid-inspiratory flattening score for a breath
                to anchor a window (lower = more flow-limited).  No cutoff when omitted.
            min_window_breaths: Minimum breaths per window (default 3).
            context_breaths_before: Context breaths before the anchor (default 3).
            context_breaths_after: Context breaths after the anchor (default 3).
            context_seconds: Context seconds on each side of the anchor (default 120.0).
                Only relevant for ``ca_centered`` and ``rera_proxy_centered``.
            min_fl_run_length: Minimum FL-class run length (default 2).
                Only relevant for ``fl_run_ending_in_recovery`` and
                ``rera_proxy_centered``.
            fl_class_threshold: Minimum flow class to count as FL (default 4).
                Only relevant for ``fl_run_ending_in_recovery`` and
                ``rera_proxy_centered``.
            recovery_amplitude_margin: Fractional peak-flow margin over the FL-run
                mean for the self-contained recovery criterion (default 0.20).
                Only relevant for ``fl_run_ending_in_recovery`` and
                ``rera_proxy_centered``.

        Returns:
            FindWindowsResponse.  ``windows`` is ordered worst-first (``ca_centered``:
            smallest CA offset from session start first).
            ``device_id`` is ``null`` when no sessions were found on the date
            (the service-internal sentinel ``0`` is never emitted).
            ``session_coverage`` lists per-session analysis status.
            ``device_capabilities`` describes what the device records.
            ``primary_mode`` is populated for all criteria; ``null`` when
            sessions on the date mix primary modes.

        Refusal semantics (successful responses with empty ``windows`` list):
            ``null_reason: "algo_version_mismatch"`` — the day has sessions analysed
                with different algorithm versions; FL-ranked criteria
                (``worst_flattening_leak_valid``, ``fl_run_ending_in_recovery``,
                ``rera_proxy_centered``) refuse comparison.
                ``ca_centered`` is unaffected — it still works.
            ``null_reason: "primary_mode_mismatch"`` — sessions differ in primary mode;
                ``fl_run_ending_in_recovery`` and ``rera_proxy_centered`` refuse; other
                criteria are unaffected.
            ``null_reason: "analysis_not_run"`` — no analysis results for this date.

        Error conditions:
            - Unknown ``criterion`` value → tool error listing valid criteria.
            - ``n`` outside 1–50 → tool error.
            - Multiple devices on date and no ``device_id`` → tool error listing device IDs.
            - Options irrelevant to the chosen criterion passed with non-default values
              → tool error; omit those options or use their defaults.
        """
        from snore.mcp.tools.windows import find_windows as _impl  # noqa: PLC0415

        therapy_date = parse_date(date, "date")
        validate_window_count(n)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="find_windows",
            therapy_date=therapy_date,
            criterion=criterion,
            n=n,
            device_id=device_id,
            include_unknown_leak=include_unknown_leak,
            flattening_threshold=flattening_threshold,
            min_window_breaths=min_window_breaths,
            context_breaths_before=context_breaths_before,
            context_breaths_after=context_breaths_after,
            context_seconds=context_seconds,
            min_fl_run_length=min_fl_run_length,
            fl_class_threshold=fl_class_threshold,
            recovery_amplitude_margin=recovery_amplitude_margin,
        )
