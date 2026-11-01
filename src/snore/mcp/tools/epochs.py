"""Epoch-comparison tools — BreathService adapters.

``compare_epochs`` wraps ``BreathService.compare_device_epochs()`` (device
channels, no analysis needed); ``experimental_compare_epochs`` wraps
``BreathService.compare_epochs()`` (SNORE breath features).

No domain computation in this module: validate input → call service → map types.
Date range parsing and metric-string validation happen here; statistics and
RX-homogeneity checks live in the service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from snore.services.breath_service import EpochRequest
from sqlalchemy.ext.asyncio import AsyncSession

from snore.mcp.errors import ValidationError
from snore.mcp.schemas import (
    CompareEpochsResponse,
    DeviceCompareEpochsResponse,
    EpochDistribution,
    EpochRxViolationRow,
    EpochSpec,
    EpochStats,
)
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
from snore.mcp.validation import parse_date_range


def _map_distribution(d: Any) -> EpochDistribution:
    """Map a DistributionStats DTO to the MCP EpochDistribution schema (field-for-field)."""
    return EpochDistribution(
        median=d.median,
        iqr=d.iqr,
        p95=d.p95,
        n_breaths=d.n_breaths,
        n_nights=d.n_nights,
    )


def _parse_epochs(epochs: list[EpochSpec]) -> list[EpochRequest]:
    """Parse and validate per-epoch date ranges.

    ``parse_date_range`` raises ValidationError with the exact epoch index in the
    parameter name so the LLM can identify which epoch is malformed.
    """
    from snore.services.breath_service import EpochRequest  # noqa: PLC0415

    epoch_requests: list[EpochRequest] = []
    for i, spec in enumerate(epochs):
        start_date, end_date = parse_date_range(
            spec.date_start,
            spec.date_end,
            start_param=f"epochs[{i}].date_start",
            end_param=f"epochs[{i}].date_end",
        )
        epoch_requests.append(
            EpochRequest(
                label=spec.label,
                date_start=start_date,
                date_end=end_date,
                device_id=spec.device_id,
            )
        )
    return epoch_requests


async def compare_epochs(
    db_session: AsyncSession,
    profile_id: int,
    epochs: list[EpochSpec],
) -> DeviceCompareEpochsResponse:
    """Compare device flow-limitation and snore distributions across epochs.

    Validates per-epoch date ranges, then delegates to
    ``BreathService.compare_device_epochs()``.

    Service ``ValueError`` ("All epochs in a comparison must target the same
    device_id") propagates to the MCP error boundary unchanged — it is a caller
    error that belongs on the wire as a ToolError.
    """
    from snore.services.breath_service import BreathService  # noqa: PLC0415

    epoch_requests = _parse_epochs(epochs)

    # DeviceAmbiguityError propagates from the service (multi-device profile, no
    # device_id given); an explicit foreign device is handled INSIDE the service
    # and returned as not_available epochs.
    try:
        result = await BreathService(db_session, profile_id).compare_device_epochs(
            epoch_requests
        )
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)

    return DeviceCompareEpochsResponse.model_validate(result.model_dump(mode="json"))


async def experimental_compare_epochs(
    db_session: AsyncSession,
    profile_id: int,
    epochs: list[EpochSpec],
    metrics: list[str] | None = None,
) -> CompareEpochsResponse:
    """Compare breath-feature distributions across therapy settings epochs.

    Validates per-epoch date ranges and metric strings, then delegates to
    ``BreathService.compare_epochs()``.  Maps the typed result DTO to the
    ``CompareEpochsResponse`` schema: dates to isoformat strings, NullReason/enum
    values to plain strings, and flow_class_distribution int keys to string keys.

    Service ``ValueError`` ("All epochs in a comparison must target the same
    device_id") propagates to the MCP error boundary unchanged — it is a caller
    error that belongs on the wire as a ToolError.
    """
    from snore.services.breath_service import (  # noqa: PLC0415
        BreathService,
        DistributionMetric,
    )

    epoch_requests = _parse_epochs(epochs)

    # Validate metric strings and map to DistributionMetric enum values.
    parsed_metrics: list[DistributionMetric] | None = None
    if metrics is not None:
        valid_names = {m.value for m in DistributionMetric}
        for m in metrics:
            if m not in valid_names:
                raise ValidationError(
                    f"Unknown metric {m!r}. Valid metrics: "
                    "mid_insp_flattening, flatness_index, tidal_volume_ml, ie_ratio"
                )
        parsed_metrics = [DistributionMetric(m) for m in metrics]

    # Delegate to service.  DeviceAmbiguityError propagates from the service
    # (multi-device profile, no device_id given); DeviceNotOwnedError for an
    # explicit foreign device is handled INSIDE the service and returned as
    # not_available epochs — no exception reaches here in that case.
    try:
        result = await BreathService(db_session, profile_id).compare_epochs(
            epoch_requests, metrics=parsed_metrics
        )
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)

    # Map CompareEpochsResult → CompareEpochsResponse.
    # Enum values → str; dates → isoformat; flow_class int keys → str keys.
    epoch_stats: list[EpochStats] = [
        EpochStats(
            label=s.label,
            date_start=s.date_start.isoformat(),
            date_end=s.date_end.isoformat(),
            nights_with_data=s.nights_with_data,
            nights_missing_analysis=s.nights_missing_analysis,
            algorithm_identity=s.algorithm_identity.model_dump(mode="json")
            if s.algorithm_identity is not None
            else None,
            null_reason=str_or_none(s.null_reason),
            primary_mode=s.primary_mode,
            mid_insp_flattening=_map_distribution(s.mid_insp_flattening),
            flatness_index=_map_distribution(s.flatness_index),
            flow_class_distribution={
                str(k): v for k, v in s.flow_class_distribution.items()
            },
            flow_class_distribution_fallback={
                str(k): v for k, v in s.flow_class_distribution_fallback.items()
            },
            tidal_volume_ml=_map_distribution(s.tidal_volume_ml),
            ie_ratio=_map_distribution(s.ie_ratio),
            rera_proxy_count=s.rera_proxy_count,
            rera_reason=str_or_none(s.rera_reason),
            rera_proxy_version=s.rera_proxy_version,
            rx_settings=s.rx_settings,
        )
        for s in result.epochs
    ]

    return CompareEpochsResponse(
        epochs=epoch_stats,
        null_reason=str_or_none(result.null_reason),
        rx_violations=[
            EpochRxViolationRow(
                epoch_label=v.epoch_label,
                changed_keys=v.changed_keys,
                change_dates=[d.isoformat() for d in v.change_dates],
            )
            for v in result.rx_violations
        ],
        version_warnings=result.version_warnings,
    )


def register(mcp: FastMCP) -> None:
    from snore.mcp.validation import validate_epoch_count  # noqa: PLC0415

    @mcp.tool()
    @tool_error_boundary
    async def compare_epochs(
        ctx: Context,
        epochs: list[EpochSpec],
    ) -> dict[str, Any]:
        """Compare device flow-limitation and snore distributions across up to 6 therapy settings epochs.

        Use this tool to see whether a settings change moved the device's own
        flow-limitation and snore channels.  Each epoch is a labelled date range; the
        tool returns descriptive statistics (median, IQR, P95) of the device's samples
        for each epoch.  No analysis is needed: every enabled session in the range
        contributes.

        Call ``get_settings_timeline`` first to identify meaningful epoch boundaries,
        then ``compare_epochs`` to quantify the difference.

        Args:
            epochs: List of 1–6 epoch specs.  Each spec has:
                ``label`` — human-readable epoch name (appears in response).
                ``date_start`` — epoch start in YYYY-MM-DD format (inclusive).
                ``date_end`` — epoch end in YYYY-MM-DD format (inclusive).
                ``device_id`` — optional; all epochs must target the same device.

        Returns:
            DeviceCompareEpochsResponse.  Each entry in ``epochs`` carries
            ``device_flg`` (device-reported flow limitation, 0–1 unitless, negative
            sentinels excluded) and ``snore_dist`` (device-reported snore level, 0–5
            unitless, zeros included).  In both, ``n_samples`` counts valid samples
            and ``n_nights`` counts calendar nights with at least one valid sample, so
            a two-session night counts once.  A channel absent from an epoch has null
            stats with ``n_samples: 0``; the epoch's ``null_reason`` stays ``null``.
            ``nights_with_data`` counts nights with at least one enabled session.
            ``rx_settings`` holds representative therapy settings observed for the epoch.
            ``rx_violations`` lists any therapy-settings changes detected within an
            epoch's date range; callers should split affected epochs at those dates.
            Sessions with no recorded settings are skipped: a change is only reported
            between sessions that both have settings.

        Refusal semantics:
            Mid-epoch RX change nulls only the affected epoch: that epoch's
                ``null_reason`` becomes ``"rx_changed_within_epoch"`` while clean epochs
                in the same call keep their computed distributions.  The top-level
                ``null_reason`` is ``"rx_changed_within_epoch"`` only when EVERY epoch was
                nulled for an RX change; otherwise it stays ``null``.  ``rx_violations``
                always lists the affected epoch label, changed keys, and change dates so
                the caller can split those ranges.
            No sessions in an epoch: that epoch's ``null_reason`` is
                ``"no_data_in_range"``.  When no session falls anywhere in the combined
                date range, the top-level ``null_reason`` is ``"no_data_in_range"`` too.
            Device not owned by the active profile: the service catches this internally
                and returns a success response with the top-level and every epoch's
                ``null_reason: "not_available"``.

        Error conditions:
            - Epochs list empty or >6 entries → tool error.
            - Multiple device IDs across epoch specs → tool error.
            - Multiple devices on the date range and no ``device_id`` → tool error
              listing device IDs so the caller can re-issue with ``device_id``.
        """
        from snore.mcp.tools.epochs import compare_epochs as _impl  # noqa: PLC0415

        validate_epoch_count(len(epochs))
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="compare_epochs",
            epochs=epochs,
        )

    @mcp.tool()
    @_with_experimental_disclaimer
    @tool_error_boundary
    async def experimental_compare_epochs(
        ctx: Context,
        epochs: list[EpochSpec],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare breath-feature distributions across up to 6 therapy settings epochs.

        Use this tool to detect whether a settings change improved or worsened
        flow-limitation metrics.  Each epoch is a labelled date range; the tool returns
        descriptive statistics (median, IQR, P95) of leak-valid breaths for each epoch.
        Only nights with OK analysis results contribute to an epoch's distributions.

        Call ``get_settings_timeline`` first to identify meaningful epoch boundaries,
        then ``experimental_compare_epochs`` to quantify the difference.

        Requires breath-level analysis results (``get_data_overview`` → ``analysis_run``
        must be true).  For the device's own flow-limitation and snore channels, use
        ``compare_epochs``, which needs no analysis.

        Args:
            epochs: List of 1–6 epoch specs.  Each spec has:
                ``label`` — human-readable epoch name (appears in response).
                ``date_start`` — epoch start in YYYY-MM-DD format (inclusive).
                ``date_end`` — epoch end in YYYY-MM-DD format (inclusive).
                ``device_id`` — optional; all epochs must target the same device.
            metrics: Optional subset of distribution metrics to compute.  Defaults
                to all four: ``"mid_insp_flattening"``, ``"flatness_index"``,
                ``"tidal_volume_ml"``, ``"ie_ratio"``.

        Returns:
            CompareEpochsResponse.  Each entry in ``epochs`` contains distributions
            and coverage metadata (``nights_with_data``, ``nights_missing_analysis``).
            ``flow_class_distribution`` counts rule-matched FL classifications only
            (breaths whose classifier confidence clears the default gate); its
            class>=4 fraction reconciles with ``experimental_get_nightly_metrics``'s
            ``fl_class_ge4_pct``.  ``flow_class_distribution_fallback`` counts the
            low-confidence fallback guesses — breaths triaged on flatness alone when
            no shape rule matched, stamped at exactly the default confidence — kept
            separate so they don't inflate FL rates.  Rows with missing or
            below-default confidence values are excluded from both distributions.
            Both use string keys (``"0"``, ``"1"``, ...) because JSON object keys
            are always strings.
            ``rx_settings`` holds representative therapy settings observed for the epoch.
            ``rx_violations`` lists any therapy-settings changes detected within an
            epoch's date range; callers should split affected epochs at those dates.
            Sessions with no recorded settings are skipped: a change is only reported
            between sessions that both have settings.
            ``version_warnings`` lists per-field messages when algorithm identity fields
            differ across epochs — distributions are still populated but callers should
            review version_warnings before drawing conclusions.

        Refusal semantics:
            Mid-epoch RX change nulls only the affected epoch: that epoch's
                ``null_reason`` becomes ``"rx_changed_within_epoch"`` while clean epochs
                in the same call keep their computed distributions.  The top-level
                ``null_reason`` is ``"rx_changed_within_epoch"`` only when EVERY epoch was
                nulled for an RX change; otherwise it stays ``null``.  ``rx_violations``
                always lists the affected epoch label, changed keys, and change dates so
                the caller can split those ranges.
            Partial degradation: when sessions within an epoch differ in ``primary_mode``,
                the epoch's ``null_reason`` stays ``null``; only ``rera_proxy_count`` is
                set to ``null`` with ``rera_reason: "primary_mode_mismatch"``; FL
                distributions remain populated.
            Device not owned by the active profile: the service catches this internally
                and returns a success response with every epoch's
                ``null_reason: "not_available"``.
            Algorithm version mismatch (``algo_version_mismatch``) is NOT a hard refusal
                — distributions are computed and ``version_warnings`` is populated instead.

        Error conditions:
            - Epochs list empty or >6 entries → tool error.
            - Multiple device IDs across epoch specs → tool error.
            - Multiple devices on the date range and no ``device_id`` → tool error
              listing device IDs so the caller can re-issue with ``device_id``.
        """
        from snore.mcp.tools.epochs import (  # noqa: PLC0415
            experimental_compare_epochs as _impl,
        )

        validate_epoch_count(len(epochs))
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="experimental_compare_epochs",
            epochs=epochs,
            metrics=metrics,
        )
