"""Epoch-comparison tools — BreathService adapters.

``compare_epochs`` wraps ``BreathService.compare_device_epochs()`` (device
channels, no analysis needed); ``experimental_compare_epochs`` wraps
``BreathService.compare_experimental_epochs()`` (SNORE breath features).

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
    EpochSpec,
    ExperimentalCompareEpochsResponse,
)
from snore.mcp.tools._scaffold import (
    _scope_and_run,
    _with_experimental_disclaimer,
    tool_error_boundary,
)
from snore.mcp.tools._service_errors import (
    MAPPED_SERVICE_ERRORS,
    raise_mapped_service_error,
)
from snore.mcp.validation import parse_date_range, validate_epoch_span


def _parse_epochs(epochs: list[EpochSpec]) -> list[EpochRequest]:
    """Parse and validate per-epoch date ranges.

    Every ValidationError names the exact epoch index in the parameter name so
    the LLM can identify which epoch is malformed or too long.
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
        validate_epoch_span(start_date, end_date, f"epochs[{i}]")
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
) -> CompareEpochsResponse:
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

    return CompareEpochsResponse.model_validate(result.model_dump(mode="json"))


async def experimental_compare_epochs(
    db_session: AsyncSession,
    profile_id: int,
    epochs: list[EpochSpec],
    metrics: list[str] | None = None,
) -> ExperimentalCompareEpochsResponse:
    """Compare breath-feature distributions across therapy settings epochs.

    Validates per-epoch date ranges and metric strings, then delegates to
    ``BreathService.compare_experimental_epochs()``.  Maps the typed result DTO to the
    ``ExperimentalCompareEpochsResponse`` schema: dates to isoformat strings, NullReason/enum
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
        result = await BreathService(
            db_session, profile_id
        ).compare_experimental_epochs(epoch_requests, metrics=parsed_metrics)
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)

    return ExperimentalCompareEpochsResponse.model_validate(
        result.model_dump(mode="json")
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

        Use it to see whether a settings change moved the device's own channels. Returns
        median, IQR and P95 of the device samples per epoch; every enabled session in
        range contributes, no analysis needed. Call ``get_settings_timeline`` first to
        find epoch boundaries.

        An epoch's ``null_reason`` is ``"no_data_in_range"`` (no sessions) or
        ``"rx_changed_within_epoch"`` (settings changed inside it; split it at the dates
        in ``rx_violations``); other epochs keep their values. An unrecorded channel has
        null stats with ``n_samples: 0``. The top-level ``null_reason`` is
        ``"no_data_in_range"`` (no session in the combined range), ``"not_available"``
        (``device_id`` not in this profile) or ``"rx_changed_within_epoch"`` (every
        epoch changed).

        Args:
            epochs: 1–6 epoch specs, each with ``label``, ``date_start`` and
                ``date_end`` (YYYY-MM-DD, inclusive) and ``device_id``, required
                when several devices have data in the range; all epochs must
                target the same device. An epoch may span at most 366 calendar
                nights; split a longer range.
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

        Returns median, IQR and P95 of leak-valid breaths per epoch, from nights with OK
        analysis results only. For the device's own channels use ``compare_epochs``.
        ``mid_insp_flattening`` is lower = more flow-limited, opposite to the device
        flow-limitation channel. ``rera_proxy_count`` applies no confidence gate: its FL
        runs include the fallback flatness-triage guesses that
        ``flow_class_distribution`` leaves out, so it inherits their sensitivity to
        waveform shape (e.g. across therapy modes or pressure-support levels).

        Null reasons are as in ``compare_epochs`` (``"no_data_in_range"``,
        ``"rx_changed_within_epoch"`` with ``rx_violations``, top-level
        ``"not_available"``), except that an epoch's ``"no_data_in_range"`` means no
        analyzed sessions. An epoch's ``rera_reason`` is ``"analysis_not_run"`` when
        sessions exist but lack current analysis; mixed primary modes null only
        ``rera_proxy_count`` (``rera_reason: "primary_mode_mismatch"``).

        Args:
            epochs: Same format as ``compare_epochs`` ``epochs``.
            metrics: Optional subset of ``"mid_insp_flattening"``, ``"flatness_index"``,
                ``"tidal_volume_ml"``, ``"ie_ratio"`` (default all four).
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
