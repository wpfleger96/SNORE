"""Pure (non-service) algorithm functions for the breath service package."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from snore.analysis.data.waveform_loader import deserialize_waveform_blob
from snore.analysis.shared.versioning import NullReason
from snore.analysis.types import AnalysisResult as AnalysisResultDTO
from snore.constants import RERAProxyConstants
from snore.services.lttb import lttb_downsample

from .dtos import (
    RawWaveformWindow,
    VentilatoryContext,
    WaveformChannel,
    WaveformChannelName,
    WaveformWindow,
)

# (offsets_seconds, values) for one waveform channel; offsets non-decreasing.
WaveformSeries = tuple[np.ndarray, np.ndarray]


def compute_waveform_window(raw: RawWaveformWindow) -> WaveformWindow:
    """Pure — no DB access. Deserializes bytes, slices window, applies LTTB."""

    request = raw.request
    channels_out: list[WaveformChannel] = []
    missing_channels: list[WaveformChannelName] = list(raw.missing_channels)

    for raw_ch in raw.channels:
        if raw_ch.sample_count <= 0 or not raw_ch.raw_bytes:
            missing_channels.append(raw_ch.waveform_type)
            continue
        try:
            timestamps, values = deserialize_waveform_blob(
                raw_ch.raw_bytes, raw_ch.sample_count
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid waveform data for channel '{raw_ch.waveform_type.value}'"
            ) from exc
        # Slice to requested window
        mask = (timestamps >= request.offset_start) & (timestamps <= request.offset_end)
        ts_slice = timestamps[mask]
        v_slice = values[mask]

        original_count = int(len(ts_slice))
        is_downsampled = False
        if request.max_points is not None and original_count > request.max_points:
            # LTTB downsampling: lttb_downsample(timestamps, values, target_points)
            if len(ts_slice) >= 3:
                ts_ds, v_ds = lttb_downsample(ts_slice, v_slice, request.max_points)
                ts_slice = ts_ds
                v_slice = v_ds
                is_downsampled = True

        channels_out.append(
            WaveformChannel(
                channel_type=raw_ch.waveform_type,
                unit=raw_ch.unit,
                sample_rate=raw_ch.sample_rate,
                offset_seconds=ts_slice.tolist(),
                values=v_slice.tolist(),
                original_sample_count=original_count,
                is_downsampled=is_downsampled,
            )
        )

    missing_reason: NullReason | None = (
        NullReason.CHANNEL_ABSENT if missing_channels else None
    )

    return WaveformWindow(
        session_id=raw.session_id,
        session_start_wall_clock=raw.session_start_wall_clock,
        timezone_status=raw.timezone_status,
        timezone_name=raw.timezone_name,
        window_start_offset=request.offset_start,
        window_end_offset=request.offset_end,
        channels=channels_out,
        missing_channels=missing_channels,
        missing_channel_reason=missing_reason,
    )


# ---------------------------------------------------------------------------
# §13 — BreathService helpers
# ---------------------------------------------------------------------------


def iter_fl_run_recoveries(
    breath_rows: Sequence[Any],
    *,
    fl_class_threshold: int = RERAProxyConstants.FL_CLASS_THRESHOLD,
    min_fl_run_length: int = RERAProxyConstants.MIN_FL_RUN_LENGTH,
    recovery_amplitude_margin: float = RERAProxyConstants.RECOVERY_AMPLITUDE_MARGIN,
) -> Iterator[tuple[int, int, int]]:
    """Yield (run_start_idx, run_last_idx, recovery_idx) per RERA-proxy event.

    A qualifying event is a run of >= ``min_fl_run_length`` consecutive breaths
    with ``flow_class >= fl_class_threshold`` whose immediately-next breath
    (no gap; a ``flow_class is None`` breath ends the run) is a recovery
    breath.  The follower is a recovery breath when EITHER:

    (a) ``is_recovery_breath is True`` — the analysis-time amplitude detector
        (``detector.py::_detect_reras``); OR
    (b) self-contained (RERA-proxy v2): the follower's ``flow_class`` drops to
        <= 2 AND its ``peak_flow_lpm`` is >= ``(1 + recovery_amplitude_margin)``
        times the mean of the run's non-null ``peak_flow_lpm`` values.

    Missing data (null ``flow_class`` or ``peak_flow_lpm`` on the follower, or
    an all-null-peak run) never satisfies (b).
    """
    n = len(breath_rows)
    i = 0
    while i < n:
        b = breath_rows[i]
        if b.flow_class is None or b.flow_class < fl_class_threshold:
            i += 1
            continue
        run_start = i
        while (
            i < n
            and breath_rows[i].flow_class is not None
            and breath_rows[i].flow_class >= fl_class_threshold
        ):
            i += 1
        if i - run_start < min_fl_run_length or i >= n:
            continue
        follower = breath_rows[i]
        is_recovery = follower.is_recovery_breath is True
        if (
            not is_recovery
            and follower.flow_class is not None
            and follower.flow_class <= 2
            and follower.peak_flow_lpm is not None
        ):
            run_peaks = [
                breath_rows[k].peak_flow_lpm
                for k in range(run_start, i)
                if breath_rows[k].peak_flow_lpm is not None
            ]
            if run_peaks:
                run_mean = sum(run_peaks) / len(run_peaks)
                is_recovery = follower.peak_flow_lpm >= (
                    (1.0 + recovery_amplitude_margin) * run_mean
                )
        if is_recovery:
            yield (run_start, i - 1, i)


def _count_fl_run_reras(
    breath_rows: Sequence[Any],
    fl_class_threshold: int = RERAProxyConstants.FL_CLASS_THRESHOLD,
    min_fl_run_length: int = RERAProxyConstants.MIN_FL_RUN_LENGTH,
    recovery_amplitude_margin: float = RERAProxyConstants.RECOVERY_AMPLITUDE_MARGIN,
) -> int:
    """Count RERA-proxy events: FL runs ending in a recovery breath."""
    return sum(
        1
        for _ in iter_fl_run_recoveries(
            breath_rows,
            fl_class_threshold=fl_class_threshold,
            min_fl_run_length=min_fl_run_length,
            recovery_amplitude_margin=recovery_amplitude_margin,
        )
    )


# ---------------------------------------------------------------------------
# §12 — Per-event ventilatory context + periodic breathing (pure)
# ---------------------------------------------------------------------------


def window_slice(series: WaveformSeries, start: float, end: float) -> WaveformSeries:
    """Samples whose offset falls in ``[start, end]`` (inclusive both ends).

    O(log n) via searchsorted — ``series`` offsets must be non-decreasing.
    """
    offsets, values = series
    lo = int(np.searchsorted(offsets, start, side="left"))
    hi = int(np.searchsorted(offsets, end, side="right"))
    return offsets[lo:hi], values[lo:hi]


def window_mean(series: WaveformSeries, start: float, end: float) -> float | None:
    """Mean of the samples in ``[start, end]``; None when the window is empty."""
    _, values = window_slice(series, start, end)
    return float(values.mean()) if values.size > 0 else None


def periodic_breathing_seconds(result_json: dict[str, Any] | None) -> float | None:
    """Total periodic-breathing episode duration from a persisted
    ``AnalysisResult.programmatic_result_json``.

    None when no result JSON was persisted (PB detection never ran); 0.0 when
    it ran and found no episodes.  Episodes use ``start_time``/``end_time``
    keys, with ``start``/``end``/``duration`` accepted as fallbacks.
    """
    if not result_json:
        return None
    dto = AnalysisResultDTO.model_validate(result_json)
    total = 0.0
    for ep in dto.periodic_breathing_episodes or []:
        start_t = float(ep.get("start_time", ep.get("start", 0)))
        end_t = float(
            ep.get("end_time", ep.get("end", start_t + ep.get("duration", 0)))
        )
        total += max(0.0, end_t - start_t)
    return total


def derive_mv_from_flow(
    offsets: np.ndarray,
    values: np.ndarray,
    *,
    window_s: float = 60.0,
    out_dt_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure — derive minute ventilation (L/min) from a flow waveform (L/min).

    MV(t) = mean of positive-clipped flow over the trailing window
    ``[t - window_s, t]``, sampled every ``out_dt_s`` seconds starting at
    ``offsets[0] + window_s`` up to the last input offset.  Output samples
    whose window contains zero input samples are omitted — merged sessions
    have timestamp gaps, so uniform sampling is never assumed.

    Returns ``(out_offsets, out_values)``; empty arrays when the input is too
    short to cover a single window or when ``offsets`` is not non-decreasing
    (searchsorted requires sorted input — unsorted offsets would silently
    produce garbage windows, so downstream metrics go null instead).  NaN
    samples in ``values`` are treated as 0.0 flow so they cannot poison the
    cumulative sum.  O(n log n): cumsum + searchsorted, no per-window scans.
    """
    if offsets.size == 0 or float(offsets[-1]) - float(offsets[0]) < window_s:
        return np.array([]), np.array([])
    if np.any(np.diff(offsets) < 0):
        return np.array([]), np.array([])

    clipped = np.clip(np.where(np.isnan(values), 0.0, values), 0.0, None)
    csum = np.concatenate(([0.0], np.cumsum(clipped, dtype=np.float64)))

    out_times = np.arange(
        float(offsets[0]) + window_s, float(offsets[-1]) + 1e-9, out_dt_s
    )
    # Window [t - window_s, t] inclusive both ends
    lo = np.searchsorted(offsets, out_times - window_s, side="left")
    hi = np.searchsorted(offsets, out_times, side="right")
    counts = hi - lo
    mask = counts > 0
    mv = (csum[hi[mask]] - csum[lo[mask]]) / counts[mask]
    return out_times[mask], mv


def derive_mv_from_flow_window(raw: RawWaveformWindow) -> WaveformSeries | None:
    """Pure — flow-derived MV series from a pre-fetched FLOW blob window.

    Deserializes the raw FLOW blob straight to numpy (bypassing the
    render-oriented ``compute_waveform_window`` avoids a numpy → list → numpy
    round trip over the full-session flow signal), slices it to the request
    window, and runs ``derive_mv_from_flow``.  Returns ``None`` when FLOW is
    absent/empty or too short to derive a single MV sample.  A corrupt blob
    raises ``ValueError``, mirroring ``compute_waveform_window``.
    """
    for flow_ch in raw.channels:
        if flow_ch.waveform_type != WaveformChannelName.FLOW:
            continue
        if flow_ch.sample_count <= 0 or not flow_ch.raw_bytes:
            return None
        try:
            flow_off, flow_val = deserialize_waveform_blob(
                flow_ch.raw_bytes, flow_ch.sample_count
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid waveform data for channel '{flow_ch.waveform_type.value}'"
            ) from exc
        in_window = (flow_off >= raw.request.offset_start) & (
            flow_off <= raw.request.offset_end
        )
        mv_off, mv_val = derive_mv_from_flow(flow_off[in_window], flow_val[in_window])
        return (mv_off, mv_val) if mv_off.size > 0 else None
    return None


def _linear_slope(xs: np.ndarray, ys: np.ndarray) -> float | None:
    """Least-squares slope of ``ys`` on ``xs``; None when ``xs`` has no spread."""
    dx = xs - xs.mean()
    den = float(np.dot(dx, dx))
    return float(np.dot(dx, ys - ys.mean())) / den if den != 0.0 else None


def compute_ventilatory_context(
    offset_s: float,
    *,
    mv: WaveformSeries | None,
    therapy_pressure: WaveformSeries | None,
    epap: WaveformSeries | None,
) -> VentilatoryContext:
    """Pure — MV slope, MV stability, and delivered PS around one event.

    - ``preceding_mv_slope_lpm_per_min``: least-squares slope of MV over the
      60 s preceding the event (``[max(0, offset_s - 60), offset_s]``), in
      L/min per minute; needs >= 2 samples.
    - ``stability_index``: stdev / mean of MV over the same window; needs
      >= 3 samples and a non-zero mean.
    - ``ps_delivered_cmh2o``: mean(THERAPY_PRESSURE − EPAP) over ±5 s around
      the event start (the two slices are truncated to equal length).

    MV metrics require ``offset_s > 0`` (an event before the session start
    has no preceding window); PS requires the ±5 s window to end after 0.
    """
    ctx = VentilatoryContext()

    if offset_s > 0.0 and mv is not None:
        mv_start = max(0.0, offset_s - 60.0)
        mv_ts, mv_vals = window_slice(mv, mv_start, offset_s)
        if mv_vals.size >= 2:
            # Offsets are seconds → per-second slope; ×60 → L/min per minute.
            slope_per_s = _linear_slope(mv_ts, mv_vals)
            if slope_per_s is not None:
                ctx.preceding_mv_slope_lpm_per_min = slope_per_s * 60.0
                ctx.preceding_mv_slope_reason = None
        if mv_vals.size >= 3:
            mean_mv = float(mv_vals.mean())
            if mean_mv != 0.0:
                ctx.stability_index = float(np.std(mv_vals, ddof=1)) / mean_mv
                ctx.stability_reason = None

    ps_start = max(0.0, offset_s - 5.0)
    ps_end = offset_s + 5.0
    if ps_end > 0.0 and therapy_pressure is not None and epap is not None:
        _, tp_vals = window_slice(therapy_pressure, ps_start, ps_end)
        _, ep_vals = window_slice(epap, ps_start, ps_end)
        n = min(tp_vals.size, ep_vals.size)
        if n > 0:
            ctx.ps_delivered_cmh2o = float(np.mean(tp_vals[:n] - ep_vals[:n]))
            ctx.ps_reason = None

    return ctx
