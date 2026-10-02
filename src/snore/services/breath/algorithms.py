"""Pure (non-service) algorithm functions for the breath service package."""

from __future__ import annotations

import logging
import math

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from snore.analysis.data.waveform_loader import deserialize_waveform_blob
from snore.analysis.shared.versioning import NullReason
from snore.constants import RERAProxyConstants
from snore.services.lttb import lttb_downsample

from .dtos import (
    RawWaveformWindow,
    VentilatoryContext,
    WaveformChannel,
    WaveformChannelName,
    WaveformWindow,
)

logger = logging.getLogger(__name__)

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
            # DTO bounds max_points to >= 2, which lttb_downsample requires
            ts_slice, v_slice = lttb_downsample(ts_slice, v_slice, request.max_points)
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


def _reason_if_null(value: float | None) -> NullReason | None:
    """``NOT_AVAILABLE`` for a value that could not be computed, else None."""
    return NullReason.NOT_AVAILABLE if value is None else None


def _finite_or_none(value: float | None) -> float | None:
    """``value`` when it is a finite number, else None (never emit NaN/inf)."""
    return value if value is not None and math.isfinite(value) else None


def raw_window_series(
    raw: RawWaveformWindow,
    *,
    tolerate_corrupt: frozenset[WaveformChannelName] = frozenset(),
) -> dict[WaveformChannelName, WaveformSeries]:
    """Pure — deserialize each raw channel straight to float64 numpy arrays,
    masked to the request window.

    Channels that are empty, or whose offsets are non-finite or not
    non-decreasing (window slicing relies on sorted offsets), are omitted —
    callers treat an omitted channel as absent.  A corrupt blob raises
    ``ValueError("Invalid waveform data for channel '<name>'")`` unless the
    channel is in ``tolerate_corrupt``, in which case it is omitted with a
    warning.
    """
    request = raw.request
    series: dict[WaveformChannelName, WaveformSeries] = {}
    for raw_ch in raw.channels:
        name = raw_ch.waveform_type
        if raw_ch.sample_count <= 0 or not raw_ch.raw_bytes:
            continue
        try:
            offsets, values = deserialize_waveform_blob(
                raw_ch.raw_bytes, raw_ch.sample_count
            )
        except ValueError as exc:
            if name in tolerate_corrupt:
                logger.warning(
                    "Corrupt '%s' waveform for session %d; treating channel as absent",
                    name.value,
                    raw.session_id,
                )
                continue
            raise ValueError(
                f"Invalid waveform data for channel '{name.value}'"
            ) from exc
        if not np.isfinite(offsets).all() or np.any(np.diff(offsets) < 0):
            continue
        in_window = (offsets >= request.offset_start) & (offsets <= request.offset_end)
        series[name] = (
            offsets[in_window].astype(np.float64),
            values[in_window].astype(np.float64),
        )
    return series


def window_slice(series: WaveformSeries, start: float, end: float) -> WaveformSeries:
    """Finite-valued samples whose offset falls in ``[start, end]`` (inclusive).

    O(log n) via searchsorted — ``series`` offsets must be non-decreasing.
    Non-finite values are dropped so they cannot poison downstream metrics.
    """
    offsets, values = series
    lo = int(np.searchsorted(offsets, start, side="left"))
    hi = int(np.searchsorted(offsets, end, side="right"))
    ts, vs = offsets[lo:hi], values[lo:hi]
    finite = np.isfinite(vs)
    return ts[finite], vs[finite]


def window_mean(series: WaveformSeries, start: float, end: float) -> float | None:
    """Mean of the finite samples in ``[start, end]``; None when there are none."""
    _, values = window_slice(series, start, end)
    return _finite_or_none(float(values.mean())) if values.size > 0 else None


def _episode_seconds(episode: object) -> float:
    """Duration of one persisted PB episode; raises TypeError/ValueError when
    malformed.  The ``duration`` fallback is read only when no end key exists."""
    if not isinstance(episode, dict):
        raise TypeError("episode is not an object")
    # float(None) raises TypeError → a null bound marks the episode malformed.
    start_raw: Any = episode.get("start_time", episode.get("start", 0))
    start_t = float(start_raw)
    if "end_time" in episode:
        end_t = float(episode["end_time"])
    elif "end" in episode:
        end_t = float(episode["end"])
    else:
        end_t = start_t + float(episode.get("duration", 0))
    if not (math.isfinite(start_t) and math.isfinite(end_t)):
        raise ValueError("non-finite episode bounds")
    return max(0.0, end_t - start_t)


def periodic_breathing_seconds(episodes: object) -> float | None:
    """Total duration of persisted ``periodic_breathing_episodes``.

    None when no episode list was persisted (PB detection never ran); 0.0
    when it ran and found no episodes.  Episodes use ``start_time``/
    ``end_time`` keys, with ``start``/``end``/``duration`` accepted as
    fallbacks.  Raises ``ValueError`` when the list or any episode is
    malformed, so callers can degrade that one session.
    """
    if episodes is None:
        return None
    if not isinstance(episodes, list):
        raise ValueError("periodic_breathing_episodes is not a list")
    total = 0.0
    for episode in episodes:
        try:
            total += _episode_seconds(episode)
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed periodic-breathing episode") from exc
    return total


def derive_mv_from_flow(
    offsets: np.ndarray,
    values: np.ndarray,
    *,
    window_s: float = 60.0,
    out_dt_s: float = 2.0,
) -> WaveformSeries:
    """Pure — derive minute ventilation (L/min) from a flow waveform (L/min).

    MV(t) = mean of positive-clipped flow over the trailing window
    ``[t - window_s, t]``, sampled every ``out_dt_s`` seconds starting at
    ``offsets[0] + window_s`` up to the last input offset.  Output samples
    whose window contains zero input samples are omitted — merged sessions
    have timestamp gaps, so uniform sampling is never assumed.

    Returns ``(out_offsets, out_values)``; empty arrays when the input is too
    short to cover a single window, when its end offsets are non-finite, or
    when ``offsets`` is not non-decreasing (searchsorted requires sorted
    input — unsorted offsets would silently produce garbage windows, so
    downstream metrics go null instead).  NaN samples in ``values`` are
    treated as 0.0 flow so they cannot poison the cumulative sum.
    O(n log n): cumsum + searchsorted, no per-window scans.
    """
    empty = (np.array([]), np.array([]))
    if offsets.size == 0 or not np.isfinite(offsets[[0, -1]]).all():
        return empty
    if float(offsets[-1]) - float(offsets[0]) < window_s:
        return empty
    if np.any(np.diff(offsets) < 0):
        return empty

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
      60 s preceding the event (``[max(0, offset_s - 60), offset_s]``,
      inclusive), in L/min per minute; needs >= 2 samples.
    - ``stability_index``: sample stdev (ddof=1) / mean of MV over the same
      window; needs >= 3 samples and a non-zero mean.
    - ``ps_delivered_cmh2o``: mean(THERAPY_PRESSURE) − mean(EPAP), each
      averaged over ±5 s around the event start (the channels need not share
      a sample rate or alignment).

    MV metrics require ``offset_s > 0`` (an event before the session start
    has no preceding window); PS requires the ±5 s window to end after 0.
    Non-finite samples are ignored; every null value carries
    ``NOT_AVAILABLE``.
    """
    slope: float | None = None
    stability: float | None = None
    ps: float | None = None

    if offset_s > 0.0 and mv is not None:
        mv_ts, mv_vals = window_slice(mv, max(0.0, offset_s - 60.0), offset_s)
        if mv_vals.size >= 2:
            # Offsets are seconds → per-second slope; ×60 → L/min per minute.
            slope_per_s = _linear_slope(mv_ts, mv_vals)
            slope = slope_per_s * 60.0 if slope_per_s is not None else None
        if mv_vals.size >= 3:
            mean_mv = float(mv_vals.mean())
            if mean_mv != 0.0:
                stability = float(np.std(mv_vals, ddof=1)) / mean_mv

    ps_start = max(0.0, offset_s - 5.0)
    ps_end = offset_s + 5.0
    if ps_end > 0.0 and therapy_pressure is not None and epap is not None:
        tp_mean = window_mean(therapy_pressure, ps_start, ps_end)
        ep_mean = window_mean(epap, ps_start, ps_end)
        if tp_mean is not None and ep_mean is not None:
            ps = tp_mean - ep_mean

    slope, stability, ps = (_finite_or_none(v) for v in (slope, stability, ps))
    return VentilatoryContext(
        preceding_mv_slope_lpm_per_min=slope,
        preceding_mv_slope_reason=_reason_if_null(slope),
        stability_index=stability,
        stability_reason=_reason_if_null(stability),
        ps_delivered_cmh2o=ps,
        ps_reason=_reason_if_null(ps),
    )
