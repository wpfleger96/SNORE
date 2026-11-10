"""Device detail derived on read for session, day and period responses.

The flow-limitation (``fl``) and snore channels have no summary columns, and
the respiratory and pulse columns hold one value per session (on ResMed, the
day's), so their statistics decode the stored waveform blobs.  Samples are
pooled across sessions, never averaged per session, as OSCAR's Statistics page
does (``Profile::calcPercentile``); the definitions match the MCP nightly
summary (``services/breath/nightly.py``).

Pooling weights each session by its sample count.  OSCAR weights by time, count
times sample rate (``session.cpp:1301-1304``), so the two agree only while the
pooled sessions share a sample rate, as the sessions of one channel do (on
ResMed, 0.5 Hz for the PLD channels and about 1 Hz for pulse).

Each session reduces to a histogram of its distinct values, which keeps a long
range's peak memory far below holding its samples.
"""

from __future__ import annotations

import logging
import math

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import NewType

import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession

from snore.constants import (
    IE_RATIO_VALID_RANGE,
    PULSE_VALID_RANGE,
    RESPIRATORY_RATE_VALID_RANGE,
    TI_VALID_RANGE_S,
    TIDAL_VOLUME_VALID_RANGE_ML,
)
from snore.database.waveform_blobs import decode_channel_blobs
from snore.services.schemas import DeviceChannelStats, MaskOnSpan
from snore.therapy_hours import parse_mask_on_segments
from snore.utils.stats import nearest_rank_index

logger = logging.getLogger(__name__)

__all__ = [
    "CHANNEL_RANGES",
    "REPORTED_CHANNELS",
    "ChannelHistogram",
    "ChannelHistograms",
    "PooledHistogram",
    "channel_histogram",
    "histogram_distribution",
    "load_channel_histograms",
    "mask_on_spans",
    "pool_channel_stats",
    "pool_histograms",
    "pooled_channel_distribution",
    "pooled_count",
    "pooled_distribution",
    "pooled_median",
    "pooled_share_above",
    "value_at_rank",
]

# Valid samples per channel, inclusive: ``fl`` and ``leak`` blobs keep their
# negative mask-off sentinels (``fl`` the raw -0.01), and the respiratory and
# pulse blobs keep the out-of-range samples the parser leaves out of session
# statistics.  Zeros are legitimate data on every channel but pulse.
# ``DeviceChannelStats`` covers the ``_CHANNEL_PREFIXES`` channels only; the
# MCP nightly summary also reduces ``leak``.
_NON_NEGATIVE = (0.0, math.inf)
CHANNEL_RANGES: dict[str, tuple[float, float]] = {
    "fl": _NON_NEGATIVE,
    "snore": (-math.inf, math.inf),
    "leak": _NON_NEGATIVE,
    "rr": RESPIRATORY_RATE_VALID_RANGE,
    "tv": TIDAL_VOLUME_VALID_RANGE_ML,
    "mv": _NON_NEGATIVE,
    "ti": TI_VALID_RANGE_S,
    "ie_ratio": IE_RATIO_VALID_RANGE,
    "pulse": PULSE_VALID_RANGE,
}
# The channels ``DeviceChannelStats`` reports, and the prefix of each one's
# ``_median``, ``_95th`` and ``_max`` fields.
_CHANNEL_PREFIXES = {
    "fl": "device_flg",
    "snore": "snore",
    "rr": "respiratory_rate",
    "tv": "tidal_volume",
    "mv": "minute_ventilation",
    "ti": "ti",
    "ie_ratio": "ie_ratio",
    "pulse": "pulse",
}
# What session and day detail load: every channel ``DeviceChannelStats`` reports.
REPORTED_CHANNELS = tuple(_CHANNEL_PREFIXES)
_STATISTICS = ("median", "95th", "max")
_ROUND_DIGITS = 4

# A session's usable samples as ascending distinct values and how often each occurs.
type ChannelHistogram = tuple[np.ndarray, np.ndarray]
# channel -> session_id -> histogram
type ChannelHistograms = dict[str, dict[int, ChannelHistogram]]
# Samples pooled by ``pool_histograms``: ascending distinct values and their
# cumulative counts.  A distinct type, so a per-session ``ChannelHistogram``
# (plain counts) cannot be passed where cumulative counts are read.
PooledHistogram = NewType("PooledHistogram", tuple[np.ndarray, np.ndarray])


def channel_histogram(channel: str, values: np.ndarray) -> ChannelHistogram:
    """Histogram of the finite ``values`` within the channel's valid range."""
    low, high = CHANNEL_RANGES[channel]
    usable = values[np.isfinite(values) & (values >= low) & (values <= high)]
    unique_values, counts = np.unique(usable, return_counts=True)
    return unique_values, counts


def pool_histograms(histograms: Sequence[ChannelHistogram]) -> PooledHistogram:
    """Pooled samples as ascending distinct values and their cumulative counts.

    The last cumulative count is the pooled sample count; both arrays are empty
    when there are no samples.  Index the result with ``value_at_rank``.
    """
    if not histograms:
        return PooledHistogram((np.empty(0), np.empty(0, dtype=np.int64)))
    values, merged_index = np.unique(
        np.concatenate([v for v, _ in histograms]), return_inverse=True
    )
    counts = np.zeros(values.size, dtype=np.int64)
    np.add.at(counts, merged_index, np.concatenate([c for _, c in histograms]))
    return PooledHistogram((values, np.cumsum(counts)))


def pooled_count(pooled: PooledHistogram) -> int:
    """Number of samples in a ``pool_histograms`` result."""
    return int(pooled[1][-1]) if pooled[1].size else 0


def value_at_rank(pooled: PooledHistogram, rank: int) -> float:
    """The ``rank``-th (0-based) smallest pooled sample: ``sorted(samples)[rank]``."""
    values, cumulative = pooled
    return float(values[np.searchsorted(cumulative, rank, side="right")])


def pooled_share_above(
    pooled: PooledHistogram, threshold: float, *, inclusive: bool
) -> float | None:
    """Fraction of pooled samples above ``threshold`` (or equal, when ``inclusive``).

    ``None`` when there are no samples.
    """
    n = pooled_count(pooled)
    if n == 0:
        return None
    values, cumulative = pooled
    n_distinct_below = int(
        np.searchsorted(values, threshold, side="left" if inclusive else "right")
    )
    n_below = int(cumulative[n_distinct_below - 1]) if n_distinct_below else 0
    return (n - n_below) / n


def pooled_median(pooled: PooledHistogram) -> float:
    """``statistics.median`` of the pooled samples (middle pair averaged as floats)."""
    n = pooled_count(pooled)
    return (value_at_rank(pooled, (n - 1) // 2) + value_at_rank(pooled, n // 2)) / 2


def pooled_distribution(pooled: PooledHistogram) -> tuple[float, float, float] | None:
    """``(median, 95th, max)`` of the pooled samples, ``None`` when there are none.

    Exactly what sorting the samples gives.  The median is
    ``statistics.median``-equivalent, so the middle pair is averaged as Python
    floats rather than float32; the 95th is nearest rank.
    """
    n = pooled_count(pooled)
    if n == 0:
        return None
    return (
        pooled_median(pooled),
        value_at_rank(pooled, nearest_rank_index(n, 0.95)),
        float(pooled[0][-1]),
    )


def histogram_distribution(
    histograms: Sequence[ChannelHistogram],
) -> tuple[float, float, float] | None:
    """``pooled_distribution`` of the sessions' ``histograms`` pooled together."""
    return pooled_distribution(pool_histograms(histograms))


async def load_channel_histograms(
    db: AsyncSession, session_ids: Sequence[int], channels: Sequence[str]
) -> ChannelHistograms:
    """Histograms of the usable ``channels`` samples of ``session_ids``.

    Each of ``channels`` is a key of the result.  A session is absent from a
    channel when it has no row for it, the blob is empty, or the blob is
    corrupt, which logs a warning.  ``decode_channel_blobs`` does no ownership
    check, so callers pass ids already scoped to the profile.
    """
    histograms, _ = await decode_channel_blobs(
        db, session_ids, channels, channel_histogram
    )
    return histograms


def pooled_channel_distribution(
    by_session: Mapping[int, ChannelHistogram], session_ids: Iterable[int]
) -> tuple[float, float, float] | None:
    """One channel's ``(median, 95th, max)`` over the pooled ``session_ids``.

    Rounded as ``DeviceChannelStats`` reports them; ``None`` when none of the
    sessions has a usable sample.
    """
    distribution = histogram_distribution(
        [by_session[sid] for sid in session_ids if sid in by_session]
    )
    if distribution is None:
        return None
    median, p95, maximum = (round(value, _ROUND_DIGITS) for value in distribution)
    return median, p95, maximum


def pool_channel_stats(
    histograms: ChannelHistograms, session_ids: Sequence[int]
) -> DeviceChannelStats | None:
    """Statistics over the combined samples of ``session_ids``.

    A reported channel not loaded into ``histograms`` reads null, and a loaded
    channel ``DeviceChannelStats`` does not report (``leak``) is ignored.
    ``None`` when no session has a usable sample on any reported channel.
    """
    fields: dict[str, float] = {}
    for channel, prefix in _CHANNEL_PREFIXES.items():
        if channel not in histograms:
            continue
        distribution = pooled_channel_distribution(histograms[channel], session_ids)
        if distribution is not None:
            for stat, value in zip(_STATISTICS, distribution, strict=True):
                fields[f"{prefix}_{stat}"] = value
    return DeviceChannelStats.model_validate(fields) if fields else None


def mask_on_spans(
    session_id: int, start_time: datetime, raw_segments: object
) -> list[MaskOnSpan] | None:
    """Wall-clock spans of a session's stored segments; ``None`` when unknown.

    Offsets too large for a timedelta or a datetime count as unknown.
    """
    segments = parse_mask_on_segments(raw_segments, session_id)
    if segments is None:
        return None
    try:
        return [
            MaskOnSpan(
                session_id=session_id,
                start=start_time + timedelta(seconds=start_offset),
                end=start_time + timedelta(seconds=end_offset),
                duration_seconds=end_offset - start_offset,
            )
            for start_offset, end_offset in segments
        ]
    except OverflowError:
        logger.warning(
            "Session %d: mask_on_segments offsets overflow the wall clock; "
            "treating segments as unknown",
            session_id,
        )
        return None
