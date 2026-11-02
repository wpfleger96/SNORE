"""Device detail derived on read for session, day and period responses.

The flow-limitation (``fl``) and snore channels have no summary columns, so
their statistics decode the stored waveform blobs.  Samples are pooled across
sessions, never averaged per session, as OSCAR's Statistics page does
(``Profile::calcPercentile``); the definitions match the MCP nightly summary
(``services/breath/nightly.py``).

Pooling weights each session by its sample count.  OSCAR weights by time, count
times sample rate (``session.cpp:1301-1304``), so the two agree only at a single
sample rate; every local session is 0.5 Hz.

Each session reduces to a histogram of its distinct values, which keeps a long
range's peak memory far below holding its samples.
"""

from __future__ import annotations

import asyncio
import logging
import math

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from itertools import batched

import numpy as np

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.data.waveform_loader import deserialize_waveform_blob
from snore.database import models
from snore.services.schemas import DeviceChannelStats, MaskOnSpan
from snore.therapy_hours import parse_mask_on_segments
from snore.utils.stats import nearest_rank_index

logger = logging.getLogger(__name__)

__all__ = [
    "CHANNEL_FLOORS",
    "ChannelHistograms",
    "channel_histogram",
    "decode_channel_blobs",
    "load_channel_histograms",
    "mask_on_spans",
    "pool_channel_stats",
]

# Lowest valid sample per channel: ``fl`` blobs keep the raw -0.01 mask-off
# sentinel.  Zeros are legitimate data on both channels.
CHANNEL_FLOORS = {"fl": 0.0, "snore": -math.inf}
_CHANNEL_FIELDS = {
    "fl": ("device_flg_median", "device_flg_95th", "device_flg_max"),
    "snore": ("snore_median", "snore_95th", "snore_max"),
}
_ROUND_DIGITS = 4

# Sessions fetched per query.  The blobs of a batch are buffered in memory until
# decoded, so this bounds that buffer, at two blobs per session for the FL and
# snore pair.
_DECODE_BATCH_SESSIONS = 64

# A session's usable samples as ascending distinct values and how often each occurs.
type ChannelHistogram = tuple[np.ndarray, np.ndarray]
# channel -> session_id -> histogram
type ChannelHistograms = dict[str, dict[int, ChannelHistogram]]


def channel_histogram(channel: str, values: np.ndarray) -> ChannelHistogram:
    """Histogram of the finite ``values`` at or above the channel's floor."""
    usable = values[np.isfinite(values) & (values >= CHANNEL_FLOORS[channel])]
    unique_values, counts = np.unique(usable, return_counts=True)
    return unique_values, counts


def histogram_distribution(
    histograms: Sequence[ChannelHistogram],
) -> tuple[float, float, float] | None:
    """``(median, 95th, max)`` of the pooled samples, ``None`` when there are none.

    Exactly what sorting the samples gives.  The median is
    ``statistics.median``-equivalent, so the middle pair is averaged as Python
    floats rather than float32; the 95th is nearest rank.
    """
    if not histograms:
        return None
    values, merged_index = np.unique(
        np.concatenate([v for v, _ in histograms]), return_inverse=True
    )
    counts = np.zeros(values.size, dtype=np.int64)
    np.add.at(counts, merged_index, np.concatenate([c for _, c in histograms]))
    n = int(counts.sum())
    if n == 0:
        return None
    cumulative = np.cumsum(counts)

    def at_rank(rank: int) -> float:
        return float(values[np.searchsorted(cumulative, rank, side="right")])

    median = (at_rank((n - 1) // 2) + at_rank(n // 2)) / 2
    return median, at_rank(nearest_rank_index(n, 0.95)), float(values[-1])


def _decode_rows[T](
    rows: Sequence[tuple[int, str, bytes | None, int | None]],
    reduce: Callable[[str, np.ndarray], T],
    values: dict[str, dict[int, T]],
    corrupt: dict[str, set[int]],
) -> None:
    """Decode and reduce fetched blobs into ``values`` and ``corrupt``.

    Runs on a worker thread, so ``rows`` holds plain values, never ORM objects.
    """
    for session_id, channel, blob, sample_count in rows:
        if not blob or not sample_count:
            continue
        try:
            _, samples = deserialize_waveform_blob(blob, sample_count)
        except ValueError:
            corrupt[channel].add(session_id)
            continue
        values[channel][session_id] = reduce(channel, samples)


async def decode_channel_blobs[T](
    db: AsyncSession,
    session_ids: Sequence[int],
    channels: Sequence[str],
    reduce: Callable[[str, np.ndarray], T],
) -> tuple[dict[str, dict[int, T]], dict[str, set[int]]]:
    """``(values, corrupt)`` of ``channels`` for ``session_ids``.

    ``values`` is channel -> session_id -> ``reduce(channel, samples)``, the raw
    samples including sentinels; ``corrupt`` is channel -> ids of sessions whose
    blob could not be decoded.  Every requested channel is a key of both, and a
    session without a usable row for a channel is absent from its ``values`` map.

    Blobs are fetched in batches and decoded off the event loop.  There is no
    ownership check here: callers pass ids already scoped to the profile.
    """
    values: dict[str, dict[int, T]] = {channel: {} for channel in channels}
    corrupt: dict[str, set[int]] = {channel: set() for channel in channels}
    waveform = models.Waveform
    for batch in batched(session_ids, _DECODE_BATCH_SESSIONS, strict=False):
        result = await db.execute(
            select(
                waveform.session_id,
                waveform.waveform_type,
                waveform.data_blob,
                waveform.sample_count,
            ).where(
                waveform.session_id.in_(batch),
                waveform.waveform_type.in_(channels),
            )
        )
        await asyncio.to_thread(
            _decode_rows,
            [(sid, channel, blob, count) for sid, channel, blob, count in result],
            reduce,
            values,
            corrupt,
        )
    return values, corrupt


async def load_channel_histograms(
    db: AsyncSession, session_ids: Sequence[int]
) -> ChannelHistograms:
    """Histograms of the usable ``fl`` and ``snore`` samples of ``session_ids``.

    Both channels are keys of the result.  A session is absent from a channel
    when it has no row for it, the blob is empty, or the blob is corrupt, which
    logs a warning.  ``decode_channel_blobs`` does no ownership check, so callers
    pass ids already scoped to the profile.
    """
    histograms, corrupt = await decode_channel_blobs(
        db, session_ids, tuple(CHANNEL_FLOORS), channel_histogram
    )
    for channel, corrupt_ids in corrupt.items():
        for session_id in sorted(corrupt_ids):
            logger.warning(
                "Skipping corrupt %s waveform of session %d", channel, session_id
            )
    return histograms


def pool_channel_stats(
    histograms: ChannelHistograms, session_ids: Sequence[int]
) -> DeviceChannelStats | None:
    """Statistics over the combined samples of ``session_ids``.

    ``None`` when no session has a usable sample on either channel.
    """
    fields: dict[str, float] = {}
    for channel, names in _CHANNEL_FIELDS.items():
        by_session = histograms[channel]
        distribution = histogram_distribution(
            [by_session[sid] for sid in session_ids if sid in by_session]
        )
        if distribution is not None:
            for name, value in zip(names, distribution, strict=True):
                fields[name] = round(value, _ROUND_DIGITS)
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
