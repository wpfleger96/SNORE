"""Device detail derived on read for session, day and period responses.

The flow-limitation (``fl``) and snore channels have no summary columns, so
their statistics decode the stored waveform blobs.  Samples are pooled across
sessions, never averaged per session, as OSCAR's Statistics page does
(``Profile::calcPercentile``); the definitions match the MCP nightly summary
(``services/breath/nightly.py``).
"""

from __future__ import annotations

import logging
import math

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.data.waveform_loader import deserialize_waveform_blob
from snore.database import models
from snore.services.schemas import DeviceChannelStats, MaskOnSpan
from snore.therapy_hours import parse_mask_on_segments
from snore.utils.db_chunk import iter_id_chunks
from snore.utils.stats import nearest_rank_index

logger = logging.getLogger(__name__)

__all__ = [
    "ChannelSamples",
    "load_channel_samples",
    "mask_on_spans",
    "pool_channel_stats",
    "sample_distribution",
    "usable_samples",
]

# Lowest valid sample per channel: ``fl`` blobs keep the raw -0.01 mask-off
# sentinel.  Zeros are legitimate data on both channels.
_CHANNEL_FLOORS = {"fl": 0.0, "snore": -math.inf}
_CHANNEL_FIELD_PREFIXES = {"fl": "device_flg", "snore": "snore"}
_ROUND_DIGITS = 4

# channel -> session_id -> usable samples
type ChannelSamples = dict[str, dict[int, np.ndarray]]


def usable_samples(channel: str, values: np.ndarray) -> np.ndarray:
    """Finite samples of ``values`` at or above the channel's floor."""
    return values[np.isfinite(values) & (values >= _CHANNEL_FLOORS[channel])]


def sample_distribution(samples: np.ndarray) -> tuple[float, float, float] | None:
    """``(median, 95th, max)`` of ``samples``, or ``None`` when there are none.

    The median is ``statistics.median``-equivalent, so the middle pair is
    averaged as Python floats rather than float32; the 95th is nearest rank.
    """
    n = samples.size
    if n == 0:
        return None
    ordered = np.sort(samples)
    median = (float(ordered[(n - 1) // 2]) + float(ordered[n // 2])) / 2
    return median, float(ordered[nearest_rank_index(n, 0.95)]), float(ordered[-1])


async def load_channel_samples(
    db: AsyncSession, session_ids: Sequence[int]
) -> ChannelSamples:
    """Usable ``fl`` and ``snore`` samples of ``session_ids``.

    Both channels are keys of the result.  A session is absent from a channel
    when it has no row for it, the blob is empty, or the blob is corrupt, which
    logs a warning.
    """
    samples: ChannelSamples = {channel: {} for channel in _CHANNEL_FLOORS}
    waveform = models.Waveform
    # The channel IN-list is a constant bound in every chunk.
    for chunk in iter_id_chunks(session_ids):
        rows = await db.execute(
            select(
                waveform.session_id,
                waveform.waveform_type,
                waveform.data_blob,
                waveform.sample_count,
            ).where(
                waveform.session_id.in_(chunk),
                waveform.waveform_type.in_(tuple(_CHANNEL_FLOORS)),
            )
        )
        for session_id, channel, blob, sample_count in rows:
            if not blob or not sample_count:
                continue
            try:
                _, values = deserialize_waveform_blob(blob, sample_count)
            except ValueError:
                logger.warning(
                    "Skipping corrupt %s waveform of session %d", channel, session_id
                )
                continue
            samples[channel][session_id] = usable_samples(channel, values)
    return samples


def pool_channel_stats(
    samples: ChannelSamples, session_ids: Sequence[int]
) -> DeviceChannelStats | None:
    """Statistics over the combined samples of ``session_ids``.

    ``None`` when no session has a usable sample on either channel.
    """
    fields: dict[str, float] = {}
    for channel, prefix in _CHANNEL_FIELD_PREFIXES.items():
        by_session = samples[channel]
        pooled = [by_session[sid] for sid in session_ids if sid in by_session]
        distribution = sample_distribution(np.concatenate(pooled)) if pooled else None
        if distribution is not None:
            for stat, value in zip(
                ("median", "95th", "max"), distribution, strict=True
            ):
                fields[f"{prefix}_{stat}"] = round(value, _ROUND_DIGITS)
    return DeviceChannelStats.model_validate(fields) if fields else None


def mask_on_spans(
    session_id: int, start_time: datetime, raw_segments: object
) -> list[MaskOnSpan] | None:
    """Wall-clock spans of a session's stored segments; ``None`` when unknown."""
    segments = parse_mask_on_segments(raw_segments, session_id)
    if segments is None:
        return None
    return [
        MaskOnSpan(
            session_id=session_id,
            start=start_time + timedelta(seconds=start_offset),
            end=start_time + timedelta(seconds=end_offset),
            duration_seconds=end_offset - start_offset,
        )
        for start_offset, end_offset in segments
    ]
