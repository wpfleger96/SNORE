"""OSCAR's leak redline views, derived on read from the stored leak waveform.

Both use the 24 L/min redline (``LEAK_REDLINE_LPM``) and cover the same
sessions: those with at least one valid leak sample.

- Time over the redline: the share of the valid leak samples at or above it,
  the MCP nightly summary's ``leak_above_24_pct``.  OSCAR's Daily page shows
  "Time over leak redline" as time above it over the day's hours (OSCAR
  ``daily.cpp:1648-1652``), so the two agree only approximately.
- Large Leak spans: OSCAR's ``flagLargeLeaks`` (``calcs.cpp:1342-1405``).  The
  Daily page shows a span channel as its summed duration over the day's hours
  (``daily.cpp:1798-1800``), which is the Large Leak index here.

Span rules, as ``flagLargeLeaks`` applies them:

- a span opens at a sample at or above the redline whose previous sample was
  below it (OSCAR ``calcs.cpp:1381-1385``);
- it closes at the next sample below the redline, and that sample's time is the
  span's end (``calcs.cpp:1386-1391``);
- a span still open when the recording ends closes one sample period after the
  last sample (``calcs.cpp:1398-1404``): OSCAR's ResMed loader writes the
  channel's final event one period past its last sample
  (``resmed_loader.cpp:4375-4381``);
- there is no minimum duration.

Four edge cases deliberately differ:

- OSCAR closes a span only when the sample before the drop is strictly above the
  redline (``calcs.cpp:1386``, ``:1398``), so a span whose last sample is
  exactly at the redline is silently dropped; here it closes like any other.
- OSCAR replaces a ``-1`` sentinel with the previous value
  (``resmed_loader.cpp:4356``), so its spans run through sentinel samples; here
  a sentinel or NaN sample counts as below the redline and ends the span.
- OSCAR imports each ResMed mask-on segment as its own session; a SNORE session
  holds every segment, so a span also closes at a recording gap
  (``recording_gaps``), one sample period after the gap's last sample, and never
  covers mask-off time.
- Durations are float seconds; OSCAR truncates them to whole seconds
  (``calcs.cpp:1390``).
"""

from __future__ import annotations

import logging
import math

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from snore.constants import LEAK_REDLINE_LPM
from snore.database.waveform_blobs import decode_channel_series, recording_gaps
from snore.services.channel_stats import (
    ChannelHistogram,
    PooledHistogram,
    channel_histogram,
    pool_histograms,
    pooled_share_above,
)
from snore.services.schemas import (
    MAX_LARGE_LEAK_SPANS,
    LargeLeakSpan,
    LeakRedlineStats,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LeakSource",
    "large_leak_offsets",
    "leak_redline_pct",
    "load_leak_redline",
]

# One session's leak: its valid-sample histogram, and its Large Leak spans as
# (start, end) offsets in seconds from session start, one row per span.
type _SessionLeak = tuple[ChannelHistogram, np.ndarray]


class LeakSource(BaseModel):
    """A session the leak redline views may cover."""

    session_id: int
    start_time: datetime
    # The Large Leak index denominator's share from this session.
    mask_on_hours: float


def large_leak_offsets(
    timestamps: np.ndarray, values: np.ndarray, sample_rate: float
) -> np.ndarray:
    """``(start, end)`` offsets of the Large Leak spans of one session's samples.

    ``timestamps`` are the sample offsets from session start and ``values`` the
    raw leak samples in L/min; a non-finite sample counts as below the redline.
    Returns a ``(spans, 2)`` float array, ascending.  Raises ``ValueError`` when
    the sample rate is not positive or the offsets are not finite, non-negative
    and strictly increasing, since spans built on them would be wrong.
    """
    if not (math.isfinite(sample_rate) and sample_rate > 0):
        raise ValueError(f"unusable sample rate {sample_rate}")
    if timestamps.size == 0:
        return np.empty((0, 2))
    if (
        not np.all(np.isfinite(timestamps))
        or timestamps[0] < 0
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("sample offsets are not finite and strictly increasing")
    offsets = timestamps.astype(np.float64)
    # joined[i]: samples i and i + 1 are in the same recording segment.
    joined = ~recording_gaps(offsets, sample_rate)
    above = values >= LEAK_REDLINE_LPM
    # continued[i]: the span holding sample i also holds sample i + 1.
    continued = above[:-1] & above[1:] & joined
    first = np.flatnonzero(above & ~np.concatenate(([False], continued)))
    last = np.flatnonzero(above & ~np.concatenate((continued, [False])))
    # The first sample below the redline ends a span; when the segment ends
    # first, the span ends one sample period after its last sample.
    closed_by_drop = np.concatenate((joined, [False]))[last]
    following = offsets[np.minimum(last + 1, offsets.size - 1)]
    end = np.where(closed_by_drop, following, offsets[last] + 1.0 / sample_rate)
    return np.column_stack((offsets[first], end))


def _reduce_leak(
    channel: str, timestamps: np.ndarray, values: np.ndarray, sample_rate: float
) -> _SessionLeak:
    return (
        channel_histogram(channel, values),
        large_leak_offsets(timestamps, values, sample_rate),
    )


def leak_redline_pct(pooled: PooledHistogram) -> float | None:
    """Percent of pooled leak samples at or above the redline, to 0.1.

    ``None`` when there are no samples.
    """
    share = pooled_share_above(pooled, LEAK_REDLINE_LPM, inclusive=True)
    return None if share is None else round(100.0 * share, 1)


def _kept_spans(
    sources: dict[int, LeakSource], offsets_by_session: dict[int, np.ndarray]
) -> list[LargeLeakSpan]:
    """The longest ``MAX_LARGE_LEAK_SPANS`` spans, chronological, as models.

    Raises ``OverflowError`` when a span's offset leaves the wall clock.
    """
    session_ids = np.concatenate(
        [np.full(len(o), sid) for sid, o in offsets_by_session.items()]
    )
    offsets = np.concatenate(list(offsets_by_session.values()))
    kept = np.argsort(offsets[:, 0] - offsets[:, 1], kind="stable")
    spans = []
    for index in kept[:MAX_LARGE_LEAK_SPANS].tolist():
        source = sources[int(session_ids[index])]
        start, end = offsets[index].tolist()
        spans.append(
            LargeLeakSpan(
                session_id=source.session_id,
                offset_seconds=start,
                start=source.start_time + timedelta(seconds=start),
                end=source.start_time + timedelta(seconds=end),
                duration_seconds=end - start,
            )
        )
    return sorted(spans, key=lambda span: span.start)


async def load_leak_redline(
    db: AsyncSession, sources: Sequence[LeakSource]
) -> LeakRedlineStats | None:
    """The leak redline views over the sessions of ``sources`` with leak data.

    ``None`` when none has a valid leak sample, or when any one's leak data is
    unusable (an undecodable blob, a bad sample rate or bad sample times): the
    others alone would be a partial value.  No ownership check: callers pass
    sessions already scoped to the profile.
    """
    by_id = {source.session_id: source for source in sources}
    reductions, corrupt = await decode_channel_series(
        db, list(by_id), ("leak",), _reduce_leak
    )
    # A session contributes when it has a valid leak sample.
    contributing = {
        session_id: offsets
        for session_id, (histogram, offsets) in reductions["leak"].items()
        if histogram[1].size
    }
    leak_above_24_pct = leak_redline_pct(
        pool_histograms(
            [reductions["leak"][session_id][0] for session_id in contributing]
        )
    )
    if corrupt["leak"] or leak_above_24_pct is None:
        return None
    mask_on_seconds = 3600.0 * sum(by_id[sid].mask_on_hours for sid in contributing)
    durations = [offsets[:, 1] - offsets[:, 0] for offsets in contributing.values()]
    large_leak_seconds = float(sum(d.sum() for d in durations))
    try:
        spans = _kept_spans(by_id, contributing)
    except OverflowError:
        logger.warning(
            "Large Leak span offsets overflow the wall clock; leaving the leak "
            "redline out"
        )
        return None
    return LeakRedlineStats(
        leak_above_24_pct=leak_above_24_pct,
        large_leak_pct=(
            round(100.0 * large_leak_seconds / mask_on_seconds, 2)
            if mask_on_seconds > 0
            else None
        ),
        large_leak_span_count=sum(d.size for d in durations),
        large_leak_spans=spans,
    )
