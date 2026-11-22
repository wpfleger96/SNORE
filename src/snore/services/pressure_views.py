"""OSCAR's Time at Pressure and the pressure, EPAP and leak 99.5th, on read.

Both read each session's ``SUMMARY_CHANNELS`` choice (``summary_channel_choices``),
the channel its stored median and 95th came from, and cover the sessions the
day's pooled median and 95th cover (``DayManager._pooled_percentiles``): those
whose stored statistics have a value for the prefix and mask-on time.  A
prefix whose covered sessions include one with no usable samples is null,
since the rest alone would be a partial value.

The 99.5th is OSCAR's Max column on ResMed (``daily.cpp:1534``,
``profiles.cpp:704``), computed here as linear ``np.percentile`` over the
pooled finite samples, the stored 95th's method, rather than OSCAR's
time-weighted percentile (``day.cpp:348-481``).

Time at Pressure follows ``Graphs/MinutesAtPressure.cpp``:

- buckets are 0.2 cmH2O wide (``INTERVALS_PER_CCMH2O``, ``:126``) and centred on
  multiples of 0.2, so a sample goes to bucket ``floor((p + 0.1) * 5)``
  (``rawToBucketId``, ``:298-302``), capped at 60 cmH2O (``:125``, ``:516``);
- a sample holds its bucket until the next sample, and a session's last sample
  adds nothing (``updateTimes``, ``:451-541``);
- pressure and EPAP samples in the first 10 seconds of each recording segment
  add nothing, as OSCAR's ResMed loader drops them while pressure settles
  (``resmed_loader.cpp:4284-4287``); the 99.5th keeps them, as the stored 95th
  does;
- an event counts in the bucket of the last pressure sample at or before its
  flag time, or of the first sample when flagged before any (``updateFlagData``,
  ``:323-357``), and not at all when flagged after the last sample; only AHI
  event types count, against pressure only, never EPAP (``:589-593``).

Differences that are deliberate:

- spans (CSR, Large Leak), which OSCAR shades behind the chart, are left out;
- a recording gap ends the time of the sample before it, and an event flagged
  inside one is not counted, since a SNORE session holds every mask-on segment
  where OSCAR makes each its own session (``resmed_loader.cpp:3266-3294``);
- a non-finite sample adds no time and an event at one is not counted, where
  OSCAR's ResMed loader carries the previous value through a sentinel
  (``resmed_loader.cpp:4351``);
- bucket positions are rounded to 1e-6 before flooring, so a sample of exactly
  9.9 lands in the 10.0 bucket whatever its float error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.day_manager import (
    POOLED_PERCENTILE_SOURCES,
    contributing_prefixes,
    has_usable_samples,
    summary_channel_choices,
)
from snore.database.models import Statistics
from snore.database.waveform_blobs import (
    check_series_offsets,
    decode_channel_series,
    recording_gaps,
)
from snore.services.event_service import EventFlags
from snore.services.schemas import TimeAtPressure

__all__ = ["PressureSource", "PressureViews", "load_pressure_views"]

_BUCKETS_PER_CMH2O = 5
_MAX_BUCKET = 60 * _BUCKETS_PER_CMH2O
_SETTLE_SECONDS = 10.0

# One session's stored series: (offsets in seconds, samples, sample rate).
type _Series = tuple[np.ndarray, np.ndarray, float]


class PressureSource(BaseModel):
    """A session the pressure views may cover."""

    session_id: int
    # The POOLED_PERCENTILE_SOURCES prefixes the session counts toward.
    prefixes: frozenset[str]

    @classmethod
    def of(
        cls, session_id: int, stats: Statistics | None, duration_seconds: float | None
    ) -> PressureSource:
        """The source for a session, from its stored statistics."""
        return cls(
            session_id=session_id,
            prefixes=(
                contributing_prefixes(stats, duration_seconds)
                if stats is not None
                else frozenset()
            ),
        )


class PressureViews(BaseModel):
    """What ``load_pressure_views`` derives; each value null when unusable."""

    pressure_99_5th: float | None = None
    epap_99_5th: float | None = None
    leak_99_5th: float | None = None
    time_at_pressure: TimeAtPressure | None = None


def _keep_series(
    _channel: str, timestamps: np.ndarray, values: np.ndarray, sample_rate: float
) -> _Series:
    check_series_offsets(timestamps, sample_rate)
    return timestamps.astype(np.float64), values.astype(np.float64), sample_rate


async def _covered_series(
    db: AsyncSession, sources: Sequence[PressureSource]
) -> dict[str, dict[int, _Series] | None]:
    """Prefix -> the covered sessions' series, or None when any is unusable."""
    usable: dict[tuple[str, int], _Series] = {}
    chosen = await summary_channel_choices(db, [s.session_id for s in sources])
    # Leak is decoded here as well as for the leak redline histogram on
    # purpose: its 99.5th needs the raw samples for the linear percentile
    # that matches the stored 95th, which a histogram cannot give.
    for channel, keys in chosen.items():
        values, _ = await decode_channel_series(
            db, [session_id for _, session_id in keys], [channel], _keep_series
        )
        for prefix, session_id in keys:
            series = values[channel].get(session_id)
            if series is not None and has_usable_samples(series[1]):
                usable[(prefix, session_id)] = series
    covered: dict[str, dict[int, _Series] | None] = {}
    for prefix in POOLED_PERCENTILE_SOURCES:
        ids = [s.session_id for s in sources if prefix in s.prefixes]
        complete = bool(ids) and all((prefix, sid) in usable for sid in ids)
        covered[prefix] = (
            {sid: usable[(prefix, sid)] for sid in ids} if complete else None
        )
    return covered


def _percentile_99_5(series: Mapping[int, _Series] | None) -> float | None:
    if series is None:
        return None
    samples = np.concatenate([v[np.isfinite(v)] for _, v, _ in series.values()])
    return float(np.percentile(samples, 99.5))


def _bucket_ids(values: np.ndarray) -> np.ndarray:
    """Bucket of each finite pressure in ``values``."""
    position = np.round((values + 0.1) * _BUCKETS_PER_CMH2O, 6)
    return np.clip(np.floor(position), 0, _MAX_BUCKET).astype(np.intp)


def _held_seconds(offsets: np.ndarray, sample_rate: float) -> np.ndarray:
    """Seconds each sample holds its bucket.

    Until the next sample, nothing for a segment's last sample, and nothing in
    a segment's first ``_SETTLE_SECONDS``.
    """
    gaps = recording_gaps(offsets, sample_rate)
    held = np.append(np.diff(offsets), 0.0)
    held[:-1][gaps] = 0.0
    starts = np.concatenate(([True], gaps))
    segment_start = offsets[starts][np.cumsum(starts) - 1]
    held[offsets - segment_start < _SETTLE_SECONDS] = 0.0
    return held


def _seconds_by_bucket(series: Mapping[int, _Series]) -> np.ndarray:
    total = np.zeros(_MAX_BUCKET + 1)
    for offsets, values, sample_rate in series.values():
        held = _held_seconds(offsets, sample_rate)
        finite = np.isfinite(values)
        total += np.bincount(
            _bucket_ids(values[finite]),
            weights=held[finite],
            minlength=_MAX_BUCKET + 1,
        )
    return total


def _events_by_bucket(
    series: Mapping[int, _Series], flags: EventFlags
) -> dict[str, np.ndarray]:
    counts: dict[str, np.ndarray] = {}
    for session_id, (offsets, values, sample_rate) in series.items():
        if not flags[session_id]:
            continue
        types = np.array([event_type for event_type, _ in flags[session_id]])
        times = np.array([offset for _, offset in flags[session_id]])
        # The last sample at or before each flag, or the first when before any.
        at = np.maximum(np.searchsorted(offsets, times, side="right") - 1, 0)
        gap_after = np.append(recording_gaps(offsets, sample_rate), False)
        in_gap = gap_after[at] & (times > offsets[at])
        kept = (times <= offsets[-1]) & ~in_gap & np.isfinite(values[at])
        buckets = _bucket_ids(values[at[kept]])
        for event_type in np.unique(types[kept]).tolist():
            counts.setdefault(event_type, np.zeros(_MAX_BUCKET + 1, dtype=np.int64))
            counts[event_type] += np.bincount(
                buckets[types[kept] == event_type], minlength=_MAX_BUCKET + 1
            )
    return counts


def _time_at_pressure(
    pressure: Mapping[int, _Series],
    epap: Mapping[int, _Series] | None,
    flags: EventFlags,
) -> TimeAtPressure | None:
    pressure_seconds = _seconds_by_bucket(pressure)
    epap_seconds = _seconds_by_bucket(epap) if epap is not None else None
    events = _events_by_bucket(pressure, flags)
    occupied = pressure_seconds > 0
    if epap_seconds is not None:
        occupied |= epap_seconds > 0
    for counts in events.values():
        occupied |= counts > 0
    if not occupied.any():
        return None
    first, last = np.flatnonzero(occupied)[[0, -1]].tolist()
    window = slice(first, last + 1)
    return TimeAtPressure(
        pressures=[k / _BUCKETS_PER_CMH2O for k in range(first, last + 1)],
        pressure_seconds=np.round(pressure_seconds[window], 1).tolist(),
        epap_seconds=(
            np.round(epap_seconds[window], 1).tolist()
            if epap_seconds is not None
            else None
        ),
        event_counts={
            event_type: events[event_type][window].tolist()
            for event_type in sorted(events)
        },
    )


async def load_pressure_views(
    db: AsyncSession,
    sources: Sequence[PressureSource],
    flags: EventFlags,
) -> PressureViews:
    """The 99.5ths and Time at Pressure over ``sources``.

    ``flags`` is ``ahi_event_flags`` for the sources' sessions.  No ownership
    check: callers pass sessions already scoped to the profile.
    """
    covered = await _covered_series(db, sources)
    pressure = covered["pressure"]
    return PressureViews(
        pressure_99_5th=_percentile_99_5(pressure),
        epap_99_5th=_percentile_99_5(covered["epap"]),
        leak_99_5th=_percentile_99_5(covered["leak"]),
        time_at_pressure=(
            _time_at_pressure(pressure, covered["epap"], flags)
            if pressure is not None
            else None
        ),
    )
