"""OSCAR's running AHI, derived on read from the device-scored AHI events.

OSCAR's ``calcAHIGraph`` (``calcs.cpp:1023-1170``) steps through each session
from its first sample and, at each step, counts the AHI events flagged in the
trailing window (``AHIWindow``, 60 minutes by default, ``profiles.h:639``),
divided by the window's full length in hours, so values in a session's first
hour read low.  Its loop advances by the 30-second ``window_step`` twice per
pass (``calcs.cpp:1124``, ``:1146``), so it plots one value per minute, the
cadence used here.  An event counts when its flag time is inside the window,
both ends included (``Session::rangeCount``, ``session.cpp:1779-1839``).

Two differences are deliberate:

- OSCAR imports each ResMed mask-on segment as its own session
  (``resmed_loader.cpp:3266-3294``); here each mask-on segment is its own
  series, and its window never reaches into an earlier segment, which is what
  OSCAR plots.  A session whose segments are unknown is one segment, from its
  start to its end, and stored segments are clipped to that span.
- OSCAR ends each series with a zero at the session's last sample
  (``calcs.cpp:1150``), which only closes the plotted line; it is left out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import numpy as np

from pydantic import BaseModel

from snore.services.schemas import RunningAhiSegment
from snore.therapy_hours import parse_mask_on_segments

__all__ = ["RunningAhiSource", "running_ahi"]

RUNNING_AHI_WINDOW_SECONDS = 3600.0
RUNNING_AHI_STEP_SECONDS = 60.0


class RunningAhiSource(BaseModel):
    """A session the running AHI covers."""

    session_id: int
    start_time: datetime
    end_time: datetime
    # The stored ``sessions.mask_on_segments`` value, decoded but unvalidated.
    mask_on_segments: object


def running_ahi(
    sources: Sequence[RunningAhiSource],
    flags: Mapping[int, Sequence[tuple[str, float]]],
) -> list[RunningAhiSegment]:
    """The running AHI of every mask-on segment of ``sources``, by start.

    ``flags`` is ``ahi_event_flags`` for the sources' sessions.
    """
    segments: list[RunningAhiSegment] = []
    for source in sources:
        length = (source.end_time - source.start_time).total_seconds()
        spans = parse_mask_on_segments(source.mask_on_segments, source.session_id)
        if spans is None:
            spans = [(0.0, length)]
        times = np.array([offset for _, offset in flags[source.session_id]])
        for start, stored_end in spans:
            end = min(stored_end, length)
            if start >= end:
                continue
            steps = np.arange(start, end, RUNNING_AHI_STEP_SECONDS)
            in_segment = times[(times >= start) & (times <= end)]
            window_start = np.maximum(steps - RUNNING_AHI_WINDOW_SECONDS, start)
            counts = np.searchsorted(in_segment, steps, side="right") - np.searchsorted(
                in_segment, window_start, side="left"
            )
            segments.append(
                RunningAhiSegment(
                    session_id=source.session_id,
                    start=source.start_time + timedelta(seconds=start),
                    values=counts.tolist(),
                )
            )
    return sorted(segments, key=lambda segment: segment.start)
