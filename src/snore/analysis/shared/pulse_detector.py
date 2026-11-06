"""
Pulse change detection for CPAP session analysis.

Implements OSCAR's calcPulseChange algorithm (calcs.cpp:1407-1480) which detects
rapid changes in heart rate during sleep. Pulse changes can indicate arousals or
other sleep disturbances.
"""

import logging

from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PulseChangeEvent:
    """
    Represents a detected pulse rate change event.

    Like OSCAR's flag events, the event is stamped at the change and its
    duration reaches back to the start of the scan window that saw it.

    Attributes:
        start_time: Time of the change in seconds from session start (the last
            qualifying sample in the scan window; OSCAR's ``lastt``)
        duration: Whole seconds from the scan-window start to the change
            (OSCAR's ``len``)
        bpm_delta: Absolute change in pulse rate (BPM) at the change
    """

    start_time: float
    duration: float
    bpm_delta: float


class PulseChangeDetector:
    """
    Detector for rapid pulse rate changes during sleep.

    Uses a sliding window approach matching OSCAR's algorithm: for each pulse
    data point, scan forward within a duration threshold for points that differ
    by more than bpm_threshold, then resume scanning after the change.

    Pulse data is typically 1Hz (~28,800 points for an 8h session), making this
    a trivially fast operation.
    """

    def __init__(
        self,
        bpm_threshold: float = 5.0,
        duration_threshold: float = 8.0,
    ):
        """
        Initialize pulse change detector.

        Args:
            bpm_threshold: BPM change that must be exceeded to count as an event
                (default 5.0)
            duration_threshold: Maximum time window in seconds to scan forward (default 8.0)
        """
        self.bpm_threshold = bpm_threshold
        self.duration_threshold = duration_threshold

    def detect(
        self,
        timestamps: np.ndarray,
        pulse_values: np.ndarray,
    ) -> list[PulseChangeEvent]:
        """
        Detect pulse change events in pulse rate waveform.

        Algorithm (from OSCAR calcs.cpp:1407-1480):
        1. For each pulse data point at time t
        2. Scan forward to points at most duration_threshold seconds later
        3. Points that differ by more than bpm_threshold BPM qualify (strict
           ``>``, calcs.cpp:1449); the last qualifying point is the change
        4. Record an event at the change's time (calcs.cpp:1461) and resume
           scanning after it (``i = li``, calcs.cpp:1462), so one heart-rate
           shift yields one event
        5. Skip NaN/artifact values (pulse data has dropouts)

        OSCAR stores the delta of the last sample scanned in the window as the
        event value; this reports the delta at the change instead.

        Args:
            timestamps: Ascending timestamp offsets in seconds from session start
            pulse_values: Array of pulse rate values in BPM

        Returns:
            List of PulseChangeEvent objects, sorted by start_time
        """
        if len(timestamps) == 0 or len(pulse_values) == 0:
            return []

        if len(timestamps) != len(pulse_values):
            logger.warning(
                f"Timestamp/value length mismatch: {len(timestamps)} vs {len(pulse_values)}"
            )
            return []

        valid_mask = ~np.isnan(pulse_values)
        if not np.any(valid_mask):
            logger.warning("No valid pulse data found (all NaN)")
            return []

        times = timestamps[valid_mask]
        bpm = pulse_values[valid_mask]
        # Exclusive end index of each point's scan window (OSCAR breaks once
        # time2 > time + window, so the window end is inclusive).
        window_ends = np.searchsorted(times, times + self.duration_threshold, "right")

        events: list[PulseChangeEvent] = []
        i = 0
        while i < len(times):
            deltas = np.abs(bpm[i + 1 : window_ends[i]] - bpm[i])
            qualifying = np.flatnonzero(deltas > self.bpm_threshold)
            if len(qualifying) > 0:
                change_idx = i + 1 + int(qualifying[-1])
                change_time = float(times[change_idx])
                events.append(
                    PulseChangeEvent(
                        start_time=change_time,
                        duration=float(int(change_time - float(times[i]))),
                        bpm_delta=float(deltas[qualifying[-1]]),
                    )
                )
                i = change_idx
            i += 1

        logger.debug(f"Detected {len(events)} pulse change events")

        return events
