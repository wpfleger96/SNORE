"""Waveform data inspection utilities."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from sqlalchemy.orm import Session

from snore.analysis.data.waveform_loader import WaveformLoader

if TYPE_CHECKING:
    from snore.analysis.shared.types import ApneaEvent, HypopneaEvent
    from snore.analysis.types import AnalysisEvent

    EventType = AnalysisEvent | ApneaEvent | HypopneaEvent


def parse_time_offset(time_str: str) -> float:
    """
    Parse time offset string to seconds.

    Args:
        time_str: Time in HH:MM:SS format

    Returns:
        Time in seconds from start of session
    """
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM:SS")

    try:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    except ValueError as e:
        raise ValueError(f"Invalid time format: {time_str}. {e}") from e

    return hours * 3600 + minutes * 60 + seconds


class WaveformInspector:
    """Inspector for waveform data extraction and event filtering."""

    def __init__(self, db_session: Session):
        """
        Initialize waveform inspector.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        self.loader = WaveformLoader(db_session)

    def get_window(
        self,
        session_id: int,
        center_seconds: float,
        window_seconds: float = 60.0,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Extract waveform data window centered on a specific time.

        Args:
            session_id: Session ID
            center_seconds: Center time in seconds from session start
            window_seconds: Window size in seconds (default: 60)

        Returns:
            Tuple of (timestamps, flow_values, metadata)
        """
        timestamps, flow_values, metadata = self.loader.load_waveform(
            session_id=session_id,
            waveform_type="flow",
            apply_filter=False,
        )

        start = center_seconds - window_seconds / 2
        end = center_seconds + window_seconds / 2

        mask = (timestamps >= start) & (timestamps <= end)
        windowed_timestamps = timestamps[mask]
        windowed_flow = flow_values[mask]

        return windowed_timestamps, windowed_flow, metadata

    def find_events_in_window(
        self,
        events: Sequence["EventType"],
        start_time: float,
        end_time: float,
    ) -> list["EventType"]:
        """
        Filter events to those within a time window.

        Args:
            events: List of analysis events
            start_time: Window start time in seconds
            end_time: Window end time in seconds

        Returns:
            List of events within window
        """
        windowed_events = []
        for event in events:
            event_end = event.start_time + event.duration
            if (
                start_time <= event.start_time <= end_time
                or start_time <= event_end <= end_time
                or (event.start_time <= start_time and event_end >= end_time)
            ):
                windowed_events.append(event)

        return windowed_events
