"""
Builders for synthetic OSCAR binary session data.

Construct ``SessionSummary`` / ``SessionEvents`` models directly and run them
through ``OscarDeviceParser._parse_single_session`` with the .000/.001 file
readers patched out, so parser tests need no real OSCAR files.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from snore.parsers.oscar_device import OscarDeviceParser
from snore.parsers.types import EventList, EventListType, SessionEvents, SessionSummary
from snore.parsers.unified import DeviceInfo, UnifiedSession

OSCAR_FILE_MAGIC = 0xC73216AB


def make_summary(first_ms: int, last_ms: int, **fields: Any) -> SessionSummary:
    """Build a .000 summary spanning ``first_ms``..``last_ms``; ``fields`` set its dicts."""
    return SessionSummary(
        magic=OSCAR_FILE_MAGIC,
        version=10,
        file_type=0,
        machine_id=1,
        session_id=first_ms // 1000,
        first_timestamp=first_ms,
        last_timestamp=last_ms,
        **fields,
    )


def make_event_list(channel_id: int, at_ms: int, data: list[int]) -> EventList:
    """Build an event list holding one event flagged at ``at_ms``.

    ``data`` is the event's duration in seconds (``[]`` for none recorded).
    """
    return EventList(
        channel_id=channel_id,
        first_timestamp=at_ms,
        last_timestamp=at_ms,
        count=1,
        event_type=EventListType.EVENT,
        sample_rate=0.0,
        gain=1.0,
        offset=0.0,
        min_value=0.0,
        max_value=60.0,
        dimension="s",
        data=data,
        time_deltas=[0],
    )


def make_waveform_list(
    channel_id: int, first_ms: int, samples: list[int], sample_rate: float = 4.0
) -> EventList:
    """Build a waveform event list of ``samples`` starting at ``first_ms``."""
    return EventList(
        channel_id=channel_id,
        first_timestamp=first_ms,
        last_timestamp=first_ms + int(len(samples) * 1000 / sample_rate),
        count=len(samples),
        event_type=EventListType.WAVEFORM,
        sample_rate=sample_rate,
        gain=1.0,
        offset=0.0,
        min_value=0.0,
        max_value=30.0,
        dimension="",
        data=samples,
    )


def make_events(first_ms: int, last_ms: int, *event_lists: EventList) -> SessionEvents:
    """Build a .001 events file holding ``event_lists``, one per channel."""
    return SessionEvents(
        magic=OSCAR_FILE_MAGIC,
        version=10,
        file_type=1,
        machine_id=1,
        session_id=first_ms // 1000,
        first_timestamp=first_ms,
        last_timestamp=last_ms,
        event_lists={el.channel_id: [el] for el in event_lists},
    )


def parse_oscar_session(
    tmp_path: Path,
    summary: SessionSummary,
    events: SessionEvents | None = None,
    timezone_name: str | None = None,
) -> UnifiedSession:
    """Parse ``summary`` (and ``events``, if given) as one OSCAR session."""
    summary_path = tmp_path / "1.000"
    summary_path.touch()
    events_path = None
    if events is not None:
        events_path = tmp_path / "1.001"
        events_path.touch()

    with (
        patch("snore.parsers.oscar_device.parse_summary_file", return_value=summary),
        patch("snore.parsers.oscar_device.parse_events_file", return_value=events),
    ):
        return OscarDeviceParser()._parse_single_session(
            session_id=summary.session_id,
            summary_path=summary_path,
            events_path=events_path,
            device_info=DeviceInfo(
                manufacturer="OSCAR", model="Test", serial_number="TEST123"
            ),
            base_path=tmp_path,
            timezone_name=timezone_name,
        )
