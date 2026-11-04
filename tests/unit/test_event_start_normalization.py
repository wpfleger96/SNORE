"""Regression tests: RespiratoryEvent.start_time is the true start when one is known.

ResMed EVE EDF annotations and OSCAR binary event lists both timestamp events at
their END.  The parsers compensate by subtracting the recorded duration so
RespiratoryEvent.start_time represents the true start.  An event with no
recorded duration has nothing to subtract: it keeps the device's time and a None
duration.

The CSL path already stores the true start (it pairs the "CSR Start" / "CSR End"
annotation texts itself and never sees an end-flagged timestamp), so it must NOT
receive any duration shift.

These tests guard all three paths against regressions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from snore.constants import CPAP_OBSTRUCTIVE
from snore.parsers.resmed_edf import ResmedEDFParser
from snore.parsers.unified import (
    DeviceInfo,
    RespiratoryEvent,
    RespiratoryEventType,
    UnifiedSession,
)
from tests.helpers.oscar_builders import (
    make_event_list,
    make_events,
    make_summary,
    parse_oscar_session,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(
    start: datetime = datetime(2025, 1, 1, 22, 0, 0),
    duration_hours: float = 7.0,
) -> UnifiedSession:
    device_info = DeviceInfo(
        manufacturer="ResMed",
        model="AirSense 11 APAP",
        serial_number="TESTEVENORM",
    )
    return UnifiedSession(
        device_info=device_info,
        start_time=start,
        end_time=start + timedelta(hours=duration_hours),
    )


def _write_annotation_edf(
    tmp_path: Path,
    filename: str,
    start_dt: datetime,
    annotations: list[tuple[float, float | None, str]],
) -> Path:
    """Create a minimal EDF+ annotation file.

    Each annotation is (onset_seconds, duration_seconds|None, text).
    """
    import pyedflib

    edf_path = tmp_path / filename
    with pyedflib.EdfWriter(str(edf_path), 0, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setStartdatetime(start_dt)
        f.setDatarecordDuration(1)
        for onset, duration, text in annotations:
            dur = duration if duration is not None else -1.0
            f.writeAnnotation(onset, dur, text)
    return edf_path


# ---------------------------------------------------------------------------
# EVE annotation path
# ---------------------------------------------------------------------------


class TestEVEEventStartNormalization:
    """EVE parsing: ResMed flags events at end; parser shifts to true start."""

    def test_eve_explicit_duration_stores_flag_time_minus_duration(self, tmp_path):
        """OA annotation at flag_time T with explicit duration D → start_time == T - D."""
        session_start = datetime(2025, 1, 1, 22, 0, 0)
        session = _make_session(start=session_start)
        parser = ResmedEDFParser()

        # Annotation: 60 seconds into the session, explicit duration 20 s.
        # The annotation onset is the flag time (end of event).
        flag_onset_s = 60.0
        duration_s = 20.0
        eve_file = _write_annotation_edf(
            tmp_path,
            "20250101_220000_EVE.edf",
            session_start,
            [
                (0.0, None, "Recording starts"),
                (flag_onset_s, duration_s, "Obstructive Apnea"),
            ],
        )

        parser._parse_eve_files_for_night([eve_file], session)

        events = [
            e
            for e in session.events
            if e.event_type == RespiratoryEventType.OBSTRUCTIVE_APNEA
        ]
        assert len(events) == 1
        flag_time = session_start + timedelta(seconds=flag_onset_s)
        assert events[0].start_time == flag_time - timedelta(seconds=duration_s)
        assert events[0].duration_seconds == pytest.approx(duration_s)

    @pytest.mark.parametrize(
        ("label", "event_type"),
        [
            ("Hypopnea", RespiratoryEventType.HYPOPNEA),
            ("Arousal", RespiratoryEventType.RERA),
        ],
    )
    def test_eve_no_duration_keeps_annotation_time(self, tmp_path, label, event_type):
        """An annotation with no duration keeps its time and gets no invented duration."""
        session_start = datetime(2025, 1, 1, 22, 0, 0)
        session = _make_session(start=session_start)
        parser = ResmedEDFParser()

        flag_onset_s = 300.0
        eve_file = _write_annotation_edf(
            tmp_path,
            "20250101_220000_EVE_nodur.edf",
            session_start,
            [
                (0.0, None, "Recording starts"),
                (flag_onset_s, None, label),
            ],
        )

        parser._parse_eve_files_for_night([eve_file], session)

        events = [e for e in session.events if e.event_type == event_type]
        assert len(events) == 1
        assert events[0].start_time == session_start + timedelta(seconds=flag_onset_s)
        assert events[0].duration_seconds is None

    def test_eve_no_duration_inside_session_first_seconds_is_kept(self, tmp_path):
        """A hypopnea flagged just after session start stays inside the session."""
        session_start = datetime(2025, 1, 1, 22, 0, 0)
        session = _make_session(start=session_start)
        eve_file = _write_annotation_edf(
            tmp_path,
            "20250101_220000_EVE_early.edf",
            session_start,
            [(0.0, None, "Recording starts"), (5.0, None, "Hypopnea")],
        )

        ResmedEDFParser()._parse_eve_files_for_night([eve_file], session)

        assert [e.start_time for e in session.events] == [
            session_start + timedelta(seconds=5)
        ]


# ---------------------------------------------------------------------------
# OSCAR binary event list path
# ---------------------------------------------------------------------------


class TestOSCAREventStartNormalization:
    """_populate_events_from_events: OSCAR timestamps are end-of-event; parser shifts to true start."""

    @staticmethod
    def _parse_event(tmp_path: Path, duration_s: int | None) -> RespiratoryEvent:
        """Parse a session with one OA flagged at 01:30:00 UTC wall clock."""
        # Session: 2024-01-15 00:30 UTC to 06:30 UTC
        first_ms = int(datetime(2024, 1, 15, 0, 30, tzinfo=UTC).timestamp() * 1000)
        last_ms = first_ms + 6 * 3600 * 1000

        # Event flagged 1 hour into the session.
        event_at_ms = first_ms + 3600 * 1000

        # No duration recorded leaves the event list without a value.
        data = [] if duration_s is None else [duration_s]
        events = make_events(
            first_ms, last_ms, make_event_list(CPAP_OBSTRUCTIVE, event_at_ms, data)
        )

        # No timezone: UTC wall clock.
        session = parse_oscar_session(tmp_path, make_summary(first_ms, last_ms), events)

        assert len(session.events) == 1
        return session.events[0]

    def test_oscar_event_flag_time_shifted_to_true_start(self, tmp_path):
        """OA event at flag_time T with duration D → start_time == T - D (UTC wall clock)."""
        event = self._parse_event(tmp_path, duration_s=30)

        assert event.duration_seconds == pytest.approx(30.0)
        assert event.start_time == datetime(2024, 1, 15, 1, 29, 30)

    @pytest.mark.parametrize("duration_s", [None, 0, -5])
    def test_oscar_missing_or_non_positive_duration_keeps_flag_time(
        self, tmp_path, duration_s
    ):
        """No usable duration: no shift and no invented value."""
        event = self._parse_event(tmp_path, duration_s=duration_s)

        assert event.duration_seconds is None
        assert event.start_time == datetime(2024, 1, 15, 1, 30, 0)


# ---------------------------------------------------------------------------
# CSL path guard
# ---------------------------------------------------------------------------


class TestCSLStartTimeUnchanged:
    """Guard: the CSL parser stores the CSR Start time as-is — no duration shift.

    The CSL path already pairs "CSR Start" / "CSR End" annotations and stores
    the true start directly.  Applying the EVE-style shift here would be wrong
    and would move the start BEFORE the CSR Start annotation.
    """

    def test_csl_pb_event_start_equals_csr_start_not_csr_end(self, tmp_path):
        """PB start_time == CSR Start onset, not CSR End and not CSR Start - duration."""
        session_start = datetime(2025, 6, 1, 23, 0, 0)
        device_info = DeviceInfo(
            manufacturer="ResMed",
            model="AirSense 11 APAP",
            serial_number="TESTGUARD",
        )
        session = UnifiedSession(
            device_info=device_info,
            start_time=session_start,
            end_time=session_start + timedelta(hours=7),
        )
        parser = ResmedEDFParser()

        # CSR span: starts 60 s in, ends 150 s in (duration 90 s).
        csr_start_onset = 60.0
        csr_end_onset = 150.0
        csl_file = _write_annotation_edf(
            tmp_path,
            "guard_CSL.edf",
            session_start,
            [
                (0.0, None, "Recording starts"),
                (csr_start_onset, None, "CSR Start"),
                (csr_end_onset, None, "CSR End"),
            ],
        )

        parser._parse_csl_files_for_night([csl_file], session)

        pb_events = [
            e
            for e in session.events
            if e.event_type == RespiratoryEventType.PERIODIC_BREATHING
        ]
        assert len(pb_events) == 1
        expected_start = session_start + timedelta(seconds=csr_start_onset)
        expected_end_time = session_start + timedelta(seconds=csr_end_onset)
        # True start is the CSR Start annotation time.
        assert pb_events[0].start_time == expected_start
        # Guard: must NOT be the CSR End time (which would mean the shift was misapplied).
        assert pb_events[0].start_time != expected_end_time
        assert pb_events[0].duration_seconds == pytest.approx(
            csr_end_onset - csr_start_onset
        )
