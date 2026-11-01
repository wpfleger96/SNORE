"""Unit tests for WaveformService."""

import logging
import sqlite3

from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from snore.analysis.modes.postprocess import EVENT_MATCH_TOLERANCE_SECONDS
from snore.analysis.modes.types import ModeResult
from snore.analysis.shared.types import ApneaEvent, HypopneaEvent, RERAEvent
from snore.analysis.types import AnalysisEvent, AnalysisResult
from snore.database.models import AnalysisResult as AnalysisResultRow
from snore.database.models import Device, Profile, Session, User, Waveform
from snore.exceptions import NotFoundError
from snore.services import waveform_service as waveform_service_module
from snore.services.schemas import EventComparisonResult
from snore.services.session_service import SessionService
from snore.services.waveform_service import WaveformService


def _make_waveform_blob(sample_count: int, sample_rate: float) -> bytes:
    """Create minimal valid waveform blob for testing."""
    timestamps = np.arange(sample_count, dtype=np.float32) / sample_rate
    values = np.sin(timestamps * 2 * np.pi * 0.2).astype(np.float32)
    data = np.column_stack([timestamps, values])
    return data.tobytes()


class TestWaveformService:
    """Tests for WaveformService."""

    async def test_list_waveforms_empty(self, async_db_session, async_test_device):
        """Empty session returns empty list."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_empty",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=1)
        result = await service.list_waveforms(session.id)

        assert result == []

    async def test_list_waveforms_with_data(self, async_db_session, async_test_device):
        """List returns correct WaveformInfo objects."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_with_data",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        wf1 = Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=25.0,
            unit="L/min",
            sample_count=1000,
            data_blob=_make_waveform_blob(1000, 25.0),
        )
        wf2 = Waveform(
            session_id=session.id,
            waveform_type="pressure",
            sample_rate=25.0,
            unit="cmH2O",
            sample_count=500,
            data_blob=_make_waveform_blob(500, 25.0),
        )
        async_db_session.add(wf1)
        async_db_session.add(wf2)
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=1)
        result = await service.list_waveforms(session.id)

        assert len(result) == 2
        assert result[0].waveform_type == "flow"
        assert result[0].sample_rate == 25.0
        assert result[0].sample_count == 1000
        assert result[0].unit == "L/min"
        assert result[1].waveform_type == "pressure"

    async def test_list_waveforms_duration_calculation(
        self, async_db_session, async_test_device
    ):
        """Duration hours computed correctly from sample_count/sample_rate."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_duration",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        wf = Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=25.0,
            unit="L/min",
            sample_count=90000,
            data_blob=_make_waveform_blob(90000, 25.0),
        )
        async_db_session.add(wf)
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=1)
        result = await service.list_waveforms(session.id)

        assert len(result) == 1
        expected_hours = 90000 / 25.0 / 3600
        assert result[0].duration_hours == pytest.approx(expected_hours, rel=1e-6)

    async def test_list_waveforms_null_sample_count_derives_from_blob(
        self, async_db_session, async_test_device
    ):
        """The listed count and duration come from the blob, not the stored count."""
        session = await _seed_session(async_db_session, async_test_device, "derived")
        wf = await _seed_waveform(async_db_session, session.id)
        wf.sample_count = None
        await async_db_session.flush()
        service = WaveformService(async_db_session, profile_id=1)

        [info] = await service.list_waveforms(session.id)

        assert info.sample_count == 1000
        assert info.duration_hours == pytest.approx(1000 / 25.0 / 3600)

    async def test_list_waveforms_partial_sample_blob_floors_count_and_warns(
        self, async_db_session, async_test_device, caplog
    ):
        """Only a blob with stray bytes warns, and it lists its whole samples."""
        session = await _seed_session(async_db_session, async_test_device, "stray")
        await _seed_waveform(
            async_db_session,
            session.id,
            blob=_make_waveform_blob(1000, 25.0) + bytes(3),
        )
        await _seed_waveform(async_db_session, session.id, waveform_type="pressure")
        service = WaveformService(async_db_session, profile_id=1)

        with caplog.at_level(logging.WARNING, logger=waveform_service_module.__name__):
            flow, pressure = await service.list_waveforms(session.id)

        assert (flow.sample_count, pressure.sample_count) == (1000, 1000)
        [warning] = [
            r for r in caplog.records if r.name == waveform_service_module.__name__
        ]
        assert warning.levelno == logging.WARNING
        message = warning.getMessage()
        assert f"session_id={session.id}" in message
        assert "type=flow" in message
        assert "stray_bytes=3" in message

    async def test_get_waveform_data_not_found(
        self, async_db_session, async_test_device
    ):
        """Invalid session raises ValueError."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_not_found",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=1)

        with pytest.raises(ValueError, match="Waveform not found"):
            await service.get_waveform_data(session.id, "nonexistent")

    async def test_get_waveform_data_success(self, async_db_session, async_test_device):
        """Load waveform returns correct arrays."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_load",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        sample_count = 1000
        sample_rate = 25.0
        wf = Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=sample_rate,
            unit="L/min",
            sample_count=sample_count,
            data_blob=_make_waveform_blob(sample_count, sample_rate),
        )
        async_db_session.add(wf)
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=1)
        timestamps, values, metadata = await service.get_waveform_data(
            session.id, "flow"
        )

        assert len(timestamps) == sample_count
        assert len(values) == sample_count
        assert isinstance(timestamps, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert metadata["sample_rate"] == sample_rate
        assert metadata["waveform_type"] == "flow"

    async def test_get_waveform_data_with_downsampling(
        self, async_db_session, async_test_device
    ):
        """Downsampling reduces array to max_points."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_downsample",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        sample_count = 10000
        sample_rate = 25.0
        wf = Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=sample_rate,
            unit="L/min",
            sample_count=sample_count,
            data_blob=_make_waveform_blob(sample_count, sample_rate),
        )
        async_db_session.add(wf)
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=1)
        max_points = 1000
        timestamps, values, metadata = await service.get_waveform_data(
            session.id, "flow", max_points=max_points
        )

        assert len(timestamps) == max_points
        assert len(values) == max_points
        assert isinstance(timestamps, np.ndarray)
        assert isinstance(values, np.ndarray)

    async def test_get_waveform_data_with_windowing(
        self, async_db_session, async_test_device
    ):
        """Windowing filters data to specified time range."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_window",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        sample_count = 1000
        sample_rate = 25.0
        wf = Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=sample_rate,
            unit="L/min",
            sample_count=sample_count,
            data_blob=_make_waveform_blob(sample_count, sample_rate),
        )
        async_db_session.add(wf)
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=1)

        timestamps, values, _ = await service.get_waveform_data(
            session.id, "flow", start_seconds=10.0, end_seconds=20.0
        )

        assert len(timestamps) > 0
        assert len(timestamps) < sample_count
        assert timestamps.min() >= 10.0
        assert timestamps.max() <= 20.0
        assert len(timestamps) == len(values)

    async def test_get_waveform_data_with_start_only(
        self, async_db_session, async_test_device
    ):
        """Windowing with only start_seconds filters correctly."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_window_start",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        sample_count = 1000
        sample_rate = 25.0
        wf = Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=sample_rate,
            unit="L/min",
            sample_count=sample_count,
            data_blob=_make_waveform_blob(sample_count, sample_rate),
        )
        async_db_session.add(wf)
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=1)

        timestamps, values, _ = await service.get_waveform_data(
            session.id, "flow", start_seconds=30.0
        )

        assert len(timestamps) > 0
        assert len(timestamps) < sample_count
        assert timestamps.min() >= 30.0
        assert len(timestamps) == len(values)

    async def test_get_waveform_data_with_end_only(
        self, async_db_session, async_test_device
    ):
        """Windowing with only end_seconds filters correctly."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_window_end",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        sample_count = 1000
        sample_rate = 25.0
        wf = Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=sample_rate,
            unit="L/min",
            sample_count=sample_count,
            data_blob=_make_waveform_blob(sample_count, sample_rate),
        )
        async_db_session.add(wf)
        await async_db_session.flush()

        service = WaveformService(async_db_session, profile_id=1)

        timestamps, values, _ = await service.get_waveform_data(
            session.id, "flow", end_seconds=10.0
        )

        assert len(timestamps) > 0
        assert len(timestamps) < sample_count
        assert timestamps.max() <= 10.0
        assert len(timestamps) == len(values)


def _machine_event(start_time: float, event_type: str = "OA") -> AnalysisEvent:
    return AnalysisEvent(
        event_type=event_type, start_time=start_time, duration=10.0, source="machine"
    )


def _apnea(start_time: float, event_type: str = "OA") -> ApneaEvent:
    return ApneaEvent(
        start_time=start_time,
        end_time=start_time + 10.0,
        duration=10.0,
        event_type=event_type,
        flow_reduction=0.9,
        confidence=0.8,
        baseline_flow=30.0,
    )


def _hypopnea(start_time: float) -> HypopneaEvent:
    return HypopneaEvent(
        start_time=start_time,
        end_time=start_time + 10.0,
        duration=10.0,
        flow_reduction=0.4,
        confidence=0.7,
        baseline_flow=30.0,
    )


def _rera(start_time: float) -> RERAEvent:
    return RERAEvent(
        start_time=start_time,
        end_time=start_time + 12.0,
        duration=12.0,
        obstructed_breath_count=3,
        recovery_amplitude_increase_pct=0.6,
        confidence=0.65,
        baseline_flow=30.0,
    )


def _make_analysis_result(
    machine_events: list[AnalysisEvent] | None = None,
    apneas: list[ApneaEvent] | None = None,
    hypopneas: list[HypopneaEvent] | None = None,
    mode: str = "aasm",
    reras: list[RERAEvent] | None = None,
) -> SimpleNamespace:
    mode_result = SimpleNamespace(
        apneas=apneas or [],
        hypopneas=hypopneas or [],
        reras=reras or [],
    )
    return SimpleNamespace(
        machine_events=machine_events or [],
        mode_results={mode: mode_result},
    )


def _stored_analysis_row(
    session: Session, apneas: list[ApneaEvent]
) -> AnalysisResultRow:
    """An analysis row of ``session`` scoring one machine apnea at 100s."""
    result = AnalysisResult(
        session_id=session.id,
        session_duration_hours=8.0,
        total_breaths=1000,
        machine_events=[_machine_event(100.0)],
        mode_results={
            "aasm": ModeResult(
                mode_name="aasm", apneas=apneas, hypopneas=[], ahi=0.0, rdi=0.0
            )
        },
    )
    return AnalysisResultRow(
        session_id=session.id,
        timestamp_start=session.start_time,
        timestamp_end=session.end_time,
        programmatic_result_json=result.model_dump(mode="json"),
        processing_time_ms=0,
        created_at=datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def compare(async_db_session, async_test_device, monkeypatch):
    """Run ``compare_events`` on a fresh session against the given analysis."""

    async def run(
        machine_events: list[AnalysisEvent] | None = None,
        apneas: list[ApneaEvent] | None = None,
        hypopneas: list[HypopneaEvent] | None = None,
        reras: list[RERAEvent] | None = None,
    ) -> EventComparisonResult:
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_compare",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()
        monkeypatch.setattr(
            WaveformService,
            "_load_analysis_result",
            AsyncMock(
                return_value=_make_analysis_result(
                    machine_events, apneas, hypopneas, reras=reras
                )
            ),
        )
        service = WaveformService(async_db_session, profile_id=1)
        return await service.compare_events(session.id)

    return run


class TestCompareEvents:
    async def test_no_analysis_result_raises_not_found(
        self, async_db_session, async_test_device, monkeypatch
    ):
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_compare_none",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        monkeypatch.setattr(
            WaveformService,
            "_load_analysis_result",
            AsyncMock(return_value=None),
        )
        service = WaveformService(async_db_session, profile_id=1)
        with pytest.raises(NotFoundError, match="No analysis results"):
            await service.compare_events(session.id)

    async def test_missing_mode_raises_not_found(
        self, async_db_session, async_test_device, monkeypatch
    ):
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_compare_mode",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        fake_result = _make_analysis_result(mode="aasm")
        monkeypatch.setattr(
            WaveformService,
            "_load_analysis_result",
            AsyncMock(return_value=fake_result),
        )
        service = WaveformService(async_db_session, profile_id=1)
        with pytest.raises(NotFoundError, match="Mode.*not found"):
            await service.compare_events(session.id, mode="resmed")

    async def test_foreign_session_raises_not_found(
        self, async_db_session, async_test_device
    ):
        session = await _seed_session(async_db_session, async_test_device, "foreign")
        async_db_session.add(_stored_analysis_row(session, [_apnea(100.0)]))
        other_user = User(canonical_email="other@example.com", role="admin")
        async_db_session.add(other_user)
        await async_db_session.flush()
        other_profile = Profile(user_id=other_user.id, name="Other Profile")
        async_db_session.add(other_profile)
        await async_db_session.flush()

        owner = WaveformService(async_db_session, async_test_device.profile_id)
        assert (await owner.compare_events(session.id)).matched_events == 1

        stranger = WaveformService(async_db_session, other_profile.id)
        with pytest.raises(NotFoundError, match=f"^Session {session.id} not found$"):
            await stranger.compare_events(session.id)

    async def test_same_timestamp_analyses_use_the_highest_id(
        self, async_db_session, async_test_device
    ):
        session = await _seed_session(async_db_session, async_test_device, "tied")
        async_db_session.add(_stored_analysis_row(session, [_apnea(100.0)]))
        await async_db_session.flush()
        async_db_session.add(_stored_analysis_row(session, []))
        await async_db_session.flush()

        service = WaveformService(async_db_session, async_test_device.profile_id)
        result = await service.compare_events(session.id)

        assert result.programmatic_event_count == 0

    async def test_no_events_returns_empty(self, compare):
        result = await compare()

        assert result.machine_event_count == 0
        assert result.programmatic_event_count == 0
        assert result.matched_events == 0
        assert result.false_negatives == []
        assert result.false_positives_apnea == []
        assert result.false_positives_hypopnea == []

    async def test_unmatched_events_keep_their_detail(self, compare):
        result = await compare(
            machine_events=[_machine_event(100.0)],
            apneas=[_apnea(1000.0, event_type="CA")],
            hypopneas=[_hypopnea(2000.0)],
        )

        assert [
            (e.event_type, e.start_time, e.source, e.confidence, e.flow_reduction)
            for e in result.false_negatives
        ] == [("OA", 100.0, "machine", None, None)]
        assert [
            (e.event_type, e.start_time, e.source, e.confidence, e.flow_reduction)
            for e in result.false_positives_apnea
        ] == [("CA", 1000.0, "programmatic", 0.8, 0.9)]
        assert [
            (e.event_type, e.start_time, e.source, e.confidence, e.flow_reduction)
            for e in result.false_positives_hypopnea
        ] == [("H", 2000.0, "programmatic", 0.7, 0.4)]

    async def test_missed_device_hypopnea_reports_type_h(self, compare):
        result = await compare(machine_events=[_machine_event(100.0, "H")])

        assert [e.event_type for e in result.false_negatives] == ["H"]

    async def test_matched_events_are_counted_and_not_listed(self, compare):
        result = await compare(
            machine_events=[_machine_event(100.0), _machine_event(200.0, "H")],
            apneas=[_apnea(100.0)],
            hypopneas=[_hypopnea(200.0)],
        )

        assert result.matched_events == 2
        assert result.false_negatives == []
        assert result.false_positives_apnea == []
        assert result.false_positives_hypopnea == []

    async def test_apnea_and_hypopnea_match_across_types(self, compare):
        """Both sides are pooled, so a device OA can pair with a SNORE hypopnea."""
        result = await compare(
            machine_events=[_machine_event(100.0)], hypopneas=[_hypopnea(101.0)]
        )

        assert result.matched_events == 1
        assert result.false_negatives == []
        assert result.false_positives_hypopnea == []

    @pytest.mark.parametrize("event_type", ["FL", "LL", "VS", "PB"])
    async def test_non_matchable_device_types_are_ignored(self, compare, event_type):
        result = await compare(
            machine_events=[_machine_event(99.0, event_type), _machine_event(100.0)],
            apneas=[_apnea(100.0)],
        )

        assert result.machine_event_count == 1
        assert result.matched_events == 1
        assert result.false_negatives == []

    async def test_pooled_matching_follows_start_time_order(self, compare):
        """Matching the apnea first would take the 101s machine event and
        strand the hypopnea, which only 101s can reach."""
        result = await compare(
            machine_events=[_machine_event(101.0), _machine_event(108.0)],
            apneas=[_apnea(104.0)],
            hypopneas=[_hypopnea(97.0)],
        )

        assert result.matched_events == 2
        assert result.false_negatives == []
        assert result.false_positives_apnea == []
        assert result.false_positives_hypopnea == []

    async def test_device_reras_are_matched_and_misses_listed(self, compare):
        result = await compare(
            machine_events=[_machine_event(100.0, "RE"), _machine_event(500.0, "RE")],
            reras=[_rera(101.0), _rera(900.0)],
        )

        assert result.machine_event_count == 2
        assert result.programmatic_event_count == 2
        assert result.matched_events == 1
        assert [(e.event_type, e.start_time) for e in result.false_negatives] == [
            ("RE", 500.0)
        ]
        assert [
            (e.event_type, e.start_time, e.source, e.confidence, e.flow_reduction)
            for e in result.false_positives_rera
        ] == [("RE", 900.0, "programmatic", 0.65, None)]
        assert (
            result.matched_events
            + len(result.false_positives_apnea)
            + len(result.false_positives_hypopnea)
            + len(result.false_positives_rera)
            == result.programmatic_event_count
        )

    async def test_programmatic_reras_are_ignored_when_device_scored_none(
        self, compare
    ):
        result = await compare(
            machine_events=[_machine_event(100.0)],
            apneas=[_apnea(100.0)],
            reras=[_rera(300.0)],
        )

        assert result.machine_event_count == 1
        assert result.programmatic_event_count == 1
        assert result.matched_events == 1
        assert result.false_positives_rera == []

    async def test_second_programmatic_event_near_one_machine_event_is_false_positive(
        self, compare
    ):
        result = await compare(
            machine_events=[_machine_event(100.0)],
            apneas=[_apnea(99.0), _apnea(101.0)],
        )

        assert result.matched_events == 1
        assert [e.start_time for e in result.false_positives_apnea] == [101.0]
        assert result.false_negatives == []

    async def test_second_machine_event_near_one_programmatic_event_is_false_negative(
        self, compare
    ):
        result = await compare(
            machine_events=[_machine_event(99.0), _machine_event(101.0)],
            apneas=[_apnea(100.0)],
        )

        assert result.matched_events == 1
        assert [e.start_time for e in result.false_negatives] == [101.0]
        assert result.false_positives_apnea == []

    @pytest.mark.parametrize(
        ("offset", "expected_matched"),
        [
            (EVENT_MATCH_TOLERANCE_SECONDS, 1),
            (EVENT_MATCH_TOLERANCE_SECONDS + 0.001, 0),
        ],
    )
    async def test_start_difference_at_tolerance_boundary(
        self, compare, offset, expected_matched
    ):
        result = await compare(
            machine_events=[_machine_event(100.0)], apneas=[_apnea(100.0 + offset)]
        )

        assert result.matched_events == expected_matched

    async def test_zero_machine_events_makes_every_detection_a_false_positive(
        self, compare
    ):
        result = await compare(apneas=[_apnea(100.0)], hypopneas=[_hypopnea(200.0)])

        assert result.machine_event_count == 0
        assert result.programmatic_event_count == 2
        assert result.matched_events == 0
        assert len(result.false_positives_apnea) == 1
        assert len(result.false_positives_hypopnea) == 1

    async def test_get_waveform_data_leaves_session_open_for_subsequent_queries(
        self, async_db_session, async_test_device
    ):
        """get_waveform_data must NOT close the injected session.

        After fetching waveform data, the caller must be able to make further
        DB queries on the same session (e.g., loading analysis overlays in
        'waveform show').  A closed session raises InvalidRequestError on any
        subsequent execute().

        Regression guard: WaveformService previously called
        ``await self.db_session.close()`` at the end of the I/O phase,
        invalidating the caller's session.
        """

        import numpy as np

        from snore.database.models import Session, Waveform
        from snore.services.waveform_service import WaveformService

        now = datetime(2025, 1, 15, 22, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_overlay_session",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        sample_count = 100
        sample_rate = 25.0
        timestamps = np.arange(sample_count, dtype=np.float32) / sample_rate
        values = np.zeros(sample_count, dtype=np.float32)
        data = np.column_stack([timestamps, values])
        wf = Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=sample_rate,
            unit="L/min",
            sample_count=sample_count,
            data_blob=data.tobytes(),
        )
        async_db_session.add(wf)
        await async_db_session.flush()

        service = WaveformService(
            async_db_session, profile_id=async_test_device.profile_id
        )

        # Fetch waveform data — must NOT close the session.
        ts, vals, meta = await service.get_waveform_data(
            session_id=session.id,
            waveform_type="flow",
        )
        assert len(ts) == sample_count

        # The session must still be usable for subsequent queries (overlay loading).
        # This would raise InvalidRequestError if the session had been closed.
        from sqlalchemy import select

        from snore.database.models import Waveform as WaveformModel

        row = (
            (
                await async_db_session.execute(
                    select(WaveformModel).where(WaveformModel.session_id == session.id)
                )
            )
            .scalars()
            .first()
        )
        assert row is not None, (
            "Session was closed after get_waveform_data — subsequent query failed"
        )


async def _seed_session(
    async_db_session: AsyncSession, device: Device, tag: str
) -> Session:
    now = datetime(2025, 1, 1, 0, 0, 0)
    session = Session(
        device_id=device.id,
        device_session_id=tag,
        start_time=now,
        end_time=now + timedelta(hours=8),
        duration_seconds=28800,
    )
    async_db_session.add(session)
    await async_db_session.flush()
    return session


async def _seed_waveform(
    async_db_session: AsyncSession,
    session_id: int,
    *,
    waveform_type: str = "flow",
    sample_count: int = 1000,
    blob: bytes | None = None,
) -> Waveform:
    wf = Waveform(
        session_id=session_id,
        waveform_type=waveform_type,
        sample_rate=25.0,
        unit="L/min",
        sample_count=sample_count,
        data_blob=blob if blob is not None else _make_waveform_blob(sample_count, 25.0),
    )
    async_db_session.add(wf)
    await async_db_session.flush()
    return wf


def _constant_blob(sample_count: int, value: float) -> bytes:
    timestamps = np.arange(sample_count, dtype=np.float32) / 25.0
    values = np.full(sample_count, value, dtype=np.float32)
    return np.column_stack([timestamps, values]).astype(np.float32).tobytes()


def _zero_overflow_pages(db_path: Path, blob: bytes) -> int:
    """Zero ``blob``'s overflow pages so any read that follows the chain fails.

    SQLite reports such a read as corruption, which lets a test prove a query
    never walked the chain.  An overflow page is a next-page pointer followed by
    payload, so this zeroes each page whose payload start matches ``blob``; a
    short final page may not match.  The blob's ascending timestamps make a match
    unique to its pages.  Returns how many pages were zeroed.
    """
    pointer_size = 4
    probe_size = 64
    image = bytearray(db_path.read_bytes())
    page_size = int.from_bytes(image[16:18], "big")
    zeroed = 0
    for offset in range(0, len(image), page_size):
        payload_start = offset + pointer_size
        if bytes(image[payload_start : payload_start + probe_size]) in blob:
            image[offset : offset + page_size] = bytes(page_size)
            zeroed += 1
    db_path.write_bytes(image)
    return zeroed


async def _cache_then_delete_without_clear(
    async_db_session: AsyncSession, device: Device
) -> int:
    """Cache a session's arrays, delete its rows, and leave the cache alone.

    A separate ``snore import`` process deletes rows this way: it cannot reach
    this process's cache.  Returns the freed ``Waveform.id``, the table's max
    rowid, so SQLite hands it to the next insert.
    """
    session = await _seed_session(async_db_session, device, "deleted")
    waveform = await _seed_waveform(async_db_session, session.id)
    freed_id = waveform.id
    service = WaveformService(async_db_session, profile_id=device.profile_id)
    _, cached_vals, _ = await service.get_waveform_data(session.id, "flow")
    assert not np.allclose(cached_vals, 42.0)  # sin data, not the sentinel

    await async_db_session.delete(session)
    await async_db_session.flush()

    assert len(waveform_service_module._waveform_array_cache._entries) == 1
    return freed_id


class TestWaveformArrayCache:
    """The module-level deserialized-array cache in get_waveform_data.

    The cache is reset between tests by the autouse ``_reset_waveform_array_cache``
    fixture in the root conftest.
    """

    async def test_warm_request_identical_to_cold(
        self, async_db_session, async_test_device
    ):
        """A cache-warm request returns byte-identical arrays and metadata."""
        session = await _seed_session(async_db_session, async_test_device, "warm")
        await _seed_waveform(async_db_session, session.id)
        service = WaveformService(async_db_session, profile_id=1)

        cold_ts, cold_vals, cold_meta = await service.get_waveform_data(
            session.id, "flow", start_seconds=5.0, end_seconds=20.0
        )
        warm_ts, warm_vals, warm_meta = await service.get_waveform_data(
            session.id, "flow", start_seconds=5.0, end_seconds=20.0
        )

        np.testing.assert_array_equal(cold_ts, warm_ts)
        np.testing.assert_array_equal(cold_vals, warm_vals)
        assert cold_meta == warm_meta
        assert cold_meta["sample_count"] == 1000  # full channel, not the window

    async def test_null_sample_count_on_miss_raises_and_caches_nothing(
        self, async_db_session, async_test_device
    ):
        """A NULL stored count fails validation; nothing bad enters the cache."""
        session = await _seed_session(async_db_session, async_test_device, "null")
        wf = await _seed_waveform(async_db_session, session.id)
        wf.sample_count = None
        await async_db_session.flush()
        service = WaveformService(async_db_session, profile_id=1)

        with pytest.raises(ValueError, match="Invalid waveform blob data"):
            await service.get_waveform_data(session.id, "flow")

        assert not waveform_service_module._waveform_array_cache._entries

    async def test_warm_request_skips_blob_fetch(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Second request for the same row does not re-read/deserialize the blob."""
        session = await _seed_session(async_db_session, async_test_device, "skip")
        await _seed_waveform(async_db_session, session.id)
        service = WaveformService(async_db_session, profile_id=1)

        # Deserialization happens only on a cold miss (the warm hit reuses the
        # cached arrays and never reads the blob), so its call count is a proxy
        # for whether the blob was fetched.
        calls = 0
        real_deser = waveform_service_module.deserialize_waveform_blob

        def counting_deser(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_deser(*args, **kwargs)

        monkeypatch.setattr(
            waveform_service_module, "deserialize_waveform_blob", counting_deser
        )

        await service.get_waveform_data(session.id, "flow")
        assert calls == 1  # cold miss reads and deserializes the blob once
        await service.get_waveform_data(session.id, "flow")
        assert calls == 1  # warm hit does not touch the blob

    async def test_replaced_row_served_fresh(self, async_db_session, async_test_device):
        """A deleted+reinserted waveform (new row id) is served fresh, not stale."""
        session = await _seed_session(async_db_session, async_test_device, "replace")
        # Seed flow first, then a decoy row that becomes the max rowid, so
        # deleting flow never frees the top rowid: SQLite reuses a rowid only for
        # the deleted max, so the reinserted flow row is guaranteed a NEW id.
        old_wf = await _seed_waveform(async_db_session, session.id)
        await _seed_waveform(async_db_session, session.id, waveform_type="pressure")
        service = WaveformService(async_db_session, profile_id=1)

        _, cold_vals, _ = await service.get_waveform_data(session.id, "flow")
        assert not np.allclose(cold_vals, 42.0)  # sin data, not the sentinel

        await async_db_session.delete(old_wf)
        await async_db_session.flush()
        new_wf = await _seed_waveform(
            async_db_session,
            session.id,
            blob=_constant_blob(1000, 42.0),
        )
        assert new_wf.id != old_wf.id

        _, fresh_vals, _ = await service.get_waveform_data(session.id, "flow")
        np.testing.assert_allclose(fresh_vals, 42.0)

    async def test_byte_cap_evicts_least_recently_used(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Exceeding the byte cap evicts the LRU entry; re-reading it re-fetches."""
        # 1000 float32 timestamps + 1000 values = 8000 bytes per waveform, so a
        # cap of 8000 holds exactly one deserialized channel.
        monkeypatch.setattr(
            waveform_service_module, "WAVEFORM_ARRAY_CACHE_MAX_BYTES", 8000
        )
        session_a = await _seed_session(async_db_session, async_test_device, "cap_a")
        session_b = await _seed_session(async_db_session, async_test_device, "cap_b")
        await _seed_waveform(async_db_session, session_a.id)
        await _seed_waveform(async_db_session, session_b.id)
        service = WaveformService(async_db_session, profile_id=1)

        calls = 0
        real_deser = waveform_service_module.deserialize_waveform_blob

        def counting_deser(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_deser(*args, **kwargs)

        monkeypatch.setattr(
            waveform_service_module, "deserialize_waveform_blob", counting_deser
        )

        await service.get_waveform_data(session_a.id, "flow")  # cache A
        await service.get_waveform_data(session_b.id, "flow")  # cache B, evict A
        assert calls == 2
        await service.get_waveform_data(session_a.id, "flow")  # A evicted → refetch
        assert calls == 3

    async def test_ownership_enforced_on_warm_hit(
        self, async_db_session, async_test_device
    ):
        """A foreign profile gets 404 even when the row's arrays are cached."""
        session = await _seed_session(async_db_session, async_test_device, "owned")
        await _seed_waveform(async_db_session, session.id)

        owner = WaveformService(async_db_session, profile_id=1)
        await owner.get_waveform_data(session.id, "flow")  # warm the cache

        intruder = WaveformService(async_db_session, profile_id=999)
        with pytest.raises(NotFoundError):
            await intruder.get_waveform_data(session.id, "flow")

    async def test_metadata_query_foreign_profile_not_found(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """The metadata query alone refuses a row the profile doesn't own."""
        session = await _seed_session(async_db_session, async_test_device, "scoped")
        await _seed_waveform(async_db_session, session.id)
        intruder = WaveformService(async_db_session, profile_id=999)
        monkeypatch.setattr(intruder, "_assert_session_owned", AsyncMock())

        with pytest.raises(NotFoundError, match="Waveform not found"):
            await intruder.get_waveform_data(session.id, "flow")

    async def test_blob_query_other_session_not_found(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """A resolved row id from another session is a 404, not that row's blob."""
        requested = await _seed_session(async_db_session, async_test_device, "asked")
        other = await _seed_session(async_db_session, async_test_device, "other")
        other_wf = await _seed_waveform(async_db_session, other.id)
        service = WaveformService(async_db_session, profile_id=1)
        monkeypatch.setattr(
            service,
            "_fetch_waveform_metadata",
            AsyncMock(
                return_value=(
                    {"waveform_id": other_wf.id},
                    (other_wf.id, other.import_date),
                )
            ),
        )

        with pytest.raises(NotFoundError, match="Waveform not found"):
            await service.get_waveform_data(requested.id, "flow")

    async def test_reused_row_id_served_fresh_without_clear(
        self, async_db_session, async_test_device
    ):
        """A reused row id serves the NEW row even though nothing cleared the cache."""
        freed_id = await _cache_then_delete_without_clear(
            async_db_session, async_test_device
        )

        new_session = await _seed_session(async_db_session, async_test_device, "new")
        new_wf = await _seed_waveform(
            async_db_session, new_session.id, blob=_constant_blob(1000, 42.0)
        )
        assert new_wf.id == freed_id  # SQLite reused the freed max rowid

        service = WaveformService(
            async_db_session, profile_id=async_test_device.profile_id
        )
        _, fresh_vals, _ = await service.get_waveform_data(new_session.id, "flow")
        np.testing.assert_allclose(fresh_vals, 42.0)

    async def test_reused_row_id_on_other_profile_served_own_data(
        self, async_db_session, async_test_device
    ):
        """A profile whose new row reuses an id never sees the old owner's arrays."""
        user = User(canonical_email="other@example.com", role="member")
        async_db_session.add(user)
        await async_db_session.flush()
        other_profile = Profile(user_id=user.id, name="Other")
        async_db_session.add(other_profile)
        await async_db_session.flush()
        other_device = Device(
            profile_id=other_profile.id,
            manufacturer="M",
            model="X",
            serial_number="SN_OTHER",
        )
        async_db_session.add(other_device)
        await async_db_session.flush()

        freed_id = await _cache_then_delete_without_clear(
            async_db_session, async_test_device
        )

        other_session = await _seed_session(async_db_session, other_device, "other")
        other_wf = await _seed_waveform(
            async_db_session, other_session.id, blob=_constant_blob(1000, 42.0)
        )
        assert other_wf.id == freed_id  # SQLite reused the freed max rowid

        service = WaveformService(async_db_session, profile_id=other_profile.id)
        _, own_vals, _ = await service.get_waveform_data(other_session.id, "flow")
        np.testing.assert_allclose(own_vals, 42.0)

    async def test_reused_rowid_after_delete_served_fresh(self, async_db_session):
        """delete_sessions frees the max rowid and its reuse serves the NEW row.

        Complements ``test_replaced_row_served_fresh`` (which keeps a decoy max
        row so no id is reused): here the cached waveform holds the max rowid and
        is removed via the real deletion path, so SQLite reuses its id for the
        next insert.
        """
        # Enable FK enforcement as the FIRST statement, before any transaction
        # begins (SQLite ignores this pragma mid-transaction), so the Core DELETE
        # in delete_sessions cascades to the session's waveforms and frees the
        # rowid.  Seed the profile/device manually for the same reason — the
        # shared fixtures would open the transaction first.
        await async_db_session.execute(text("PRAGMA foreign_keys=ON"))

        user = User(canonical_email="reuse@example.com", role="admin")
        async_db_session.add(user)
        await async_db_session.flush()
        profile = Profile(user_id=user.id, name="Reuse")
        async_db_session.add(profile)
        await async_db_session.flush()
        device = Device(
            profile_id=profile.id,
            manufacturer="M",
            model="X",
            serial_number="SN_REUSE",
        )
        async_db_session.add(device)
        await async_db_session.flush()

        session = await _seed_session(async_db_session, device, "reuse")
        old_wf = await _seed_waveform(async_db_session, session.id)
        old_id = old_wf.id

        service = WaveformService(async_db_session, profile_id=profile.id)
        _, cold_vals, _ = await service.get_waveform_data(session.id, "flow")
        assert not np.allclose(cold_vals, 42.0)  # warm the cache with sin data

        deleted = await SessionService(
            async_db_session, profile_id=profile.id
        ).delete_sessions([session.id])
        assert deleted == 1
        assert not waveform_service_module._waveform_array_cache._entries
        # The Core DELETE leaves the now-deleted row in the ORM identity map;
        # drop it so the reused-id insert below does not collide with it.
        async_db_session.expunge(old_wf)

        new_session = await _seed_session(async_db_session, device, "reuse2")
        new_wf = await _seed_waveform(
            async_db_session, new_session.id, blob=_constant_blob(1000, 42.0)
        )
        assert new_wf.id == old_id  # SQLite reused the freed max rowid

        _, fresh_vals, _ = await service.get_waveform_data(new_session.id, "flow")
        np.testing.assert_allclose(fresh_vals, 42.0)


class TestOverflowChainReads:
    """Metadata paths must not read past ``data_blob`` (#435).

    ``sample_count`` is stored after the blob, so reading it walks the blob's
    whole overflow chain even when the blob itself is never returned.
    """

    async def test_warm_read_and_list_succeed_with_zeroed_overflow_pages(
        self, async_db_session, async_test_device, temp_db
    ):
        """Warm metadata and listing work when any read of the chain would fail."""
        sample_count = 5000  # 40 KB, so the row spills onto overflow pages
        blob = _make_waveform_blob(sample_count, 25.0)
        session = await _seed_session(async_db_session, async_test_device, "zeroed")
        session_id = session.id
        profile_id = async_test_device.profile_id
        await _seed_waveform(
            async_db_session, session_id, sample_count=sample_count, blob=blob
        )
        await async_db_session.commit()
        warmer = WaveformService(async_db_session, profile_id=profile_id)
        await warmer.get_waveform_data(session_id, "flow")

        # Drop every connection so no page cache survives the on-disk patch.
        await async_db_session.close()
        await async_db_session.bind.dispose()
        assert _zero_overflow_pages(temp_db, blob) > 0

        # Control: the corruption is real, so the assertions below can't pass
        # vacuously.  "malformed" rules out OperationalError's "no such table".
        with closing(sqlite3.connect(temp_db)) as conn:
            with pytest.raises(sqlite3.DatabaseError, match="malformed"):
                conn.execute("SELECT sample_count FROM waveforms").fetchall()

        engine = create_async_engine(f"sqlite+aiosqlite:///{temp_db}")
        try:
            async with AsyncSession(engine) as fresh_session:
                service = WaveformService(fresh_session, profile_id=profile_id)
                timestamps, _, metadata = await service.get_waveform_data(
                    session_id, "flow", start_seconds=5.0, end_seconds=20.0
                )
                listed = await service.list_waveforms(session_id)
        finally:
            await engine.dispose()

        assert len(timestamps) < sample_count
        assert metadata["sample_count"] == sample_count
        assert [info.sample_count for info in listed] == [sample_count]
