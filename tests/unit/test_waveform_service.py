"""Unit tests for WaveformService."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Device, Profile, Session, User, Waveform
from snore.exceptions import NotFoundError
from snore.services import waveform_service as waveform_service_module
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


def _make_event(
    start_time: float,
    duration: float = 10.0,
    event_type: str = "OA",
    confidence: float | None = None,
    flow_reduction: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        start_time=start_time,
        duration=duration,
        event_type=event_type,
        confidence=confidence,
        flow_reduction=flow_reduction,
    )


def _make_machine_event(
    start_time: float, duration: float = 10.0, event_type: str = "OA"
) -> SimpleNamespace:
    return SimpleNamespace(
        start_time=start_time,
        duration=duration,
        event_type=event_type,
    )


def _make_analysis_result(
    machine_events: list[SimpleNamespace] | None = None,
    apneas: list[SimpleNamespace] | None = None,
    hypopneas: list[SimpleNamespace] | None = None,
    mode: str = "aasm",
) -> SimpleNamespace:
    mode_result = SimpleNamespace(
        apneas=apneas or [],
        hypopneas=hypopneas or [],
    )
    return SimpleNamespace(
        machine_events=machine_events or [],
        mode_results={mode: mode_result},
    )


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

    async def test_no_events_returns_empty(
        self, async_db_session, async_test_device, monkeypatch
    ):
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_compare_empty",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        fake_result = _make_analysis_result()
        monkeypatch.setattr(
            WaveformService,
            "_load_analysis_result",
            AsyncMock(return_value=fake_result),
        )
        monkeypatch.setattr(
            "snore.analysis.utils.convert_machine_events",
            lambda events: ([], []),
        )

        service = WaveformService(async_db_session, profile_id=1)
        result = await service.compare_events(session.id)
        assert result.false_negatives == []
        assert result.false_positives_apnea == []
        assert result.false_positives_hypopnea == []

    async def test_false_negative_detected(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Machine event with no matching programmatic event → false negative."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_compare_fn",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        m_event = _make_machine_event(start_time=100.0, event_type="OA")
        fake_result = _make_analysis_result(machine_events=[m_event])
        monkeypatch.setattr(
            WaveformService,
            "_load_analysis_result",
            AsyncMock(return_value=fake_result),
        )
        monkeypatch.setattr(
            "snore.analysis.utils.convert_machine_events",
            lambda events: ([m_event], []),
        )

        service = WaveformService(async_db_session, profile_id=1)
        result = await service.compare_events(session.id)
        assert len(result.false_negatives) == 1
        assert result.false_negatives[0].start_time == 100.0

    async def test_false_positive_detected(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Programmatic event with no matching machine event → false positive."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_compare_fp",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        prog_event = _make_event(start_time=200.0, event_type="OA")
        fake_result = _make_analysis_result(apneas=[prog_event])
        monkeypatch.setattr(
            WaveformService,
            "_load_analysis_result",
            AsyncMock(return_value=fake_result),
        )
        monkeypatch.setattr(
            "snore.analysis.utils.convert_machine_events",
            lambda events: ([], []),
        )

        service = WaveformService(async_db_session, profile_id=1)
        result = await service.compare_events(session.id)
        assert len(result.false_positives_apnea) == 1
        assert result.false_positives_apnea[0].start_time == 200.0

    async def test_matching_events_excluded(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Events at same time should not appear in false positive/negative lists."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_compare_match",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        m_event = _make_machine_event(start_time=100.0)
        prog_event = _make_event(start_time=100.0, event_type="OA")
        fake_result = _make_analysis_result(
            machine_events=[m_event], apneas=[prog_event]
        )
        monkeypatch.setattr(
            WaveformService,
            "_load_analysis_result",
            AsyncMock(return_value=fake_result),
        )
        monkeypatch.setattr(
            "snore.analysis.utils.convert_machine_events",
            lambda events: ([m_event], []),
        )

        service = WaveformService(async_db_session, profile_id=1)
        result = await service.compare_events(session.id)
        assert result.false_negatives == []
        assert result.false_positives_apnea == []

    async def test_event_source_names_machine_or_programmatic(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Unmatched machine events are tagged "machine"; detected ones "programmatic"."""
        now = datetime(2025, 1, 1, 0, 0, 0)
        session = Session(
            device_id=async_test_device.id,
            device_session_id="test_compare_source",
            start_time=now,
            end_time=now + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        m_event = _make_machine_event(start_time=100.0)
        prog_apnea = _make_event(start_time=1000.0, event_type="CA")
        prog_hypopnea = _make_event(
            start_time=2000.0, confidence=0.8, flow_reduction=0.4
        )
        fake_result = _make_analysis_result(
            machine_events=[m_event], apneas=[prog_apnea], hypopneas=[prog_hypopnea]
        )
        monkeypatch.setattr(
            WaveformService,
            "_load_analysis_result",
            AsyncMock(return_value=fake_result),
        )
        monkeypatch.setattr(
            "snore.analysis.utils.convert_machine_events",
            lambda events: ([m_event], []),
        )

        service = WaveformService(async_db_session, profile_id=1)
        result = await service.compare_events(session.id)

        assert [e.source for e in result.false_negatives] == ["machine"]
        assert [e.source for e in result.false_positives_apnea] == ["programmatic"]
        assert [e.source for e in result.false_positives_hypopnea] == ["programmatic"]

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

    async def test_reused_rowid_after_delete_served_fresh(self, async_db_session):
        """delete_sessions clears the cache so a reused rowid serves the NEW row.

        Complements ``test_replaced_row_served_fresh`` (which keeps a decoy max
        row so no id is reused): here the cached waveform holds the max rowid and
        is removed via the real deletion path, so SQLite reuses its id for the
        next insert — the case id-keying alone cannot distinguish.
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
