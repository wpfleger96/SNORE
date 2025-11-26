"""
Integration tests for programmatic analysis engine.
"""

import numpy as np
import pytest

from oscar_mcp.analysis.engines.programmatic_engine import (
    ProgrammaticAnalysisEngine,
    ProgrammaticAnalysisResult,
)


class TestProgrammaticAnalysisEngine:
    """Test complete programmatic analysis pipeline."""

    @pytest.fixture
    def engine(self):
        return ProgrammaticAnalysisEngine(
            min_breath_duration=1.0, min_event_duration=10.0, confidence_threshold=0.6
        )

    def test_engine_initialization(self, engine):
        assert engine is not None
        assert engine.breath_segmenter is not None
        assert engine.feature_extractor is not None
        assert engine.flow_classifier is not None
        assert engine.event_detector is not None
        assert engine.pattern_detector is not None

    def test_analyze_session_basic(self, engine):
        timestamps = np.arange(0, 300, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert isinstance(result, ProgrammaticAnalysisResult)
        assert result.session_id == 123
        assert result.duration_hours > 0
        assert result.total_breaths > 0

    def test_analyze_session_with_apneas(self, engine):
        timestamps = np.arange(0, 300, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        flow_values[1250:1500] = 1.0
        flow_values[3000:3250] = 1.0

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert result.total_breaths > 0
        assert "event_timeline" in result.event_timeline or isinstance(
            result.event_timeline, dict
        )

    def test_analyze_session_with_spo2(self, engine):
        timestamps = np.arange(0, 300, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)
        spo2_values = np.ones(len(timestamps)) * 95.0

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
            spo2_values=spo2_values,
        )

        assert result is not None
        assert result.total_breaths > 0

    def test_analyze_session_short_duration(self, engine):
        timestamps = np.arange(0, 30, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert result is not None
        assert result.duration_hours > 0

    def test_result_contains_all_fields(self, engine):
        timestamps = np.arange(0, 300, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert hasattr(result, "session_id")
        assert hasattr(result, "timestamp_start")
        assert hasattr(result, "timestamp_end")
        assert hasattr(result, "duration_hours")
        assert hasattr(result, "flow_analysis")
        assert hasattr(result, "event_timeline")
        assert hasattr(result, "total_breaths")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "confidence_summary")
        assert hasattr(result, "clinical_summary")

    def test_processing_time_recorded(self, engine):
        timestamps = np.arange(0, 300, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert result.processing_time_ms > 0

    def test_confidence_summary_present(self, engine):
        timestamps = np.arange(0, 300, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert isinstance(result.confidence_summary, dict)

    def test_clinical_summary_present(self, engine):
        timestamps = np.arange(0, 300, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert isinstance(result.clinical_summary, str)
        assert len(result.clinical_summary) > 0

    def test_flow_analysis_dictionary(self, engine):
        timestamps = np.arange(0, 300, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert isinstance(result.flow_analysis, dict)

    def test_event_timeline_dictionary(self, engine):
        timestamps = np.arange(0, 300, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert isinstance(result.event_timeline, dict)

    def test_timestamp_range_correct(self, engine):
        timestamps = np.arange(100, 400, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        assert result.timestamp_start == pytest.approx(timestamps[0], abs=0.1)
        assert result.timestamp_end == pytest.approx(timestamps[-1], abs=0.1)

    def test_duration_calculation(self, engine):
        timestamps = np.arange(0, 3600, 0.04)
        flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

        result = engine.analyze_session(
            session_id=123,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=25.0,
        )

        expected_hours = (timestamps[-1] - timestamps[0]) / 3600.0
        assert result.duration_hours == pytest.approx(expected_hours, rel=0.01)

    def test_multiple_sessions(self, engine):
        for session_id in [1, 2, 3]:
            timestamps = np.arange(0, 300, 0.04)
            flow_values = self._generate_synthetic_flow(timestamps, breaths_per_min=15)

            result = engine.analyze_session(
                session_id=session_id,
                timestamps=timestamps,
                flow_values=flow_values,
                sample_rate=25.0,
            )

            assert result.session_id == session_id

    def test_empty_timestamps_handled(self, engine):
        timestamps = np.array([])
        flow_values = np.array([])

        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            engine.analyze_session(
                session_id=123,
                timestamps=timestamps,
                flow_values=flow_values,
                sample_rate=25.0,
            )

    def test_mismatched_lengths_handled(self, engine):
        timestamps = np.arange(0, 100, 0.04)
        flow_values = np.arange(0, 50, 0.04)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises((ValueError, IndexError)):
                engine.analyze_session(
                    session_id=123,
                    timestamps=timestamps,
                    flow_values=flow_values,
                    sample_rate=25.0,
                )

    def _generate_synthetic_flow(
        self, timestamps: np.ndarray, breaths_per_min: float
    ) -> np.ndarray:
        breath_freq = breaths_per_min / 60.0
        phase = 2 * np.pi * breath_freq * timestamps

        flow = 30 * np.sin(phase)

        flow[flow < 0] *= 0.5

        noise = np.random.normal(0, 2, len(timestamps))

        return flow + noise


class TestEngineConfiguration:
    """Test engine configuration options."""

    def test_custom_min_breath_duration(self):
        engine = ProgrammaticAnalysisEngine(min_breath_duration=2.0)

        assert engine.breath_segmenter is not None

    def test_custom_min_event_duration(self):
        engine = ProgrammaticAnalysisEngine(min_event_duration=15.0)

        assert engine.event_detector is not None

    def test_custom_confidence_threshold(self):
        engine = ProgrammaticAnalysisEngine(confidence_threshold=0.7)

        assert engine.flow_classifier is not None

    def test_default_parameters(self):
        engine = ProgrammaticAnalysisEngine()

        assert engine is not None
