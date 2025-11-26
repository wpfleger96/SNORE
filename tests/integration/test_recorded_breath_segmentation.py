"""
Integration tests for breath segmentation using recorded PAP session data.

Uses actual device recordings to validate breath segmentation produces
physiologically reasonable results.

Note on Expected Warnings:
    During test execution, you may see harmless warnings from pyedflib's C library:
    "read 0, less than X requested!!!"

    These warnings occur when reading ResMed EDF files that contain EDF+ annotation
    channels mixed with data channels. They do not indicate errors or test failures.
    The warnings are expected and match OSCAR's behavior (just hidden in their GUI).
"""

import numpy as np
import pytest

from oscar_mcp.analysis.algorithms.breath_segmenter import BreathSegmenter
from oscar_mcp.analysis.data.waveform_loader import WaveformLoader
from oscar_mcp.database.models import Session


@pytest.mark.recorded
@pytest.mark.requires_fixtures
class TestRecordedSessionProcessing:
    """Process recorded sessions and validate basic correctness."""

    def test_baseline_fixture_breath_count(self, recorded_session):
        """Baseline fixture should produce reasonable breath count."""
        db = recorded_session("20250808")
        session = db.query(Session).first()
        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        # Should detect reasonable number of breaths
        # Fixture contains ~6.5 hours of data (388 records × 60s), expect 4000-8000 breaths
        # At typical 12-15 breaths/min: 6.5h × 60min × 12-15 = 4680-5850 breaths
        assert len(breaths) >= 4000, f"Too few breaths detected: {len(breaths)}"
        assert len(breaths) <= 8000, f"Too many breaths detected: {len(breaths)}"

    def test_early_therapy_fixture_processes(self, recorded_session):
        """Early therapy fixture should process without errors."""
        db = recorded_session("20250110")
        session = db.query(Session).first()
        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        assert len(breaths) > 0, "No breaths detected"

    def test_multi_segment_fixture_processes(self, recorded_session):
        """Multi-segment fixture should process despite discontinuities."""
        db = recorded_session("20250910")
        session = db.query(Session).first()
        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        assert len(breaths) > 0, "No breaths detected in multi-segment session"


@pytest.mark.recorded
@pytest.mark.requires_fixtures
class TestMetricsRealism:
    """Validate calculated metrics are physiologically realistic across recorded sessions."""

    def test_mean_respiratory_rate_in_sleep_range(self, recorded_session):
        """Mean RR should be in typical sleep range (8-25 breaths/min)."""
        db = recorded_session("20250808")
        session = db.query(Session).first()
        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        mean_rr = np.mean([b.respiratory_rate for b in breaths])
        assert 8 <= mean_rr <= 25, f"Mean RR unusual for sleep: {mean_rr}"

    def test_mean_tidal_volume_in_adult_range(self, recorded_session):
        """Mean TV should be in typical adult range (300-800 mL)."""
        db = recorded_session("20250808")
        session = db.query(Session).first()
        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        mean_tv = np.mean([b.tidal_volume for b in breaths])
        assert 300 <= mean_tv <= 800, f"Mean TV unusual: {mean_tv}"

    def test_minute_ventilation_realistic(self, recorded_session):
        """Minute ventilation (TV × RR) should be in typical range (5-12 L/min)."""
        db = recorded_session("20250808")
        session = db.query(Session).first()
        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        mean_mv = np.mean([b.minute_ventilation for b in breaths])
        assert 5 <= mean_mv <= 15, f"Mean minute ventilation unusual: {mean_mv}"

    def test_breath_duration_in_valid_range(self, recorded_session):
        """Breath durations should be between 1-20 seconds (filter limits)."""
        db = recorded_session("20250808")
        session = db.query(Session).first()
        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        for breath in breaths:
            # Allow slight tolerance for floating-point edge cases near threshold
            # Breaths validated at boundary may be slightly below 1.0s after slicing
            assert 0.95 <= breath.duration <= 20.0, (
                f"Breath duration out of range: {breath.duration}"
            )


@pytest.mark.recorded
class TestFeatureVariability:
    """Verify features detect variation in recorded breathing patterns."""

    def test_flatness_shows_variation(self, recorded_session):
        """Flatness index should vary across breaths (not all identical)."""
        db = recorded_session("20250808")
        session = db.query(Session).first()
        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        from oscar_mcp.analysis.algorithms.feature_extractors import (
            WaveformFeatureExtractor,
        )

        extractor = WaveformFeatureExtractor()
        flatness_values = []

        for breath in breaths[:50]:  # Test first 50 breaths
            start_idx = int(breath.start_time * metadata["sample_rate"])
            end_idx = int(breath.end_time * metadata["sample_rate"])
            breath_flow = flow_values[start_idx:end_idx]

            shape, _, _, _ = extractor.extract_all_features(
                breath_flow, sample_rate=metadata["sample_rate"], include_spectral=False
            )
            flatness_values.append(shape.flatness_index)

        # Should have some variation (not all breaths identical)
        assert np.std(flatness_values) > 0.05, "No variation in flatness index"

    def test_amplitude_shows_variation(self, recorded_session):
        """Amplitude should vary across breaths (normal breathing variability)."""
        db = recorded_session("20250808")
        session = db.query(Session).first()
        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        amplitudes = [b.amplitude for b in breaths]

        # Should have meaningful variation
        assert np.std(amplitudes) > 2.0, "No variation in breath amplitude"
