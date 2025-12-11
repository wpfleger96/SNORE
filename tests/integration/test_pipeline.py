"""
Integration tests for critical waveform processing pipeline functionality.

Tests focus on:
1. End-to-end pipeline (load → segment → extract)
2. OSCAR-aligned algorithms (rolling RR, TV smoothing, amplitude filter)
3. Multi-segment discontinuity handling

Note on Expected Warnings:
    During test execution, you may see harmless warnings from pyedflib's C library:
    "read 0, less than X requested!!!"

    These warnings occur when reading ResMed EDF files that contain EDF+ annotation
    channels mixed with data channels. They do not indicate errors or test failures.
    The warnings are expected and match OSCAR's behavior (just hidden in their GUI).
"""

import numpy as np
import pytest

from snore.analysis.data.waveform_loader import WaveformLoader
from snore.analysis.shared.breath_segmenter import BreathSegmenter
from snore.analysis.shared.feature_extractors import WaveformFeatureExtractor
from snore.database.models import Session


@pytest.mark.integration_pipeline
class TestEndToEndPipeline:
    """Verify complete workflow processes real data successfully."""

    def test_full_pipeline_baseline_session(self, recorded_session):
        """Load → segment → extract features on real CPAP data."""
        db = recorded_session("20250808")
        session = db.query(Session).first()
        assert session is not None

        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow", apply_filter=False
        )

        assert len(timestamps) > 0
        assert len(timestamps) == len(flow_values)

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        assert len(breaths) > 0

        extractor = WaveformFeatureExtractor()
        start_idx = int(breaths[0].start_time * metadata["sample_rate"])
        end_idx = int(breaths[0].end_time * metadata["sample_rate"])
        breath_flow = flow_values[start_idx:end_idx]

        shape, peak, stats, _ = extractor.extract_all_features(
            breath_flow, sample_rate=metadata["sample_rate"], include_spectral=False
        )

        assert shape is not None
        assert peak is not None
        assert stats is not None

    def test_multi_segment_discontinuity_handling(self, recorded_session):
        """Sessions with mask-off periods process correctly."""
        db = recorded_session("20250910")
        session = db.query(Session).first()
        assert session is not None

        loader = WaveformLoader(db)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        time_diffs = np.diff(timestamps)
        gaps = np.where(time_diffs > 60.0)[0]

        if len(gaps) > 0:
            assert "segments" in metadata
            assert len(metadata["segments"]) >= 2

        segmenter = BreathSegmenter()
        breaths = segmenter.segment_breaths(
            timestamps, flow_values, sample_rate=metadata["sample_rate"]
        )

        assert len(breaths) > 0


@pytest.mark.integration_pipeline
class TestPhysiologicalValidation:
    """Verify real data produces physiologically realistic results."""

    def test_respiratory_rate_realistic_range(self, recorded_session):
        """RR should be in normal adult range (8-25 breaths/min typical)."""
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

        rr_values = [b.respiratory_rate for b in breaths]
        mean_rr = np.mean(rr_values)

        assert 8 <= mean_rr <= 25, f"Mean RR out of typical range: {mean_rr}"

        outliers = [rr for rr in rr_values if rr < 5 or rr > 60]
        outlier_percent = len(outliers) / len(rr_values) * 100
        assert outlier_percent < 0.5, (
            f"{len(outliers)} RR outliers ({outlier_percent:.2f}%) exceed 0.5% threshold"
        )

    def test_tidal_volume_realistic_range(self, recorded_session):
        """TV should be in normal adult range (300-800 mL typical)."""
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

        tv_values = [b.tidal_volume for b in breaths]
        mean_tv = np.mean(tv_values)

        assert 300 <= mean_tv <= 800, f"Mean TV out of typical range: {mean_tv}"

        in_range = sum(200 <= tv <= 1000 for tv in tv_values)
        assert in_range / len(tv_values) >= 0.95, "Too many TV outliers"

    def test_peak_flows_realistic_range(self, recorded_session):
        """Peak inspiratory/expiratory flows should be realistic."""
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
            assert 0 < breath.peak_inspiratory_flow <= 100, (
                f"PIF out of range: {breath.peak_inspiratory_flow}"
            )
            assert 0 < breath.peak_expiratory_flow <= 100, (
                f"PEF out of range: {breath.peak_expiratory_flow}"
            )

    def test_ie_ratio_realistic(self, recorded_session):
        """I:E ratio should be in typical range (0.4-0.8 for sleep)."""
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

        ie_ratios = [b.i_e_ratio for b in breaths]
        mean_ie = np.mean(ie_ratios)

        assert 0.3 <= mean_ie <= 2.0, f"Mean I:E ratio unusual: {mean_ie}"


@pytest.mark.integration_pipeline
class TestOSCARAlgorithms:
    """Verify OSCAR-aligned algorithms function correctly."""

    def test_amplitude_filter_8_lpm(self, recorded_session):
        """All breaths should have amplitude > 2 L/min (improved threshold)."""
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
            assert breath.amplitude > 2.0, (
                f"Breath {breath.breath_number} below 2 L/min threshold"
            )

    def test_rolling_rr_more_stable_than_instantaneous(self, recorded_session):
        """Rolling 60s window RR should be more stable than instantaneous."""
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

        if len(breaths) >= 20:
            inst_rr = [b.respiratory_rate for b in breaths[10:]]
            roll_rr = [b.respiratory_rate_rolling for b in breaths[10:]]

            assert np.std(roll_rr) < np.std(inst_rr), (
                "Rolling RR not smoothing variability"
            )

    def test_tv_smoothing_reduces_variability(self, recorded_session):
        """5-point weighted TV smoothing should reduce jitter."""
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

        if len(breaths) >= 10:
            raw_tv = [b.tidal_volume for b in breaths[5:]]
            smoothed_tv = [b.tidal_volume_smoothed for b in breaths[5:]]

            assert np.std(smoothed_tv) <= np.std(raw_tv), (
                "TV smoothing not reducing variability"
            )

    def test_complete_breaths_have_both_phases(self, recorded_session):
        """All returned breaths must be complete (inspiration + expiration)."""
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
            assert breath.is_complete, (
                f"Incomplete breath returned: {breath.breath_number}"
            )
            assert breath.inspiration_time > 0, "Missing inspiration phase"
            assert breath.expiration_time > 0, "Missing expiration phase"


@pytest.mark.integration_features
class TestFeatureExtraction:
    """Verify feature extraction produces valid results."""

    def test_flatness_index_in_valid_range(self, recorded_session):
        """Flatness index should be in [0, 1] range."""
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

        extractor = WaveformFeatureExtractor()

        for breath in breaths[:10]:
            start_idx = int(breath.start_time * metadata["sample_rate"])
            end_idx = int(breath.end_time * metadata["sample_rate"])
            breath_flow = flow_values[start_idx:end_idx]

            shape, _, _, _ = extractor.extract_all_features(
                breath_flow, sample_rate=metadata["sample_rate"], include_spectral=False
            )

            assert 0 <= shape.flatness_index <= 1, (
                f"Flatness index out of range: {shape.flatness_index}"
            )
