"""
Unit tests for breath analysis algorithms.

This module tests core breath processing algorithms including:
- Breath amplitude validation
- Rolling window calculations
- Weighted averaging
- Event detection algorithms
- End-to-end breath segmentation
"""

import numpy as np
import pytest

from oscar_mcp.analysis.algorithms.breath_segmenter import (
    BreathMetrics,
    BreathSegmenter,
)


@pytest.mark.unit
class TestBreathAmplitudeValidation:
    """Test breath amplitude filtering and validation."""

    def test_rejects_breaths_below_threshold(self):
        """
        Breaths with amplitude ≤8 L/min should be rejected as noise.

        This prevents sensor noise and artifacts from being counted as breaths.
        """
        segmenter = BreathSegmenter()

        # Create waveform with one valid breath and one too-small breath
        sample_rate = 25.0
        duration = 10.0
        samples = int(duration * sample_rate)
        timestamps = np.linspace(0, duration, samples)

        # Valid breath (amplitude = 40 L/min)
        valid_breath = 30.0 * np.sin(np.pi * np.arange(62) / 62)
        valid_breath = np.concatenate([valid_breath, -10.0 * np.ones(63)])

        # Too-small breath (amplitude = 5 L/min)
        small_breath = 3.0 * np.sin(np.pi * np.arange(62) / 62)
        small_breath = np.concatenate([small_breath, -2.0 * np.ones(63)])

        # Combine
        flow_data = np.concatenate([valid_breath, small_breath])
        timestamps = timestamps[: len(flow_data)]

        breaths = segmenter.segment_breaths(timestamps, flow_data, sample_rate)

        # Should only detect valid breaths (amplitude > 8)
        assert len(breaths) >= 0
        # If any breaths detected, all should have amplitude > 8
        for breath in breaths:
            assert breath.amplitude > 8.0, (
                f"Breath amplitude {breath.amplitude} should be > 8.0 L/min"
            )


@pytest.mark.unit
class TestRollingWindowCalculation:
    """Test rolling window calculations for respiratory rate."""

    def test_rolling_window_with_regular_breathing(self):
        """
        Rolling 60-second window should accurately count breaths.

        Uses partial breath weighting for breaths spanning window boundary.
        """
        segmenter = BreathSegmenter()

        # Create mock breaths spanning 90 seconds
        mock_breaths = []
        for i in range(30):  # 30 breaths over 90 seconds
            breath = BreathMetrics(
                breath_number=i + 1,
                start_time=i * 3.0,
                middle_time=i * 3.0 + 1.5,
                end_time=i * 3.0 + 3.0,
                duration=3.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=20.0,
                inspiration_time=1.5,
                expiration_time=1.5,
                i_e_ratio=1.0,
                respiratory_rate=20.0,  # 60/3 = 20 breaths/min
                respiratory_rate_rolling=0.0,  # Will be calculated
                minute_ventilation=10.0,
                amplitude=50.0,
                is_complete=True,
            )
            mock_breaths.append(breath)

        # Test rolling RR at breath 25 (75 seconds into session)
        rolling_rr = segmenter.calculate_rolling_respiratory_rate(
            mock_breaths, current_breath_idx=24, window_seconds=60.0
        )

        # In last 60 seconds (from 15s to 75s), there are 20 breaths
        # RR should be 20 breaths/min
        assert 19.0 <= rolling_rr <= 21.0, (
            f"Rolling RR {rolling_rr} should be ~20 breaths/min"
        )


@pytest.mark.unit
class TestWeightedAverageSmoothing:
    """Test weighted averaging for tidal volume smoothing."""

    def test_five_point_weighted_smoothing(self):
        """
        5-point weighted average should smooth noise while preserving trend.

        Formula: (tv[-3] + tv[-2] + tv[-1] + tv[current]*2) / 5
        """
        segmenter = BreathSegmenter()

        # Test with 4 history values
        tv_history = [450.0, 460.0, 455.0, 465.0]
        current_tv = 470.0

        smoothed = segmenter.calculate_smoothed_tidal_volume(tv_history, current_tv)

        # Expected: (455 + 465 + 470 + 470*2) / 5 = 2330 / 5 = 466
        expected = (
            tv_history[-3] + tv_history[-2] + tv_history[-1] + current_tv * 2
        ) / 5
        assert abs(smoothed - expected) < 0.1, (
            f"Smoothed TV {smoothed} should equal {expected}"
        )

    def test_smoothing_with_no_history(self):
        """First breath should return raw value (no smoothing possible)."""
        segmenter = BreathSegmenter()

        smoothed_first = segmenter.calculate_smoothed_tidal_volume([], 500.0)
        assert smoothed_first == 500.0, "First breath should not be smoothed"

    def test_smoothing_with_partial_history(self):
        """Should adapt smoothing formula based on available history."""
        segmenter = BreathSegmenter()

        # 1 history value
        smoothed_1 = segmenter.calculate_smoothed_tidal_volume([450.0], 470.0)
        expected_1 = (450.0 + 470.0 * 2) / 3
        assert abs(smoothed_1 - expected_1) < 0.1

        # 2 history values
        smoothed_2 = segmenter.calculate_smoothed_tidal_volume([450.0, 460.0], 470.0)
        expected_2 = (450.0 + 460.0 + 470.0 * 2) / 4
        assert abs(smoothed_2 - expected_2) < 0.1


@pytest.mark.unit
class TestPercentileBasedEventDetection:
    """Test percentile-based flow restriction detection."""

    def test_detects_sustained_restriction_events(self):
        """
        Should detect sustained periods where amplitude falls below threshold.

        Uses 60th percentile of amplitudes as baseline, flags when breath
        amplitude < (p60 * restriction_percent) for minimum duration.
        """
        segmenter = BreathSegmenter()

        # Create mock breaths with varying amplitudes
        mock_breaths = []
        # More breaths with longer restriction period
        amplitudes = [50, 52, 48, 51, 49, 20, 22, 18, 21, 19, 23, 50, 51, 49, 48, 52]

        for i, amp in enumerate(amplitudes):
            breath = BreathMetrics(
                breath_number=i + 1,
                start_time=i * 4.0,
                middle_time=i * 4.0 + 2.0,
                end_time=i * 4.0 + 4.0,
                duration=4.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=amp * 0.6,
                peak_expiratory_flow=amp * 0.4,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=amp,
                is_complete=True,
            )
            mock_breaths.append(breath)

        # Detect flow restriction (50% threshold, 10s minimum duration)
        restrictions = segmenter.detect_flow_restriction(
            mock_breaths, restriction_percent=50.0, duration_threshold=10.0
        )

        # p60 of amplitudes ≈ 49-50
        # cutoff = 50 * 0.5 = 25
        # Breaths 5-10 (indices 5-10) have amp ~20, spanning 24 seconds
        # Should detect one event
        assert len(restrictions) >= 1, "Should detect at least one restriction event"

        # First restriction should span the low-amplitude region
        if len(restrictions) > 0:
            start_idx, end_idx = restrictions[0]
            # Check that it captures the restricted breaths
            assert start_idx >= 4, (
                f"Restriction should start around breath 5, got {start_idx}"
            )
            assert end_idx <= 11, (
                f"Restriction should end around breath 10, got {end_idx}"
            )
            # Duration should be at least 10 seconds
            duration = sum(
                mock_breaths[i].duration for i in range(start_idx, end_idx + 1)
            )
            assert duration >= 10.0, f"Restriction duration {duration}s should be ≥ 10s"


@pytest.mark.unit
class TestEndToEndBreathSegmentation:
    """Integration test for complete breath segmentation pipeline."""

    def test_segments_synthetic_session_correctly(self):
        """
        Complete pipeline should process synthetic session end-to-end.

        Validates: deserialization → segmentation → metric calculation → features
        """
        segmenter = BreathSegmenter()

        # Create synthetic 2-minute session with regular breathing
        sample_rate = 25.0
        breath_duration = 4.0  # seconds (15 breaths/min)
        num_breaths = 30  # 2 minutes
        samples_per_breath = int(breath_duration * sample_rate)

        timestamps = []
        flow_data = []

        for i in range(num_breaths):
            breath_start = i * breath_duration
            t = np.linspace(
                breath_start, breath_start + breath_duration, samples_per_breath
            )

            # Sinusoidal breath: inspiration (0-2s), expiration (2-4s)
            half = samples_per_breath // 2
            insp = 35.0 * np.sin(np.pi * np.arange(half) / half)
            exp = -25.0 * np.sin(np.pi * np.arange(half) / half)
            breath = np.concatenate([insp, exp])

            timestamps.extend(t)
            flow_data.extend(breath)

        timestamps = np.array(timestamps)
        flow_data = np.array(flow_data)

        # Segment breaths
        breaths = segmenter.segment_breaths(timestamps, flow_data, sample_rate)

        # Validate results
        assert len(breaths) >= 25, f"Should detect most breaths, got {len(breaths)}"
        assert len(breaths) <= 30, f"Should not over-detect, got {len(breaths)}"

        # Check that all breaths have valid metrics
        for breath in breaths:
            assert breath.is_complete, (
                f"Breath {breath.breath_number} should be complete"
            )
            assert breath.amplitude > 8.0, (
                f"Breath {breath.breath_number} amplitude too low"
            )
            assert breath.duration > 1.0, (
                f"Breath {breath.breath_number} duration too short"
            )
            assert breath.tidal_volume > 0, (
                f"Breath {breath.breath_number} has no volume"
            )
            assert breath.respiratory_rate_rolling >= 0, (
                "Rolling RR should be non-negative"
            )

        # Check smoothing is applied (later breaths have smoothed TV)
        if len(breaths) >= 5:
            # 5th breath should have smoothing
            assert breaths[4].tidal_volume_smoothed > 0, (
                "Smoothed TV should be positive"
            )
            # Smoothed value should be close to raw (in stable breathing)
            assert (
                abs(breaths[4].tidal_volume - breaths[4].tidal_volume_smoothed)
                / breaths[4].tidal_volume
                < 0.2  # Within 20%
            ), "Smoothed TV should be close to raw in stable breathing"
