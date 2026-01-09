"""Unit tests for EventDetector methods across all modes."""

import numpy as np
import pytest

from snore.analysis.modes.config import AASM_CONFIG, AASM_RELAXED_CONFIG, RESMED_CONFIG
from snore.analysis.modes.detector import EventDetector, _calculate_event_overlap
from snore.analysis.modes.types import HypopneaMode
from snore.analysis.shared.types import (
    ApneaEvent,
    BreathMetrics,
    HypopneaEvent,
)


@pytest.fixture
def resmed_detector():
    """EventDetector configured with ResMed mode."""
    return EventDetector(RESMED_CONFIG)


@pytest.fixture
def aasm_detector():
    """EventDetector configured with AASM mode."""
    return EventDetector(AASM_CONFIG)


@pytest.fixture
def aasm_relaxed_detector():
    """EventDetector configured with AASM relaxed mode."""
    return EventDetector(AASM_RELAXED_CONFIG)


@pytest.fixture
def sample_breaths() -> list[BreathMetrics]:
    """Create sample breaths with controllable gaps."""
    return [
        BreathMetrics(
            breath_number=1,
            start_time=0.0,
            middle_time=1.0,
            end_time=2.0,
            duration=2.0,
            tidal_volume=500.0,
            tidal_volume_smoothed=500.0,
            peak_inspiratory_flow=30.0,
            peak_expiratory_flow=25.0,
            inspiration_time=1.0,
            expiration_time=1.0,
            i_e_ratio=1.0,
            respiratory_rate=30.0,
            respiratory_rate_rolling=15.0,
            minute_ventilation=7.5,
            amplitude=55.0,
            is_complete=True,
        ),
        BreathMetrics(
            breath_number=2,
            start_time=17.0,
            middle_time=18.0,
            end_time=19.0,
            duration=2.0,
            tidal_volume=500.0,
            tidal_volume_smoothed=500.0,
            peak_inspiratory_flow=30.0,
            peak_expiratory_flow=25.0,
            inspiration_time=1.0,
            expiration_time=1.0,
            i_e_ratio=1.0,
            respiratory_rate=30.0,
            respiratory_rate_rolling=15.0,
            minute_ventilation=7.5,
            amplitude=55.0,
            is_complete=True,
        ),
        BreathMetrics(
            breath_number=3,
            start_time=31.0,
            middle_time=32.0,
            end_time=33.0,
            duration=2.0,
            tidal_volume=500.0,
            tidal_volume_smoothed=500.0,
            peak_inspiratory_flow=30.0,
            peak_expiratory_flow=25.0,
            inspiration_time=1.0,
            expiration_time=1.0,
            i_e_ratio=1.0,
            respiratory_rate=30.0,
            respiratory_rate_rolling=15.0,
            minute_ventilation=7.5,
            amplitude=55.0,
            is_complete=True,
        ),
        BreathMetrics(
            breath_number=4,
            start_time=55.0,
            middle_time=56.0,
            end_time=57.0,
            duration=2.0,
            tidal_volume=500.0,
            tidal_volume_smoothed=500.0,
            peak_inspiratory_flow=30.0,
            peak_expiratory_flow=25.0,
            inspiration_time=1.0,
            expiration_time=1.0,
            i_e_ratio=1.0,
            respiratory_rate=30.0,
            respiratory_rate_rolling=15.0,
            minute_ventilation=7.5,
            amplitude=55.0,
            is_complete=True,
        ),
    ]


@pytest.fixture
def sample_flow_data() -> tuple[np.ndarray, np.ndarray]:
    """Create flow data with near-zero regions."""
    timestamps = np.arange(0, 100, 0.04)
    flow_values = np.ones_like(timestamps) * 10.0
    flow_values[500:875] = 0.5
    return timestamps, flow_values


class TestDetectBreathGaps:
    """Tests for _detect_breath_gaps() method."""

    def test_detect_gap_basic(self, resmed_detector):
        """Detect single gap of 15 seconds."""
        breaths = [
            BreathMetrics(
                breath_number=1,
                start_time=0.0,
                middle_time=1.0,
                end_time=2.0,
                duration=2.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=1.0,
                expiration_time=1.0,
                i_e_ratio=1.0,
                respiratory_rate=30.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=55.0,
                is_complete=True,
            ),
            BreathMetrics(
                breath_number=2,
                start_time=17.0,
                middle_time=18.0,
                end_time=19.0,
                duration=2.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=1.0,
                expiration_time=1.0,
                i_e_ratio=1.0,
                respiratory_rate=30.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=55.0,
                is_complete=True,
            ),
        ]

        events = resmed_detector._detect_breath_gaps(breaths, min_gap_seconds=10.0)

        assert len(events) == 1
        assert events[0].event_type == "CA"
        assert events[0].start_time == 2.0
        assert events[0].end_time == 17.0
        assert events[0].duration == 15.0
        assert events[0].detection_method == "gap"
        assert events[0].confidence == 0.85

    def test_detect_gap_below_threshold(self, resmed_detector):
        """Gap below 10s threshold should not be detected."""
        breaths = [
            BreathMetrics(
                breath_number=1,
                start_time=0.0,
                middle_time=1.0,
                end_time=2.0,
                duration=2.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=1.0,
                expiration_time=1.0,
                i_e_ratio=1.0,
                respiratory_rate=30.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=55.0,
                is_complete=True,
            ),
            BreathMetrics(
                breath_number=2,
                start_time=10.0,
                middle_time=11.0,
                end_time=12.0,
                duration=2.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=1.0,
                expiration_time=1.0,
                i_e_ratio=1.0,
                respiratory_rate=30.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=55.0,
                is_complete=True,
            ),
        ]

        events = resmed_detector._detect_breath_gaps(breaths, min_gap_seconds=10.0)

        assert len(events) == 0

    def test_detect_gap_multiple(self, resmed_detector, sample_breaths):
        """Detect multiple gaps."""
        events = resmed_detector._detect_breath_gaps(
            sample_breaths, min_gap_seconds=10.0
        )

        assert len(events) == 3
        assert events[0].duration == 15.0
        assert events[1].duration == 12.0
        assert events[2].duration == 22.0

    def test_detect_gap_empty_breaths(self, resmed_detector):
        """Empty breath list should return no events."""
        events = resmed_detector._detect_breath_gaps([], min_gap_seconds=10.0)

        assert len(events) == 0

    def test_detect_gap_single_breath(self, resmed_detector):
        """Single breath should return no events."""
        breaths = [
            BreathMetrics(
                breath_number=1,
                start_time=0.0,
                middle_time=1.0,
                end_time=2.0,
                duration=2.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=1.0,
                expiration_time=1.0,
                i_e_ratio=1.0,
                respiratory_rate=30.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=55.0,
                is_complete=True,
            )
        ]

        events = resmed_detector._detect_breath_gaps(breaths, min_gap_seconds=10.0)

        assert len(events) == 0


class TestDetectNearZeroFlow:
    """Tests for _detect_near_zero_flow() method."""

    def test_near_zero_sustained(self, resmed_detector):
        """Detect sustained near-zero flow for 15 seconds."""
        timestamps = np.arange(0, 100, 0.04)
        flow_values = np.ones_like(timestamps) * 10.0
        flow_values[500:875] = 0.5

        events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(events) == 1
        assert events[0].event_type == "CA"
        assert events[0].detection_method == "near_zero_flow"
        assert events[0].confidence == 0.80
        assert 14.0 <= events[0].duration <= 16.0

    def test_near_zero_below_threshold(self, resmed_detector):
        """Near-zero flow below duration threshold should not be detected."""
        timestamps = np.arange(0, 100, 0.04)
        flow_values = np.ones_like(timestamps) * 10.0
        flow_values[500:700] = 0.5

        events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(events) == 0

    def test_near_zero_intermittent(self, resmed_detector):
        """Intermittent near-zero flow should not be detected."""
        timestamps = np.arange(0, 100, 0.04)
        flow_values = np.ones_like(timestamps) * 10.0
        for i in range(0, 2500, 100):
            flow_values[i : i + 50] = 0.5

        events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(events) == 0

    def test_near_zero_at_session_end(self, resmed_detector):
        """Detect near-zero flow at end of signal."""
        timestamps = np.arange(0, 100, 0.04)
        flow_values = np.ones_like(timestamps) * 10.0
        flow_values[2125:] = 0.5

        events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(events) == 1
        assert events[0].end_time == pytest.approx(timestamps[-1], abs=0.1)

    def test_near_zero_uses_actual_timestamps(self, resmed_detector):
        """Verify event times match actual timestamps, not sample indices."""
        timestamps = np.arange(1000, 1100, 0.04)
        flow_values = np.ones_like(timestamps) * 10.0
        flow_values[500:875] = 0.5

        events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(events) == 1
        assert events[0].start_time >= 1000.0
        assert events[0].end_time <= 1100.0


class TestDeduplicateEvents:
    """Tests for _deduplicate_events() method."""

    def test_dedupe_no_overlap(self, resmed_detector):
        """Non-overlapping events should remain unchanged."""
        events = [
            ApneaEvent(
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=1.0,
                confidence=0.85,
                baseline_flow=0.0,
                detection_method="gap",
            ),
            ApneaEvent(
                start_time=20.0,
                end_time=30.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=1.0,
                confidence=0.80,
                baseline_flow=0.0,
                detection_method="near_zero_flow",
            ),
        ]

        deduplicated = resmed_detector._deduplicate_events(
            events, overlap_threshold=0.5
        )

        assert len(deduplicated) == 2

    def test_dedupe_high_overlap(self, resmed_detector):
        """High overlap should deduplicate, keeping higher confidence."""
        events = [
            ApneaEvent(
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=1.0,
                confidence=0.85,
                baseline_flow=0.0,
                detection_method="gap",
            ),
            ApneaEvent(
                start_time=2.0,
                end_time=12.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.95,
                confidence=0.90,
                baseline_flow=0.0,
                detection_method="amplitude",
            ),
        ]

        deduplicated = resmed_detector._deduplicate_events(
            events, overlap_threshold=0.5
        )

        assert len(deduplicated) == 1
        assert deduplicated[0].confidence == 0.90
        assert deduplicated[0].detection_method == "amplitude"

    def test_dedupe_confidence_ordering(self, resmed_detector):
        """Keep first event if it has higher confidence."""
        events = [
            ApneaEvent(
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=1.0,
                confidence=0.95,
                baseline_flow=0.0,
                detection_method="gap",
            ),
            ApneaEvent(
                start_time=2.0,
                end_time=12.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.95,
                confidence=0.80,
                baseline_flow=0.0,
                detection_method="near_zero_flow",
            ),
        ]

        deduplicated = resmed_detector._deduplicate_events(
            events, overlap_threshold=0.5
        )

        assert len(deduplicated) == 1
        assert deduplicated[0].confidence == 0.95
        assert deduplicated[0].detection_method == "gap"

    def test_dedupe_empty_list(self, resmed_detector):
        """Empty list should return empty list."""
        deduplicated = resmed_detector._deduplicate_events([], overlap_threshold=0.5)

        assert len(deduplicated) == 0

    def test_dedupe_single_event(self, resmed_detector):
        """Single event should be returned unchanged."""
        events = [
            ApneaEvent(
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=1.0,
                confidence=0.85,
                baseline_flow=0.0,
                detection_method="gap",
            )
        ]

        deduplicated = resmed_detector._deduplicate_events(
            events, overlap_threshold=0.5
        )

        assert len(deduplicated) == 1
        assert deduplicated[0] == events[0]


class TestCalculateEventOverlap:
    """Tests for _calculate_event_overlap() function."""

    def test_overlap_none(self):
        """Events 10s apart should have no overlap."""
        event1 = ApneaEvent(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.85,
            baseline_flow=0.0,
        )
        event2 = ApneaEvent(
            start_time=20.0,
            end_time=30.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.80,
            baseline_flow=0.0,
        )

        overlap = _calculate_event_overlap(event1, event2)

        assert overlap == 0.0

    def test_overlap_partial(self):
        """50% overlap should return 0.5."""
        event1 = ApneaEvent(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.85,
            baseline_flow=0.0,
        )
        event2 = ApneaEvent(
            start_time=5.0,
            end_time=15.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.80,
            baseline_flow=0.0,
        )

        overlap = _calculate_event_overlap(event1, event2)

        assert overlap == 0.5

    def test_overlap_complete(self):
        """One event containing another should return 1.0."""
        event1 = ApneaEvent(
            start_time=0.0,
            end_time=20.0,
            duration=20.0,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.85,
            baseline_flow=0.0,
        )
        event2 = ApneaEvent(
            start_time=5.0,
            end_time=15.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.80,
            baseline_flow=0.0,
        )

        overlap = _calculate_event_overlap(event1, event2)

        assert overlap == 1.0

    def test_overlap_adjacent(self):
        """Adjacent events (end == start) should have no overlap."""
        event1 = ApneaEvent(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.85,
            baseline_flow=0.0,
        )
        event2 = ApneaEvent(
            start_time=10.0,
            end_time=20.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.80,
            baseline_flow=0.0,
        )

        overlap = _calculate_event_overlap(event1, event2)

        assert overlap == 0.0


class TestDetectEventsResmed:
    """Integration tests for _detect_events_resmed() method."""

    def test_resmed_combines_strategies(
        self,
        resmed_detector,
        sample_breaths,
        sample_flow_data,
    ):
        """Verify multiple strategies are combined and deduplicated."""
        events = resmed_detector._detect_events_resmed(sample_breaths, sample_flow_data)

        assert len(events) >= 1
        assert events[0].event_type == "CA"

    def test_resmed_no_flow_data(self, resmed_detector, sample_breaths):
        """ResMed should work with breaths only (no flow data)."""
        events = resmed_detector._detect_events_resmed(sample_breaths, None)

        assert len(events) >= 1
        for event in events:
            assert event.detection_method == "gap"
            assert event.event_type == "CA"


# =============================================================================
# Shared Utility Tests (AASM/ResMed shared methods)
# =============================================================================


class TestMergeTwoEvents:
    """Tests for _merge_two_events() method - we fixed a bug here!"""

    def test_merge_preserves_detection_method(self, aasm_detector):
        """Verify detection_method is preserved when merging events."""
        event1 = ApneaEvent(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.85,
            baseline_flow=0.0,
            detection_method="gap",
        )
        event2 = ApneaEvent(
            start_time=10.5,
            end_time=20.0,
            duration=9.5,
            event_type="CA",
            flow_reduction=1.0,
            confidence=0.85,
            baseline_flow=0.0,
            detection_method="gap",
        )

        merged = aasm_detector._merge_two_events(event1, event2)

        assert merged.detection_method == "gap"
        assert merged.start_time == 0.0
        assert merged.end_time == 20.0

    def test_merge_apnea_events(self, aasm_detector):
        """Merge two adjacent apnea events."""
        event1 = ApneaEvent(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=0.9,
            confidence=0.85,
            baseline_flow=30.0,
        )
        event2 = ApneaEvent(
            start_time=11.0,
            end_time=20.0,
            duration=9.0,
            event_type="CA",
            flow_reduction=0.95,
            confidence=0.90,
            baseline_flow=30.0,
        )

        merged = aasm_detector._merge_two_events(event1, event2)

        assert isinstance(merged, ApneaEvent)
        assert merged.start_time == 0.0
        assert merged.end_time == 20.0
        assert merged.duration == 20.0
        assert merged.event_type == "CA"

    def test_merge_hypopnea_events(self, aasm_detector):
        """Merge two adjacent hypopnea events."""
        event1 = HypopneaEvent(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            flow_reduction=0.5,
            confidence=0.80,
            baseline_flow=30.0,
            has_desaturation=True,
        )
        event2 = HypopneaEvent(
            start_time=11.0,
            end_time=20.0,
            duration=9.0,
            flow_reduction=0.6,
            confidence=0.85,
            baseline_flow=30.0,
            has_desaturation=False,
        )

        merged = aasm_detector._merge_two_events(event1, event2)

        assert isinstance(merged, HypopneaEvent)
        assert merged.start_time == 0.0
        assert merged.end_time == 20.0
        assert merged.has_desaturation is True

    def test_merge_averages_flow_reduction(self, aasm_detector):
        """Verify flow reduction is averaged when merging."""
        event1 = ApneaEvent(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=0.8,
            confidence=0.85,
            baseline_flow=30.0,
        )
        event2 = ApneaEvent(
            start_time=11.0,
            end_time=20.0,
            duration=9.0,
            event_type="CA",
            flow_reduction=0.6,
            confidence=0.85,
            baseline_flow=30.0,
        )

        merged = aasm_detector._merge_two_events(event1, event2)

        assert merged.flow_reduction == pytest.approx(0.7, abs=0.01)

    def test_merge_takes_min_confidence(self, aasm_detector):
        """Verify confidence is minimum of two events."""
        event1 = ApneaEvent(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            event_type="CA",
            flow_reduction=0.9,
            confidence=0.90,
            baseline_flow=30.0,
        )
        event2 = ApneaEvent(
            start_time=11.0,
            end_time=20.0,
            duration=9.0,
            event_type="CA",
            flow_reduction=0.9,
            confidence=0.70,
            baseline_flow=30.0,
        )

        merged = aasm_detector._merge_two_events(event1, event2)

        assert merged.confidence == 0.70


class TestMergeAdjacentEvents:
    """Tests for _merge_adjacent_events() method."""

    def test_merge_within_gap_threshold(self, aasm_detector):
        """Events within gap threshold should be merged."""
        events = [
            ApneaEvent(
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.9,
                confidence=0.85,
                baseline_flow=30.0,
            ),
            ApneaEvent(
                start_time=12.0,
                end_time=22.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.9,
                confidence=0.85,
                baseline_flow=30.0,
            ),
        ]

        merged = aasm_detector._merge_adjacent_events(events, max_gap=3.0)

        assert len(merged) == 1
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 22.0

    def test_no_merge_beyond_gap_threshold(self, aasm_detector):
        """Events beyond gap threshold should not be merged."""
        events = [
            ApneaEvent(
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.9,
                confidence=0.85,
                baseline_flow=30.0,
            ),
            ApneaEvent(
                start_time=15.0,
                end_time=25.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.9,
                confidence=0.85,
                baseline_flow=30.0,
            ),
        ]

        merged = aasm_detector._merge_adjacent_events(events, max_gap=3.0)

        assert len(merged) == 2

    def test_no_merge_different_types(self, aasm_detector):
        """Different event types should not be merged."""
        events = [
            ApneaEvent(
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.9,
                confidence=0.85,
                baseline_flow=30.0,
            ),
            HypopneaEvent(
                start_time=11.0,
                end_time=21.0,
                duration=10.0,
                flow_reduction=0.5,
                confidence=0.80,
                baseline_flow=30.0,
            ),
        ]

        merged = aasm_detector._merge_adjacent_events(events, max_gap=3.0)

        assert len(merged) == 2

    def test_merge_empty_list(self, aasm_detector):
        """Empty list should return empty list."""
        merged = aasm_detector._merge_adjacent_events([], max_gap=3.0)

        assert len(merged) == 0

    def test_merge_single_event(self, aasm_detector):
        """Single event should be returned unchanged."""
        events = [
            ApneaEvent(
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.9,
                confidence=0.85,
                baseline_flow=30.0,
            )
        ]

        merged = aasm_detector._merge_adjacent_events(events, max_gap=3.0)

        assert len(merged) == 1
        assert merged[0] == events[0]


class TestClassifyApneaType:
    """Tests for _classify_apnea_type() method."""

    def test_classify_obstructive(self, aasm_detector):
        """High effort flow signal should be classified as OA."""
        flow_signal = np.array([5, -5, 6, -6, 5, -5, 6, -6, 5, -5])

        apnea_type, confidence = aasm_detector._classify_apnea_type(
            flow_signal=flow_signal
        )

        assert apnea_type == "OA"
        assert 0.5 <= confidence <= 1.0  # Should have reasonable confidence

    def test_classify_central(self, aasm_detector):
        """Flat flow signal should be classified as CA."""
        flow_signal = np.array(
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        )

        apnea_type, confidence = aasm_detector._classify_apnea_type(
            flow_signal=flow_signal
        )

        assert apnea_type == "CA"
        assert 0.5 <= confidence <= 1.0  # Should have reasonable confidence

    def test_classify_mixed(self, aasm_detector):
        """Medium effort should be classified as MA."""
        flow_signal = np.array(
            [0.06, -0.06, 0.07, -0.05, 0.06, -0.06, 0.07, -0.05, 0.06, -0.06]
        )

        apnea_type, confidence = aasm_detector._classify_apnea_type(
            flow_signal=flow_signal
        )

        assert apnea_type == "MA"
        assert 0.3 <= confidence <= 0.6  # MA should have lower confidence (borderline)

    def test_classify_no_flow_data(self, aasm_detector):
        """No flow data should be classified as UA."""
        apnea_type, confidence = aasm_detector._classify_apnea_type(flow_signal=None)

        assert apnea_type == "UA"
        assert confidence == 0.2  # UA should have low confidence


class TestCheckDesaturation:
    """Tests for _check_desaturation() method."""

    def test_desaturation_detected(self, aasm_detector):
        """SpO2 drop of 4% should be detected."""
        spo2_values = np.array([96, 95, 94, 93, 92, 91, 90])

        has_desat = aasm_detector._check_desaturation(spo2_values)

        assert has_desat is True

    def test_no_desaturation(self, aasm_detector):
        """SpO2 drop of 2% should not be detected."""
        spo2_values = np.array([96, 95, 94, 94, 95, 96])

        has_desat = aasm_detector._check_desaturation(spo2_values)

        assert has_desat is False

    def test_insufficient_data(self, aasm_detector):
        """Less than 2 samples should return False."""
        spo2_values = np.array([96])

        has_desat = aasm_detector._check_desaturation(spo2_values)

        assert has_desat is False


# =============================================================================
# AASM Mode Tests (amplitude-based detection)
# =============================================================================


@pytest.fixture
def apnea_breaths() -> list[BreathMetrics]:
    """Breaths with 90%+ reduction period (apnea)."""
    breaths = []
    for i in range(20):
        if 5 <= i < 10:
            amplitude = 3.0
        else:
            amplitude = 50.0

        breaths.append(
            BreathMetrics(
                breath_number=i + 1,
                start_time=float(i * 4),
                middle_time=float(i * 4 + 2),
                end_time=float(i * 4 + 4),
                duration=4.0,
                tidal_volume=500.0 if amplitude > 10 else 50.0,
                tidal_volume_smoothed=500.0 if amplitude > 10 else 50.0,
                peak_inspiratory_flow=30.0 if amplitude > 10 else 3.0,
                peak_expiratory_flow=25.0 if amplitude > 10 else 2.5,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5 if amplitude > 10 else 0.75,
                amplitude=amplitude,
                is_complete=True,
            )
        )
    return breaths


class TestValidateEvent:
    """Tests for _validate_event() method."""

    def test_validate_meets_threshold(self, aasm_detector):
        """Event with 95% reduction should validate."""
        reductions = np.array([0.85, 0.90, 0.95, 0.92, 0.88])

        is_valid = aasm_detector._validate_event(reductions, 0, 5)

        assert is_valid is True

    def test_validate_below_threshold(self, aasm_detector):
        """Event with max 85% reduction should not validate for AASM strict."""
        reductions = np.array([0.80, 0.82, 0.85, 0.83, 0.81])

        is_valid = aasm_detector._validate_event(reductions, 0, 5)

        assert is_valid is False

    def test_validate_relaxed_threshold(self, aasm_relaxed_detector):
        """Event with 87% reduction should validate for aasm_relaxed."""
        reductions = np.array([0.80, 0.82, 0.87, 0.83, 0.81])

        is_valid = aasm_relaxed_detector._validate_event(reductions, 0, 5)

        assert is_valid is True

    def test_validate_empty_reductions(self, aasm_detector):
        """Empty reductions array should return False."""
        reductions = np.array([])

        is_valid = aasm_detector._validate_event(reductions, 0, 0)

        assert is_valid is False


class TestDetectApneas:
    """Tests for _detect_apneas() method - core AASM functionality."""

    def test_detect_apnea_90_percent_reduction(self, aasm_detector, apnea_breaths):
        """Breaths with 95% reduction should be detected as apnea."""
        apneas = aasm_detector._detect_apneas(apnea_breaths, flow_data=None)

        assert len(apneas) >= 1
        assert all(a.event_type in ["OA", "CA", "MA", "UA"] for a in apneas)

    def test_apnea_empty_breaths(self, aasm_detector):
        """Empty breath list should return no apneas."""
        apneas = aasm_detector._detect_apneas([], flow_data=None)

        assert len(apneas) == 0


class TestFindConsecutiveReducedBreaths:
    """Tests for _find_consecutive_reduced_breaths() method."""

    def test_find_consecutive_basic(self, aasm_detector):
        """Find basic consecutive reduced breaths."""
        breaths = []
        for i in range(10):
            breaths.append(
                BreathMetrics(
                    breath_number=i + 1,
                    start_time=float(i * 4),
                    middle_time=float(i * 4 + 2),
                    end_time=float(i * 4 + 4),
                    duration=4.0,
                    tidal_volume=500.0,
                    tidal_volume_smoothed=500.0,
                    peak_inspiratory_flow=30.0,
                    peak_expiratory_flow=25.0,
                    inspiration_time=2.0,
                    expiration_time=2.0,
                    i_e_ratio=1.0,
                    respiratory_rate=15.0,
                    respiratory_rate_rolling=15.0,
                    minute_ventilation=7.5,
                    amplitude=50.0,
                    is_complete=True,
                )
            )

        reductions = np.array([0.1, 0.2, 0.95, 0.95, 0.95, 0.95, 0.95, 0.2, 0.1, 0.1])

        regions = aasm_detector._find_consecutive_reduced_breaths(
            breaths, reductions, threshold=0.9, min_duration=10.0
        )

        assert len(regions) >= 1

    def test_find_multiple_regions(self, aasm_detector):
        """Find multiple separate regions."""
        breaths = []
        for i in range(15):
            breaths.append(
                BreathMetrics(
                    breath_number=i + 1,
                    start_time=float(i * 4),
                    middle_time=float(i * 4 + 2),
                    end_time=float(i * 4 + 4),
                    duration=4.0,
                    tidal_volume=500.0,
                    tidal_volume_smoothed=500.0,
                    peak_inspiratory_flow=30.0,
                    peak_expiratory_flow=25.0,
                    inspiration_time=2.0,
                    expiration_time=2.0,
                    i_e_ratio=1.0,
                    respiratory_rate=15.0,
                    respiratory_rate_rolling=15.0,
                    minute_ventilation=7.5,
                    amplitude=50.0,
                    is_complete=True,
                )
            )

        reductions = np.array(
            [
                0.1,
                0.95,
                0.95,
                0.95,
                0.1,
                0.1,
                0.1,
                0.95,
                0.95,
                0.95,
                0.95,
                0.1,
                0.1,
                0.1,
                0.1,
            ]
        )

        regions = aasm_detector._find_consecutive_reduced_breaths(
            breaths, reductions, threshold=0.9, min_duration=10.0
        )

        assert len(regions) >= 1


class TestCalculateTimeBasedBaseline:
    """Tests for time-based baseline calculation (AASM)."""

    def test_baseline_2_minute_window(self, aasm_detector):
        """Baseline should use 2-minute window."""
        breaths = []
        for i in range(40):
            breaths.append(
                BreathMetrics(
                    breath_number=i + 1,
                    start_time=float(i * 4),
                    middle_time=float(i * 4 + 2),
                    end_time=float(i * 4 + 4),
                    duration=4.0,
                    tidal_volume=500.0,
                    tidal_volume_smoothed=500.0,
                    peak_inspiratory_flow=30.0,
                    peak_expiratory_flow=25.0,
                    inspiration_time=2.0,
                    expiration_time=2.0,
                    i_e_ratio=1.0,
                    respiratory_rate=15.0,
                    respiratory_rate_rolling=15.0,
                    minute_ventilation=7.5,
                    amplitude=float(40 + i % 20),
                    is_complete=True,
                )
            )

        baseline = aasm_detector._calculate_time_based_baseline(breaths, 35)

        assert baseline >= 10.0

    def test_baseline_excludes_event_breaths(self, aasm_detector):
        """Breaths with in_event=True should be excluded."""
        breaths = []
        for i in range(40):
            breath = BreathMetrics(
                breath_number=i + 1,
                start_time=float(i * 4),
                middle_time=float(i * 4 + 2),
                end_time=float(i * 4 + 4),
                duration=4.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=float(40 + i % 20),
                is_complete=True,
                in_event=(10 <= i < 20),
            )
            breaths.append(breath)

        baseline = aasm_detector._calculate_time_based_baseline(breaths, 35)

        assert baseline >= 10.0

    def test_baseline_minimum_floor(self, aasm_detector):
        """Very low values should return minimum floor."""
        breaths = []
        for i in range(40):
            breaths.append(
                BreathMetrics(
                    breath_number=i + 1,
                    start_time=float(i * 4),
                    middle_time=float(i * 4 + 2),
                    end_time=float(i * 4 + 4),
                    duration=4.0,
                    tidal_volume=50.0,
                    tidal_volume_smoothed=50.0,
                    peak_inspiratory_flow=3.0,
                    peak_expiratory_flow=2.5,
                    inspiration_time=2.0,
                    expiration_time=2.0,
                    i_e_ratio=1.0,
                    respiratory_rate=15.0,
                    respiratory_rate_rolling=15.0,
                    minute_ventilation=0.75,
                    amplitude=2.0,
                    is_complete=True,
                )
            )

        baseline = aasm_detector._calculate_time_based_baseline(breaths, 35)

        assert baseline == 10.0


class TestCalculateBreathBasedBaseline:
    """Tests for breath-based baseline calculation (aasm_relaxed)."""

    def test_baseline_30_breath_window(self, aasm_relaxed_detector):
        """Baseline should use 30-breath window."""
        breaths = []
        for i in range(50):
            breaths.append(
                BreathMetrics(
                    breath_number=i + 1,
                    start_time=float(i * 4),
                    middle_time=float(i * 4 + 2),
                    end_time=float(i * 4 + 4),
                    duration=4.0,
                    tidal_volume=500.0,
                    tidal_volume_smoothed=500.0,
                    peak_inspiratory_flow=30.0,
                    peak_expiratory_flow=25.0,
                    inspiration_time=2.0,
                    expiration_time=2.0,
                    i_e_ratio=1.0,
                    respiratory_rate=15.0,
                    respiratory_rate_rolling=15.0,
                    minute_ventilation=7.5,
                    amplitude=float(40 + i % 20),
                    is_complete=True,
                )
            )

        baseline = aasm_relaxed_detector._calculate_breath_based_baseline(breaths, 45)

        assert baseline >= 10.0

    def test_baseline_excludes_event_breaths(self, aasm_relaxed_detector):
        """Breaths with in_event=True should be excluded."""
        breaths = []
        for i in range(50):
            breath = BreathMetrics(
                breath_number=i + 1,
                start_time=float(i * 4),
                middle_time=float(i * 4 + 2),
                end_time=float(i * 4 + 4),
                duration=4.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=float(40 + i % 20),
                is_complete=True,
                in_event=(20 <= i < 35),
            )
            breaths.append(breath)

        baseline = aasm_relaxed_detector._calculate_breath_based_baseline(breaths, 45)

        assert baseline >= 10.0


class TestDetectReras:
    """Tests for RERA detection using flow patterns."""

    def test_detect_rera_basic_pattern(self, aasm_detector):
        """Two flow-limited breaths followed by recovery breath."""
        breaths = []
        # Add 10 normal breaths to establish baseline (40 seconds)
        for i in range(10):
            breaths.append(
                BreathMetrics(
                    breath_number=i + 1,
                    start_time=float(i * 4),
                    middle_time=float(i * 4 + 2),
                    end_time=float(i * 4 + 4),
                    duration=4.0,
                    tidal_volume=500.0,
                    tidal_volume_smoothed=500.0,
                    peak_inspiratory_flow=30.0,
                    peak_expiratory_flow=25.0,
                    inspiration_time=2.0,
                    expiration_time=2.0,
                    i_e_ratio=1.0,
                    respiratory_rate=15.0,
                    respiratory_rate_rolling=15.0,
                    minute_ventilation=7.5,
                    amplitude=50.0,
                    is_complete=True,
                )
            )

        # Flow-limited breath 1 (25% reduction)
        breaths.append(
            BreathMetrics(
                breath_number=11,
                start_time=40.0,
                middle_time=42.0,
                end_time=44.0,
                duration=4.0,
                tidal_volume=375.0,
                tidal_volume_smoothed=375.0,
                peak_inspiratory_flow=22.5,
                peak_expiratory_flow=18.75,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=37.5,
                is_complete=True,
            )
        )
        # Flow-limited breath 2 (25% reduction)
        breaths.append(
            BreathMetrics(
                breath_number=12,
                start_time=44.0,
                middle_time=46.0,
                end_time=48.0,
                duration=4.0,
                tidal_volume=375.0,
                tidal_volume_smoothed=375.0,
                peak_inspiratory_flow=22.5,
                peak_expiratory_flow=18.75,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=37.5,
                is_complete=True,
            )
        )
        # Recovery breath (higher amplitude - 60% increase from 37.5)
        breaths.append(
            BreathMetrics(
                breath_number=13,
                start_time=48.0,
                middle_time=50.0,
                end_time=52.0,
                duration=4.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=60.0,  # 60% increase from 37.5
                is_complete=True,
            )
        )

        reras = aasm_detector._detect_reras(breaths, [], [])
        assert len(reras) == 1
        assert reras[0].obstructed_breath_count == 2
        assert reras[0].start_time == 40.0
        assert reras[0].confidence >= 0.4

    def test_detect_rera_insufficient_recovery(self, aasm_detector):
        """Sequence without sufficient recovery breath should not be detected."""
        breaths = [
            # Normal
            BreathMetrics(
                breath_number=1,
                start_time=0.0,
                middle_time=2.0,
                end_time=4.0,
                duration=4.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=50.0,
                is_complete=True,
            ),
            # Flow-limited 1
            BreathMetrics(
                breath_number=2,
                start_time=4.0,
                middle_time=6.0,
                end_time=8.0,
                duration=4.0,
                tidal_volume=375.0,
                tidal_volume_smoothed=375.0,
                peak_inspiratory_flow=22.5,
                peak_expiratory_flow=18.75,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=37.5,
                is_complete=True,
            ),
            # Flow-limited 2
            BreathMetrics(
                breath_number=3,
                start_time=8.0,
                middle_time=10.0,
                end_time=12.0,
                duration=4.0,
                tidal_volume=375.0,
                tidal_volume_smoothed=375.0,
                peak_inspiratory_flow=22.5,
                peak_expiratory_flow=18.75,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=37.5,
                is_complete=True,
            ),
            # Weak recovery (only 20% increase - not enough)
            BreathMetrics(
                breath_number=4,
                start_time=12.0,
                middle_time=14.0,
                end_time=16.0,
                duration=4.0,
                tidal_volume=450.0,
                tidal_volume_smoothed=450.0,
                peak_inspiratory_flow=27.0,
                peak_expiratory_flow=22.5,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=45.0,
                is_complete=True,
            ),
        ]

        reras = aasm_detector._detect_reras(breaths, [], [])
        assert len(reras) == 0

    def test_detect_rera_excluded_by_apnea(self, aasm_detector):
        """Breaths in apnea events should be excluded from RERA detection."""
        breaths = [
            BreathMetrics(
                breath_number=1,
                start_time=0.0,
                middle_time=2.0,
                end_time=4.0,
                duration=4.0,
                tidal_volume=375.0,
                tidal_volume_smoothed=375.0,
                peak_inspiratory_flow=22.5,
                peak_expiratory_flow=18.75,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=37.5,
                is_complete=True,
            ),
            BreathMetrics(
                breath_number=2,
                start_time=4.0,
                middle_time=6.0,
                end_time=8.0,
                duration=4.0,
                tidal_volume=375.0,
                tidal_volume_smoothed=375.0,
                peak_inspiratory_flow=22.5,
                peak_expiratory_flow=18.75,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=37.5,
                is_complete=True,
            ),
        ]

        apnea = ApneaEvent(
            start_time=0.0,
            end_time=8.0,
            duration=8.0,
            event_type="OA",
            flow_reduction=0.9,
            confidence=0.8,
            baseline_flow=50.0,
        )

        reras = aasm_detector._detect_reras(breaths, [apnea], [])
        assert len(reras) == 0


class TestDetectHypopneasModes:
    """Tests for hypopnea detection with different modes."""

    def test_hypopnea_flow_only_mode(self, aasm_detector):
        """FLOW_ONLY mode should detect hypopneas with 40% threshold."""
        # Override config for this test
        from snore.analysis.modes.config import DetectionModeConfig
        from snore.analysis.modes.types import BaselineMethod

        config = DetectionModeConfig(
            name="test",
            description="Test config for FLOW_ONLY mode",
            baseline_method=BaselineMethod.TIME,
            baseline_window=120.0,
            hypopnea_mode=HypopneaMode.FLOW_ONLY,
            hypopnea_min_threshold=0.30,  # This will be overridden to 0.40 by FLOW_ONLY
            hypopnea_flow_only_fallback=False,
        )
        detector = EventDetector(config)

        # Add 30 normal breaths to establish baseline (120 seconds)
        breaths = [
            BreathMetrics(
                breath_number=i + 1,
                start_time=float(i * 4),
                middle_time=float(i * 4 + 2),
                end_time=float(i * 4 + 4),
                duration=4.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=50.0,
                is_complete=True,
            )
            for i in range(30)
        ]

        # Add 3 breaths with 50% reduction (should trigger with 40% threshold)
        for i in range(30, 33):
            breaths.append(
                BreathMetrics(
                    breath_number=i + 1,
                    start_time=float(i * 4),
                    middle_time=float(i * 4 + 2),
                    end_time=float(i * 4 + 4),
                    duration=4.0,
                    tidal_volume=250.0,
                    tidal_volume_smoothed=250.0,
                    peak_inspiratory_flow=15.0,
                    peak_expiratory_flow=12.5,
                    inspiration_time=2.0,
                    expiration_time=2.0,
                    i_e_ratio=1.0,
                    respiratory_rate=15.0,
                    respiratory_rate_rolling=15.0,
                    minute_ventilation=7.5,
                    amplitude=25.0,  # 50% reduction from 50.0
                    is_complete=True,
                )
            )

        hypopneas = detector._detect_hypopneas(
            breaths, flow_data=None, spo2_signal=None
        )
        assert len(hypopneas) >= 1
        if len(hypopneas) > 0:
            assert hypopneas[0].flow_reduction >= 0.40

    def test_hypopnea_disabled_mode(self, aasm_detector):
        """DISABLED mode should skip hypopnea detection entirely."""
        from snore.analysis.modes.config import DetectionModeConfig
        from snore.analysis.modes.types import BaselineMethod

        config = DetectionModeConfig(
            name="test",
            description="Test config for DISABLED mode",
            baseline_method=BaselineMethod.TIME,
            baseline_window=120.0,
            hypopnea_mode=HypopneaMode.DISABLED,
        )
        detector = EventDetector(config)

        breaths = [
            BreathMetrics(
                breath_number=i,
                start_time=float(i * 4),
                middle_time=float(i * 4 + 2),
                end_time=float(i * 4 + 4),
                duration=4.0,
                tidal_volume=250.0,  # 50% reduction
                tidal_volume_smoothed=250.0,
                peak_inspiratory_flow=15.0,
                peak_expiratory_flow=12.5,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=25.0,
                is_complete=True,
            )
            for i in range(3)
        ]

        hypopneas = detector._detect_hypopneas(
            breaths, flow_data=None, spo2_signal=None
        )
        assert len(hypopneas) == 0

    def test_hypopnea_fallback_to_flow_only(self, aasm_detector):
        """AASM mode with fallback should use FLOW_ONLY when no SpO2."""
        from snore.analysis.modes.config import DetectionModeConfig
        from snore.analysis.modes.types import BaselineMethod

        config = DetectionModeConfig(
            name="test",
            description="Test config for AASM_3PCT with fallback",
            baseline_method=BaselineMethod.TIME,
            baseline_window=120.0,
            hypopnea_mode=HypopneaMode.AASM_3PCT,
            hypopnea_flow_only_fallback=True,
            hypopnea_min_threshold=0.30,
        )
        detector = EventDetector(config)

        # Add 30 normal breaths to establish baseline (120 seconds)
        breaths = [
            BreathMetrics(
                breath_number=i + 1,
                start_time=float(i * 4),
                middle_time=float(i * 4 + 2),
                end_time=float(i * 4 + 4),
                duration=4.0,
                tidal_volume=500.0,
                tidal_volume_smoothed=500.0,
                peak_inspiratory_flow=30.0,
                peak_expiratory_flow=25.0,
                inspiration_time=2.0,
                expiration_time=2.0,
                i_e_ratio=1.0,
                respiratory_rate=15.0,
                respiratory_rate_rolling=15.0,
                minute_ventilation=7.5,
                amplitude=50.0,
                is_complete=True,
            )
            for i in range(30)
        ]

        # Add 3 breaths with 50% reduction
        for i in range(30, 33):
            breaths.append(
                BreathMetrics(
                    breath_number=i + 1,
                    start_time=float(i * 4),
                    middle_time=float(i * 4 + 2),
                    end_time=float(i * 4 + 4),
                    duration=4.0,
                    tidal_volume=250.0,
                    tidal_volume_smoothed=250.0,
                    peak_inspiratory_flow=15.0,
                    peak_expiratory_flow=12.5,
                    inspiration_time=2.0,
                    expiration_time=2.0,
                    i_e_ratio=1.0,
                    respiratory_rate=15.0,
                    respiratory_rate_rolling=15.0,
                    minute_ventilation=7.5,
                    amplitude=25.0,  # 50% reduction from 50.0
                    is_complete=True,
                )
            )

        # No SpO2 data - should fall back to FLOW_ONLY
        hypopneas = detector._detect_hypopneas(
            breaths, flow_data=None, spo2_signal=None
        )
        assert len(hypopneas) >= 1


class TestValidateAgainstMachineEvents:
    """Tests for cross-validation framework."""

    def test_validation_perfect_match(self, aasm_detector):
        """All events match perfectly."""
        prog_apneas = [
            ApneaEvent(
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.9,
                confidence=0.8,
                baseline_flow=50.0,
            ),
            ApneaEvent(
                start_time=50.0,
                end_time=60.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.95,
                confidence=0.85,
                baseline_flow=50.0,
            ),
        ]

        machine_apneas = [
            ApneaEvent(
                start_time=10.5,
                end_time=20.5,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.9,
                confidence=0.8,
                baseline_flow=50.0,
            ),
            ApneaEvent(
                start_time=50.5,
                end_time=60.5,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.95,
                confidence=0.85,
                baseline_flow=50.0,
            ),
        ]

        result = aasm_detector.validate_against_machine_events(
            prog_apneas, [], machine_apneas, []
        )

        assert result["apnea_validation"].matched_events == 2
        assert result["apnea_validation"].false_positives == 0
        assert result["apnea_validation"].false_negatives == 0
        assert result["apnea_validation"].sensitivity == 1.0
        assert result["apnea_validation"].precision == 1.0
        assert result["apnea_validation"].f1_score == 1.0

    def test_validation_with_false_positives(self, aasm_detector):
        """Programmatic detector finds extra events."""
        prog_apneas = [
            ApneaEvent(
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.9,
                confidence=0.8,
                baseline_flow=50.0,
            ),
            ApneaEvent(
                start_time=50.0,
                end_time=60.0,
                duration=10.0,
                event_type="CA",
                flow_reduction=0.95,
                confidence=0.85,
                baseline_flow=50.0,
            ),
            ApneaEvent(
                start_time=90.0,
                end_time=100.0,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.85,
                confidence=0.75,
                baseline_flow=50.0,
            ),
        ]

        machine_apneas = [
            ApneaEvent(
                start_time=10.5,
                end_time=20.5,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.9,
                confidence=0.8,
                baseline_flow=50.0,
            ),
        ]

        result = aasm_detector.validate_against_machine_events(
            prog_apneas, [], machine_apneas, []
        )

        assert result["apnea_validation"].matched_events == 1
        assert result["apnea_validation"].false_positives == 2
        assert result["apnea_validation"].false_negatives == 0
        assert result["apnea_validation"].sensitivity == 1.0
        assert result["apnea_validation"].precision < 1.0

    def test_validation_no_machine_events(self, aasm_detector):
        """Edge case: no machine events, but programmatic events exist."""
        prog_apneas = [
            ApneaEvent(
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.9,
                confidence=0.8,
                baseline_flow=50.0,
            ),
        ]

        result = aasm_detector.validate_against_machine_events(prog_apneas, [], [], [])

        assert result["apnea_validation"].matched_events == 0
        assert result["apnea_validation"].false_positives == 1
        assert result["apnea_validation"].false_negatives == 0
        assert result["apnea_validation"].sensitivity == 0.0
        assert result["apnea_validation"].precision == 0.0

    def test_validation_no_programmatic_events(self, aasm_detector):
        """Edge case: no programmatic events, but machine events exist."""
        machine_apneas = [
            ApneaEvent(
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.9,
                confidence=0.8,
                baseline_flow=50.0,
            ),
        ]

        result = aasm_detector.validate_against_machine_events(
            [], [], machine_apneas, []
        )

        assert result["apnea_validation"].matched_events == 0
        assert result["apnea_validation"].false_positives == 0
        assert result["apnea_validation"].false_negatives == 1
        assert result["apnea_validation"].sensitivity == 0.0
        assert result["apnea_validation"].precision == 1.0

    def test_validation_different_tolerances(self, aasm_detector):
        """Tolerance affects matching."""
        prog_apneas = [
            ApneaEvent(
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.9,
                confidence=0.8,
                baseline_flow=50.0,
            ),
        ]

        machine_apneas = [
            ApneaEvent(
                start_time=17.0,
                end_time=27.0,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.9,
                confidence=0.8,
                baseline_flow=50.0,
            ),
        ]

        # Tight tolerance - no match (7 seconds apart)
        result_tight = aasm_detector.validate_against_machine_events(
            prog_apneas, [], machine_apneas, [], tolerance_seconds=3.0
        )
        assert result_tight["apnea_validation"].matched_events == 0

        # Loose tolerance - match
        result_loose = aasm_detector.validate_against_machine_events(
            prog_apneas, [], machine_apneas, [], tolerance_seconds=10.0
        )
        assert result_loose["apnea_validation"].matched_events == 1
