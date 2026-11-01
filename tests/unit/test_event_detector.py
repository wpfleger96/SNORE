"""Unit tests for EventDetector methods across all modes."""

import numpy as np
import pytest

from snore.analysis.modes.baseline import (
    _calculate_breath_based_baseline,
    _calculate_time_based_baseline,
)
from snore.analysis.modes.classification import (
    _check_desaturation,
    _classify_apnea_type,
)
from snore.analysis.modes.config import (
    AASM_CONFIG,
    AASM_RELAXED_CONFIG,
    RESMED_CONFIG,
    DetectionModeConfig,
)
from snore.analysis.modes.detector import EventDetector
from snore.analysis.modes.postprocess import (
    EVENT_MATCH_TOLERANCE_SECONDS,
    _calculate_event_overlap,
    _deduplicate_events,
    _merge_adjacent_events,
    _merge_two_events,
    _validate_event,
)
from snore.analysis.modes.types import HypopneaMode
from snore.analysis.shared.types import (
    ApneaEvent,
    BreathMetrics,
    HypopneaEvent,
    RERAEvent,
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

        deduplicated = _deduplicate_events(
            resmed_detector.config, events, overlap_threshold=0.5
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

        deduplicated = _deduplicate_events(
            resmed_detector.config, events, overlap_threshold=0.5
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

        deduplicated = _deduplicate_events(
            resmed_detector.config, events, overlap_threshold=0.5
        )

        assert len(deduplicated) == 1
        assert deduplicated[0].confidence == 0.95
        assert deduplicated[0].detection_method == "gap"

    def test_dedupe_empty_list(self, resmed_detector):
        """Empty list should return empty list."""
        deduplicated = _deduplicate_events(
            resmed_detector.config, [], overlap_threshold=0.5
        )

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

        deduplicated = _deduplicate_events(
            resmed_detector.config, events, overlap_threshold=0.5
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
        events = resmed_detector._detect_events_resmed(
            sample_breaths, sample_flow_data, sample_rate=25.0
        )

        assert len(events) >= 1
        assert events[0].event_type == "CA"

    def test_resmed_no_flow_data(self, resmed_detector, sample_breaths):
        """ResMed should work with breaths only (minimal flow data)."""
        timestamps = np.array([0.0])
        flow_values = np.array([0.0])
        events = resmed_detector._detect_events_resmed(
            sample_breaths, (timestamps, flow_values), sample_rate=25.0
        )

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

        merged = _merge_two_events(event1, event2)

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

        merged = _merge_two_events(event1, event2)

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

        merged = _merge_two_events(event1, event2)

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

        merged = _merge_two_events(event1, event2)

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

        merged = _merge_two_events(event1, event2)

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

        merged = _merge_adjacent_events(events, max_gap=3.0)

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

        merged = _merge_adjacent_events(events, max_gap=3.0)

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

        merged = _merge_adjacent_events(events, max_gap=3.0)

        assert len(merged) == 2

    def test_merge_empty_list(self, aasm_detector):
        """Empty list should return empty list."""
        merged = _merge_adjacent_events([], max_gap=3.0)

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

        merged = _merge_adjacent_events(events, max_gap=3.0)

        assert len(merged) == 1
        assert merged[0] == events[0]


class TestClassifyApneaType:
    """Tests for _classify_apnea_type() method."""

    def test_classify_obstructive(self, aasm_detector):
        """High effort flow signal should be classified as OA."""
        flow_signal = np.array([5, -5, 6, -6, 5, -5, 6, -6, 5, -5])

        apnea_type, confidence = _classify_apnea_type(
            flow_signal=flow_signal, sample_rate=25.0
        )

        assert apnea_type == "OA"
        assert 0.5 <= confidence <= 1.0  # Should have reasonable confidence

    def test_classify_central(self, aasm_detector):
        """Flat flow signal should be classified as CA."""
        flow_signal = np.array(
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        )

        apnea_type, confidence = _classify_apnea_type(
            flow_signal=flow_signal, sample_rate=25.0
        )

        assert apnea_type == "CA"
        assert 0.5 <= confidence <= 1.0  # Should have reasonable confidence

    def test_classify_mixed(self, aasm_detector):
        """Medium effort should be classified as MA."""
        # Signal with moderate variation to produce effort score in MA range (0.05-0.15)
        # std ≈ 3.5, range ≈ 7, normalizes to effort ≈ 0.08-0.10
        flow_signal = np.array([3.0, -3.0, 3.5, -2.5, 3.0, -3.0, 3.5, -2.5, 3.0, -3.0])

        apnea_type, confidence = _classify_apnea_type(
            flow_signal=flow_signal, sample_rate=25.0
        )

        assert apnea_type == "MA"
        assert 0.3 <= confidence <= 0.6  # MA should have lower confidence (borderline)

    def test_classify_no_flow_data(self, aasm_detector):
        """No flow data should be classified as UA."""
        apnea_type, confidence = _classify_apnea_type(
            flow_signal=None, sample_rate=25.0
        )

        assert apnea_type == "UA"
        assert confidence == 0.2  # UA should have low confidence


class TestCheckDesaturation:
    """Tests for _check_desaturation() method."""

    def test_desaturation_detected(self, aasm_detector):
        """SpO2 drop of 4% should be detected."""
        spo2_values = np.array([96, 95, 94, 93, 92, 91, 90])

        has_desat = _check_desaturation(spo2_values)

        assert has_desat is True

    def test_no_desaturation(self, aasm_detector):
        """SpO2 drop of 2% should not be detected."""
        spo2_values = np.array([96, 95, 94, 94, 95, 96])

        has_desat = _check_desaturation(spo2_values)

        assert has_desat is False

    def test_insufficient_data(self, aasm_detector):
        """Less than 2 samples should return False."""
        spo2_values = np.array([96])

        has_desat = _check_desaturation(spo2_values)

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

        is_valid = _validate_event(aasm_detector.config, reductions, 0, 5)

        assert is_valid is True

    def test_validate_below_threshold(self, aasm_detector):
        """Event with max 85% reduction should not validate for AASM strict."""
        reductions = np.array([0.80, 0.82, 0.85, 0.83, 0.81])

        is_valid = _validate_event(aasm_detector.config, reductions, 0, 5)

        assert is_valid is False

    def test_validate_relaxed_threshold(self, aasm_relaxed_detector):
        """Event with 87% reduction should validate for aasm_relaxed."""
        reductions = np.array([0.80, 0.82, 0.87, 0.83, 0.81])

        is_valid = _validate_event(aasm_relaxed_detector.config, reductions, 0, 5)

        assert is_valid is True

    def test_validate_empty_reductions(self, aasm_detector):
        """Empty reductions array should return False."""
        reductions = np.array([])

        is_valid = _validate_event(aasm_detector.config, reductions, 0, 0)

        assert is_valid is False


class TestDetectApneas:
    """Tests for _detect_apneas() method - core AASM functionality."""

    def test_detect_apnea_90_percent_reduction(self, aasm_detector, apnea_breaths):
        """Breaths with 95% reduction should be detected as apnea."""
        timestamps = np.arange(0, 80, 0.04)
        flow_values = np.zeros(len(timestamps))
        apneas = aasm_detector._detect_apneas(
            apnea_breaths, flow_data=(timestamps, flow_values), sample_rate=25.0
        )

        assert len(apneas) >= 1
        assert all(a.event_type in ["OA", "CA", "MA", "UA"] for a in apneas)

    def test_apnea_empty_breaths(self, aasm_detector):
        """Empty breath list should return no apneas."""
        timestamps = np.array([])
        flow_values = np.array([])
        apneas = aasm_detector._detect_apneas(
            [], flow_data=(timestamps, flow_values), sample_rate=25.0
        )

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

        baseline = _calculate_time_based_baseline(aasm_detector.config, breaths, 35)

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

        baseline = _calculate_time_based_baseline(aasm_detector.config, breaths, 35)

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

        baseline = _calculate_time_based_baseline(aasm_detector.config, breaths, 35)

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

        baseline = _calculate_breath_based_baseline(
            aasm_relaxed_detector.config, breaths, 45
        )

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

        baseline = _calculate_breath_based_baseline(
            aasm_relaxed_detector.config, breaths, 45
        )

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
        assert result["apnea_validation"].sensitivity == 1.0
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

    def test_validation_with_hypopneas(self, aasm_detector):
        """Validate hypopnea events against machine events."""
        prog_hypopneas = [
            HypopneaEvent(
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                flow_reduction=0.4,
                confidence=0.75,
                baseline_flow=50.0,
                has_arousal=False,
                has_desaturation=True,
            ),
            HypopneaEvent(
                start_time=50.0,
                end_time=60.0,
                duration=10.0,
                flow_reduction=0.5,
                confidence=0.8,
                baseline_flow=50.0,
                has_arousal=True,
                has_desaturation=False,
            ),
        ]

        machine_hypopneas = [
            HypopneaEvent(
                start_time=10.5,
                end_time=20.5,
                duration=10.0,
                flow_reduction=0.4,
                confidence=0.75,
                baseline_flow=50.0,
                has_arousal=False,
                has_desaturation=True,
            ),
        ]

        result = aasm_detector.validate_against_machine_events(
            [], prog_hypopneas, [], machine_hypopneas
        )

        assert result["hypopnea_validation"].matched_events == 1
        assert result["hypopnea_validation"].false_positives == 1
        assert result["hypopnea_validation"].false_negatives == 0
        assert result["hypopnea_validation"].sensitivity == 1.0
        assert result["hypopnea_validation"].precision < 1.0

    @staticmethod
    def _make_rera(start_time: float) -> RERAEvent:
        return RERAEvent(
            start_time=start_time,
            end_time=start_time + 12.0,
            duration=12.0,
            obstructed_breath_count=3,
            recovery_amplitude_increase_pct=0.6,
            confidence=0.7,
            baseline_flow=50.0,
        )

    def test_validation_reras_within_tolerance_match(self, aasm_detector):
        """Programmatic and machine RERAs within tolerance match; status is 'ok'."""
        prog_reras = [self._make_rera(100.0)]
        machine_reras = [self._make_rera(100.5)]

        result = aasm_detector.validate_against_machine_events(
            [], [], [], [], programmatic_reras=prog_reras, machine_reras=machine_reras
        )

        assert result["rera_validation_status"] == "ok"
        assert result["rera_validation"].matched_events == 1
        assert result["rera_validation"].false_positives == 0
        assert result["rera_validation"].false_negatives == 0
        # RERAs fold into the combined lists → matched, nothing spurious.
        assert result["false_positive_events"] == []
        assert result["false_negative_events"] == []

    def test_validation_reras_outside_tolerance(self, aasm_detector):
        """RERAs more than tolerance apart → false positive and false negative."""
        prog_reras = [self._make_rera(100.0)]
        machine_reras = [self._make_rera(120.0)]

        result = aasm_detector.validate_against_machine_events(
            [], [], [], [], programmatic_reras=prog_reras, machine_reras=machine_reras
        )

        assert result["rera_validation_status"] == "ok"
        assert result["rera_validation"].matched_events == 0
        assert result["rera_validation"].false_positives == 1
        assert result["rera_validation"].false_negatives == 1
        assert len(result["false_positive_events"]) == 1
        assert len(result["false_negative_events"]) == 1

    def test_validation_reras_at_exactly_tolerance_boundary_match(self, aasm_detector):
        """A RERA exactly EVENT_MATCH_TOLERANCE_SECONDS apart matches (<= is inclusive)."""
        prog_reras = [self._make_rera(100.0)]
        machine_reras = [self._make_rera(100.0 + EVENT_MATCH_TOLERANCE_SECONDS)]

        result = aasm_detector.validate_against_machine_events(
            [], [], [], [], programmatic_reras=prog_reras, machine_reras=machine_reras
        )

        assert result["rera_validation_status"] == "ok"
        assert result["rera_validation"].matched_events == 1
        assert result["rera_validation"].false_positives == 0
        assert result["rera_validation"].false_negatives == 0
        assert result["false_positive_events"] == []
        assert result["false_negative_events"] == []

    @staticmethod
    def _make_apnea(start_time: float) -> ApneaEvent:
        return ApneaEvent(
            start_time=start_time,
            end_time=start_time + 10.0,
            duration=10.0,
            event_type="OA",
            flow_reduction=0.9,
            confidence=0.8,
            baseline_flow=50.0,
        )

    @staticmethod
    def _make_hypopnea(start_time: float) -> HypopneaEvent:
        return HypopneaEvent(
            start_time=start_time,
            end_time=start_time + 10.0,
            duration=10.0,
            flow_reduction=0.4,
            confidence=0.7,
            baseline_flow=50.0,
        )

    def test_combined_lists_pair_a_machine_event_only_once(self, aasm_detector):
        """A second programmatic event near a taken machine event is a false positive."""
        result = aasm_detector.validate_against_machine_events(
            [self._make_apnea(100.0), self._make_apnea(102.0)],
            [],
            [self._make_apnea(101.0)],
            [],
        )

        assert [e.start_time for e in result["false_positive_events"]] == [102.0]
        assert result["false_negative_events"] == []

    def test_combined_lists_pair_a_programmatic_event_only_once(self, aasm_detector):
        """A second machine event near a taken programmatic event is a false negative."""
        result = aasm_detector.validate_against_machine_events(
            [self._make_apnea(100.0)],
            [],
            [self._make_apnea(99.0), self._make_apnea(101.0)],
            [],
        )

        assert [e.start_time for e in result["false_negative_events"]] == [101.0]
        assert result["false_positive_events"] == []

    def test_combined_lists_match_across_types_in_start_time_order(self, aasm_detector):
        """Pooled matching follows time order, as ``snore waveform compare`` does.

        Matching the apnea first would take the 101s machine event and strand
        the hypopnea, which only 101s can reach.
        """
        result = aasm_detector.validate_against_machine_events(
            [self._make_apnea(104.0)],
            [self._make_hypopnea(97.0)],
            [self._make_apnea(101.0), self._make_apnea(108.0)],
            [],
        )

        assert result["false_positive_events"] == []
        assert result["false_negative_events"] == []

    def test_validation_reras_average_is_three_way_when_machine_present(
        self, aasm_detector
    ):
        """A missed machine RERA pulls overall average sensitivity below the 2-way value."""
        machine_reras = [self._make_rera(100.0)]

        result = aasm_detector.validate_against_machine_events(
            [], [], [], [], machine_reras=machine_reras
        )

        # apnea (empty→1.0) + hypopnea (empty→1.0) + rera (missed→0.0), /3
        assert result["rera_validation_status"] == "ok"
        assert result["rera_validation"].sensitivity == 0.0
        assert (
            abs(result["overall_agreement"]["average_sensitivity"] - (2.0 / 3.0)) < 1e-9
        )

    def test_validation_no_machine_reras_status_and_two_way_average(
        self, aasm_detector
    ):
        """No machine RERAs → status flag set, prog RERAs excluded from combined lists and averages."""
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
                start_time=10.5,
                end_time=20.5,
                duration=10.0,
                event_type="OA",
                flow_reduction=0.9,
                confidence=0.8,
                baseline_flow=50.0,
            ),
        ]
        prog_reras = [self._make_rera(200.0)]

        result = aasm_detector.validate_against_machine_events(
            prog_apneas,
            [],
            machine_apneas,
            [],
            programmatic_reras=prog_reras,
            machine_reras=[],
        )

        assert result["rera_validation_status"] == "no_machine_re_events"
        # Programmatic RERA must NOT surface as a cross-type false positive.
        assert result["false_positive_events"] == []
        # Average stays 2-way (apnea 1.0 + hypopnea 1.0) / 2 = 1.0.
        assert abs(result["overall_agreement"]["average_sensitivity"] - 1.0) < 1e-9

    def test_hypopnea_aasm_3pct_mode(self, aasm_detector):
        """Hypopnea detection with AASM_3PCT mode and SpO2 desaturation."""
        from snore.analysis.modes.config import DetectionModeConfig
        from snore.analysis.modes.types import BaselineMethod

        config = DetectionModeConfig(
            name="test",
            description="Test config for AASM_3PCT mode",
            baseline_method=BaselineMethod.TIME,
            baseline_window=120.0,
            hypopnea_mode=HypopneaMode.AASM_3PCT,
        )
        detector = EventDetector(config)

        # Create 30 normal breaths for baseline
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

        # Add 4 breaths with 50% reduction (16 seconds > 10 second minimum)
        for i in range(30, 34):
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
                    minute_ventilation=3.75,
                    amplitude=25.0,
                    is_complete=True,
                )
            )

        # Create SpO2 signal with desaturation during hypopnea
        timestamps = np.array([float(i * 4) for i in range(34)])
        spo2_signal = np.full(34, 97.0)
        # Gradual drop within event window: 97 -> 93 = 4% drop
        spo2_signal[30] = 97.0
        spo2_signal[31] = 95.0
        spo2_signal[32] = 94.0
        spo2_signal[33] = 93.0

        flow_data = (timestamps, np.zeros(34))

        hypopneas = detector._detect_hypopneas(breaths, flow_data, spo2_signal)

        assert len(hypopneas) >= 1
        if len(hypopneas) > 0:
            assert hypopneas[0].has_desaturation is True

    def test_hypopnea_aasm_4pct_mode(self, aasm_detector):
        """Hypopnea detection with AASM_4PCT mode and SpO2 desaturation."""
        from snore.analysis.modes.config import DetectionModeConfig
        from snore.analysis.modes.types import BaselineMethod

        config = DetectionModeConfig(
            name="test",
            description="Test config for AASM_4PCT mode",
            baseline_method=BaselineMethod.TIME,
            baseline_window=120.0,
            hypopnea_mode=HypopneaMode.AASM_4PCT,
        )
        detector = EventDetector(config)

        # Create 30 normal breaths for baseline
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

        # Add 4 breaths with 50% reduction (16 seconds > 10 second minimum)
        for i in range(30, 34):
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
                    minute_ventilation=3.75,
                    amplitude=25.0,
                    is_complete=True,
                )
            )

        # Create SpO2 signal with desaturation during hypopnea
        timestamps = np.array([float(i * 4) for i in range(34)])
        spo2_signal = np.full(34, 97.0)
        # Gradual drop within event window: 97 -> 92 = 5% drop
        spo2_signal[30] = 97.0
        spo2_signal[31] = 95.0
        spo2_signal[32] = 93.0
        spo2_signal[33] = 92.0

        flow_data = (timestamps, np.zeros(34))

        hypopneas = detector._detect_hypopneas(breaths, flow_data, spo2_signal)

        assert len(hypopneas) >= 1
        if len(hypopneas) > 0:
            assert hypopneas[0].has_desaturation is True

    def test_multiple_reras_in_session(self, aasm_detector):
        """Detect multiple RERA events in a single session."""
        # Create baseline breaths
        breaths = []
        for i in range(50):
            amplitude = 50.0
            # First RERA: breaths 10-12 with recovery at 13
            if 10 <= i < 13:
                amplitude = 38.0  # 24% reduction
            elif i == 13:
                amplitude = 70.0  # >50% recovery increase

            # Second RERA: breaths 30-32 with recovery at 33
            if 30 <= i < 33:
                amplitude = 37.0  # 26% reduction
            elif i == 33:
                amplitude = 68.0  # >50% recovery increase

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
                    amplitude=amplitude,
                    is_complete=True,
                )
            )

        apneas: list[ApneaEvent] = []
        hypopneas: list[HypopneaEvent] = []

        reras = aasm_detector._detect_reras(breaths, apneas, hypopneas)

        assert len(reras) >= 2


# =============================================================================
# Vectorized near-zero-flow oracle tests
# =============================================================================


def _near_zero_flow_scalar_oracle(
    flow_signal: np.ndarray,
    timestamps: np.ndarray,
    zero_threshold: float = 2.0,
    min_duration: float = 10.0,
) -> list[tuple[float, float]]:
    """Verbatim copy of the original scalar loop; returns (start_time, end_time) pairs."""
    total_duration = timestamps[-1] - timestamps[0]
    sample_rate = len(timestamps) / total_duration
    min_samples = int(min_duration * sample_rate)
    near_zero_mask = np.abs(flow_signal) < zero_threshold

    results: list[tuple[float, float]] = []
    in_event = False
    event_start = 0
    for i, is_near_zero in enumerate(near_zero_mask):
        if is_near_zero and not in_event:
            in_event = True
            event_start = i
        elif not is_near_zero and in_event:
            in_event = False
            event_length = i - event_start
            if event_length >= min_samples:
                results.append(
                    (float(timestamps[event_start]), float(timestamps[i - 1]))
                )
    if in_event:
        event_length = len(near_zero_mask) - event_start
        if event_length >= min_samples:
            results.append((float(timestamps[event_start]), float(timestamps[-1])))
    return results


class TestDetectNearZeroFlowVectorized:
    """Oracle-comparison and edge-case tests for the vectorized _detect_near_zero_flow()."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_near_zero_matches_oracle_random_signals(self, resmed_detector, seed):
        """Vectorized result matches scalar oracle across random background signals."""
        rng = np.random.default_rng(seed)
        n = 2500  # 100 seconds at 25 Hz
        timestamps = np.linspace(0.0, 100.0, n)
        flow_values = rng.uniform(5.0, 20.0, n)

        # Short run below min_samples (75 < 250): not detected
        flow_values[200:275] = 0.5
        # Long interior run above min_samples (375 > 250): detected
        flow_values[400:775] = 0.5
        # Another short run below threshold: not detected
        flow_values[900:950] = 0.5
        # Trailing run reaching end of signal (400 > 250): detected
        flow_values[2100:] = 0.5

        oracle_runs = _near_zero_flow_scalar_oracle(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )
        actual_events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(actual_events) == len(oracle_runs)
        for event, (oracle_start, oracle_end) in zip(
            actual_events, oracle_runs, strict=True
        ):
            assert event.start_time == pytest.approx(oracle_start, abs=1e-9)
            assert event.end_time == pytest.approx(oracle_end, abs=1e-9)

    def test_near_zero_leading_run_at_index_zero_matches_oracle(self, resmed_detector):
        """Run starting at index 0 is handled identically to oracle."""
        timestamps = np.arange(0.0, 100.0, 0.04)  # 2500 samples, 25 Hz
        flow_values = np.ones_like(timestamps) * 10.0
        flow_values[:500] = 0.5  # Leading run: 500 samples = 20 s > 10 s minimum

        oracle_runs = _near_zero_flow_scalar_oracle(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )
        actual_events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(actual_events) == len(oracle_runs)
        for event, (oracle_start, oracle_end) in zip(
            actual_events, oracle_runs, strict=True
        ):
            assert event.start_time == pytest.approx(oracle_start, abs=1e-9)
            assert event.end_time == pytest.approx(oracle_end, abs=1e-9)

    def test_near_zero_trailing_run_end_time_equals_last_timestamp(
        self, resmed_detector
    ):
        """Trailing run that reaches end of signal closes at timestamps[-1]."""
        timestamps = np.arange(0.0, 100.0, 0.04)
        flow_values = np.ones_like(timestamps) * 10.0
        flow_values[2000:] = 0.5  # 500 trailing samples = 20 s

        events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(events) == 1
        assert events[0].end_time == pytest.approx(float(timestamps[-1]), abs=1e-9)

    def test_near_zero_no_near_zero_samples_returns_no_events(self, resmed_detector):
        """Signal entirely above threshold produces no events."""
        timestamps = np.arange(0.0, 100.0, 0.04)
        flow_values = np.ones_like(timestamps) * 10.0  # All well above 2.0

        events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(events) == 0

    def test_near_zero_entire_signal_near_zero_produces_one_spanning_event(
        self, resmed_detector
    ):
        """Signal entirely below threshold yields one event spanning the full signal."""
        timestamps = np.arange(0.0, 100.0, 0.04)
        flow_values = np.ones_like(timestamps) * 0.5  # All below threshold

        events = resmed_detector._detect_near_zero_flow(
            flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
        )

        assert len(events) == 1
        assert events[0].start_time == pytest.approx(float(timestamps[0]), abs=1e-9)
        assert events[0].end_time == pytest.approx(float(timestamps[-1]), abs=1e-9)


def _uniform_breath(number: int, start: float, amplitude: float) -> BreathMetrics:
    """Build a 4-second breath at a fixed amplitude for detector tests."""
    return BreathMetrics(
        breath_number=number,
        start_time=start,
        middle_time=start + 2.0,
        end_time=start + 4.0,
        duration=4.0,
        tidal_volume=amplitude * 10.0,
        tidal_volume_smoothed=amplitude * 10.0,
        peak_inspiratory_flow=amplitude,
        peak_expiratory_flow=amplitude * 0.8,
        inspiration_time=2.0,
        expiration_time=2.0,
        i_e_ratio=1.0,
        respiratory_rate=15.0,
        respiratory_rate_rolling=15.0,
        minute_ventilation=7.5,
        amplitude=amplitude,
        is_complete=True,
    )


class TestHypopneaBaselineContamination:
    """Hypopnea-contained breaths must be excluded from downstream baselines."""

    def test_hypopnea_breaths_marked_in_event_and_excluded_from_baseline(self):
        """Detected hypopnea breaths are flagged in_event, decontaminating baselines.

        The fix marks hypopnea-contained breaths in_event so they no longer feed
        the p90 baselines used by downstream RERA detection. Here the eval window
        straddles the flagged run: pre-fix (flags cleared) all six breaths feed a
        percentile; post-fix the three flagged breaths are dropped, leaving fewer
        than the time-based 5-sample floor so the clean fallback baseline is
        returned. The computed baseline therefore differs pre/post fix.
        """
        # Time-based baseline over the previous 24 s (~6 breaths at 4 s each).
        config = AASM_CONFIG.model_copy(update={"baseline_window": 24.0})
        detector = EventDetector(config)

        breaths = [_uniform_breath(i + 1, float(i * 4), 50.0) for i in range(10)]
        # 3 hypopnea breaths (amplitude 15 → 70% reduction vs the 50 baseline);
        # short enough that baseline drift never terminates the run early.
        breaths += [_uniform_breath(i + 1, float(i * 4), 15.0) for i in range(10, 13)]
        breaths += [_uniform_breath(i + 1, float(i * 4), 50.0) for i in range(13, 19)]

        hypopneas = detector._detect_hypopneas(
            breaths, flow_data=None, spo2_signal=None, exclude_events=[]
        )

        assert len(hypopneas) >= 1
        # The whole 3-breath run is contained and flagged; neighbours are not.
        assert [b.in_event for b in breaths[10:13]] == [True, True, True]
        assert breaths[9].in_event is False
        assert breaths[13].in_event is False

        # Downstream baseline at index 13 (window breaths[7:13] = 3 normal + the
        # 3 flagged breaths). Post-fix: 3 valid values < 5 → fallback 30.0.
        baseline_post_fix = _calculate_time_based_baseline(config, breaths, 13)

        # Pre-fix simulation: clear the flags the detector set, so the reduced
        # breaths contaminate the same window (6 values → p90 = 50.0).
        for breath in breaths:
            breath.in_event = False
        baseline_pre_fix = _calculate_time_based_baseline(config, breaths, 13)

        assert baseline_pre_fix == pytest.approx(50.0)
        assert baseline_post_fix == pytest.approx(30.0)
        assert baseline_post_fix != baseline_pre_fix


class TestReraThresholdConfig:
    """RERA detection reads its thresholds from DetectionModeConfig."""

    def _rera_pattern(self) -> list[BreathMetrics]:
        """10 normal breaths, two 25%-reduced breaths, then a recovery breath."""
        breaths = [_uniform_breath(i + 1, float(i * 4), 50.0) for i in range(10)]
        breaths.append(_uniform_breath(11, 40.0, 37.5))  # 25% reduction
        breaths.append(_uniform_breath(12, 44.0, 37.5))  # 25% reduction
        breaths.append(_uniform_breath(13, 48.0, 60.0))  # recovery: +60% over 37.5
        breaths += [_uniform_breath(i + 1, float(i * 4), 50.0) for i in range(13, 15)]
        return breaths

    def test_default_thresholds_detect_rera(self):
        """The default [0.20, 0.30) band detects the 25%-reduction sequence."""
        breaths = self._rera_pattern()
        reras = EventDetector(AASM_CONFIG)._detect_reras(breaths, [], [])
        assert len(reras) == 1

    def test_raised_reduction_min_suppresses_rera(self):
        """Raising rera_reduction_min above 25% removes the detection."""
        config = AASM_CONFIG.model_copy(
            update={"rera_reduction_min": 0.30, "rera_reduction_max": 0.40}
        )
        breaths = self._rera_pattern()
        reras = EventDetector(config)._detect_reras(breaths, [], [])
        assert len(reras) == 0

    def test_raised_recovery_increase_min_suppresses_rera(self):
        """Requiring a larger recovery increase than 60% removes the detection."""
        config = AASM_CONFIG.model_copy(update={"rera_recovery_increase_min": 0.80})
        breaths = self._rera_pattern()
        reras = EventDetector(config)._detect_reras(breaths, [], [])
        assert len(reras) == 0

    def test_lowered_recovery_reduction_max_suppresses_rera(self):
        """A recovery-reduction ceiling of 0 rejects the recovery breath.

        The recovery breath exceeds baseline, so its clamped reduction is 0.0;
        the default 0.10 ceiling admits it, but 0.0 (strict <) rejects it. This
        isolates rera_recovery_reduction_max — no other threshold changes.
        """
        config = AASM_CONFIG.model_copy(update={"rera_recovery_reduction_max": 0.0})
        breaths = self._rera_pattern()
        reras = EventDetector(config)._detect_reras(breaths, [], [])
        assert len(reras) == 0


class TestDetectEventsModeIsolation:
    """detect_events resets per-breath state so multi-mode runs stay independent."""

    def _flow_data(self, breaths: list[BreathMetrics]) -> tuple[np.ndarray, np.ndarray]:
        """Flat flow signal spanning the breath timeline (no apnea to classify)."""
        timestamps = np.arange(0.0, breaths[-1].end_time + 1.0, 0.1)
        return timestamps, np.zeros_like(timestamps)

    def _hypopnea_pattern(self) -> list[BreathMetrics]:
        """15 normal breaths, a 4-breath 50%-reduced run, then normal breaths."""
        breaths = [_uniform_breath(i + 1, float(i * 4), 50.0) for i in range(15)]
        breaths += [_uniform_breath(i + 1, float(i * 4), 25.0) for i in range(15, 19)]
        breaths += [_uniform_breath(i + 1, float(i * 4), 50.0) for i in range(19, 30)]
        return breaths

    def test_preset_in_event_flags_do_not_change_results(self):
        """Contaminated in_event flags from a prior mode do not alter detection.

        Without the reset, pre-set flags would starve the baselines (falling to
        the fixed floor) and suppress the hypopnea; the reset makes each
        detect_events call self-contained and mode-order-independent.
        """
        detector = EventDetector(AASM_CONFIG)

        fresh = self._hypopnea_pattern()
        flow_data = self._flow_data(fresh)
        expected = detector.detect_events(
            fresh,
            flow_data=flow_data,
            sample_rate=25.0,
            session_duration_hours=1.0,
        )
        # Guard: the fixture must actually produce an event, else the test is vacuous.
        assert len(expected.hypopneas) >= 1

        contaminated = self._hypopnea_pattern()
        for breath in contaminated:
            breath.in_event = True
        actual = detector.detect_events(
            contaminated,
            flow_data=flow_data,
            sample_rate=25.0,
            session_duration_hours=1.0,
        )

        assert len(actual.apneas) == len(expected.apneas)
        assert len(actual.hypopneas) == len(expected.hypopneas)
        assert len(actual.reras) == len(expected.reras)
        assert actual.ahi == pytest.approx(expected.ahi)
        assert actual.rdi == pytest.approx(expected.rdi)


class TestThresholdBandValidation:
    """DetectionModeConfig rejects inverted [min, max) threshold bands."""

    def _config_dict(self, **overrides: float) -> dict:
        base = AASM_CONFIG.model_dump()
        base.update(overrides)
        return base

    def test_inverted_rera_band_rejected(self):
        """rera_reduction_min >= rera_reduction_max is rejected at construction."""
        with pytest.raises(ValueError, match="rera_reduction_min"):
            DetectionModeConfig.model_validate(
                self._config_dict(rera_reduction_min=0.40, rera_reduction_max=0.30)
            )

    def test_inverted_hypopnea_band_rejected(self):
        """hypopnea_min_threshold >= hypopnea_max_threshold is rejected."""
        with pytest.raises(ValueError, match="hypopnea_min_threshold"):
            DetectionModeConfig.model_validate(
                self._config_dict(
                    hypopnea_min_threshold=0.90, hypopnea_max_threshold=0.30
                )
            )
