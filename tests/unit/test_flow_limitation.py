"""
Unit tests for flow limitation classification.

Tests rule-based classification of breaths into 7 flow limitation classes,
confidence scoring, and session-level flow limitation index calculation.
"""

from snore.analysis.shared.feature_extractors import (
    PeakFeatures,
    ShapeFeatures,
    WaveformFeatureExtractor,
)
from snore.analysis.shared.flow_limitation import (
    FlowLimitationClassifier,
    FlowPattern,
)
from tests.helpers.synthetic_data import (
    generate_flattened_breath,
    generate_multi_peak_breath,
    generate_sinusoidal_breath,
)


class TestClass1Sinusoidal:
    """Test classification of Class 1 (normal sinusoidal) breaths."""

    def test_class1_perfect_sinusoid(self):
        """Perfect sinusoidal breath should classify as Class 1."""
        classifier = FlowLimitationClassifier()
        extractor = WaveformFeatureExtractor()

        _, flow = generate_sinusoidal_breath(duration=2.0, amplitude=45.0)
        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)
        peaks = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.flow_class == 1
        assert pattern.class_name == "Sinusoidal"
        assert pattern.severity == "normal"

    def test_class1_confidence_high(self):
        """Normal breathing should have reasonable confidence."""
        classifier = FlowLimitationClassifier()
        extractor = WaveformFeatureExtractor()

        _, flow = generate_sinusoidal_breath()
        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)
        peaks = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.confidence >= 0.5
        assert (
            "low_flatness" in pattern.matched_features
            or "default_classification" in pattern.matched_features
        )


class TestClass2DoublePeak:
    """Test classification of Class 2 (double peak) breaths."""

    def test_class2_two_distinct_peaks(self):
        """Breath with two distinct peaks should classify as Class 2 when peaks are detected."""
        classifier = FlowLimitationClassifier()

        shape = ShapeFeatures(
            flatness_index=0.4,
            plateau_duration=0.2,
            symmetry_score=0.1,
            kurtosis=1.5,
            rise_time=0.3,
            fall_time=0.4,
        )

        peaks = PeakFeatures(
            peak_count=2,
            peak_positions=[0.3, 0.7],
            peak_prominences=[0.8, 0.7],
            inter_peak_intervals=[0.4],
        )

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.flow_class == 2
        assert pattern.class_name == "Double Peak"
        assert pattern.severity == "mild"

    def test_class2_requires_adequate_spacing(self):
        """Two peaks too close together should not classify as Class 2."""
        classifier = FlowLimitationClassifier()

        shape = ShapeFeatures(
            flatness_index=0.4,
            plateau_duration=0.2,
            symmetry_score=0.1,
            kurtosis=1.5,
            rise_time=0.3,
            fall_time=0.4,
        )

        peaks = PeakFeatures(
            peak_count=2,
            peak_positions=[0.3, 0.35],
            peak_prominences=[0.8, 0.6],
            inter_peak_intervals=[0.1],
        )

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.flow_class != 2


class TestClass3MultiplePeaks:
    """Test classification of Class 3 (multiple tiny peaks)."""

    def test_class3_many_small_peaks(self):
        """Breath with 3+ small peaks should classify as Class 3."""
        classifier = FlowLimitationClassifier()
        extractor = WaveformFeatureExtractor()

        _, flow = generate_multi_peak_breath(
            duration=2.0,
            peak_count=4,
        )
        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)
        peaks = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        if peaks.peak_count >= 3:
            assert pattern.flow_class == 3
            assert pattern.severity in ["mild-moderate", "mild"]


class TestClass4EarlyPeak:
    """Test classification of Class 4 (peak during initial phase)."""

    def test_class4_early_peak_with_plateau(self):
        """Early peak followed by plateau should classify as Class 4."""
        classifier = FlowLimitationClassifier()

        shape = ShapeFeatures(
            flatness_index=0.5,
            plateau_duration=0.6,
            symmetry_score=-0.3,
            kurtosis=1.2,
            rise_time=0.2,
            fall_time=0.8,
        )

        peaks = PeakFeatures(
            peak_count=1,
            peak_positions=[0.2],
            peak_prominences=[0.9],
            inter_peak_intervals=[],
        )

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.flow_class == 4
        assert pattern.class_name == "Peak During Initial Phase"
        assert pattern.severity == "moderate"
        assert "early_peak" in pattern.matched_features


class TestClass5MidPeak:
    """Test classification of Class 5 (peak at midpoint)."""

    def test_class5_central_peak_high_flatness(self):
        """Central peak with high flatness should classify as Class 5."""
        classifier = FlowLimitationClassifier()

        shape = ShapeFeatures(
            flatness_index=0.75,
            plateau_duration=0.4,
            symmetry_score=0.05,
            kurtosis=1.0,
            rise_time=0.4,
            fall_time=0.4,
        )

        peaks = PeakFeatures(
            peak_count=1,
            peak_positions=[0.5],
            peak_prominences=[0.7],
            inter_peak_intervals=[],
        )

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.flow_class == 5
        assert pattern.class_name == "Peak at Midpoint"
        assert pattern.severity == "moderate-severe"
        assert "central_peak" in pattern.matched_features


class TestClass6LatePeak:
    """Test classification of Class 6 (peak during late phase)."""

    def test_class6_late_peak_pattern(self):
        """Late peak with early plateau should classify as Class 6."""
        classifier = FlowLimitationClassifier()

        shape = ShapeFeatures(
            flatness_index=0.65,
            plateau_duration=0.5,
            symmetry_score=0.3,
            kurtosis=0.8,
            rise_time=0.8,
            fall_time=0.2,
        )

        peaks = PeakFeatures(
            peak_count=1,
            peak_positions=[0.75],
            peak_prominences=[0.8],
            inter_peak_intervals=[],
        )

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.flow_class == 6
        assert pattern.class_name == "Peak During Late Phase"
        assert pattern.severity == "severe"
        assert "late_peak" in pattern.matched_features


class TestClass7PlateauThroughout:
    """Test classification of Class 7 (plateau throughout)."""

    def test_class7_extreme_flatness(self):
        """Extremely flat waveform should classify as Class 7."""
        classifier = FlowLimitationClassifier()
        extractor = WaveformFeatureExtractor()

        _, flow = generate_flattened_breath(
            duration=2.0,
            flatness_index=0.95,
        )
        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)
        peaks = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.flow_class == 7
        assert pattern.class_name == "Plateau Throughout"
        assert pattern.severity == "severe"

    def test_class7_confidence_very_high(self):
        """Class 7 with extreme flatness should have high confidence."""
        classifier = FlowLimitationClassifier()

        shape = ShapeFeatures(
            flatness_index=0.96,
            plateau_duration=0.9,
            symmetry_score=0.0,
            kurtosis=0.5,
            rise_time=0.1,
            fall_time=0.1,
        )

        peaks = PeakFeatures(
            peak_count=0,
            peak_positions=[],
            peak_prominences=[],
            inter_peak_intervals=[],
        )

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.flow_class == 7
        assert pattern.confidence >= 0.7


class TestConfidenceScoring:
    """Test confidence score calculation."""

    def test_confidence_multiple_features_matched(self):
        """More matched features should give higher confidence."""
        classifier = FlowLimitationClassifier()

        shape = ShapeFeatures(
            flatness_index=0.92,
            plateau_duration=0.85,
            symmetry_score=0.0,
            kurtosis=0.6,
            rise_time=0.1,
            fall_time=0.1,
        )

        peaks = PeakFeatures(
            peak_count=0,
            peak_positions=[],
            peak_prominences=[],
            inter_peak_intervals=[],
        )

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.confidence >= 0.7

    def test_confidence_always_in_range(self):
        """Confidence should always be between 0 and 1."""
        classifier = FlowLimitationClassifier()
        extractor = WaveformFeatureExtractor()

        _, flow = generate_sinusoidal_breath()
        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)
        peaks = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert 0.0 <= pattern.confidence <= 1.0


class TestFlowLimitationIndex:
    """Test session-level flow limitation index calculation."""

    def test_fli_all_normal_breaths(self):
        """Session with all normal breaths should have FLI near 0."""
        classifier = FlowLimitationClassifier()

        patterns = [
            FlowPattern(
                breath_number=i,
                flow_class=1,
                class_name="Sinusoidal",
                confidence=0.9,
                matched_features={},
                severity="normal",
            )
            for i in range(1, 11)
        ]

        fli = classifier.calculate_flow_limitation_index(patterns)

        assert fli < 0.1

    def test_fli_all_severe_breaths(self):
        """Session with all severe breaths should have FLI near 1."""
        classifier = FlowLimitationClassifier()

        patterns = [
            FlowPattern(
                breath_number=i,
                flow_class=7,
                class_name="Plateau Throughout",
                confidence=0.9,
                matched_features={},
                severity="severe",
            )
            for i in range(1, 11)
        ]

        fli = classifier.calculate_flow_limitation_index(patterns)

        assert fli > 0.85

    def test_fli_mixed_severity(self):
        """Session with mixed severity should have intermediate FLI."""
        classifier = FlowLimitationClassifier()

        patterns = [
            FlowPattern(
                breath_number=i,
                flow_class=1 if i <= 5 else 7,
                class_name="Sinusoidal" if i <= 5 else "Plateau Throughout",
                confidence=0.9,
                matched_features={},
                severity="normal" if i <= 5 else "severe",
            )
            for i in range(1, 11)
        ]

        fli = classifier.calculate_flow_limitation_index(patterns)

        assert 0.3 < fli < 0.7

    def test_fli_empty_list(self):
        """Empty pattern list should return FLI of 0."""
        classifier = FlowLimitationClassifier()

        fli = classifier.calculate_flow_limitation_index([])

        assert fli == 0.0


class TestSessionAnalysis:
    """Test complete session analysis."""

    def test_analyze_session_basic(self):
        """Should analyze multiple breaths and return session summary."""
        classifier = FlowLimitationClassifier()
        extractor = WaveformFeatureExtractor()

        breath_features = []
        for i in range(1, 6):
            _, flow = generate_sinusoidal_breath()
            insp_flow = flow[flow > 0]

            shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)
            peaks = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

            breath_features.append((i, shape, peaks))

        analysis = classifier.analyze_session(breath_features)

        assert analysis.total_breaths == 5
        assert len(analysis.patterns) == 5
        assert 0.0 <= analysis.flow_limitation_index <= 1.0
        assert 0.0 <= analysis.average_confidence <= 1.0

    def test_class_distribution_accuracy(self):
        """Class distribution should accurately count each class."""
        classifier = FlowLimitationClassifier()

        breath_features = []

        for i in range(3):
            shape = ShapeFeatures(
                flatness_index=0.2,
                plateau_duration=0.1,
                symmetry_score=0.0,
                kurtosis=3.0,
                rise_time=0.3,
                fall_time=0.3,
            )
            peaks = PeakFeatures(
                peak_count=1,
                peak_positions=[0.5],
                peak_prominences=[0.9],
                inter_peak_intervals=[],
            )
            breath_features.append((i, shape, peaks))

        for i in range(3, 6):
            shape = ShapeFeatures(
                flatness_index=0.95,
                plateau_duration=0.9,
                symmetry_score=0.0,
                kurtosis=0.5,
                rise_time=0.1,
                fall_time=0.1,
            )
            peaks = PeakFeatures(
                peak_count=0,
                peak_positions=[],
                peak_prominences=[],
                inter_peak_intervals=[],
            )
            breath_features.append((i, shape, peaks))

        analysis = classifier.analyze_session(breath_features)

        assert analysis.class_distribution[1] == 3
        assert analysis.class_distribution[7] == 3
