"""
Unit tests for flow limitation classification.

Tests rule-based classification of breaths into 7 flow limitation classes,
confidence scoring, and session-level flow limitation index calculation.
"""

import pytest

from snore.analysis.shared.feature_extractors import (
    PeakFeatures,
    ShapeFeatures,
    WaveformFeatureExtractor,
)
from snore.analysis.shared.flow_limitation import (
    FlowLimitationClassifier,
    FlowPattern,
)
from snore.constants import FlowLimitationConstants as FLC
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
        # Regression: the class-1 rule must actually match (confidence > 0.5),
        # not fall through to the 0.5 fallback as it did when the rule required
        # positive kurtosis (unsatisfiable for smooth platykurtic breaths).
        assert pattern.confidence > 0.5
        assert "low_flatness" in pattern.matched_features

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
            plateau_fraction=0.1,
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
            plateau_fraction=0.1,
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
            plateau_fraction=0.4,
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
            plateau_fraction=0.3,
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
            plateau_fraction=0.3,
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

    def test_dominant_late_peak_drives_class6(self):
        """The most-prominent peak — not the first — selects the shape class.

        A small early peak (position 0.2) precedes a dominant late peak
        (position 0.8, far higher prominence).  Indexing by the first peak
        would read 0.2 and miss Class 6; the dominant-peak rule must read 0.8.
        """
        classifier = FlowLimitationClassifier()

        shape = ShapeFeatures(
            flatness_index=0.65,
            plateau_duration=0.5,
            plateau_fraction=0.3,
            symmetry_score=0.3,
            kurtosis=0.8,
            rise_time=0.8,
            fall_time=0.2,
        )

        peaks = PeakFeatures(
            peak_count=2,
            peak_positions=[0.2, 0.8],
            peak_prominences=[0.2, 0.9],
            inter_peak_intervals=[0.5],
        )

        pattern = classifier.classify_flow_pattern(
            breath_number=1,
            shape_features=shape,
            peak_features=peaks,
        )

        assert pattern.flow_class == 6
        assert pattern.matched_features["late_peak"] == 0.8


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
            plateau_fraction=1.0,
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


class TestPlateauFractionControls:
    """Classifier keys plateau on plateau_fraction (ratio), not plateau_duration."""

    def test_classification_follows_plateau_fraction_not_duration(self):
        """When plateau_duration (seconds) and plateau_fraction (ratio) disagree,
        classification tracks plateau_fraction.

        Both breaths share a flatness in (FL_CLASS4_FLATNESS_MIN,
        FL_CLASS1_FLATNESS_MAX) and an early peak, so only the plateau gate
        separates Class 4 from Class 1.  A large plateau_duration that would
        clear an absolute-seconds threshold must not promote a breath whose
        plateau_fraction is below the ratio threshold, and a tiny
        plateau_duration must not block one whose fraction clears it.
        """
        classifier = FlowLimitationClassifier()

        peaks = PeakFeatures(
            peak_count=1,
            peak_positions=[0.2],
            peak_prominences=[0.9],
            inter_peak_intervals=[],
        )

        # Big duration, sub-threshold fraction → NOT Class 4.
        low_fraction = ShapeFeatures(
            flatness_index=0.42,
            plateau_duration=5.0,
            plateau_fraction=0.1,
            symmetry_score=0.0,
            kurtosis=1.0,
            rise_time=0.2,
            fall_time=0.8,
        )
        # Identical except the two plateau fields: tiny duration, supra-threshold
        # fraction → Class 4.
        high_fraction = ShapeFeatures(
            flatness_index=0.42,
            plateau_duration=0.01,
            plateau_fraction=0.5,
            symmetry_score=0.0,
            kurtosis=1.0,
            rise_time=0.2,
            fall_time=0.8,
        )

        low = classifier.classify_flow_pattern(
            breath_number=1, shape_features=low_fraction, peak_features=peaks
        )
        high = classifier.classify_flow_pattern(
            breath_number=2, shape_features=high_fraction, peak_features=peaks
        )

        assert low.flow_class != 4
        assert high.flow_class == 4
        assert "plateau_sustained" in high.matched_features


class TestConfidenceScoring:
    """Test confidence score calculation."""

    def test_confidence_large_margins_high(self):
        """Confidence follows the margin formula, not a per-class constant.

        Values far past their thresholds must match FL_CONFIDENCE_BASE +
        FL_CONFIDENCE_MARGIN_SCALE * mean(margins); the superseded feature-count
        formula ignored how far past the threshold each value sat.
        """
        classifier = FlowLimitationClassifier()

        flatness = 0.92
        plateau_fraction = 1.0
        shape = ShapeFeatures(
            flatness_index=flatness,
            plateau_duration=0.85,
            plateau_fraction=plateau_fraction,
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

        # Class 7 matches on its two continuous thresholds; reproduce their
        # margins (each clipped to [0, 1]) and the mean-margin confidence.
        margins = [
            (flatness - FLC.FL_CLASS7_FLATNESS_MIN) / FLC.FL_CLASS7_FLATNESS_MIN,
            min(
                (plateau_fraction - FLC.FL_CLASS7_PLATEAU_FRAC_MIN)
                / FLC.FL_CLASS7_PLATEAU_FRAC_MIN,
                1.0,
            ),
        ]
        expected = FLC.FL_CONFIDENCE_BASE + FLC.FL_CONFIDENCE_MARGIN_SCALE * (
            sum(margins) / len(margins)
        )
        assert pattern.flow_class == 7
        assert pattern.confidence == pytest.approx(expected)
        assert pattern.confidence >= 0.7

    def test_rule_match_at_threshold_yields_base_confidence(self):
        """A rule match with every value at its threshold has ~zero margin, so
        confidence collapses to FL_CONFIDENCE_BASE (the margin term vanishes)."""
        classifier = FlowLimitationClassifier()

        eps = 1e-9
        shape = ShapeFeatures(
            flatness_index=FLC.FL_CLASS7_FLATNESS_MIN + eps,
            plateau_duration=0.9,
            plateau_fraction=FLC.FL_CLASS7_PLATEAU_FRAC_MIN + eps,
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
        assert pattern.confidence == pytest.approx(FLC.FL_CONFIDENCE_BASE, abs=1e-6)

    def test_base_confidence_exceeds_fallback_invariant(self):
        """Rule-matched confidence must sit strictly above the fallback the
        nightly fl_class_ge4_pct gate keys on; guards against a lowered BASE."""
        assert FLC.FL_CONFIDENCE_BASE > FLC.FL_DEFAULT_CONFIDENCE

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

    def test_fli_strictly_increasing_in_class(self):
        """Uniform sessions must score monotonically by class severity.

        Guards against the confidence-weighting inversion where a confident
        Class 6 outscored a less-certain Class 7.  The index now uses class
        weights only, so it must be strictly increasing in class number.
        """
        classifier = FlowLimitationClassifier()

        def uniform_index(flow_class: int) -> float:
            patterns = [
                FlowPattern(
                    breath_number=i,
                    flow_class=flow_class,
                    class_name="x",
                    # Confidence deliberately varies inversely with severity to
                    # prove it does not affect the ordering.
                    confidence=1.0 - 0.1 * flow_class,
                    matched_features={},
                    severity="normal",
                )
                for i in range(1, 6)
            ]
            return classifier.calculate_flow_limitation_index(patterns)

        indices = [uniform_index(c) for c in range(1, 8)]
        assert all(a < b for a, b in zip(indices, indices[1:], strict=False))


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
                plateau_fraction=0.1,
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
                plateau_fraction=0.6,
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
