"""
Flow limitation classification algorithm.

This module implements rule-based classification of breath waveforms into
7 flow limitation classes based on extracted features. Classes range from
normal (Class 1) to severe flow limitation (Class 7).
"""

import logging

import numpy as np

from snore.analysis.shared.feature_extractors import (
    PeakFeatures,
    ShapeFeatures,
    StatisticalFeatures,
)
from snore.analysis.shared.types import FlowPattern, SessionFlowAnalysis
from snore.constants import FLOW_LIMITATION_CLASSES
from snore.constants import FlowLimitationConstants as FLC

logger = logging.getLogger(__name__)


class FlowLimitationClassifier:
    """
    Classifies breath waveforms into 7 flow limitation classes.

    Uses rule-based algorithms that map extracted features (flatness,
    peak count/position, plateau duration, etc.) to clinical flow
    limitation classifications.

    Example:
        >>> classifier = FlowLimitationClassifier()
        >>> pattern = classifier.classify_flow_pattern(
        ...     breath_number=1,
        ...     shape_features=shape_features,
        ...     peak_features=peak_features
        ... )
        >>> print(f"Class {pattern.flow_class}: {pattern.class_name}")
    """

    def __init__(self, confidence_threshold: float = FLC.CONFIDENCE_THRESHOLD):
        """
        Initialize the classifier.

        Args:
            confidence_threshold: Minimum confidence for reliable classification
        """
        self.confidence_threshold = confidence_threshold
        self.classes = FLOW_LIMITATION_CLASSES
        logger.info(
            f"FlowLimitationClassifier initialized with {len(self.classes)} classes"
        )

    def classify_flow_pattern(
        self,
        breath_number: int,
        shape_features: ShapeFeatures,
        peak_features: PeakFeatures,
        statistical_features: StatisticalFeatures | None = None,
    ) -> FlowPattern:
        """
        Classify a single breath into one of 7 flow limitation classes.

        Args:
            breath_number: Sequential breath number
            shape_features: Shape characteristics (flatness, plateau, etc.)
            peak_features: Peak analysis results
            statistical_features: Optional statistical features

        Returns:
            FlowPattern with classification and confidence score
        """
        matched_features: dict[str, float | int | str] = {}

        flow_class = self._apply_classification_rules(
            shape_features, peak_features, matched_features
        )

        confidence = self._calculate_confidence(flow_class, matched_features)

        class_info = self.classes[flow_class]

        return FlowPattern(
            breath_number=breath_number,
            flow_class=flow_class,
            class_name=class_info["name"],
            confidence=confidence,
            matched_features=matched_features,
            severity=class_info["severity"],
        )

    def _apply_classification_rules(
        self,
        shape: ShapeFeatures,
        peaks: PeakFeatures,
        matched_features: dict[str, Any],
    ) -> int:
        """
        Apply rule-based logic to determine flow limitation class.

        Rules are ordered from most specific (complex patterns) to most
        general (normal breathing). Returns the first matching class.

        Args:
            shape: Shape features
            peaks: Peak features
            matched_features: Dictionary to populate with matched features

        Returns:
            Flow limitation class (1-7)
        """
        flatness = shape.flatness_index
        plateau = shape.plateau_duration
        peak_count = peaks.peak_count
        peak_positions = peaks.peak_positions
        symmetry = shape.symmetry_score
        kurtosis = shape.kurtosis

        peak_position = peak_positions[0] if peak_positions else 0.5

        if (
            flatness > FLC.FL_CLASS7_FLATNESS_MIN
            and plateau > FLC.FL_CLASS7_PLATEAU_MIN
        ):
            matched_features["flatness_very_high"] = flatness
            matched_features["plateau_extensive"] = plateau
            return 7

        if (
            peak_position > FLC.FL_CLASS6_PEAK_POSITION_MIN
            and flatness > FLC.FL_CLASS6_FLATNESS_MIN
            and plateau > FLC.FL_CLASS6_PLATEAU_MIN
        ):
            matched_features["late_peak"] = peak_position
            matched_features["flatness_high"] = flatness
            matched_features["plateau_present"] = plateau
            return 6

        if (
            flatness > FLC.FL_CLASS5_FLATNESS_MIN
            and peak_count == 1
            and FLC.FL_CLASS5_PEAK_POSITION_MIN
            <= peak_position
            <= FLC.FL_CLASS5_PEAK_POSITION_MAX
            and plateau > FLC.FL_CLASS5_PLATEAU_MIN
        ):
            matched_features["flatness_high"] = flatness
            matched_features["central_peak"] = peak_position
            matched_features["plateau_both_sides"] = plateau
            return 5

        if (
            flatness > FLC.FL_CLASS4_FLATNESS_MIN
            and peak_position < FLC.FL_CLASS4_PEAK_POSITION_MAX
            and plateau > FLC.FL_CLASS4_PLATEAU_MIN
        ):
            matched_features["early_peak"] = peak_position
            matched_features["plateau_sustained"] = plateau
            matched_features["flatness_moderate"] = flatness
            return 4

        if (
            peak_count >= FLC.FL_CLASS3_PEAK_COUNT_MIN
            and flatness > FLC.FL_CLASS3_FLATNESS_MIN
        ):
            max_prominence = (
                max(peaks.peak_prominences) if peaks.peak_prominences else 0
            )
            if max_prominence < FLC.FL_CLASS3_PROMINENCE_MAX:
                matched_features["multiple_small_peaks"] = peak_count
                matched_features["low_prominence"] = max_prominence
                matched_features["flatness_mild"] = flatness
                return 3

        if peak_count == FLC.FL_CLASS2_PEAK_COUNT:
            if len(peaks.inter_peak_intervals) > 0:
                spacing = peaks.inter_peak_intervals[0]
                if spacing > FLC.FL_CLASS2_PEAK_SPACING_MIN:
                    matched_features["double_peak"] = peak_count
                    matched_features["peak_spacing"] = spacing
                    return 2

        if (
            flatness < FLC.FL_CLASS1_FLATNESS_MAX
            and abs(symmetry) < FLC.FL_CLASS1_SYMMETRY_MAX
            and kurtosis > FLC.FL_CLASS1_KURTOSIS_MIN
        ):
            matched_features["low_flatness"] = flatness
            matched_features["symmetric"] = symmetry
            matched_features["high_kurtosis"] = kurtosis
            return 1

        matched_features["default_classification"] = "no_clear_match"
        if flatness < 0.5:
            return 1
        elif flatness < 0.7:
            return 4
        else:
            return 7

    def _calculate_confidence(
        self, flow_class: int, matched_features: dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for classification.

        Confidence is based on:
        - Number of features that matched the classification rules
        - Strength of feature values relative to thresholds
        - Absence of conflicting features

        Args:
            flow_class: Assigned flow class
            matched_features: Features that matched during classification

        Returns:
            Confidence score (0-1)
        """
        if "default_classification" in matched_features:
            return FLC.FL_DEFAULT_CONFIDENCE

        feature_count = len(matched_features)

        if feature_count >= FLC.FL_HIGH_FEATURE_COUNT:
            base_confidence = FLC.FL_HIGH_CONFIDENCE
        elif feature_count == FLC.FL_MEDIUM_FEATURE_COUNT:
            base_confidence = FLC.FL_MEDIUM_CONFIDENCE
        else:
            base_confidence = FLC.FL_LOW_CONFIDENCE

        if "flatness_very_high" in matched_features:
            if matched_features["flatness_very_high"] > FLC.FL_VERY_HIGH_FLATNESS:
                base_confidence = min(1.0, base_confidence + FLC.FL_CONFIDENCE_BONUS)

        if "double_peak" in matched_features:
            spacing = matched_features.get("peak_spacing", 0)
            if spacing > FLC.FL_HIGH_PEAK_SPACING:
                base_confidence = min(1.0, base_confidence + FLC.FL_CONFIDENCE_BONUS)

        return base_confidence

    def calculate_flow_limitation_index(self, patterns: list[FlowPattern]) -> float:
        """
        Calculate session-level flow limitation index.

        The index is a weighted average of flow limitation severity across
        all breaths, ranging from 0 (no limitation) to 1 (severe limitation).

        Args:
            patterns: List of classified breath patterns

        Returns:
            Flow limitation index (0-1)
        """
        if not patterns:
            return 0.0

        total_weight = 0.0
        for pattern in patterns:
            class_weight = self.classes[pattern.flow_class]["weight"]
            total_weight += class_weight * pattern.confidence

        return total_weight / len(patterns)

    def analyze_session(
        self,
        breath_features: list[tuple[Any, ...]],
    ) -> SessionFlowAnalysis:
        """
        Analyze all breaths in a session.

        Args:
            breath_features: List of (breath_number, shape_features, peak_features)
                tuples for each breath in the session

        Returns:
            SessionFlowAnalysis with complete classification results
        """
        patterns = []
        class_distribution = {i: 0 for i in range(1, 8)}

        for breath_number, shape_features, peak_features in breath_features:
            pattern = self.classify_flow_pattern(
                breath_number, shape_features, peak_features
            )
            patterns.append(pattern)
            class_distribution[pattern.flow_class] += 1

        fl_index = self.calculate_flow_limitation_index(patterns)
        avg_confidence = np.mean([p.confidence for p in patterns])

        return SessionFlowAnalysis(
            total_breaths=len(patterns),
            class_distribution=class_distribution,
            flow_limitation_index=fl_index,
            average_confidence=float(avg_confidence),
            patterns=patterns,
        )
