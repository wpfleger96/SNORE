"""
Flow limitation classification algorithm.

This module implements rule-based classification of breath waveforms into
7 flow limitation classes based on extracted features. Classes range from
normal (Class 1) to severe flow limitation (Class 7).
"""

import logging

from typing import Any

import numpy as np

from snore.analysis.shared.feature_extractors import PeakFeatures, ShapeFeatures
from snore.analysis.shared.types import FlowPattern, SessionFlowAnalysis
from snore.constants import FLOW_LIMITATION_CLASSES
from snore.constants import FlowLimitationConstants as FLC

logger = logging.getLogger(__name__)

__all__ = ["FlowLimitationClassifier", "SessionFlowAnalysis"]


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

    def __init__(self) -> None:
        """Initialize the classifier."""
        self.classes = FLOW_LIMITATION_CLASSES
        logger.info(
            f"FlowLimitationClassifier initialized with {len(self.classes)} classes"
        )

    def classify_flow_pattern(
        self,
        breath_number: int,
        shape_features: ShapeFeatures,
        peak_features: PeakFeatures,
    ) -> FlowPattern:
        """
        Classify a single breath into one of 7 flow limitation classes.

        Args:
            breath_number: Sequential breath number
            shape_features: Shape characteristics (flatness, plateau, etc.)
            peak_features: Peak analysis results

        Returns:
            FlowPattern with classification and confidence score
        """
        matched_features: dict[str, float | int | str] = {}
        margins: list[float] = []

        flow_class = self._apply_classification_rules(
            shape_features, peak_features, matched_features, margins
        )

        confidence = self._calculate_confidence(matched_features, margins)

        class_info = self.classes[flow_class]

        return FlowPattern(
            breath_number=breath_number,
            flow_class=flow_class,
            class_name=class_info["name"],
            confidence=confidence,
            matched_features=matched_features,
            severity=class_info["severity"],
        )

    @staticmethod
    def _margin_above(value: float, threshold: float) -> float:
        """Fractional distance of ``value`` above ``threshold`` for a ``>`` rule."""
        if threshold <= 0:
            return 0.0
        return float(np.clip((value - threshold) / threshold, 0.0, 1.0))

    @staticmethod
    def _margin_below(value: float, threshold: float) -> float:
        """Fractional distance of ``value`` below ``threshold`` for a ``<`` rule."""
        if threshold <= 0:
            return 0.0
        return float(np.clip((threshold - value) / threshold, 0.0, 1.0))

    def _apply_classification_rules(
        self,
        shape: ShapeFeatures,
        peaks: PeakFeatures,
        matched_features: dict[str, Any],
        margins: list[float],
    ) -> int:
        """
        Apply rule-based logic to determine flow limitation class.

        Rules are ordered from most specific (complex patterns) to most
        general (normal breathing). Returns the first matching class.

        Args:
            shape: Shape features
            peaks: Peak features
            matched_features: Dictionary to populate with matched features
            margins: List to populate with per-condition threshold margins
                (evidence strength) of the matched rule

        Returns:
            Flow limitation class (1-7)
        """
        flatness = shape.flatness_index
        plateau_fraction = shape.plateau_fraction
        peak_count = peaks.peak_count
        peak_positions = peaks.peak_positions
        symmetry = shape.symmetry_score

        # The dominant (most prominent) peak defines the shape class; a small
        # early peak must not hide a dominant late peak (e.g. Class 6).
        if peaks.peak_prominences:
            dominant_idx = int(np.argmax(peaks.peak_prominences))
            peak_position = peak_positions[dominant_idx]
        elif peak_positions:
            peak_position = peak_positions[0]
        else:
            peak_position = 0.5

        # Confidence margins are contributed only by continuous threshold
        # comparisons (``>``/``<``), whose distance past the threshold measures
        # evidence strength.  Equality, range-membership, and count checks are
        # structural gates and append no margin.
        if (
            flatness > FLC.FL_CLASS7_FLATNESS_MIN
            and plateau_fraction > FLC.FL_CLASS7_PLATEAU_FRAC_MIN
        ):
            matched_features["flatness_very_high"] = flatness
            matched_features["plateau_extensive"] = plateau_fraction
            margins.append(self._margin_above(flatness, FLC.FL_CLASS7_FLATNESS_MIN))
            margins.append(
                self._margin_above(plateau_fraction, FLC.FL_CLASS7_PLATEAU_FRAC_MIN)
            )
            return 7

        if (
            peak_position > FLC.FL_CLASS6_PEAK_POSITION_MIN
            and flatness > FLC.FL_CLASS6_FLATNESS_MIN
            and plateau_fraction > FLC.FL_CLASS6_PLATEAU_FRAC_MIN
        ):
            matched_features["late_peak"] = peak_position
            matched_features["flatness_high"] = flatness
            matched_features["plateau_present"] = plateau_fraction
            margins.append(
                self._margin_above(peak_position, FLC.FL_CLASS6_PEAK_POSITION_MIN)
            )
            margins.append(self._margin_above(flatness, FLC.FL_CLASS6_FLATNESS_MIN))
            margins.append(
                self._margin_above(plateau_fraction, FLC.FL_CLASS6_PLATEAU_FRAC_MIN)
            )
            return 6

        if (
            flatness > FLC.FL_CLASS5_FLATNESS_MIN
            and peak_count == 1
            and FLC.FL_CLASS5_PEAK_POSITION_MIN
            <= peak_position
            <= FLC.FL_CLASS5_PEAK_POSITION_MAX
            and plateau_fraction > FLC.FL_CLASS5_PLATEAU_FRAC_MIN
        ):
            matched_features["flatness_high"] = flatness
            matched_features["central_peak"] = peak_position
            matched_features["plateau_both_sides"] = plateau_fraction
            margins.append(self._margin_above(flatness, FLC.FL_CLASS5_FLATNESS_MIN))
            margins.append(
                self._margin_above(plateau_fraction, FLC.FL_CLASS5_PLATEAU_FRAC_MIN)
            )
            return 5

        if (
            flatness > FLC.FL_CLASS4_FLATNESS_MIN
            and peak_position < FLC.FL_CLASS4_PEAK_POSITION_MAX
            and plateau_fraction > FLC.FL_CLASS4_PLATEAU_FRAC_MIN
        ):
            matched_features["early_peak"] = peak_position
            matched_features["plateau_sustained"] = plateau_fraction
            matched_features["flatness_moderate"] = flatness
            margins.append(self._margin_above(flatness, FLC.FL_CLASS4_FLATNESS_MIN))
            margins.append(
                self._margin_below(peak_position, FLC.FL_CLASS4_PEAK_POSITION_MAX)
            )
            margins.append(
                self._margin_above(plateau_fraction, FLC.FL_CLASS4_PLATEAU_FRAC_MIN)
            )
            return 4

        if (
            peak_count >= FLC.FL_CLASS3_PEAK_COUNT_MIN
            and flatness > FLC.FL_CLASS3_FLATNESS_MIN
        ):
            max_prominence = (
                peaks.peak_prominences[dominant_idx] if peaks.peak_prominences else 0
            )
            if max_prominence < FLC.FL_CLASS3_PROMINENCE_MAX:
                matched_features["multiple_small_peaks"] = peak_count
                matched_features["low_prominence"] = max_prominence
                matched_features["flatness_mild"] = flatness
                margins.append(self._margin_above(flatness, FLC.FL_CLASS3_FLATNESS_MIN))
                margins.append(
                    self._margin_below(max_prominence, FLC.FL_CLASS3_PROMINENCE_MAX)
                )
                return 3

        if peak_count == FLC.FL_CLASS2_PEAK_COUNT:
            if len(peaks.inter_peak_intervals) > 0:
                spacing = peaks.inter_peak_intervals[0]
                if spacing > FLC.FL_CLASS2_PEAK_SPACING_MIN:
                    matched_features["double_peak"] = peak_count
                    matched_features["peak_spacing"] = spacing
                    margins.append(
                        self._margin_above(spacing, FLC.FL_CLASS2_PEAK_SPACING_MIN)
                    )
                    return 2

        if (
            flatness < FLC.FL_CLASS1_FLATNESS_MAX
            and abs(symmetry) < FLC.FL_CLASS1_SYMMETRY_MAX
            and peak_count == 1
        ):
            matched_features["low_flatness"] = flatness
            matched_features["symmetric"] = symmetry
            matched_features["single_peak"] = peak_count
            margins.append(self._margin_below(flatness, FLC.FL_CLASS1_FLATNESS_MAX))
            margins.append(
                self._margin_below(abs(symmetry), FLC.FL_CLASS1_SYMMETRY_MAX)
            )
            return 1

        matched_features["default_classification"] = "no_clear_match"
        if flatness < FLC.FL_FALLBACK_FLATNESS_CLASS1_MAX:
            return 1
        elif flatness < FLC.FL_FALLBACK_FLATNESS_CLASS4_MAX:
            return 4
        else:
            return 7

    def _calculate_confidence(
        self, matched_features: dict[str, Any], margins: list[float]
    ) -> float:
        """
        Calculate confidence score for classification from threshold margins.

        A rule matches on how far each value sits past its threshold: the
        confidence scales with the mean margin (evidence strength) rather than a
        per-class constant.  Rule-matched confidence is always strictly above the
        fallback (``FL_DEFAULT_CONFIDENCE``), which is what nightly metrics gate on.

        Args:
            matched_features: Features that matched during classification
            margins: Per-condition threshold margins of the matched rule

        Returns:
            Confidence score (0-1)
        """
        if "default_classification" in matched_features or not margins:
            return FLC.FL_DEFAULT_CONFIDENCE

        mean_margin = float(np.mean(margins))
        return FLC.FL_CONFIDENCE_BASE + FLC.FL_CONFIDENCE_MARGIN_SCALE * mean_margin

    def calculate_flow_limitation_index(self, patterns: list[FlowPattern]) -> float:
        """
        Calculate session-level flow limitation index.

        The index is the mean class-severity weight across all breaths, ranging
        from 0 (no limitation) to 1 (severe limitation).  Confidence is
        deliberately excluded: it measures certainty, not severity, and folding
        it in inverts the ordering (a confident Class 6 outscoring a less-certain
        Class 7).

        Args:
            patterns: List of classified breath patterns

        Returns:
            Flow limitation index (0-1)
        """
        if not patterns:
            return 0.0

        total_weight = 0.0
        for pattern in patterns:
            total_weight += self.classes[pattern.flow_class]["weight"]

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
