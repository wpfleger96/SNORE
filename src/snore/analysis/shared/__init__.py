"""
Algorithm modules for waveform analysis.

This module provides breath segmentation, feature extraction, and pattern
detection algorithms for CPAP data analysis.
"""

from snore.analysis.shared.breath_segmenter import (
    BreathMetrics,
    BreathSegmenter,
)
from snore.analysis.shared.feature_extractors import (
    PeakFeatures,
    ShapeFeatures,
    WaveformFeatureExtractor,
)

__all__ = [
    "BreathSegmenter",
    "BreathMetrics",
    "WaveformFeatureExtractor",
    "ShapeFeatures",
    "PeakFeatures",
]
