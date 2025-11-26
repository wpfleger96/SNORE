"""
Algorithm modules for waveform analysis.

This module provides breath segmentation, feature extraction, and pattern
detection algorithms for CPAP data analysis.
"""

from oscar_mcp.analysis.algorithms.breath_segmenter import (
    BreathMetrics,
    BreathPhases,
    BreathSegmenter,
)
from oscar_mcp.analysis.algorithms.feature_extractors import (
    PeakFeatures,
    ShapeFeatures,
    SpectralFeatures,
    StatisticalFeatures,
    WaveformFeatureExtractor,
)

__all__ = [
    "BreathSegmenter",
    "BreathPhases",
    "BreathMetrics",
    "WaveformFeatureExtractor",
    "ShapeFeatures",
    "PeakFeatures",
    "StatisticalFeatures",
    "SpectralFeatures",
]
