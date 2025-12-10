"""
Respiratory event detection types.

This module re-exports event dataclasses from the shared types module.
The old RespiratoryEventDetector class has been replaced by the
config-based EventDetector in analysis.modes.detector.
"""

from snore.analysis.shared.types import ApneaEvent, EventTimeline, HypopneaEvent

__all__ = [
    "ApneaEvent",
    "HypopneaEvent",
    "EventTimeline",
]
