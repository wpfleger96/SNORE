"""Detection mode registry."""

from typing import Any

from snore.analysis.modes.config import AVAILABLE_CONFIGS, DEFAULT_MODE
from snore.analysis.modes.detector import EventDetector
from snore.analysis.modes.types import (
    BaselineMethod,
    DetectionModeConfig,
    ModeResult,
)

__all__ = [
    "EventDetector",
    "ModeResult",
    "DetectionModeConfig",
    "BaselineMethod",
    "AVAILABLE_CONFIGS",
    "DEFAULT_MODE",
    "get_mode",
    "get_all_modes",
]


def get_mode(name: str, **kwargs: Any) -> EventDetector:
    """
    Factory function to get a detection mode by name.

    Args:
        name: Mode name (e.g., "aasm", "aasm_relaxed")
        **kwargs: Reserved for future config overrides

    Returns:
        EventDetector instance configured for the mode

    Raises:
        ValueError: If mode name is not recognized
    """
    if name not in AVAILABLE_CONFIGS:
        raise ValueError(
            f"Unknown mode: {name}. Available: {list(AVAILABLE_CONFIGS.keys())}"
        )

    config = AVAILABLE_CONFIGS[name]
    return EventDetector(config)


def get_all_modes(**kwargs: Any) -> list[EventDetector]:
    """
    Get instances of all available modes.

    Args:
        **kwargs: Reserved for future config overrides

    Returns:
        List of all available EventDetector instances
    """
    return [EventDetector(config) for config in AVAILABLE_CONFIGS.values()]
