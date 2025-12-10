"""
Detection mode registry for respiratory event detection.

Provides factory functions to instantiate and manage different detection modes.
"""

from typing import Any

from .aasm import AASMDetectionMode
from .aasm_relaxed import AASMRelaxedDetectionMode
from .base import DetectionMode, ModeResult

# Mode imports will be added as they're implemented
# from .resmed import ResMedDetectionMode

__all__ = [
    "DetectionMode",
    "ModeResult",
    "AASMDetectionMode",
    "AASMRelaxedDetectionMode",
    "AVAILABLE_MODES",
    "DEFAULT_MODE",
    "get_mode",
    "get_all_modes",
]

AVAILABLE_MODES: dict[str, type[DetectionMode]] = {
    "aasm": AASMDetectionMode,
    "aasm_relaxed": AASMRelaxedDetectionMode,
}

DEFAULT_MODE = "aasm"


def get_mode(name: str, **kwargs: Any) -> DetectionMode:
    """
    Factory function to get a detection mode by name.

    Args:
        name: Mode name (e.g., "aasm", "resmed")
        **kwargs: Mode-specific initialization parameters

    Returns:
        DetectionMode instance

    Raises:
        ValueError: If mode name is not recognized
    """
    if name not in AVAILABLE_MODES:
        raise ValueError(
            f"Unknown mode: {name}. Available: {list(AVAILABLE_MODES.keys())}"
        )
    return AVAILABLE_MODES[name](**kwargs)


def get_all_modes(**kwargs: Any) -> list[DetectionMode]:
    """
    Get instances of all available modes.

    Args:
        **kwargs: Mode-specific initialization parameters

    Returns:
        List of all available DetectionMode instances
    """
    return [mode_class(**kwargs) for mode_class in AVAILABLE_MODES.values()]
