"""
Configuration consistency validation.

This module validates that all configuration sources are consistent,
preventing the dual-source-of-truth issue where constants.py and patterns.py
had conflicting threshold values.
"""

import logging
from oscar_mcp.constants import EventDetectionConstants as EDC
from oscar_mcp.knowledge.patterns import RESPIRATORY_EVENTS

logger = logging.getLogger(__name__)


def validate_event_detection_config() -> bool:
    """
    Validate that event detection configuration is consistent across all sources.

    Checks that patterns.py respiratory event definitions match the threshold
    values defined in constants.py (the single source of truth).

    Returns:
        True if configuration is consistent

    Raises:
        ValueError: If configuration inconsistencies are detected
    """
    errors = []

    pattern_apnea_threshold = RESPIRATORY_EVENTS["obstructive_apnea"]["flow_reduction_percent"]
    expected_apnea_threshold = int(EDC.APNEA_FLOW_REDUCTION_THRESHOLD * 100)

    if pattern_apnea_threshold != expected_apnea_threshold:
        errors.append(
            f"Apnea threshold mismatch: patterns.py has {pattern_apnea_threshold}% "
            f"but constants.py defines {expected_apnea_threshold}%"
        )

    pattern_hypopnea_min = RESPIRATORY_EVENTS["hypopnea"]["flow_reduction_percent_min"]
    expected_hypopnea_min = int(EDC.HYPOPNEA_MIN_REDUCTION * 100)

    if pattern_hypopnea_min != expected_hypopnea_min:
        errors.append(
            f"Hypopnea min threshold mismatch: patterns.py has {pattern_hypopnea_min}% "
            f"but constants.py defines {expected_hypopnea_min}%"
        )

    pattern_hypopnea_max = RESPIRATORY_EVENTS["hypopnea"]["flow_reduction_percent_max"]
    expected_hypopnea_max = int(EDC.HYPOPNEA_MAX_REDUCTION * 100)

    if pattern_hypopnea_max != expected_hypopnea_max:
        errors.append(
            f"Hypopnea max threshold mismatch: patterns.py has {pattern_hypopnea_max}% "
            f"but constants.py defines {expected_hypopnea_max}%"
        )

    pattern_min_duration = RESPIRATORY_EVENTS["obstructive_apnea"]["min_duration_seconds"]
    expected_min_duration = int(EDC.MIN_EVENT_DURATION)

    if pattern_min_duration != expected_min_duration:
        errors.append(
            f"Min duration mismatch: patterns.py has {pattern_min_duration}s "
            f"but constants.py defines {expected_min_duration}s"
        )

    if errors:
        error_message = "Configuration inconsistencies detected:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info("Event detection configuration validated successfully")
    return True


def validate_all_config() -> bool:
    """
    Validate all configuration sources are consistent.

    Returns:
        True if all configuration is consistent

    Raises:
        ValueError: If any configuration inconsistencies are detected
    """
    validate_event_detection_config()

    return True
