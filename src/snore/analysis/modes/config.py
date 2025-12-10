"""Predefined detection mode configurations."""

from snore.analysis.modes.types import BaselineMethod, DetectionModeConfig
from snore.constants import EventDetectionConstants as EDC

__all__ = [
    "DetectionModeConfig",
    "AASM_CONFIG",
    "AASM_RELAXED_CONFIG",
    "AVAILABLE_CONFIGS",
    "DEFAULT_MODE",
]

# ============================================================================
# AASM-Compliant Configuration
# ============================================================================

AASM_CONFIG = DetectionModeConfig(
    name="aasm",
    description="AASM Scoring Manual v2.6 compliant detection",
    baseline_method=BaselineMethod.TIME,
    baseline_window=EDC.BASELINE_WINDOW_SECONDS,  # 120.0 seconds (2 minutes)
    baseline_percentile=90,
    apnea_threshold=EDC.APNEA_FLOW_REDUCTION_THRESHOLD,  # 0.90 (90%)
    apnea_validation_threshold=0.90,  # Strict AASM compliance
    hypopnea_min_threshold=EDC.HYPOPNEA_MIN_REDUCTION,  # 0.30 (30%)
    hypopnea_max_threshold=EDC.HYPOPNEA_MAX_REDUCTION,  # 0.89 (89%)
    min_event_duration=EDC.MIN_EVENT_DURATION,  # 10.0 seconds
    merge_gap=EDC.MERGE_GAP_SECONDS,  # 3.0 seconds
    metric="amplitude",
)

# ============================================================================
# AASM-Based Relaxed Configuration
# ============================================================================

AASM_RELAXED_CONFIG = DetectionModeConfig(
    name="aasm_relaxed",
    description="AASM-based with relaxed thresholds for machine matching",
    baseline_method=BaselineMethod.BREATH,
    baseline_window=30,  # 30 breaths (~2 min at 15 breaths/min)
    baseline_percentile=90,
    apnea_threshold=EDC.APNEA_FLOW_REDUCTION_THRESHOLD,  # 0.90 (90%)
    apnea_validation_threshold=0.85,  # Relaxed to catch borderline events
    hypopnea_min_threshold=EDC.HYPOPNEA_MIN_REDUCTION,  # 0.30 (30%)
    hypopnea_max_threshold=EDC.HYPOPNEA_MAX_REDUCTION,  # 0.89 (89%)
    min_event_duration=EDC.MIN_EVENT_DURATION,  # 10.0 seconds
    merge_gap=EDC.MERGE_GAP_SECONDS,  # 3.0 seconds
    metric="amplitude",
)

# ============================================================================
# Mode Registry
# ============================================================================

AVAILABLE_CONFIGS: dict[str, DetectionModeConfig] = {
    "aasm": AASM_CONFIG,
    "aasm_relaxed": AASM_RELAXED_CONFIG,
}

DEFAULT_MODE = "aasm"
