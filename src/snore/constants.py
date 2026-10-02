"""
Constants and mappings for OSCAR CPAP data analysis.

Based on OSCAR's schema.h and machine_common.h definitions.
"""

from pathlib import Path
from typing import Literal, TypedDict

# CPAP Pressure Channels
CPAP_PRESSURE = 0x1000
CPAP_EPAP = 0x1002

# CPAP Event Flags
CPAP_OBSTRUCTIVE = 0x1100
CPAP_HYPOPNEA = 0x1101
CPAP_CLEAR_AIRWAY = 0x1102
CPAP_RERA = 0x1104  # Respiratory effort related arousal
CPAP_FLOW_LIMIT = 0x1106
CPAP_CSR = 0x1107  # Cheyne-Stokes Respiration
CPAP_PERIODIC_BREATHING = 0x1108

# CPAP Waveforms
CPAP_FLOW_RATE = 0x1200
CPAP_MASK_PRESSURE = 0x1201
CPAP_LEAK = 0x1202
CPAP_RESPRATE = 0x1203
CPAP_TIDAL_VOLUME = 0x1204
CPAP_MINUTE_VENT = 0x1205

# CPAP Settings
CPAP_MODE = 0x1400
CPAP_PRESSURE_MIN = 0x1401
CPAP_PRESSURE_MAX = 0x1402
CPAP_EPR_LEVEL = 0x1403  # Expiratory pressure relief

# Oximetry Channels
OXI_SPO2 = 0x2000
OXI_PULSE = 0x2001


# ============================================================================
# Measurement Units
# ============================================================================

# Note: ResMed EDF parsing stores the ASCII pressure unit ("cmH2O") while
# OSCAR data uses the Unicode subscript form ("cmH₂O"). Both spellings are
# kept so each parser continues to store exactly what it stored before.
UNIT_FLOW = "L/min"
UNIT_PRESSURE = "cmH2O"
UNIT_PRESSURE_DISPLAY = "cmH₂O"
UNIT_PERCENT = "%"
UNIT_BPM = "bpm"
UNIT_ML = "mL"
UNIT_SECONDS = "seconds"


# ============================================================================
# Analysis Algorithm Constants
# ============================================================================


class BreathSegmentationConstants:
    """Constants for breath segmentation (breath_segmenter.py)."""

    MIN_BREATH_DURATION = 1.0
    MAX_BREATH_DURATION = 20.0
    ZERO_CROSSING_HYSTERESIS = 2.0
    MIN_BREATH_AMPLITUDE = (
        2.0  # Lowered from 8.0 to detect breaths during low-flow periods
    )

    TIDAL_VOLUME_SMOOTHING_POINTS = 5
    RESPIRATORY_RATE_WINDOW_SECONDS = 60.0


class EventDetectionConstants:
    """
    Constants for respiratory event detection (analysis/modes/detector.py).

    ⚠️ IMPORTANT: This is the SINGLE SOURCE OF TRUTH for event detection thresholds.
    When modifying detection behavior, change values HERE, not in implementation files.

    All threshold values are in decimal format (0.7 = 70%, 0.3 = 30%, etc.)
    """

    MIN_EVENT_DURATION = 10.0
    BASELINE_WINDOW_SECONDS = 120.0
    MERGE_GAP_SECONDS = 3.0

    APNEA_FLOW_REDUCTION_THRESHOLD = 0.9
    HYPOPNEA_MIN_REDUCTION = 0.3
    HYPOPNEA_MAX_REDUCTION = 0.89

    APNEA_EFFORT_HIGH_THRESHOLD = 0.5
    APNEA_EFFORT_LOW_THRESHOLD = 0.1

    SPO2_DESATURATION_DROP = 3.0

    APNEA_BASE_CONFIDENCE = 0.7
    APNEA_HIGH_REDUCTION_BONUS = 0.1
    APNEA_LONG_DURATION_BONUS = 0.1
    APNEA_BASELINE_FLOW_BONUS = 0.1
    APNEA_HIGH_REDUCTION_THRESHOLD = 0.95
    APNEA_LONG_DURATION_THRESHOLD = 15.0
    APNEA_HIGH_BASELINE_THRESHOLD = 20.0

    HYPOPNEA_BASE_CONFIDENCE = 0.6
    HYPOPNEA_IDEAL_MIN_REDUCTION = 0.5
    HYPOPNEA_IDEAL_MAX_REDUCTION = 0.7
    HYPOPNEA_LONG_DURATION_THRESHOLD = 15.0
    HYPOPNEA_DESATURATION_BONUS = 0.2

    SPECTRAL_MIN_SAMPLES = 50
    BREATHING_FREQ_MIN = 0.1
    BREATHING_FREQ_MAX = 0.5

    EVENT_DURATION_RULE_THRESHOLD = 0.9
    EVENT_TERMINATION_RECOVERY = 0.5
    EVENT_TERMINATION_MIN_BREATHS = 2

    # Analysis-time RERA detector (detector.py::_detect_reras): amplitude-
    # crescendo proxy. A run of breaths with amplitude reduction in
    # [RERA_REDUCTION_MIN, RERA_REDUCTION_MAX) followed by a recovery breath
    # (reduction < RERA_RECOVERY_REDUCTION_MAX AND amplitude increase >=
    # RERA_RECOVERY_INCREASE_MIN over the run mean). Distinct from the
    # query-time FL-run proxy in RERAProxyConstants.
    RERA_REDUCTION_MIN = 0.20
    RERA_REDUCTION_MAX = 0.30
    RERA_RECOVERY_REDUCTION_MAX = 0.10
    RERA_RECOVERY_INCREASE_MIN = 0.50


class RERAProxyConstants:
    """Constants for the query-time RERA proxy (breath_service.py).

    Criterion: runs of >= MIN_FL_RUN_LENGTH consecutive stored breaths with
    flow_class >= FL_CLASS_THRESHOLD ending in a recovery breath. Independent of
    the analysis-time amplitude-crescendo RERA detector (EventDetectionConstants
    RERA_* values); versioned separately as RERA_PROXY_ALGO_VERSION.
    """

    FL_CLASS_THRESHOLD = 4
    MIN_FL_RUN_LENGTH = 2
    RECOVERY_AMPLITUDE_MARGIN = 0.20


class PatternDetectionConstants:
    """Constants for complex pattern detection (pattern_detector.py)."""

    MIN_CYCLE_COUNT = 3
    AUTOCORR_THRESHOLD = 0.6

    CSR_MIN_CYCLE_LENGTH = 45.0
    CSR_MAX_CYCLE_LENGTH = 90.0
    CSR_WINDOW_MINUTES = 10.0

    SIGNAL_SMOOTHING_WINDOW = 5
    WAXING_WANING_MIN_SCORE = 0.5

    PERIODIC_MIN_CYCLE = 30.0
    PERIODIC_MAX_CYCLE = 120.0

    REGULARITY_MIN_SCORE = 0.5

    CLUSTER_THRESHOLD_SECONDS = 300.0
    MIN_EVENTS_FOR_POSITIONAL = 5
    MIN_CLUSTER_SIZE = 3

    ENVELOPE_VARIATION_MIN = 0.2

    CSR_THRESHOLD_FACTOR = 0.5

    LOW_TV_THRESHOLD_FACTOR = 0.1
    APNEA_PRESENCE_THRESHOLD = 0.1

    CSR_MIN_AMPLITUDE_VAR = 0.3
    CSR_MIN_WAXING_WANING = 0.7
    CSR_MIN_CYCLES_HIGH_CONF = 5

    PERIODIC_HIGH_REGULARITY = 0.7

    CSR_WINDOW_STEP_MINUTES = 2.0
    CSR_EPISODE_MERGE_GAP_SECONDS = 60.0
    CSR_MIN_ENVELOPE_RATIO = 2.0


class FlowLimitationConstants:
    """Constants for flow limitation classification (flow_limitation.py)."""

    CONFIDENCE_THRESHOLD = 0.6

    # Plateau thresholds are fractions of inspiration time (plateau_fraction),
    # not absolute seconds, so slow deep breaths no longer trivially clear them.
    FL_CLASS7_FLATNESS_MIN = 0.9
    FL_CLASS7_PLATEAU_FRAC_MIN = 0.5

    FL_CLASS6_PEAK_POSITION_MIN = 0.7
    FL_CLASS6_FLATNESS_MIN = 0.6
    FL_CLASS6_PLATEAU_FRAC_MIN = 0.25

    FL_CLASS5_FLATNESS_MIN = 0.7
    FL_CLASS5_PEAK_POSITION_MIN = 0.4
    FL_CLASS5_PEAK_POSITION_MAX = 0.6
    FL_CLASS5_PLATEAU_FRAC_MIN = 0.2

    FL_CLASS4_FLATNESS_MIN = 0.4
    FL_CLASS4_PEAK_POSITION_MAX = 0.3
    FL_CLASS4_PLATEAU_FRAC_MIN = 0.3

    FL_CLASS3_PEAK_COUNT_MIN = 3
    FL_CLASS3_FLATNESS_MIN = 0.3
    FL_CLASS3_PROMINENCE_MAX = 0.3

    FL_CLASS2_PEAK_COUNT = 2
    FL_CLASS2_PEAK_SPACING_MIN = 0.3

    FL_CLASS1_FLATNESS_MAX = 0.45
    FL_CLASS1_SYMMETRY_MAX = 0.3

    # Fallback flatness-triage cutoffs: when no shape rule matches, the breath is
    # binned by flatness alone (class 1 / 4 / 7) at FL_DEFAULT_CONFIDENCE.
    FL_FALLBACK_FLATNESS_CLASS1_MAX = 0.5
    FL_FALLBACK_FLATNESS_CLASS4_MAX = 0.7

    # Threshold-margin confidence: rule-matched confidence is
    # FL_CONFIDENCE_BASE + FL_CONFIDENCE_MARGIN_SCALE * mean(margins), where each
    # margin is the fractional distance of a value past its threshold, clipped to
    # [0, 1].  Base > FL_DEFAULT_CONFIDENCE guarantees rule-matched breaths always
    # score strictly above the fallback (the gate nightly metrics key on).
    FL_DEFAULT_CONFIDENCE = 0.5
    FL_CONFIDENCE_BASE = 0.6
    FL_CONFIDENCE_MARGIN_SCALE = 0.3


class PulseChangeConstants:
    """Constants for pulse change detection (pulse_detector.py)."""

    BPM_THRESHOLD = 5.0
    DURATION_THRESHOLD = 8.0


# ============================================================================
# Flow Limitation Classes
# ============================================================================


class FlowLimitationClassInfo(TypedDict):
    """Type definition for flow limitation class information."""

    name: str
    description: str
    visual_characteristics: str
    clinical_significance: str
    severity: str
    weight: float


FLOW_LIMITATION_CLASSES: dict[int, FlowLimitationClassInfo] = {
    1: {
        "name": "Sinusoidal",
        "description": "Normal, rounded inspiration with smooth sinusoidal curve",
        "visual_characteristics": "Smooth rounded peak, symmetric rise and fall",
        "clinical_significance": "Healthy unobstructed breathing pattern",
        "severity": "normal",
        "weight": 0.0,
    },
    2: {
        "name": "Double Peak",
        "description": "Two distinct peaks during inspiration phase",
        "visual_characteristics": "Two separate peaks with valley between, soft tissue vibration",
        "clinical_significance": "Mild flow limitation - upper airway reopening after initial collapse",
        "severity": "mild",
        "weight": 0.3,
    },
    3: {
        "name": "Flattened with Multiple Peaks",
        "description": "Multiple tiny peaks across flattened inspiratory curve",
        "visual_characteristics": "Many small peaks/oscillations, irregular amplitude",
        "clinical_significance": "Mild-moderate flow limitation - soft tissue vibration during inspiration",
        "severity": "mild-moderate",
        "weight": 0.4,
    },
    4: {
        "name": "Peak During Initial Phase",
        "description": "Early sharp peak followed by sustained plateau",
        "visual_characteristics": "Peak in first 30% of inspiration, then flat plateau",
        "clinical_significance": "Moderate flow limitation - initial opening followed by restricted flow",
        "severity": "moderate",
        "weight": 0.6,
    },
    5: {
        "name": "Peak at Midpoint",
        "description": "Single peak at midpoint with plateaus on both sides",
        "visual_characteristics": "Central peak (40-60% position), flat on both sides",
        "clinical_significance": "Moderate-severe flow limitation - intensive phasic muscle activity",
        "severity": "moderate-severe",
        "weight": 0.7,
    },
    6: {
        "name": "Peak During Late Phase",
        "description": "Initial plateau with late-phase peak",
        "visual_characteristics": "Flat early phase, peak in final 30% of inspiration",
        "clinical_significance": "Severe flow limitation - marked tracheal support during lung inflation",
        "severity": "severe",
        "weight": 0.9,
    },
    7: {
        "name": "Plateau Throughout",
        "description": "Nearly flat plateau throughout entire inspiration",
        "visual_characteristics": "Minimal amplitude variation, flat-top waveform throughout",
        "clinical_significance": "Severe flow limitation - collapsed noncompliant upper airway",
        "severity": "severe",
        "weight": 1.0,
    },
}


# ============================================================================
# File Format Constants
# ============================================================================

# OSCAR file format magic number
OSCAR_MAGIC_NUMBER = 0xC73216AB


# ============================================================================
# Respiratory Event Types
# ============================================================================

# Event type string constants (from RespiratoryEventType enum values)
EVENT_TYPE_OBSTRUCTIVE_APNEA = "OA"
EVENT_TYPE_CENTRAL_APNEA = "CA"
EVENT_TYPE_CLEAR_AIRWAY = "CAA"
EVENT_TYPE_MIXED_APNEA = "MA"
EVENT_TYPE_HYPOPNEA = "H"
EVENT_TYPE_RERA = "RE"
EVENT_TYPE_FLOW_LIMITATION = "FL"
EVENT_TYPE_UNCLASSIFIED_APNEA = "UA"

# Type alias for valid apnea event types (used by ApneaEvent model)
type ApneaEventType = Literal["OA", "CA", "MA", "UA"]

# Mapping from storage event types to ApneaEventType
APNEA_TYPE_MAP: dict[str, ApneaEventType] = {
    EVENT_TYPE_OBSTRUCTIVE_APNEA: "OA",
    EVENT_TYPE_CENTRAL_APNEA: "CA",
    EVENT_TYPE_CLEAR_AIRWAY: "CA",
    EVENT_TYPE_MIXED_APNEA: "MA",
    EVENT_TYPE_UNCLASSIFIED_APNEA: "UA",
}


def get_apnea_type(event_type: str) -> ApneaEventType | None:
    """Convert storage event type to ApneaEventType, or None if not an apnea."""
    return APNEA_TYPE_MAP.get(event_type)


def is_apnea_type(event_type: str) -> bool:
    """Check if event type is an apnea (not hypopnea/RERA)."""
    return event_type in APNEA_TYPE_MAP


EVENT_TYPE_ABBREVIATIONS: dict[str, str] = {
    EVENT_TYPE_OBSTRUCTIVE_APNEA: "OA",
    EVENT_TYPE_CENTRAL_APNEA: "CA",
    EVENT_TYPE_CLEAR_AIRWAY: "CA",
    EVENT_TYPE_MIXED_APNEA: "MA",
    EVENT_TYPE_HYPOPNEA: "H",
    EVENT_TYPE_RERA: "RE",
    EVENT_TYPE_UNCLASSIFIED_APNEA: "UA",
}


def abbreviate_event_type(event_type: str) -> str:
    """Convert event type constant to display abbreviation."""
    return EVENT_TYPE_ABBREVIATIONS.get(event_type, event_type)


# ============================================================================
# Default Settings
# ============================================================================

# Database stored in user's home directory
DEFAULT_DATABASE_PATH = str(Path.home() / ".snore" / "snore.db")

# Raw file backup directory
DEFAULT_RAW_BACKUP_DIR = Path.home() / ".snore" / "raw"

# Marker file written before a DB commit that will trigger a post-commit VACUUM.
# Persists across restarts: startup checks for this file and reschedules the
# VACUUM when a prior container restart interrupted it.  Content is the SQLite
# file path so startup can re-derive the target without environment variables.
# Removed by _vacuum_background after a successful VACUUM.
DEFAULT_VACUUM_PENDING_MARKER = Path.home() / ".snore" / "vacuum.pending"
# Marker file written by scripts/watchtower-pre-update.sh when a container
# update is deferred because the app is mid-operation.  Content is the UTC
# timestamp of the first deferral (written once; mtime is touched on every
# subsequent retry as the liveness signal).  Read by GET /api/v1/about, which
# treats the marker as fresh only when mtime is under 30 minutes old.  Removed
# unconditionally at app startup; after a same-image crash-restart watchtower
# recreates it within one poll cycle if a deploy is still pending.
DEFAULT_DEPLOY_DEFERRED_MARKER = Path.home() / ".snore" / "deploy-deferred.pending"

# Upload staging spool directory (durable across restarts)
DEFAULT_UPLOAD_SPOOL_DIR = Path.home() / ".snore" / "spool"

# Logging configuration
DEFAULT_LOG_DIR = Path.home() / ".snore" / "logs"
DEFAULT_LOG_FILE = "snore.log"
DEFAULT_LOG_BACKUP_COUNT = 5

# CLI display defaults
DEFAULT_LIST_SESSIONS_LIMIT = 20

# ============================================================================
# Parser Configuration
# ============================================================================

# Directory search depth for finding CPAP data roots
# Supports OSCAR structure: Profiles/user/device/Backup (5 levels deep)
# Also supports raw SD card structure which is typically shallower
PARSER_MAX_SEARCH_DEPTH = 5
