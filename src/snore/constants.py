"""
Constants and mappings for OSCAR CPAP data analysis.

Based on OSCAR's schema.h and machine_common.h definitions.
"""

from enum import Enum
from pathlib import Path
from typing import TypedDict

# ============================================================================
# Machine Types
# ============================================================================


class MachineType(str, Enum):
    """CPAP device types supported by OSCAR."""

    CPAP = "CPAP"
    BIPAP = "BiPAP"
    AUTO_CPAP = "AutoCPAP"
    VENTILATOR = "Ventilator"
    OXIMETER = "Oximeter"
    POSITION = "Position"
    SLEEP_STAGE = "SleepStage"


# ============================================================================
# Channel Types
# ============================================================================


class ChannelType(str, Enum):
    """Types of data channels in OSCAR."""

    DATA = "DATA"  # Single values (height, weight)
    SETTING = "SETTING"  # Device settings (pressure, EPR level)
    FLAG = "FLAG"  # Event markers (apnea, hypopnea)
    MINOR_FLAG = "MINOR_FLAG"  # Less significant events
    SPAN = "SPAN"  # Time-span events (CSR periods)
    WAVEFORM = "WAVEFORM"  # High-frequency time-series data
    UNKNOWN = "UNKNOWN"  # Unclassified


class CalculationType(str, Enum):
    """Statistical calculation types available for channels."""

    MIN = "min"
    MAX = "max"
    AVG = "avg"
    WAVG = "wavg"  # Weighted average
    SUM = "sum"
    MEDIAN = "median"
    P90 = "p90"  # 90th percentile
    P95 = "p95"  # 95th percentile
    CPH = "cph"  # Count per hour
    SPH = "sph"  # Sum per hour


# ============================================================================
# Channel IDs
# ============================================================================

# CPAP Pressure Channels
CPAP_PRESSURE = 0x1000
CPAP_IPAP = 0x1001
CPAP_EPAP = 0x1002
CPAP_PS = 0x1003  # Pressure support

# CPAP Event Flags
CPAP_OBSTRUCTIVE = 0x1100
CPAP_HYPOPNEA = 0x1101
CPAP_CLEAR_AIRWAY = 0x1102
CPAP_APNEA = 0x1103  # Generic apnea
CPAP_RERA = 0x1104  # Respiratory effort related arousal
CPAP_VIBRATORY_SNORE = 0x1105
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
CPAP_TARGET_VENT = 0x1206
CPAP_FLG = 0x1207  # Flow limitation graph
CPAP_IE = 0x1208  # Inspiratory/expiratory ratio

# CPAP Statistics
CPAP_AHI = 0x1300
CPAP_RDI = 0x1301  # Respiratory disturbance index
CPAP_PTB = 0x1302  # Periodic breathing percentage

# CPAP Settings
CPAP_MODE = 0x1400
CPAP_PRESSURE_MIN = 0x1401
CPAP_PRESSURE_MAX = 0x1402
CPAP_EPR_LEVEL = 0x1403  # Expiratory pressure relief
CPAP_RAMP_TIME = 0x1404
CPAP_RAMP_PRESSURE = 0x1405

# Oximetry Channels
OXI_SPO2 = 0x2000
OXI_PULSE = 0x2001
OXI_PLETHY = 0x2002  # Plethysmograph waveform
OXI_SPO2DROP = 0x2003  # Desaturation events

# Position Sensor
POS_POSITION = 0x3000

# Sleep Stages
SLEEP_STAGE = 0x4000


# ============================================================================
# Channel Definitions
# ============================================================================


class ChannelDefinition:
    """Definition of a data channel."""

    def __init__(
        self,
        channel_id: int,
        code: str,
        name: str,
        description: str,
        channel_type: ChannelType,
        unit: str = "",
        default_color: str = "#ffffff",
        calculations: list[CalculationType] | None = None,
    ):
        self.channel_id = channel_id
        self.code = code
        self.name = name
        self.description = description
        self.channel_type = channel_type
        self.unit = unit
        self.default_color = default_color
        self.calculations = calculations or []


# Map of channel IDs to their definitions
CHANNEL_DEFINITIONS: dict[int, ChannelDefinition] = {
    # Pressure channels
    CPAP_PRESSURE: ChannelDefinition(
        CPAP_PRESSURE,
        "CPAP_Pressure",
        "Pressure",
        "Therapy pressure",
        ChannelType.WAVEFORM,
        "cmH₂O",
        "#00ff00",
        [
            CalculationType.MIN,
            CalculationType.MAX,
            CalculationType.MEDIAN,
            CalculationType.P95,
        ],
    ),
    CPAP_IPAP: ChannelDefinition(
        CPAP_IPAP,
        "CPAP_IPAP",
        "IPAP",
        "Inspiratory positive airway pressure",
        ChannelType.WAVEFORM,
        "cmH₂O",
        "#00ff00",
        [
            CalculationType.MIN,
            CalculationType.MAX,
            CalculationType.MEDIAN,
            CalculationType.P95,
        ],
    ),
    CPAP_EPAP: ChannelDefinition(
        CPAP_EPAP,
        "CPAP_EPAP",
        "EPAP",
        "Expiratory positive airway pressure",
        ChannelType.WAVEFORM,
        "cmH₂O",
        "#0080ff",
        [
            CalculationType.MIN,
            CalculationType.MAX,
            CalculationType.MEDIAN,
            CalculationType.P95,
        ],
    ),
    # Event flags
    CPAP_OBSTRUCTIVE: ChannelDefinition(
        CPAP_OBSTRUCTIVE,
        "CPAP_Obstructive",
        "Obstructive Apnea",
        "Obstructive apnea events",
        ChannelType.FLAG,
        "events",
        "#ff0000",
        [CalculationType.SUM, CalculationType.CPH],
    ),
    CPAP_HYPOPNEA: ChannelDefinition(
        CPAP_HYPOPNEA,
        "CPAP_Hypopnea",
        "Hypopnea",
        "Hypopnea events (reduced airflow)",
        ChannelType.FLAG,
        "events",
        "#ff8000",
        [CalculationType.SUM, CalculationType.CPH],
    ),
    CPAP_CLEAR_AIRWAY: ChannelDefinition(
        CPAP_CLEAR_AIRWAY,
        "CPAP_ClearAirway",
        "Clear Airway Apnea",
        "Central/clear airway apnea events",
        ChannelType.FLAG,
        "events",
        "#ff00ff",
        [CalculationType.SUM, CalculationType.CPH],
    ),
    CPAP_APNEA: ChannelDefinition(
        CPAP_APNEA,
        "CPAP_Apnea",
        "Apnea",
        "Generic apnea events",
        ChannelType.FLAG,
        "events",
        "#ff0000",
        [CalculationType.SUM, CalculationType.CPH],
    ),
    CPAP_RERA: ChannelDefinition(
        CPAP_RERA,
        "CPAP_RERA",
        "RERA",
        "Respiratory effort related arousals",
        ChannelType.FLAG,
        "events",
        "#ffff00",
        [CalculationType.SUM, CalculationType.CPH],
    ),
    CPAP_FLOW_LIMIT: ChannelDefinition(
        CPAP_FLOW_LIMIT,
        "CPAP_FlowLimit",
        "Flow Limitation",
        "Flow limitation events",
        ChannelType.FLAG,
        "events",
        "#8080ff",
        [CalculationType.SUM, CalculationType.CPH],
    ),
    # Waveforms
    CPAP_FLOW_RATE: ChannelDefinition(
        CPAP_FLOW_RATE,
        "CPAP_FlowRate",
        "Flow Rate",
        "Respiratory flow rate",
        ChannelType.WAVEFORM,
        "L/min",
        "#00ffff",
        [CalculationType.MIN, CalculationType.MAX, CalculationType.AVG],
    ),
    CPAP_LEAK: ChannelDefinition(
        CPAP_LEAK,
        "CPAP_Leak",
        "Leak Rate",
        "Mask leak rate",
        ChannelType.WAVEFORM,
        "L/min",
        "#ff0000",
        [
            CalculationType.MIN,
            CalculationType.MAX,
            CalculationType.MEDIAN,
            CalculationType.P95,
        ],
    ),
    CPAP_RESPRATE: ChannelDefinition(
        CPAP_RESPRATE,
        "CPAP_RespRate",
        "Respiratory Rate",
        "Breaths per minute",
        ChannelType.WAVEFORM,
        "bpm",
        "#ffffff",
        [CalculationType.MIN, CalculationType.MAX, CalculationType.AVG],
    ),
    CPAP_TIDAL_VOLUME: ChannelDefinition(
        CPAP_TIDAL_VOLUME,
        "CPAP_TidalVolume",
        "Tidal Volume",
        "Volume of air per breath",
        ChannelType.WAVEFORM,
        "mL",
        "#00ff00",
        [CalculationType.MIN, CalculationType.MAX, CalculationType.AVG],
    ),
    CPAP_MINUTE_VENT: ChannelDefinition(
        CPAP_MINUTE_VENT,
        "CPAP_MinuteVent",
        "Minute Ventilation",
        "Volume of air per minute",
        ChannelType.WAVEFORM,
        "L/min",
        "#00ff80",
        [CalculationType.MIN, CalculationType.MAX, CalculationType.AVG],
    ),
    # Statistics
    CPAP_AHI: ChannelDefinition(
        CPAP_AHI,
        "CPAP_AHI",
        "AHI",
        "Apnea-Hypopnea Index (events per hour)",
        ChannelType.DATA,
        "events/hr",
        "#ff0000",
        [CalculationType.AVG],
    ),
    CPAP_RDI: ChannelDefinition(
        CPAP_RDI,
        "CPAP_RDI",
        "RDI",
        "Respiratory Disturbance Index",
        ChannelType.DATA,
        "events/hr",
        "#ff8000",
        [CalculationType.AVG],
    ),
    # Oximetry
    OXI_SPO2: ChannelDefinition(
        OXI_SPO2,
        "OXI_SPO2",
        "SpO₂",
        "Blood oxygen saturation",
        ChannelType.WAVEFORM,
        "%",
        "#0080ff",
        [
            CalculationType.MIN,
            CalculationType.MAX,
            CalculationType.AVG,
            CalculationType.MEDIAN,
        ],
    ),
    OXI_PULSE: ChannelDefinition(
        OXI_PULSE,
        "OXI_Pulse",
        "Pulse Rate",
        "Heart rate",
        ChannelType.WAVEFORM,
        "bpm",
        "#ff0000",
        [CalculationType.MIN, CalculationType.MAX, CalculationType.AVG],
    ),
}


# ============================================================================
# Units and Conversions
# ============================================================================

# Map of unit names to their display strings
UNITS = {
    "cmH2O": "cmH₂O",
    "L/min": "L/min",
    "mL": "mL",
    "bpm": "bpm",
    "events": "events",
    "events/hr": "events/hr",
    "%": "%",
    "hours": "hours",
    "minutes": "minutes",
    "seconds": "seconds",
}


# ============================================================================
# Clinical Thresholds
# ============================================================================

# AHI severity thresholds
AHI_SEVERITY = {
    "normal": (0, 5),
    "mild": (5, 15),
    "moderate": (15, 30),
    "severe": (30, float("inf")),
}

# SpO2 thresholds
SPO2_NORMAL_MIN = 95  # Below this is considered desaturation
SPO2_CRITICAL_MIN = 88  # Critical desaturation threshold

# Leak thresholds (L/min)
LEAK_ACCEPTABLE_MAX = 24  # Maximum acceptable leak rate
LEAK_LARGE_THRESHOLD = 30  # Large leak threshold

# Compliance thresholds
COMPLIANCE_MIN_HOURS = 4  # Minimum hours per night for compliance
COMPLIANCE_MIN_DAYS = 21  # Minimum days per month (70% of 30)


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
    Constants for respiratory event detection (event_detector.py).

    ⚠️ IMPORTANT: This is the SINGLE SOURCE OF TRUTH for event detection thresholds.
    The patterns.py file references these values for consistency.
    When modifying detection behavior, change values HERE, not in patterns.py.

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


class FlowLimitationConstants:
    """Constants for flow limitation classification (flow_limitation.py)."""

    CONFIDENCE_THRESHOLD = 0.6

    FL_CLASS7_FLATNESS_MIN = 0.9
    FL_CLASS7_PLATEAU_MIN = 0.8

    FL_CLASS6_PEAK_POSITION_MIN = 0.7
    FL_CLASS6_FLATNESS_MIN = 0.6
    FL_CLASS6_PLATEAU_MIN = 0.4

    FL_CLASS5_FLATNESS_MIN = 0.7
    FL_CLASS5_PEAK_POSITION_MIN = 0.4
    FL_CLASS5_PEAK_POSITION_MAX = 0.6
    FL_CLASS5_PLATEAU_MIN = 0.3

    FL_CLASS4_FLATNESS_MIN = 0.4
    FL_CLASS4_PEAK_POSITION_MAX = 0.3
    FL_CLASS4_PLATEAU_MIN = 0.5

    FL_CLASS3_PEAK_COUNT_MIN = 3
    FL_CLASS3_FLATNESS_MIN = 0.3
    FL_CLASS3_PROMINENCE_MAX = 0.3

    FL_CLASS2_PEAK_COUNT = 2
    FL_CLASS2_PEAK_SPACING_MIN = 0.3

    FL_CLASS1_FLATNESS_MAX = 0.3
    FL_CLASS1_SYMMETRY_MAX = 0.3
    FL_CLASS1_KURTOSIS_MIN = 2.0

    FL_DEFAULT_CONFIDENCE = 0.5
    FL_HIGH_FEATURE_COUNT = 3
    FL_MEDIUM_FEATURE_COUNT = 2
    FL_HIGH_CONFIDENCE = 0.9
    FL_MEDIUM_CONFIDENCE = 0.7
    FL_LOW_CONFIDENCE = 0.6

    FL_VERY_HIGH_FLATNESS = 0.95
    FL_HIGH_PEAK_SPACING = 0.4
    FL_CONFIDENCE_BONUS = 0.05


class AnalysisEngineConstants:
    """Constants for analysis engine coordination (programmatic_engine.py)."""

    MIN_BREATH_DURATION = 1.0
    MIN_EVENT_DURATION = 10.0
    CONFIDENCE_THRESHOLD = 0.6

    DEFAULT_SAMPLE_RATE = 25.0

    MIN_INSPIRATORY_SAMPLES = 10

    FLI_SEVERITY_MINIMAL = 0.2
    FLI_SEVERITY_MILD = 0.4
    FLI_SEVERITY_MODERATE = 0.6

    CSR_MIN_CONFIDENCE = 0.6
    PERIODIC_MIN_CONFIDENCE = 0.6
    POSITIONAL_MIN_CONFIDENCE = 0.6


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
    reference_image: str
    reference_section: str


FLOW_LIMITATION_CLASSES: dict[int, FlowLimitationClassInfo] = {
    1: {
        "name": "Sinusoidal",
        "description": "Normal, rounded inspiration with smooth sinusoidal curve",
        "visual_characteristics": "Smooth rounded peak, symmetric rise and fall",
        "clinical_significance": "Healthy unobstructed breathing pattern",
        "severity": "normal",
        "weight": 0.0,
        "reference_image": "data/guidelines/images/flow_limitation/OSCAR_flow_limitation_classes.png",
        "reference_section": "Class 1",
    },
    2: {
        "name": "Double Peak",
        "description": "Two distinct peaks during inspiration phase",
        "visual_characteristics": "Two separate peaks with valley between, soft tissue vibration",
        "clinical_significance": "Mild flow limitation - upper airway reopening after initial collapse",
        "severity": "mild",
        "weight": 0.3,
        "reference_image": "data/guidelines/images/flow_limitation/OSCAR_flow_limitation_classes.png",
        "reference_section": "Class 2",
    },
    3: {
        "name": "Flattened with Multiple Peaks",
        "description": "Multiple tiny peaks across flattened inspiratory curve",
        "visual_characteristics": "Many small peaks/oscillations, irregular amplitude",
        "clinical_significance": "Mild-moderate flow limitation - soft tissue vibration during inspiration",
        "severity": "mild-moderate",
        "weight": 0.4,
        "reference_image": "data/guidelines/images/flow_limitation/OSCAR_flow_limitation_classes.png",
        "reference_section": "Class 3",
    },
    4: {
        "name": "Peak During Initial Phase",
        "description": "Early sharp peak followed by sustained plateau",
        "visual_characteristics": "Peak in first 30% of inspiration, then flat plateau",
        "clinical_significance": "Moderate flow limitation - initial opening followed by restricted flow",
        "severity": "moderate",
        "weight": 0.6,
        "reference_image": "data/guidelines/images/flow_limitation/OSCAR_flow_limitation_classes.png",
        "reference_section": "Class 4",
    },
    5: {
        "name": "Peak at Midpoint",
        "description": "Single peak at midpoint with plateaus on both sides",
        "visual_characteristics": "Central peak (40-60% position), flat on both sides",
        "clinical_significance": "Moderate-severe flow limitation - intensive phasic muscle activity",
        "severity": "moderate-severe",
        "weight": 0.7,
        "reference_image": "data/guidelines/images/flow_limitation/OSCAR_flow_limitation_classes.png",
        "reference_section": "Class 5",
    },
    6: {
        "name": "Peak During Late Phase",
        "description": "Initial plateau with late-phase peak",
        "visual_characteristics": "Flat early phase, peak in final 30% of inspiration",
        "clinical_significance": "Severe flow limitation - marked tracheal support during lung inflation",
        "severity": "severe",
        "weight": 0.9,
        "reference_image": "data/guidelines/images/flow_limitation/OSCAR_flow_limitation_classes.png",
        "reference_section": "Class 6",
    },
    7: {
        "name": "Plateau Throughout",
        "description": "Nearly flat plateau throughout entire inspiration",
        "visual_characteristics": "Minimal amplitude variation, flat-top waveform throughout",
        "clinical_significance": "Severe flow limitation - collapsed noncompliant upper airway",
        "severity": "severe",
        "weight": 1.0,
        "reference_image": "data/guidelines/images/flow_limitation/OSCAR_flow_limitation_classes.png",
        "reference_section": "Class 7",
    },
}


# ============================================================================
# File Format Constants
# ============================================================================

# OSCAR file format magic number
OSCAR_MAGIC_NUMBER = 0xC73216AB

# File extensions
SUMMARY_FILE_EXT = ".000"  # Summary data file
EVENT_FILE_EXT = ".001"  # Event/waveform data file


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

# Event type display names
EVENT_TYPE_NAMES = {
    EVENT_TYPE_OBSTRUCTIVE_APNEA: "Obstructive Apnea",
    EVENT_TYPE_CENTRAL_APNEA: "Central Apnea",
    EVENT_TYPE_CLEAR_AIRWAY: "Clear Airway",
    EVENT_TYPE_MIXED_APNEA: "Mixed Apnea",
    EVENT_TYPE_HYPOPNEA: "Hypopnea",
    EVENT_TYPE_RERA: "RERA",
    EVENT_TYPE_FLOW_LIMITATION: "Flow Limitation",
    EVENT_TYPE_UNCLASSIFIED_APNEA: "Unclassified Apnea",
}

# Event types that count toward AHI
AHI_EVENT_TYPES = {
    EVENT_TYPE_OBSTRUCTIVE_APNEA,
    EVENT_TYPE_CENTRAL_APNEA,
    EVENT_TYPE_CLEAR_AIRWAY,
    EVENT_TYPE_MIXED_APNEA,
    EVENT_TYPE_HYPOPNEA,
    EVENT_TYPE_UNCLASSIFIED_APNEA,
}

# Event types that count toward RDI (same as AHI without RERA detection)
RDI_EVENT_TYPES = AHI_EVENT_TYPES

# ============================================================================
# Default Settings
# ============================================================================

# Database stored in user's home directory
DEFAULT_DATABASE_PATH = str(Path.home() / ".snore" / "snore.db")
DEFAULT_PROFILE_NAME = "Default"

# Logging configuration
DEFAULT_LOG_DIR = Path.home() / ".snore" / "logs"
DEFAULT_LOG_FILE = "snore.log"
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_LOG_BACKUP_COUNT = 5

# CLI display defaults
DEFAULT_LIST_SESSIONS_LIMIT = 20

# Knowledge base image paths (relative to project root)
KNOWLEDGE_IMAGE_BASE = Path("data/guidelines/images")
IMAGE_DIRS = {
    "flow_limitation": KNOWLEDGE_IMAGE_BASE / "flow_limitation",
    "events": KNOWLEDGE_IMAGE_BASE / "events",
    "patterns": KNOWLEDGE_IMAGE_BASE / "patterns",
    "charts": KNOWLEDGE_IMAGE_BASE / "charts",
    "ui": KNOWLEDGE_IMAGE_BASE / "ui",
}

# Time calculations
SECONDS_PER_HOUR = 3600
MILLISECONDS_PER_SECOND = 1000

# ============================================================================
# Parser Configuration
# ============================================================================

# Directory search depth for finding CPAP data roots
# Supports OSCAR structure: Profiles/user/device/Backup (5 levels deep)
# Also supports raw SD card structure which is typically shallower
PARSER_MAX_SEARCH_DEPTH = 5
