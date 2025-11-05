"""
Constants and mappings for OSCAR CPAP data analysis.

Based on OSCAR's schema.h and machine_common.h definitions.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List

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
        calculations: List[CalculationType] = None,
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
CHANNEL_DEFINITIONS: Dict[int, ChannelDefinition] = {
    # Pressure channels
    CPAP_PRESSURE: ChannelDefinition(
        CPAP_PRESSURE,
        "CPAP_Pressure",
        "Pressure",
        "Therapy pressure",
        ChannelType.WAVEFORM,
        "cmH₂O",
        "#00ff00",
        [CalculationType.MIN, CalculationType.MAX, CalculationType.MEDIAN, CalculationType.P95],
    ),
    CPAP_IPAP: ChannelDefinition(
        CPAP_IPAP,
        "CPAP_IPAP",
        "IPAP",
        "Inspiratory positive airway pressure",
        ChannelType.WAVEFORM,
        "cmH₂O",
        "#00ff00",
        [CalculationType.MIN, CalculationType.MAX, CalculationType.MEDIAN, CalculationType.P95],
    ),
    CPAP_EPAP: ChannelDefinition(
        CPAP_EPAP,
        "CPAP_EPAP",
        "EPAP",
        "Expiratory positive airway pressure",
        ChannelType.WAVEFORM,
        "cmH₂O",
        "#0080ff",
        [CalculationType.MIN, CalculationType.MAX, CalculationType.MEDIAN, CalculationType.P95],
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
        [CalculationType.MIN, CalculationType.MAX, CalculationType.MEDIAN, CalculationType.P95],
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
        "Respiratory Disturbance Index (includes RERA)",
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
        [CalculationType.MIN, CalculationType.MAX, CalculationType.AVG, CalculationType.MEDIAN],
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
# File Format Constants
# ============================================================================

# OSCAR file format magic number
OSCAR_MAGIC_NUMBER = 0xC73216AB

# File extensions
SUMMARY_FILE_EXT = ".000"  # Summary data file
EVENT_FILE_EXT = ".001"  # Event/waveform data file


# ============================================================================
# Default Settings
# ============================================================================

# Database stored in user's home directory
DEFAULT_DATABASE_PATH = str(Path.home() / ".oscar-mcp" / "oscar_mcp.db")
DEFAULT_PROFILE_NAME = "Default"

# Time calculations
SECONDS_PER_HOUR = 3600
MILLISECONDS_PER_SECOND = 1000
