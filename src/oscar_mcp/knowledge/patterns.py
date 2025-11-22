"""
Medical Pattern Definitions

Flow limitation classes, respiratory events, and pattern characteristics
extracted from OSCAR Guide and clinical guidelines.

Note: Numeric threshold values (e.g., flow_reduction_percent) reference
constants.py as the single source of truth for runtime configuration.
"""

from typing import Dict, List, Tuple, TypedDict
from oscar_mcp.constants import EventDetectionConstants as EDC


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


class RespiratoryEventInfo(TypedDict, total=False):
    """Type definition for respiratory event information."""

    name: str
    abbreviation: str
    criteria: str
    characteristics: str
    clinical_significance: str
    severity: str
    min_duration_seconds: int
    flow_reduction_percent: int
    flow_reduction_percent_min: int
    flow_reduction_percent_max: int


class ComplexPatternInfo(TypedDict, total=False):
    """Type definition for complex pattern information."""

    name: str
    abbreviation: str
    description: str
    characteristics: str
    clinical_significance: str
    severity: str
    cycle_length_range: Tuple[int, int]
    reference_images: List[str]


# Flow Limitation Classes (1-7)
# Based on OSCAR Guide classification system
FLOW_LIMITATION_CLASSES: Dict[int, FlowLimitationClassInfo] = {
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


# Respiratory Event Types
RESPIRATORY_EVENTS: Dict[str, RespiratoryEventInfo] = {
    "obstructive_apnea": {
        "name": "Obstructive Apnea (OA)",
        "abbreviation": "OA",
        "criteria": f"≥{int(EDC.APNEA_FLOW_REDUCTION_THRESHOLD * 100)}% reduction in airflow for ≥{int(EDC.MIN_EVENT_DURATION)} seconds with continued respiratory effort",
        "characteristics": "Flow cessation with persistent chest/abdominal movement",
        "clinical_significance": "Complete upper airway obstruction despite breathing effort",
        "severity": "severe",
        "min_duration_seconds": int(EDC.MIN_EVENT_DURATION),
        "flow_reduction_percent": int(EDC.APNEA_FLOW_REDUCTION_THRESHOLD * 100),
    },
    "central_apnea": {
        "name": "Central Apnea (CA)",
        "abbreviation": "CA",
        "criteria": f"≥{int(EDC.APNEA_FLOW_REDUCTION_THRESHOLD * 100)}% reduction in airflow for ≥{int(EDC.MIN_EVENT_DURATION)} seconds without respiratory effort",
        "characteristics": "Both flow and effort signals cease simultaneously",
        "clinical_significance": "Loss of central drive to breathe - neurological origin",
        "severity": "severe",
        "min_duration_seconds": int(EDC.MIN_EVENT_DURATION),
        "flow_reduction_percent": int(EDC.APNEA_FLOW_REDUCTION_THRESHOLD * 100),
    },
    "mixed_apnea": {
        "name": "Mixed Apnea",
        "abbreviation": "MA",
        "criteria": f"Begins as central apnea, transitions to obstructive pattern (≥{int(EDC.APNEA_FLOW_REDUCTION_THRESHOLD * 100)}% flow reduction)",
        "characteristics": "Initial absence of effort, then effort resumes without airflow",
        "clinical_significance": "Combined central and obstructive pathology",
        "severity": "severe",
        "min_duration_seconds": int(EDC.MIN_EVENT_DURATION),
        "flow_reduction_percent": int(EDC.APNEA_FLOW_REDUCTION_THRESHOLD * 100),
    },
    "hypopnea": {
        "name": "Hypopnea",
        "abbreviation": "H",
        "criteria": f"{int(EDC.HYPOPNEA_MIN_REDUCTION * 100)}-{int(EDC.HYPOPNEA_MAX_REDUCTION * 100)}% reduction in airflow for ≥{int(EDC.MIN_EVENT_DURATION)} seconds",
        "characteristics": "Partial airflow reduction, may include arousal or desaturation",
        "clinical_significance": "Partial airway obstruction or reduced respiratory effort",
        "severity": "moderate",
        "min_duration_seconds": int(EDC.MIN_EVENT_DURATION),
        "flow_reduction_percent_min": int(EDC.HYPOPNEA_MIN_REDUCTION * 100),
        "flow_reduction_percent_max": int(EDC.HYPOPNEA_MAX_REDUCTION * 100),
    },
    "rera": {
        "name": "Respiratory Effort Related Arousal (RERA)",
        "abbreviation": "RERA",
        "criteria": "Flow limitation with increased effort terminated by arousal, not meeting apnea/hypopnea criteria",
        "characteristics": "Flattened flow waveform, increased effort, arousal termination",
        "clinical_significance": "Subtle flow limitation causing sleep fragmentation",
        "severity": "mild",
        "min_duration_seconds": 10,
    },
}


# Complex Breathing Patterns
COMPLEX_PATTERNS: Dict[str, ComplexPatternInfo] = {
    "cheyne_stokes_respiration": {
        "name": "Cheyne-Stokes Respiration (CSR)",
        "abbreviation": "CSR",
        "description": "Cyclical waxing and waning of tidal volume with central apneas",
        "characteristics": "45-90 second cycles, crescendo-decrescendo pattern",
        "clinical_significance": "Heart failure, stroke, or high-altitude exposure indicator",
        "cycle_length_range": (45, 90),
        "severity": "moderate-severe",
    },
    "periodic_breathing": {
        "name": "Periodic Breathing",
        "abbreviation": "PB",
        "description": "Regular oscillations in breathing amplitude without clear central apneas",
        "characteristics": "30-120 second cycles, waxing/waning tidal volume",
        "clinical_significance": "Ventilatory instability, may indicate cardiac or neurological issues",
        "cycle_length_range": (30, 120),
        "severity": "mild-moderate",
        "reference_images": [
            "data/guidelines/images/patterns/OSCAR_periodic_breathing_chart_example.png",
            "data/guidelines/images/patterns/OSCAR_periodic_breathing_chart_example_resmed.png",
        ],
    },
    "positional_apnea": {
        "name": "Positional Apnea",
        "description": "Events clustered during specific sleep positions",
        "characteristics": "Temporal clustering of events, position-dependent severity",
        "clinical_significance": "Anatomical obstruction worse in certain positions (typically supine)",
        "severity": "variable",
        "reference_images": [
            "data/guidelines/images/patterns/OSCAR_positional_apnea_chart_example.png",
            "data/guidelines/images/patterns/OSCAR_positional_apnea_chart_example_detailed.png",
        ],
    },
    "palatal_prolapse": {
        "name": "Palatal Prolapse",
        "description": "Sudden expiratory flow cutoff due to soft palate collapse",
        "characteristics": "Abrupt cessation during expiration, sharp flow drop",
        "clinical_significance": "Soft palate instability during expiration",
        "severity": "mild",
        "reference_images": [
            "data/guidelines/images/patterns/OSCAR_palatal_prolapse_graph_example.png",
        ],
    },
}


# Static Pattern Relationships
# Describes how patterns relate to each other causally or sequentially
PATTERN_RELATIONSHIPS = [
    {
        "from": "flow_limitation_class_2",
        "to": "hypopnea",
        "type": "can_progress_to",
        "strength": 0.6,
        "description": "Mild flow limitation can progress to hypopnea events",
    },
    {
        "from": "flow_limitation_class_7",
        "to": "obstructive_apnea",
        "type": "can_progress_to",
        "strength": 0.9,
        "description": "Severe flow limitation strongly associated with obstructive apneas",
    },
    {
        "from": "obstructive_apnea",
        "to": "oxygen_desaturation",
        "type": "causes",
        "strength": 0.9,
        "description": "Obstructive apneas typically cause oxygen desaturation",
    },
    {
        "from": "hypopnea",
        "to": "arousal",
        "type": "causes",
        "strength": 0.8,
        "description": "Hypopneas often trigger arousal from sleep",
    },
    {
        "from": "rera",
        "to": "arousal",
        "type": "causes",
        "strength": 0.95,
        "description": "RERAs by definition terminate in arousal",
    },
]


# Arousal and Sleep Fragmentation Patterns
# Additional patterns related to sleep quality and arousals
AROUSAL_PATTERNS = {
    "arousal_recovery_breathing": {
        "name": "Arousal Recovery Breathing",
        "description": "Breathing pattern changes following arousal from sleep",
        "characteristics": "Increased respiratory rate and tidal volume immediately post-arousal",
        "clinical_significance": "Normal physiological response to arousal, indicates sleep fragmentation",
        "severity": "marker",  # Not a pathology, but a marker of sleep disruption
        "reference_images": [
            "data/guidelines/images/patterns/OSCAR_arousal_recovery_breathing_chart_example.png",
            "data/guidelines/images/patterns/OSCAR_arousal_recovery_breathing_chart_example_2.png",
        ],
    },
    "periodic_leg_movement": {
        "name": "Periodic Leg Movement (PLM)",
        "abbreviation": "PLM",
        "description": "Repetitive stereotyped leg movements during sleep",
        "characteristics": "Periodic limb movements, often clustered in NREM sleep",
        "clinical_significance": "May cause sleep fragmentation and arousals, associated with restless leg syndrome (RLS)",
        "severity": "mild-moderate",
        "cycle_length_range": (20, 40),  # Typical PLM periodicity in seconds
        "reference_images": [
            "data/guidelines/images/patterns/OSCAR_periodic_leg_movement_chart_example.png",
        ],
    },
}
