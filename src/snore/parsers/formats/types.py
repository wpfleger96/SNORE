"""EDF format type definitions."""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class EDFHeader:
    """EDF file header information."""

    version: str
    patient_info: str
    recording_info: str
    start_datetime: datetime
    num_data_records: int
    record_duration: float  # seconds
    num_signals: int
    is_edf_plus: bool = False


@dataclass
class EDFSignalInfo:
    """Information about a single EDF signal/channel."""

    label: str  # Signal name
    transducer: str  # Transducer type
    physical_dimension: str  # Units (e.g., "cmH2O", "L/min")
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    prefiltering: str
    samples_per_record: int
    signal_index: int  # Index in EDF file

    # Computed values
    gain: float = 0.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        """Calculate gain and offset for digital->physical conversion."""
        digital_range = self.digital_max - self.digital_min
        physical_range = self.physical_max - self.physical_min

        if digital_range > 0:
            self.gain = physical_range / digital_range
            self.offset = self.physical_min - (self.digital_min * self.gain)
        else:
            self.gain = 1.0
            self.offset = 0.0

    def digital_to_physical(self, digital_value: int) -> float:
        """Convert a digital value to physical units."""
        return (digital_value * self.gain) + self.offset


@dataclass
class EDFAnnotation:
    """An EDF+ annotation (event marker with optional duration)."""

    onset_time: float  # Seconds from recording start
    duration: float | None  # Duration in seconds (if specified)
    annotations: list[str]  # List of annotation strings

    def to_datetime(self, recording_start: datetime) -> datetime:
        """Convert onset time to absolute datetime."""
        return recording_start + timedelta(seconds=self.onset_time)
