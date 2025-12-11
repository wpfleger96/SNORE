"""EDF format type definitions."""

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EDFHeader(BaseModel):
    """EDF file header information."""

    version: str = Field(description="EDF version")
    patient_info: str = Field(description="Patient identification")
    recording_info: str = Field(description="Recording identification")
    start_datetime: datetime = Field(description="Recording start time")
    num_data_records: int = Field(ge=0, description="Number of data records")
    record_duration: float = Field(ge=0, description="Record duration (seconds)")
    num_signals: int = Field(ge=0, description="Number of signals")
    is_edf_plus: bool = Field(default=False, description="EDF+ format flag")


class EDFSignalInfo(BaseModel):
    """Information about a single EDF signal/channel."""

    label: str = Field(description="Signal name")
    transducer: str = Field(description="Transducer type")
    physical_dimension: str = Field(description="Units (e.g., 'cmH2O', 'L/min')")
    physical_min: float = Field(description="Physical minimum value")
    physical_max: float = Field(description="Physical maximum value")
    digital_min: int = Field(description="Digital minimum value")
    digital_max: int = Field(description="Digital maximum value")
    prefiltering: str = Field(description="Prefiltering info")
    samples_per_record: int = Field(ge=0, description="Samples per data record")
    signal_index: int = Field(ge=0, description="Signal index in EDF file")

    gain: float = Field(default=0.0, description="Digital->physical gain")
    offset: float = Field(default=0.0, description="Digital->physical offset")

    @field_validator("gain", "offset", mode="before")
    @classmethod
    def compute_conversion_params(cls, v: float, info: Any) -> float:
        """Calculate gain and offset for digital->physical conversion."""
        values = info.data
        if "digital_min" in values and "digital_max" in values:
            digital_range = values["digital_max"] - values["digital_min"]
            physical_range = values["physical_max"] - values["physical_min"]

            if digital_range > 0:
                if info.field_name == "gain":
                    return float(physical_range / digital_range)
                elif info.field_name == "offset":
                    computed_gain = physical_range / digital_range
                    return float(
                        values["physical_min"] - (values["digital_min"] * computed_gain)
                    )
        return float(v) if v != 0.0 else (1.0 if info.field_name == "gain" else 0.0)

    def digital_to_physical(self, digital_value: int) -> float:
        """Convert a digital value to physical units."""
        return (digital_value * self.gain) + self.offset


class EDFAnnotation(BaseModel):
    """An EDF+ annotation (event marker with optional duration)."""

    onset_time: float = Field(description="Seconds from recording start")
    duration: float | None = Field(default=None, description="Duration (seconds)")
    annotations: list[str] = Field(description="Annotation strings")

    def to_datetime(self, recording_start: datetime) -> datetime:
        """Convert onset time to absolute datetime."""
        return recording_start + timedelta(seconds=self.onset_time)
