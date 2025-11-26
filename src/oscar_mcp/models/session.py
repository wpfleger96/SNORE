"""Pydantic models for Session data."""

from pydantic import BaseModel, Field


class SessionSummary(BaseModel):
    """Summary information about a therapy session."""

    id: int
    session_id: str = Field(description="Unique session identifier from OSCAR")
    machine_brand: str | None = None
    machine_model: str | None = None

    # Timing
    start_time: int = Field(
        description="Session start time (Unix timestamp in milliseconds)"
    )
    end_time: int = Field(
        description="Session end time (Unix timestamp in milliseconds)"
    )
    duration_hours: float = Field(description="Session duration in hours")

    # Key metrics
    ahi: float | None = Field(default=None, description="Apnea-Hypopnea Index")
    pressure_median: float | None = Field(
        default=None, description="Median pressure (cmH₂O)"
    )
    leak_median: float | None = Field(
        default=None, description="Median leak rate (L/min)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "session_id": "20240115_220015",
                "machine_brand": "ResMed",
                "machine_model": "AirSense 10 AutoSet",
                "start_time": 1705357215000,
                "end_time": 1705384815000,
                "duration_hours": 7.67,
                "ahi": 2.3,
                "pressure_median": 10.2,
                "leak_median": 8.5,
            }
        }


class SessionDetail(BaseModel):
    """Detailed information about a therapy session."""

    id: int
    session_id: str
    machine_brand: str | None = None
    machine_model: str | None = None

    # Timing
    start_time: int = Field(description="Unix timestamp in milliseconds")
    end_time: int = Field(description="Unix timestamp in milliseconds")
    duration_hours: float

    # Event counts
    obstructive_apneas: int = 0
    hypopneas: int = 0
    central_apneas: int = 0
    reras: int = 0

    # Indices
    ahi: float | None = None

    # Pressure statistics
    pressure_min: float | None = None
    pressure_max: float | None = None
    pressure_median: float | None = None
    pressure_95th: float | None = None

    # Leak statistics
    leak_median: float | None = None
    leak_95th: float | None = None
    leak_max: float | None = None

    # Respiratory statistics
    resp_rate_avg: float | None = None
    tidal_volume_avg: float | None = None
    minute_vent_avg: float | None = None

    # SpO2 statistics (if available)
    spo2_avg: float | None = None
    spo2_min: float | None = None
    pulse_avg: float | None = None

    # Settings
    settings: dict[str, str] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "session_id": "20240115_220015",
                "machine_brand": "ResMed",
                "machine_model": "AirSense 10 AutoSet",
                "start_time": 1705357215000,
                "end_time": 1705384815000,
                "duration_hours": 7.67,
                "obstructive_apneas": 12,
                "hypopneas": 6,
                "central_apneas": 0,
                "reras": 2,
                "ahi": 2.3,
                "pressure_min": 8.0,
                "pressure_max": 14.2,
                "pressure_median": 10.2,
                "pressure_95th": 12.8,
                "leak_median": 8.5,
                "leak_95th": 18.2,
                "leak_max": 24.0,
                "resp_rate_avg": 14.5,
                "tidal_volume_avg": 520,
                "minute_vent_avg": 7.5,
                "settings": {
                    "Mode": "AutoSet",
                    "MinPressure": "8.0",
                    "MaxPressure": "15.0",
                    "EPR": "3",
                },
            }
        }
