"""Pydantic models for Day/Daily report data."""

from datetime import date

from pydantic import BaseModel, Field


class DaySummary(BaseModel):
    """Summary of therapy data for a single day."""

    id: int
    date: date
    total_therapy_hours: float | None = Field(
        default=None, description="Total hours of therapy"
    )
    ahi: float | None = Field(default=None, description="Apnea-Hypopnea Index")
    compliance: bool = Field(
        default=False, description="Met minimum 4-hour usage requirement"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "date": "2024-01-15",
                "total_therapy_hours": 7.67,
                "ahi": 2.3,
                "compliance": True,
            }
        }


class DayReport(BaseModel):
    """Detailed therapy report for a single day."""

    id: int
    date: date
    total_therapy_hours: float | None = None
    session_count: int = Field(default=0, description="Number of therapy sessions")

    ahi: float | None = Field(default=None, description="Apnea-Hypopnea Index")
    rdi: float | None = Field(default=None, description="Respiratory Disturbance Index")

    obstructive_apneas: int = 0
    hypopneas: int = 0
    central_apneas: int = 0
    reras: int = 0
    flow_limitations: int = 0

    pressure_median: float | None = Field(
        default=None, description="Median pressure (cmH₂O)"
    )
    pressure_95th: float | None = Field(
        default=None, description="95th percentile pressure (cmH₂O)"
    )
    pressure_max: float | None = Field(
        default=None, description="Maximum pressure (cmH₂O)"
    )

    leak_median: float | None = Field(
        default=None, description="Median leak rate (L/min)"
    )
    leak_95th: float | None = Field(
        default=None, description="95th percentile leak rate (L/min)"
    )
    leak_max: float | None = Field(
        default=None, description="Maximum leak rate (L/min)"
    )

    spo2_avg: float | None = Field(default=None, description="Average SpO₂ (%)")
    spo2_min: float | None = Field(default=None, description="Minimum SpO₂ (%)")
    spo2_median: float | None = Field(default=None, description="Median SpO₂ (%)")
    pulse_avg: float | None = Field(
        default=None, description="Average pulse rate (bpm)"
    )

    compliant: bool = Field(
        default=False, description="Met minimum 4-hour usage requirement"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "date": "2024-01-15",
                "total_therapy_hours": 7.67,
                "session_count": 1,
                "ahi": 2.3,
                "rdi": 2.8,
                "obstructive_apneas": 12,
                "hypopneas": 6,
                "central_apneas": 0,
                "reras": 2,
                "flow_limitations": 15,
                "pressure_median": 10.2,
                "pressure_95th": 12.8,
                "pressure_max": 14.2,
                "leak_median": 8.5,
                "leak_95th": 18.2,
                "leak_max": 24.0,
                "spo2_avg": 96.5,
                "spo2_min": 91,
                "spo2_median": 97,
                "pulse_avg": 68,
                "compliant": True,
            }
        }


class DayTextReport(BaseModel):
    """Human-readable text report for a single day."""

    date: date
    summary: str = Field(description="Human-readable summary of the day's therapy")

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-01-15",
                "summary": "On January 15, 2024, therapy was used for 7.7 hours with good compliance. "
                "The AHI was 2.3 events per hour (normal range, <5). "
                "There were 12 obstructive apneas and 6 hypopneas. "
                "Median pressure was 10.2 cmH₂O with a 95th percentile of 12.8 cmH₂O. "
                "Leak rates were well controlled with a median of 8.5 L/min. "
                "Average SpO₂ was 96.5% with a minimum of 91%. "
                "Overall, this represents effective therapy with minimal events.",
            }
        }
