"""Pydantic models for Profile data."""

from datetime import date, datetime

from pydantic import BaseModel, Field


class ProfileSummary(BaseModel):
    """Summary information about a user profile."""

    id: int
    name: str
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None
    height_cm: float | None = None
    created_at: datetime
    updated_at: datetime

    # Computed fields
    machine_count: int = Field(default=0, description="Number of devices")
    total_days: int = Field(default=0, description="Total days of therapy data")
    date_range_start: date | None = Field(default=None, description="First day of data")
    date_range_end: date | None = Field(default=None, description="Last day of data")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "first_name": "John",
                "last_name": "Doe",
                "date_of_birth": "1975-05-15",
                "height_cm": 180.0,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-15T12:30:00",
                "machine_count": 2,
                "total_days": 365,
                "date_range_start": "2023-01-01",
                "date_range_end": "2023-12-31",
            }
        }


class ProfileDetail(BaseModel):
    """Detailed profile information including therapy overview."""

    id: int
    name: str
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None
    height_cm: float | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime

    # Therapy summary
    total_therapy_hours: float = Field(default=0, description="Total hours of therapy")
    avg_ahi: float | None = Field(
        default=None, description="Average AHI across all days"
    )
    compliance_rate: float | None = Field(
        default=None, description="Percentage of days with >4 hours usage"
    )
    days_with_data: int = Field(
        default=0, description="Number of days with therapy data"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "first_name": "John",
                "last_name": "Doe",
                "date_of_birth": "1975-05-15",
                "height_cm": 180.0,
                "notes": "Patient reports improved sleep quality",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-15T12:30:00",
                "total_therapy_hours": 2920.5,
                "avg_ahi": 3.2,
                "compliance_rate": 95.0,
                "days_with_data": 365,
            }
        }
