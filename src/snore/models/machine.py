"""Pydantic models for Machine/Device data."""

from datetime import datetime

from pydantic import BaseModel, Field


class MachineSummary(BaseModel):
    """Summary information about a CPAP/BiPAP device."""

    id: int
    machine_id: str = Field(description="Unique machine identifier from OSCAR")
    serial_number: str | None = None
    brand: str | None = None
    model: str | None = None
    machine_type: str = Field(description="Type: CPAP, BiPAP, AutoCPAP, Oximeter, etc.")
    created_at: datetime
    last_import: datetime | None = Field(
        default=None, description="Last data import timestamp"
    )

    session_count: int = Field(
        default=0, description="Total number of therapy sessions"
    )
    total_hours: float = Field(
        default=0, description="Total therapy hours from this device"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "machine_id": "ResMed_AirSense10_12345678",
                "serial_number": "12345678",
                "brand": "ResMed",
                "model": "AirSense 10 AutoSet",
                "machine_type": "AutoCPAP",
                "created_at": "2024-01-01T00:00:00",
                "last_import": "2024-01-15T08:30:00",
                "session_count": 365,
                "total_hours": 2920.5,
            }
        }
