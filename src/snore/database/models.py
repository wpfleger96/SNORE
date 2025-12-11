"""
SQLAlchemy ORM models for SNORE database.

Defines the complete database schema including:
- Core CPAP data tables (devices, sessions, waveforms, events, statistics, settings)
- Medical analysis infrastructure (knowledge base, patterns, analysis results)
"""

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from snore.database.types import ValidatedJSONWithDefault


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


def utc_now() -> datetime:
    """Return current UTC timestamp for database defaults."""
    return datetime.now(UTC)


class Profile(Base):
    """User/patient profile (OSCAR-compatible single-user model)."""

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    first_name: Mapped[str | None] = mapped_column(String(100))
    last_name: Mapped[str | None] = mapped_column(String(100))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    height_cm: Mapped[int | None] = mapped_column(Integer)
    settings: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    devices = relationship("Device", back_populates="profile")
    days = relationship("Day", back_populates="profile", cascade="all, delete-orphan")

    __table_args__ = (CheckConstraint("length(username) > 0", name="chk_username"),)

    def __repr__(self) -> str:
        return f"<Profile(id={self.id}, username={self.username})>"


class Device(Base):
    """CPAP/BiPAP/Oximeter device."""

    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE")
    )
    manufacturer: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    serial_number: Mapped[str] = mapped_column(String, unique=True)
    firmware_version: Mapped[str | None] = mapped_column(String)
    hardware_version: Mapped[str | None] = mapped_column(String)
    product_code: Mapped[str | None] = mapped_column(String)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    last_import: Mapped[datetime | None] = mapped_column(DateTime)

    profile = relationship("Profile", back_populates="devices")
    sessions = relationship(
        "Session", back_populates="device", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("length(manufacturer) > 0", name="chk_manufacturer"),
        CheckConstraint("length(serial_number) > 0", name="chk_serial"),
    )

    def __repr__(self) -> str:
        return f"<Device(id={self.id}, manufacturer={self.manufacturer}, model={self.model}, serial={self.serial_number})>"


class Day(Base):
    """Daily aggregated statistics (OSCAR-compatible pre-calculated cache)."""

    __tablename__ = "days"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE")
    )
    date: Mapped[date] = mapped_column(Date)

    # Pre-calculated statistics (cached for performance)
    session_count: Mapped[int] = mapped_column(Integer, default=0)
    total_therapy_hours: Mapped[float] = mapped_column(Float, default=0.0)

    obstructive_apneas: Mapped[int] = mapped_column(Integer, default=0)
    central_apneas: Mapped[int] = mapped_column(Integer, default=0)
    hypopneas: Mapped[int] = mapped_column(Integer, default=0)
    reras: Mapped[int] = mapped_column(Integer, default=0)

    ahi: Mapped[float | None] = mapped_column(Float)
    oai: Mapped[float | None] = mapped_column(Float)
    cai: Mapped[float | None] = mapped_column(Float)
    hi: Mapped[float | None] = mapped_column(Float)

    pressure_min: Mapped[float | None] = mapped_column(Float)
    pressure_max: Mapped[float | None] = mapped_column(Float)
    pressure_median: Mapped[float | None] = mapped_column(Float)
    pressure_mean: Mapped[float | None] = mapped_column(Float)
    pressure_95th: Mapped[float | None] = mapped_column(Float)

    leak_min: Mapped[float | None] = mapped_column(Float)
    leak_max: Mapped[float | None] = mapped_column(Float)
    leak_median: Mapped[float | None] = mapped_column(Float)
    leak_mean: Mapped[float | None] = mapped_column(Float)
    leak_95th: Mapped[float | None] = mapped_column(Float)

    spo2_min: Mapped[float | None] = mapped_column(Float)
    spo2_max: Mapped[float | None] = mapped_column(Float)
    spo2_mean: Mapped[float | None] = mapped_column(Float)
    spo2_avg: Mapped[float | None] = mapped_column(Float)  # Alias for compatibility

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    profile = relationship("Profile", back_populates="days")
    sessions = relationship("Session", back_populates="day")

    __table_args__ = (UniqueConstraint("profile_id", "date", name="uq_profile_date"),)

    def __repr__(self) -> str:
        return f"<Day(id={self.id}, profile_id={self.profile_id}, date={self.date}, ahi={self.ahi})>"


class Session(Base):
    """Individual therapy session."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"))
    day_id: Mapped[int | None] = mapped_column(
        ForeignKey("days.id", ondelete="CASCADE")
    )
    device_session_id: Mapped[str] = mapped_column(String)
    start_time: Mapped[datetime] = mapped_column(DateTime)
    end_time: Mapped[datetime] = mapped_column(DateTime)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    therapy_mode: Mapped[str | None] = mapped_column(String)
    import_date: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    import_source: Mapped[str | None] = mapped_column(String)
    parser_version: Mapped[str | None] = mapped_column(String)
    data_quality_notes: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    has_waveform_data: Mapped[bool] = mapped_column(Boolean, default=False)
    has_event_data: Mapped[bool] = mapped_column(Boolean, default=False)
    has_statistics: Mapped[bool] = mapped_column(Boolean, default=False)

    device = relationship("Device", back_populates="sessions")
    day = relationship("Day", back_populates="sessions")
    waveforms = relationship(
        "Waveform", back_populates="session", cascade="all, delete-orphan"
    )
    events = relationship(
        "Event", back_populates="session", cascade="all, delete-orphan"
    )
    statistics = relationship(
        "Statistics",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )
    settings = relationship(
        "Setting", back_populates="session", cascade="all, delete-orphan"
    )
    analysis_results = relationship(
        "AnalysisResult", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("device_id", "device_session_id", name="uq_device_session"),
        CheckConstraint("end_time >= start_time", name="chk_time_range"),
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0", name="chk_duration"
        ),
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, device_id={self.device_id}, start={self.start_time})>"


class Waveform(Base):
    """Time-series waveform data."""

    __tablename__ = "waveforms"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE")
    )
    waveform_type: Mapped[str] = mapped_column(String)
    sample_rate: Mapped[float] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String)
    min_value: Mapped[float | None] = mapped_column(Float)
    max_value: Mapped[float | None] = mapped_column(Float)
    mean_value: Mapped[float | None] = mapped_column(Float)
    data_blob: Mapped[bytes] = mapped_column(LargeBinary)
    sample_count: Mapped[int | None] = mapped_column(Integer)

    session = relationship("Session", back_populates="waveforms")

    __table_args__ = (
        UniqueConstraint("session_id", "waveform_type", name="uq_session_waveform"),
        CheckConstraint("sample_rate > 0", name="chk_sample_rate"),
    )

    def __repr__(self) -> str:
        return f"<Waveform(id={self.id}, session_id={self.session_id}, type={self.waveform_type})>"


class Event(Base):
    """Respiratory events and flags."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE")
    )
    event_type: Mapped[str] = mapped_column(String)
    start_time: Mapped[datetime] = mapped_column(DateTime)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    spo2_drop: Mapped[float | None] = mapped_column(Float)
    peak_flow_limitation: Mapped[float | None] = mapped_column(Float)

    session = relationship("Session", back_populates="events")

    __table_args__ = (
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0", name="chk_duration"
        ),
    )

    def __repr__(self) -> str:
        return f"<Event(id={self.id}, session_id={self.session_id}, type={self.event_type}, start={self.start_time})>"


class Statistics(Base):
    """Session statistics and pre-calculated summary data."""

    __tablename__ = "statistics"

    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True
    )

    obstructive_apneas: Mapped[int] = mapped_column(Integer, default=0)
    central_apneas: Mapped[int] = mapped_column(Integer, default=0)
    mixed_apneas: Mapped[int] = mapped_column(Integer, default=0)
    hypopneas: Mapped[int] = mapped_column(Integer, default=0)
    reras: Mapped[int] = mapped_column(Integer, default=0)
    flow_limitations: Mapped[int] = mapped_column(Integer, default=0)

    ahi: Mapped[float | None] = mapped_column(Float)
    oai: Mapped[float | None] = mapped_column(Float)
    cai: Mapped[float | None] = mapped_column(Float)
    hi: Mapped[float | None] = mapped_column(Float)
    rei: Mapped[float | None] = mapped_column(Float)

    pressure_min: Mapped[float | None] = mapped_column(Float)
    pressure_max: Mapped[float | None] = mapped_column(Float)
    pressure_median: Mapped[float | None] = mapped_column(Float)
    pressure_mean: Mapped[float | None] = mapped_column(Float)
    pressure_95th: Mapped[float | None] = mapped_column(Float)

    leak_min: Mapped[float | None] = mapped_column(Float)
    leak_max: Mapped[float | None] = mapped_column(Float)
    leak_median: Mapped[float | None] = mapped_column(Float)
    leak_mean: Mapped[float | None] = mapped_column(Float)
    leak_95th: Mapped[float | None] = mapped_column(Float)
    leak_percentile_70: Mapped[float | None] = mapped_column(Float)

    respiratory_rate_min: Mapped[float | None] = mapped_column(Float)
    respiratory_rate_max: Mapped[float | None] = mapped_column(Float)
    respiratory_rate_mean: Mapped[float | None] = mapped_column(Float)

    tidal_volume_min: Mapped[float | None] = mapped_column(Float)
    tidal_volume_max: Mapped[float | None] = mapped_column(Float)
    tidal_volume_mean: Mapped[float | None] = mapped_column(Float)

    minute_ventilation_min: Mapped[float | None] = mapped_column(Float)
    minute_ventilation_max: Mapped[float | None] = mapped_column(Float)
    minute_ventilation_mean: Mapped[float | None] = mapped_column(Float)

    spo2_min: Mapped[float | None] = mapped_column(Float)
    spo2_max: Mapped[float | None] = mapped_column(Float)
    spo2_mean: Mapped[float | None] = mapped_column(Float)
    spo2_time_below_90: Mapped[int | None] = mapped_column(Integer)
    pulse_min: Mapped[float | None] = mapped_column(Float)
    pulse_max: Mapped[float | None] = mapped_column(Float)
    pulse_mean: Mapped[float | None] = mapped_column(Float)

    usage_hours: Mapped[float | None] = mapped_column(Float)

    session = relationship("Session", back_populates="statistics")

    def __repr__(self) -> str:
        return f"<Statistics(session_id={self.session_id}, ahi={self.ahi})>"


class Setting(Base):
    """Device configuration settings."""

    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE")
    )
    key: Mapped[str] = mapped_column(String)
    value: Mapped[str | None] = mapped_column(String)

    session = relationship("Session", back_populates="settings")

    __table_args__ = (UniqueConstraint("session_id", "key", name="uq_session_key"),)

    def __repr__(self) -> str:
        return f"<Setting(id={self.id}, session_id={self.session_id}, key={self.key})>"


class AnalysisResult(Base):
    """Track programmatic analysis results for sessions."""

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE")
    )
    timestamp_start: Mapped[datetime] = mapped_column(DateTime)
    timestamp_end: Mapped[datetime] = mapped_column(DateTime)
    programmatic_result_json: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    processing_time_ms: Mapped[int | None] = mapped_column(Integer)
    engine_versions_json: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    session = relationship("Session", back_populates="analysis_results")
    detected_patterns = relationship(
        "DetectedPattern", back_populates="analysis", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<AnalysisResult(id={self.id}, session_id={self.session_id})>"


class DetectedPattern(Base):
    """
    Individual pattern detections from analysis.

    Pattern definitions are stored in code (snore.knowledge.patterns), not database.
    This table only stores runtime detections with references to pattern IDs.
    """

    __tablename__ = "detected_patterns"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    analysis_result_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_results.id", ondelete="CASCADE")
    )
    pattern_id: Mapped[str] = mapped_column(String(100))
    start_time: Mapped[datetime] = mapped_column(DateTime)
    duration: Mapped[float | None] = mapped_column(Float)  # seconds
    confidence: Mapped[float] = mapped_column(Float)
    detected_by: Mapped[str] = mapped_column(String(20))  # programmatic
    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    notes: Mapped[str | None] = mapped_column(Text)

    analysis = relationship("AnalysisResult", back_populates="detected_patterns")

    def __repr__(self) -> str:
        return f"<DetectedPattern(id={self.id}, pattern={self.pattern_id}, confidence={self.confidence})>"
