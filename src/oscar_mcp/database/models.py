"""
SQLAlchemy ORM models for OSCAR-MCP database.

Defines the complete database schema including:
- Core CPAP data tables (devices, sessions, waveforms, events, statistics, settings)
- Medical analysis infrastructure (knowledge base, patterns, analysis results)
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Text,
    Date,
    DateTime,
    Boolean,
    ForeignKey,
    LargeBinary,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship, declarative_base

from oscar_mcp.database.types import ValidatedJSON, ValidatedJSONWithDefault

Base = declarative_base()


# Standardized datetime helpers
def utc_now():
    """Return current UTC timestamp for database defaults."""
    return datetime.now(timezone.utc)


class Profile(Base):
    """User/patient profile (OSCAR-compatible single-user model)."""

    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    date_of_birth = Column(Date)
    height_cm = Column(Integer)
    settings = Column(ValidatedJSONWithDefault)  # Profile-specific settings (e.g., day_split_time)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    devices = relationship("Device", back_populates="profile")
    days = relationship("Day", back_populates="profile", cascade="all, delete-orphan")

    __table_args__ = (CheckConstraint("length(username) > 0", name="chk_username"),)

    def __repr__(self):
        return f"<Profile(id={self.id}, username={self.username})>"


class Device(Base):
    """CPAP/BiPAP/Oximeter device."""

    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=True)
    manufacturer = Column(String, nullable=False)
    model = Column(String, nullable=False)
    serial_number = Column(String, unique=True, nullable=False)
    firmware_version = Column(String)
    hardware_version = Column(String)
    product_code = Column(String)
    first_seen = Column(DateTime, default=utc_now)
    last_import = Column(DateTime)

    # Relationships
    profile = relationship("Profile", back_populates="devices")
    sessions = relationship("Session", back_populates="device", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("length(manufacturer) > 0", name="chk_manufacturer"),
        CheckConstraint("length(serial_number) > 0", name="chk_serial"),
    )

    def __repr__(self):
        return f"<Device(id={self.id}, manufacturer={self.manufacturer}, model={self.model}, serial={self.serial_number})>"


class Day(Base):
    """Daily aggregated statistics (OSCAR-compatible pre-calculated cache)."""

    __tablename__ = "days"

    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)

    # Pre-calculated statistics (cached for performance)
    session_count = Column(Integer, default=0)
    total_therapy_hours = Column(Float, default=0.0)

    # Respiratory event counts
    obstructive_apneas = Column(Integer, default=0)
    central_apneas = Column(Integer, default=0)
    hypopneas = Column(Integer, default=0)
    reras = Column(Integer, default=0)

    # Respiratory indices
    ahi = Column(Float)
    oai = Column(Float)
    cai = Column(Float)
    hi = Column(Float)

    # Pressure statistics
    pressure_min = Column(Float)
    pressure_max = Column(Float)
    pressure_median = Column(Float)
    pressure_mean = Column(Float)
    pressure_95th = Column(Float)

    # Leak statistics
    leak_min = Column(Float)
    leak_max = Column(Float)
    leak_median = Column(Float)
    leak_mean = Column(Float)
    leak_95th = Column(Float)

    # Oximetry statistics
    spo2_min = Column(Float)
    spo2_max = Column(Float)
    spo2_mean = Column(Float)
    spo2_avg = Column(Float)  # Alias for compatibility

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    profile = relationship("Profile", back_populates="days")
    sessions = relationship("Session", back_populates="day")

    __table_args__ = (UniqueConstraint("profile_id", "date", name="uq_profile_date"),)

    def __repr__(self):
        return (
            f"<Day(id={self.id}, profile_id={self.profile_id}, date={self.date}, ahi={self.ahi})>"
        )


class Session(Base):
    """Individual therapy session."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("devices.id", ondelete="CASCADE"), nullable=False)
    day_id = Column(Integer, ForeignKey("days.id", ondelete="CASCADE"), nullable=True)
    device_session_id = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration_seconds = Column(Float)
    therapy_mode = Column(String)
    import_date = Column(DateTime, default=utc_now)
    import_source = Column(String)
    parser_version = Column(String)
    data_quality_notes = Column(ValidatedJSONWithDefault)  # JSON array of quality issues
    has_waveform_data = Column(Boolean, default=False)
    has_event_data = Column(Boolean, default=False)
    has_statistics = Column(Boolean, default=False)

    # Relationships
    device = relationship("Device", back_populates="sessions")
    day = relationship("Day", back_populates="sessions")
    waveforms = relationship("Waveform", back_populates="session", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="session", cascade="all, delete-orphan")
    statistics = relationship(
        "Statistics", back_populates="session", uselist=False, cascade="all, delete-orphan"
    )
    settings = relationship("Setting", back_populates="session", cascade="all, delete-orphan")
    analysis_results = relationship(
        "AnalysisResult", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("device_id", "device_session_id", name="uq_device_session"),
        CheckConstraint("end_time >= start_time", name="chk_time_range"),
        CheckConstraint("duration_seconds IS NULL OR duration_seconds >= 0", name="chk_duration"),
    )

    def __repr__(self):
        return f"<Session(id={self.id}, device_id={self.device_id}, start={self.start_time})>"


class Waveform(Base):
    """Time-series waveform data."""

    __tablename__ = "waveforms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    waveform_type = Column(String, nullable=False)
    sample_rate = Column(Float, nullable=False)
    unit = Column(String)
    min_value = Column(Float)
    max_value = Column(Float)
    mean_value = Column(Float)
    data_blob = Column(LargeBinary, nullable=False)
    sample_count = Column(Integer)

    # Relationships
    session = relationship("Session", back_populates="waveforms")

    __table_args__ = (
        UniqueConstraint("session_id", "waveform_type", name="uq_session_waveform"),
        CheckConstraint("sample_rate > 0", name="chk_sample_rate"),
    )

    def __repr__(self):
        return f"<Waveform(id={self.id}, session_id={self.session_id}, type={self.waveform_type})>"


class Event(Base):
    """Respiratory events and flags."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    event_type = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    duration_seconds = Column(Float)
    spo2_drop = Column(Float)
    peak_flow_limitation = Column(Float)

    # Relationships
    session = relationship("Session", back_populates="events")

    __table_args__ = (
        CheckConstraint("duration_seconds IS NULL OR duration_seconds >= 0", name="chk_duration"),
    )

    def __repr__(self):
        return f"<Event(id={self.id}, session_id={self.session_id}, type={self.event_type}, start={self.start_time})>"


class Statistics(Base):
    """Session statistics and pre-calculated summary data."""

    __tablename__ = "statistics"

    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True)

    # Respiratory event counts
    obstructive_apneas = Column(Integer, default=0)
    central_apneas = Column(Integer, default=0)
    mixed_apneas = Column(Integer, default=0)
    hypopneas = Column(Integer, default=0)
    reras = Column(Integer, default=0)
    flow_limitations = Column(Integer, default=0)

    # Respiratory indices
    ahi = Column(Float)
    oai = Column(Float)
    cai = Column(Float)
    hi = Column(Float)
    rei = Column(Float)

    # Pressure statistics
    pressure_min = Column(Float)
    pressure_max = Column(Float)
    pressure_median = Column(Float)
    pressure_mean = Column(Float)
    pressure_95th = Column(Float)

    # Leak statistics
    leak_min = Column(Float)
    leak_max = Column(Float)
    leak_median = Column(Float)
    leak_mean = Column(Float)
    leak_95th = Column(Float)
    leak_percentile_70 = Column(Float)

    # Respiratory rate statistics
    respiratory_rate_min = Column(Float)
    respiratory_rate_max = Column(Float)
    respiratory_rate_mean = Column(Float)

    # Tidal volume statistics
    tidal_volume_min = Column(Float)
    tidal_volume_max = Column(Float)
    tidal_volume_mean = Column(Float)

    # Minute ventilation statistics
    minute_ventilation_min = Column(Float)
    minute_ventilation_max = Column(Float)
    minute_ventilation_mean = Column(Float)

    # Oximetry statistics
    spo2_min = Column(Float)
    spo2_max = Column(Float)
    spo2_mean = Column(Float)
    spo2_time_below_90 = Column(Integer)
    pulse_min = Column(Float)
    pulse_max = Column(Float)
    pulse_mean = Column(Float)

    # Usage
    usage_hours = Column(Float)

    # Relationships
    session = relationship("Session", back_populates="statistics")

    def __repr__(self):
        return f"<Statistics(session_id={self.session_id}, ahi={self.ahi})>"


class Setting(Base):
    """Device configuration settings."""

    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    key = Column(String, nullable=False)
    value = Column(String)

    # Relationships
    session = relationship("Session", back_populates="settings")

    __table_args__ = (UniqueConstraint("session_id", "key", name="uq_session_key"),)

    def __repr__(self):
        return f"<Setting(id={self.id}, session_id={self.session_id}, key={self.key})>"


# ============================================================================
# MEDICAL ANALYSIS INFRASTRUCTURE
# ============================================================================


class KnowledgeBase(Base):
    """Core knowledge registry for clinical patterns and guidelines."""

    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_id = Column(String(100), unique=True, nullable=False)
    category = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    source = Column(String(255))
    version = Column(String(50))
    content_type = Column(String(50))
    priority = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
    )

    # Relationships
    contents = relationship(
        "KnowledgeContent", back_populates="knowledge", cascade="all, delete-orphan"
    )
    media = relationship("KnowledgeMedia", back_populates="knowledge", cascade="all, delete-orphan")

    def __repr__(self):
        return (
            f"<KnowledgeBase(id={self.knowledge_id}, category={self.category}, title={self.title})>"
        )


class AnalysisPattern(Base):
    """Pattern definitions for both programmatic and LLM engines."""

    __tablename__ = "analysis_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_id = Column(String(100), unique=True, nullable=False)
    pattern_name = Column(String(255), nullable=False)
    category = Column(String(50), nullable=False)
    visual_description = Column(Text)
    clinical_significance = Column(Text)
    algorithm_rules_json = Column(ValidatedJSONWithDefault)  # Algorithmic detection rules
    threshold_values_json = Column(ValidatedJSONWithDefault)  # Clinical threshold values
    example_metrics_json = Column(ValidatedJSONWithDefault)  # Example metric ranges
    severity_level = Column(String(20))
    confidence_threshold = Column(Float, default=0.6)
    reference_image_path = Column(String(500))
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
    )

    # Relationships
    detected = relationship("DetectedPattern", back_populates="pattern")
    relationships_from = relationship(
        "PatternRelationship",
        foreign_keys="PatternRelationship.from_pattern_id",
        back_populates="from_pattern",
    )
    relationships_to = relationship(
        "PatternRelationship",
        foreign_keys="PatternRelationship.to_pattern_id",
        back_populates="to_pattern",
    )

    def __repr__(self):
        return f"<AnalysisPattern(id={self.pattern_id}, name={self.pattern_name}, category={self.category})>"


class KnowledgeContent(Base):
    """Structured content storage for knowledge base entries."""

    __tablename__ = "knowledge_content"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_id = Column(
        String(100), ForeignKey("knowledge_base.knowledge_id", ondelete="CASCADE"), nullable=False
    )
    content_key = Column(String(100), nullable=False)
    content_value = Column(Text, nullable=False)
    data_type = Column(String(20), nullable=False)  # json_array, json_object, markdown, html
    display_order = Column(Integer, default=0)
    is_required = Column(Boolean, default=False)

    # Relationships
    knowledge = relationship("KnowledgeBase", back_populates="contents")

    def __repr__(self):
        return f"<KnowledgeContent(id={self.id}, knowledge_id={self.knowledge_id}, key={self.content_key})>"


class KnowledgeMedia(Base):
    """Image and media file references for knowledge base."""

    __tablename__ = "knowledge_media"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_id = Column(
        String(100), ForeignKey("knowledge_base.knowledge_id", ondelete="CASCADE"), nullable=False
    )
    media_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    caption = Column(Text)
    metadata_json = Column(ValidatedJSONWithDefault)  # File metadata (resolution, format, etc.)
    file_hash = Column(String(64))  # SHA256
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    knowledge = relationship("KnowledgeBase", back_populates="media")

    def __repr__(self):
        return f"<KnowledgeMedia(id={self.id}, knowledge_id={self.knowledge_id}, type={self.media_type})>"


class AnalysisResult(Base):
    """Track dual-engine analysis results (programmatic + LLM)."""

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    timestamp_start = Column(DateTime, nullable=False)
    timestamp_end = Column(DateTime, nullable=False)
    programmatic_result_json = Column(ValidatedJSONWithDefault)  # Algorithmic analysis results
    llm_result_json = Column(ValidatedJSONWithDefault)  # LLM analysis results
    combined_result_json = Column(ValidatedJSONWithDefault)  # Combined/reconciled results
    agreement_score = Column(Float)
    processing_time_ms = Column(Integer)
    engine_versions_json = Column(ValidatedJSONWithDefault)  # Version info for reproducibility
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    session = relationship("Session", back_populates="analysis_results")
    detected_patterns = relationship(
        "DetectedPattern", back_populates="analysis", cascade="all, delete-orphan"
    )
    feedback = relationship(
        "AnalysisFeedback", back_populates="analysis", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, session_id={self.session_id}, agreement={self.agreement_score})>"


class DetectedPattern(Base):
    """Individual pattern detections from analysis."""

    __tablename__ = "detected_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_result_id = Column(
        Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), nullable=False
    )
    pattern_id = Column(
        String(100),
        ForeignKey("analysis_patterns.pattern_id", ondelete="CASCADE"),
        nullable=False,
    )
    start_time = Column(DateTime, nullable=False)
    duration = Column(Float)  # seconds
    confidence = Column(Float, nullable=False)
    detected_by = Column(String(20), nullable=False)  # programmatic, llm, both
    metrics_json = Column(ValidatedJSONWithDefault)  # Pattern-specific metrics
    notes = Column(Text)

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="detected_patterns")
    pattern = relationship("AnalysisPattern", back_populates="detected")

    def __repr__(self):
        return f"<DetectedPattern(id={self.id}, pattern={self.pattern_id}, confidence={self.confidence})>"


class AnalysisFeedback(Base):
    """Learning and improvement tracking for analysis system."""

    __tablename__ = "analysis_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_result_id = Column(
        Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), nullable=False
    )
    feedback_type = Column(String(50), nullable=False)
    discrepancy_description = Column(Text)
    suggested_improvement = Column(Text)
    implemented = Column(Boolean, default=False)
    reviewed_by = Column(String(100))
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="feedback")

    def __repr__(self):
        return f"<AnalysisFeedback(id={self.id}, type={self.feedback_type}, implemented={self.implemented})>"


class AlgorithmConfig(Base):
    """Algorithm parameters and configuration."""

    __tablename__ = "algorithm_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm_name = Column(String(100), unique=True, nullable=False)
    version = Column(String(50), nullable=False)
    parameters_json = Column(ValidatedJSON, nullable=False)  # Algorithm parameters
    is_active = Column(Boolean, default=True)
    performance_metrics_json = Column(ValidatedJSONWithDefault)  # Performance metrics
    last_updated = Column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
    )

    def __repr__(self):
        return f"<AlgorithmConfig(name={self.algorithm_name}, version={self.version}, active={self.is_active})>"


class PatternRelationship(Base):
    """Connections and relationships between patterns."""

    __tablename__ = "pattern_relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    from_pattern_id = Column(
        String(100), ForeignKey("analysis_patterns.pattern_id", ondelete="CASCADE"), nullable=False
    )
    to_pattern_id = Column(
        String(100), ForeignKey("analysis_patterns.pattern_id", ondelete="CASCADE"), nullable=False
    )
    relationship_type = Column(String(50), nullable=False)  # causes, indicates, precedes, etc.
    strength = Column(Float, default=0.5)  # 0-1
    notes = Column(Text)

    # Relationships
    from_pattern = relationship(
        "AnalysisPattern", foreign_keys=[from_pattern_id], back_populates="relationships_from"
    )
    to_pattern = relationship(
        "AnalysisPattern", foreign_keys=[to_pattern_id], back_populates="relationships_to"
    )

    __table_args__ = (
        CheckConstraint("from_pattern_id != to_pattern_id", name="chk_no_self_reference"),
    )

    def __repr__(self):
        return f"<PatternRelationship(from={self.from_pattern_id}, to={self.to_pattern_id}, type={self.relationship_type})>"
