"""
SQLAlchemy ORM models for SNORE database.

Defines the complete database schema including:
- Auth/identity tables (users, auth_identities, invites, oauth_attempts)
- Core CPAP data tables (profiles, devices, sessions, waveforms, events, statistics, settings)
- Medical analysis infrastructure (knowledge base, patterns, analysis results)

Relationship loading policy
---------------------------
All relationships use ``lazy="raise"`` to prevent implicit N+1 queries and to
guarantee that ``MissingGreenlet`` errors cannot occur when these models are
used from async sessions (PR-2).  Every traversal must be explicit: use
``selectinload`` / ``joinedload`` in the calling query or call the relationship
within the open session.

Timestamp classification
------------------------
*Absolute instants* (audit times, import times) use ``UTCDateTime``.  These
values are always written with ``utc_now()`` and must survive a SQLite round-
trip with full timezone information intact.

*Device / session wall-clock columns* (``Session.start_time``, ``Event.start_time``,
``AnalysisResult.timestamp_start`` / ``timestamp_end``,
``DetectedPattern.start_time``) use plain ``DateTime`` without timezone because
CPAP devices record local wall-clock time without a timezone offset.  Storing
them as UTC would invent timezone facts the source data does not contain.
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
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from snore.database.types import UTCDateTime, ValidatedJSON, ValidatedJSONWithDefault

# Constraint naming convention — must live here on Base.metadata (not env.py)
# so that Base.metadata.create_all emits the same deterministic constraint
# names that Alembic autogenerate produces from the migration chain.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def utc_now() -> datetime:
    """Return current UTC timestamp for database defaults."""
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Auth / identity tables
# ---------------------------------------------------------------------------


class User(Base):
    """Auth identity and account record.  Owns one or more Profiles."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Canonical email: stripped + lower-cased at one normalization point.
    canonical_email: Mapped[str] = mapped_column(String(254), unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(150))
    # admin | member | demo
    role: Mapped[str] = mapped_column(String(20), default="member")
    # Bumped on password-change/disable/role-change; invalidates all cookies.
    session_version: Mapped[int] = mapped_column(Integer, default=0)
    disabled_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    # FK to profiles — set after the first profile is created.
    default_profile_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(
            "profiles.id",
            ondelete="SET NULL",
            use_alter=True,
            name="fk_users_default_profile_id_profiles",
        ),
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now
    )
    preferences: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, nullable=True, default=dict
    )
    # Set to True when the user explicitly unlinks Google; blocks the email
    # auto-link path in resolve_login so the next "Sign in with Google" cannot
    # silently re-establish a severed identity.  Cleared to False when the user
    # deliberately re-links via the invite-signup flow.
    google_link_disabled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0"
    )
    # TOTP 2FA columns.
    # Non-null + totp_enabled_at is None  = pending unconfirmed setup.
    # Non-null + totp_enabled_at is set   = active 2FA enrollment.
    # Stored as plaintext Base32 by design: the symmetric secret must be
    # recoverable for verification; defense is DB file access control.
    totp_secret: Mapped[str | None] = mapped_column(String(64))
    totp_enabled_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    # Last successfully verified time-step — used for replay prevention.
    totp_last_used_step: Mapped[int | None] = mapped_column(Integer)

    profiles = relationship(
        "Profile",
        back_populates="user",
        foreign_keys="Profile.user_id",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    auth_identities = relationship(
        "AuthIdentity",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    invites_created = relationship(
        "Invite",
        back_populates="created_by_user",
        foreign_keys="Invite.created_by",
        lazy="raise",
    )
    totp_recovery_codes = relationship(
        "TotpRecoveryCode",
        cascade="all, delete-orphan",
        lazy="raise",
    )

    __table_args__ = (
        CheckConstraint("length(canonical_email) > 0", name="chk_user_email"),
        CheckConstraint("role IN ('admin','member','demo')", name="chk_user_role"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.canonical_email}, role={self.role})>"


class AuthIdentity(Base):
    """OAuth identity linked to a User (provider + subject pair)."""

    __tablename__ = "auth_identities"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    provider: Mapped[str] = mapped_column(String(50))  # "google"
    subject: Mapped[str] = mapped_column(String(255))  # provider's user ID
    email: Mapped[str | None] = mapped_column(String(254))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)

    user = relationship("User", back_populates="auth_identities", lazy="raise")

    __table_args__ = (
        UniqueConstraint(
            "provider", "subject", name="uq_auth_identity_provider_subject"
        ),
        # Covers the (user_id, provider) predicate used by get_me and unlink_google.
        Index("ix_auth_identities_user_provider", "user_id", "provider"),
    )

    def __repr__(self) -> str:
        return f"<AuthIdentity(id={self.id}, provider={self.provider}, user_id={self.user_id})>"


class Invite(Base):
    """Invitation to create a SNORE account."""

    __tablename__ = "invites"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(254))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    role: Mapped[str] = mapped_column(String(20), default="member")
    created_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    redeemed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)

    created_by_user = relationship(
        "User",
        back_populates="invites_created",
        foreign_keys=[created_by],
        lazy="raise",
    )

    __table_args__ = (
        CheckConstraint("role IN ('admin','member','demo')", name="chk_invite_role"),
    )

    def __repr__(self) -> str:
        return f"<Invite(id={self.id}, email={self.email})>"


class TotpRecoveryCode(Base):
    """One-time recovery code for a TOTP-enrolled user.

    Each row represents a single code.  ``used_at`` non-null means the code
    has been redeemed and cannot be used again.  Rows are cascade-deleted with
    their owning user.
    """

    __tablename__ = "totp_recovery_codes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    code_hash: Mapped[str] = mapped_column(
        String(255)
    )  # argon2 encoded hash of raw code
    used_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)

    __table_args__ = (Index("ix_totp_recovery_codes_user_id", "user_id"),)

    def __repr__(self) -> str:
        return f"<TotpRecoveryCode(id={self.id}, user_id={self.user_id})>"


class OauthAttempt(Base):
    """Server-side OAuth flow state (one-use, browser-bound)."""

    __tablename__ = "oauth_attempts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    state: Mapped[str] = mapped_column(String(128), unique=True)
    kind: Mapped[str] = mapped_column(String(20))  # "login" | "signup"
    invite_id: Mapped[int | None] = mapped_column(
        ForeignKey("invites.id", ondelete="SET NULL")
    )
    expected_canonical_email: Mapped[str | None] = mapped_column(String(254))
    nonce: Mapped[str | None] = mapped_column(String(128))
    pkce_verifier: Mapped[str | None] = mapped_column(String(128))
    # SHA-256 of the pre-auth browser session cookie value
    browser_session_hash: Mapped[str | None] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    consumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    # Non-NULL marks a connect-kind flow: link Google to this already-authenticated
    # user.  kind stays "login" so the CHECK constraint holds on pre-existing DBs.
    connect_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)

    __table_args__ = (
        Index("ix_oauth_attempts_expires_at", "expires_at"),
        CheckConstraint("kind IN ('login','signup')", name="chk_oauth_kind"),
    )

    def __repr__(self) -> str:
        return (
            f"<OauthAttempt(id={self.id}, kind={self.kind}, state={self.state[:8]}...)>"
        )


# ---------------------------------------------------------------------------
# Profile (dataset container, owned by a User)
# ---------------------------------------------------------------------------


class Profile(Base):
    """Dataset container for one user's CPAP data (OSCAR-compatible)."""

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # Human-readable display name (e.g. "My ResMed AirSense 11")
    name: Mapped[str] = mapped_column(String(150))
    # Legacy OSCAR-compatible fields (kept for import compatibility)
    username: Mapped[str | None] = mapped_column(String(100))
    first_name: Mapped[str | None] = mapped_column(String(100))
    last_name: Mapped[str | None] = mapped_column(String(100))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    height_cm: Mapped[int | None] = mapped_column(Integer)
    # User-declared IANA timezone name (e.g. "America/New_York").  Labeling
    # metadata only (A6): device wall-clock timestamps stay naive — this never
    # rewrites timestamps or fabricates UTC offsets.
    timezone: Mapped[str | None] = mapped_column(String(64))
    settings: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    # Deletion tombstone: non-null means profile is being deleted.
    # Tombstoned profiles are invisible to context construction and all queries.
    deleting_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now
    )

    user = relationship(
        "User", back_populates="profiles", foreign_keys=[user_id], lazy="raise"
    )
    devices = relationship(
        "Device",
        back_populates="profile",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    mask_log_entries = relationship(
        "MaskLogEntry",
        back_populates="profile",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    health_samples = relationship(
        "HealthSample",
        back_populates="profile",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    health_nightly_summaries = relationship(
        "HealthNightlySummary",
        back_populates="profile",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_profile_user_name"),
        CheckConstraint("length(name) > 0", name="chk_profile_name"),
    )

    def __repr__(self) -> str:
        return f"<Profile(id={self.id}, user_id={self.user_id}, name={self.name})>"


# ---------------------------------------------------------------------------
# Core CPAP data tables
# ---------------------------------------------------------------------------


class Device(Base):
    """CPAP/BiPAP/Oximeter device, owned by a Profile."""

    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE")
    )
    manufacturer: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    serial_number: Mapped[str] = mapped_column(String)
    firmware_version: Mapped[str | None] = mapped_column(String)
    hardware_version: Mapped[str | None] = mapped_column(String)
    product_code: Mapped[str | None] = mapped_column(String)
    # Absolute audit instants.
    first_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    last_import: Mapped[datetime | None] = mapped_column(UTCDateTime)

    profile = relationship("Profile", back_populates="devices", lazy="raise")
    sessions = relationship(
        "Session",
        back_populates="device",
        cascade="all, delete-orphan",
        lazy="raise",
        overlaps="day,sessions",
    )
    days = relationship(
        "Day",
        back_populates="device",
        cascade="all, delete-orphan",
        lazy="raise",
    )

    __table_args__ = (
        # Serial number is unique per profile (not globally).
        UniqueConstraint(
            "profile_id", "serial_number", name="uq_device_profile_serial"
        ),
        CheckConstraint("length(manufacturer) > 0", name="chk_manufacturer"),
        CheckConstraint("length(serial_number) > 0", name="chk_serial"),
    )

    def __repr__(self) -> str:
        return f"<Device(id={self.id}, profile_id={self.profile_id}, manufacturer={self.manufacturer}, model={self.model}, serial={self.serial_number})>"


class MaskLogEntry(Base):
    """User-entered mask equipment log entry, owned by a Profile."""

    __tablename__ = "mask_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE")
    )
    brand: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model: Mapped[str | None] = mapped_column(String(150), nullable=True)
    size: Mapped[str | None] = mapped_column(String(50))
    # pillows | nasal | full_face — None means not yet declared
    style: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Date the user started using this mask (user-declared, no timezone).
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text)
    # Absolute audit instants.
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now
    )

    profile = relationship("Profile", back_populates="mask_log_entries", lazy="raise")

    __table_args__ = (
        # Style vocabulary: keep in sync with api/schemas.py MaskStyle, migrations 008/009 CHECKs, services/mask_epoch_service.py map, ui/src/utils/maskOptions.ts.
        CheckConstraint(
            "style IS NULL OR style IN ('pillows','nasal','full_face')",
            name="chk_mask_style",
        ),
        CheckConstraint("brand IS NULL OR length(brand) > 0", name="chk_mask_brand"),
        CheckConstraint("model IS NULL OR length(model) > 0", name="chk_mask_model"),
        Index("ix_mask_log_profile_start_date", "profile_id", "start_date"),
    )

    def __repr__(self) -> str:
        return f"<MaskLogEntry(id={self.id}, profile_id={self.profile_id}, brand={self.brand}, model={self.model}, style={self.style})>"


# ---------------------------------------------------------------------------
# Apple Health data tables
# ---------------------------------------------------------------------------


class HealthSample(Base):
    """Raw Apple Health sample, one row per HKSample; source-preserving."""

    __tablename__ = "health_samples"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE")
    )
    # Canonical HealthKit type identifier, e.g. HKCategoryTypeIdentifierSleepAnalysis.
    record_type: Mapped[str] = mapped_column(String(100))
    source_name: Mapped[str] = mapped_column(String(200))
    source_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    device_info: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Apple Health local wall-clock times: no source timezone, never convert.
    start_time: Mapped[datetime] = mapped_column(DateTime)
    end_time: Mapped[datetime] = mapped_column(DateTime)
    # value_text: sleep stage name for category records; NULL for quantity records.
    value_text: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # value_num: quantity value for numeric records; NULL for category records.
    value_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Therapy-night date derived via noon-split at parse time.
    night_date: Mapped[date] = mapped_column(Date)
    utc_offset_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingest_channel: Mapped[str] = mapped_column(String(20))
    # Absolute audit instant.
    imported_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)

    profile = relationship("Profile", back_populates="health_samples", lazy="raise")

    __table_args__ = (
        # Dedup unique expression index with COALESCE sentinels closing the NULL hole.
        # SQLite treats NULLs as distinct in UNIQUE constraints, so category rows
        # (value_num IS NULL) would never conflict on re-import without the sentinel.
        # -1.0 is safe: SpO2 %, respiratory rate, and disturbance counts are all >= 0.
        Index(
            "uq_health_sample_dedup",
            "profile_id",
            "record_type",
            "source_name",
            "start_time",
            "end_time",
            text("coalesce(value_text, '')"),
            text("coalesce(value_num, -1.0)"),
            unique=True,
        ),
        Index(
            "ix_health_samples_profile_type_night",
            "profile_id",
            "record_type",
            "night_date",
        ),
        Index(
            "ix_health_samples_profile_night_source",
            "profile_id",
            "night_date",
            "source_name",
        ),
        CheckConstraint(
            "value_text IS NOT NULL OR value_num IS NOT NULL",
            name="chk_health_sample_value",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<HealthSample(id={self.id}, profile_id={self.profile_id}, "
            f"record_type={self.record_type}, night_date={self.night_date})>"
        )


class HealthNightlySummary(Base):
    """Derived per-night sleep cache; rebuilt by delete-and-recompute."""

    __tablename__ = "health_nightly_summaries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE")
    )
    night_date: Mapped[date] = mapped_column(Date)
    preferred_source: Mapped[str | None] = mapped_column(String(200), nullable=True)
    time_in_bed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_sleep_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    core_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    deep_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    rem_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    awake_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    unspecified_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_efficiency_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage_coverage_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Absolute audit instant.
    computed_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)

    profile = relationship(
        "Profile", back_populates="health_nightly_summaries", lazy="raise"
    )

    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "night_date",
            name="uq_health_nightly_summaries_profile_night",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<HealthNightlySummary(id={self.id}, profile_id={self.profile_id}, "
            f"night_date={self.night_date})>"
        )


class Day(Base):
    """Daily aggregated statistics (OSCAR-compatible pre-calculated cache).

    Rows no Session references are pruned by ``DayManager.recalculate_day``,
    which documents the lifecycle rule.
    """

    __tablename__ = "days"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"))
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

    epap_min: Mapped[float | None] = mapped_column(Float)
    epap_max: Mapped[float | None] = mapped_column(Float)
    epap_median: Mapped[float | None] = mapped_column(Float)
    epap_mean: Mapped[float | None] = mapped_column(Float)
    epap_95th: Mapped[float | None] = mapped_column(Float)

    leak_min: Mapped[float | None] = mapped_column(Float)
    leak_max: Mapped[float | None] = mapped_column(Float)
    leak_median: Mapped[float | None] = mapped_column(Float)
    leak_mean: Mapped[float | None] = mapped_column(Float)
    leak_95th: Mapped[float | None] = mapped_column(Float)

    spo2_min: Mapped[float | None] = mapped_column(Float)
    spo2_max: Mapped[float | None] = mapped_column(Float)
    spo2_mean: Mapped[float | None] = mapped_column(Float)

    # Absolute audit instants.
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now
    )

    device = relationship("Device", back_populates="days", lazy="raise")
    sessions = relationship(
        "Session", back_populates="day", lazy="raise", overlaps="device,sessions"
    )

    __table_args__ = (
        UniqueConstraint("device_id", "date", name="uq_device_date"),
        # Supports composite FK from Session(day_id, device_id) → Day(id, device_id)
        # so both ownership joins are provably identical.
        UniqueConstraint("id", "device_id", name="uq_day_id_device"),
    )

    def __repr__(self) -> str:
        return f"<Day(id={self.id}, device_id={self.device_id}, date={self.date}, ahi={self.ahi})>"


class Session(Base):
    """Individual therapy session."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"))
    day_id: Mapped[int | None] = mapped_column(Integer)
    device_session_id: Mapped[str] = mapped_column(String)
    # Device wall-clock times: no source timezone, never convert.
    start_time: Mapped[datetime] = mapped_column(DateTime)
    end_time: Mapped[datetime] = mapped_column(DateTime)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    therapy_mode: Mapped[str | None] = mapped_column(String)
    # Absolute audit instant.
    import_date: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    import_source: Mapped[str | None] = mapped_column(String)
    parser_version: Mapped[str | None] = mapped_column(String)
    data_quality_notes: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    # Ascending, disjoint [start_offset_s, end_offset_s] mask-on intervals in session
    # offset seconds (list of 2-element lists). NULL = unknown (OSCAR imports,
    # pre-change data); a single-segment session stores [[0.0, duration]].
    mask_on_segments: Mapped[list[Any] | None] = mapped_column(
        ValidatedJSON, nullable=True
    )
    has_waveform_data: Mapped[bool] = mapped_column(Boolean, default=False)
    has_event_data: Mapped[bool] = mapped_column(Boolean, default=False)
    has_statistics: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    device = relationship(
        "Device", back_populates="sessions", lazy="raise", overlaps="day,sessions"
    )
    day = relationship(
        "Day",
        back_populates="sessions",
        foreign_keys="[Session.day_id]",
        lazy="raise",
    )
    waveforms = relationship(
        "Waveform",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    events = relationship(
        "Event",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    statistics = relationship(
        "Statistics",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="raise",
    )
    settings = relationship(
        "Setting",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    analysis_results = relationship(
        "AnalysisResult",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    breaths = relationship(
        "Breath",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="raise",
        overlaps="analysis_result",
    )

    __table_args__ = (
        UniqueConstraint("device_id", "device_session_id", name="uq_device_session"),
        CheckConstraint("end_time >= start_time", name="chk_time_range"),
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0", name="chk_duration"
        ),
        # Composite FK: ensures session.day_id and session.device_id agree on ownership.
        # Day.UNIQUE(id, device_id) makes this constraint enforceable.
        # When day_id IS NULL (session not yet linked to a day), the FK is skipped.
        ForeignKeyConstraint(
            ["day_id", "device_id"],
            ["days.id", "days.device_id"],
            name="fk_session_day_device",
            ondelete="CASCADE",
            use_alter=True,
        ),
        # Supports the _find_overlapping range predicate: device_id equality filter +
        # start_time range scan.  Run per imported session; index prevents full table
        # scan growth as diagnostic-blip sessions accumulate.
        Index("ix_sessions_device_id_start_time", "device_id", "start_time"),
        # Day aggregation and the orphan-day probe both look sessions up by
        # day_id; without this index each is a full sessions scan, making
        # ``db recompute-days`` O(days x sessions).
        Index("ix_sessions_day_id", "day_id"),
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

    session = relationship("Session", back_populates="waveforms", lazy="raise")

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
    # Device wall-clock time: no source timezone, never convert.
    start_time: Mapped[datetime] = mapped_column(DateTime)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    spo2_drop: Mapped[float | None] = mapped_column(Float)
    peak_flow_limitation: Mapped[float | None] = mapped_column(Float)

    session = relationship("Session", back_populates="events", lazy="raise")

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

    # Device-reported (ResMed STR) indices, preserved alongside the computed
    # ahi/oai/cai/hi above (which finalize_statistics recomputes from events).
    ahi_device: Mapped[float | None] = mapped_column(Float)
    oai_device: Mapped[float | None] = mapped_column(Float)
    cai_device: Mapped[float | None] = mapped_column(Float)
    hi_device: Mapped[float | None] = mapped_column(Float)

    pressure_min: Mapped[float | None] = mapped_column(Float)
    pressure_max: Mapped[float | None] = mapped_column(Float)
    pressure_median: Mapped[float | None] = mapped_column(Float)
    pressure_mean: Mapped[float | None] = mapped_column(Float)
    pressure_95th: Mapped[float | None] = mapped_column(Float)

    epap_min: Mapped[float | None] = mapped_column(Float)
    epap_max: Mapped[float | None] = mapped_column(Float)
    epap_median: Mapped[float | None] = mapped_column(Float)
    epap_mean: Mapped[float | None] = mapped_column(Float)
    epap_95th: Mapped[float | None] = mapped_column(Float)

    ipap_median: Mapped[float | None] = mapped_column(Float)
    ipap_95th: Mapped[float | None] = mapped_column(Float)
    ipap_max: Mapped[float | None] = mapped_column(Float)

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
    spo2_median: Mapped[float | None] = mapped_column(Float)
    spo2_95th: Mapped[float | None] = mapped_column(Float)
    spo2_time_below_90: Mapped[int | None] = mapped_column(Integer)
    pulse_min: Mapped[float | None] = mapped_column(Float)
    pulse_max: Mapped[float | None] = mapped_column(Float)
    pulse_mean: Mapped[float | None] = mapped_column(Float)

    usage_hours: Mapped[float | None] = mapped_column(Float)

    # --- STR daily summary extras ---
    # Apnea indices
    uai: Mapped[float | None] = mapped_column(Float)
    ai: Mapped[float | None] = mapped_column(Float)
    # APAP-only
    rin: Mapped[float | None] = mapped_column(Float)
    csr_pct: Mapped[float | None] = mapped_column(Float)
    # VAuto-only
    spont_cyc_pct: Mapped[float | None] = mapped_column(Float)
    # Respiratory rate extras (95th only; max already present above)
    respiratory_rate_95th: Mapped[float | None] = mapped_column(Float)
    # Tidal volume extras (95th only; max already present above)
    tidal_volume_95th: Mapped[float | None] = mapped_column(Float)
    # Minute ventilation extras (95th only; max already present above)
    minute_ventilation_95th: Mapped[float | None] = mapped_column(Float)
    # I:E ratio stats (VAuto-only)
    ie_ratio_median: Mapped[float | None] = mapped_column(Float)
    ie_ratio_95th: Mapped[float | None] = mapped_column(Float)
    ie_ratio_max: Mapped[float | None] = mapped_column(Float)
    # Inspiratory time stats (VAuto-only)
    ti_median: Mapped[float | None] = mapped_column(Float)
    ti_95th: Mapped[float | None] = mapped_column(Float)
    ti_max: Mapped[float | None] = mapped_column(Float)
    # Flow percentiles
    flow_5th: Mapped[float | None] = mapped_column(Float)
    flow_95th: Mapped[float | None] = mapped_column(Float)
    # Blow-side pressure/flow
    blow_press_5th: Mapped[float | None] = mapped_column(Float)
    blow_press_95th: Mapped[float | None] = mapped_column(Float)
    blow_flow_median: Mapped[float | None] = mapped_column(Float)
    # Climate / humidifier stats
    amb_humidity_median: Mapped[float | None] = mapped_column(Float)
    hum_temp_median: Mapped[float | None] = mapped_column(Float)
    htube_temp_median: Mapped[float | None] = mapped_column(Float)
    htube_pow_median: Mapped[float | None] = mapped_column(Float)
    hum_pow_median: Mapped[float | None] = mapped_column(Float)
    # Mask events (count of mask-on events per session)
    mask_events: Mapped[float | None] = mapped_column(Float)

    session = relationship("Session", back_populates="statistics", lazy="raise")

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

    session = relationship("Session", back_populates="settings", lazy="raise")

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
    # Device/session wall-clock windows: no source timezone, never convert.
    timestamp_start: Mapped[datetime] = mapped_column(DateTime)
    timestamp_end: Mapped[datetime] = mapped_column(DateTime)
    programmatic_result_json: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    processing_time_ms: Mapped[int | None] = mapped_column(Integer)
    engine_versions_json: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    # Absolute audit instant — used for latest-result selection.
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)

    session = relationship("Session", back_populates="analysis_results", lazy="raise")
    detected_patterns = relationship(
        "DetectedPattern",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    breaths = relationship(
        "Breath",
        back_populates="analysis_result",
        cascade="all, delete-orphan",
        lazy="raise",
    )

    def __repr__(self) -> str:
        return f"<AnalysisResult(id={self.id}, session_id={self.session_id})>"


class Breath(Base):
    """Immutable per-breath metrics persisted at analysis time.

    Breaths are immutable children of a specific AnalysisResult (analysis run).
    Re-analysis appends a new AnalysisResult + new Breath children; prior runs
    and their breaths are never deleted except via the explicit deletion API
    (cascade handles children).

    **No Alembic migration** (ruling #4): this model ships model-only; fresh DBs
    get the schema via Base.metadata.create_all; pre-existing DBs receive a
    capability-honest error → drop + reimport.

    Denormalised session_id is permitted for query efficiency; uniqueness is on
    (analysis_result_id, breath_number).
    """

    __tablename__ = "breaths"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    analysis_result_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_results.id", ondelete="CASCADE"), nullable=False
    )
    # Denormalised for efficient per-session queries.
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    breath_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timing (session-relative offsets in seconds)
    start_offset_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_offset_s: Mapped[float] = mapped_column(Float, nullable=False)
    inspiration_time_s: Mapped[float | None] = mapped_column(Float)  # Ti
    expiration_time_s: Mapped[float | None] = mapped_column(Float)  # Te
    total_time_s: Mapped[float | None] = mapped_column(Float)  # Ttot
    i_e_ratio: Mapped[float | None] = mapped_column(Float)
    duty_cycle: Mapped[float | None] = mapped_column(Float)  # Ti/Ttot

    # Amplitude
    peak_flow_lpm: Mapped[float | None] = mapped_column(Float)  # peak inspiratory
    peak_exp_flow_lpm: Mapped[float | None] = mapped_column(Float)  # peak expiratory
    tidal_volume_ml: Mapped[float | None] = mapped_column(Float)
    respiratory_rate_rolling: Mapped[float | None] = mapped_column(
        Float
    )  # RR from rolling window

    # Flow shape — time-above-80%-peak (existing ShapeFeatures.flatness_index)
    flatness_index: Mapped[float | None] = mapped_column(Float)
    # NEW: mid-inspiratory flattening = mid-insp flow ÷ peak (different metric)
    mid_insp_flattening: Mapped[float | None] = mapped_column(Float)

    # Flow classification (FlowLimitationClassifier 7-class taxonomy)
    flow_class: Mapped[int | None] = mapped_column(Integer)
    flow_confidence: Mapped[float | None] = mapped_column(Float)

    # Recovery-breath flag (from primary mode's RERA detector)
    is_recovery_breath: Mapped[bool | None] = mapped_column(Boolean)

    # Trigger/cycle inference (experimental, heuristic — non-ResMed → null)
    inferred_trigger_type: Mapped[str | None] = mapped_column(String)
    trigger_confidence: Mapped[float | None] = mapped_column(Float)
    inferred_cycle_type: Mapped[str | None] = mapped_column(String)
    cycle_confidence: Mapped[float | None] = mapped_column(Float)
    # Per-device applicability: null + reason for unvalidated devices
    trigger_cycle_applicable: Mapped[bool | None] = mapped_column(Boolean)
    trigger_cycle_reason: Mapped[str | None] = mapped_column(String)

    # Quality flags (all nullable + reason; see plan step 3)
    leak_valid: Mapped[bool | None] = mapped_column(Boolean)
    leak_valid_reason: Mapped[str | None] = mapped_column(String)
    ramp_active: Mapped[bool | None] = mapped_column(Boolean)
    ramp_active_reason: Mapped[str | None] = mapped_column(String)
    mask_off: Mapped[bool | None] = mapped_column(Boolean)
    mask_off_reason: Mapped[str | None] = mapped_column(String)

    analysis_result = relationship(
        "AnalysisResult",
        back_populates="breaths",
        lazy="raise",
    )
    session = relationship(
        "Session",
        back_populates="breaths",
        lazy="raise",
        overlaps="analysis_result",
    )

    __table_args__ = (
        UniqueConstraint(
            "analysis_result_id",
            "breath_number",
            name="uq_breath_run_number",
        ),
        Index("ix_breath_analysis_start", "analysis_result_id", "start_offset_s"),
        Index("ix_breath_session_id", "session_id"),
        CheckConstraint(
            "flow_class IS NULL OR (flow_class >= 1 AND flow_class <= 7)",
            name="chk_breath_flow_class",
        ),
        CheckConstraint(
            "flow_confidence IS NULL OR (flow_confidence >= 0 AND flow_confidence <= 1)",
            name="chk_breath_flow_confidence",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Breath(id={self.id}, analysis_result_id={self.analysis_result_id}, "
            f"breath_number={self.breath_number})>"
        )


class ImportJobRecord(Base):
    """Persisted record of an import job, updated at each state transition.

    Written at PENDING, RUNNING, and terminal states so the job survives server
    restarts.  Orphaned non-terminal rows are marked failed at startup by
    ``_recover_orphaned_import_jobs``.
    """

    __tablename__ = "import_job_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    job_type: Mapped[str] = mapped_column(String(20), nullable=False)
    # No FK: records must survive user/profile deletion.
    owner_user_id: Mapped[int | None] = mapped_column(Integer)
    target_profile_id: Mapped[int | None] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sessions_imported: Mapped[int | None] = mapped_column(Integer)
    import_result_json: Mapped[dict[str, Any] | None] = mapped_column(
        ValidatedJSON, nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    analysis_queued: Mapped[bool | None] = mapped_column(Boolean)
    spool_dir_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    # None until the job reaches RUNNING; set once when it starts executing.
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    # None for non-terminal states; set when the job reaches terminal state.
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    __table_args__ = (
        Index("ix_import_job_records_owner_user_id", "owner_user_id"),
        Index("ix_import_job_records_user_created", "owner_user_id", "created_at"),
        CheckConstraint(
            "state IN ('pending_upload','pending','running','succeeded','failed','cancelled')",
            name="chk_import_job_record_state",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ImportJobRecord(id={self.id}, job_id={self.job_id}, state={self.state})>"
        )


class AnalysisJobRecord(Base):
    """Persisted record of an analysis job, updated at each state transition.

    Written at RUNNING and terminal states so the job survives server restarts.
    Orphaned non-terminal rows are marked failed at startup.
    """

    __tablename__ = "analysis_job_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    # No FK: records must survive user/profile deletion.
    profile_id: Mapped[int] = mapped_column(Integer, nullable=False)
    owner_user_id: Mapped[int | None] = mapped_column(Integer)
    session_ids_json: Mapped[list[int]] = mapped_column(ValidatedJSON, nullable=False)
    modes: Mapped[list[str] | None] = mapped_column(ValidatedJSON, nullable=True)
    primary_mode: Mapped[str | None] = mapped_column(String(50))
    store_results: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    progress_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    __table_args__ = (
        Index("ix_analysis_job_records_owner_user_id", "owner_user_id"),
        Index("ix_analysis_job_records_profile_state", "profile_id", "state"),
        CheckConstraint(
            "state IN ('running','succeeded','failed','cancelled')",
            name="chk_analysis_job_record_state",
        ),
    )

    def __repr__(self) -> str:
        return f"<AnalysisJobRecord(id={self.id}, job_id={self.job_id}, state={self.state})>"


class ValidationRun(Base):
    """A persisted validator run, comparable across algorithm versions.

    Unlike the ``*_job_records`` durability mirrors, this row *is* the result:
    the background job (or the synchronous POST for sync validator types) writes
    its state and full ``report_json`` directly here — there is no separate
    job-record table.  Sync runs carry ``job_id = NULL``.

    Comparison / dedup key: two runs are the "same" run only when
    ``(profile_id, validator_type, date_from, date_to)`` match AND both
    ``engine_identity_json`` (``AlgorithmIdentity.current()``) and
    ``validator_params_json`` (query-time knobs not covered by the identity)
    are equal.  The identity component alone is insufficient — without the
    params component, a threshold change would silently compare unlike runs.

    ``report_json`` stores the whole Pydantic report blob.  Reports are read
    whole by the UI and are bounded (≤~650 session rows), so there is no
    per-session child table.  This is a reversible decision: a child table can
    be introduced later without changing the run identity or the dedup key.
    """

    __tablename__ = "validation_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # NULL for synchronous runs (computed inline in the POST, never queued).
    job_id: Mapped[str | None] = mapped_column(String(32), unique=True)
    # No FK: runs must survive user/profile deletion for historical comparison.
    profile_id: Mapped[int] = mapped_column(Integer, nullable=False)
    owner_user_id: Mapped[int | None] = mapped_column(Integer)
    validator_type: Mapped[str] = mapped_column(String(20), nullable=False)
    date_from: Mapped[date] = mapped_column(Date, nullable=False)
    date_to: Mapped[date] = mapped_column(Date, nullable=False)
    engine_identity_json: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSON, nullable=False
    )
    validator_params_json: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSON, nullable=False
    )
    # NULL until the run reaches SUCCEEDED; the whole report blob otherwise.
    report_json: Mapped[dict[str, Any] | None] = mapped_column(ValidatedJSON)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    __table_args__ = (
        Index("ix_validation_runs_owner_user_id", "owner_user_id"),
        Index(
            "ix_validation_runs_profile_type_created",
            "profile_id",
            "validator_type",
            "created_at",
        ),
        Index(
            "ix_validation_runs_dedup",
            "profile_id",
            "validator_type",
            "date_from",
            "date_to",
        ),
        CheckConstraint(
            "state IN ('queued','running','succeeded','failed','cancelled')",
            name="chk_validation_run_state",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ValidationRun(id={self.id}, type={self.validator_type}, "
            f"state={self.state})>"
        )


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
    # Device/session wall-clock time: no source timezone, never convert.
    start_time: Mapped[datetime] = mapped_column(DateTime)
    duration: Mapped[float | None] = mapped_column(Float)  # seconds
    confidence: Mapped[float] = mapped_column(Float)
    detected_by: Mapped[str] = mapped_column(String(20))  # programmatic
    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        ValidatedJSONWithDefault, default=dict
    )
    notes: Mapped[str | None] = mapped_column(Text)

    analysis = relationship(
        "AnalysisResult", back_populates="detected_patterns", lazy="raise"
    )

    def __repr__(self) -> str:
        return f"<DetectedPattern(id={self.id}, pattern={self.pattern_id}, confidence={self.confidence})>"
