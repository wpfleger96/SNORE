from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Literal, cast

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    StringConstraints,
    model_validator,
)

from snore.provenance import Provenance, provenance_field
from snore.services.schemas import (
    DayDetail,
    DayListItem,
    HealthNightDetailRead,
    HealthNightSummaryRead,
    HealthSampleRead,
    ImportSource,
    MaskLogEntryResponse,
    RxAllResponse,
    RxChangesResponse,
    RxComparisonResponse,
    RxPeriodResponse,
    RxSettingChange,
)

DISPLAY_NAME_MAX_LEN: int = 150

__all__ = [
    "DISPLAY_NAME_MAX_LEN",
    "MessageResponse",
    "AnalysisMode",
    "PaginatedResponse",
    "WaveformDataResponse",
    "SessionEnabledRequest",
    "SessionDeleteRequest",
    "BulkDeletePreviewRequest",
    "AnalysisRunRequest",
    "AnalysisDeleteRequest",
    "BatchAnalysisRequest",
    "ValidationRequest",
    "DateRangeValidationRequest",
    "FlValidationRequest",
    "BreathTrendsValidationRequest",
    "ReraValidationRequest",
    "EventItem",
    "DayDetail",
    "DayListItem",
    "RxAllResponse",
    "RxPeriodResponse",
    "RxComparisonResponse",
    "RxSettingChange",
    "RxChangesResponse",
    "MaskLogEntryResponse",
    "MaskLogCreateRequest",
    "MaskLogUpdateRequest",
    "AnalysisJobStatus",
    "AnalysisJobsListResponse",
    "AnalysisJobEnqueued",
    "ValidatorType",
    "ValidationRunRequest",
    "ValidationRunStatus",
    "ValidationRunDetail",
    "ValidationRunsListResponse",
    "ImportSourceResultSummary",
    "ImportResultSummary",
    "LinkedAnalysisSummary",
    "HealthImportResultSummary",
    "PipelineJobStatus",
    "PipelineJobsListResponse",
    "DateListResponse",
    # Re-exported Apple Health read schemas
    "HealthNightSummaryRead",
    "HealthNightDetailRead",
    "HealthSampleRead",
]


class MessageResponse(BaseModel):
    message: str


class PaginatedResponse[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int


class WaveformDataResponse(BaseModel):
    timestamps: list[float]
    values: list[float] = provenance_field(
        Provenance.DEVICE, "Waveform samples (possibly LTTB-downsampled)"
    )
    sample_rate: float = provenance_field(Provenance.DEVICE, "Sample rate (Hz)")
    unit: str
    total_samples: int
    downsampled: bool
    returned_samples: int


class SessionEnabledRequest(BaseModel):
    enabled: bool


class SessionDeleteRequest(BaseModel):
    session_ids: list[int]


class BulkDeletePreviewRequest(BaseModel):
    session_ids: list[int] | None = None
    device: str | None = None
    from_date: date | None = None
    to_date: date | None = None
    delete_all: bool = False


class AnalysisRunRequest(BaseModel):
    modes: list[str] = Field(default_factory=lambda: ["aasm"])
    primary_mode: str | None = Field(
        default=None,
        description=(
            "Mode whose recovery markers are persisted. "
            "Must be a member of `modes` when supplied; "
            "defaults to 'aasm' when 'aasm' is in modes, required otherwise."
        ),
    )
    store_results: bool = True


class AnalysisDeleteRequest(BaseModel):
    session_ids: list[int] = Field(default_factory=list)
    all_versions: bool = False


class EventItem(BaseModel):
    id: int
    event_type: str
    start_time: float
    duration_seconds: float = provenance_field(
        Provenance.DEVICE, "Device-scored event duration (s)"
    )
    offset_seconds: float
    spo2_drop: float | None = provenance_field(
        Provenance.DEVICE, "SpO2 drop (%)", default=None
    )
    peak_flow_limitation: float | None = provenance_field(
        Provenance.DEVICE, "Peak flow limitation", default=None
    )


AnalysisMode = Literal["aasm", "aasm_relaxed", "resmed"]


class BatchAnalysisRequest(BaseModel):
    from_date: date | None = None
    to_date: date | None = None
    missing_only: bool = Field(
        default=False,
        description=(
            "When true, restrict the batch to sessions that have a flow waveform "
            "but no analysis result yet. Composable with from_date/to_date. "
            "When true, from_date and to_date are not required."
        ),
    )
    modes: list[AnalysisMode] = Field(
        default_factory=lambda: cast(list[AnalysisMode], ["aasm"])
    )
    primary_mode: str | None = Field(
        default=None,
        description=(
            "Mode whose recovery markers are persisted. "
            "Must be a member of `modes` when supplied; "
            "defaults to 'aasm' when 'aasm' is in modes, required otherwise."
        ),
    )
    store_results: bool = True


class ValidationRequest(BaseModel):
    from_date: date
    to_date: date
    mode: AnalysisMode = "aasm"


class DateRangeValidationRequest(BaseModel):
    """Shared base for signal-validation requests over a ``[from_date, to_date]`` range."""

    from_date: date
    to_date: date

    @model_validator(mode="after")
    def validate_date_order(self) -> DateRangeValidationRequest:
        if self.to_date < self.from_date:
            raise ValueError("to_date must be >= from_date")
        return self


class FlValidationRequest(DateRangeValidationRequest):
    pass


class BreathTrendsValidationRequest(DateRangeValidationRequest):
    pass


class ReraValidationRequest(DateRangeValidationRequest):
    pass


# Apple cross-validation walks its calendar span in 90-night pages in the
# request path, so an unbounded span means thousands of sequential queries.
# One year of nights is far beyond any real Apple Watch export horizon.
_APPLE_CROSS_MAX_SPAN_NIGHTS = 366


class AppleCrossValidationRequest(BaseModel):
    from_date: date
    to_date: date
    device_id: int | None = None

    @model_validator(mode="after")
    def validate_date_order(self) -> AppleCrossValidationRequest:
        if self.to_date < self.from_date:
            raise ValueError("to_date must be >= from_date")
        span_nights = (self.to_date - self.from_date).days + 1
        if span_nights > _APPLE_CROSS_MAX_SPAN_NIGHTS:
            raise ValueError(
                f"Date range spans {span_nights} nights; the maximum is "
                f"{_APPLE_CROSS_MAX_SPAN_NIGHTS}. Narrow the range."
            )
        return self


# Style vocabulary must stay in sync: DB CHECKs (models.py, migrations 008/009), services/mask_epoch_service.py map, ui/src/utils/maskOptions.ts.
MaskStyle = Literal["pillows", "nasal", "full_face"]

_MASK_START_DATE_MIN = date(2000, 1, 1)
_MASK_START_DATE_MAX_FUTURE_DAYS = 366


def _validate_plausible_start_date(value: date) -> date:
    """Reject start dates outside the CPAP-era clinical window."""
    if value < _MASK_START_DATE_MIN:
        raise ValueError("start_date must be on or after 2000-01-01")
    if value > date.today() + timedelta(days=_MASK_START_DATE_MAX_FUTURE_DAYS):
        raise ValueError("start_date must not be more than 366 days in the future")
    return value


PlausibleStartDate = Annotated[date, AfterValidator(_validate_plausible_start_date)]


class _MaskLogFields(BaseModel):
    """Shared optional field set for mask log request bodies."""

    model_config = ConfigDict(extra="forbid")

    brand: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
        ]
        | None
    ) = None
    model: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=150)
        ]
        | None
    ) = None
    style: MaskStyle | None = None
    start_date: PlausibleStartDate | None = None
    size: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)
        ]
        | None
    ) = None
    notes: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
        ]
        | None
    ) = None


class MaskLogCreateRequest(_MaskLogFields):
    """POST body: all fields are optional — an entirely empty create is accepted."""


class MaskLogUpdateRequest(_MaskLogFields):
    """PATCH body: omitted fields are unchanged; explicit null clears any field."""


def _normalize_response_timestamp(value: datetime) -> datetime:
    """Normalize an aware response timestamp to UTC."""
    return value.astimezone(UTC)


UTCResponseTimestamp = Annotated[
    AwareDatetime,
    Strict(),
    AfterValidator(_normalize_response_timestamp),
]


class AnalysisJobStatus(BaseModel):
    """Snapshot of one analysis job.

    created_at, started_at, and finished_at are tz-aware UTC datetimes
    serialized as ISO 8601 strings.
    """

    job_id: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    source: str
    session_count: int
    progress_completed: int
    progress_total: int
    error_message: str | None
    created_at: UTCResponseTimestamp
    started_at: UTCResponseTimestamp | None
    finished_at: UTCResponseTimestamp | None
    owner_user_id: int | None


class AnalysisJobsListResponse(BaseModel):
    jobs: list[AnalysisJobStatus]


class AnalysisJobEnqueued(BaseModel):
    job_id: str
    session_count: int


# Accepted validator types. events/fl/breaths run today; rera/apple are accepted
# here but rejected with 400 until their sibling validators are registered.
ValidatorType = Literal["events", "fl", "breaths", "rera", "apple"]


class ValidationRunRequest(BaseModel):
    validator_type: ValidatorType
    from_date: date
    to_date: date
    # Query-time knobs (e.g. {"mode": "resmed"} for events); interpreted per
    # validator type. Combined with the engine identity to form the dedup key.
    params: dict[str, object] | None = None
    # Force a fresh run even when a matching succeeded run already exists.
    force: bool = False

    @model_validator(mode="after")
    def validate_date_order(self) -> ValidationRunRequest:
        if self.to_date < self.from_date:
            raise ValueError("to_date must be >= from_date")
        return self


class ValidationRunStatus(BaseModel):
    """Snapshot of one validation run.

    created_at, started_at, and finished_at are tz-aware UTC datetimes
    serialized as ISO 8601 strings.
    """

    run_id: int
    job_id: str | None
    validator_type: str
    date_from: str
    date_to: str
    state: str
    error_message: str | None
    engine_identity: dict[str, object]
    validator_params: dict[str, object]
    owner_user_id: int | None
    created_at: UTCResponseTimestamp
    started_at: UTCResponseTimestamp | None
    finished_at: UTCResponseTimestamp | None
    # True when this run was returned by dedup instead of computed anew.
    reused: bool = False


class ValidationRunDetail(ValidationRunStatus):
    # The whole report blob; None until the run has succeeded.
    report_json: dict[str, object] | None


class ValidationRunsListResponse(BaseModel):
    runs: list[ValidationRunStatus]
    total: int
    limit: int
    offset: int


class ImportSourceResultSummary(BaseModel):
    source: ImportSource
    imported: int
    skipped: int
    failed: int
    warnings: list[str]


class ImportResultSummary(BaseModel):
    """Trimmed ImportResult excluding imported_session_ids (poll-bandwidth)."""

    total_imported: int
    total_skipped: int
    total_failed: int
    warnings: list[str]
    sources: list[ImportSourceResultSummary]


class LinkedAnalysisSummary(BaseModel):
    job_id: str
    state: str
    progress_completed: int
    progress_total: int
    error_message: str | None


class HealthImportResultSummary(BaseModel):
    """Summary of an Apple Health import result attached to a pipeline job."""

    inserted: int
    skipped: int
    nights_recomputed: int


class PipelineJobStatus(BaseModel):
    """Stitched view of one import job and its downstream analysis job.

    created_at, started_at, and finished_at are tz-aware UTC datetimes
    serialized as ISO 8601 strings.
    """

    job_id: str
    job_type: str
    state: str
    stage: str
    file_count: int
    created_at: UTCResponseTimestamp
    started_at: UTCResponseTimestamp | None
    finished_at: UTCResponseTimestamp | None
    progress_message: str | None
    sessions_imported: int | None
    import_result: ImportResultSummary | None
    health_import_result: HealthImportResultSummary | None = None
    error_message: str | None
    analysis_job_id: str | None
    analysis_queued: bool | None
    linked_analysis: LinkedAnalysisSummary | None


class PipelineJobsListResponse(BaseModel):
    jobs: list[PipelineJobStatus]


class DateListResponse(BaseModel):
    dates: list[date]
