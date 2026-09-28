"""Algorithm versioning types and version constants.

Binding typed contracts for how analysis runs are versioned, stored, and
compared for staleness.

The stored engine_versions_json shape is the nested AlgoVersions composition:

    {
        "identity": AlgorithmIdentity.model_dump(),   # §1
        "run":      AnalysisRunMetadata.model_dump(),  # §1
    }

Legacy flat rows (format_version / modes at top level) are detected by absence
of the "identity" key and treated as STALE_VERSION — no conversion code because
fresh-DB / reimport is the mandated upgrade path.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator

from snore.analysis.modes.config import AVAILABLE_CONFIGS

# ---------------------------------------------------------------------------
# Versioned algorithm constants
# Each constant is stamped into AlgorithmIdentity; bumping any value makes
# prior rows stale and prevents cross-version aggregation.
# ---------------------------------------------------------------------------

SEGMENTER_ALGO_VERSION: str = "v1"
FL_CLASSIFIER_ALGO_VERSION: str = "v2"
FLATTENING_ALGO_VERSION: str = "v2"  # mid-insp flattening
TRIGGER_CYCLE_ALGO_VERSION: str = "v1"  # trigger/cycle heuristic (new, experimental)
LEAK_VALID_ALGO: str = "v1"
# v2: hypopnea-contained breaths are now excluded from downstream baselines, so
# RERA/recovery outputs shift for sessions with hypopneas. No dedicated
# apnea/hypopnea event-detector identity key exists; the in_event marking is
# applied after hypopnea merging, and detect_events resets in_event at entry so
# each mode's detection is self-contained. AHI and persisted apneas/hypopneas are
# therefore unchanged regardless of mode order — only recovery/RERA outputs move,
# which this key covers.
RECOVERY_DETECTOR_ALGO_VERSION: str = "v2"
# ramp_active / mask_off breath validity flags: settings-driven timed ramp
# heuristic + persisted mask-on-segment gap overlap.
VALIDITY_FLAGS_ALGO_VERSION: str = "v1"

# Query-time RERA-proxy criterion (FL runs ending in recovery). NOT part of
# AlgorithmIdentity — it labels query-time derivation only. v2 adds the
# self-contained recovery criterion (class drop to <=2 + peak-flow margin over
# the run mean) alongside the analysis-time recovery flag.
RERA_PROXY_ALGO_VERSION: str = "v2"

# Threshold used by leak_valid derivation (v1).
LEAK_VALID_THRESHOLD_LPM: float = 24.0

# Maximum gap (seconds) for nearest-neighbour leak alignment.
LEAK_VALID_MAX_ALIGNMENT_GAP_S: float = 5.0


# ---------------------------------------------------------------------------
# AlgorithmIdentity — the stable "what algorithm" fingerprint
# ---------------------------------------------------------------------------


class AlgorithmIdentity(BaseModel):
    """Stable algorithm fingerprint persisted with each analysis run.

    Compares structurally.  All query-driving features are included so that
    bumping any version string marks old rows as STALE_VERSION instead of
    silently accepting them.

    Note: run-specific metadata (primary_mode, modes) lives in
    AnalysisRunMetadata — not here.
    """

    # 4 = validity_flags stamped; 3 = PR-A nested format; 2 = flat legacy rows.
    # The bump to 4 is mandatory: pydantic validation back-fills missing fields
    # from defaults, so without it legacy rows lacking validity_flags would
    # compare equal to the current identity and silently stay OK.
    #
    # RELEASE NOTE: bumping format_version marks every stored AnalysisResult
    # STALE_VERSION — the next batch analysis re-processes all sessions.
    # Call this out explicitly in release notes whenever format_version is bumped.
    format_version: int = 4
    segmenter: str = SEGMENTER_ALGO_VERSION
    fl_classifier: str = FL_CLASSIFIER_ALGO_VERSION
    flattening: str = FLATTENING_ALGO_VERSION
    trigger_cycle: str = TRIGGER_CYCLE_ALGO_VERSION
    leak_valid: str = LEAK_VALID_ALGO
    recovery_detector: str = RECOVERY_DETECTOR_ALGO_VERSION
    validity_flags: str = VALIDITY_FLAGS_ALGO_VERSION

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlgorithmIdentity):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    @classmethod
    def current(cls) -> AlgorithmIdentity:
        """Return the identity for the current algorithm versions."""
        return cls()


# Fields whose mismatch triggers different behavior per call site: find_windows
# hard-refuses for all criteria except ca_centered; compare_epochs emits
# non-blocking version_warnings (distributions are still computed).
# trigger_cycle and validity_flags are intentionally excluded: neither feeds a
# cross-epoch distribution (trigger/cycle labels are per-breath experimental
# metadata; ramp_active/mask_off validity flags gate rows, not aggregates), so
# a version bump in either need not refuse comparisons.  A solo bump of an
# excluded key MUST be accompanied by a format_version bump so old rows still
# go stale — format_version is in this set and catches it.
CROSS_VERSION_REFUSAL_KEYS: frozenset[str] = frozenset(
    {
        "format_version",
        "segmenter",
        "fl_classifier",
        "flattening",
        "leak_valid",
        "recovery_detector",
    }
)


# ---------------------------------------------------------------------------
# AnalysisRunMetadata — per-invocation context
# ---------------------------------------------------------------------------


class AnalysisRunMetadata(BaseModel):
    """Per-invocation metadata recorded alongside AlgorithmIdentity.

    primary_mode: the single mode whose recovery markers are persisted.
    modes: all modes that were run in this invocation.
    """

    primary_mode: str
    modes: list[str]

    @model_validator(mode="after")
    def _validate(self) -> AnalysisRunMetadata:
        if self.primary_mode not in AVAILABLE_CONFIGS:
            raise ValueError(
                f"primary_mode {self.primary_mode!r} is not a recognised mode. "
                f"Available: {list(AVAILABLE_CONFIGS)}"
            )
        if self.primary_mode not in self.modes:
            raise ValueError(
                f"primary_mode {self.primary_mode!r} must be a member of modes {self.modes}"
            )
        return self


# ---------------------------------------------------------------------------
# AlgoVersions — the composite stored in engine_versions_json
# ---------------------------------------------------------------------------


class AlgoVersions(BaseModel):
    """The composite stored in AnalysisResult.engine_versions_json (PR-A format).

    Validates that the stored JSON contains both the 'identity' and 'run' keys.
    Legacy flat rows (no 'identity' key) do NOT parse here — callers detect them
    by attempting model_validate and catching ValidationError.
    """

    identity: AlgorithmIdentity
    run: AnalysisRunMetadata

    @classmethod
    def from_stored(cls, raw: dict[str, Any]) -> AlgoVersions | None:
        """Parse stored engine_versions_json.  Returns None for legacy/malformed rows."""
        if not isinstance(raw, dict) or "identity" not in raw:
            return None  # legacy flat row or None/non-dict input → STALE_VERSION
        try:
            return cls.model_validate(raw)
        except Exception:
            return None  # nested but unparseable → classify as stale


# ---------------------------------------------------------------------------
# AnalysisStatus / DayAnalysisStatus / NullReason (§1, §10)
# ---------------------------------------------------------------------------


class AnalysisStatus(StrEnum):
    """Single-session analysis status."""

    OK = "ok"
    NOT_RUN = "not_run"
    STALE_VERSION = "stale_version"


class DayAnalysisStatus(StrEnum):
    """Day-level aggregated analysis status (§10 coverage policy)."""

    OK = "ok"
    PARTIAL = "partial"
    NOT_RUN = "not_run"
    STALE = "stale"
    MIXED_VERSION = "mixed_version"


class NullReason(StrEnum):
    """Codes explaining why an optional field is null."""

    ANALYSIS_NOT_RUN = "analysis_not_run"
    ANALYSIS_STALE = "analysis_stale"
    ALGO_VERSION_MISMATCH = "algo_version_mismatch"
    PRIMARY_MODE_MISMATCH = "primary_mode_mismatch"
    CHANNEL_ABSENT = "channel_absent"
    CHANNEL_UNALIGNED = "channel_unaligned"
    NOT_AVAILABLE = "not_available"
    SMART_RAMP_INDETERMINATE = "smart_ramp_indeterminate"
    SEGMENTS_UNKNOWN = "segments_unknown"
    DURATION_ZERO = "duration_zero"
    NO_DATA_IN_RANGE = "no_data_in_range"
    MULTI_SESSION_AMBIGUITY = "multi_session_ambiguity"
    UNVALIDATED_DEVICE = "unvalidated_device"
    TABLE_MISSING = "table_missing"
    RX_CHANGED_WITHIN_EPOCH = "rx_changed_within_epoch"
    NO_SESSIONS = "no_sessions"


class TimezoneStatus(StrEnum):
    """Timestamp timezone classification (A6)."""

    UTC = "utc"
    UNKNOWN = "unknown"
    # Profile-level user-declared IANA timezone applies (timestamps stay naive;
    # the companion timezone_name field carries the declared zone).
    USER_DECLARED = "user_declared"
