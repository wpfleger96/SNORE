"""BreathService — query layer over the breaths table.

This module is the single home for the typed seam contract consumed by the MCP
tool layer: every DTO, enum, and seam function the tool adapters depend on is
defined (or re-exported) here, so adapters import one surface and never reach
into ORM models or analysis internals.

The implementation lives in the ``snore.services.breath`` package; this module
is a pure re-export shim kept for import-path stability.
"""

from __future__ import annotations

from snore.services.breath import (
    CROSS_VERSION_REFUSAL_KEYS,
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisRunMetadata,
    AnalysisStatus,
    BreathBin,
    BreathPage,
    BreathQueryRange,
    BreathRow,
    BreathService,
    CompareEpochsResult,
    ContextualEvent,
    CycleType,
    DayAnalysisStatus,
    DeviceAmbiguityError,
    DeviceCapabilities,
    DeviceNotOwnedError,
    DistributionMetric,
    DistributionStats,
    EpochBreathStats,
    EpochRequest,
    EpochRxViolation,
    FindWindowsResult,
    MultiSessionAmbiguityError,
    MvSource,
    NightlyAnalysisSummary,
    NightlyRangeSummary,
    NoSessionsInRangeError,
    NullReason,
    RawWaveformChannel,
    RawWaveformWindow,
    SessionCoverage,
    SessionSummary,
    TimezoneStatus,
    TriggerCycleApplicability,
    TriggerType,
    VentilatoryContext,
    WaveformChannel,
    WaveformChannelName,
    WaveformWindow,
    WaveformWindowRequest,
    WindowCriterion,
    WindowCriterionOptions,
    WindowResult,
    compute_ventilatory_context,
    compute_waveform_window,
    derive_mv_from_flow,
    fetch_waveform_window_raw,
)
from snore.services.breath import (
    _count_fl_run_reras as _count_fl_run_reras,
)
from snore.services.breath import (
    _fetch_waveform_blobs as _fetch_waveform_blobs,
)
from snore.services.breath import (
    iter_fl_run_recoveries as iter_fl_run_recoveries,
)

__all__ = [
    # Enums / shared types
    "TimezoneStatus",
    "NullReason",
    "AnalysisStatus",
    "DayAnalysisStatus",
    "AlgorithmIdentity",
    "AnalysisRunMetadata",
    "AlgoVersions",
    "CROSS_VERSION_REFUSAL_KEYS",
    "TriggerType",
    "CycleType",
    "TriggerCycleApplicability",
    "WaveformChannelName",
    # DTOs
    "SessionCoverage",
    "BreathQueryRange",
    "SessionSummary",
    "MultiSessionAmbiguityError",
    "DeviceAmbiguityError",
    "DeviceNotOwnedError",
    "NoSessionsInRangeError",
    "BreathRow",
    "BreathBin",
    "BreathPage",
    "WindowCriterion",
    "WindowCriterionOptions",
    "WindowResult",
    "FindWindowsResult",
    "EpochRequest",
    "DistributionMetric",
    "DistributionStats",
    "EpochRxViolation",
    "EpochBreathStats",
    "CompareEpochsResult",
    "ContextualEvent",
    "RawWaveformChannel",
    "RawWaveformWindow",
    "WaveformChannel",
    "WaveformWindow",
    "WaveformWindowRequest",
    "NightlyAnalysisSummary",
    "NightlyRangeSummary",
    "DeviceCapabilities",
    "MvSource",
    "VentilatoryContext",
    # Functions
    "fetch_waveform_window_raw",
    "compute_waveform_window",
    "compute_ventilatory_context",
    "derive_mv_from_flow",
    # Service
    "BreathService",
]
