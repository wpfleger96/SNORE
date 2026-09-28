"""Breath service package — implementation behind ``snore.services.breath_service``.

The public import surface is the ``snore.services.breath_service`` shim; this
package holds the split implementation (DTOs, pure algorithms, and the
``BreathService`` method groups as mixins over ``_BreathServiceCore``).
"""

from __future__ import annotations

from snore.analysis.shared.versioning import (
    CROSS_VERSION_REFUSAL_KEYS,
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisRunMetadata,
    AnalysisStatus,
    DayAnalysisStatus,
    NullReason,
    TimezoneStatus,
)

from ._core import _resolve_timezone as _resolve_timezone
from .algorithms import (
    _count_fl_run_reras as _count_fl_run_reras,
)
from .algorithms import (
    compute_ventilatory_context,
    compute_waveform_window,
    derive_mv_from_flow,
)
from .algorithms import (
    iter_fl_run_recoveries as iter_fl_run_recoveries,
)
from .capabilities import CapabilitiesMixin
from .dtos import (
    BreathBin,
    BreathPage,
    BreathQueryRange,
    BreathRow,
    CompareEpochsResult,
    ContextualEvent,
    CycleType,
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
    RawWaveformChannel,
    RawWaveformWindow,
    SessionCoverage,
    SessionSummary,
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
)
from .epochs import EpochsMixin
from .nightly import NightlyMixin
from .table import TableMixin
from .waveform_io import (
    WaveformMixin,
    fetch_waveform_window_raw,
)
from .waveform_io import (
    _fetch_waveform_blobs as _fetch_waveform_blobs,
)
from .windows import WindowsMixin

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


class BreathService(
    TableMixin,
    WindowsMixin,
    EpochsMixin,
    NightlyMixin,
    CapabilitiesMixin,
    WaveformMixin,
):
    """Query layer over the breaths table. All methods are async.

    Every public method enforces profile ownership: all Session/Device/Day
    queries join through ``Device.profile_id == self._profile_id`` so that
    foreign-profile data is never returned.
    """
