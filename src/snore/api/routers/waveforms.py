from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from snore.api.deps import service_dep
from snore.api.schemas import WaveformDataResponse
from snore.services import WaveformService
from snore.services.schemas import EventComparisonResult, WaveformInfo

router = APIRouter()

WaveformServiceDep = Annotated[WaveformService, Depends(service_dep(WaveformService))]

VALID_WAVEFORM_TYPES = Literal[
    "flow",
    "pressure",
    "therapy_pressure",
    "epap",
    "leak",
    "mv",
    "rr",
    "tv",
    "spo2",
    "pulse",
    "fl",
    "snore",
    "ie_ratio",
    "ti",
    "pressure_hr",
    "trigger_cycle",
]


@router.get("/{session_id}/waveforms", response_model=list[WaveformInfo])
async def list_waveforms(
    session_id: int, service: WaveformServiceDep
) -> list[WaveformInfo]:
    return await service.list_waveforms(session_id)


@router.get("/{session_id}/waveforms/compare", response_model=EventComparisonResult)
async def compare_waveform_events(
    session_id: int,
    service: WaveformServiceDep,
    mode: Literal["aasm", "aasm_relaxed", "resmed"] = Query(default="aasm"),
) -> EventComparisonResult:
    return await service.compare_events(session_id, mode=mode)


@router.get(
    "/{session_id}/waveforms/{waveform_type}", response_model=WaveformDataResponse
)
async def get_waveform(
    session_id: int,
    waveform_type: VALID_WAVEFORM_TYPES,
    service: WaveformServiceDep,
    max_points: int = Query(default=2000, ge=100, le=10000),
    start_seconds: float | None = Query(default=None, ge=0, allow_inf_nan=False),
    end_seconds: float | None = Query(default=None, ge=0, allow_inf_nan=False),
) -> WaveformDataResponse:
    timestamps, values, metadata = await service.get_waveform_data(
        session_id=session_id,
        waveform_type=waveform_type,
        max_points=max_points,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    total_samples = metadata["sample_count"]
    returned = len(timestamps)

    return WaveformDataResponse(
        timestamps=timestamps.tolist(),
        values=values.tolist(),
        sample_rate=metadata.get("sample_rate", 0.0),
        unit=metadata.get("unit", "") or "",
        total_samples=total_samples,
        downsampled=returned < total_samples,
        returned_samples=returned,
    )
