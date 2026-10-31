from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from snore.api.deps import DateRangeParams, PaginationParams, service_dep
from snore.api.schemas import (
    DateListResponse,
    DayDetail,
    DayExperimentalMetrics,
    DayListItem,
    PaginatedResponse,
)
from snore.services import DayService

router = APIRouter()

DayServiceDep = Annotated[DayService, Depends(service_dep(DayService))]


@router.get("/", response_model=PaginatedResponse[DayListItem])
async def list_days(
    service: DayServiceDep,
    pagination: PaginationParams = Depends(),
    dates: DateRangeParams = Depends(),
    device_id: int | None = Query(default=None),
) -> PaginatedResponse[DayListItem]:
    items, total = await service.list_days(
        from_date=dates.from_date,
        to_date=dates.to_date,
        device_id=device_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return PaginatedResponse(
        items=items, total=total, limit=pagination.limit, offset=pagination.offset
    )


@router.get("/dates", response_model=DateListResponse)
async def list_dates(service: DayServiceDep) -> DateListResponse:
    dates = await service.list_dates()
    return DateListResponse(dates=dates)


@router.get("/{day_date}", response_model=DayDetail)
async def get_day(
    day_date: date,
    service: DayServiceDep,
    device_id: int | None = Query(default=None),
) -> DayDetail:
    return await service.get_day(day_date, device_id=device_id)


@router.get(
    "/{day_date}/experimental",
    response_model=DayExperimentalMetrics,
    tags=["experimental"],
)
async def get_day_experimental(
    day_date: date,
    service: DayServiceDep,
    device_id: int | None = Query(default=None),
) -> DayExperimentalMetrics:
    return await service.get_day_experimental(day_date, device_id=device_id)
