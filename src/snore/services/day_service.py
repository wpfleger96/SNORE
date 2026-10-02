"""Day aggregate service — queries Day table and returns Pydantic models."""

from __future__ import annotations

import logging

from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select

from snore.database import models
from snore.exceptions import NotFoundError
from snore.services._base import ProfileScopedService, paginate
from snore.services.schemas import DayDetail, DayListItem, HealthNightSummaryRead

if TYPE_CHECKING:
    from snore.analysis.shared.versioning import NullReason

logger = logging.getLogger(__name__)

__all__ = ["DayService"]

# DayDetail fields stored on the Day row under the same name; get_day copies
# them by identity and passes only renamed, defaulted, or joined fields.
_DAY_DETAIL_IDENTITY_FIELDS = tuple(
    name
    for name in DayDetail.model_fields
    if name in sa_inspect(models.Day).column_attrs.keys()
)


def _null_fl_rera(reason: str) -> dict[str, Any]:
    """The six FL/RERA fields all null, sharing one companion reason code."""
    return {
        "fl_class_ge4_pct": None,
        "fl_class_ge4_pct_reason": reason,
        "rera_index": None,
        "rera_index_reason": reason,
        "rera_count": None,
        "rera_count_reason": reason,
    }


def _reason_value(reason: NullReason | None) -> str | None:
    """Plain-string form of a NullReason companion, or None when computable."""
    return reason.value if reason is not None else None


class DayService(ProfileScopedService):
    async def list_days(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
        device_id: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[DayListItem], int]:
        """Return paginated list of days with optional filters."""
        query = (
            select(models.Day)
            .join(models.Device, models.Day.device_id == models.Device.id)
            .where(self._profile_filter())
        )

        if from_date is not None:
            query = query.where(models.Day.date >= from_date)
        if to_date is not None:
            query = query.where(models.Day.date <= to_date)
        if device_id is not None:
            query = query.where(models.Day.device_id == device_id)

        result, total = await paginate(
            self.db_session,
            query,
            order_by=models.Day.date.desc(),
            limit=limit,
            offset=offset,
        )
        items = [DayListItem.model_validate(d) for d in result.scalars().all()]
        return items, total

    async def list_dates(self) -> list[date]:
        query = (
            select(models.Day.date)
            .distinct()
            .join(models.Device, models.Day.device_id == models.Device.id)
            .where(self._profile_filter())
            .order_by(models.Day.date)
        )
        rows = (await self.db_session.execute(query)).scalars().all()
        return list(rows)

    async def get_day(self, day_date: date, device_id: int | None = None) -> DayDetail:
        """Return detailed day record with session IDs.

        When multiple devices have Day rows on the same date (e.g. a machine-switch
        date), returns the first row ordered by device_id with a warning rather than
        raising MultipleResultsFound.  Pass device_id to select a specific device.

        Raises NotFoundError if no day exists for this date in the actor's profile.
        """
        stmt = (
            select(models.Day)
            .join(models.Device, models.Day.device_id == models.Device.id)
            .where(self._profile_filter(), models.Day.date == day_date)
        )
        if device_id is not None:
            stmt = stmt.where(models.Day.device_id == device_id)

        stmt = stmt.order_by(models.Day.device_id)

        rows = (await self.db_session.execute(stmt)).scalars().all()

        if not rows:
            if device_id is not None:
                raise NotFoundError(
                    f"No data found for device_id={device_id} on date {day_date}"
                )
            raise NotFoundError(f"No data found for date {day_date}")

        if len(rows) > 1:
            device_ids = [r.device_id for r in rows]
            logger.warning(
                f"Multiple Day rows for date {day_date}: device_ids={device_ids}; "
                f"returning device_id={rows[0].device_id}"
            )

        day = rows[0]

        session_rows = (
            await self.db_session.execute(
                select(models.Session.id, models.Session.enabled)
                .where(models.Session.day_id == day.id)
                .order_by(models.Session.start_time)
            )
        ).all()
        session_ids = [row.id for row in session_rows]

        health_summary_row = (
            await self.db_session.execute(
                select(models.HealthNightlySummary).where(
                    models.HealthNightlySummary.profile_id == self.profile_id,
                    models.HealthNightlySummary.night_date == day_date,
                )
            )
        ).scalar_one_or_none()

        health_sleep = (
            HealthNightSummaryRead.model_validate(health_summary_row)
            if health_summary_row is not None
            else None
        )

        fl_rera = await self._nightly_fl_rera(
            day_date,
            day.device_id,
            all_sessions_disabled=bool(session_rows)
            and not any(row.enabled for row in session_rows),
        )

        identity = {name: getattr(day, name) for name in _DAY_DETAIL_IDENTITY_FIELDS}
        return DayDetail.model_validate(
            {
                **identity,
                "avg_pressure": day.pressure_median,
                "avg_leak": day.leak_median,
                "avg_spo2": day.spo2_mean,
                "obstructive_apneas": day.obstructive_apneas or 0,
                "central_apneas": day.central_apneas or 0,
                "hypopneas": day.hypopneas or 0,
                "reras": day.reras or 0,
                "session_ids": session_ids,
                "health_sleep": health_sleep,
                **fl_rera,
            }
        )

    async def _nightly_fl_rera(
        self, day_date: date, device_id: int, *, all_sessions_disabled: bool
    ) -> dict[str, Any]:
        """Read-time flow-limitation / RERA-proxy metrics for one night.

        Sourced from BreathService.get_nightly_summary — the same latest-run
        aggregation the MCP nightly summary uses — so day detail never
        recomputes breath logic.  Waveform stats are skipped
        (``include_waveform_stats=False``): the FL/RERA fields come from the
        breath table alone, and loading the full-night waveform blobs on every
        ``GET /days/{date}`` is pure overhead here.

        Two distinct null-with-reason outcomes are possible:

        * An ordinary un-analyzed night (sessions exist, none with an OK
          analysis) returns a valid summary whose FL/RERA reasons are
          ``not_available`` — this is the success path, not an error.
        * A genuine lookup failure (the device has no sessions on the date, a
          breath-table DB error, or device resolution declining) degrades to
          null values with an ``analysis_not_run`` reason, so day detail never
          fails on missing breath analysis.

        A night whose sessions are all disabled (``all_sessions_disabled``) is
        a normal user choice the breath service skips, so it short-circuits to
        the same ``analysis_not_run`` nulls without the anomaly warning; a Day
        row with no sessions at all still warns.
        """
        from sqlalchemy.exc import SQLAlchemyError  # noqa: PLC0415

        from snore.analysis.shared.versioning import NullReason  # noqa: PLC0415
        from snore.services.breath_service import (  # noqa: PLC0415
            BreathService,
            DeviceAmbiguityError,
            DeviceNotOwnedError,
        )

        if all_sessions_disabled:
            return _null_fl_rera(NullReason.ANALYSIS_NOT_RUN.value)

        try:
            night = await BreathService(
                self.db_session, self.profile_id
            ).get_nightly_summary(
                day_date, device_id=device_id, include_waveform_stats=False
            )
        except (DeviceAmbiguityError, DeviceNotOwnedError):
            # Device resolution declined — degrade quietly (expected edge case).
            return _null_fl_rera(NullReason.ANALYSIS_NOT_RUN.value)
        except SQLAlchemyError:
            logger.warning(
                "FL/RERA nightly lookup for %s hit a breath-table DB error; "
                "returning null metrics",
                day_date,
                exc_info=True,
            )
            return _null_fl_rera(NullReason.ANALYSIS_NOT_RUN.value)
        except ValueError:
            # Day row exists but the device has no sessions on this date —
            # anomalous, so surface it rather than silently nulling.
            logger.warning(
                "FL/RERA nightly lookup for %s found no analyzable sessions; "
                "returning null metrics",
                day_date,
            )
            return _null_fl_rera(NullReason.ANALYSIS_NOT_RUN.value)

        return {
            "fl_class_ge4_pct": night.fl_class_ge4_pct,
            "fl_class_ge4_pct_reason": _reason_value(night.fl_class_ge4_pct_reason),
            "rera_index": night.rera_index,
            "rera_index_reason": _reason_value(night.rera_index_reason),
            "rera_count": night.rera_count,
            "rera_count_reason": _reason_value(night.rera_reason),
        }
