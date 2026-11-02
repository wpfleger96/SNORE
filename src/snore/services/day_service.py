"""Day aggregate service — queries Day table and returns Pydantic models."""

from __future__ import annotations

import logging

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select

from snore.database import models
from snore.exceptions import NotFoundError
from snore.services._base import ProfileScopedService, paginate
from snore.services.channel_stats import (
    load_channel_samples,
    mask_on_spans,
    pool_channel_stats,
)
from snore.services.event_service import device_reported_types, reported_counts
from snore.services.schemas import (
    DayDetail,
    DayExperimentalMetrics,
    DayListItem,
    HealthNightSummaryRead,
)

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


def _null_fl_rera(day_date: date, reason: str) -> DayExperimentalMetrics:
    """The six FL/RERA fields all null, sharing one companion reason code."""
    return DayExperimentalMetrics(
        date=day_date,
        fl_class_ge4_pct_reason=reason,
        rera_index_reason=reason,
        rera_count_reason=reason,
    )


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

    async def _select_day(self, day_date: date, device_id: int | None) -> models.Day:
        """Return the Day row for a date in the actor's profile.

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

        return rows[0]

    async def get_day(self, day_date: date, device_id: int | None = None) -> DayDetail:
        """Return detailed day record with session IDs (see ``_select_day``)."""
        day = await self._select_day(day_date, device_id)

        session_rows = (
            await self.db_session.execute(
                select(
                    models.Session.id,
                    models.Session.enabled,
                    models.Session.start_time,
                    models.Session.mask_on_segments,
                )
                .where(models.Session.day_id == day.id)
                .order_by(models.Session.start_time)
            )
        ).all()
        # Like the Day aggregates, device detail covers enabled sessions only.
        enabled_rows = [row for row in session_rows if row.enabled]
        enabled_ids = [row.id for row in enabled_rows]
        samples = await load_channel_samples(self.db_session, enabled_ids)
        spans = sorted(
            (
                span
                for row in enabled_rows
                for span in mask_on_spans(row.id, row.start_time, row.mask_on_segments)
                or []
            ),
            key=lambda span: span.start,
        )

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

        reported = await device_reported_types(self.db_session, day.device_id)
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
                **reported_counts(day, reported),
                "session_ids": [row.id for row in session_rows],
                "health_sleep": health_sleep,
                "channel_stats": pool_channel_stats(samples, enabled_ids),
                "mask_on_spans": spans,
            }
        )

    async def get_day_experimental(
        self, day_date: date, device_id: int | None = None
    ) -> DayExperimentalMetrics:
        """Return FL / RERA-proxy metrics for the Day row ``get_day`` would select."""
        day = await self._select_day(day_date, device_id)
        enabled_flags = (
            (
                await self.db_session.execute(
                    select(models.Session.enabled).where(
                        models.Session.day_id == day.id
                    )
                )
            )
            .scalars()
            .all()
        )
        return await self._nightly_fl_rera(
            day_date,
            day.device_id,
            all_sessions_disabled=bool(enabled_flags) and not any(enabled_flags),
        )

    async def _nightly_fl_rera(
        self, day_date: date, device_id: int, *, all_sessions_disabled: bool
    ) -> DayExperimentalMetrics:
        """Read-time flow-limitation / RERA-proxy metrics for one night.

        Sourced from BreathService.get_experimental_nightly_summary — the same
        latest-run aggregation the experimental MCP nightly tool uses — so this
        never recomputes breath logic.

        Two distinct null-with-reason outcomes are possible:

        * An ordinary un-analyzed night (sessions exist, none with an OK
          analysis) returns a valid summary whose FL/RERA reasons are
          ``not_available`` — this is the success path, not an error.
        * A genuine lookup failure (the device has no sessions on the date, a
          breath-table DB error, or device resolution declining) degrades to
          null values with an ``analysis_not_run`` reason, so this never fails
          on missing breath analysis.

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

        not_run = _null_fl_rera(day_date, NullReason.ANALYSIS_NOT_RUN.value)
        if all_sessions_disabled:
            return not_run

        try:
            night = await BreathService(
                self.db_session, self.profile_id
            ).get_experimental_nightly_summary(day_date, device_id=device_id)
        except (DeviceAmbiguityError, DeviceNotOwnedError):
            # Device resolution declined — degrade quietly (expected edge case).
            return not_run
        except SQLAlchemyError:
            logger.warning(
                "FL/RERA nightly lookup for %s hit a breath-table DB error; "
                "returning null metrics",
                day_date,
                exc_info=True,
            )
            return not_run
        except ValueError:
            # Day row exists but the device has no sessions on this date —
            # anomalous, so surface it rather than silently nulling.
            logger.warning(
                "FL/RERA nightly lookup for %s found no analyzable sessions; "
                "returning null metrics",
                day_date,
            )
            return not_run

        return DayExperimentalMetrics(
            date=day_date,
            fl_class_ge4_pct=night.fl_class_ge4_pct,
            fl_class_ge4_pct_reason=_reason_value(night.fl_class_ge4_pct_reason),
            rera_index=night.rera_index,
            rera_index_reason=_reason_value(night.rera_index_reason),
            rera_count=night.rera_count,
            rera_count_reason=_reason_value(night.rera_reason),
        )
