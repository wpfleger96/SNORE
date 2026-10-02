"""Health service — reads Apple Health tables for the authenticated profile."""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

from sqlalchemy import case, func, select

from snore.database import models
from snore.exceptions import NotFoundError
from snore.parsers.apple_health.type_handlers import SLEEP_TYPE
from snore.services._base import ProfileScopedService
from snore.services.schemas import (
    HealthNightDetailRead,
    HealthNightSummaryRead,
    HealthSampleRead,
)

__all__ = ["HealthService", "NightFragmentation"]


class NightFragmentation(NamedTuple):
    """Watch-derived sleep-fragmentation components for one night.

    Either component may be ``None`` when the summary lacked stage data.
    """

    awake_seconds: float | None
    sleep_efficiency_pct: float | None


SPO2_RECORD_TYPE = "HKQuantityTypeIdentifierOxygenSaturation"
_RR_RECORD_TYPE = "HKQuantityTypeIdentifierRespiratoryRate"
_BREATHING_DISTURBANCE_RECORD_TYPE = (
    "HKQuantityTypeIdentifierAppleSleepingBreathingDisturbances"
)


# Plausible SpO₂ range in percent; samples outside it (after fraction→percent
# conversion) are treated as sensor or encoding errors and dropped.
_SPO2_MIN_PCT = 50.0
_SPO2_MAX_PCT = 100.0


def spo2_display_pct(value: float) -> float:
    """Return a stored SpO2 value on the percent scale for display.

    Apple Health sources store SpO2 as either a fraction (0.95) or a percent
    (95). Fractions are scaled to percent; anything else is returned as stored
    so implausible samples stay visible rather than hidden. The fraction test
    mirrors the first branch of the ``spo2_pct`` case in ``get_night_detail``.
    """
    return value * 100 if _SPO2_MIN_PCT <= value * 100 <= _SPO2_MAX_PCT else value


class HealthService(ProfileScopedService):
    async def list_nights(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[HealthNightSummaryRead], int]:
        """Return paginated nightly summaries for the profile, most-recent first."""
        query = select(models.HealthNightlySummary).where(
            models.HealthNightlySummary.profile_id == self.profile_id
        )
        if from_date is not None:
            query = query.where(models.HealthNightlySummary.night_date >= from_date)
        if to_date is not None:
            query = query.where(models.HealthNightlySummary.night_date <= to_date)

        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db_session.execute(count_query)).scalar_one()

        query = query.order_by(models.HealthNightlySummary.night_date.desc())
        if limit > 0:
            query = query.limit(limit)
        query = query.offset(offset)

        rows = (await self.db_session.execute(query)).scalars().all()
        items = [HealthNightSummaryRead.model_validate(r) for r in rows]
        return items, total

    async def list_night_dates(self) -> list[date]:
        """Return all night dates for the profile that have sleep summaries, ascending.

        Unbounded by design — mirrors GET /days/dates; payloads are tiny date strings.
        """
        query = (
            select(models.HealthNightlySummary.night_date)
            .where(models.HealthNightlySummary.profile_id == self.profile_id)
            .order_by(models.HealthNightlySummary.night_date)
        )
        rows = (await self.db_session.execute(query)).scalars().all()
        return list(rows)

    async def get_night_detail(self, night_date: date) -> HealthNightDetailRead:
        """Return nightly sleep summary with aggregated SpO2 and respiratory rate.

        SpO2 sources disagree on encoding (fraction 0.95 vs percent 95), so each
        sample is normalized to percent before aggregating; samples outside
        50–100% after normalization are ignored.

        Raises NotFoundError when no summary exists for this night.
        """
        summary = (
            await self.db_session.execute(
                select(models.HealthNightlySummary).where(
                    models.HealthNightlySummary.profile_id == self.profile_id,
                    models.HealthNightlySummary.night_date == night_date,
                )
            )
        ).scalar_one_or_none()

        if summary is None:
            raise NotFoundError(f"No health data found for night {night_date}")

        value = models.HealthSample.value_num
        is_spo2 = models.HealthSample.record_type == SPO2_RECORD_TYPE
        # Value ranges are disjoint: [0.5, 1] is a fraction, [50, 100] a percent.
        spo2_pct = case(
            (
                is_spo2 & (value * 100).between(_SPO2_MIN_PCT, _SPO2_MAX_PCT),
                value * 100,
            ),
            (is_spo2 & value.between(_SPO2_MIN_PCT, _SPO2_MAX_PCT), value),
        )
        rr = case((models.HealthSample.record_type == _RR_RECORD_TYPE, value))

        # Single-pass conditional aggregation for SpO2 avg/min and RR avg.
        agg = (
            await self.db_session.execute(
                select(
                    func.avg(spo2_pct).label("avg_spo2"),
                    func.min(spo2_pct).label("min_spo2"),
                    func.avg(rr).label("avg_rr"),
                ).where(
                    models.HealthSample.profile_id == self.profile_id,
                    models.HealthSample.night_date == night_date,
                    models.HealthSample.record_type.in_(
                        [SPO2_RECORD_TYPE, _RR_RECORD_TYPE]
                    ),
                )
            )
        ).one()

        return HealthNightDetailRead(
            **HealthNightSummaryRead.model_validate(summary).model_dump(),
            avg_spo2_pct=round(agg.avg_spo2, 1) if agg.avg_spo2 is not None else None,
            min_spo2_pct=round(agg.min_spo2, 1) if agg.min_spo2 is not None else None,
            avg_rr=round(agg.avg_rr, 2) if agg.avg_rr is not None else None,
        )

    async def get_breathing_disturbance_by_night(
        self, date_from: date, date_to: date
    ) -> dict[date, float]:
        """Return mean Apple sleeping-breathing-disturbance value per night.

        Aggregates ``value_num`` of
        ``HKQuantityTypeIdentifierAppleSleepingBreathingDisturbances`` samples
        grouped by ``night_date`` (profile-scoped, inclusive date range).  Apple
        writes this metric sparsely — most nights have no row and are simply
        absent from the returned mapping (never keyed to ``None``).

        The mean collapses the rare multi-source night to one value; single-row
        nights (the common case) pass through unchanged.
        """
        rows = (
            await self.db_session.execute(
                select(
                    models.HealthSample.night_date,
                    func.avg(models.HealthSample.value_num),
                )
                .where(
                    models.HealthSample.profile_id == self.profile_id,
                    models.HealthSample.record_type
                    == _BREATHING_DISTURBANCE_RECORD_TYPE,
                    models.HealthSample.night_date >= date_from,
                    models.HealthSample.night_date <= date_to,
                    models.HealthSample.value_num.is_not(None),
                )
                .group_by(models.HealthSample.night_date)
            )
        ).all()
        return {night: float(value) for night, value in rows if value is not None}

    async def get_fragmentation_by_night(
        self, date_from: date, date_to: date
    ) -> dict[date, NightFragmentation]:
        """Return ``NightFragmentation(awake_seconds, sleep_efficiency_pct)`` per night.

        Sourced from cached ``HealthNightlySummary`` rows (profile-scoped,
        inclusive range); either component may be ``None`` when the summary
        lacked stage data.  Nights without a summary row are absent from the map.
        """
        rows = (
            await self.db_session.execute(
                select(
                    models.HealthNightlySummary.night_date,
                    models.HealthNightlySummary.awake_seconds,
                    models.HealthNightlySummary.sleep_efficiency_pct,
                ).where(
                    models.HealthNightlySummary.profile_id == self.profile_id,
                    models.HealthNightlySummary.night_date >= date_from,
                    models.HealthNightlySummary.night_date <= date_to,
                )
            )
        ).all()
        return {
            night: NightFragmentation(awake, efficiency)
            for night, awake, efficiency in rows
        }

    async def get_night_samples(
        self, night_date: date, source_name: str | None = None
    ) -> list[HealthSampleRead]:
        """Return sleep-stage samples for the night, ordered by start time.

        Source filter: explicit source_name overrides; when omitted the night's
        preferred_source is used (no filter when preferred_source is also None).

        Raises NotFoundError when no summary row exists for this night.
        """
        row = (
            await self.db_session.execute(
                select(
                    models.HealthNightlySummary.id,
                    models.HealthNightlySummary.preferred_source,
                ).where(
                    models.HealthNightlySummary.profile_id == self.profile_id,
                    models.HealthNightlySummary.night_date == night_date,
                )
            )
        ).one_or_none()

        if row is None:
            raise NotFoundError(f"No health data found for night {night_date}")

        effective_source = (
            source_name if source_name is not None else row.preferred_source
        )

        query = (
            select(models.HealthSample)
            .where(
                models.HealthSample.profile_id == self.profile_id,
                models.HealthSample.night_date == night_date,
                models.HealthSample.record_type == SLEEP_TYPE,
            )
            .order_by(models.HealthSample.start_time)
        )
        if effective_source is not None:
            query = query.where(models.HealthSample.source_name == effective_source)

        rows = (await self.db_session.execute(query)).scalars().all()
        return [HealthSampleRead.model_validate(r) for r in rows]
