"""Apple Health cross-source night-level validator.

Correlates SNORE's experimental nightly FL/RERA indices against *independent*
Apple Watch signals already in the database.  RERAs terminate in cortical
arousal, so watch-detected fragmentation (awake time, sleep efficiency) and
Apple's own sleeping-breathing-disturbance metric form a second, genuinely
independent validity axis — the device's own FLG signal is not.

This is measurement infrastructure only: it consumes existing nightly outputs
and changes no algorithm or threshold.  It is night-level and cheap (no
waveforms), so it runs synchronously everywhere.

Night-date join
---------------
``HealthSample.night_date`` and ``HealthNightlySummary.night_date`` are assigned
by ``apply_noon_split`` (noon boundary), which by construction mirrors
``DayManager.get_day_for_session`` / ``DEFAULT_SPLIT_TIME = time(12, 0)`` — the
same convention behind ``NightlyAnalysisSummary.therapy_date``.  The two axes
therefore key on identical night dates and are joined directly.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.shared.versioning import DayAnalysisStatus
from snore.services.breath.dtos import DeviceAmbiguityError, NightlyAnalysisSummary
from snore.services.breath.nightly import MAX_NIGHTS_PER_CALL
from snore.services.breath_service import BreathService
from snore.services.health_service import HealthService, NightFragmentation
from snore.validation.apple_cross_report import (
    AppleCrossAggregate,
    AppleCrossNightRecord,
    AppleCrossValidationReport,
    correlate_night_pairs,
)


def _skip_reason_for(day_status: DayAnalysisStatus) -> str | None:
    """Map a day's analysis status to a visible skip reason (None when usable)."""
    if day_status == DayAnalysisStatus.NOT_RUN:
        return "analysis_not_run"
    if day_status in (DayAnalysisStatus.STALE, DayAnalysisStatus.MIXED_VERSION):
        return "analysis_stale"
    return None


def _reason_str(reason: object | None) -> str | None:
    """Render a NullReason (StrEnum) as its plain string value, passing None."""
    return str(reason) if reason is not None else None


class AppleCrossValidator:
    """Validates SNORE FL/RERA indices against independent Apple Health signals."""

    def __init__(self, db_session: AsyncSession, profile_id: int) -> None:
        self._db = db_session
        self._profile_id = profile_id

    async def validate_date_range(
        self, date_from: str, date_to: str, device_id: int | None = None
    ) -> AppleCrossValidationReport:
        """Run cross-source validation across an inclusive date range.

        Args:
            date_from: Start date (YYYY-MM-DD).
            date_to: End date (YYYY-MM-DD).
            device_id: Pin the SNORE device to resolve nights against.  When None
                (the default) and a night has sessions from more than one device,
                that night degrades to a ``device_ambiguous`` skip rather than
                aborting the whole range.
        """
        start = date.fromisoformat(date_from)
        end = date.fromisoformat(date_to)

        nights, ambiguous_dates = await self._collect_snore_nights(
            start, end, device_id
        )
        health = HealthService(self._db, self._profile_id)
        apple_bd = await health.get_breathing_disturbance_by_night(start, end)
        fragmentation = await health.get_fragmentation_by_night(start, end)

        records = [self._build_record(n, apple_bd, fragmentation) for n in nights]
        records.extend(
            self._ambiguous_record(d, apple_bd, fragmentation) for d in ambiguous_dates
        )
        records.sort(key=lambda r: r.night_date)
        aggregate = self._aggregate(records)

        return AppleCrossValidationReport(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_range_start=date_from,
            date_range_end=date_to,
            aggregate=aggregate,
            nights=records,
        )

    async def _collect_snore_nights(
        self, start: date, end: date, device_id: int | None
    ) -> tuple[list[NightlyAnalysisSummary], list[date]]:
        """Page ``get_experimental_nightly_range`` over the range in <=90-night windows.

        Returns ``(resolved_nights, ambiguous_dates)``.  A chunk that raises
        ``DeviceAmbiguityError`` (only possible when ``device_id`` is None) is
        retried night-by-night so the ambiguity is isolated to the specific
        nights that carry multiple devices; every other night is still scored.
        """
        breath_svc = BreathService(self._db, self._profile_id)
        nights: list[NightlyAnalysisSummary] = []
        ambiguous: list[date] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=MAX_NIGHTS_PER_CALL - 1), end)
            chunk_nights, chunk_ambiguous = await self._resolve_chunk(
                breath_svc, cursor, chunk_end, device_id
            )
            nights.extend(chunk_nights)
            ambiguous.extend(chunk_ambiguous)
            cursor = chunk_end + timedelta(days=1)
        return nights, ambiguous

    @staticmethod
    async def _resolve_chunk(
        breath_svc: BreathService,
        start: date,
        end: date,
        device_id: int | None,
    ) -> tuple[list[NightlyAnalysisSummary], list[date]]:
        """Resolve one chunk, degrading device-ambiguous nights to skips.

        With ``device_id`` pinned, a ``DeviceAmbiguityError`` cannot arise from
        the pin itself (it selects one device), so any other service error
        propagates unchanged.
        """
        try:
            summary = await breath_svc.get_experimental_nightly_range(
                start, end, device_id=device_id
            )
            return list(summary.nights), []
        except DeviceAmbiguityError:
            if device_id is not None:
                raise
            nights: list[NightlyAnalysisSummary] = []
            ambiguous: list[date] = []
            cursor = start
            while cursor <= end:
                try:
                    single = await breath_svc.get_experimental_nightly_range(
                        cursor, cursor, device_id=None
                    )
                    nights.extend(single.nights)
                except DeviceAmbiguityError:
                    ambiguous.append(cursor)
                cursor += timedelta(days=1)
            return nights, ambiguous

    @staticmethod
    def _build_record(
        night: NightlyAnalysisSummary,
        apple_bd: dict[date, float],
        fragmentation: dict[date, NightFragmentation],
    ) -> AppleCrossNightRecord:
        bd = apple_bd.get(night.therapy_date)
        frag = fragmentation.get(night.therapy_date, NightFragmentation(None, None))
        return AppleCrossNightRecord(
            night_date=night.therapy_date.isoformat(),
            rera_index=night.rera_index,
            rera_index_reason=_reason_str(night.rera_index_reason),
            fl_class_ge4_pct=night.fl_class_ge4_pct,
            fl_class_ge4_pct_reason=_reason_str(night.fl_class_ge4_pct_reason),
            apple_breathing_disturbances=bd,
            apple_bd_reason="no_apple_bd" if bd is None else None,
            awake_seconds=frag.awake_seconds,
            sleep_efficiency_pct=frag.sleep_efficiency_pct,
            skip_reason=_skip_reason_for(night.day_status),
        )

    @staticmethod
    def _ambiguous_record(
        night_date: date,
        apple_bd: dict[date, float],
        fragmentation: dict[date, NightFragmentation],
    ) -> AppleCrossNightRecord:
        """A device-ambiguous night: null SNORE side, Apple side still joined."""
        bd = apple_bd.get(night_date)
        frag = fragmentation.get(night_date, NightFragmentation(None, None))
        return AppleCrossNightRecord(
            night_date=night_date.isoformat(),
            rera_index=None,
            rera_index_reason="device_ambiguous",
            fl_class_ge4_pct=None,
            fl_class_ge4_pct_reason="device_ambiguous",
            apple_breathing_disturbances=bd,
            apple_bd_reason="no_apple_bd" if bd is None else None,
            awake_seconds=frag.awake_seconds,
            sleep_efficiency_pct=frag.sleep_efficiency_pct,
            skip_reason="device_ambiguous",
        )

    @staticmethod
    def _aggregate(records: list[AppleCrossNightRecord]) -> AppleCrossAggregate:
        # Build per-metric night→value maps from the records themselves.  The
        # SNORE side (rera, fl) is gated on ``skip_reason is None`` so a skipped
        # night — including a stale / version-mismatched one that still carries
        # non-null indices — contributes to no correlation, upholding the
        # skip_reason invariant.  The Apple side is joined regardless of skip.
        rera: dict[date, float] = {}
        fl: dict[date, float] = {}
        awake: dict[date, float] = {}
        efficiency: dict[date, float] = {}
        apple_bd: dict[date, float] = {}
        for r in records:
            key = date.fromisoformat(r.night_date)
            if r.apple_breathing_disturbances is not None:
                apple_bd[key] = r.apple_breathing_disturbances
            if r.awake_seconds is not None:
                awake[key] = r.awake_seconds
            if r.sleep_efficiency_pct is not None:
                efficiency[key] = r.sleep_efficiency_pct
            if r.skip_reason is not None:
                continue
            if r.rera_index is not None:
                rera[key] = r.rera_index
            if r.fl_class_ge4_pct is not None:
                fl[key] = r.fl_class_ge4_pct

        n_with_apple_bd = len(apple_bd)

        return AppleCrossAggregate(
            total_nights=len(records),
            n_analysis_not_run=sum(
                1 for r in records if r.skip_reason == "analysis_not_run"
            ),
            n_analysis_stale=sum(
                1 for r in records if r.skip_reason == "analysis_stale"
            ),
            n_device_ambiguous=sum(
                1 for r in records if r.skip_reason == "device_ambiguous"
            ),
            n_skipped_no_apple_bd=len(records) - n_with_apple_bd,
            n_with_apple_bd=n_with_apple_bd,
            rera_vs_apple_bd=correlate_night_pairs(rera, apple_bd),
            fl_vs_apple_bd=correlate_night_pairs(fl, apple_bd),
            rera_vs_awake_seconds=correlate_night_pairs(rera, awake),
            fl_vs_sleep_efficiency=correlate_night_pairs(fl, efficiency),
        )
