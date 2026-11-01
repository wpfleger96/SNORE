"""Integration tests: device FL/snore aggregates in get_nightly_summary and compare_epochs.

Tests seed waveform rows and verify:
- device_flg_* and snore_* fields appear in NightlyRow
- CHANNEL_ABSENT reason when channel not recorded
- snore_pct_time computed correctly
- FL negative sentinel filter applied
- no_sessions reason when Day has no enabled sessions
- leak_above_24_pct from leak samples, across a night's sessions
- get_nightly_summary and compliance never reach the analysis path
- compare_epochs device_flg and snore_dist distributions, with no analysis
- unknown settings are not an RX change in compare_epochs
- default tools execute no SQL against the breaths table
"""

from __future__ import annotations

import re
import uuid

from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Breath
from tests.integration.conftest import (
    _make_analysis_result,
    _make_day_session,
    _make_device,
    _make_event,
    _make_profile,
)


def _make_waveform_blob(values: list[float]) -> tuple[bytes, int]:
    """Serialize a list of float values into the [timestamp, value] blob format."""
    n = len(values)
    arr = np.zeros((n, 2), dtype=np.float32)
    arr[:, 0] = np.arange(n, dtype=np.float32)  # timestamps: 0, 1, 2, ...
    arr[:, 1] = np.array(values, dtype=np.float32)
    return arr.tobytes(), n


async def _add_waveform(
    db: AsyncSession,
    session_id: int,
    waveform_type: str,
    values: list[float],
) -> None:
    from snore.database.models import Waveform  # noqa: PLC0415

    blob, sample_count = _make_waveform_blob(values)
    wf = Waveform(
        session_id=session_id,
        waveform_type=waveform_type,
        sample_rate=1.0,
        sample_count=sample_count,
        data_blob=blob,
    )
    db.add(wf)
    await db.flush()


@pytest.mark.integration
class TestNightlySummaryDeviceWaveforms:
    async def test_fl_channel_aggregates_populated(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """FL waveform values appear in device_flg_* fields of NightlyRow."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 1)
        _day, sess = await _make_day_session(async_db_session, device, d)

        # FL values: 0.0, 0.2, 0.4, 0.6, 0.8 — no negatives
        await _add_waveform(async_db_session, sess.id, "fl", [0.0, 0.2, 0.4, 0.6, 0.8])

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        assert len(result.nights) == 1
        night = result.nights[0]
        assert night.device_flg_median is not None
        assert night.device_flg_median == pytest.approx(0.4)
        assert night.device_flg_95th is not None
        assert night.device_flg_max == pytest.approx(0.8)
        assert night.device_flg_reason is None

    async def test_fl_negative_sentinels_filtered(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """FL values < 0 are filtered; remaining values are aggregated correctly."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 2)
        _day, sess = await _make_day_session(async_db_session, device, d)

        # Two sentinel values (−0.01 from digital −1) mixed with real data
        await _add_waveform(
            async_db_session, sess.id, "fl", [-0.01, 0.3, 0.5, -0.01, 0.7]
        )

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        # Only [0.3, 0.5, 0.7] remain; median = 0.5
        assert night.device_flg_median == pytest.approx(0.5)
        assert night.device_flg_max == pytest.approx(0.7)

    async def test_snore_channel_aggregates_populated(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Snore waveform values appear in snore_* fields including snore_pct_time."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 3)
        _day, sess = await _make_day_session(async_db_session, device, d)

        # 4 out of 10 samples > 0.5: 0.6, 0.7, 0.8, 1.0
        await _add_waveform(
            async_db_session,
            sess.id,
            "snore",
            [0.0, 0.0, 0.5, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 1.0],
        )

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        assert night.snore_median is not None
        assert night.snore_pct_time == pytest.approx(0.4)  # 4/10
        assert night.snore_reason is None

    async def test_channel_absent_gives_null_with_reason(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Night with no FL or snore waveform rows has null fields and reason string."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 4)
        await _make_day_session(async_db_session, device, d)

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        assert night.device_flg_median is None
        assert night.device_flg_reason == "channel_absent"
        assert night.snore_median is None
        assert night.snore_reason == "channel_absent"

    async def test_multi_session_night_merges_samples(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """FL values from two sessions of the same night are aggregated together."""
        from snore.database.models import Session  # noqa: PLC0415
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 5)
        day, sess1 = await _make_day_session(async_db_session, device, d)

        # Create a second session for the same day
        import uuid  # noqa: PLC0415

        from datetime import datetime, timedelta  # noqa: PLC0415

        start2 = datetime(d.year, d.month, d.day, 23, 0, 0)
        sess2 = Session(
            device_id=device.id,
            day_id=day.id,
            device_session_id=f"test_{d.isoformat()}_{uuid.uuid4().hex[:6]}",
            start_time=start2,
            end_time=start2 + timedelta(hours=2),
            duration_seconds=7200,
            enabled=True,
        )
        async_db_session.add(sess2)
        await async_db_session.flush()

        # Session 1 FL: [0.1, 0.2], Session 2 FL: [0.3, 0.4]
        await _add_waveform(async_db_session, sess1.id, "fl", [0.1, 0.2])
        await _add_waveform(async_db_session, sess2.id, "fl", [0.3, 0.4])

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        night = result.nights[0]
        # All 4 merged: median of [0.1, 0.2, 0.3, 0.4] = 0.25
        assert night.device_flg_median == pytest.approx(0.25)
        assert night.device_flg_max == pytest.approx(0.4)


@pytest.mark.integration
class TestNightlySummaryNoSessions:
    async def test_day_with_no_sessions_gives_no_sessions_reason(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Day that has no enabled sessions gets device_flg_reason='no_sessions'."""
        from snore.database.models import Day  # noqa: PLC0415
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 8, 1)
        # Create a Day with no sessions attached to it
        day = Day(device_id=device.id, date=d, total_therapy_hours=0)
        async_db_session.add(day)
        await async_db_session.flush()

        result = await get_nightly_summary(
            async_db_session,
            start=d,
            end=d,
            profile_id=async_test_profile.id,
        )
        assert len(result.nights) == 1
        night = result.nights[0]
        assert night.device_flg_median is None
        assert night.device_flg_reason == "no_sessions"
        assert night.snore_median is None
        assert night.snore_reason == "no_sessions"
        assert night.leak_above_24_pct is None
        assert night.leak_above_24_pct_reason == "no_sessions"

    async def test_other_device_with_disabled_session_keeps_null_channels(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """With no device_id, a Day of a device whose session is disabled gets
        no_sessions, not the channel stats of the device resolved on that date."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device_a = await _make_device(async_db_session, async_test_profile.id)
        device_b = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 8, 2)
        _, sess_a = await _make_day_session(async_db_session, device_a, d, ahi=1.0)
        _, sess_b = await _make_day_session(async_db_session, device_b, d, ahi=9.0)
        await _add_waveform(async_db_session, sess_a.id, "leak", [30.0, 30.0])
        await _add_waveform(async_db_session, sess_a.id, "fl", [0.9, 0.9])
        sess_b.enabled = False
        await async_db_session.flush()

        result = await get_nightly_summary(
            async_db_session, start=d, end=d, profile_id=async_test_profile.id
        )

        rows = {night.device_id: night for night in result.nights}
        assert rows[device_a.id].leak_above_24_pct == pytest.approx(100.0)
        assert rows[device_a.id].device_flg_median == pytest.approx(0.9)
        row_b = rows[device_b.id]
        assert row_b.leak_above_24_pct is None
        assert row_b.device_flg_median is None
        assert (
            row_b.leak_above_24_pct_reason
            == row_b.device_flg_reason
            == row_b.snore_reason
            == row_b.ti_median_reason
            == row_b.ie_ratio_reason
            == "no_sessions"
        )


@pytest.mark.integration
class TestNightlySummaryLeakAbove24Pct:
    async def test_percent_counts_boundary_and_skips_sentinels(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """24.0 counts as above; negative and non-finite samples count nowhere."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 6)
        _day, sess = await _make_day_session(async_db_session, device, d)
        await _add_waveform(
            async_db_session,
            sess.id,
            "leak",
            [24.0, 23.9, 0.0, 40.0, -1.0, float("nan"), float("inf")],
        )

        result = await get_nightly_summary(
            async_db_session, start=d, end=d, profile_id=async_test_profile.id
        )

        # Usable samples [24.0, 23.9, 0.0, 40.0]: 2 of 4 at or above 24.
        assert result.nights[0].leak_above_24_pct == pytest.approx(50.0)
        assert result.nights[0].leak_above_24_pct_reason is None

    async def test_percent_pools_samples_across_sessions(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A short leaky session weighs by its samples, not as half the night."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 7)
        day, first = await _make_day_session(async_db_session, device, d)
        second = await _add_second_session(async_db_session, day, 23)
        await _add_waveform(async_db_session, first.id, "leak", [0.0] * 9)
        await _add_waveform(async_db_session, second.id, "leak", [30.0])

        result = await get_nightly_summary(
            async_db_session, start=d, end=d, profile_id=async_test_profile.id
        )

        assert result.nights[0].leak_above_24_pct == pytest.approx(10.0)

    async def test_channel_absent_reason_without_leak_waveform(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 8)
        await _make_day_session(async_db_session, device, d)

        result = await get_nightly_summary(
            async_db_session, start=d, end=d, profile_id=async_test_profile.id
        )

        assert result.nights[0].leak_above_24_pct is None
        assert result.nights[0].leak_above_24_pct_reason == "channel_absent"


@pytest.mark.integration
class TestNightlySummaryWithoutAnalysis:
    async def test_device_data_returned_when_analysis_path_raises(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """The default tool never reaches the analysis bulk query."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415
        from snore.services.breath_service import BreathService  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 6, 9)
        _day, sess = await _make_day_session(async_db_session, device, d, ahi=4.2)
        await _add_waveform(async_db_session, sess.id, "fl", [0.1, 0.3])
        await _add_waveform(async_db_session, sess.id, "leak", [30.0, 0.0])

        with patch.object(
            BreathService,
            "_latest_analysis_bulk",
            side_effect=AssertionError("default path read analysis"),
        ) as analysis_bulk:
            result = await get_nightly_summary(
                async_db_session, start=d, end=d, profile_id=async_test_profile.id
            )

        analysis_bulk.assert_not_called()
        night = result.nights[0]
        assert night.ahi == pytest.approx(4.2)
        assert night.device_flg_median == pytest.approx(0.2)
        assert night.leak_above_24_pct == pytest.approx(50.0)

    async def test_compliance_on_nights_with_no_analysis(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Compliance needs only therapy hours: 8 h counts, 2 h does not."""
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(async_db_session, device, date(2025, 6, 10))
        await _make_day_session(
            async_db_session, device, date(2025, 6, 11), duration_hours=2.0
        )

        result = await get_nightly_summary(
            async_db_session,
            start=date(2025, 6, 10),
            end=date(2025, 6, 13),
            profile_id=async_test_profile.id,
        )

        assert result.compliance is not None
        assert result.compliance.days_compliant == 1
        assert result.compliance.days_total == 4
        assert result.compliance.compliance_pct == pytest.approx(25.0)
        assert result.device_capabilities is not None
        assert result.device_capabilities.has_analysis is False


async def _add_second_session(db: AsyncSession, day: Any, start_hour: int) -> Any:
    """Add another enabled session to an existing Day, as a split night has."""
    from snore.database.models import Session  # noqa: PLC0415

    start = datetime(day.date.year, day.date.month, day.date.day, start_hour)
    sess = Session(
        device_id=day.device_id,
        day_id=day.id,
        device_session_id=f"second_{uuid.uuid4().hex[:6]}",
        start_time=start,
        end_time=start + timedelta(hours=1),
        duration_seconds=3600.0,
        enabled=True,
    )
    db.add(sess)
    await db.flush()
    return sess


async def _add_rx(db: AsyncSession, session_id: int, value: str) -> None:
    from snore.database.models import Setting  # noqa: PLC0415

    db.add(Setting(session_id=session_id, key="pressure_min", value=value))
    await db.flush()


def _epoch(label: str, start: date, end: date, device_id: int | None = None) -> Any:
    from snore.mcp.schemas import EpochSpec  # noqa: PLC0415

    return EpochSpec(
        label=label,
        date_start=start.isoformat(),
        date_end=end.isoformat(),
        device_id=device_id,
    )


@pytest.mark.integration
class TestCompareEpochsDeviceWaveforms:
    async def test_distributions_populate_with_zero_analysis_rows(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Default compare_epochs needs no analysis: both channels aggregate across
        nights, FLG sentinels drop out, and snore zeros stay in."""
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess_1 = await _make_day_session(async_db_session, device, date(2025, 9, 1))
        _, sess_2 = await _make_day_session(async_db_session, device, date(2025, 9, 2))
        await _add_waveform(async_db_session, sess_1.id, "fl", [-0.01, 0.2, 0.4, 0.6])
        await _add_waveform(async_db_session, sess_1.id, "snore", [0.0, 0.6, 0.8, 1.0])
        await _add_waveform(async_db_session, sess_2.id, "fl", [0.1, 0.3, 0.5])
        await _add_waveform(async_db_session, sess_2.id, "snore", [0.0, 0.0, 0.7])

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[_epoch("A", date(2025, 9, 1), date(2025, 9, 30))],
        )

        assert result.null_reason is None
        epoch = result.epochs[0]
        assert epoch.null_reason is None
        assert epoch.nights_with_data == 2
        assert epoch.device_flg.n_samples == 6
        assert epoch.device_flg.n_nights == 2
        assert epoch.device_flg.median == pytest.approx(0.35)
        assert epoch.snore_dist.n_samples == 7
        assert epoch.snore_dist.n_nights == 2
        assert epoch.snore_dist.median == pytest.approx(0.6)

    async def test_analysis_state_is_never_read(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """An analysis lookup anywhere on the default path fails this test."""
        from snore.services.breath_service import (  # noqa: PLC0415
            BreathService,
            EpochRequest,
        )

        device = await _make_device(async_db_session, async_test_profile.id)
        therapy_date = date(2025, 9, 1)
        _, sess = await _make_day_session(async_db_session, device, therapy_date)
        await _add_waveform(async_db_session, sess.id, "fl", [0.2])

        with (
            patch.object(
                BreathService,
                "_latest_analysis_for_session",
                side_effect=AssertionError("analysis read"),
            ),
            patch.object(
                BreathService,
                "_latest_analysis_bulk",
                side_effect=AssertionError("analysis read"),
            ),
        ):
            result = await BreathService(
                async_db_session, async_test_profile.id
            ).compare_device_epochs(
                [
                    EpochRequest(
                        label="A", date_start=therapy_date, date_end=therapy_date
                    )
                ]
            )

        assert result.epochs[0].device_flg.n_samples == 1

    async def test_two_session_night_counts_as_one_night(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        therapy_date = date(2025, 9, 1)
        day, sess_1 = await _make_day_session(async_db_session, device, therapy_date)
        sess_2 = await _add_second_session(async_db_session, day, start_hour=23)
        await _add_waveform(async_db_session, sess_1.id, "fl", [0.1, 0.2])
        await _add_waveform(async_db_session, sess_2.id, "fl", [0.3])
        await _add_waveform(async_db_session, sess_1.id, "snore", [1.0])
        await _add_waveform(async_db_session, sess_2.id, "snore", [2.0, 3.0])

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[_epoch("A", therapy_date, therapy_date)],
        )

        epoch = result.epochs[0]
        assert epoch.nights_with_data == 1
        assert epoch.device_flg.n_samples == 3
        assert epoch.device_flg.n_nights == 1
        assert epoch.snore_dist.n_samples == 3
        assert epoch.snore_dist.n_nights == 1

    async def test_night_with_no_valid_samples_is_not_counted(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A night of FLG sentinels counts toward nights_with_data, not n_nights;
        non-finite samples drop from both channels."""
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess_1 = await _make_day_session(async_db_session, device, date(2025, 9, 1))
        _, sess_2 = await _make_day_session(async_db_session, device, date(2025, 9, 2))
        await _add_waveform(async_db_session, sess_1.id, "fl", [0.5, float("nan")])
        await _add_waveform(async_db_session, sess_1.id, "snore", [float("inf"), 0.0])
        await _add_waveform(async_db_session, sess_2.id, "fl", [-1.0, -0.5])

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[_epoch("A", date(2025, 9, 1), date(2025, 9, 2))],
        )

        epoch = result.epochs[0]
        assert epoch.nights_with_data == 2
        assert epoch.device_flg.n_samples == 1
        assert epoch.device_flg.n_nights == 1
        assert epoch.device_flg.median == pytest.approx(0.5)
        assert epoch.snore_dist.n_samples == 1
        assert epoch.snore_dist.median == pytest.approx(0.0)

    async def test_absent_channel_nulls_its_stats_not_the_epoch(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        therapy_date = date(2025, 9, 1)
        _, sess = await _make_day_session(async_db_session, device, therapy_date)
        await _add_waveform(async_db_session, sess.id, "fl", [0.2])

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[_epoch("A", therapy_date, therapy_date)],
        )

        assert result.null_reason is None
        epoch = result.epochs[0]
        assert epoch.null_reason is None
        assert epoch.nights_with_data == 1
        assert epoch.device_flg.n_samples == 1
        assert epoch.snore_dist.median is None
        assert epoch.snore_dist.n_samples == 0
        assert epoch.snore_dist.n_nights == 0

    async def test_disabled_session_does_not_contribute(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        _, kept = await _make_day_session(async_db_session, device, date(2025, 9, 1))
        _, skipped = await _make_day_session(async_db_session, device, date(2025, 9, 2))
        skipped.enabled = False
        await _add_waveform(async_db_session, kept.id, "fl", [0.2])
        await _add_waveform(async_db_session, skipped.id, "fl", [0.9, 0.9])

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[_epoch("A", date(2025, 9, 1), date(2025, 9, 2))],
        )

        epoch = result.epochs[0]
        assert epoch.nights_with_data == 1
        assert epoch.device_flg.n_samples == 1

    async def test_rx_change_without_analysis_nulls_only_its_epoch(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """The RX check covers unanalyzed sessions; the violating epoch stays null
        and never loads its waveforms, and the clean epoch still computes."""
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415
        from snore.services.breath_service import BreathService  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        _, clean = await _make_day_session(async_db_session, device, date(2025, 9, 1))
        await _add_waveform(async_db_session, clean.id, "fl", [0.2])

        changed_on = date(2025, 10, 2)
        _, before = await _make_day_session(async_db_session, device, date(2025, 10, 1))
        _, after = await _make_day_session(async_db_session, device, changed_on)
        await _add_rx(async_db_session, before.id, "8.0")
        await _add_rx(async_db_session, after.id, "12.0")
        await _add_waveform(async_db_session, before.id, "fl", [0.3])

        real_fetch = BreathService._fetch_channel_vals
        fetched_session_ids: list[int] = []

        async def _recording_fetch(
            db: AsyncSession, session_ids: list[int], channels: Any
        ) -> Any:
            fetched_session_ids.extend(session_ids)
            return await real_fetch(db, session_ids, channels)

        with patch.object(
            BreathService, "_fetch_channel_vals", staticmethod(_recording_fetch)
        ):
            result = await compare_epochs(
                db_session=async_db_session,
                profile_id=async_test_profile.id,
                epochs=[
                    _epoch("clean", date(2025, 9, 1), date(2025, 9, 30)),
                    _epoch("violating", date(2025, 10, 1), date(2025, 10, 31)),
                ],
            )

        assert result.null_reason is None
        clean_epoch, violating = result.epochs
        assert clean_epoch.null_reason is None
        assert clean_epoch.device_flg.n_samples == 1
        assert violating.null_reason == "rx_changed_within_epoch"
        assert violating.nights_with_data == 2
        assert violating.rx_settings == {"pressure_min": "8.0"}
        assert violating.device_flg.median is None
        assert violating.device_flg.n_samples == 0
        assert [v.epoch_label for v in result.rx_violations] == ["violating"]
        assert result.rx_violations[0].changed_keys == ["pressure_min"]
        assert result.rx_violations[0].change_dates == [changed_on.isoformat()]
        assert fetched_session_ids == [clean.id]

    async def test_every_epoch_rx_changed_sets_top_level_reason(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        for month in (9, 10):
            for day_of_month, value in ((1, "8.0"), (2, "12.0")):
                _, sess = await _make_day_session(
                    async_db_session, device, date(2025, month, day_of_month)
                )
                await _add_rx(async_db_session, sess.id, value)

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                _epoch("A", date(2025, 9, 1), date(2025, 9, 2)),
                _epoch("B", date(2025, 10, 1), date(2025, 10, 2)),
            ],
        )

        assert result.null_reason == "rx_changed_within_epoch"
        assert {e.null_reason for e in result.epochs} == {"rx_changed_within_epoch"}
        assert len(result.rx_violations) == 2

    async def _seed_nights(
        self, db: AsyncSession, profile_id: int, rx_values: list[str | None]
    ) -> None:
        """One session per night from 2025-02-01; a None value seeds no settings."""
        device = await _make_device(db, profile_id)
        for offset, value in enumerate(rx_values):
            night = date(2025, 2, 1) + timedelta(days=offset)
            _, sess = await _make_day_session(db, device, night)
            await _add_waveform(db, sess.id, "fl", [0.2])
            if value is not None:
                await _add_rx(db, sess.id, value)

    @pytest.mark.parametrize(
        ("rx_values", "expected_rx"),
        [
            pytest.param([None, "8.0"], {"pressure_min": "8.0"}, id="unknown_first"),
            pytest.param(["8.0", None], {"pressure_min": "8.0"}, id="unknown_last"),
            pytest.param([None, None], {}, id="all_unknown"),
        ],
    )
    async def test_unknown_settings_are_not_an_rx_change(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Any,
        rx_values: list[str | None],
        expected_rx: dict[str, str],
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        await self._seed_nights(async_db_session, async_test_profile.id, rx_values)

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[_epoch("A", date(2025, 2, 1), date(2025, 2, 28))],
        )

        epoch = result.epochs[0]
        assert result.null_reason is None
        assert result.rx_violations == []
        assert epoch.null_reason is None
        assert epoch.rx_settings == expected_rx
        assert epoch.device_flg.n_samples == 2

    async def test_change_across_unknown_session_reports_later_known_date(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        await self._seed_nights(
            async_db_session, async_test_profile.id, ["8.0", None, "12.0"]
        )

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[_epoch("A", date(2025, 2, 1), date(2025, 2, 28))],
        )

        assert result.epochs[0].null_reason == "rx_changed_within_epoch"
        assert result.epochs[0].rx_settings == {"pressure_min": "8.0"}
        violation = result.rx_violations[0]
        assert violation.changed_keys == ["pressure_min"]
        assert violation.change_dates == ["2025-02-03"]

    async def test_foreign_device_is_not_available_at_both_levels(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        other_profile = await _make_profile(async_db_session)
        foreign = await _make_device(async_db_session, other_profile.id)
        therapy_date = date(2025, 9, 1)
        await _make_day_session(async_db_session, foreign, therapy_date)

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                _epoch("A", therapy_date, therapy_date, foreign.id),
                _epoch("B", therapy_date, therapy_date, foreign.id),
            ],
        )

        assert result.null_reason == "not_available"
        assert {e.null_reason for e in result.epochs} == {"not_available"}

    @pytest.mark.parametrize("explicit_device", [False, True])
    async def test_no_sessions_in_union_range_is_no_data_at_both_levels(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Any,
        explicit_device: bool,
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        await _make_day_session(async_db_session, device, date(2025, 1, 1))
        device_id = device.id if explicit_device else None

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                _epoch("A", date(2025, 9, 1), date(2025, 9, 2), device_id),
                _epoch("B", date(2025, 10, 1), date(2025, 10, 2), device_id),
            ],
        )

        assert result.null_reason == "no_data_in_range"
        assert {e.null_reason for e in result.epochs} == {"no_data_in_range"}

    async def test_epoch_without_sessions_leaves_other_epochs_populated(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, date(2025, 9, 1))
        await _add_waveform(async_db_session, sess.id, "fl", [0.2])

        result = await compare_epochs(
            db_session=async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                _epoch("A", date(2025, 9, 1), date(2025, 9, 30)),
                _epoch("B", date(2025, 10, 1), date(2025, 10, 31)),
            ],
        )

        assert result.null_reason is None
        populated, empty = result.epochs
        assert populated.device_flg.n_samples == 1
        assert empty.null_reason == "no_data_in_range"
        assert empty.nights_with_data == 0

    async def test_two_devices_without_device_id_is_a_tool_error(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.errors import ValidationError  # noqa: PLC0415
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415

        device_a = await _make_device(async_db_session, async_test_profile.id)
        device_b = await _make_device(async_db_session, async_test_profile.id)
        therapy_date = date(2025, 9, 1)
        await _make_day_session(async_db_session, device_a, therapy_date)
        await _make_day_session(async_db_session, device_b, therapy_date)

        with pytest.raises(ValidationError, match="Multiple devices"):
            await compare_epochs(
                db_session=async_db_session,
                profile_id=async_test_profile.id,
                epochs=[_epoch("A", therapy_date, therapy_date)],
            )

    async def test_service_rejects_reversed_and_multi_device_epochs(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.services.breath_service import (  # noqa: PLC0415
            BreathService,
            EpochRequest,
        )

        svc = BreathService(async_db_session, async_test_profile.id)
        start, end = date(2025, 9, 1), date(2025, 9, 2)

        with pytest.raises(ValueError, match="must be <="):
            await svc.compare_device_epochs(
                [EpochRequest(label="A", date_start=end, date_end=start)]
            )
        with pytest.raises(ValueError, match="same device_id"):
            await svc.compare_device_epochs(
                [
                    EpochRequest(
                        label="A", date_start=start, date_end=end, device_id=1
                    ),
                    EpochRequest(
                        label="B", date_start=start, date_end=end, device_id=2
                    ),
                ]
            )


@pytest.mark.integration
class TestDefaultToolsReadNoBreaths:
    async def test_default_tools_execute_no_sql_against_breaths_table(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Default tools answer from device data even when breath rows exist."""
        from snore.mcp.tools.epochs import compare_epochs  # noqa: PLC0415
        from snore.mcp.tools.events import get_events  # noqa: PLC0415
        from snore.mcp.tools.summary import get_nightly_summary  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        d = date(2025, 11, 1)
        _, sess = await _make_day_session(async_db_session, device, d, ahi=4.2)
        for channel in ("flow", "pressure", "mv", "leak", "fl", "snore"):
            await _add_waveform(async_db_session, sess.id, channel, [0.5] * 600)
        await _make_event(async_db_session, sess)
        analysis = await _make_analysis_result(async_db_session, sess)
        async_db_session.add(
            Breath(
                analysis_result_id=analysis.id,
                session_id=sess.id,
                breath_number=1,
                start_offset_s=0.0,
                end_offset_s=4.0,
                leak_valid=True,
            )
        )
        await async_db_session.flush()

        statements: list[str] = []

        def capture_statement(
            _conn: Any, _cursor: Any, statement: str, *_args: Any
        ) -> None:
            statements.append(statement)

        engine = async_db_session.get_bind()
        event.listen(engine, "before_cursor_execute", capture_statement)
        try:
            nightly = await get_nightly_summary(
                async_db_session, start=d, end=d, profile_id=async_test_profile.id
            )
            events = await get_events(
                async_db_session, d, profile_id=async_test_profile.id
            )
            epochs = await compare_epochs(
                db_session=async_db_session,
                profile_id=async_test_profile.id,
                epochs=[_epoch("A", d, d)],
            )
        finally:
            event.remove(engine, "before_cursor_execute", capture_statement)

        assert nightly.nights[0].device_flg_median == pytest.approx(0.5)
        assert len(events.events) == 1
        assert epochs.epochs[0].device_flg.n_samples == 600
        assert statements
        breaths_table = re.compile(rf"\b{Breath.__tablename__}\b", re.IGNORECASE)
        assert [s for s in statements if breaths_table.search(s)] == []
