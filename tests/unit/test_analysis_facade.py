"""Unit tests for AnalysisFacade."""

from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import (
    AnalysisResult,
    Day,
    DetectedPattern,
    Device,
    Session,
)
from snore.services.analysis_facade import AnalysisFacade


async def _create_session_with_analysis(
    db_session: AsyncSession, device: Device, day_date: date, num_analyses: int = 1
) -> tuple[Day, Session]:
    """Helper to create a session with analysis results."""
    day = Day(device_id=device.id, date=day_date, total_therapy_hours=8.0)
    db_session.add(day)
    await db_session.flush()
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"test_{day_date.isoformat()}",
        start_time=datetime.combine(day_date, datetime.min.time()),
        end_time=datetime.combine(day_date, datetime.min.time()) + timedelta(hours=8),
        duration_seconds=28800,
    )
    db_session.add(sess)
    await db_session.flush()
    for i in range(num_analyses):
        ar = AnalysisResult(
            session_id=sess.id,
            timestamp_start=sess.start_time,
            timestamp_end=sess.end_time,
            programmatic_result_json={"version": i + 1},
            created_at=datetime.now(UTC) + timedelta(minutes=i),
        )
        db_session.add(ar)
    await db_session.flush()
    return day, sess


class TestListSessionsWithStatus:
    """Tests for list_sessions_with_status method."""

    async def test_list_empty(self, async_db_session):
        """Empty database returns empty list."""
        service = AnalysisFacade(async_db_session, profile_id=1)
        results = await service.list_sessions_with_status()

        assert results == []

    async def test_list_with_analysis(self, async_db_session, async_test_device):
        """Sessions with and without analysis show correct has_analysis flags."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        day1, sess1 = await _create_session_with_analysis(
            async_db_session, async_test_device, yesterday
        )
        day2 = Day(device_id=async_test_device.id, date=today, total_therapy_hours=7.0)
        async_db_session.add(day2)
        await async_db_session.flush()

        sess2 = Session(
            device_id=async_test_device.id,
            day_id=day2.id,
            device_session_id="test_no_analysis",
            start_time=datetime.combine(today, datetime.min.time()),
            end_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=7),
            duration_seconds=25200,
        )
        async_db_session.add(sess2)
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        results = await service.list_sessions_with_status()

        assert len(results) == 2
        assert results[0].session_id == sess2.id
        assert results[0].has_analysis is False
        assert results[0].analysis_id is None

        assert results[1].session_id == sess1.id
        assert results[1].has_analysis is True
        assert results[1].analysis_id is not None

    async def test_list_analyzed_only(self, async_db_session, async_test_device):
        """Only returns sessions with analysis when analyzed_only is True."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        await _create_session_with_analysis(
            async_db_session, async_test_device, yesterday
        )

        day2 = Day(device_id=async_test_device.id, date=today, total_therapy_hours=7.0)
        async_db_session.add(day2)
        await async_db_session.flush()

        sess2 = Session(
            device_id=async_test_device.id,
            day_id=day2.id,
            device_session_id="test_no_analysis",
            start_time=datetime.combine(today, datetime.min.time()),
            end_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=7),
            duration_seconds=25200,
        )
        async_db_session.add(sess2)
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        results = await service.list_sessions_with_status(analyzed_only=True)

        assert len(results) == 1
        assert results[0].has_analysis is True

    async def test_list_date_filter(self, async_db_session, async_test_device):
        """Filter by start and end dates works correctly."""
        d1 = date(2024, 1, 1)
        d2 = date(2024, 1, 5)
        d3 = date(2024, 1, 10)

        await _create_session_with_analysis(async_db_session, async_test_device, d1)
        await _create_session_with_analysis(async_db_session, async_test_device, d2)
        await _create_session_with_analysis(async_db_session, async_test_device, d3)
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        results = await service.list_sessions_with_status(
            start=datetime(2024, 1, 2), end=datetime(2024, 1, 9)
        )

        assert len(results) == 1
        assert results[0].session_date == d2


class TestGetDeletePreview:
    """Tests for get_delete_preview method."""

    async def test_delete_preview_no_filters(self, async_db_session):
        """Raises ValueError when no filters are specified."""
        service = AnalysisFacade(async_db_session, profile_id=1)

        with pytest.raises(ValueError) as exc:
            await service.get_delete_preview()

        assert "at least one filter" in str(exc.value).lower()

    async def test_delete_preview_with_data(self, async_db_session, async_test_device):
        """Preview shows correct counts for sessions with analysis."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        _, sess1 = await _create_session_with_analysis(
            async_db_session, async_test_device, yesterday, num_analyses=2
        )
        _, sess2 = await _create_session_with_analysis(
            async_db_session, async_test_device, today
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(delete_all=True)

        assert preview.sessions_with_analysis == 2
        assert preview.total_analysis_records == 3
        assert preview.records_to_delete == 2
        assert len(preview.session_details) == 2

        detail1 = next(d for d in preview.session_details if d.id == sess1.id)
        assert detail1.version_count == 2

        detail2 = next(d for d in preview.session_details if d.id == sess2.id)
        assert detail2.version_count == 1

    async def test_delete_preview_latest_only(
        self, async_db_session, async_test_device
    ):
        """Preview with all_versions=False only counts latest analysis."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        await _create_session_with_analysis(
            async_db_session, async_test_device, yesterday, num_analyses=3
        )
        await _create_session_with_analysis(
            async_db_session, async_test_device, today, num_analyses=2
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(delete_all=True, all_versions=False)

        assert preview.sessions_with_analysis == 2
        assert preview.total_analysis_records == 5
        assert preview.records_to_delete == 2


def _stale_engine_json() -> dict:
    """Legacy flat engine_versions_json (no 'identity' key) → always stale."""
    return {"format_version": 3, "segmenter": "v1"}


def _current_engine_json() -> dict:
    """engine_versions_json matching the current AlgorithmIdentity."""
    from snore.analysis.shared.versioning import (  # noqa: PLC0415
        AlgorithmIdentity,
        AlgoVersions,
        AnalysisRunMetadata,
    )

    return AlgoVersions(
        identity=AlgorithmIdentity.current(),
        run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
    ).model_dump()


async def _create_session_with_mixed_analyses(
    db_session: AsyncSession,
    device: Device,
    day_date: date,
) -> tuple[Day, Session, AnalysisResult, AnalysisResult]:
    """Create a session with one stale and one current AnalysisResult row.

    Returns (day, session, stale_ar, current_ar).
    """
    day = Day(device_id=device.id, date=day_date, total_therapy_hours=8.0)
    db_session.add(day)
    await db_session.flush()
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"mixed_{day_date.isoformat()}",
        start_time=datetime.combine(day_date, datetime.min.time()),
        end_time=datetime.combine(day_date, datetime.min.time()) + timedelta(hours=8),
        duration_seconds=28800,
    )
    db_session.add(sess)
    await db_session.flush()

    stale_ar = AnalysisResult(
        session_id=sess.id,
        timestamp_start=sess.start_time,
        timestamp_end=sess.end_time,
        engine_versions_json=_stale_engine_json(),
        created_at=datetime.now(UTC),
    )
    db_session.add(stale_ar)

    current_ar = AnalysisResult(
        session_id=sess.id,
        timestamp_start=sess.start_time,
        timestamp_end=sess.end_time,
        engine_versions_json=_current_engine_json(),
        created_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    db_session.add(current_ar)

    await db_session.flush()
    return day, sess, stale_ar, current_ar


def _outdated_engine_json() -> dict:
    """Nested engine_versions_json whose identity differs from the current one."""
    from snore.analysis.shared.versioning import (  # noqa: PLC0415
        AlgorithmIdentity,
        AlgoVersions,
        AnalysisRunMetadata,
    )

    return AlgoVersions(
        identity=AlgorithmIdentity.current().model_copy(update={"segmenter": "v0"}),
        run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
    ).model_dump()


_SELECTION_BASE_TIME = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)


async def _create_selection_session(
    db_session: AsyncSession,
    device: Device,
    day_date: date,
    engine_jsons: list[dict] | None = None,
    has_flow: bool = True,
) -> Session:
    """Create a session with optional flow waveform and analyses, oldest first.

    ``engine_jsons[i]`` is stored with ``created_at = base + i minutes`` so the
    last entry is deterministically the latest analysis.
    """
    from snore.database.models import Waveform  # noqa: PLC0415

    day = Day(device_id=device.id, date=day_date, total_therapy_hours=8.0)
    db_session.add(day)
    await db_session.flush()
    start = datetime.combine(day_date, datetime.min.time())
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"select_{day_date.isoformat()}",
        start_time=start,
        end_time=start + timedelta(hours=8),
        duration_seconds=28800,
    )
    db_session.add(sess)
    await db_session.flush()
    if has_flow:
        db_session.add(
            Waveform(
                session_id=sess.id,
                waveform_type="flow",
                sample_rate=25.0,
                data_blob=b"",
            )
        )
    for i, engine_json in enumerate(engine_jsons or []):
        db_session.add(
            AnalysisResult(
                session_id=sess.id,
                timestamp_start=sess.start_time,
                timestamp_end=sess.end_time,
                engine_versions_json=engine_json,
                created_at=_SELECTION_BASE_TIME + timedelta(minutes=i),
            )
        )
    await db_session.flush()
    return sess


class TestListSessionIdsOnly:
    """list_session_ids(only=...) selects missing or stale analyzable sessions."""

    async def test_only_stale_selects_sessions_whose_latest_analysis_is_stale(
        self, async_db_session, async_test_device
    ):
        """Only sessions whose latest row fails the identity check are returned."""
        device = async_test_device
        await _create_selection_session(async_db_session, device, date(2025, 1, 1))
        await _create_selection_session(
            async_db_session, device, date(2025, 1, 2), [_current_engine_json()]
        )
        latest_stale = await _create_selection_session(
            async_db_session, device, date(2025, 1, 3), [_outdated_engine_json()]
        )
        await _create_selection_session(
            async_db_session,
            device,
            date(2025, 1, 4),
            [_outdated_engine_json(), _current_engine_json()],
        )
        stale_after_current = await _create_selection_session(
            async_db_session,
            device,
            date(2025, 1, 5),
            [_current_engine_json(), _outdated_engine_json()],
        )
        legacy_flat = await _create_selection_session(
            async_db_session, device, date(2025, 1, 6), [_stale_engine_json()]
        )
        column_default = await _create_selection_session(
            async_db_session, device, date(2025, 1, 7), [{}]
        )
        await _create_selection_session(
            async_db_session,
            device,
            date(2025, 1, 8),
            [_outdated_engine_json()],
            has_flow=False,
        )
        await async_db_session.commit()

        facade = AnalysisFacade(async_db_session, profile_id=1)
        ids = await facade.list_session_ids(only="stale")

        assert ids == [
            latest_stale.id,
            stale_after_current.id,
            legacy_flat.id,
            column_default.id,
        ]

    async def test_only_stale_orders_latest_by_created_at_not_id(
        self, async_db_session, async_test_device
    ):
        """The latest row is the newest ``created_at``, even when its id is lower."""
        device = async_test_device

        async def _session_with_rows_newest_first(
            day_date: date, newest: dict, older: dict
        ) -> Session:
            # Insert the newest row first so the older row gets the higher id.
            sess = await _create_selection_session(async_db_session, device, day_date)
            for engine_json, minutes in ((newest, 10), (older, 0)):
                async_db_session.add(
                    AnalysisResult(
                        session_id=sess.id,
                        timestamp_start=sess.start_time,
                        timestamp_end=sess.end_time,
                        engine_versions_json=engine_json,
                        created_at=_SELECTION_BASE_TIME + timedelta(minutes=minutes),
                    )
                )
                await async_db_session.flush()
            return sess

        await _session_with_rows_newest_first(
            date(2025, 4, 1),
            newest=_current_engine_json(),
            older=_outdated_engine_json(),
        )
        newest_stale = await _session_with_rows_newest_first(
            date(2025, 4, 2),
            newest=_outdated_engine_json(),
            older=_current_engine_json(),
        )
        await async_db_session.commit()

        facade = AnalysisFacade(async_db_session, profile_id=1)

        assert await facade.list_session_ids(only="stale") == [newest_stale.id]

    async def test_only_stale_with_date_range_excludes_out_of_range(
        self, async_db_session, async_test_device
    ):
        """Date filters compose with the stale selector."""
        device = async_test_device
        in_range = await _create_selection_session(
            async_db_session, device, date(2025, 2, 10), [_outdated_engine_json()]
        )
        await _create_selection_session(
            async_db_session, device, date(2025, 3, 10), [_outdated_engine_json()]
        )
        await async_db_session.commit()

        facade = AnalysisFacade(async_db_session, profile_id=1)
        ids = await facade.list_session_ids(
            from_date=datetime(2025, 2, 1),
            to_date=datetime(2025, 2, 28, 23, 59, 59),
            only="stale",
        )

        assert ids == [in_range.id]

    async def test_only_missing_selects_unanalyzed_flow_sessions(
        self, async_db_session, async_test_device
    ):
        """Missing excludes analyzed sessions of any version and flow-less sessions."""
        device = async_test_device
        unanalyzed = await _create_selection_session(
            async_db_session, device, date(2025, 5, 1)
        )
        await _create_selection_session(
            async_db_session, device, date(2025, 5, 2), [_outdated_engine_json()]
        )
        await _create_selection_session(
            async_db_session, device, date(2025, 5, 3), has_flow=False
        )
        await async_db_session.commit()

        facade = AnalysisFacade(async_db_session, profile_id=1)
        ids = await facade.list_session_ids(only="missing")

        assert ids == [unanalyzed.id]


class TestStartupGapFill:
    """Startup resume backfills never-analyzed sessions, never stale ones."""

    async def test_startup_resume_enqueues_only_never_analyzed_sessions(
        self, async_db_session, async_test_device
    ):
        from contextlib import asynccontextmanager  # noqa: PLC0415
        from unittest.mock import patch  # noqa: PLC0415

        from snore.api.app import _startup_resume_analysis  # noqa: PLC0415

        device = async_test_device
        unanalyzed = await _create_selection_session(
            async_db_session, device, date(2025, 6, 1)
        )
        await _create_selection_session(
            async_db_session, device, date(2025, 6, 2), [_outdated_engine_json()]
        )
        await _create_selection_session(
            async_db_session, device, date(2025, 6, 3), [_current_engine_json()]
        )
        await async_db_session.commit()

        @asynccontextmanager
        async def _scope(*args: Any, **kwargs: Any) -> AsyncIterator[AsyncSession]:
            yield async_db_session

        with (
            patch("snore.database.session.session_scope", _scope),
            patch("snore.api.analysis_jobs.enqueue") as enqueue,
        ):
            await _startup_resume_analysis({device.profile_id})

        enqueue.assert_called_once()
        assert enqueue.call_args.kwargs["session_ids"] == [unanalyzed.id]


class TestDeleteStaleVersions:
    """Tests for get_delete_preview and delete_analysis with stale_versions=True."""

    async def test_preview_stale_versions_counts_only_stale_rows(
        self, async_db_session, async_test_device
    ):
        """Preview shows only stale rows in records_to_delete; current row survives."""
        today = date.today()
        _, sess, stale_ar, current_ar = await _create_session_with_mixed_analyses(
            async_db_session, async_test_device, today
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(delete_all=True, stale_versions=True)

        assert preview.sessions_with_analysis == 1
        assert preview.total_analysis_records == 2  # both rows exist
        assert preview.records_to_delete == 1  # only the stale one
        assert len(preview.session_details) == 1
        assert preview.session_details[0].id == sess.id
        assert preview.session_details[0].version_count == 1  # 1 stale row

    async def test_preview_stale_versions_no_stale_rows_returns_zero(
        self, async_db_session, async_test_device
    ):
        """When all rows are current, records_to_delete is 0 and sessions empty."""
        today = date.today()
        _, sess = await _create_session_with_analysis(
            async_db_session, async_test_device, today
        )
        # Overwrite with a current-version engine_versions_json.
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        ar = (
            (
                await async_db_session.execute(
                    sa_select(AnalysisResult).filter_by(session_id=sess.id)
                )
            )
            .scalars()
            .first()
        )
        ar.engine_versions_json = _current_engine_json()
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(delete_all=True, stale_versions=True)

        assert preview.sessions_with_analysis == 0
        assert preview.records_to_delete == 0

    async def test_delete_stale_versions_removes_only_stale_rows(
        self, async_db_session, async_test_device
    ):
        """delete_analysis(stale_versions=True) removes stale rows; current row survives."""
        from sqlalchemy import select as sa_select

        today = date.today()
        _, sess, stale_ar, current_ar = await _create_session_with_mixed_analyses(
            async_db_session, async_test_device, today
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis([sess.id], stale_versions=True)

        assert deleted == 1

        remaining_ids = (
            (
                await async_db_session.execute(
                    sa_select(AnalysisResult.id).filter_by(session_id=sess.id)
                )
            )
            .scalars()
            .all()
        )
        # Only the current-version row survives.
        assert remaining_ids == [current_ar.id]

    async def test_delete_stale_versions_dry_run_deletes_nothing(
        self, async_db_session, async_test_device
    ):
        """get_delete_preview with stale_versions=True never mutates the DB."""
        from sqlalchemy import func  # noqa: PLC0415
        from sqlalchemy import select as sa_select

        today = date.today()
        _, sess, stale_ar, current_ar = await _create_session_with_mixed_analyses(
            async_db_session, async_test_device, today
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(delete_all=True, stale_versions=True)

        # Preview should identify the stale row…
        assert preview.records_to_delete == 1

        # …but the DB must be unchanged.
        count = (
            await async_db_session.execute(
                sa_select(func.count())
                .select_from(AnalysisResult)
                .filter_by(session_id=sess.id)
            )
        ).scalar()
        assert count == 2

    async def test_delete_stale_versions_with_force_flag(
        self, async_db_session, async_test_device
    ):
        """Deleting stale rows when all rows are stale leaves no analysis for session."""
        from sqlalchemy import func  # noqa: PLC0415
        from sqlalchemy import select as sa_select

        today = date.today()
        _, sess = await _create_session_with_analysis(
            async_db_session, async_test_device, today, num_analyses=2
        )
        # Both rows use the default engine_versions_json={} from the helper,
        # which lacks an 'identity' key → stale.
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis([sess.id], stale_versions=True)

        assert deleted == 2

        remaining = (
            await async_db_session.execute(
                sa_select(func.count())
                .select_from(AnalysisResult)
                .filter_by(session_id=sess.id)
            )
        ).scalar()
        assert remaining == 0

    async def test_malformed_nested_engine_json_classified_stale(
        self, async_db_session, async_test_device
    ):
        """A row with 'identity' key but unparseable nested content is classified stale.

        Exercises AlgoVersions.from_stored robustness: the ValidationError path
        must not propagate — the row must be treated as stale without crashing
        get_delete_preview or delete_analysis.
        """
        today = date.today()
        day = Day(device_id=async_test_device.id, date=today, total_therapy_hours=8.0)
        async_db_session.add(day)
        await async_db_session.flush()

        sess = Session(
            device_id=async_test_device.id,
            day_id=day.id,
            device_session_id=f"malformed_{today.isoformat()}",
            start_time=datetime.combine(today, datetime.min.time()),
            end_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=8),
            duration_seconds=28800,
        )
        async_db_session.add(sess)
        await async_db_session.flush()

        # "identity" key present but the nested content is unparseable — this
        # used to raise ValidationError in from_stored; after the fix it returns None.
        malformed_ar = AnalysisResult(
            session_id=sess.id,
            timestamp_start=sess.start_time,
            timestamp_end=sess.end_time,
            engine_versions_json={"identity": "not_a_dict", "run": "also_invalid"},
            created_at=datetime.now(UTC),
        )
        async_db_session.add(malformed_ar)
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)

        # Must not raise; malformed row must be classified stale.
        preview = await service.get_delete_preview(delete_all=True, stale_versions=True)
        assert preview.sessions_with_analysis == 1
        assert preview.records_to_delete == 1

        # Delete must also succeed without crash.
        deleted = await service.delete_analysis([sess.id], stale_versions=True)
        assert deleted == 1

    async def test_delete_stale_versions_with_all_scope(
        self, async_db_session, async_test_device
    ):
        """stale_versions=True scoped across multiple sessions deletes only stale rows."""
        from sqlalchemy import select as sa_select

        today = date.today()
        yesterday = today - timedelta(days=1)

        _, sess1, stale1, current1 = await _create_session_with_mixed_analyses(
            async_db_session, async_test_device, today
        )
        _, sess2, stale2, current2 = await _create_session_with_mixed_analyses(
            async_db_session, async_test_device, yesterday
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis(
            [sess1.id, sess2.id], stale_versions=True
        )

        assert deleted == 2  # one stale row per session

        surviving_ids = set(
            (await async_db_session.execute(sa_select(AnalysisResult.id)))
            .scalars()
            .all()
        )
        assert current1.id in surviving_ids
        assert current2.id in surviving_ids
        assert stale1.id not in surviving_ids
        assert stale2.id not in surviving_ids


class TestDeleteAnalysis:
    """Tests for delete_analysis method."""

    async def test_delete_all_versions(self, async_db_session, async_test_device):
        """Deletes all analysis records when all_versions=True."""
        from sqlalchemy import func  # noqa: PLC0415
        from sqlalchemy import select as sa_select

        today = date.today()
        _, sess = await _create_session_with_analysis(
            async_db_session, async_test_device, today, num_analyses=3
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis([sess.id], all_versions=True)

        assert deleted == 3

        remaining = (
            await async_db_session.execute(
                sa_select(func.count())
                .select_from(AnalysisResult)
                .filter_by(session_id=sess.id)
            )
        ).scalar()
        assert remaining == 0

    async def test_delete_latest_only(self, async_db_session, async_test_device):
        """Only deletes most recent analysis per session when all_versions=False."""
        from sqlalchemy import func  # noqa: PLC0415
        from sqlalchemy import select as sa_select

        today = date.today()
        _, sess = await _create_session_with_analysis(
            async_db_session, async_test_device, today, num_analyses=3
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis([sess.id], all_versions=False)

        assert deleted == 1

        remaining = (
            await async_db_session.execute(
                sa_select(func.count())
                .select_from(AnalysisResult)
                .filter_by(session_id=sess.id)
            )
        ).scalar()
        assert remaining == 2

        latest = (
            (
                await async_db_session.execute(
                    sa_select(AnalysisResult)
                    .filter_by(session_id=sess.id)
                    .order_by(AnalysisResult.created_at.desc())
                )
            )
            .scalars()
            .first()
        )
        assert latest.programmatic_result_json["version"] == 2

    async def test_delete_empty(self, async_db_session):
        """Returns 0 when no matching sessions exist."""
        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis([999], all_versions=False)

        assert deleted == 0

    async def test_delete_multiple_sessions(self, async_db_session, async_test_device):
        """Deletes analysis for multiple sessions."""
        from sqlalchemy import func  # noqa: PLC0415
        from sqlalchemy import select as sa_select

        today = date.today()
        yesterday = today - timedelta(days=1)

        _, sess1 = await _create_session_with_analysis(
            async_db_session, async_test_device, yesterday
        )
        _, sess2 = await _create_session_with_analysis(
            async_db_session, async_test_device, today
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis([sess1.id, sess2.id], all_versions=True)

        assert deleted == 2

        remaining = (
            await async_db_session.execute(
                sa_select(func.count()).select_from(AnalysisResult)
            )
        ).scalar()
        assert remaining == 0


def _make_session_scope_patcher(async_db_session: AsyncSession) -> Any:
    """Return an async context manager factory that yields *async_db_session*.

    Used to patch ``snore.database.session.session_scope`` in tests that call
    ``run_batch_analysis``, which now opens its own short-lived scope for the
    id-list query instead of using the caller-provided session.
    """
    from contextlib import asynccontextmanager  # noqa: PLC0415

    @asynccontextmanager
    async def _fake_scope(*args, **kwargs):
        yield async_db_session

    return _fake_scope


class TestRunBatchAnalysis:
    async def test_no_sessions_returns_empty_result(
        self, async_db_session, monkeypatch
    ):
        """No sessions in range returns BatchAnalysisResult with all zeros."""
        monkeypatch.setattr(
            "snore.database.session.session_scope",
            _make_session_scope_patcher(async_db_session),
        )
        facade = AnalysisFacade(async_db_session, profile_id=1)
        result = await facade.run_batch_analysis(
            from_date=datetime(2099, 1, 1), to_date=datetime(2099, 12, 31)
        )
        assert result.total == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.results == []

    async def test_successful_analysis(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Coordinator.submit() success propagates through run_batch_analysis."""
        from unittest.mock import AsyncMock, MagicMock

        from snore.services.schemas import BatchAnalysisResult, BatchSessionResult

        day, sess = await _create_session_with_analysis(
            async_db_session, async_test_device, date(2025, 1, 10), num_analyses=0
        )
        await async_db_session.commit()

        mock_result = BatchAnalysisResult(
            total=1,
            successful=1,
            failed=0,
            cancelled=0,
            results=[
                BatchSessionResult(
                    session_id=sess.id, session_date=date(2025, 1, 10), success=True
                )
            ],
        )
        coord_mock = MagicMock()
        coord_mock.submit = AsyncMock(return_value=mock_result)
        coord_mock.progress = (1, 1)

        monkeypatch.setattr(
            "snore.database.session.session_scope",
            _make_session_scope_patcher(async_db_session),
        )
        monkeypatch.setattr(
            "snore.services.analysis_facade.BatchAnalysisCoordinator",
            lambda: coord_mock,
        )
        facade = AnalysisFacade(async_db_session, profile_id=1)
        result = await facade.run_batch_analysis(
            from_date=datetime(2025, 1, 1), to_date=datetime(2025, 1, 31)
        )
        assert result.total == 1
        assert result.successful == 1
        assert result.failed == 0

    async def test_failed_analysis(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Coordinator.submit() failure propagates through run_batch_analysis."""
        from unittest.mock import AsyncMock, MagicMock

        from snore.services.schemas import BatchAnalysisResult, BatchSessionResult

        day, sess = await _create_session_with_analysis(
            async_db_session, async_test_device, date(2025, 1, 10), num_analyses=0
        )
        await async_db_session.commit()

        mock_result = BatchAnalysisResult(
            total=1,
            successful=0,
            failed=1,
            cancelled=0,
            results=[
                BatchSessionResult(
                    session_id=sess.id,
                    session_date=date(2025, 1, 10),
                    success=False,
                    error="test error",
                )
            ],
        )
        coord_mock = MagicMock()
        coord_mock.submit = AsyncMock(return_value=mock_result)
        coord_mock.progress = (1, 1)

        monkeypatch.setattr(
            "snore.database.session.session_scope",
            _make_session_scope_patcher(async_db_session),
        )
        monkeypatch.setattr(
            "snore.services.analysis_facade.BatchAnalysisCoordinator",
            lambda: coord_mock,
        )
        facade = AnalysisFacade(async_db_session, profile_id=1)
        result = await facade.run_batch_analysis(
            from_date=datetime(2025, 1, 1), to_date=datetime(2025, 1, 31)
        )
        assert result.total == 1
        assert result.failed == 1
        assert result.results[0].success is False

    async def test_progress_callback_called(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Progress callback is forwarded to coordinator.submit()."""
        from unittest.mock import MagicMock

        from snore.services.schemas import BatchAnalysisResult, BatchSessionResult

        day, sess = await _create_session_with_analysis(
            async_db_session, async_test_device, date(2025, 1, 10), num_analyses=0
        )
        await async_db_session.commit()

        calls: list[tuple[int, int | None]] = []

        async def fake_submit(**kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                cb(1, None)
            return BatchAnalysisResult(
                total=1,
                successful=1,
                failed=0,
                cancelled=0,
                results=[
                    BatchSessionResult(
                        session_id=sess.id, session_date=date(2025, 1, 10), success=True
                    )
                ],
            )

        coord_mock = MagicMock()
        coord_mock.submit = fake_submit
        coord_mock.progress = (1, 1)

        monkeypatch.setattr(
            "snore.database.session.session_scope",
            _make_session_scope_patcher(async_db_session),
        )
        monkeypatch.setattr(
            "snore.services.analysis_facade.BatchAnalysisCoordinator",
            lambda: coord_mock,
        )
        facade = AnalysisFacade(async_db_session, profile_id=1)
        await facade.run_batch_analysis(
            from_date=datetime(2025, 1, 1),
            to_date=datetime(2025, 1, 31),
            progress_callback=lambda completed, total: calls.append((completed, total)),
        )
        assert len(calls) == 1
        assert calls[0] == (1, None)  # total is None until exhausted (unknown)


class TestBatchCoordinatorCancellationAccounting:
    """Tests for BatchAnalysisCoordinator.submit() cancellation and totals."""

    async def test_all_pairs_cancelled_when_predicate_fires_before_dispatch(self):
        """If cancel_predicate returns True from the start, all pairs land as cancelled.

        The coordinator checks the predicate in _fill_window before creating any
        tasks.  All unstarted pairs are drained and counted as cancelled so the
        total stays honest.
        """
        from snore.services.analysis_facade import BatchAnalysisCoordinator

        pairs = [
            (1, date(2025, 1, 1)),
            (2, date(2025, 1, 2)),
            (3, date(2025, 1, 3)),
        ]
        coord = BatchAnalysisCoordinator()
        result = await coord.submit(
            session_pairs=pairs,
            profile_id=1,
            cancel_predicate=lambda: True,  # fires immediately
            max_workers=4,
            store_results=False,
        )

        assert result.total == 3
        assert result.cancelled == 3
        assert result.successful == 0
        assert result.failed == 0
        assert len(result.results) == 3
        assert all(r.cancelled for r in result.results)

    async def test_empty_pairs_list_returns_zero_total(self):
        """Empty session_pairs list returns all-zero BatchAnalysisResult."""
        from snore.services.analysis_facade import BatchAnalysisCoordinator

        coord = BatchAnalysisCoordinator()
        result = await coord.submit(
            session_pairs=[],
            profile_id=1,
            store_results=False,
        )

        assert result.total == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.cancelled == 0
        assert result.results == []


class TestStoreWithRetry:
    """Tests for the _store_with_retry helper."""

    def _make_patches(
        self, monkeypatch: pytest.MonkeyPatch, store_side_effects: list[Any]
    ) -> Any:
        """Set up all mocks needed to exercise _store_with_retry in isolation.

        Args:
            store_side_effects: List of side effects for write_svc.store_result.
                Each entry is either an exception to raise or None to succeed.
        """
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        @asynccontextmanager
        async def _fake_write_gate():
            yield

        @asynccontextmanager
        async def _fake_scope(*args, **kwargs):
            yield MagicMock()

        call_iter = iter(store_side_effects)

        async def _fake_store(computation, processing_time_ms):
            effect = next(call_iter, None)
            if isinstance(effect, BaseException):
                raise effect
            if isinstance(effect, type) and issubclass(effect, BaseException):
                raise effect()

        write_svc_mock = AsyncMock()
        write_svc_mock.store_result = _fake_store

        monkeypatch.setattr("snore.database.write_gate.write_gate", _fake_write_gate)
        monkeypatch.setattr("snore.database.session.session_scope", _fake_scope)
        monkeypatch.setattr(
            "snore.database.txn.is_sqlite_contention",
            lambda exc: "database is locked" in str(exc).lower(),
        )
        monkeypatch.setattr("asyncio.sleep", AsyncMock())

        analysis_svc_patcher = patch(
            "snore.analysis.service.AnalysisService", return_value=write_svc_mock
        )

        return analysis_svc_patcher

    async def test_one_transient_lock_then_success_ends_as_success(self, monkeypatch):
        """A store that raises 'database is locked' once then succeeds is retried."""
        from unittest.mock import AsyncMock

        from sqlalchemy.exc import OperationalError

        from snore.services.analysis_facade import _store_with_retry

        lock_err = OperationalError("database is locked", None, None)
        patcher = self._make_patches(monkeypatch, [lock_err, None])

        # Override the sleep mock with one we can inspect — proves a retry happened.
        sleep_mock = AsyncMock()
        monkeypatch.setattr("asyncio.sleep", sleep_mock)

        with patcher:
            await _store_with_retry(
                profile_id=1, computation=object(), processing_time_ms=100
            )

        # Exactly one retry sleep: first attempt failed (transient), second succeeded.
        sleep_mock.assert_called_once()

    async def test_persistent_lock_fails_after_bounded_attempts(self, monkeypatch):
        """A store that always raises 'database is locked' fails after MAX_ATTEMPTS."""
        from sqlalchemy.exc import OperationalError

        from snore.database.txn import MAX_ATTEMPTS
        from snore.services.analysis_facade import _store_with_retry

        lock_err = OperationalError("database is locked", None, None)
        # Enough errors to exhaust all attempts
        patcher = self._make_patches(monkeypatch, [lock_err] * (MAX_ATTEMPTS + 1))

        with patcher, pytest.raises(OperationalError):
            await _store_with_retry(
                profile_id=1, computation=object(), processing_time_ms=100
            )

    async def test_non_contention_error_propagates_immediately(self, monkeypatch):
        """A non-lock error propagates on the first attempt without retrying."""
        from unittest.mock import AsyncMock

        from sqlalchemy.exc import OperationalError

        from snore.services.analysis_facade import _store_with_retry

        boom = OperationalError("disk I/O error", None, None)
        # Only one error in the list; retry would need more but shouldn't happen
        patcher = self._make_patches(monkeypatch, [boom])

        # Replace the sleep mock set inside _make_patches with a fresh one we can inspect.
        sleep_mock = AsyncMock()
        monkeypatch.setattr("asyncio.sleep", sleep_mock)

        with patcher, pytest.raises(OperationalError, match="disk I/O error"):
            await _store_with_retry(
                profile_id=1, computation=object(), processing_time_ms=100
            )

        # sleep must not have been called — no retry happened
        sleep_mock.assert_not_called()


class TestAnalysisFacadeTherapyDay:
    """Tests that session_date in AnalysisListItem reflects Day.date (the therapy day)."""

    async def test_early_morning_session_returns_day_date_not_start_date(
        self, async_db_session, async_test_device
    ):
        """An early-morning session returns Day.date (therapy day), not start_time.date().

        A session starting at 01:12 on Aug 10 belongs to the Aug 9 therapy day.
        The facade must surface Day.date so the analysis list matches the Days view.
        """
        therapy_day_date = date(2025, 8, 9)
        start = datetime(2025, 8, 10, 1, 12, 0)

        day = Day(
            device_id=async_test_device.id,
            date=therapy_day_date,
            total_therapy_hours=7.5,
        )
        async_db_session.add(day)
        await async_db_session.flush()

        sess = Session(
            device_id=async_test_device.id,
            day_id=day.id,
            device_session_id="test_early_morning",
            start_time=start,
            end_time=start + timedelta(hours=7, minutes=30),
            duration_seconds=7.5 * 3600,
        )
        async_db_session.add(sess)
        await async_db_session.flush()

        ar = AnalysisResult(
            session_id=sess.id,
            timestamp_start=sess.start_time,
            timestamp_end=sess.end_time,
            programmatic_result_json={"version": 1},
            created_at=datetime.now(UTC),
        )
        async_db_session.add(ar)
        await async_db_session.flush()

        service = AnalysisFacade(async_db_session, profile_id=1)
        results = await service.list_sessions_with_status()

        assert len(results) == 1
        # Must be therapy_day (Aug 9), NOT start_time.date() (Aug 10).
        assert results[0].session_date == therapy_day_date
        assert results[0].session_date != start.date()


class TestChunkedIdBinds:
    """Fix #280: unbounded IN(...) id lists are chunked under SQLite's bind cap.

    Each test forces multi-chunk execution via ``ID_CHUNK_SIZE=2`` so the combine
    logic (rowcount-sum / dict-merge / scalar-sum / concat + re-sort) is exercised
    across at least one chunk boundary — a regression that dropped chunks or
    mis-combined them would change these assertions.
    """

    async def test_delete_all_versions_sums_rowcount_across_chunks(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """all_versions delete over a chunk boundary removes every row, summing rowcount."""
        from sqlalchemy import func  # noqa: PLC0415
        from sqlalchemy import select as sa_select

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        base = date(2025, 3, 1)
        sessions = []
        for i in range(3):  # 3 sessions → 2 chunks at size 2
            _, sess = await _create_session_with_analysis(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                num_analyses=2,
            )
            sessions.append(sess)
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis(
            [s.id for s in sessions], all_versions=True
        )

        assert deleted == 6  # 3 sessions × 2 versions, summed across chunks

        remaining = (
            await async_db_session.execute(
                sa_select(func.count()).select_from(AnalysisResult)
            )
        ).scalar()
        assert remaining == 0

    async def test_delete_stale_versions_chunks_inner_stale_ids(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Stale delete chunks stale_ids inside a session chunk and sums across sessions."""
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        # Session 1: 3 stale rows (helper's default json lacks 'identity' → stale)
        # plus 1 current row.  The 3 stale ids straddle the inner id-chunk boundary
        # (size 2); the current row must survive.
        _, sess1 = await _create_session_with_analysis(
            async_db_session, async_test_device, date(2025, 4, 1), num_analyses=3
        )
        current_ar = AnalysisResult(
            session_id=sess1.id,
            timestamp_start=sess1.start_time,
            timestamp_end=sess1.end_time,
            engine_versions_json=_current_engine_json(),
            created_at=datetime.now(UTC) + timedelta(hours=1),
        )
        async_db_session.add(current_ar)
        # Two more sessions → the session list also straddles the outer session
        # chunk boundary (3 sessions at chunk size 2).
        _, sess2 = await _create_session_with_analysis(
            async_db_session, async_test_device, date(2025, 4, 2), num_analyses=1
        )
        _, sess3 = await _create_session_with_analysis(
            async_db_session, async_test_device, date(2025, 4, 3), num_analyses=2
        )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis(
            [sess1.id, sess2.id, sess3.id], stale_versions=True
        )

        assert deleted == 6  # 3 + 1 + 2 stale rows, summed across every chunk

        surviving = set(
            (await async_db_session.execute(sa_select(AnalysisResult.id)))
            .scalars()
            .all()
        )
        assert surviving == {current_ar.id}  # only the current row survives

    async def test_get_delete_preview_merges_counts_and_sums_patterns(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Preview merges per-session version counts and sums pattern counts across chunks."""
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        base = date(2025, 5, 1)
        version_counts = [1, 2, 3]  # 3 sessions → 2 chunks at size 2
        sessions = []
        for i, n in enumerate(version_counts):
            _, sess = await _create_session_with_analysis(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                num_analyses=n,
            )
            sessions.append(sess)

        # One DetectedPattern per AnalysisResult so the summed pattern count equals
        # the total analysis rows (1 + 2 + 3 = 6).
        ar_ids = (
            (await async_db_session.execute(sa_select(AnalysisResult.id)))
            .scalars()
            .all()
        )
        for ar_id in ar_ids:
            async_db_session.add(
                DetectedPattern(
                    analysis_result_id=ar_id,
                    pattern_id="p",
                    start_time=datetime(2025, 5, 1),
                    confidence=0.9,
                    detected_by="programmatic",
                )
            )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(
            [s.id for s in sessions], all_versions=True
        )

        assert preview.sessions_with_analysis == 3
        assert preview.total_analysis_records == 6  # 1 + 2 + 3, merged dict
        assert preview.patterns_count == 6  # summed across chunks
        counts = {d.id: d.version_count for d in preview.session_details}
        assert counts == {
            sessions[0].id: 1,
            sessions[1].id: 2,
            sessions[2].id: 3,
        }

    async def test_run_batch_analysis_reorders_by_day_date_after_chunking(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Chunked session_id batch concat is re-sorted by day_date ascending."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        from snore.services.schemas import BatchAnalysisResult  # noqa: PLC0415

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        # day_date order differs from insertion (id) order, so an unsorted concat
        # would be detectable.  5 sessions → 3 chunks at size 2.
        days = [
            date(2025, 6, 5),
            date(2025, 6, 1),
            date(2025, 6, 4),
            date(2025, 6, 2),
            date(2025, 6, 3),
        ]
        sessions = []
        for d in days:
            _, sess = await _create_session_with_analysis(
                async_db_session, async_test_device, d, num_analyses=0
            )
            sessions.append(sess)
        await async_db_session.commit()

        captured: dict[str, Any] = {}

        async def fake_submit(**kwargs):
            captured["session_pairs"] = kwargs["session_pairs"]
            return BatchAnalysisResult(
                total=0, successful=0, failed=0, cancelled=0, results=[]
            )

        coord_mock = MagicMock()
        coord_mock.submit = fake_submit
        coord_mock.progress = (0, 0)

        monkeypatch.setattr(
            "snore.database.session.session_scope",
            _make_session_scope_patcher(async_db_session),
        )
        monkeypatch.setattr(
            "snore.services.analysis_facade.BatchAnalysisCoordinator",
            lambda: coord_mock,
        )

        facade = AnalysisFacade(async_db_session, profile_id=1)
        await facade.run_batch_analysis(session_ids=[s.id for s in sessions])

        day_dates = [day_date for _sid, day_date in captured["session_pairs"]]
        assert day_dates == [
            date(2025, 6, 1),
            date(2025, 6, 2),
            date(2025, 6, 3),
            date(2025, 6, 4),
            date(2025, 6, 5),
        ]

    async def test_delete_latest_only_dedupes_duplicate_ids_across_chunks(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """A duplicate id split across chunks must not delete a second version.

        Without the entry-point dedup, [s1, s2, s1] chunks to [s1, s2] then [s1]:
        chunk 1 deletes s1's latest, chunk 2 re-ranks the survivors and deletes
        s1's now-latest — a second, unintended version.  Deduping to [s1, s2]
        restores the single-IN "one latest per session" semantics.
        """
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        base = date(2025, 7, 1)
        sessions = []
        for i in range(3):  # third session proves scoping — it must stay untouched
            _, sess = await _create_session_with_analysis(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                num_analyses=2,
            )
            sessions.append(sess)
        sess1, sess2, sess3 = sessions
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)

        # Preview must not double-count the duplicated id: [s1, s2, s1] == [s1, s2].
        dup_preview = await service.get_delete_preview(
            session_ids=[sess1.id, sess2.id, sess1.id]
        )
        deduped_preview = await service.get_delete_preview(
            session_ids=[sess1.id, sess2.id]
        )
        assert dup_preview.sessions_with_analysis == 2
        assert len(dup_preview.session_details) == 2
        assert {d.id for d in dup_preview.session_details} == {sess1.id, sess2.id}
        assert (
            dup_preview.sessions_with_analysis == deduped_preview.sessions_with_analysis
        )
        assert (
            dup_preview.total_analysis_records == deduped_preview.total_analysis_records
        )

        deleted = await service.delete_analysis(
            [sess1.id, sess2.id, sess1.id], all_versions=False, stale_versions=False
        )

        # One latest per distinct session — never two for the duplicated s1.
        assert deleted == 2

        async def _versions(session_id: int) -> list[int]:
            rows = (
                (
                    await async_db_session.execute(
                        sa_select(AnalysisResult).filter_by(session_id=session_id)
                    )
                )
                .scalars()
                .all()
            )
            return sorted(r.programmatic_result_json["version"] for r in rows)

        # s1's older version survives (bug would delete it); s2 likewise; s3 untouched.
        assert await _versions(sess1.id) == [1]
        assert await _versions(sess2.id) == [1]
        assert await _versions(sess3.id) == [1, 2]

    async def test_delete_latest_only_sums_rowcount_across_chunks(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """latest-only delete over a chunk boundary removes one row per session.

        Proves the per-chunk rowcount accumulation for the default (latest-only)
        branch: 3 distinct sessions at chunk size 2 span two chunks.
        """
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        base = date(2025, 8, 1)
        sessions = []
        for i in range(3):  # 3 sessions → 2 chunks at size 2
            _, sess = await _create_session_with_analysis(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                num_analyses=2,
            )
            sessions.append(sess)
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        deleted = await service.delete_analysis(
            [s.id for s in sessions], all_versions=False, stale_versions=False
        )

        assert deleted == 3  # one latest per session, summed across chunks

        for sess in sessions:
            versions = sorted(
                r.programmatic_result_json["version"]
                for r in (
                    await async_db_session.execute(
                        sa_select(AnalysisResult).filter_by(session_id=sess.id)
                    )
                )
                .scalars()
                .all()
            )
            assert versions == [1]  # only the older version remains

    async def test_stale_versions_preview_concats_rows_and_sums_patterns_across_chunks(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Stale preview concats all_rows and sums stale patterns across chunks.

        3 sessions (one stale + one current row each) span two chunks at size 2.
        Only the stale rows and their patterns are counted; the current rows'
        patterns are excluded.
        """
        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        base = date(2025, 9, 1)
        sessions = []
        stale_pattern_counts = [2, 1, 2]  # 5 stale patterns, spanning chunks
        for i, n_patterns in enumerate(stale_pattern_counts):
            _, sess, stale_ar, current_ar = await _create_session_with_mixed_analyses(
                async_db_session, async_test_device, base + timedelta(days=i)
            )
            sessions.append(sess)
            for _ in range(n_patterns):
                async_db_session.add(
                    DetectedPattern(
                        analysis_result_id=stale_ar.id,
                        pattern_id="p",
                        start_time=datetime(2025, 9, 1),
                        confidence=0.9,
                        detected_by="programmatic",
                    )
                )
            # A pattern on the current row must NOT be counted.
            async_db_session.add(
                DetectedPattern(
                    analysis_result_id=current_ar.id,
                    pattern_id="c",
                    start_time=datetime(2025, 9, 1),
                    confidence=0.5,
                    detected_by="programmatic",
                )
            )
        await async_db_session.commit()

        service = AnalysisFacade(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(
            session_ids=[s.id for s in sessions], stale_versions=True
        )

        assert preview.sessions_with_analysis == 3  # one stale row per session
        assert preview.total_analysis_records == 6  # all rows, concat across chunks
        assert preview.records_to_delete == 3  # only the stale rows
        assert preview.patterns_count == 5  # 2+1+2 stale patterns; current excluded
        assert {d.id for d in preview.session_details} == {s.id for s in sessions}

    async def test_get_delete_preview_respects_real_bind_cap_without_monkeypatch(
        self, async_db_session
    ):
        """1500 non-existent ids never trip SQLite's bind cap at the real chunk size.

        No monkeypatch: a single unchunked IN(1..1500) would overrun SQLite's
        999-parameter floor.  Chunks of 500 stay under the cap, so the preview
        returns a normal empty result instead of raising OperationalError.
        """
        service = AnalysisFacade(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(session_ids=list(range(1, 1501)))

        assert preview.sessions_with_analysis == 0
        assert preview.total_analysis_records == 0
        assert preview.records_to_delete == 0
        assert preview.patterns_count == 0
        assert preview.session_details == []
