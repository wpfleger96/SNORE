"""Unit tests for snore.analysis.queries latest-row lookups."""

from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.queries import latest_analysis_ids, latest_analysis_rows
from snore.database.models import AnalysisResult, Day, Device, Session


async def _seed_session_with_versions(
    db: AsyncSession, device: Device, day_date: date, num_versions: int
) -> Session:
    """Create a session with ``num_versions`` AnalysisResult rows (ascending created_at)."""
    day = Day(device_id=device.id, date=day_date, total_therapy_hours=8.0)
    db.add(day)
    await db.flush()
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"q_{day_date.isoformat()}",
        start_time=datetime.combine(day_date, datetime.min.time()),
        end_time=datetime.combine(day_date, datetime.min.time()) + timedelta(hours=8),
        duration_seconds=28800,
    )
    db.add(sess)
    await db.flush()
    for i in range(num_versions):
        db.add(
            AnalysisResult(
                session_id=sess.id,
                timestamp_start=sess.start_time,
                timestamp_end=sess.end_time,
                programmatic_result_json={"version": i + 1},
                engine_versions_json={"version": i + 1},
                created_at=datetime.now(UTC) + timedelta(minutes=i),
            )
        )
    await db.flush()
    return sess


class TestLatestAnalysisIdsChunking:
    """latest_analysis_ids chunks its id list and merges disjoint per-session results."""

    async def test_merges_disjoint_per_session_results_across_chunks(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Covers every session and picks the latest AnalysisResult id per session.

        With ``ID_CHUNK_SIZE=2`` and 3 sessions the id list spans two chunks; the
        per-chunk ranked results have disjoint session keys and must merge into a
        complete map, each value being that session's newest row.
        """
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        sessions = []
        for i in range(3):  # 3 sessions → 2 chunks at size 2
            sess = await _seed_session_with_versions(
                async_db_session,
                async_test_device,
                date(2025, 7, 1) + timedelta(days=i),
                num_versions=2,
            )
            sessions.append(sess)
        await async_db_session.commit()

        result = await latest_analysis_ids(async_db_session, [s.id for s in sessions])

        assert set(result.keys()) == {s.id for s in sessions}

        # Each mapped id must be the (created_at DESC, id DESC) winner for its session.
        for sess in sessions:
            expected_latest = (
                (
                    await async_db_session.execute(
                        sa_select(AnalysisResult.id)
                        .where(AnalysisResult.session_id == sess.id)
                        .order_by(
                            AnalysisResult.created_at.desc(),
                            AnalysisResult.id.desc(),
                        )
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            assert result[sess.id] == expected_latest

    async def test_empty_input_returns_empty_map(self, async_db_session):
        """Empty session id list short-circuits to an empty map without querying."""
        result = await latest_analysis_ids(async_db_session, [])
        assert result == {}


class TestLatestAnalysisRows:
    """latest_analysis_rows yields one latest row per analyzed session."""

    async def test_id_list_and_select_scopes_yield_latest_rows(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Chunked id lists and a session-id SELECT give the same latest rows,
        and extra columns are read from that same row."""
        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)
        sessions = [
            await _seed_session_with_versions(
                async_db_session,
                async_test_device,
                date(2025, 8, 1) + timedelta(days=i),
                num_versions=i + 1,
            )
            for i in range(3)
        ]
        unanalyzed = await _seed_session_with_versions(
            async_db_session, async_test_device, date(2025, 8, 9), num_versions=0
        )
        await async_db_session.commit()
        version = func.json_extract(
            AnalysisResult.programmatic_result_json, "$.version"
        ).label("version")
        latest_id = await latest_analysis_ids(
            async_db_session, [s.id for s in sessions]
        )
        expected = {
            s.id: (latest_id[s.id], {"version": i + 1}, i + 1)
            for i, s in enumerate(sessions)
        }
        session_ids = [s.id for s in sessions] + [unanalyzed.id]

        for scope in (
            session_ids,
            select(Session.id).where(Session.device_id == async_test_device.id),
        ):
            rows = [
                row
                async for row in latest_analysis_rows(async_db_session, scope, version)
            ]

            assert {
                row.session_id: (row.id, row.engine_versions_json, row.version)
                for row in rows
            } == expected
            assert len(rows) == len(sessions)
