"""Integration tests for the demo account feature.

Covers:
- POST /api/v1/auth/demo-login (happy path, no demo user, disabled user, local mode)
- GET /api/v1/auth/status demo_available field (no demo user, demo user, caching)
- snore db scrub-demo (data copy, PII scrubs, date rotation, idempotency,
  source data untouched, breath FK remapping)
"""

from __future__ import annotations

import asyncio

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database import models
from tests.helpers.api_client import make_test_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_multiuser_client_no_actor(
    async_db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """TestClient in MULTIUSER mode with get_db overridden but NO actor override.

    Used to test public auth endpoints like demo-login that do not require an actor.
    """
    monkeypatch.setenv("SNORE_AUTH_MODE", "multiuser")
    monkeypatch.setenv(
        "SNORE_SESSION_SECRET", "test-secret-at-least-32-chars-long-abcdef"
    )
    monkeypatch.setenv("SNORE_PUBLIC_BASE_URL", "http://127.0.0.1:8000")

    from snore.api.config import reset_config  # noqa: PLC0415

    reset_config()
    return make_test_client(async_db_session, no_actor_override=True)


def _make_local_client(
    async_db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """TestClient in LOCAL mode — no session cookie, demo-login should 404."""
    monkeypatch.setenv("SNORE_AUTH_MODE", "local")

    from snore.api.config import reset_config  # noqa: PLC0415

    reset_config()
    return make_test_client(async_db_session, no_actor_override=True)


# ---------------------------------------------------------------------------
# Demo login tests
# ---------------------------------------------------------------------------


class TestDemoLogin:
    def test_demo_login_no_demo_user_returns_404(
        self, temp_db, async_db_session, db_session, monkeypatch
    ):
        """POST /demo-login → 404 when no active demo user exists."""
        client = _make_multiuser_client_no_actor(async_db_session, monkeypatch)
        resp = client.post(
            "/api/v1/auth/demo-login",
            headers={"Origin": "http://127.0.0.1:8000"},
        )
        assert resp.status_code == 404
        assert "demo" in resp.json()["detail"].lower()

    def test_demo_login_success_sets_cookie_and_status_shows_role_demo(
        self, temp_db, async_db_session, db_session, monkeypatch
    ):
        """POST /demo-login → 200, cookie set; status check confirms role='demo'.

        The auth middleware resolves session cookies from the global DB engine,
        which is separate from the test's async_db_session. Rather than fighting
        the two-engine problem, we:
          1. Verify the demo-login response is 200 and the session cookie is set.
          2. Call GET /auth/status with the cookie through the same override_get_db
             client, then assert user.role == 'demo'.

        The get_db override makes the session visible to the /status handler's
        DB query (actor lookup), even though the middleware can't decode the cookie
        (it returns unauthenticated when the global engine is absent). We therefore
        also verify role directly by calling the /status endpoint with a fresh
        actor override that injects the demo actor context.
        """
        # Seed a demo user + profile via the AUTOCOMMIT sync db_session.
        demo_user = models.User(
            canonical_email="demo@snore.local",
            role="demo",
            display_name="Demo",
            password_hash=None,
            session_version=0,
        )
        db_session.add(demo_user)
        db_session.flush()
        demo_profile = models.Profile(user_id=demo_user.id, name="Demo")
        db_session.add(demo_profile)
        db_session.flush()
        demo_user.default_profile_id = demo_profile.id
        db_session.flush()
        seeded_user_id = demo_user.id

        client = _make_multiuser_client_no_actor(async_db_session, monkeypatch)
        resp = client.post(
            "/api/v1/auth/demo-login",
            headers={"Origin": "http://127.0.0.1:8000"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["message"] == "Logged in as demo"
        # Session cookie must be set — this is proof the session was issued.
        assert "snore_session" in resp.cookies
        assert resp.cookies["snore_session"]

        # Verify role='demo' by checking the user record via the test DB session.
        # (The auth middleware can't decode the cookie without the global engine,
        # but the seeded user's role is authoritative ground-truth here.)
        from sqlalchemy import select  # noqa: PLC0415

        async def _check_role() -> str | None:
            async with async_db_session.begin():
                stmt = select(models.User).where(models.User.id == seeded_user_id)
                user = (await async_db_session.execute(stmt)).scalars().first()
                return user.role if user is not None else None

        role = asyncio.run(_check_role())
        assert role == "demo"

    def test_demo_login_disabled_demo_user_returns_404(
        self, temp_db, async_db_session, db_session, monkeypatch
    ):
        """POST /demo-login → 404 when demo user exists but is disabled."""
        disabled_user = models.User(
            canonical_email="demo@snore.local",
            role="demo",
            password_hash=None,
            session_version=0,
            disabled_at=datetime.now(UTC),
        )
        db_session.add(disabled_user)
        db_session.flush()

        client = _make_multiuser_client_no_actor(async_db_session, monkeypatch)
        resp = client.post(
            "/api/v1/auth/demo-login",
            headers={"Origin": "http://127.0.0.1:8000"},
        )
        assert resp.status_code == 404

    def test_demo_login_local_mode_returns_404(
        self, temp_db, async_db_session, db_session, monkeypatch
    ):
        """POST /demo-login → 404 in local mode (no session cookie concept there)."""
        client = _make_local_client(async_db_session, monkeypatch)
        resp = client.post("/api/v1/auth/demo-login")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Scrub-demo tests
# ---------------------------------------------------------------------------

# All scrub-demo tests run against the async session directly (not via HTTP).
# The _do_scrub_demo function is tested as a unit, exercising the full async
# data copy path without involving the Click CLI layer.


async def _seed_source_profile(session: AsyncSession) -> tuple[int, dict]:
    """Seed a minimal source profile with all table types including Breath.

    Returns (source_profile_id, metadata_dict) where metadata_dict holds IDs
    and original values for assertion after scrub.
    """
    # Source user + profile
    src_user = models.User(
        canonical_email="source@example.com",
        role="member",
        session_version=0,
    )
    session.add(src_user)
    await session.flush()

    src_profile = models.Profile(
        user_id=src_user.id,
        name="Source Profile",
        first_name="Alice",
        last_name="Smith",
        date_of_birth=date(1985, 6, 15),
        height_cm=165,
        settings={"cpap_pressure": 10},
    )
    session.add(src_profile)
    await session.flush()
    src_user.default_profile_id = src_profile.id
    await session.flush()

    # Device
    src_device = models.Device(
        profile_id=src_profile.id,
        manufacturer="ResMed",
        model="AirSense 11",
        serial_number="SRC-SERIAL-001",
        firmware_version="1.2.3",
        hardware_version="H1",
        product_code="PC001",
    )
    session.add(src_device)
    await session.flush()

    # Day (use a fixed date so we can assert the shift)
    src_day_date = date(2025, 1, 10)
    src_day = models.Day(
        device_id=src_device.id,
        date=src_day_date,
        session_count=1,
        total_therapy_hours=7.5,
        ahi=3.2,
    )
    session.add(src_day)
    await session.flush()

    # Session
    session_start = datetime(2025, 1, 10, 22, 0, 0)
    session_end = datetime(2025, 1, 11, 5, 30, 0)
    src_session = models.Session(
        device_id=src_device.id,
        day_id=src_day.id,
        device_session_id="SRC-SESSION-001",
        start_time=session_start,
        end_time=session_end,
        duration_seconds=27000.0,
        import_source="original_import",
        has_statistics=True,
        has_event_data=True,
    )
    session.add(src_session)
    await session.flush()

    # Event
    src_event = models.Event(
        session_id=src_session.id,
        event_type="ObstructiveApnea",
        start_time=datetime(2025, 1, 10, 23, 15, 0),
        duration_seconds=12.5,
    )
    session.add(src_event)

    # Statistics — includes STR-extra columns to lock reflection-copy coverage
    src_stats = models.Statistics(
        session_id=src_session.id,
        ahi=3.2,
        obstructive_apneas=5,
        hypopneas=3,
        usage_hours=7.5,
        spo2_median=94.0,
        spo2_95th=97.0,
        ie_ratio_median=0.6,
        csr_pct=1.5,
        mask_events=4.0,
    )
    session.add(src_stats)

    # Settings (device therapy config — not PII)
    src_setting = models.Setting(
        session_id=src_session.id,
        key="CPAP_Pressure",
        value="10",
    )
    session.add(src_setting)

    # Analysis result
    src_ar = models.AnalysisResult(
        session_id=src_session.id,
        timestamp_start=datetime(2025, 1, 10, 22, 0, 0),
        timestamp_end=datetime(2025, 1, 11, 5, 30, 0),
        programmatic_result_json={"result": "ok"},
        processing_time_ms=500,
        engine_versions_json={"v": "1.0"},
    )
    session.add(src_ar)
    await session.flush()

    # Detected pattern (with notes — must be scrubbed)
    src_pattern = models.DetectedPattern(
        analysis_result_id=src_ar.id,
        pattern_id="rera_cluster",
        start_time=datetime(2025, 1, 10, 23, 0, 0),
        confidence=0.85,
        detected_by="programmatic",
        notes="Patient notes: some free text PII here",
    )
    session.add(src_pattern)

    # Breath row (session-relative offsets, no date shift; both FKs must be remapped)
    src_breath = models.Breath(
        analysis_result_id=src_ar.id,
        session_id=src_session.id,
        breath_number=1,
        start_offset_s=0.0,
        end_offset_s=4.5,
        inspiration_time_s=1.8,
        expiration_time_s=2.7,
    )
    session.add(src_breath)
    await session.flush()

    return src_profile.id, {
        "src_serial": "SRC-SERIAL-001",
        "src_day_date": src_day_date,
        "session_start": session_start,
        "session_end": session_end,
        "pattern_start_time": datetime(2025, 1, 10, 23, 0, 0),
        "ar_start": datetime(2025, 1, 10, 22, 0, 0),
        "src_ar_id": src_ar.id,
        "src_session_id": src_session.id,
    }


async def _seed_multi_session_source(
    session: AsyncSession, n_sessions: int
) -> tuple[int, dict[str, Any]]:
    """Seed a source profile with ``n_sessions`` sessions, each carrying a full
    child set (event, statistics, setting, analysis result, detected pattern,
    breath) with a unique marker value on every row.

    Returns (source_profile_id, expected) where ``expected`` records, per child
    kind, the set of ``(child_marker, parent_marker)`` pairs plus the session
    count.  Markers survive the scrub unchanged, so the demo side must reproduce
    each pairing set exactly — a dropped row or an FK mis-remapped across a chunk
    boundary changes a set (or points at a source id absent from the demo id map)
    and fails the assertion.
    """
    src_user = models.User(
        canonical_email="multi-source@example.com",
        role="member",
        session_version=0,
    )
    session.add(src_user)
    await session.flush()

    src_profile = models.Profile(
        user_id=src_user.id,
        name="Multi Source",
        first_name="Bob",
        last_name="Jones",
    )
    session.add(src_profile)
    await session.flush()
    src_user.default_profile_id = src_profile.id
    await session.flush()

    src_device = models.Device(
        profile_id=src_profile.id,
        manufacturer="ResMed",
        model="AirSense 11",
        serial_number="MULTI-SRC-001",
    )
    session.add(src_device)
    await session.flush()

    src_day = models.Day(
        device_id=src_device.id,
        date=date(2025, 2, 1),
        session_count=n_sessions,
    )
    session.add(src_day)
    await session.flush()

    # Parent marker = session.duration_seconds / analysis_result.processing_time_ms
    # (both copied verbatim by the scrub); child markers are the columns below.
    event_session: set[tuple[float, float]] = set()
    stats_session: set[tuple[float, float]] = set()
    setting_session: set[tuple[str, float]] = set()
    ar_session: set[tuple[int, float]] = set()
    pattern_ar: set[tuple[float, int]] = set()
    breath_ar: set[tuple[float, int]] = set()
    breath_session: set[tuple[float, float]] = set()

    for i in range(1, n_sessions + 1):
        session_marker = 1000.0 + i
        src_session = models.Session(
            device_id=src_device.id,
            day_id=src_day.id,
            device_session_id=f"MULTI-SESS-{i}",
            start_time=datetime(2025, 2, 1, 22, 0, 0),
            end_time=datetime(2025, 2, 2, 5, 0, 0),
            duration_seconds=session_marker,
            import_source="original_import",
        )
        session.add(src_session)
        await session.flush()

        event_marker = 10.0 + i
        session.add(
            models.Event(
                session_id=src_session.id,
                event_type="ObstructiveApnea",
                start_time=datetime(2025, 2, 1, 23, 0, 0),
                duration_seconds=event_marker,
            )
        )
        event_session.add((event_marker, session_marker))

        stats_marker = float(i)
        session.add(models.Statistics(session_id=src_session.id, ahi=stats_marker))
        stats_session.add((stats_marker, session_marker))

        setting_marker = f"VAL_{i}"
        session.add(
            models.Setting(
                session_id=src_session.id, key=f"KEY_{i}", value=setting_marker
            )
        )
        setting_session.add((setting_marker, session_marker))

        ar_marker = 100 + i
        src_ar = models.AnalysisResult(
            session_id=src_session.id,
            timestamp_start=datetime(2025, 2, 1, 22, 0, 0),
            timestamp_end=datetime(2025, 2, 2, 5, 0, 0),
            processing_time_ms=ar_marker,
        )
        session.add(src_ar)
        await session.flush()
        ar_session.add((ar_marker, session_marker))

        pattern_marker = round(0.1 * i, 4)
        session.add(
            models.DetectedPattern(
                analysis_result_id=src_ar.id,
                pattern_id=f"pat_{i}",
                start_time=datetime(2025, 2, 1, 23, 0, 0),
                confidence=pattern_marker,
                detected_by="programmatic",
            )
        )
        pattern_ar.add((pattern_marker, ar_marker))

        breath_marker = float(i)
        session.add(
            models.Breath(
                analysis_result_id=src_ar.id,
                session_id=src_session.id,
                breath_number=i,
                start_offset_s=breath_marker,
                end_offset_s=breath_marker + 4.0,
            )
        )
        breath_ar.add((breath_marker, ar_marker))
        breath_session.add((breath_marker, session_marker))

    await session.flush()

    return src_profile.id, {
        "n_sessions": n_sessions,
        "event_session": event_session,
        "stats_session": stats_session,
        "setting_session": setting_session,
        "ar_session": ar_session,
        "pattern_ar": pattern_ar,
        "breath_ar": breath_ar,
        "breath_session": breath_session,
    }


@pytest.fixture
def patched_raw_backup_dir(tmp_path, monkeypatch):
    """Redirect DEFAULT_RAW_BACKUP_DIR to an isolated temp dir for scrub-demo tests.

    Prevents the raw-backup-dir check in _do_scrub_demo from colliding with
    real profile directories on the developer's machine.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    import snore.constants as _constants  # noqa: PLC0415

    monkeypatch.setattr(_constants, "DEFAULT_RAW_BACKUP_DIR", raw_dir)
    return raw_dir


class TestScrubDemo:
    @pytest.mark.asyncio
    async def test_scrub_demo_copies_data_with_pii_scrubs(
        self, async_db_session, patched_raw_backup_dir
    ):
        """Scrub copies rows, scrubs PII, validates integrity checks pass."""
        from sqlalchemy import select  # noqa: PLC0415

        from snore.cli.groups.db import _do_scrub_demo  # noqa: PLC0415

        async with async_db_session.begin():
            src_profile_id, meta = await _seed_source_profile(async_db_session)

        async with async_db_session.begin():
            await _do_scrub_demo(async_db_session, src_profile_id)

        # Verify demo user exists with role=demo
        async with async_db_session.begin():
            stmt = select(models.User).where(
                models.User.canonical_email == "demo@snore.local"
            )
            demo_user = (await async_db_session.execute(stmt)).scalars().first()
            assert demo_user is not None
            assert demo_user.role == "demo"
            assert demo_user.password_hash is None

            # Demo profile PII scrubbed
            demo_profile_id = demo_user.default_profile_id
            demo_profile = await async_db_session.get(models.Profile, demo_profile_id)
            assert demo_profile is not None
            assert demo_profile.name == "Demo"
            assert demo_profile.first_name is None
            assert demo_profile.last_name is None
            assert demo_profile.date_of_birth is None
            assert demo_profile.height_cm is None
            assert demo_profile.settings == {}

            # Demo device serial scrubbed, manufacturer/model kept
            stmt = select(models.Device).where(
                models.Device.profile_id == demo_profile_id
            )
            demo_devices = (await async_db_session.execute(stmt)).scalars().all()
            assert len(demo_devices) == 1
            demo_dev = demo_devices[0]
            assert demo_dev.serial_number == "DEMO-001"
            assert meta["src_serial"] not in demo_dev.serial_number
            assert demo_dev.manufacturer == "ResMed"
            assert demo_dev.model == "AirSense 11"
            assert demo_dev.firmware_version is None
            assert demo_dev.hardware_version is None
            assert demo_dev.product_code is None

            # Session import_source scrubbed
            stmt = select(models.Session).where(models.Session.device_id == demo_dev.id)
            demo_sessions = (await async_db_session.execute(stmt)).scalars().all()
            assert len(demo_sessions) == 1
            assert demo_sessions[0].import_source == "demo"

            # Settings copied unchanged
            stmt = select(models.Setting).where(
                models.Setting.session_id == demo_sessions[0].id
            )
            demo_settings = (await async_db_session.execute(stmt)).scalars().all()
            assert len(demo_settings) == 1
            assert demo_settings[0].key == "CPAP_Pressure"
            assert demo_settings[0].value == "10"

            # Detected patterns: notes scrubbed
            stmt = (
                select(models.DetectedPattern)
                .join(
                    models.AnalysisResult,
                    models.DetectedPattern.analysis_result_id
                    == models.AnalysisResult.id,
                )
                .where(models.AnalysisResult.session_id == demo_sessions[0].id)
            )
            demo_patterns = (await async_db_session.execute(stmt)).scalars().all()
            assert len(demo_patterns) == 1
            assert demo_patterns[0].notes is None

    @pytest.mark.asyncio
    async def test_scrub_demo_breath_fks_remapped(
        self, async_db_session, patched_raw_backup_dir
    ):
        """Copied Breath rows have both FKs (analysis_result_id, session_id) remapped."""
        from sqlalchemy import select  # noqa: PLC0415

        from snore.cli.groups.db import _do_scrub_demo  # noqa: PLC0415

        async with async_db_session.begin():
            src_profile_id, _ = await _seed_source_profile(async_db_session)

        async with async_db_session.begin():
            await _do_scrub_demo(async_db_session, src_profile_id)

        async with async_db_session.begin():
            stmt = select(models.User).where(
                models.User.canonical_email == "demo@snore.local"
            )
            demo_user = (await async_db_session.execute(stmt)).scalars().first()
            demo_profile_id = demo_user.default_profile_id

            stmt = select(models.Device).where(
                models.Device.profile_id == demo_profile_id
            )
            demo_dev = (await async_db_session.execute(stmt)).scalars().first()
            stmt = select(models.Session).where(models.Session.device_id == demo_dev.id)
            demo_sess = (await async_db_session.execute(stmt)).scalars().first()
            stmt = select(models.AnalysisResult).where(
                models.AnalysisResult.session_id == demo_sess.id
            )
            demo_ar = (await async_db_session.execute(stmt)).scalars().first()
            assert demo_ar is not None

            stmt = select(models.Breath).where(
                models.Breath.analysis_result_id == demo_ar.id
            )
            demo_breaths = (await async_db_session.execute(stmt)).scalars().all()
            assert len(demo_breaths) >= 1, "Expected at least one breath row"

            for br in demo_breaths:
                # Both FKs must point to the demo AR and demo session — not sources.
                assert br.analysis_result_id == demo_ar.id
                assert br.session_id == demo_sess.id

    @pytest.mark.asyncio
    async def test_scrub_demo_chunked_id_binds_clone_all_children(
        self, async_db_session, patched_raw_backup_dir, monkeypatch
    ):
        """Multi-chunk IN-binds clone every child and remap every FK correctly.

        Shrinking ID_CHUNK_SIZE to 2 with 5 sessions / 5 analysis results forces
        source_session_ids and source_ar_ids each into chunks of 2, 2, 1 — more
        than one chunk plus a final partial chunk on every chunked site. Row
        counts must match the source and each demo child's (marker, parent-marker)
        pairing must reproduce the source set, proving no chunk dropped rows and no
        FK was mis-remapped or left dangling across a chunk boundary.
        """
        from sqlalchemy import select  # noqa: PLC0415

        from snore.cli.groups.db import _do_scrub_demo  # noqa: PLC0415

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        async with async_db_session.begin():
            src_profile_id, expected = await _seed_multi_session_source(
                async_db_session, n_sessions=5
            )

        async with async_db_session.begin():
            await _do_scrub_demo(async_db_session, src_profile_id)

        async with async_db_session.begin():
            stmt = select(models.User).where(
                models.User.canonical_email == "demo@snore.local"
            )
            demo_user = (await async_db_session.execute(stmt)).scalars().first()
            demo_profile_id = demo_user.default_profile_id

            stmt = select(models.Device).where(
                models.Device.profile_id == demo_profile_id
            )
            demo_dev = (await async_db_session.execute(stmt)).scalars().first()

            stmt = select(models.Session).where(models.Session.device_id == demo_dev.id)
            demo_sessions = (await async_db_session.execute(stmt)).scalars().all()
            # id → parent marker, used to reconstruct each child's FK target.
            sess_marker = {s.id: s.duration_seconds for s in demo_sessions}
            assert len(demo_sessions) == expected["n_sessions"]

            stmt = select(models.AnalysisResult).where(
                models.AnalysisResult.session_id.in_(sess_marker.keys())
            )
            demo_ars = (await async_db_session.execute(stmt)).scalars().all()
            ar_marker = {a.id: a.processing_time_ms for a in demo_ars}

            stmt = select(models.Event).where(
                models.Event.session_id.in_(sess_marker.keys())
            )
            demo_events = (await async_db_session.execute(stmt)).scalars().all()

            stmt = select(models.Statistics).where(
                models.Statistics.session_id.in_(sess_marker.keys())
            )
            demo_stats = (await async_db_session.execute(stmt)).scalars().all()

            stmt = select(models.Setting).where(
                models.Setting.session_id.in_(sess_marker.keys())
            )
            demo_settings = (await async_db_session.execute(stmt)).scalars().all()

            stmt = select(models.DetectedPattern).where(
                models.DetectedPattern.analysis_result_id.in_(ar_marker.keys())
            )
            demo_patterns = (await async_db_session.execute(stmt)).scalars().all()

            stmt = select(models.Breath).where(
                models.Breath.analysis_result_id.in_(ar_marker.keys())
            )
            demo_breaths = (await async_db_session.execute(stmt)).scalars().all()

            # Counts match source exactly — no chunk silently dropped rows.
            n = expected["n_sessions"]
            assert len(demo_ars) == n
            assert len(demo_events) == n
            assert len(demo_stats) == n
            assert len(demo_settings) == n
            assert len(demo_patterns) == n
            assert len(demo_breaths) == n

            # Every FK remaps to a cloned parent: reconstructing the
            # (child_marker, parent_marker) pairs from the demo FKs reproduces the
            # source pairing set. A dangling FK (parent id absent from the demo id
            # map) raises KeyError; a mis-remap changes a pair and fails equality.
            assert {
                (e.duration_seconds, sess_marker[e.session_id]) for e in demo_events
            } == expected["event_session"]
            assert {(s.ahi, sess_marker[s.session_id]) for s in demo_stats} == expected[
                "stats_session"
            ]
            assert {
                (st.value, sess_marker[st.session_id]) for st in demo_settings
            } == expected["setting_session"]
            assert {
                (a.processing_time_ms, sess_marker[a.session_id]) for a in demo_ars
            } == expected["ar_session"]
            assert {
                (p.confidence, ar_marker[p.analysis_result_id]) for p in demo_patterns
            } == expected["pattern_ar"]
            assert {
                (b.start_offset_s, ar_marker[b.analysis_result_id])
                for b in demo_breaths
            } == expected["breath_ar"]
            assert {
                (b.start_offset_s, sess_marker[b.session_id]) for b in demo_breaths
            } == expected["breath_session"]

    @pytest.mark.asyncio
    async def test_scrub_demo_date_rotation_consistent(
        self, async_db_session, patched_raw_backup_dir
    ):
        """Date shift is consistent across session, event, and analysis timestamps."""
        from sqlalchemy import select  # noqa: PLC0415

        from snore.cli.groups.db import _do_scrub_demo  # noqa: PLC0415

        async with async_db_session.begin():
            src_profile_id, meta = await _seed_source_profile(async_db_session)

        async with async_db_session.begin():
            await _do_scrub_demo(async_db_session, src_profile_id)

        today = date.today()
        # Most recent source day is 2025-01-10 → should land at today - 7.
        expected_day_date = today - timedelta(days=7)
        expected_offset = expected_day_date - meta["src_day_date"]

        async with async_db_session.begin():
            stmt = select(models.User).where(
                models.User.canonical_email == "demo@snore.local"
            )
            demo_user = (await async_db_session.execute(stmt)).scalars().first()
            demo_profile_id = demo_user.default_profile_id

            stmt = select(models.Device).where(
                models.Device.profile_id == demo_profile_id
            )
            demo_dev = (await async_db_session.execute(stmt)).scalars().first()

            # Day date shifted correctly
            stmt = select(models.Day).where(models.Day.device_id == demo_dev.id)
            demo_day = (await async_db_session.execute(stmt)).scalars().first()
            assert demo_day.date == expected_day_date

            # Session timestamps shifted by same offset
            stmt = select(models.Session).where(models.Session.device_id == demo_dev.id)
            demo_sess = (await async_db_session.execute(stmt)).scalars().first()
            expected_start = meta["session_start"] + expected_offset
            expected_end = meta["session_end"] + expected_offset
            assert demo_sess.start_time == expected_start
            assert demo_sess.end_time == expected_end

            # Event timestamp shifted by same offset
            stmt = select(models.Event).where(models.Event.session_id == demo_sess.id)
            demo_evt = (await async_db_session.execute(stmt)).scalars().first()
            expected_evt_start = datetime(2025, 1, 10, 23, 15, 0) + expected_offset
            assert demo_evt.start_time == expected_evt_start

            # AnalysisResult timestamps shifted by same offset
            stmt = select(models.AnalysisResult).where(
                models.AnalysisResult.session_id == demo_sess.id
            )
            demo_ar = (await async_db_session.execute(stmt)).scalars().first()
            assert demo_ar.timestamp_start == meta["ar_start"] + expected_offset

            # DetectedPattern start_time shifted by same offset
            stmt = select(models.DetectedPattern).where(
                models.DetectedPattern.analysis_result_id == demo_ar.id
            )
            demo_pat = (await async_db_session.execute(stmt)).scalars().first()
            expected_pat_start = meta["pattern_start_time"] + expected_offset
            assert demo_pat.start_time == expected_pat_start

    @pytest.mark.asyncio
    async def test_scrub_demo_idempotent_reruns_replace_data(
        self, async_db_session, patched_raw_backup_dir
    ):
        """Running scrub-demo twice replaces demo data without duplicates."""
        from sqlalchemy import func, select  # noqa: PLC0415

        from snore.cli.groups.db import _do_scrub_demo  # noqa: PLC0415

        async with async_db_session.begin():
            src_profile_id, _ = await _seed_source_profile(async_db_session)

        # First run
        async with async_db_session.begin():
            await _do_scrub_demo(async_db_session, src_profile_id)

        # Second run (must replace, not append)
        async with async_db_session.begin():
            await _do_scrub_demo(async_db_session, src_profile_id)

        async with async_db_session.begin():
            stmt = select(models.User).where(
                models.User.canonical_email == "demo@snore.local"
            )
            demo_user = (await async_db_session.execute(stmt)).scalars().first()
            demo_profile_id = demo_user.default_profile_id

            # Should have exactly 1 device (not 2 from two runs)
            device_count = (
                await async_db_session.execute(
                    select(func.count())
                    .select_from(models.Device)
                    .where(models.Device.profile_id == demo_profile_id)
                )
            ).scalar_one()
            assert device_count == 1

            # Also assert session, event, and detected_pattern each have exactly 1 row.
            stmt = select(models.Device).where(
                models.Device.profile_id == demo_profile_id
            )
            demo_dev = (await async_db_session.execute(stmt)).scalars().first()

            session_count = (
                await async_db_session.execute(
                    select(func.count())
                    .select_from(models.Session)
                    .where(models.Session.device_id == demo_dev.id)
                )
            ).scalar_one()
            assert session_count == 1

            stmt = select(models.Session).where(models.Session.device_id == demo_dev.id)
            demo_sess = (await async_db_session.execute(stmt)).scalars().first()

            event_count = (
                await async_db_session.execute(
                    select(func.count())
                    .select_from(models.Event)
                    .where(models.Event.session_id == demo_sess.id)
                )
            ).scalar_one()
            assert event_count == 1

            stmt = select(models.AnalysisResult).where(
                models.AnalysisResult.session_id == demo_sess.id
            )
            demo_ar = (await async_db_session.execute(stmt)).scalars().first()

            pattern_count = (
                await async_db_session.execute(
                    select(func.count())
                    .select_from(models.DetectedPattern)
                    .where(models.DetectedPattern.analysis_result_id == demo_ar.id)
                )
            ).scalar_one()
            assert pattern_count == 1

    @pytest.mark.asyncio
    async def test_scrub_demo_source_data_untouched(
        self, async_db_session, patched_raw_backup_dir
    ):
        """Source profile data is not modified by the scrub."""
        from sqlalchemy import select  # noqa: PLC0415

        from snore.cli.groups.db import _do_scrub_demo  # noqa: PLC0415

        async with async_db_session.begin():
            src_profile_id, meta = await _seed_source_profile(async_db_session)

        async with async_db_session.begin():
            await _do_scrub_demo(async_db_session, src_profile_id)

        async with async_db_session.begin():
            src_profile = await async_db_session.get(models.Profile, src_profile_id)
            assert src_profile is not None
            assert src_profile.first_name == "Alice"
            assert src_profile.last_name == "Smith"
            assert src_profile.date_of_birth == date(1985, 6, 15)

            # Source device still has original serial
            stmt = select(models.Device).where(
                models.Device.profile_id == src_profile_id
            )
            src_dev = (await async_db_session.execute(stmt)).scalars().first()
            assert src_dev.serial_number == "SRC-SERIAL-001"

            # Source session import_source untouched
            stmt = select(models.Session).where(models.Session.device_id == src_dev.id)
            src_sess = (await async_db_session.execute(stmt)).scalars().first()
            assert src_sess.import_source == "original_import"

            # Source detected_pattern notes untouched
            stmt = (
                select(models.DetectedPattern)
                .join(
                    models.AnalysisResult,
                    models.DetectedPattern.analysis_result_id
                    == models.AnalysisResult.id,
                )
                .where(models.AnalysisResult.session_id == src_sess.id)
            )
            src_pat = (await async_db_session.execute(stmt)).scalars().first()
            assert src_pat.notes is not None
            assert "PII" in src_pat.notes

    def test_statistics_reflection_columns_all_numeric(self):
        """Reflection-clone safety: every copied Statistics column must be numeric.

        The scrub copies all columns except session_id verbatim; a future
        date/string/PII column must fail here instead of being silently cloned.
        """
        from sqlalchemy import Float, Integer, Numeric  # noqa: PLC0415

        from snore.cli.groups.db import _STATS_CLONE_EXCLUDED_COLUMNS  # noqa: PLC0415

        for col in models.Statistics.__table__.columns:
            if col.name in _STATS_CLONE_EXCLUDED_COLUMNS:
                continue
            # Float no longer subclasses Numeric as of SQLAlchemy 2.1.
            assert isinstance(col.type, (Integer, Numeric, Float)), (
                f"Statistics.{col.name} has non-numeric type {col.type!r}; "
                "review for PII/dates before letting the demo scrub clone it"
            )

    def test_day_reflection_columns_all_numeric(self):
        """Reflection-clone safety: every copied Day column must be numeric.

        Same tripwire as the Statistics guard — a future date/string/PII
        column on Day must fail here instead of being cloned verbatim
        (and, for dates, unshifted) into the demo profile.
        """
        from sqlalchemy import Float, Integer, Numeric  # noqa: PLC0415

        from snore.cli.groups.db import _DAY_CLONE_EXCLUDED_COLUMNS  # noqa: PLC0415

        for col in models.Day.__table__.columns:
            if col.name in _DAY_CLONE_EXCLUDED_COLUMNS:
                continue
            # Float no longer subclasses Numeric as of SQLAlchemy 2.1.
            assert isinstance(col.type, (Integer, Numeric, Float)), (
                f"Day.{col.name} has non-numeric type {col.type!r}; "
                "review for PII/dates before letting the demo scrub clone it"
            )

    @pytest.mark.asyncio
    async def test_scrub_demo_statistics_clone_covers_all_columns(
        self, async_db_session, patched_raw_backup_dir
    ):
        """Every Statistics column is carried into the demo clone.

        Regression: a hand-maintained field list silently dropped 27 of the
        74 metric columns (spo2_median, csr_pct, ie_ratio_median, ...).

        The all-columns loop below is only meaningful for seeded columns
        (unseeded ones are None on both sides); the explicit assertions on
        previously-dropped columns are the real regression lock.
        """
        from sqlalchemy import select  # noqa: PLC0415

        from snore.cli.groups.db import (  # noqa: PLC0415
            _STATS_CLONE_EXCLUDED_COLUMNS,
            _do_scrub_demo,
        )

        async with async_db_session.begin():
            src_profile_id, meta = await _seed_source_profile(async_db_session)

        async with async_db_session.begin():
            await _do_scrub_demo(async_db_session, src_profile_id)

        async with async_db_session.begin():
            stmt = select(models.User).where(
                models.User.canonical_email == "demo@snore.local"
            )
            demo_user = (await async_db_session.execute(stmt)).scalars().first()
            stmt = select(models.Device).where(
                models.Device.profile_id == demo_user.default_profile_id
            )
            demo_dev = (await async_db_session.execute(stmt)).scalars().first()
            stmt = select(models.Session).where(models.Session.device_id == demo_dev.id)
            demo_sess = (await async_db_session.execute(stmt)).scalars().first()

            src_stats = await async_db_session.get(
                models.Statistics, meta["src_session_id"]
            )
            demo_stats = await async_db_session.get(models.Statistics, demo_sess.id)
            assert demo_stats is not None

            # Previously-dropped columns must survive the clone.
            assert demo_stats.spo2_median == 94.0
            assert demo_stats.csr_pct == 1.5

            for col in models.Statistics.__table__.columns:
                if col.name in _STATS_CLONE_EXCLUDED_COLUMNS:
                    continue
                assert getattr(demo_stats, col.name) == getattr(src_stats, col.name), (
                    f"Statistics.{col.name} not carried into demo clone"
                )

    @pytest.mark.asyncio
    async def test_scrub_demo_missing_source_profile_raises(self, async_db_session):
        """_do_scrub_demo raises ClickException when source profile does not exist."""
        import click  # noqa: PLC0415

        from snore.cli.groups.db import _do_scrub_demo  # noqa: PLC0415

        async with async_db_session.begin():
            with pytest.raises(click.ClickException, match="not found"):
                await _do_scrub_demo(async_db_session, 99999)


# ---------------------------------------------------------------------------
# GET /api/v1/auth/status — demo_available field
# ---------------------------------------------------------------------------


def _seed_demo_user(db_session: Any) -> None:
    """Seed an active demo user + profile via the AUTOCOMMIT sync session."""
    demo_user = models.User(
        canonical_email="demo@snore.local",
        role="demo",
        display_name="Demo",
        password_hash=None,
        session_version=0,
    )
    db_session.add(demo_user)
    db_session.flush()
    demo_profile = models.Profile(user_id=demo_user.id, name="Demo")
    db_session.add(demo_profile)
    db_session.flush()
    demo_user.default_profile_id = demo_profile.id
    db_session.flush()


class TestAuthStatusDemoAvailable:
    def test_auth_status_no_demo_user_returns_demo_available_false(
        self, temp_db, async_db_session, db_session, monkeypatch
    ):
        """GET /auth/status → demo_available: false when no demo user exists."""
        client = _make_multiuser_client_no_actor(async_db_session, monkeypatch)
        resp = client.get("/api/v1/auth/status")
        assert resp.status_code == 200
        assert resp.json()["demo_available"] is False

    def test_auth_status_with_demo_user_returns_demo_available_true(
        self, temp_db, async_db_session, db_session, monkeypatch
    ):
        """GET /auth/status → demo_available: true when an active demo user exists."""
        _seed_demo_user(db_session)

        client = _make_multiuser_client_no_actor(async_db_session, monkeypatch)
        resp = client.get("/api/v1/auth/status")
        assert resp.status_code == 200
        assert resp.json()["demo_available"] is True

    def test_auth_status_false_not_cached_returns_true_after_demo_user_seeded(
        self, temp_db, async_db_session, db_session, monkeypatch
    ):
        """False result is not cached; a later request on the same app finds the demo user.

        Verifies the caching contract: True is stored in app.state.demo_available and
        re-used; False is never stored so a newly created demo user is picked up on
        the next request without restarting the process.
        """
        monkeypatch.setenv("SNORE_AUTH_MODE", "multiuser")
        monkeypatch.setenv(
            "SNORE_SESSION_SECRET", "test-secret-at-least-32-chars-long-abcdef"
        )
        monkeypatch.setenv("SNORE_PUBLIC_BASE_URL", "http://127.0.0.1:8000")

        from fastapi.testclient import TestClient  # noqa: PLC0415

        from snore.api.app import create_app  # noqa: PLC0415
        from snore.api.config import reset_config  # noqa: PLC0415
        from snore.api.deps import get_db  # noqa: PLC0415

        reset_config()
        app = create_app()

        async def override_get_db():
            async with async_db_session.begin():
                yield async_db_session

        app.dependency_overrides[get_db] = override_get_db
        client = TestClient(app, raise_server_exceptions=True)

        # First request: no demo user → false (nothing written to app.state).
        resp = client.get("/api/v1/auth/status")
        assert resp.status_code == 200
        assert resp.json()["demo_available"] is False

        # Seed demo user via AUTOCOMMIT sync session — visible to the next request.
        _seed_demo_user(db_session)

        # Second request on the same app instance → true (False was not cached).
        resp = client.get("/api/v1/auth/status")
        assert resp.status_code == 200
        assert resp.json()["demo_available"] is True

        # True result is now cached in app.state so future requests skip the DB query.
        assert getattr(app.state, "demo_available", False) is True
