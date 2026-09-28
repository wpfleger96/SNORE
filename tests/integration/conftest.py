"""Shared fixtures and seed helpers for integration tests.

Seed helpers (_make_profile, _make_device, _make_day_session, _make_analysis_result,
_make_event) are plain functions — not fixtures — so test modules can import and call them directly:

    from tests.integration.conftest import _make_profile, _make_device, ...
"""

from __future__ import annotations

import uuid

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import (
    AnalysisResult,
    Day,
    Device,
    Event,
    Profile,
    Session,
    User,
)
from snore.database.session import cleanup_database


@pytest.fixture(autouse=True)
async def reset_database_state():
    """Reset global database state before and after each test.

    Runs cleanup in the same event loop as the test so engine disposal is
    loop-correct; pytest-asyncio's auto mode serves sync tests transparently.
    """
    await cleanup_database()
    yield
    await cleanup_database()


# ---------------------------------------------------------------------------
# Shared seed helpers — import from here; do not copy across test modules.
# ---------------------------------------------------------------------------


async def _make_profile(db: AsyncSession) -> Any:
    """Create a User + Profile and return the Profile."""
    user = User(
        canonical_email=f"test_{uuid.uuid4().hex[:8]}@example.com",
        role="member",
    )
    db.add(user)
    await db.flush()
    profile = Profile(user_id=user.id, name="Test Profile")
    db.add(profile)
    await db.flush()
    return profile


async def _make_device(
    db: AsyncSession,
    profile_id: int,
    manufacturer: str = "TestMfr",
    model: str = "TestModel",
    serial_number: str | None = None,
) -> Device:
    device = Device(
        profile_id=profile_id,
        manufacturer=manufacturer,
        model=model,
        serial_number=serial_number or f"SN_{uuid.uuid4().hex[:8]}",
    )
    db.add(device)
    await db.flush()
    return device


async def _make_day_session(
    db: AsyncSession,
    device: Device,
    day_date: date,
    duration_hours: float = 8.0,
    start_hour: int = 22,
    **day_kwargs: Any,
) -> tuple[Day, Session]:
    """Create a linked Day + enabled Session pair."""
    day = Day(
        device_id=device.id,
        date=day_date,
        total_therapy_hours=duration_hours,
        **day_kwargs,
    )
    db.add(day)
    await db.flush()

    start_dt = datetime(day_date.year, day_date.month, day_date.day, start_hour, 0, 0)
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"test_{day_date.isoformat()}_{uuid.uuid4().hex[:6]}",
        start_time=start_dt,
        end_time=start_dt + timedelta(hours=duration_hours),
        duration_seconds=duration_hours * 3600,
        enabled=True,
    )
    db.add(sess)
    await db.flush()
    return day, sess


async def _make_analysis_result(
    db: AsyncSession,
    session: Session,
    primary_mode: str = "aasm",
) -> AnalysisResult:
    """Create an AnalysisResult with the current algorithm identity (status=OK)."""
    from snore.analysis.shared.versioning import (  # noqa: PLC0415
        AlgorithmIdentity,
        AlgoVersions,
        AnalysisRunMetadata,
    )

    algo_versions = AlgoVersions(
        identity=AlgorithmIdentity.current(),
        run=AnalysisRunMetadata(primary_mode=primary_mode, modes=[primary_mode]),
    )
    ar = AnalysisResult(
        session_id=session.id,
        timestamp_start=session.start_time,
        timestamp_end=session.end_time,
        engine_versions_json=algo_versions.model_dump(),
    )
    db.add(ar)
    await db.flush()
    return ar


async def _make_event(
    db: AsyncSession,
    session: Session,
    *,
    event_type: str = "CA",
    offset_seconds: float = 120.0,
    duration_seconds: float = 15.0,
) -> Event:
    """Create a machine Event at ``offset_seconds`` from session start."""
    event = Event(
        session_id=session.id,
        event_type=event_type,
        start_time=session.start_time + timedelta(seconds=offset_seconds),
        duration_seconds=duration_seconds,
    )
    db.add(event)
    await db.flush()
    return event
