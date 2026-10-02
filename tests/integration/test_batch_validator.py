"""Integration tests: batch validation report device indices.

``SessionValidation.device_*`` carry the device-reported (STR) indices from the
Statistics ``*_device`` columns and ``uai`` — never SNORE's mask-on recount in
``ahi``/``oai``/``cai``/``hi``.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Statistics
from snore.validation.batch import BatchValidator
from snore.validation.report import SessionValidation
from tests.integration.conftest import _make_day_session, _make_device


async def _validate_one_night(
    db: AsyncSession, profile_id: int, d: date, stats: dict[str, float | None] | None
) -> SessionValidation:
    """Seed one session (optionally with Statistics) and validate its night."""
    device = await _make_device(db, profile_id)
    _day, sess = await _make_day_session(db, device, d, duration_hours=8.0)
    if stats is not None:
        db.add(Statistics(session_id=sess.id, **stats))
        await db.flush()

    mock_ar = MagicMock()
    mock_ar.session_duration_hours = 8.0
    mock_ar.machine_events = []
    mock_ar.mode_results = {"aasm": MagicMock(apneas=[], hypopneas=[])}
    scores = MagicMock(sensitivity=0.9, precision=0.9, f1_score=0.9)
    mock_validation = {"apnea_validation": scores, "hypopnea_validation": scores}

    with (
        patch(
            "snore.services.analysis_facade.AnalysisFacade.get_analysis_result",
            new=AsyncMock(return_value=mock_ar),
        ),
        patch(
            "snore.analysis.modes.detector.EventDetector.validate_against_machine_events",
            return_value=mock_validation,
        ),
    ):
        report = await BatchValidator(db, profile_id).validate_date_range(
            d.isoformat(), d.isoformat()
        )

    assert len(report.sessions) == 1
    return report.sessions[0]


@pytest.mark.integration
class TestValidationReportDeviceIndices:
    async def test_device_indices_come_from_device_columns_not_recount(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        sv = await _validate_one_night(
            async_db_session,
            async_test_profile.id,
            date(2025, 7, 1),
            {
                "ahi": 9.9,
                "oai": 9.9,
                "cai": 9.9,
                "hi": 9.9,
                "ahi_device": 4.5,
                "oai_device": 1.2,
                "cai_device": 0.3,
                "hi_device": 3.0,
                "uai": 0.5,
            },
        )

        assert sv.device_ahi == pytest.approx(4.5)
        assert sv.device_oai == pytest.approx(1.2)
        assert sv.device_cai == pytest.approx(0.3)
        assert sv.device_hi == pytest.approx(3.0)
        assert sv.device_uai == pytest.approx(0.5)

    async def test_device_indices_null_when_device_did_not_report(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """A recount without device STR indices must not leak into device_*."""
        sv = await _validate_one_night(
            async_db_session,
            async_test_profile.id,
            date(2025, 7, 2),
            {"ahi": 3.1, "oai": 0.8, "cai": 0.1, "hi": 2.2, "ahi_device": 3.0},
        )

        assert sv.device_ahi == pytest.approx(3.0)
        assert sv.device_oai is None
        assert sv.device_cai is None
        assert sv.device_hi is None
        assert sv.device_uai is None

    async def test_device_indices_null_when_statistics_absent(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        sv = await _validate_one_night(
            async_db_session, async_test_profile.id, date(2025, 7, 3), None
        )

        assert sv.device_ahi is None
        assert sv.device_oai is None
        assert sv.device_cai is None
        assert sv.device_hi is None
        assert sv.device_uai is None
