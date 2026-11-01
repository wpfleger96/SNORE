import uuid

from datetime import date, datetime, timedelta
from typing import Any, NoReturn

import pytest

from sqlalchemy.orm import Session as OrmSession

from snore.analysis.shared.versioning import (
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisRunMetadata,
)
from snore.database.models import (
    AnalysisResult,
    Breath,
    Day,
    Device,
    Profile,
    Session,
    User,
)
from snore.provenance import IndexSource
from snore.services.breath_service import BreathService
from snore.services.day_service import DayService

_EXPERIMENTAL_FIELDS = {
    "fl_class_ge4_pct",
    "fl_class_ge4_pct_reason",
    "rera_index",
    "rera_index_reason",
    "rera_count",
    "rera_count_reason",
}


def _seed_night(
    db_session: OrmSession, device: Device, night: date, *, enabled: bool = True
) -> Session:
    """A two-hour night: one Day row and one linked Session."""
    day = Day(
        device_id=device.id,
        date=night,
        session_count=1,
        total_therapy_hours=2.0 if enabled else 0.0,
    )
    db_session.add(day)
    db_session.flush()
    start = datetime.combine(night, datetime.min.time()).replace(hour=22)
    session = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"sess_{uuid.uuid4().hex[:8]}",
        start_time=start,
        end_time=start + timedelta(hours=2),
        duration_seconds=7200,
        enabled=enabled,
    )
    db_session.add(session)
    db_session.flush()
    return session


def _seed_rera_proxies(db_session: OrmSession, session: Session, count: int) -> None:
    """An OK analysis with ``count`` proxies: two FL breaths, then a recovery."""
    analysis = AnalysisResult(
        session_id=session.id,
        timestamp_start=session.start_time,
        timestamp_end=session.end_time,
        engine_versions_json=AlgoVersions(
            identity=AlgorithmIdentity.current(),
            run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
        ).model_dump(),
    )
    db_session.add(analysis)
    db_session.flush()
    for n in range(count * 3):
        is_recovery = n % 3 == 2
        db_session.add(
            Breath(
                analysis_result_id=analysis.id,
                session_id=session.id,
                breath_number=n,
                start_offset_s=n * 5.0,
                end_offset_s=n * 5.0 + 4.0,
                leak_valid=True,
                flow_class=1 if is_recovery else 5,
                flow_confidence=0.9,
                is_recovery_breath=is_recovery,
            )
        )
    db_session.flush()


def _seed_foreign_device(db_session: OrmSession) -> Device:
    """A device on a second user's profile, not visible to ``api_client``."""
    user = User(
        canonical_email=f"foreign_{uuid.uuid4().hex[:8]}@test.com", role="member"
    )
    db_session.add(user)
    db_session.flush()
    profile = Profile(user_id=user.id, name="Foreign")
    db_session.add(profile)
    db_session.flush()
    device = Device(
        profile_id=profile.id,
        manufacturer="Mfr",
        model="M",
        serial_number=f"SN_{uuid.uuid4().hex[:8]}",
    )
    db_session.add(device)
    db_session.flush()
    return device


class TestDaysRouter:
    def test_list_days_empty(self, api_client):
        response = api_client.get("/api/v1/days/")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_days_with_data(self, api_client, db_session, test_device):
        from snore.database.models import Day

        day = Day(
            device_id=test_device.id,
            date=date(2025, 1, 10),
            session_count=1,
            total_therapy_hours=8.0,
            ahi=2.5,
        )
        db_session.add(day)
        db_session.flush()

        response = api_client.get("/api/v1/days/")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["date"] == "2025-01-10"
        assert data["items"][0]["session_count"] == 1

    def test_list_days_filter_by_device(self, api_client, db_session, test_device):
        from snore.database.models import Day

        day = Day(device_id=test_device.id, date=date(2025, 1, 10), session_count=1)
        db_session.add(day)
        db_session.flush()

        response = api_client.get(f"/api/v1/days/?device_id={test_device.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

        response = api_client.get("/api/v1/days/?device_id=99999")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_get_day_not_found(self, api_client):
        response = api_client.get("/api/v1/days/2025-01-01")
        assert response.status_code == 404

    def test_get_day_with_data(self, api_client, db_session, test_device):
        from snore.database.models import Day

        day = Day(
            device_id=test_device.id,
            date=date(2025, 1, 10),
            session_count=2,
            total_therapy_hours=7.5,
            ahi=1.8,
        )
        db_session.add(day)
        db_session.flush()

        response = api_client.get("/api/v1/days/2025-01-10")
        assert response.status_code == 200
        data = response.json()
        assert data["date"] == "2025-01-10"
        assert data["session_count"] == 2
        assert data["ahi"] == 1.8
        assert data["session_ids"] == []
        assert "pressure_min" in data
        assert "pressure_max" in data
        assert "pressure_median" in data
        assert "pressure_95th" in data
        assert "epap_min" in data
        assert "epap_max" in data
        assert "epap_median" in data
        assert "epap_mean" in data
        assert "epap_95th" in data
        assert "leak_min" in data
        assert "leak_max" in data
        assert "leak_mean" in data
        assert "leak_95th" in data
        assert "spo2_min" in data
        assert "spo2_max" in data
        assert data["obstructive_apneas"] == 0
        assert data["central_apneas"] == 0
        assert data["hypopneas"] == 0
        assert data["reras"] == 0

    def test_days_endpoints_return_index_source_and_recount(
        self, api_client, db_session, test_device
    ):
        from snore.database.models import Day

        db_session.add(
            Day(
                device_id=test_device.id,
                date=date(2025, 1, 11),
                session_count=1,
                total_therapy_hours=8.0,
                ahi=1.2,
                hi=0.5,
                index_source=IndexSource.DEVICE,
                ahi_computed=1.6,
                hi_computed=0.7,
            )
        )
        db_session.flush()

        item = api_client.get("/api/v1/days/").json()["items"][0]
        detail = api_client.get("/api/v1/days/2025-01-11").json()

        assert (item["ahi"], item["index_source"], item["ahi_computed"]) == (
            1.2,
            "device",
            1.6,
        )
        assert detail["index_source"] == "device"
        assert (detail["ahi"], detail["ahi_computed"]) == (1.2, 1.6)
        assert (detail["hi"], detail["hi_computed"]) == (0.5, 0.7)

    def test_get_day_omits_experimental_fields(
        self, api_client, db_session, test_device
    ):
        _seed_night(db_session, test_device, date(2025, 1, 12))

        data = api_client.get("/api/v1/days/2025-01-12").json()

        assert not _EXPERIMENTAL_FIELDS & data.keys()

    def test_get_day_never_reads_breath_analysis(
        self, api_client, db_session, test_device, monkeypatch
    ):
        session = _seed_night(db_session, test_device, date(2025, 1, 13))
        _seed_rera_proxies(db_session, session, count=2)

        def _fail(*args: Any, **kwargs: Any) -> NoReturn:
            raise AssertionError("GET /days/{date} must not read breath analysis")

        monkeypatch.setattr(BreathService, "get_experimental_nightly_summary", _fail)
        monkeypatch.setattr(DayService, "_nightly_fl_rera", _fail)

        response = api_client.get("/api/v1/days/2025-01-13")

        assert response.status_code == 200
        assert response.json()["session_ids"] == [session.id]


class TestDayExperimental:
    def test_analyzed_night_returns_metrics(self, api_client, db_session, test_device):
        session = _seed_night(db_session, test_device, date(2025, 2, 1))
        _seed_rera_proxies(db_session, session, count=4)

        response = api_client.get("/api/v1/days/2025-02-01/experimental")

        assert response.status_code == 200
        assert response.json() == {
            "date": "2025-02-01",
            "fl_class_ge4_pct": pytest.approx(100.0 * 2 / 3),
            "fl_class_ge4_pct_reason": None,
            "rera_index": pytest.approx(2.0),
            "rera_index_reason": None,
            "rera_count": 4,
            "rera_count_reason": None,
        }

    def test_unanalyzed_night_reports_not_available(
        self, api_client, db_session, test_device
    ):
        _seed_night(db_session, test_device, date(2025, 2, 2))

        data = api_client.get("/api/v1/days/2025-02-02/experimental").json()

        assert data == {
            "date": "2025-02-02",
            "fl_class_ge4_pct": None,
            "fl_class_ge4_pct_reason": "not_available",
            "rera_index": None,
            "rera_index_reason": "not_available",
            "rera_count": None,
            "rera_count_reason": "not_available",
        }

    def test_all_sessions_disabled_reports_analysis_not_run(
        self, api_client, db_session, test_device
    ):
        session = _seed_night(db_session, test_device, date(2025, 2, 3), enabled=False)
        _seed_rera_proxies(db_session, session, count=2)

        data = api_client.get("/api/v1/days/2025-02-03/experimental").json()

        assert data == {
            "date": "2025-02-03",
            "fl_class_ge4_pct": None,
            "fl_class_ge4_pct_reason": "analysis_not_run",
            "rera_index": None,
            "rera_index_reason": "analysis_not_run",
            "rera_count": None,
            "rera_count_reason": "analysis_not_run",
        }

    def test_unknown_date_returns_404(self, api_client):
        response = api_client.get("/api/v1/days/2025-02-04/experimental")

        assert response.status_code == 404

    def test_device_id_selects_that_devices_night(
        self, api_client, db_session, test_device, test_profile
    ):
        other_device = Device(
            profile_id=test_profile.id,
            manufacturer="Mfr",
            model="Other",
            serial_number=f"SN_{uuid.uuid4().hex[:8]}",
        )
        db_session.add(other_device)
        db_session.flush()
        _seed_night(db_session, test_device, date(2025, 2, 5))
        analyzed = _seed_night(db_session, other_device, date(2025, 2, 5))
        _seed_rera_proxies(db_session, analyzed, count=2)

        response = api_client.get(
            f"/api/v1/days/2025-02-05/experimental?device_id={other_device.id}"
        )

        assert response.json()["rera_count"] == 2

    def test_foreign_profile_night_returns_404(self, api_client, db_session):
        device = _seed_foreign_device(db_session)
        session = _seed_night(db_session, device, date(2025, 2, 6))
        _seed_rera_proxies(db_session, session, count=2)

        by_date = api_client.get("/api/v1/days/2025-02-06/experimental")
        by_device = api_client.get(
            f"/api/v1/days/2025-02-06/experimental?device_id={device.id}"
        )

        assert by_date.status_code == 404
        assert by_device.status_code == 404
