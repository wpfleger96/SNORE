import uuid

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.orm import Session as OrmSession

from snore.database.models import (
    Day,
    Device,
    Event,
    HealthNightlySummary,
    Profile,
    Session,
    User,
)

_TREND_KEYS = {
    "ahi",
    "usage",
    "spo2",
    "leak",
    "pressure",
    "oai",
    "cai",
    "hi",
    "rera",
    "epap",
    "rr",
    "pulse",
    "mv",
    "tv",
    "uai",
    "time_in_apnea",
    "bedtime",
    "wake_time",
}


class TestStatsSummary:
    def test_summary_empty(self, api_client):
        response = api_client.get("/api/v1/stats/summary")
        assert response.status_code == 204

    def test_summary_with_data(
        self, api_client, db_session, test_device, test_session_factory
    ):
        _session = test_session_factory(
            test_device.id,
            start_time=datetime(2024, 1, 1, 22, 0),
            ahi=3.5,
            usage_hours=7.0,
        )
        day = Day(
            device_id=test_device.id,
            date=date(2024, 1, 1),
            session_count=1,
            total_therapy_hours=7.0,
        )
        db_session.add(day)
        db_session.flush()

        response = api_client.get("/api/v1/stats/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["days_with_data"] == 1
        assert data["total_hours"] == pytest.approx(7.0)
        assert not {"effectiveness", "ahi_trend_direction"} & data.keys()

    def test_summary_and_periods_report_compliance(
        self, api_client, db_session, test_device
    ):
        # Jan 1-4: 4 h exactly, 3.5 h, no data, 6 h.
        for day_of_month, hours in ((1, 4.0), (2, 3.5), (4, 6.0)):
            db_session.add(
                Day(
                    device_id=test_device.id,
                    date=date(2024, 1, day_of_month),
                    session_count=1,
                    total_therapy_hours=hours,
                )
            )
        db_session.flush()

        summary = api_client.get("/api/v1/stats/summary").json()
        [period] = api_client.get(
            "/api/v1/stats/periods", params={"period_type": "month"}
        ).json()

        assert (summary["days_compliant"], summary["compliance_pct"]) == (2, 50.0)
        assert (period["days_compliant"], period["compliance_pct"]) == (2, 50.0)

    def test_disabled_only_edge_day_is_outside_both_compliance_ranges(
        self, api_client, db_session, test_device
    ):
        # 8 h on Mar 1 and 2; Mar 10's sessions are all disabled (0 hours).
        for day_of_month, hours in ((1, 8.0), (2, 8.0), (10, 0.0)):
            db_session.add(
                Day(
                    device_id=test_device.id,
                    date=date(2024, 3, day_of_month),
                    session_count=1,
                    total_therapy_hours=hours,
                )
            )
        db_session.flush()

        summary = api_client.get("/api/v1/stats/summary").json()
        [period] = api_client.get(
            "/api/v1/stats/periods", params={"period_type": "month"}
        ).json()

        assert summary["compliance_pct"] == period["compliance_pct"] == 100.0


class TestStatsExperimental:
    def test_range_with_data(self, api_client, db_session, test_device):
        """Weeks at AHI 8, 8, 2 average 6 ("good"); the last week is improving."""
        for week, ahi in enumerate([8.0, 8.0, 2.0]):
            db_session.add(
                Day(
                    device_id=test_device.id,
                    date=date(2024, 1, 1) + timedelta(weeks=week),
                    session_count=1,
                    total_therapy_hours=7.0,
                    ahi=ahi,
                )
            )
        db_session.flush()

        response = api_client.get("/api/v1/stats/experimental")

        assert response.status_code == 200
        assert response.json() == {
            "effectiveness": "good",
            "ahi_trend_direction": "improving",
        }

    def test_empty_range(self, api_client):
        response = api_client.get("/api/v1/stats/experimental")

        assert response.status_code == 200
        assert response.json() == {
            "effectiveness": "unknown",
            "ahi_trend_direction": None,
        }

    def test_days_limit_is_not_a_parameter(self, api_client):
        """Always all-time, so even a limit no date arithmetic can hold is ignored."""
        response = api_client.get("/api/v1/stats/experimental?days_limit=1000000")

        assert response.status_code == 200


@pytest.mark.parametrize("route", ["summary", "periods", "records", "trends"])
class TestDaysLimitBounds:
    @pytest.mark.parametrize("days_limit", [-1, 36501, 1000000])
    def test_out_of_range_limit_returns_422(self, api_client, route, days_limit):
        response = api_client.get(f"/api/v1/stats/{route}?days_limit={days_limit}")

        assert response.status_code == 422

    def test_largest_limit_is_accepted(self, api_client, route):
        response = api_client.get(f"/api/v1/stats/{route}?days_limit=36500")

        assert response.is_success

    def test_zero_limit_means_all_time(
        self, api_client, db_session, test_device, route
    ):
        db_session.add(
            Day(
                device_id=test_device.id,
                date=date.today() - timedelta(days=4000),
                total_therapy_hours=7.0,
                ahi=2.0,
            )
        )
        db_session.flush()

        all_time = api_client.get(f"/api/v1/stats/{route}").json()
        response = api_client.get(f"/api/v1/stats/{route}?days_limit=0")

        assert response.status_code == 200
        assert response.json() == all_time
        assert all_time


class TestStatsPeriods:
    def test_periods_empty(self, api_client):
        response = api_client.get("/api/v1/stats/periods")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_periods_with_data(self, api_client, db_session, test_device):
        day = Day(
            device_id=test_device.id,
            date=date(2024, 1, 15),
            session_count=1,
            total_therapy_hours=7.0,
        )
        db_session.add(day)
        db_session.flush()

        response = api_client.get("/api/v1/stats/periods?period_type=month")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        period = data[0]
        assert "period_type" in period
        assert "period_start" in period
        assert "period_end" in period

    def test_periods_day_type_with_data(self, api_client, db_session, test_device):
        """day granularity on /periods returns one entry per therapy date."""
        for i in range(3):
            db_session.add(
                Day(
                    device_id=test_device.id,
                    date=date(2024, 5, 1) + timedelta(days=i),
                    session_count=1,
                    total_therapy_hours=7.0,
                )
            )
        db_session.flush()

        response = api_client.get("/api/v1/stats/periods?period_type=day")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        # Each period covers exactly one day (period_start == period_end)
        for period in data:
            assert period["period_start"] == period["period_end"]


class TestStatsTrends:
    def test_trends_empty(self, api_client):
        response = api_client.get("/api/v1/stats/trends")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_trends_payload_has_every_series_key(
        self, api_client, db_session, test_device
    ):
        """/trends always returns every device metric key."""
        db_session.add(
            Day(
                device_id=test_device.id,
                date=date(2024, 3, 15),
                session_count=1,
                total_therapy_hours=7.0,
                ahi=2.5,
            )
        )
        db_session.flush()

        response = api_client.get("/api/v1/stats/trends?period_type=month")
        assert response.status_code == 200
        data = response.json()

        expected_keys = {
            "ahi",
            "usage",
            "spo2",
            "leak",
            "pressure",
            "oai",
            "cai",
            "hi",
            "rera",
            "epap",
            "rr",
            "pulse",
            "mv",
            "tv",
            "uai",
            "time_in_apnea",
            "bedtime",
            "wake_time",
        }
        assert set(data.keys()) == expected_keys

    def test_trends_day_default_limit_excludes_old_days(
        self, api_client, db_session, test_device
    ):
        """When period_type=day and no days_limit, days older than 180 are excluded."""
        old_date = date.today() - timedelta(days=200)
        recent_date = date.today() - timedelta(days=5)

        db_session.add(
            Day(
                device_id=test_device.id,
                date=old_date,
                session_count=1,
                total_therapy_hours=7.0,
                ahi=3.0,
            )
        )
        db_session.add(
            Day(
                device_id=test_device.id,
                date=recent_date,
                session_count=1,
                total_therapy_hours=7.0,
                ahi=2.0,
            )
        )
        db_session.flush()

        # No explicit days_limit → default 180 for day granularity
        response = api_client.get("/api/v1/stats/trends?period_type=day")
        assert response.status_code == 200
        data = response.json()
        ahi_dates = [entry[0] for entry in data["ahi"]]
        assert str(old_date) not in ahi_dates
        assert str(recent_date) in ahi_dates

    def test_trends_day_explicit_limit_includes_old_days(
        self, api_client, db_session, test_device
    ):
        """An explicit large days_limit overrides the day-granularity default."""
        old_date = date.today() - timedelta(days=200)

        db_session.add(
            Day(
                device_id=test_device.id,
                date=old_date,
                session_count=1,
                total_therapy_hours=7.0,
                ahi=3.0,
            )
        )
        db_session.flush()

        response = api_client.get("/api/v1/stats/trends?period_type=day&days_limit=365")
        assert response.status_code == 200
        data = response.json()
        ahi_dates = [entry[0] for entry in data["ahi"]]
        assert str(old_date) in ahi_dates

    def test_trends_wire_shape_without_health_data(
        self, api_client, db_session, test_device
    ):
        """Apple Health series keys are omitted, not null, when no night has data."""
        db_session.add(
            Day(
                device_id=test_device.id,
                date=date(2024, 3, 15),
                session_count=1,
                total_therapy_hours=7.0,
                ahi=2.5,
            )
        )
        db_session.flush()

        data = api_client.get("/api/v1/stats/trends?period_type=month").json()

        assert set(data) == _TREND_KEYS
        assert data["ahi"] == [["2024-03-01", 2.5]]
        assert data["spo2"] == [["2024-03-01", None]]

    def test_trends_wire_shape_with_health_data(
        self, api_client, db_session, test_device
    ):
        """Apple Health series appear alongside the 15 device keys when present."""
        db_session.add(
            Day(
                device_id=test_device.id,
                date=date(2024, 3, 15),
                session_count=1,
                total_therapy_hours=7.0,
                ahi=2.5,
            )
        )
        db_session.add(
            HealthNightlySummary(
                profile_id=test_device.profile_id,
                night_date=date(2024, 3, 15),
                total_sleep_seconds=7.5 * 3600,
                sleep_efficiency_pct=90.0,
                computed_at=datetime.now(UTC),
            )
        )
        db_session.flush()

        data = api_client.get("/api/v1/stats/trends?period_type=month").json()

        assert set(data) == _TREND_KEYS | {"total_sleep_hours", "sleep_efficiency"}
        assert data["ahi"] == [["2024-03-01", 2.5]]
        assert data["total_sleep_hours"] == [["2024-03-01", 7.5]]
        assert data["sleep_efficiency"] == [["2024-03-01", 90.0]]


class TestStatsSessionValues:
    """TV, time in apnea, session times and UAI on /periods and /trends."""

    @pytest.fixture
    def _night(self, db_session, test_device, test_session_factory):
        """One ResMed night 23:30 → 07:00 with 30 s of apnea and two UAs."""
        day = Day(
            device_id=test_device.id,
            date=date(2024, 3, 15),
            session_count=1,
            total_therapy_hours=7.5,
            ahi=1.0,
            oai=0.4,
            cai=0.1,
            hi=0.2,
            unclassified_apneas=2,
        )
        db_session.add(day)
        db_session.flush()
        session = test_session_factory(
            test_device.id,
            start_time=datetime(2024, 3, 15, 23, 30),
            duration_hours=7.5,
            usage_hours=7.5,
            tidal_volume_mean=0.45,
        )
        session.day_id = day.id
        start = datetime(2024, 3, 16, 2, 0)
        db_session.add_all(
            [
                Event(
                    session_id=session.id,
                    event_type="OA",
                    start_time=start,
                    duration_seconds=12.0,
                ),
                Event(
                    session_id=session.id,
                    event_type="UA",
                    start_time=start,
                    duration_seconds=18.0,
                ),
                Event(session_id=session.id, event_type="UA", start_time=start),
                Event(session_id=session.id, event_type="H", start_time=start),
            ]
        )
        db_session.flush()

    @pytest.mark.usefixtures("_night")
    def test_periods_carry_session_values(self, api_client):
        (period,) = api_client.get("/api/v1/stats/periods?period_type=day").json()

        assert period["avg_tidal_volume_ml"] == pytest.approx(450.0)
        assert period["avg_time_in_apnea_minutes"] == pytest.approx(0.5)
        assert period["avg_bedtime_hours"] == pytest.approx(11.5)
        assert period["avg_wake_time_hours"] == pytest.approx(19.0)
        assert period["avg_uai"] == pytest.approx(2 / 7.5)

    @pytest.mark.usefixtures("_night")
    def test_trends_carry_session_values(self, api_client):
        data = api_client.get("/api/v1/stats/trends?period_type=month").json()

        assert data["tv"] == [["2024-03-01", pytest.approx(450.0)]]
        assert data["time_in_apnea"] == [["2024-03-01", pytest.approx(0.5)]]
        assert data["bedtime"] == [["2024-03-01", pytest.approx(11.5)]]
        assert data["wake_time"] == [["2024-03-01", pytest.approx(19.0)]]
        assert data["uai"] == [["2024-03-01", pytest.approx(2 / 7.5)]]

    def test_multi_device_night_counts_as_one_night(
        self, api_client, db_session, test_device, test_session_factory
    ):
        other = Device(
            profile_id=test_device.profile_id,
            manufacturer="Other",
            model="No UA",
            serial_number=f"SN_{uuid.uuid4().hex[:8]}",
        )
        db_session.add(other)
        db_session.flush()
        # Device A 22:00 → 02:00 scores UAs; device B 02:30 → 07:00 never does.
        first = _night_session(
            db_session,
            test_session_factory,
            _day(db_session, test_device, 4.0, unclassified_apneas=2),
            datetime(2024, 3, 15, 22, 0),
            4.0,
            tidal_volume_mean=0.4,
        )
        second = _night_session(
            db_session,
            test_session_factory,
            _day(db_session, other, 4.5),
            datetime(2024, 3, 16, 2, 30),
            4.5,
            tidal_volume_mean=0.5,
        )
        _add_event(db_session, first, "UA", 20.0)
        _add_event(db_session, second, "CA", 40.0)

        (period,) = api_client.get("/api/v1/stats/periods?period_type=day").json()

        assert period["avg_bedtime_hours"] == pytest.approx(10.0)
        assert period["avg_wake_time_hours"] == pytest.approx(19.0)
        assert period["avg_time_in_apnea_minutes"] == pytest.approx(1.0)
        assert period["avg_tidal_volume_ml"] == pytest.approx(
            round((400 * 4 + 500 * 4.5) / 8.5, 1)
        )
        # Device B's hours count toward UAI with no UA events, as they do for OAI.
        assert period["avg_uai"] == pytest.approx(2 / 8.5)

    def test_disabled_session_excluded_from_session_values(
        self, api_client, db_session, test_device, test_session_factory
    ):
        day = _day(db_session, test_device, 6.0)
        enabled = _night_session(
            db_session,
            test_session_factory,
            day,
            datetime(2024, 3, 15, 23, 0),
            6.0,
            tidal_volume_mean=0.4,
        )
        disabled = _night_session(
            db_session,
            test_session_factory,
            day,
            datetime(2024, 3, 16, 5, 30),
            3.5,
            enabled=False,
            tidal_volume_mean=0.6,
        )
        _add_event(db_session, enabled, "OA", 30.0)
        _add_event(db_session, disabled, "CA", 60.0)

        (period,) = api_client.get("/api/v1/stats/periods?period_type=day").json()

        assert period["avg_bedtime_hours"] == pytest.approx(11.0)
        assert period["avg_wake_time_hours"] == pytest.approx(17.0)
        assert period["avg_time_in_apnea_minutes"] == pytest.approx(0.5)
        assert period["avg_tidal_volume_ml"] == pytest.approx(400.0)

    def test_session_without_statistics_keeps_times_and_apnea(
        self, api_client, db_session, test_device, test_session_factory
    ):
        day = _day(db_session, test_device, 8.0)
        _night_session(
            db_session,
            test_session_factory,
            day,
            datetime(2024, 3, 15, 22, 0),
            4.0,
            tidal_volume_mean=0.5,
        )
        bare = _night_session(
            db_session, test_session_factory, day, datetime(2024, 3, 16, 3, 0), 4.0
        )
        _add_event(db_session, bare, "CA", 30.0)

        (period,) = api_client.get("/api/v1/stats/periods?period_type=day").json()

        assert period["avg_bedtime_hours"] == pytest.approx(10.0)
        assert period["avg_wake_time_hours"] == pytest.approx(19.0)
        assert period["avg_time_in_apnea_minutes"] == pytest.approx(0.5)
        assert period["avg_tidal_volume_ml"] == pytest.approx(500.0)

    def test_session_ending_after_noon_wakes_past_24_hours(
        self, api_client, db_session, test_device, test_session_factory
    ):
        # 05:00 belongs to the March 15 night; ending at 13:00 is 25 h after its noon.
        _night_session(
            db_session,
            test_session_factory,
            _day(db_session, test_device, 8.0),
            datetime(2024, 3, 16, 5, 0),
            8.0,
        )

        (period,) = api_client.get("/api/v1/stats/periods?period_type=day").json()
        data = api_client.get("/api/v1/stats/trends?period_type=month").json()

        assert period["avg_bedtime_hours"] == pytest.approx(17.0)
        assert period["avg_wake_time_hours"] == pytest.approx(25.0)
        assert data["wake_time"] == [["2024-03-01", pytest.approx(25.0)]]

    def test_device_never_scoring_ua_leaves_uai_null(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = _night_session(
            db_session,
            test_session_factory,
            _day(db_session, test_device, 8.0, oai=0.5),
            datetime(2024, 3, 15, 22, 0),
            8.0,
        )
        _add_event(db_session, session, "OA", 10.0)

        (period,) = api_client.get("/api/v1/stats/periods?period_type=month").json()
        data = api_client.get("/api/v1/stats/trends?period_type=month").json()

        assert period["avg_uai"] is None
        assert period["avg_oai"] == pytest.approx(0.5)
        assert data["uai"] == [["2024-03-01", None]]
        assert data["oai"] == [["2024-03-01", pytest.approx(0.5)]]


def _day(
    db_session: OrmSession, device: Device, hours: float, **day_kwargs: Any
) -> Day:
    """The March 15, 2024 night of ``device`` with ``hours`` of therapy."""
    day = Day(
        device_id=device.id,
        date=date(2024, 3, 15),
        total_therapy_hours=hours,
        **day_kwargs,
    )
    db_session.add(day)
    db_session.flush()
    return day


def _night_session(
    db_session: OrmSession,
    session_factory: Callable[..., Session],
    day: Day,
    start: datetime,
    hours: float,
    *,
    enabled: bool = True,
    **stats_kwargs: float,
) -> Session:
    """A session of ``day``; ``stats_kwargs`` add a Statistics row with usage hours."""
    if stats_kwargs:
        stats_kwargs["usage_hours"] = hours
    session = session_factory(
        day.device_id, start, duration_hours=hours, **stats_kwargs
    )
    session.day_id = day.id
    session.enabled = enabled
    db_session.flush()
    return session


def _add_event(
    db_session: OrmSession, session: Session, event_type: str, seconds: float
) -> None:
    db_session.add(
        Event(
            session_id=session.id,
            event_type=event_type,
            start_time=session.start_time,
            duration_seconds=seconds,
        )
    )
    db_session.flush()


class TestStatsRecords:
    def test_records_empty(self, api_client):
        response = api_client.get("/api/v1/stats/records")
        assert response.status_code == 200
        assert response.json() == {}

    def test_records_wire_shape(self, api_client, db_session, test_device):
        """Only metrics with a qualifying day appear; pairs serialize as [date, value]."""
        for i, ahi in enumerate([1.0, 4.5]):
            db_session.add(
                Day(
                    device_id=test_device.id,
                    date=date(2024, 3, 1) + timedelta(days=i),
                    session_count=1,
                    total_therapy_hours=6.0 + i,
                    ahi=ahi,
                )
            )
        db_session.flush()

        data = api_client.get("/api/v1/stats/records").json()

        assert data == {
            "ahi": {
                "best": [["2024-03-01", 1.0], ["2024-03-02", 4.5]],
                "worst": [["2024-03-02", 4.5], ["2024-03-01", 1.0]],
            },
            "therapy_hours": {
                "best": [["2024-03-02", 7.0], ["2024-03-01", 6.0]],
                "worst": [["2024-03-01", 6.0], ["2024-03-02", 7.0]],
            },
        }


class TestStatsDataRange:
    def test_data_range_empty(self, api_client):
        """/data-range returns 200 with both dates null when no data exists."""
        response = api_client.get("/api/v1/stats/data-range")
        assert response.status_code == 200
        assert response.json() == {"earliest_date": None, "latest_date": None}

    def test_data_range_returns_both_bounds(self, api_client, db_session, test_device):
        """/data-range returns earliest and latest Day.date across all profile data."""
        older = date.today() - timedelta(days=30)
        newer = date.today() - timedelta(days=5)
        db_session.add(
            Day(device_id=test_device.id, date=older, total_therapy_hours=7.0)
        )
        db_session.add(
            Day(device_id=test_device.id, date=newer, total_therapy_hours=7.0)
        )
        db_session.flush()

        response = api_client.get("/api/v1/stats/data-range")
        assert response.status_code == 200
        data = response.json()
        assert data["earliest_date"] == str(older)
        assert data["latest_date"] == str(newer)

    def test_data_range_returns_all_time_bounds_for_old_data(
        self, api_client, db_session, test_device
    ):
        """/data-range returns the all-time bounds even when data is >400 days old."""
        old_date = date.today() - timedelta(days=450)
        db_session.add(
            Day(device_id=test_device.id, date=old_date, total_therapy_hours=7.0)
        )
        db_session.flush()

        response = api_client.get("/api/v1/stats/data-range")
        assert response.status_code == 200
        data = response.json()
        assert data["earliest_date"] == str(old_date)
        assert data["latest_date"] == str(old_date)


class TestStatsDataRangeProfileIsolation:
    def test_data_range_excludes_foreign_profile_days(
        self, api_client, db_session, test_device
    ):
        """/data-range does not bleed days from a second profile into the response.

        ``api_client`` operates as the actor backed by ``test_profile`` (the
        first admin user's profile).  A foreign Day seeded on a second profile's
        device must not affect the returned bounds.
        """
        today = date.today()
        own_date = today - timedelta(days=10)

        # Seed one day for the actor's profile (via test_device).
        db_session.add(
            Day(device_id=test_device.id, date=own_date, total_therapy_hours=7.0)
        )

        # Create a completely separate user + profile + device with a newer date.
        foreign_user = User(
            canonical_email=f"foreign_{uuid.uuid4().hex[:8]}@test.com",
            role="member",
        )
        db_session.add(foreign_user)
        db_session.flush()
        foreign_profile = Profile(user_id=foreign_user.id, name="Foreign")
        db_session.add(foreign_profile)
        db_session.flush()
        foreign_device = Device(
            profile_id=foreign_profile.id,
            manufacturer="Mfr",
            model="M",
            serial_number=f"SN_{uuid.uuid4().hex[:8]}",
        )
        db_session.add(foreign_device)
        db_session.flush()
        db_session.add(
            Day(
                device_id=foreign_device.id,
                date=today - timedelta(days=1),
                total_therapy_hours=7.0,
            )
        )
        db_session.flush()

        response = api_client.get("/api/v1/stats/data-range")
        assert response.status_code == 200
        data = response.json()
        # Both bounds must reflect only the actor's own day — not the foreign one.
        assert data["earliest_date"] == str(own_date)
        assert data["latest_date"] == str(own_date)


class TestStatsSummaryDaysLimitEdgeCase:
    def test_summary_returns_204_when_data_older_than_days_limit(
        self, api_client, db_session, test_device
    ):
        """/summary?days_limit=90 returns 204 when the only data is ~200 days old."""
        old_date = date.today() - timedelta(days=200)
        db_session.add(
            Day(device_id=test_device.id, date=old_date, total_therapy_hours=7.0)
        )
        db_session.flush()

        response = api_client.get("/api/v1/stats/summary?days_limit=90")
        assert response.status_code == 204
