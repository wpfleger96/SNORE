import uuid

from datetime import UTC, date, datetime, timedelta

import pytest

from snore.database.models import Day, Device, HealthNightlySummary, Profile, User

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
        assert data["total_hours"] == pytest.approx(0.0, abs=0.1)
        assert "ahi_trend_direction" in data


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

    def test_trends_payload_has_13_keys(self, api_client, db_session, test_device):
        """/trends always returns exactly 13 metric keys."""
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
        """Apple Health series appear alongside the 13 device keys when present."""
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
