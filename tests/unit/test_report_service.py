"""Unit tests for ReportService."""

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Day, Device, Session
from snore.services.report_service import ReportService
from snore.services.stats_service import StatsService


async def _create_day(
    db_session: AsyncSession,
    device: Device,
    day_date: date,
    duration_hours: float = 7.5,
    ahi: float = 3.0,
    **day_kwargs: Any,
) -> Day:
    """Create a Day with an associated Session so StatsService can aggregate it."""
    day = Day(
        device_id=device.id,
        date=day_date,
        total_therapy_hours=duration_hours,
        ahi=ahi,
        **day_kwargs,
    )
    db_session.add(day)
    await db_session.flush()

    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"test_{day_date.isoformat()}_{device.id}",
        start_time=datetime.combine(day_date, datetime.min.time()),
        end_time=datetime.combine(day_date, datetime.min.time())
        + timedelta(hours=duration_hours),
        duration_seconds=duration_hours * 3600,
    )
    db_session.add(sess)
    await db_session.flush()
    return day


async def _seed_days(
    db_session: AsyncSession,
    device: Device,
    from_date: date,
    to_date: date,
    step_days: int = 7,
) -> None:
    """Seed one Day per step across [from_date, to_date]."""
    current = from_date
    while current <= to_date:
        await _create_day(db_session, device, current)
        current += timedelta(days=step_days)


class TestReportService:
    """Tests for ReportService.generate_summary_report and generate_comparison_report."""

    async def test_summary_starts_with_doctype_and_contains_dates(
        self, async_db_session, async_test_device
    ):
        """Generated summary HTML starts with <!DOCTYPE html> and includes the range dates."""
        today = date.today()
        from_date = today - timedelta(days=30)
        to_date = today

        service = ReportService(async_db_session, profile_id=1)
        html = await service.generate_summary_report(from_date, to_date)

        assert html.startswith("<!DOCTYPE html>")
        assert str(from_date) in html
        assert str(to_date) in html

    async def test_summary_empty_db_renders_no_data_message(self, async_db_session):
        """Empty database renders a valid report with the no-data notice instead of raising."""
        today = date.today()
        from_date = today - timedelta(days=30)
        to_date = today

        service = ReportService(async_db_session, profile_id=1)
        html = await service.generate_summary_report(from_date, to_date)

        assert html.startswith("<!DOCTYPE html>")
        assert "No therapy data" in html

    async def test_summary_with_data_contains_device_model_svg_and_monthly_row(
        self, async_db_session, async_test_device
    ):
        """With seeded data the report contains the device model, an SVG chart, and a monthly table row."""
        today = date.today()
        from_date = today - timedelta(days=60)
        to_date = today

        await _seed_days(
            async_db_session, async_test_device, from_date, to_date, step_days=7
        )

        service = ReportService(async_db_session, profile_id=1)
        html = await service.generate_summary_report(from_date, to_date)

        assert async_test_device.model in html
        assert "<svg" in html
        assert "<td>" in html
        assert "days" in html.lower() or "/</td>" in html or "<td>7" in html

    async def test_reports_do_not_render_effectiveness(
        self, async_db_session, async_test_device
    ):
        """The effectiveness band is experimental, so no report shows it."""
        today = date.today()
        range_a = (today - timedelta(days=60), today - timedelta(days=31))
        range_b = (today - timedelta(days=30), today)
        await _seed_days(async_db_session, async_test_device, range_a[0], range_b[1])

        service = ReportService(async_db_session, profile_id=1)
        summary_html = await service.generate_summary_report(range_a[0], range_b[1])
        comparison_html = await service.generate_comparison_report(range_a, range_b)

        assert "Avg AHI" in summary_html
        assert "Avg AHI" in comparison_html
        assert "Effectiveness" not in summary_html
        assert "Effectiveness" not in comparison_html

    async def test_comparison_contains_both_range_headings_and_delta_sign(
        self, async_db_session, async_test_device
    ):
        """Comparison report includes both range headings and at least one delta cell with +/-."""
        today = date.today()
        range_a = (today - timedelta(days=60), today - timedelta(days=31))
        range_b = (today - timedelta(days=30), today)

        await _seed_days(
            async_db_session, async_test_device, range_a[0], range_a[1], step_days=7
        )
        await _seed_days(
            async_db_session, async_test_device, range_b[0], range_b[1], step_days=7
        )

        service = ReportService(async_db_session, profile_id=1)
        html = await service.generate_comparison_report(range_a, range_b)

        assert str(range_a[0]) in html
        assert str(range_b[0]) in html
        assert "+" in html or "−" in html

    async def test_comparison_one_empty_range_renders_without_raising(
        self, async_db_session, async_test_device
    ):
        """Comparison with one empty range renders successfully and shows the no-data notice."""
        today = date.today()
        range_a = (today - timedelta(days=30), today)
        range_b = (today + timedelta(days=1), today + timedelta(days=30))

        await _seed_days(
            async_db_session, async_test_device, range_a[0], range_a[1], step_days=7
        )

        service = ReportService(async_db_session, profile_id=1)
        html = await service.generate_comparison_report(range_a, range_b)

        assert html.startswith("<!DOCTYPE html>")
        assert "No therapy data" in html

    async def test_summary_autoescape_escapes_device_model_script_tag(
        self, async_db_session, async_test_profile
    ):
        """Device model containing <script> arrives HTML-escaped in the report."""
        import uuid

        device = Device(
            profile_id=async_test_profile.id,
            manufacturer="ACME",
            model="<script>alert(1)</script>",
            serial_number=f"XSS_{uuid.uuid4().hex[:8]}",
        )
        async_db_session.add(device)
        await async_db_session.flush()

        today = date.today()
        from_date = today - timedelta(days=7)
        to_date = today

        service = ReportService(async_db_session, profile_id=1)
        html = await service.generate_summary_report(from_date, to_date)

        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    async def test_stats_service_from_date_to_date_filters_correctly(
        self, async_db_session, async_test_device
    ):
        """get_summary(from_date=..., to_date=...) includes only in-range days."""
        today = date.today()
        in_range_date = today - timedelta(days=5)
        out_of_range_date = today - timedelta(days=30)

        await _create_day(
            async_db_session,
            async_test_device,
            in_range_date,
            duration_hours=8.0,
            ahi=2.0,
        )
        await _create_day(
            async_db_session,
            async_test_device,
            out_of_range_date,
            duration_hours=6.0,
            ahi=10.0,
        )

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary(
            from_date=today - timedelta(days=7), to_date=today
        )

        assert summary is not None
        assert summary.days_with_data == 1
        assert summary.avg_ahi == pytest.approx(2.0, rel=1e-3)


class TestReportServiceSessionOwnership:
    """ReportService must not close its injected (caller-owned) session.

    The session lifecycle is owned by get_db()/db_session()/session_scope();
    closing it inside a service method silently rolls back any provisioning
    done in the same transaction and breaks subsequent queries on the same
    session.
    """

    async def test_generate_summary_report_leaves_session_valid_for_subsequent_query(
        self, async_db_session, async_test_device
    ):
        """A query on the same session after generate_summary_report must not raise.

        Regression for the defect where ReportService closed its injected
        session after the fetch phase, causing ``InvalidRequestError`` on any
        subsequent use of the same session.
        """
        from sqlalchemy import select  # noqa: PLC0415

        from snore.database.models import Device  # noqa: PLC0415

        service = ReportService(async_db_session, profile_id=1)
        await service.generate_summary_report(
            date.today() - timedelta(days=7), date.today()
        )

        # Any query on the same session must succeed without raising
        # InvalidRequestError or similar closed-transaction errors.
        result = (await async_db_session.execute(select(Device))).scalars().all()
        # The exact rows don't matter — what matters is the query completed.
        assert isinstance(result, list)

    async def test_generate_comparison_report_leaves_session_valid_for_subsequent_query(
        self, async_db_session, async_test_device
    ):
        """A query on the same session after generate_comparison_report must not raise."""
        from sqlalchemy import select  # noqa: PLC0415

        from snore.database.models import Device  # noqa: PLC0415

        service = ReportService(async_db_session, profile_id=1)
        today = date.today()
        await service.generate_comparison_report(
            range_a=(today - timedelta(days=30), today - timedelta(days=15)),
            range_b=(today - timedelta(days=14), today),
        )

        result = (await async_db_session.execute(select(Device))).scalars().all()
        assert isinstance(result, list)
