"""Report generation service for self-contained HTML therapy reports."""

from __future__ import annotations

from datetime import date
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.svg_charts import render_trend_line
from snore.database import models
from snore.services._base import ProfileScopedService
from snore.services.schemas import DeviceInfo, TherapySummary, TrendsResponse
from snore.services.stats_service import StatsService

__all__ = ["ReportService"]

# ---------------------------------------------------------------------------
# Jinja2 environment (module-level, shared across instances)
# ---------------------------------------------------------------------------


def _fmt_f(value: float | None, decimals: int = 1) -> str:
    """Format a float to fixed decimal places, returning '—' for None."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _fmt_i(value: int | None) -> str:
    """Format an integer with thousands separator, returning '—' for None."""
    if value is None:
        return "—"
    return f"{value:,}"


_ENV: Environment = Environment(
    loader=PackageLoader("snore", "templates/reports"),
    autoescape=select_autoescape(["html"]),
)
_ENV.filters["fmt_f"] = _fmt_f
_ENV.filters["fmt_i"] = _fmt_i


# ---------------------------------------------------------------------------
# Delta helpers
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER = {"avg_ahi", "avg_leak"}
_HIGHER_IS_BETTER = {"avg_hours", "avg_spo2"}

_DELTA_METRICS: list[tuple[str, str, str]] = [
    # (label, attr on TherapySummary, direction-key)
    ("Avg Usage (hr)", "avg_hours", "avg_hours"),
    ("Avg AHI", "avg_ahi", "avg_ahi"),
    ("Avg Pressure", "avg_pressure", ""),
    ("Avg EPAP", "avg_epap", ""),
    ("Avg Leak", "avg_leak", "avg_leak"),
    ("Avg SpO₂ (%)", "avg_spo2", "avg_spo2"),
    ("Min SpO₂ (%)", "min_spo2", ""),
    ("Avg Pulse (bpm)", "avg_pulse", ""),
    ("Avg Resp Rate", "avg_respiratory_rate", ""),
]


def _delta_color(delta: float, direction_key: str) -> str:
    """Return a CSS hex color string for a delta value, or empty string."""
    if direction_key in _LOWER_IS_BETTER:
        return "#16a34a" if delta < 0 else "#dc2626"
    if direction_key in _HIGHER_IS_BETTER:
        return "#16a34a" if delta > 0 else "#dc2626"
    return ""


def _build_deltas(
    summary_a: TherapySummary | None,
    summary_b: TherapySummary | None,
) -> list[dict[str, Any]]:
    """Compute per-metric delta rows for the comparison table."""
    rows: list[dict[str, Any]] = []
    for label, attr, direction_key in _DELTA_METRICS:
        val_a: float | None = getattr(summary_a, attr, None) if summary_a else None
        val_b: float | None = getattr(summary_b, attr, None) if summary_b else None
        a_str = _fmt_f(val_a)
        b_str = _fmt_f(val_b)
        if val_a is None or val_b is None:
            rows.append(
                {
                    "label": label,
                    "a_str": a_str,
                    "b_str": b_str,
                    "delta_str": "—",
                    "color": "",
                }
            )
            continue
        delta = val_b - val_a
        sign = "+" if delta >= 0 else "−"
        delta_str = f"{sign}{abs(delta):.1f}"
        color = _delta_color(delta, direction_key) if delta != 0.0 else ""
        rows.append(
            {
                "label": label,
                "a_str": a_str,
                "b_str": b_str,
                "delta_str": delta_str,
                "color": color,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# ReportService
# ---------------------------------------------------------------------------


class ReportService(ProfileScopedService):
    """Generate self-contained HTML therapy reports.

    I/O–compute split (§7)
    -----------------------
    Each ``generate_*_report`` method is structured as an explicit two-phase call:

    1. **Fetch phase** (``_fetch_summary_data`` / ``_fetch_comparison_data``): all
       database queries run via ``StatsService`` and return plain Python objects
       (Pydantic schemas).  The session is not needed after this phase completes.
    2. **Render phase** (``_render_summary`` / ``_render_comparison``): pure
       Jinja2 template rendering that takes only plain Python objects.  No session
       access occurs here, making it safe to call after the session is closed.

    PR-2 can adapt the fetch phase to an async session and keep the render phase
    unchanged.
    """

    def __init__(self, db_session: AsyncSession, profile_id: int) -> None:
        super().__init__(db_session, profile_id)
        self._stats = StatsService(db_session, profile_id)

    async def _first_device(self) -> DeviceInfo | None:
        """Return the first device in this profile as a plain detached schema, or None."""
        device = (
            (
                await self.db_session.execute(
                    select(models.Device)
                    .where(self._profile_filter())
                    .order_by(models.Device.first_seen)
                )
            )
            .scalars()
            .first()
        )
        if device is None:
            return None
        return DeviceInfo.model_validate(device)

    # --- Fetch helpers (I/O phase) ---

    async def _fetch_summary_data(
        self, from_date: date, to_date: date
    ) -> dict[str, Any]:
        """Fetch all data needed to render a summary report.

        Returns plain Python objects; no ORM session access after this call.
        """
        summary = await self._stats.get_summary(from_date=from_date, to_date=to_date)
        monthly = await self._stats.get_period_statistics(
            "month", from_date=from_date, to_date=to_date
        )
        trends = await self._stats.get_trends(
            "week", from_date=from_date, to_date=to_date
        )
        device = await self._first_device()
        return {
            "summary": summary,
            "monthly": monthly,
            "trends": trends,
            "device": device,
        }

    async def _fetch_comparison_data(
        self,
        range_a: tuple[date, date],
        range_b: tuple[date, date],
    ) -> dict[str, Any]:
        """Fetch all data needed to render a comparison report.

        Returns plain Python objects; no ORM session access after this call.
        """
        summary_a = await self._stats.get_summary(
            from_date=range_a[0], to_date=range_a[1]
        )
        summary_b = await self._stats.get_summary(
            from_date=range_b[0], to_date=range_b[1]
        )
        device = await self._first_device()
        deltas = _build_deltas(summary_a, summary_b)
        return {
            "summary_a": summary_a,
            "summary_b": summary_b,
            "device": device,
            "deltas": deltas,
        }

    # --- Render helpers (compute phase — no session needed) ---

    @staticmethod
    def _render_summary(
        from_date: date,
        to_date: date,
        data: dict[str, Any],
    ) -> str:
        """Render summary HTML from pre-fetched data.  No DB access."""
        trends: TrendsResponse = data["trends"]
        ahi_chart = render_trend_line(
            trends.ahi,
            color="#dc2626",
            y_label="AHI",
        )
        usage_chart = render_trend_line(
            trends.usage,
            color="#2563eb",
            y_label="Hours",
        )
        tmpl = _ENV.get_template("summary.html")
        return tmpl.render(
            from_date=from_date,
            to_date=to_date,
            generated_on=date.today(),
            summary=data["summary"],
            monthly=data["monthly"],
            ahi_chart=ahi_chart,
            usage_chart=usage_chart,
            device=data["device"],
        )

    @staticmethod
    def _render_comparison(
        range_a: tuple[date, date],
        range_b: tuple[date, date],
        data: dict[str, Any],
    ) -> str:
        """Render comparison HTML from pre-fetched data.  No DB access."""
        tmpl = _ENV.get_template("comparison.html")
        return tmpl.render(
            range_a=range_a,
            range_b=range_b,
            generated_on=date.today(),
            summary_a=data["summary_a"],
            summary_b=data["summary_b"],
            device=data["device"],
            deltas=data["deltas"],
        )

    # --- Public API (fetch then render) ---

    async def generate_summary_report(self, from_date: date, to_date: date) -> str:
        """
        Render a complete HTML summary therapy report for the given date range.

        Structured as fetch (DB I/O, session required) then render (pure
        Jinja2, no session needed).  The injected session is caller-owned —
        only the owning scope (``get_db()`` / ``db_session()``) may close it.
        """
        data = await self._fetch_summary_data(from_date, to_date)
        return self._render_summary(from_date, to_date, data)

    async def generate_comparison_report(
        self,
        range_a: tuple[date, date],
        range_b: tuple[date, date],
    ) -> str:
        """
        Render a complete HTML comparison report for two date ranges.

        Structured as fetch (DB I/O, session required) then render (pure
        Jinja2, no session needed).  The injected session is caller-owned —
        only the owning scope (``get_db()`` / ``db_session()``) may close it.
        """
        data = await self._fetch_comparison_data(range_a, range_b)
        return self._render_comparison(range_a, range_b, data)
