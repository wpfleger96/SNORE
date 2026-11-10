"""Import date ranges select sessions by night date (issue #488).

A ``date_from``/``date_to`` range passed to ``parse_sessions`` must select the
same nights a session is filed under (``Day.date``, noon split): a session that
starts at 01:34 on 2024-06-21 belongs to the 2024-06-20 night, so a one-day
range of 06-20 imports it and a one-day range of 06-21 does not.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from snore.parsers.oscar_device import OscarDeviceParser
from snore.parsers.resmed_edf import ResmedEDFParser
from snore.parsers.unified import DeviceInfo
from tests.helpers.oscar_builders import make_summary

pytestmark = pytest.mark.unit

AFTER_MIDNIGHT = "20240621_013400"
BEFORE_MIDNIGHT = "20240620_220000"


def _resmed_card(tmp_path: Path, *segment_ids: str) -> Path:
    """An SD card whose DATALOG holds one (header-less) BRP file per segment."""
    datalog = tmp_path / "DATALOG" / "2024"
    datalog.mkdir(parents=True)
    for segment_id in segment_ids:
        (datalog / f"{segment_id}_BRP.edf").touch()
    return tmp_path


def _resmed_start_times(
    card: Path,
    date_from: str,
    date_to: str,
    *,
    parallel: bool = False,
    header_starts: dict[str, datetime] | None = None,
) -> list[datetime]:
    """Parse ``card`` through ``parse_sessions`` and return yielded start times.

    Only the EDF decode is faked: each chain parses to a session starting at
    its chain id, or at ``header_starts[chain_id]`` when given, so discovery,
    range filtering and the post-parse range guard all run for real.  With
    ``parallel`` and two or more chains, the guard runs in the pool worker.
    """

    def fake_parse_night(
        self: ResmedEDFParser, night_date: str, chain_id: str, *args: object
    ) -> MagicMock:
        session = MagicMock()
        session.start_time = (header_starts or {}).get(
            chain_id, datetime.strptime(chain_id, "%Y%m%d_%H%M%S")
        )
        return session

    with (
        ThreadPoolExecutor(max_workers=2) as pool,
        patch.object(ResmedEDFParser, "_parse_night_session", fake_parse_night),
        patch.object(ResmedEDFParser, "_load_str_caches", return_value=({}, {})),
        patch.object(ResmedEDFParser, "_detect_series11", return_value=False),
        patch.object(ResmedEDFParser, "get_device_info", return_value=MagicMock()),
        patch("snore.parsers.base.get_pool", return_value=pool),
    ):
        sessions = ResmedEDFParser().parse_sessions(
            card, date_from=date_from, date_to=date_to, parallel=parallel
        )
        return sorted(s.start_time for s in sessions)


class TestResmedNightRange:
    def test_after_midnight_session_in_previous_night_range_is_imported(self, tmp_path):
        card = _resmed_card(tmp_path, AFTER_MIDNIGHT)

        starts = _resmed_start_times(card, "2024-06-20", "2024-06-20")

        assert starts == [datetime(2024, 6, 21, 1, 34)]

    def test_after_midnight_session_in_its_calendar_day_range_is_skipped(
        self, tmp_path
    ):
        card = _resmed_card(tmp_path, AFTER_MIDNIGHT)

        starts = _resmed_start_times(card, "2024-06-21", "2024-06-21")

        assert starts == []

    def test_pre_midnight_session_in_its_night_range_is_imported(self, tmp_path):
        card = _resmed_card(tmp_path, BEFORE_MIDNIGHT)

        starts = _resmed_start_times(card, "2024-06-20", "2024-06-20")

        assert starts == [datetime(2024, 6, 20, 22, 0)]

    @pytest.mark.parametrize("parallel", [False, True])
    def test_pre_and_post_midnight_sessions_of_one_night_are_imported(
        self, tmp_path, parallel
    ):
        # More than the 4 h chaining gap apart, so they are separate chains.
        card = _resmed_card(tmp_path, "20240620_200000", AFTER_MIDNIGHT)

        starts = _resmed_start_times(
            card, "2024-06-20", "2024-06-20", parallel=parallel
        )

        assert starts == [datetime(2024, 6, 20, 20, 0), datetime(2024, 6, 21, 1, 34)]

    @pytest.mark.parametrize("parallel", [False, True])
    def test_header_start_past_noon_on_other_night_is_skipped(self, tmp_path, parallel):
        """The filename says the 06-20 night, but the parsed start is 06-21's.

        That session is filed under Day 2024-06-21, so a 06-20 range must not
        import it, while the other 06-20 chain still is.
        """
        card = _resmed_card(tmp_path, "20240620_200000", "20240621_115950")

        starts = _resmed_start_times(
            card,
            "2024-06-20",
            "2024-06-20",
            parallel=parallel,
            header_starts={"20240621_115950": datetime(2024, 6, 21, 12, 0, 5)},
        )

        assert starts == [datetime(2024, 6, 20, 20, 0)]

    @pytest.mark.parametrize("parallel", [False, True])
    def test_chain_with_pre_noon_stub_is_imported_on_parsed_night(
        self, tmp_path, parallel
    ):
        """A stub at 11:59:50 chained to therapy that starts at 12:00:05.

        The chain's first segment is on the 06-20 night, but the session starts
        at its first valid segment, so it is filed under Day 2024-06-21: the
        06-21 range imports it and the 06-20 range does not.
        """
        card = _resmed_card(
            tmp_path,
            "20240620_200000",
            "20240621_115950",
            "20240621_120005",
            "20240621_200000",
        )
        header_starts = {"20240621_115950": datetime(2024, 6, 21, 12, 0, 5)}

        night_21 = _resmed_start_times(
            card,
            "2024-06-21",
            "2024-06-21",
            parallel=parallel,
            header_starts=header_starts,
        )
        night_20 = _resmed_start_times(
            card,
            "2024-06-20",
            "2024-06-20",
            parallel=parallel,
            header_starts=header_starts,
        )

        assert night_21 == [datetime(2024, 6, 21, 12, 0, 5), datetime(2024, 6, 21, 20)]
        assert night_20 == [datetime(2024, 6, 20, 20)]

    @pytest.mark.parametrize("parallel", [False, True])
    def test_chain_with_pre_noon_stub_is_skipped_on_stub_night(
        self, tmp_path, parallel
    ):
        """The same noon-spanning chain one night earlier, seen from the stub side.

        Filed under Day 2024-06-20 by its 12:00:05 start, so the 06-20 range
        imports it and the 06-19 range (the stub's night) does not.
        """
        card = _resmed_card(
            tmp_path,
            "20240619_200000",
            "20240620_115950",
            "20240620_120005",
            "20240620_200000",
        )
        header_starts = {"20240620_115950": datetime(2024, 6, 20, 12, 0, 5)}

        night_20 = _resmed_start_times(
            card,
            "2024-06-20",
            "2024-06-20",
            parallel=parallel,
            header_starts=header_starts,
        )
        night_19 = _resmed_start_times(
            card,
            "2024-06-19",
            "2024-06-19",
            parallel=parallel,
            header_starts=header_starts,
        )

        assert night_20 == [datetime(2024, 6, 20, 12, 0, 5), datetime(2024, 6, 20, 20)]
        assert night_19 == [datetime(2024, 6, 19, 20)]


def _oscar_start_times(
    tmp_path: Path,
    start_utc: datetime,
    date_from: str,
    date_to: str,
    timezone_name: str | None = None,
) -> list[datetime]:
    """Parse one OSCAR session starting at ``start_utc`` through ``parse_sessions``."""
    first_ms = int(start_utc.timestamp() * 1000)
    summary = make_summary(first_ms, first_ms + 6 * 3600 * 1000)
    summary_path = tmp_path / "1.000"
    summary_path.touch()
    unit = (summary.session_id, summary_path, None)

    parser = OscarDeviceParser()
    with (
        patch.object(parser, "_resolve_units", return_value=(tmp_path, [unit])),
        patch.object(
            parser,
            "get_device_info",
            return_value=DeviceInfo(
                manufacturer="OSCAR", model="Test", serial_number="TEST123"
            ),
        ),
        patch("snore.parsers.oscar_device.parse_summary_file", return_value=summary),
    ):
        sessions = parser.parse_sessions(
            tmp_path,
            date_from=date_from,
            date_to=date_to,
            parallel=False,
            timezone_name=timezone_name,
        )
        return [s.start_time for s in sessions]


class TestOscarNightRange:
    def test_after_midnight_session_in_previous_night_range_is_imported(self, tmp_path):
        starts = _oscar_start_times(
            tmp_path,
            datetime(2024, 6, 21, 1, 34, tzinfo=UTC),
            "2024-06-20",
            "2024-06-20",
        )

        assert starts == [datetime(2024, 6, 21, 1, 34)]

    def test_after_midnight_session_in_its_calendar_day_range_is_skipped(
        self, tmp_path
    ):
        starts = _oscar_start_times(
            tmp_path,
            datetime(2024, 6, 21, 1, 34, tzinfo=UTC),
            "2024-06-21",
            "2024-06-21",
        )

        assert starts == []

    def test_declared_timezone_selects_by_local_night(self, tmp_path):
        # 05:34 UTC on 06-21 is 01:34 EDT on 06-21: the local 06-20 night.
        start_utc = datetime(2024, 6, 21, 5, 34, tzinfo=UTC)

        in_range = _oscar_start_times(
            tmp_path, start_utc, "2024-06-20", "2024-06-20", "America/New_York"
        )
        out_of_range = _oscar_start_times(
            tmp_path, start_utc, "2024-06-21", "2024-06-21", "America/New_York"
        )

        assert in_range == [datetime(2024, 6, 21, 1, 34)]
        assert out_of_range == []
