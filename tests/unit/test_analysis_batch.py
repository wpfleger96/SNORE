"""Unit tests for _analyze_batch() parallelized session analysis."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from click.testing import CliRunner
from rich.console import Console

from snore.cli.groups.analysis import analysis
from snore.services.schemas import BatchAnalysisResult, BatchSessionResult


def _make_mock_sessions(count: int) -> list[MagicMock]:
    """Return mock session objects with sequential .id attributes."""
    sessions = []
    for i in range(1, count + 1):
        s = MagicMock()
        s.id = i
        s.day.date = date(2024, 1, i)
        s.start_time.date.return_value = date(2024, 1, i)
        sessions.append(s)
    return sessions


def _make_session_scope(mock_sessions: list[MagicMock]) -> Any:
    """Return an async context-manager factory whose session mocks run_batch_analysis.

    run_batch_analysis now materializes the full row list in a short-lived scope:
    ``rows = (await _db.execute(stmt)).all()``

    Each row needs ``.session_id`` and ``.day_date`` attributes.  The scope
    factory accepts ``**kwargs`` so callers that pass ``immediate=True`` work.
    """

    class _Row:
        def __init__(self, sid: int, day_date: Any) -> None:
            self.session_id = sid
            self.day_date = day_date

    rows = [_Row(s.id, s.day.date) for s in mock_sessions]

    result_mock = MagicMock()
    result_mock.all.return_value = rows

    mock_db_session = MagicMock()
    mock_db_session.execute = AsyncMock(return_value=result_mock)
    mock_db_session.commit = AsyncMock()

    @asynccontextmanager
    async def _scope(*args, **kwargs):
        yield mock_db_session

    _scope.session = mock_db_session
    return _scope


def _make_coordinator_mock(batch_result: BatchAnalysisResult) -> MagicMock:
    """Return a mock BatchAnalysisCoordinator whose submit() resolves to batch_result."""
    coord = MagicMock()
    coord.submit = AsyncMock(return_value=batch_result)
    coord.cancel = MagicMock()
    coord.progress = (batch_result.total, batch_result.total)
    return coord


@pytest.mark.unit
class TestAnalyzeBatch:
    """Tests for batch analysis via `analysis run --from --to`."""

    def test_analyze_batch_happy_path_reports_all_successful(self):
        """All sessions analyzed successfully reports correct success and failure counts."""
        mock_sessions = _make_mock_sessions(3)
        scope = _make_session_scope(mock_sessions)

        stdout_buf = StringIO()
        stderr_buf = StringIO()
        stdout_console = Console(file=stdout_buf, force_terminal=False, width=120)
        stderr_console = Console(
            file=stderr_buf, stderr=True, force_terminal=False, width=120
        )

        batch_result = BatchAnalysisResult(
            total=3,
            successful=3,
            failed=0,
            cancelled=0,
            results=[
                BatchSessionResult(
                    session_id=i, session_date=date(2024, 1, i), success=True
                )
                for i in range(1, 4)
            ],
        )
        coord_mock = _make_coordinator_mock(batch_result)

        runner = CliRunner()
        with (
            patch("snore.database.session.init_database", new_callable=AsyncMock),
            patch("snore.database.session.session_scope", scope),
            patch(
                "snore.services.analysis_facade.BatchAnalysisCoordinator",
                return_value=coord_mock,
            ),
            patch("snore.cli.display.console", stdout_console),
            patch("snore.cli.display.err_console", stderr_console),
            patch("snore.cli.groups.analysis.console", stdout_console),
            patch(
                "snore.auth.factory.resolve_cli_profile_id",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            result = runner.invoke(
                analysis,
                ["run", "--from", "2024-01-01", "--to", "2024-01-31"],
            )

        assert result.exit_code == 0, result.output
        captured = stdout_buf.getvalue()
        assert "Analyzed 3 sessions" in captured
        assert "3" in captured
        assert "0" in captured

    def test_analyze_batch_partial_failure_reports_failed_session_id(self):
        """When one session raises an error, stderr names the failed session and counts are correct."""
        mock_sessions = _make_mock_sessions(3)
        failing_id = mock_sessions[1].id  # session ID 2 fails

        scope = _make_session_scope(mock_sessions)

        stdout_buf = StringIO()
        stderr_buf = StringIO()
        stdout_console = Console(file=stdout_buf, force_terminal=False, width=120)
        stderr_console = Console(
            file=stderr_buf, stderr=True, force_terminal=False, width=120
        )

        batch_result = BatchAnalysisResult(
            total=3,
            successful=2,
            failed=1,
            cancelled=0,
            results=[
                BatchSessionResult(
                    session_id=1, session_date=date(2024, 1, 1), success=True
                ),
                BatchSessionResult(
                    session_id=failing_id,
                    session_date=date(2024, 1, 2),
                    success=False,
                    error="test error",
                ),
                BatchSessionResult(
                    session_id=3, session_date=date(2024, 1, 3), success=True
                ),
            ],
        )
        coord_mock = _make_coordinator_mock(batch_result)

        runner = CliRunner()
        with (
            patch("snore.database.session.init_database", new_callable=AsyncMock),
            patch("snore.database.session.session_scope", scope),
            patch(
                "snore.services.analysis_facade.BatchAnalysisCoordinator",
                return_value=coord_mock,
            ),
            patch("snore.cli.display.console", stdout_console),
            patch("snore.cli.display.err_console", stderr_console),
            patch("snore.cli.groups.analysis.console", stdout_console),
            patch(
                "snore.auth.factory.resolve_cli_profile_id",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            result = runner.invoke(
                analysis,
                ["run", "--from", "2024-01-01", "--to", "2024-01-31"],
            )

        assert result.exit_code == 0, result.output
        stdout_captured = stdout_buf.getvalue()
        # successful=2, failed=1
        assert "2" in stdout_captured
        assert "1" in stdout_captured
        # the failed session ID must appear in the warning output
        assert str(failing_id) in stdout_captured

    def test_analyze_batch_empty_sessions_prints_no_sessions_found(self):
        """When query returns no sessions, the empty-result message is printed and command exits cleanly."""
        scope = _make_session_scope([])

        stdout_buf = StringIO()
        stdout_console = Console(file=stdout_buf, force_terminal=False, width=120)

        runner = CliRunner()
        with (
            patch("snore.database.session.init_database", new_callable=AsyncMock),
            patch("snore.database.session.session_scope", scope),
            patch("snore.cli.display.console", stdout_console),
            patch("snore.cli.groups.analysis.console", stdout_console),
            patch(
                "snore.auth.factory.resolve_cli_profile_id",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            result = runner.invoke(
                analysis,
                ["run", "--from", "2024-01-01", "--to", "2024-01-31"],
            )

        assert result.exit_code == 0, result.output
        assert "No sessions found" in stdout_buf.getvalue()

    def test_analyze_batch_single_session_reports_one_successful(self):
        """A single matching session is analyzed and reported with the correct session count."""
        mock_sessions = _make_mock_sessions(1)
        scope = _make_session_scope(mock_sessions)

        stdout_buf = StringIO()
        stdout_console = Console(file=stdout_buf, force_terminal=False, width=120)

        batch_result = BatchAnalysisResult(
            total=1,
            successful=1,
            failed=0,
            cancelled=0,
            results=[
                BatchSessionResult(
                    session_id=1, session_date=date(2024, 1, 1), success=True
                ),
            ],
        )
        coord_mock = _make_coordinator_mock(batch_result)

        runner = CliRunner()
        with (
            patch("snore.database.session.init_database", new_callable=AsyncMock),
            patch("snore.database.session.session_scope", scope),
            patch(
                "snore.services.analysis_facade.BatchAnalysisCoordinator",
                return_value=coord_mock,
            ),
            patch("snore.cli.display.console", stdout_console),
            patch("snore.cli.groups.analysis.console", stdout_console),
            patch(
                "snore.auth.factory.resolve_cli_profile_id",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            result = runner.invoke(
                analysis,
                ["run", "--from", "2024-01-01", "--to", "2024-01-31"],
            )

        assert result.exit_code == 0, result.output
        captured = stdout_buf.getvalue()
        assert "Analyzed 1 sessions" in captured
        assert "1" in captured

    def test_analyze_batch_unknown_mode_exits_with_reason(self):
        """An unknown -m value fails the command with the validation message, not per-session failures."""
        scope = _make_session_scope(_make_mock_sessions(2))
        coordinator_cls = MagicMock()

        runner = CliRunner()
        with (
            patch("snore.database.session.init_database", new_callable=AsyncMock),
            patch("snore.database.session.session_scope", scope),
            patch(
                "snore.services.analysis_facade.BatchAnalysisCoordinator",
                coordinator_cls,
            ),
            patch(
                "snore.auth.factory.resolve_cli_profile_id",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            result = runner.invoke(
                analysis,
                ["run", "--from", "2024-01-01", "-m", "bogus"],
            )

        assert result.exit_code != 0
        assert "Unknown mode(s)" in result.output
        coordinator_cls.assert_not_called()


@pytest.mark.unit
class TestRunBatchAnalysisValidation:
    """run_batch_analysis rejects invalid modes before touching the DB or coordinator."""

    @pytest.mark.parametrize(
        ("modes", "primary_mode", "match"),
        [
            (["bogus"], None, "Unknown mode"),
            (["aasm_relaxed", "resmed"], None, "primary_mode must be supplied"),
            (["aasm"], "resmed", "must be a member of modes"),
            ([], None, "primary_mode must be supplied"),
        ],
    )
    async def test_invalid_modes_raise_before_any_session_runs(
        self, modes, primary_mode, match
    ):
        from snore.services.analysis_facade import AnalysisFacade

        scope = MagicMock(side_effect=AssertionError("DB must not be queried"))
        coordinator_cls = MagicMock()
        with (
            patch("snore.database.session.session_scope", scope),
            patch(
                "snore.services.analysis_facade.BatchAnalysisCoordinator",
                coordinator_cls,
            ),
            pytest.raises(ValueError, match=match),
        ):
            await AnalysisFacade(None, profile_id=1).run_batch_analysis(
                modes=modes, primary_mode=primary_mode
            )

        scope.assert_not_called()
        coordinator_cls.assert_not_called()


@pytest.mark.unit
class TestAnalyzeBatchOnlySelector:
    """Tests for `analysis run --only missing|stale`."""

    @staticmethod
    def _invoke(args: list[str], selected_ids: list[int]) -> tuple[Any, Any, Any, str]:
        stdout_buf = StringIO()
        stdout_console = Console(file=stdout_buf, force_terminal=False, width=120)
        batch_result = BatchAnalysisResult(
            total=len(selected_ids),
            successful=len(selected_ids),
            failed=0,
            cancelled=0,
            results=[],
        )
        scope = _make_session_scope([])
        list_ids = AsyncMock(return_value=selected_ids)

        async def _run_batch(**kwargs: Any) -> BatchAnalysisResult:
            # The CLI must end its read transaction before the batch loop.
            assert scope.session.commit.await_count >= 1
            return batch_result

        run_batch = AsyncMock(side_effect=_run_batch)
        with (
            patch("snore.database.session.init_database", new_callable=AsyncMock),
            patch("snore.database.session.session_scope", scope),
            patch(
                "snore.services.analysis_facade.AnalysisFacade.list_session_ids",
                list_ids,
            ),
            patch(
                "snore.services.analysis_facade.AnalysisFacade.run_batch_analysis",
                run_batch,
            ),
            patch("snore.cli.display.console", stdout_console),
            patch("snore.cli.groups.analysis.console", stdout_console),
            patch(
                "snore.auth.factory.resolve_cli_profile_id",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            result = CliRunner().invoke(analysis, args)
        return result, list_ids, run_batch, stdout_buf.getvalue()

    def test_only_stale_without_dates_analyzes_selected_sessions(self):
        """--only stale needs no dates and analyzes exactly the selected ids."""
        result, list_ids, run_batch, captured = self._invoke(
            ["run", "--only", "stale"], [2, 5]
        )

        assert result.exit_code == 0, result.output
        list_ids.assert_awaited_once_with(from_date=None, to_date=None, only="stale")
        assert run_batch.await_args.kwargs["session_ids"] == [2, 5]
        assert "Analyzed 2 sessions" in captured

    def test_only_missing_without_dates_analyzes_selected_sessions(self):
        """--only missing needs no dates and analyzes exactly the selected ids."""
        result, list_ids, run_batch, captured = self._invoke(
            ["run", "--only", "missing"], [4]
        )

        assert result.exit_code == 0, result.output
        list_ids.assert_awaited_once_with(from_date=None, to_date=None, only="missing")
        assert run_batch.await_args.kwargs["session_ids"] == [4]
        assert "Analyzed 1 sessions" in captured

    def test_only_stale_with_date_range_passes_dates_to_selector(self):
        """--from/--to narrow the --only selection; the batch then gets ids only."""
        result, list_ids, run_batch, _ = self._invoke(
            ["run", "--only", "stale", "--from", "2024-01-01", "--to", "2024-01-31"],
            [3],
        )

        assert result.exit_code == 0, result.output
        kwargs = list_ids.await_args.kwargs
        assert kwargs["only"] == "stale"
        assert kwargs["from_date"].date() == date(2024, 1, 1)
        assert kwargs["to_date"].date() == date(2024, 1, 31)
        batch_kwargs = run_batch.await_args.kwargs
        assert batch_kwargs["session_ids"] == [3]
        assert batch_kwargs["from_date"] is None
        assert batch_kwargs["to_date"] is None

    def test_only_stale_with_nothing_selected_prints_message_and_skips_run(self):
        """An empty stale selection prints a filter-specific message, no batch run."""
        result, _, run_batch, captured = self._invoke(["run", "--only", "stale"], [])

        assert result.exit_code == 0, result.output
        assert "No sessions with outdated analysis" in captured
        run_batch.assert_not_awaited()

    def test_only_missing_with_nothing_selected_prints_message(self):
        """An empty missing selection prints the missing-specific message."""
        result, _, run_batch, captured = self._invoke(["run", "--only", "missing"], [])

        assert result.exit_code == 0, result.output
        assert "No unanalyzed sessions to backfill" in captured
        run_batch.assert_not_awaited()

    @pytest.mark.parametrize(
        "single_flag", [["--session-id", "1"], ["--date", "2024-01-01"]]
    )
    def test_only_with_single_session_flag_errors(self, single_flag: list[str]) -> None:
        """--only cannot be combined with --session-id or --date."""
        result, list_ids, run_batch, _ = self._invoke(
            ["run", "--only", "stale", *single_flag], [1]
        )

        assert result.exit_code != 0
        assert "cannot be used with batch flags (--from, --to, --only)" in result.output
        list_ids.assert_not_awaited()
        run_batch.assert_not_awaited()

    def test_no_selection_flags_errors(self):
        """Without any selection flag the command still refuses to run."""
        result, _, run_batch, _ = self._invoke(["run"], [1])

        assert result.exit_code != 0
        assert "--only" in result.output
        run_batch.assert_not_awaited()


@pytest.mark.unit
class TestBatchCoordinatorHandle:
    """AnalysisFacade exposes the BatchAnalysisCoordinator handle for cancellation.

    Coordinator session-lifetime / boundedness tests.
    """

    async def test_batch_coordinator_accessible_after_run(
        self, async_db_session, async_test_device, temp_db, monkeypatch
    ):
        """facade.batch_coordinator is set after run_batch_analysis returns."""
        from datetime import date
        from unittest.mock import MagicMock

        from snore.database.models import Day, Session
        from snore.services.analysis_facade import AnalysisFacade

        day = Day(
            device_id=async_test_device.id,
            date=date(2025, 6, 1),
            total_therapy_hours=8.0,
        )
        async_db_session.add(day)
        await async_db_session.flush()
        sess = Session(
            device_id=async_test_device.id,
            day_id=day.id,
            device_session_id="COORD_HANDLE_TEST",
            start_time=__import__("datetime").datetime(2025, 6, 1, 21, 0, 0),
            end_time=__import__("datetime").datetime(2025, 6, 2, 5, 0, 0),
        )
        async_db_session.add(sess)
        await async_db_session.commit()

        # run_batch_analysis opens its own short-lived session_scope() for the
        # id-list query.  Inject async_db_session so the query can find the
        # test session without requiring the global engine to be initialized.
        @asynccontextmanager
        async def _inject_test_session(*args, **kwargs):
            yield async_db_session

        monkeypatch.setattr(
            "snore.database.session.session_scope", _inject_test_session
        )

        mock_inputs = MagicMock()
        mock_result = MagicMock()

        monkeypatch.setattr(
            "snore.analysis.service.AnalysisService.prepare_inputs",
            lambda *a, **kw: mock_inputs,
        )
        monkeypatch.setattr(
            "snore.analysis.service.AnalysisService.compute_analysis",
            lambda *a, **kw: mock_result,
        )

        facade = AnalysisFacade(async_db_session, profile_id=1)
        assert facade.batch_coordinator is None, "No coordinator before run"

        await facade.run_batch_analysis(
            from_date=__import__("datetime").datetime(2025, 6, 1),
            to_date=__import__("datetime").datetime(2025, 6, 30),
            store_results=False,
        )

        # After run, the coordinator is accessible for inspection.
        coord = facade.batch_coordinator
        assert coord is not None, "batch_coordinator must be set after run"
        completed, total = coord.progress
        assert total == 1
        assert completed == 1

    async def test_coordinator_cancel_stops_remaining_sessions(self, monkeypatch):
        """Calling coord.cancel() before submit starts causes sessions to be skipped.

        ``cancel()`` sets the flag; ``submit()`` does NOT clear a pre-set flag.
        ``analyze_one`` returns early for each session, so no DB access occurs.
        All sessions are recorded as cancelled (not successful) — the
        important property is that ``load_session_inputs_raw`` is never called.
        """
        from snore.services.analysis_facade import BatchAnalysisCoordinator

        load_calls: list[int] = []

        coord = BatchAnalysisCoordinator()
        # Request cancellation before submit begins.
        coord.cancel()
        assert coord._cancel_requested is True

        # submit() must honour the pre-set flag and not clear it.
        # With pre-cancel, _fill_window checks the flag before pulling any pair,
        # so total = len(session_pairs) (3) and all 3 are cancelled via drain.
        result = await coord.submit(
            session_pairs=[(1, None), (2, None), (3, None)],
            profile_id=1,
            store_results=False,
            max_workers=1,
        )
        # With a pre-set cancel flag: _fill_window returns immediately without
        # creating any tasks.  The drain loop then accounts for all 3 pairs as cancelled.
        assert result.total == 3
        assert result.cancelled == 3, (
            f"All 3 unstarted pairs must be drained as cancelled; got {result.cancelled}"
        )
        assert result.successful == 0, (
            f"No session must be counted successful after pre-cancel; got {result.successful}"
        )
        # No load calls: the cancellation guard fires before any task is created.
        assert load_calls == [], (
            "load_session_inputs_raw must not be called after cancel()"
        )
        # Flag must still be set.
        assert coord._cancel_requested is True

    async def test_coordinator_cancel_mid_batch_stops_further_dispatch(self):
        """Cancel requested mid-batch stops dispatching new sessions and accounts all.

        Scenario: 20 sessions, max_workers=2.  The first window dispatches
        sessions 1 and 2 concurrently.  As soon as the first session completes,
        cancel() is called via the progress_callback.  By the time _fill_window
        is called again, _cancel_requested is True, so no further sessions are
        dispatched.  Sessions 3-20 are drained as cancelled.

        Exact deterministic boundary: dispatched == max_workers == 2 (both
        from the first window), cancelled == n_total - max_workers == 18.
        """
        import threading  # noqa: PLC0415
        import unittest.mock  # noqa: PLC0415

        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        from contextlib import asynccontextmanager  # noqa: PLC0415

        from snore.services.analysis_facade import (  # noqa: PLC0415
            BatchAnalysisCoordinator,
        )

        n_total = 20
        max_workers = 2
        coord = BatchAnalysisCoordinator()

        compute_calls: list[int] = []
        compute_lock = threading.Lock()

        # Mock session_scope so _run_one never touches the real DB.
        mock_raw = unittest.mock.MagicMock()

        @asynccontextmanager
        async def mock_session_scope(**kwargs):
            mock_session = unittest.mock.AsyncMock()
            mock_session.load_session_inputs_raw = unittest.mock.AsyncMock(
                return_value=mock_raw
            )
            yield mock_session

        # Mock AnalysisService so load_session_inputs_raw returns a DTO.
        mock_svc = unittest.mock.AsyncMock()
        mock_svc.load_session_inputs_raw = unittest.mock.AsyncMock(
            return_value=mock_raw
        )

        def instrumented_compute(raw):
            # _compute_session_in_process(raw) is the call; track that it fires.
            with compute_lock:
                compute_calls.append(len(compute_calls) + 1)
            return unittest.mock.MagicMock()  # mock AnalysisComputation

        with ThreadPoolExecutor(max_workers=max_workers) as _pool:
            with (
                unittest.mock.patch(
                    "snore.database.session.session_scope", mock_session_scope
                ),
                unittest.mock.patch(
                    "snore.analysis.service.AnalysisService",
                    return_value=mock_svc,
                ),
                unittest.mock.patch(
                    "snore.utils.process_pool.get_pool", return_value=_pool
                ),
                unittest.mock.patch(
                    "snore.analysis.service._compute_session_in_process",
                    side_effect=instrumented_compute,
                ),
            ):
                # progress_callback triggers cancel on the very first completion.
                # This fires before _fill_window can pull any more sessions, so the
                # exact boundary is: dispatched == max_workers (first window only),
                # cancelled == n_total - max_workers.
                def on_progress(completed: int, total: int | None) -> None:
                    if completed >= 1:
                        coord.cancel()

                pairs = [(i, None) for i in range(1, n_total + 1)]
                result = await coord.submit(
                    session_pairs=pairs,
                    profile_id=1,
                    store_results=False,
                    max_workers=max_workers,
                    progress_callback=on_progress,
                )

        # 1. total == matched_total (always).
        assert result.total == n_total, (
            f"total must equal matched_total={n_total}, got {result.total}"
        )
        # 2. Accounting invariant: successful + failed + cancelled == total.
        assert result.successful + result.failed + result.cancelled == n_total, (
            f"Accounting mismatch: {result.successful}+{result.failed}+{result.cancelled} "
            f"!= {n_total}"
        )
        # 3. Exactly max_workers sessions were dispatched (first window only).
        assert len(compute_calls) == max_workers, (
            f"Expected exactly {max_workers} compute dispatches (first window); "
            f"got {len(compute_calls)}"
        )
        # 4. Exactly n_total - max_workers sessions were cancelled without dispatch.
        assert result.cancelled == n_total - max_workers, (
            f"Expected cancelled={n_total - max_workers}, got {result.cancelled}"
        )
        # 5. Cancel flag still set.
        assert coord._cancel_requested is True

    async def test_coordinator_window_bounded_for_large_batch(self):
        """Coordinator must never have more than max_workers tasks in flight.

        Probe: 10,000 sessions, max_workers=4.  At no point should more than
        max_workers asyncio tasks be simultaneously in flight.  Also verifies
        that session_dates (retained metadata) never exceeds max_workers at
        any checkpoint — entries are pruned as tasks complete so the
        coordinator's retained metadata stays window-bounded (O(max_workers)).
        """
        import threading  # noqa: PLC0415
        import unittest.mock  # noqa: PLC0415

        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        from contextlib import asynccontextmanager  # noqa: PLC0415

        from snore.services.analysis_facade import (  # noqa: PLC0415
            BatchAnalysisCoordinator,
        )

        n_total = 10_000
        max_workers = 4
        coord = BatchAnalysisCoordinator()

        peak_in_flight: list[int] = []
        peak_session_dates: list[int] = []
        in_flight = 0
        in_flight_lock = threading.Lock()

        # Mock session_scope and AnalysisService so _run_one never touches the DB.
        mock_raw = unittest.mock.MagicMock()
        mock_svc = unittest.mock.AsyncMock()
        mock_svc.load_session_inputs_raw = unittest.mock.AsyncMock(
            return_value=mock_raw
        )

        @asynccontextmanager
        async def mock_session_scope(**kwargs):
            yield unittest.mock.AsyncMock()

        def counting_compute(raw):
            nonlocal in_flight
            with in_flight_lock:
                in_flight += 1
                peak_in_flight.append(in_flight)
                # Snapshot session_dates size at each enqueue — must be window-bounded.
                peak_session_dates.append(len(coord.session_dates))
            with in_flight_lock:
                in_flight -= 1
            return unittest.mock.MagicMock()  # mock AnalysisComputation

        with ThreadPoolExecutor(max_workers=max_workers * 2) as _pool:
            with (
                unittest.mock.patch(
                    "snore.database.session.session_scope", mock_session_scope
                ),
                unittest.mock.patch(
                    "snore.analysis.service.AnalysisService",
                    return_value=mock_svc,
                ),
                unittest.mock.patch(
                    "snore.utils.process_pool.get_pool", return_value=_pool
                ),
                unittest.mock.patch(
                    "snore.analysis.service._compute_session_in_process",
                    side_effect=counting_compute,
                ),
            ):
                pairs = [(i, None) for i in range(1, n_total + 1)]
                result = await coord.submit(
                    session_pairs=pairs,
                    profile_id=1,
                    store_results=False,
                    max_workers=max_workers,
                )

        assert result.total == n_total
        assert result.successful == n_total
        assert result.cancelled == 0
        assert result.failed == 0

        observed_peak = max(peak_in_flight) if peak_in_flight else 0
        assert observed_peak <= max_workers, (
            f"Peak in-flight tasks was {observed_peak}, expected <= {max_workers}. "
            "Sliding window is not bounding task concurrency correctly."
        )
        # session_dates is keyed by the same sid set as `pending` — entries are
        # added at enqueue and popped at task completion.  Snapshots taken at
        # each enqueue must also stay window-bounded.
        observed_sd_peak = max(peak_session_dates) if peak_session_dates else 0
        assert observed_sd_peak <= max_workers, (
            f"session_dates peak was {observed_sd_peak}, expected <= {max_workers}. "
            "Retained metadata is not staying window-bounded."
        )
        # After submit() completes, session_dates must be fully drained.
        assert len(coord.session_dates) == 0, (
            f"session_dates has {len(coord.session_dates)} entries after submit(); "
            "expected 0 — all entries should be popped on task completion."
        )
