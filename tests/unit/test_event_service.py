"""Unit tests for EventService."""

from datetime import datetime, timedelta

from snore.database.models import Event
from snore.services.event_service import EVENT_MATCH_TOLERANCE_SECONDS, EventService


class TestEventService:
    """Tests for EventService.match_events()."""

    def test_empty_events(self):
        """Empty event lists return all zeros."""
        result = EventService.match_events([], [])

        assert result.machine_count == 0
        assert result.programmatic_count == 0
        assert result.matched == 0
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_all_matched(self):
        """Identical timestamps result in all matched, no false positives or negatives."""
        machine_times = [10.0, 20.0, 30.0]
        programmatic_times = [10.0, 20.0, 30.0]

        result = EventService.match_events(machine_times, programmatic_times)

        assert result.machine_count == 3
        assert result.programmatic_count == 3
        assert result.matched == 3
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_within_tolerance(self):
        """Events within tolerance are matched."""
        machine_times = [10.0, 20.0]
        programmatic_times = [13.0, 22.0]

        result = EventService.match_events(machine_times, programmatic_times)

        assert result.machine_count == 2
        assert result.programmatic_count == 2
        assert result.matched == 2
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_outside_tolerance(self):
        """Events outside tolerance result in false positives and false negatives."""
        machine_times = [10.0]
        programmatic_times = [16.1]

        result = EventService.match_events(machine_times, programmatic_times)

        assert result.machine_count == 1
        assert result.programmatic_count == 1
        assert result.matched == 0
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_false_negatives_only(self):
        """Machine events with no programmatic events result in false negatives."""
        machine_times = [10.0, 20.0, 30.0]
        programmatic_times = []

        result = EventService.match_events(machine_times, programmatic_times)

        assert result.machine_count == 3
        assert result.programmatic_count == 0
        assert result.matched == 0
        assert result.false_positives == 0
        assert result.false_negatives == 3

    def test_false_positives_only(self):
        """Programmatic events with no machine events result in false positives."""
        machine_times = []
        programmatic_times = [10.0, 20.0, 30.0]

        result = EventService.match_events(machine_times, programmatic_times)

        assert result.machine_count == 0
        assert result.programmatic_count == 3
        assert result.matched == 0
        assert result.false_positives == 3
        assert result.false_negatives == 0

    def test_mixed_match(self):
        """Mixed scenario with some matched and some unmatched events."""
        machine_times = [10.0, 20.0, 30.0, 40.0]
        programmatic_times = [11.0, 35.0, 50.0]

        result = EventService.match_events(machine_times, programmatic_times)

        assert result.machine_count == 4
        assert result.programmatic_count == 3
        assert result.matched == 2
        assert result.false_positives == 1
        assert result.false_negatives == 2

    def test_custom_tolerance(self):
        """Custom tolerance parameter allows tighter matching."""
        machine_times = [10.0, 20.0]
        programmatic_times = [12.5, 22.5]

        result_tight = EventService.match_events(
            machine_times, programmatic_times, tolerance=2.0
        )
        assert result_tight.matched == 0
        assert result_tight.false_positives == 2
        assert result_tight.false_negatives == 2

        result_loose = EventService.match_events(
            machine_times, programmatic_times, tolerance=3.0
        )
        assert result_loose.matched == 2
        assert result_loose.false_positives == 0
        assert result_loose.false_negatives == 0

    def test_unsorted_input(self):
        """Unsorted input lists are handled correctly."""
        machine_times = [30.0, 10.0, 20.0]
        programmatic_times = [20.0, 30.0, 10.0]

        result = EventService.match_events(machine_times, programmatic_times)

        assert result.machine_count == 3
        assert result.programmatic_count == 3
        assert result.matched == 3
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_default_tolerance_constant(self):
        """Default tolerance uses the module constant."""
        machine_times = [10.0]
        programmatic_times = [10.0 + EVENT_MATCH_TOLERANCE_SECONDS]

        result = EventService.match_events(machine_times, programmatic_times)

        assert result.matched == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_edge_case_exact_tolerance_boundary(self):
        """Events exactly at tolerance boundary are matched."""
        machine_times = [10.0]
        programmatic_times = [15.0]

        result = EventService.match_events(
            machine_times, programmatic_times, tolerance=5.0
        )

        assert result.matched == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_extra_programmatic_events_are_false_positives(self):
        """A machine event pairs with one programmatic event; the rest are FPs."""
        machine_times = [10.0]
        programmatic_times = [9.0, 10.0, 11.0]

        result = EventService.match_events(machine_times, programmatic_times)

        assert result.machine_count == 1
        assert result.programmatic_count == 3
        assert result.matched == 1
        assert result.false_positives == 2
        assert result.false_negatives == 0

    def test_one_detection_cannot_match_two_machine_events(self):
        """Two machine events near one detection leave one false negative."""
        result = EventService.match_events([100.0, 105.0], [102.0])

        assert result.matched == 1
        assert result.false_positives == 0
        assert result.false_negatives == 1


class TestMachineApneaHypopneaTimes:
    """Tests for EventService.get_machine_apnea_hypopnea_times()."""

    async def test_non_apnea_device_flags_do_not_steal_matches(
        self,
        async_db_session,
        async_test_profile,
        async_test_device,
        async_test_session_factory,
    ):
        """VS/FL flags just before an OA are dropped, so the OA still matches."""
        start = datetime(2025, 1, 10, 22, 0)
        session = await async_test_session_factory(async_test_device.id, start)
        oa_time = start + timedelta(hours=1)
        for event_type, offset_s in (("VS", -2.0), ("FL", -1.0), ("OA", 0.0)):
            async_db_session.add(
                Event(
                    session_id=session.id,
                    event_type=event_type,
                    start_time=oa_time + timedelta(seconds=offset_s),
                    duration_seconds=10.0,
                )
            )
        await async_db_session.flush()
        svc = EventService(async_db_session, profile_id=async_test_profile.id)

        machine_times = await svc.get_machine_apnea_hypopnea_times(session.id)
        result = EventService.match_events(machine_times, [oa_time.timestamp()])

        assert machine_times == [oa_time.timestamp()]
        assert result.matched == 1
        assert result.false_negatives == 0
        assert result.false_positives == 0
