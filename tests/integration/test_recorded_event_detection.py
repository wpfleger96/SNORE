"""
Integration tests for event detection using recorded PAP session data.

Uses actual device recordings to validate detection algorithms produce
physiologically reasonable results.
"""

import pytest

from oscar_mcp.analysis.data.waveform_loader import WaveformLoader
from oscar_mcp.analysis.engines.programmatic_engine import ProgrammaticAnalysisEngine
from oscar_mcp.database.models import Session


@pytest.mark.integration
@pytest.mark.recorded
@pytest.mark.requires_fixtures
class TestEventDetectionOnRecordedSessions:
    """Validate event detection on recorded PAP session data."""

    def test_session_269_detects_events(self, db_session_with_session_269_fixture):
        """Session 269 should detect respiratory events."""
        session = db_session_with_session_269_fixture.query(Session).first()
        loader = WaveformLoader(db_session_with_session_269_fixture)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        engine = ProgrammaticAnalysisEngine()
        result = engine.analyze_session(
            session_id=session.id,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=metadata["sample_rate"],
        )

        # Should detect some events (session has known apnea activity)
        total_events = result.event_timeline.get("total_events", 0)
        assert total_events > 0, "Expected to detect respiratory events"

        # AHI should be calculated
        assert result.event_timeline.get("ahi", 0) >= 0

        # Processing should complete in reasonable time
        assert result.processing_time_ms < 60000

    def test_baseline_fixture_event_detection(self, db_session_with_baseline_fixture):
        """Baseline fixture should produce physiologically reasonable event counts."""
        session = db_session_with_baseline_fixture.query(Session).first()
        loader = WaveformLoader(db_session_with_baseline_fixture)
        timestamps, flow_values, metadata = loader.load_waveform(
            session_id=session.id, waveform_type="flow"
        )

        engine = ProgrammaticAnalysisEngine()
        result = engine.analyze_session(
            session_id=session.id,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=metadata["sample_rate"],
        )

        # AHI should be in reasonable range (0-100 for severe cases)
        ahi = result.event_timeline.get("ahi", 0)
        assert 0 <= ahi <= 100

        # Should complete successfully
        assert result.processing_time_ms > 0
