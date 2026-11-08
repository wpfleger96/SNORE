"""SpO2 reaches hypopnea detection through the analysis pipeline."""

import numpy as np

from snore.analysis.service import AnalysisInputs, AnalysisService, RawSessionBlobs
from snore.analysis.shared.types import HypopneaEvent

FLOW_RATE_HZ = 25.0
BREATH_S = 4.0
# 40 normal breaths, 4 breaths at half amplitude (a 16 s hypopnea), 20 normal.
BREATH_AMPLITUDES = [30.0] * 40 + [15.0] * 4 + [30.0] * 20
HYPOPNEA_START_S = 40 * BREATH_S
HYPOPNEA_END_S = 44 * BREATH_S


def _blob(timestamps: np.ndarray, values: np.ndarray) -> bytes:
    """Serialize like ``database.waveform_blobs.serialize_waveform``."""
    return np.column_stack([timestamps, values]).astype(np.float32).tobytes()


def _flow() -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(0.0, len(BREATH_AMPLITUDES) * BREATH_S, 1 / FLOW_RATE_HZ)
    breath_index = (timestamps // BREATH_S).astype(int)
    amplitude = np.asarray(BREATH_AMPLITUDES)[breath_index]
    return timestamps, amplitude * np.sin(2 * np.pi * timestamps / BREATH_S)


def _spo2_1hz(dip_to: float) -> tuple[np.ndarray, np.ndarray]:
    """1 Hz SpO2 at 97% that falls to ``dip_to`` during the hypopnea."""
    timestamps = np.arange(0.0, len(BREATH_AMPLITUDES) * BREATH_S, 1.0)
    values = np.full_like(timestamps, 97.0)
    values[(timestamps >= HYPOPNEA_START_S + 4) & (timestamps < HYPOPNEA_END_S)] = (
        dip_to
    )
    return timestamps, values


def _analyze_hypopneas(
    spo2: tuple[np.ndarray, np.ndarray] | None,
) -> list[HypopneaEvent]:
    flow_ts, flow_values = _flow()
    raw = RawSessionBlobs(
        session_id=1,
        flow_blob=_blob(flow_ts, flow_values),
        flow_sample_count=len(flow_ts),
        flow_metadata={"sample_rate": FLOW_RATE_HZ},
        machine_events=[],
        spo2_blob=None if spo2 is None else _blob(*spo2),
        spo2_sample_count=0 if spo2 is None else len(spo2[0]),
        modes=["aasm"],
        primary_mode="aasm",
    )
    computation = AnalysisService().compute_analysis(
        AnalysisService.prepare_inputs(raw)
    )
    return computation.summary.mode_results["aasm"].hypopneas


def _the_hypopnea(hypopneas: list[HypopneaEvent]) -> HypopneaEvent:
    matching = [
        h
        for h in hypopneas
        if h.start_time < HYPOPNEA_END_S and h.end_time > HYPOPNEA_START_S
    ]
    assert len(matching) == 1
    return matching[0]


def test_hypopnea_with_spo2_dip_has_desaturation():
    hypopnea = _the_hypopnea(_analyze_hypopneas(_spo2_1hz(dip_to=93.0)))

    assert hypopnea.has_desaturation is True


def test_hypopnea_without_dip_has_no_desaturation():
    hypopnea = _the_hypopnea(_analyze_hypopneas(_spo2_1hz(dip_to=96.0)))

    assert hypopnea.has_desaturation is False


def test_hypopnea_without_oximetry_is_unchanged_flow_only():
    with_spo2 = _the_hypopnea(_analyze_hypopneas(_spo2_1hz(dip_to=93.0)))
    without_spo2 = _the_hypopnea(_analyze_hypopneas(None))

    assert without_spo2.has_desaturation is None
    assert (without_spo2.start_time, without_spo2.end_time) == (
        with_spo2.start_time,
        with_spo2.end_time,
    )


def test_oximeter_not_connected_falls_back_to_flow_only():
    """ResMed writes -1 for every SpO2 sample when no oximeter is attached."""
    spo2_ts, _ = _spo2_1hz(dip_to=93.0)

    hypopnea = _the_hypopnea(_analyze_hypopneas((spo2_ts, np.full_like(spo2_ts, -1))))

    assert hypopnea.has_desaturation is None


class TestPrepareSpo2Inputs:
    def _inputs(self, timestamps: np.ndarray, values: np.ndarray) -> AnalysisInputs:
        flow_ts, flow_values = _flow()
        return AnalysisService.prepare_inputs(
            RawSessionBlobs(
                session_id=1,
                flow_blob=_blob(flow_ts, flow_values),
                flow_sample_count=len(flow_ts),
                flow_metadata={"sample_rate": FLOW_RATE_HZ},
                machine_events=[],
                spo2_blob=_blob(timestamps, values),
                spo2_sample_count=len(timestamps),
            )
        )

    def test_invalid_readings_are_dropped_from_own_timeline(self):
        inputs = self._inputs(
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([97.0, -1.0, 0.0, 101.0, 95.0]),
        )

        np.testing.assert_array_equal(inputs.spo2_timestamps, [0.0, 4.0])
        np.testing.assert_array_equal(inputs.spo2_values, [97.0, 95.0])

    def test_single_valid_reading_is_treated_as_absent(self):
        inputs = self._inputs(np.array([0.0, 1.0, 2.0]), np.array([97.0, -1.0, -1.0]))

        assert inputs.spo2_timestamps is None
        assert inputs.spo2_values is None
