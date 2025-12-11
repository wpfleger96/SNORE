"""
Waveform data loading and preprocessing utilities.

This module provides functions for loading waveform data from the database,
deserializing binary data, and applying preprocessing operations like filtering
and resampling.
"""

import logging

from typing import Any

import numpy as np

from scipy import signal
from sqlalchemy.orm import Session

from snore.database.models import Waveform

logger = logging.getLogger(__name__)


class WaveformLoader:
    """
    High-level interface for loading and preprocessing waveform data.

    Example:
        >>> loader = WaveformLoader(db_session)
        >>> timestamps, values, metadata = loader.load_waveform(
        ...     session_id=123,
        ...     waveform_type="flow",
        ...     apply_filter=True
        ... )
    """

    def __init__(self, db_session: Session):
        """
        Initialize waveform loader.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session

    def load_waveform(
        self,
        session_id: int,
        waveform_type: str,
        apply_filter: bool = False,
        target_sample_rate: float | None = None,
        detect_artifacts: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Load waveform from database with optional preprocessing.

        IMPORTANT: Filtering is OFF by default to match OSCAR's approach.
        OSCAR uses no filters on flow waveforms (calcs.cpp:951-958) as raw
        waveform characteristics contain clinically important information for
        flow limitation pattern detection.

        Args:
            session_id: Database session ID
            waveform_type: Type of waveform ("flow", "pressure", etc.)
            apply_filter: Whether to apply noise filtering (default False, matches OSCAR)
            target_sample_rate: Resample to this rate (Hz), None to keep original
            detect_artifacts: Whether to detect and mark artifacts

        Returns:
            Tuple of (timestamps, values, metadata) where:
                - timestamps: 1D array of time offsets in seconds
                - values: 1D array of waveform values
                - metadata: Dict with sample_rate, unit, min_value, max_value, etc.

        Raises:
            ValueError: If waveform not found in database
        """
        logger.debug(
            f"Loading waveform: session_id={session_id}, "
            f"type={waveform_type}, filter={apply_filter}"
        )

        timestamps, values, metadata = load_waveform_from_db(
            self.db_session, session_id, waveform_type
        )

        logger.debug(f"Loaded {len(values)} samples at {metadata['sample_rate']} Hz")

        if apply_filter:
            values = apply_noise_filter(values, sample_rate=metadata["sample_rate"])
            logger.debug("Applied Butterworth filter")

        if target_sample_rate and target_sample_rate != metadata["sample_rate"]:
            timestamps, values = handle_sample_rate_conversion(
                timestamps,
                values,
                from_rate=metadata["sample_rate"],
                to_rate=target_sample_rate,
            )
            metadata["sample_rate"] = target_sample_rate
            metadata["original_sample_rate"] = metadata["sample_rate"]
            logger.debug(f"Resampled to {target_sample_rate} Hz")

        if detect_artifacts:
            artifact_mask = detect_and_mark_artifacts(values, waveform_type)
            metadata["artifact_indices"] = np.where(artifact_mask)[0]
            logger.debug(
                f"Detected {artifact_mask.sum()} artifact samples "
                f"({100 * artifact_mask.sum() / len(values):.2f}%)"
            )

        return timestamps, values, metadata


def deserialize_waveform_blob(
    blob_data: bytes, sample_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deserialize waveform blob data into timestamps and values arrays.

    This reverses the serialization performed by serialize_waveform() in
    database/importers.py. The blob format is:
    - Float32 numpy array stored as raw bytes
    - Shape: (sample_count, 2) where columns are [timestamp, value]

    Args:
        blob_data: Raw bytes from database BLOB column
        sample_count: Number of samples (for validation and reshaping)

    Returns:
        Tuple of (timestamps, values) as 1D float32 numpy arrays

    Raises:
        ValueError: If blob data is invalid or corrupted

    Example:
        >>> timestamps, values = deserialize_waveform_blob(
        ...     waveform_record.data_blob,
        ...     waveform_record.sample_count
        ... )
    """
    try:
        data = np.frombuffer(blob_data, dtype=np.float32)

        expected_size = sample_count * 2  # 2 columns: timestamp, value
        if len(data) != expected_size:
            raise ValueError(
                f"Blob size mismatch: expected {expected_size} values "
                f"({sample_count} samples × 2 columns), got {len(data)}"
            )

        data = data.reshape((sample_count, 2))

        timestamps = data[:, 0]  # Seconds from session start
        values = data[:, 1]  # Signal values

        logger.debug(
            f"Deserialized {sample_count} samples: "
            f"time range [{timestamps[0]:.2f}, {timestamps[-1]:.2f}]s, "
            f"value range [{values.min():.2f}, {values.max():.2f}]"
        )

        return timestamps, values

    except Exception as e:
        logger.error(f"Failed to deserialize waveform blob: {e}")
        raise ValueError(f"Invalid waveform blob data: {e}") from e


def load_waveform_from_db(
    db_session: Session, session_id: int, waveform_type: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load waveform from database and deserialize.

    Args:
        db_session: SQLAlchemy database session
        session_id: Database session ID
        waveform_type: Type of waveform ("flow", "pressure", "leak", etc.)

    Returns:
        Tuple of (timestamps, values, metadata) where:
            - timestamps: 1D array of time offsets in seconds
            - values: 1D array of waveform values
            - metadata: Dict with sample_rate, unit, min_value, max_value, etc.

    Raises:
        ValueError: If waveform not found in database

    Example:
        >>> timestamps, values, metadata = load_waveform_from_db(
        ...     session, session_id=123, waveform_type="flow"
        ... )
    """
    waveform = (
        db_session.query(Waveform)
        .filter_by(session_id=session_id, waveform_type=waveform_type)
        .first()
    )

    if not waveform:
        raise ValueError(
            f"Waveform not found: session_id={session_id}, type={waveform_type}"
        )

    timestamps, values = deserialize_waveform_blob(
        waveform.data_blob, waveform.sample_count or 0
    )

    metadata = {
        "waveform_id": waveform.id,
        "session_id": waveform.session_id,
        "waveform_type": waveform.waveform_type,
        "sample_rate": waveform.sample_rate,
        "unit": waveform.unit,
        "min_value": waveform.min_value,
        "max_value": waveform.max_value,
        "mean_value": waveform.mean_value,
        "sample_count": waveform.sample_count,
        "duration": timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0,
    }

    return timestamps, values, metadata


def apply_noise_filter(
    data: np.ndarray,
    sample_rate: float,
    filter_type: str = "butterworth",
    cutoff_hz: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """
    Apply noise filter to waveform data.

    Uses a low-pass Butterworth filter to remove high-frequency noise while
    preserving the breath waveform characteristics. The default cutoff of 0.5 Hz
    removes noise above breathing frequency (~12-20 breaths/min = 0.2-0.33 Hz).

    Args:
        data: 1D array of waveform values
        sample_rate: Sample rate in Hz
        filter_type: Type of filter (currently only "butterworth" supported)
        cutoff_hz: Cutoff frequency in Hz (frequencies above this are attenuated)
        order: Filter order (higher = sharper cutoff but more ringing)

    Returns:
        Filtered waveform as 1D numpy array (same length as input)

    Example:
        >>> filtered = apply_noise_filter(flow_data, sample_rate=25.0)
    """
    if filter_type != "butterworth":
        raise ValueError(f"Unsupported filter type: {filter_type}")

    if cutoff_hz >= sample_rate / 2:
        raise ValueError(
            f"Cutoff frequency ({cutoff_hz} Hz) must be less than "
            f"Nyquist frequency ({sample_rate / 2} Hz)"
        )

    # Design Butterworth low-pass filter
    # Wn is the cutoff frequency normalized to Nyquist frequency
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist

    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)

    # Apply filter (filtfilt for zero phase distortion)
    filtered_data = signal.filtfilt(b, a, data)

    logger.debug(
        f"Applied {filter_type} filter: "
        f"cutoff={cutoff_hz}Hz, order={order}, "
        f"sample_rate={sample_rate}Hz"
    )

    return filtered_data


def handle_sample_rate_conversion(
    timestamps: np.ndarray,
    values: np.ndarray,
    from_rate: float,
    to_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample waveform to a different sample rate.

    Uses scipy.signal.resample to perform high-quality resampling with
    Fourier method. Maintains signal fidelity and properly handles timestamps.

    Args:
        timestamps: 1D array of time offsets in seconds
        values: 1D array of waveform values
        from_rate: Original sample rate in Hz
        to_rate: Target sample rate in Hz

    Returns:
        Tuple of (new_timestamps, new_values) at target sample rate

    Example:
        >>> new_t, new_v = handle_sample_rate_conversion(
        ...     timestamps, values, from_rate=25.0, to_rate=10.0
        ... )
    """
    if from_rate == to_rate:
        return timestamps, values

    duration = timestamps[-1] - timestamps[0]
    new_sample_count = int(duration * to_rate)

    resampled_values = signal.resample(values, new_sample_count)

    resampled_timestamps = np.linspace(timestamps[0], timestamps[-1], new_sample_count)

    logger.debug(
        f"Resampled from {from_rate}Hz ({len(values)} samples) "
        f"to {to_rate}Hz ({new_sample_count} samples)"
    )

    return resampled_timestamps, resampled_values


def detect_and_mark_artifacts(data: np.ndarray, waveform_type: str) -> np.ndarray:
    """
    Detect and mark artifacts in waveform data.

    Identifies unrealistic values that likely indicate sensor disconnections,
    measurement errors, or other artifacts. Detection criteria vary by
    waveform type.

    Args:
        data: 1D array of waveform values
        waveform_type: Type of waveform ("flow", "pressure", etc.)

    Returns:
        Boolean mask array (same shape as data) where True = artifact

    Example:
        >>> artifact_mask = detect_and_mark_artifacts(flow_data, "flow")
        >>> clean_data = flow_data[~artifact_mask]
    """
    artifact_mask = np.zeros(len(data), dtype=bool)

    thresholds = {
        "flow": {"min": -120, "max": 120},  # L/min (typical -60 to +60)
        "pressure": {"min": 0, "max": 30},  # cmH2O (typical 4-20)
        "leak": {"min": 0, "max": 200},  # L/min (typical 0-50)
        "spo2": {"min": 0, "max": 100},  # % (typical 88-100)
        "pulse": {"min": 30, "max": 200},  # bpm (typical 40-100)
    }

    if waveform_type in thresholds:
        min_val = thresholds[waveform_type]["min"]
        max_val = thresholds[waveform_type]["max"]

        artifact_mask |= (data < min_val) | (data > max_val)

    # Detect NaN or inf values
    artifact_mask |= ~np.isfinite(data)

    # Detect sudden large jumps (sensor disconnection)
    # Only perform jump detection on arrays with sufficient samples
    # Small arrays don't have enough data for reliable baseline calculation
    if len(data) > 10:
        diffs = np.abs(np.diff(data))
        baseline_change = np.percentile(diffs, 50)
        if baseline_change > 0:
            large_jumps = diffs > (10 * baseline_change)
            artifact_mask[:-1] |= large_jumps
            artifact_mask[1:] |= large_jumps

    return artifact_mask


def handle_discontinuities(
    timestamps: np.ndarray,
    values: np.ndarray,
    gap_threshold: float = 60.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split waveform into continuous segments at discontinuities.

    Identifies gaps in the timeline (e.g., mask-off periods) and splits the
    waveform into continuous segments. Useful for handling multi-segment
    sessions where the user removed the mask.

    Args:
        timestamps: 1D array of time offsets in seconds
        values: 1D array of waveform values
        gap_threshold: Gap size in seconds to consider a discontinuity

    Returns:
        List of (timestamps, values) tuples for each continuous segment

    Example:
        >>> segments = handle_discontinuities(timestamps, values, gap_threshold=60)
        >>> print(f"Found {len(segments)} continuous segments")
    """
    if len(timestamps) == 0:
        return []
    if len(timestamps) < 2:
        return [(timestamps, values)]

    time_diffs = np.diff(timestamps)

    gap_indices = np.where(time_diffs > gap_threshold)[0]

    if len(gap_indices) == 0:
        return [(timestamps, values)]

    segments = []
    start_idx = 0

    for gap_idx in gap_indices:
        end_idx = gap_idx + 1
        segments.append((timestamps[start_idx:end_idx], values[start_idx:end_idx]))
        start_idx = end_idx

    segments.append((timestamps[start_idx:], values[start_idx:]))

    logger.info(
        f"Split waveform into {len(segments)} segments "
        f"(found {len(gap_indices)} gaps > {gap_threshold}s)"
    )

    return segments
