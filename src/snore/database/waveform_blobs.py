"""Stored waveform blob format and batched decoding.

A ``Waveform.data_blob`` is a float32 ``(sample_count, 2)`` array of
``(timestamp, value)`` pairs, written by ``serialize_waveform`` and read back by
``deserialize_waveform_blob``.  This module depends only on the ORM models and
the parser data model, so the database layer (import, day aggregation) can
encode and decode blobs without importing the analysis or service packages.
"""

import asyncio
import logging

from collections.abc import Callable, Sequence
from itertools import batched

import numpy as np

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Waveform
from snore.parsers.unified import WaveformData

logger = logging.getLogger(__name__)

# Blob layout: float32 (timestamp, value) pairs; see deserialize_waveform_blob.
WAVEFORM_BLOB_BYTES_PER_SAMPLE = 2 * np.dtype(np.float32).itemsize

# Sessions fetched per query.  The blobs of a batch are buffered in memory until
# decoded, so this bounds that buffer, at one blob per session per channel.
DECODE_BATCH_SESSIONS = 64


def serialize_waveform(waveform: WaveformData) -> bytes:
    """
    Serialize waveform data to bytes for database storage.

    Stores timestamps and values as float32 numpy arrays.
    No compression - SQLite and filesystem handle that efficiently.

    Args:
        waveform: WaveformData object

    Returns:
        Serialized bytes
    """
    if isinstance(waveform.timestamps, list):
        timestamps = np.array(waveform.timestamps, dtype=np.float32)
    else:
        timestamps = waveform.timestamps.astype(np.float32)

    if isinstance(waveform.values, list):
        values = np.array(waveform.values, dtype=np.float32)
    else:
        values = waveform.values.astype(np.float32)

    data = np.column_stack([timestamps, values])
    return data.tobytes()


def deserialize_waveform_blob(
    blob_data: bytes, sample_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deserialize waveform blob data into timestamps and values arrays.

    This reverses the serialization performed by serialize_waveform(). The
    blob format is:
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
        expected_bytes = sample_count * WAVEFORM_BLOB_BYTES_PER_SAMPLE
        if len(blob_data) != expected_bytes:
            raise ValueError(
                f"Blob size mismatch: expected {expected_bytes} bytes "
                f"({sample_count} samples × {WAVEFORM_BLOB_BYTES_PER_SAMPLE} bytes), "
                f"got {len(blob_data)}"
            )

        if sample_count == 0:
            raise ValueError("Blob has no samples")

        data = np.frombuffer(blob_data, dtype=np.float32).reshape((sample_count, 2))

        timestamps = data[:, 0]  # Seconds from session start
        values = data[:, 1]  # Signal values

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Deserialized {sample_count} samples: "
                f"time range [{timestamps[0]:.2f}, {timestamps[-1]:.2f}]s, "
                f"value range [{values.min():.2f}, {values.max():.2f}]"
            )

        return timestamps, values

    except Exception as e:
        raise ValueError(f"Invalid waveform blob data: {e}") from e


def _decode_rows[T](
    rows: Sequence[tuple[int, str, bytes | None, int | None]],
    reduce: Callable[[str, np.ndarray], T],
    values: dict[str, dict[int, T]],
    corrupt: dict[str, set[int]],
) -> None:
    """Decode and reduce fetched blobs into ``values`` and ``corrupt``.

    Runs on a worker thread, so ``rows`` holds plain values, never ORM objects.
    """
    for session_id, channel, blob, sample_count in rows:
        if not blob or not sample_count:
            continue
        try:
            _, samples = deserialize_waveform_blob(blob, sample_count)
        except ValueError as e:
            logger.warning(
                "Skipping corrupt %s waveform of session %s: %s", channel, session_id, e
            )
            corrupt[channel].add(session_id)
            continue
        values[channel][session_id] = reduce(channel, samples)


async def decode_channel_blobs[T](
    db: AsyncSession,
    session_ids: Sequence[int],
    channels: Sequence[str],
    reduce: Callable[[str, np.ndarray], T],
) -> tuple[dict[str, dict[int, T]], dict[str, set[int]]]:
    """``(values, corrupt)`` of ``channels`` for ``session_ids``.

    ``values`` is channel -> session_id -> ``reduce(channel, samples)``, the raw
    samples including sentinels; ``corrupt`` is channel -> ids of sessions whose
    blob could not be decoded, each logged once as a warning.  Every requested
    channel is a key of both, and a session without a usable row for a channel
    is absent from its ``values`` map.

    Blobs are fetched in batches and decoded off the event loop.  There is no
    ownership check here: callers pass ids already scoped to the profile.
    """
    values: dict[str, dict[int, T]] = {channel: {} for channel in channels}
    corrupt: dict[str, set[int]] = {channel: set() for channel in channels}
    for batch in batched(session_ids, DECODE_BATCH_SESSIONS, strict=False):
        result = await db.execute(
            select(
                Waveform.session_id,
                Waveform.waveform_type,
                Waveform.data_blob,
                Waveform.sample_count,
            ).where(
                Waveform.session_id.in_(batch),
                Waveform.waveform_type.in_(channels),
            )
        )
        await asyncio.to_thread(
            _decode_rows,
            [(sid, channel, blob, count) for sid, channel, blob, count in result],
            reduce,
            values,
            corrupt,
        )
    return values, corrupt
