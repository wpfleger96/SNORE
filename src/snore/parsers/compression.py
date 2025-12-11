"""Compression and decompression utilities for OSCAR data."""

import gzip
import struct
import zlib


class QtCompressionError(Exception):
    """Exception raised when Qt compression/decompression fails."""

    pass


def decompress_gzip(data: bytes) -> bytes:
    """
    Decompress gzip-compressed data.

    Args:
        data: Compressed data bytes

    Returns:
        Decompressed data bytes
    """
    return gzip.decompress(data)


def compress_gzip(data: bytes) -> bytes:
    """
    Compress data using gzip.

    Args:
        data: Uncompressed data bytes

    Returns:
        Compressed data bytes
    """
    return gzip.compress(data)


def encode_int16_array(values: list[int]) -> bytes:
    """
    Encode a list of integers as int16 array.

    Args:
        values: List of integer values

    Returns:
        Packed binary data
    """
    return struct.pack(f"<{len(values)}h", *values)


def decode_int16_array(data: bytes) -> list[int]:
    """
    Decode int16 array from binary data.

    Args:
        data: Packed binary data

    Returns:
        List of integer values
    """
    count = len(data) // 2
    return list(struct.unpack(f"<{count}h", data))


def encode_delta_times(timestamps: list[int]) -> bytes:
    """
    Encode timestamps using delta encoding.

    Delta encoding stores the difference between consecutive values,
    which provides better compression for monotonically increasing data.

    Args:
        timestamps: List of timestamps (in milliseconds)

    Returns:
        Packed delta-encoded binary data
    """
    if not timestamps:
        return b""

    deltas = [timestamps[0]]

    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        deltas.append(delta)

    return struct.pack(f"<{len(deltas)}I", *deltas)


def decode_delta_times(data: bytes) -> list[int]:
    """
    Decode delta-encoded timestamps.

    Args:
        data: Packed delta-encoded binary data

    Returns:
        List of absolute timestamps (in milliseconds)
    """
    if not data:
        return []

    count = len(data) // 4
    deltas = list(struct.unpack(f"<{count}I", data))

    if not deltas:
        return []

    timestamps = [deltas[0]]

    for i in range(1, len(deltas)):
        timestamps.append(timestamps[-1] + deltas[i])

    return timestamps


def apply_gain_offset(values: list[int], gain: float, offset: float) -> list[float]:
    """
    Apply gain and offset transformation to integer values.

    OSCAR stores values as int16 with gain/offset for compression.
    Actual value = (stored_value * gain) + offset

    Args:
        values: List of stored integer values
        gain: Gain factor
        offset: Offset value

    Returns:
        List of actual floating-point values
    """
    return [(v * gain) + offset for v in values]


def remove_gain_offset(values: list[float], gain: float, offset: float) -> list[int]:
    """
    Remove gain and offset to convert to integer storage values.

    Stored value = (actual_value - offset) / gain

    Args:
        values: List of actual floating-point values
        gain: Gain factor
        offset: Offset value

    Returns:
        List of integer storage values
    """
    return [int((v - offset) / gain) for v in values]


def qUncompress(data: bytes) -> bytes:
    """
    Decompress data compressed with Qt's qCompress().

    Qt's qCompress format:
    - 4 bytes: Uncompressed data size (big-endian uint32)
    - N bytes: zlib-compressed data

    Args:
        data: Compressed data bytes

    Returns:
        Decompressed data bytes

    Raises:
        QtCompressionError: If decompression fails
    """
    if len(data) < 4:
        raise QtCompressionError("Data too short for Qt compressed format")

    uncompressed_size = struct.unpack(">I", data[:4])[0]

    compressed_data = data[4:]

    try:
        decompressed = zlib.decompress(compressed_data)
    except zlib.error as e:
        raise QtCompressionError(f"zlib decompression failed: {e}") from e

    if len(decompressed) != uncompressed_size:
        raise QtCompressionError(
            f"Decompressed size mismatch: expected {uncompressed_size}, got {len(decompressed)}"
        )

    return decompressed


def qCompress(data: bytes, compression_level: int = 6) -> bytes:
    """
    Compress data using Qt's qCompress() format.

    Qt's qCompress format:
    - 4 bytes: Uncompressed data size (big-endian uint32)
    - N bytes: zlib-compressed data

    Args:
        data: Data to compress
        compression_level: zlib compression level (1-9, default 6)

    Returns:
        Compressed data bytes in Qt format

    Raises:
        QtCompressionError: If compression fails
    """
    if compression_level < 1 or compression_level > 9:
        raise ValueError("Compression level must be between 1 and 9")

    try:
        compressed = zlib.compress(data, level=compression_level)
    except zlib.error as e:
        raise QtCompressionError(f"zlib compression failed: {e}") from e

    uncompressed_size = len(data)
    header = struct.pack(">I", uncompressed_size)

    return header + compressed


def calculate_crc16(data: bytes) -> int:
    """
    Calculate CRC16 checksum.

    Uses CRC-16-CCITT polynomial (0x1021).

    Args:
        data: Data to checksum

    Returns:
        CRC16 checksum value
    """
    crc = 0xFFFF
    polynomial = 0x1021

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ polynomial
            else:
                crc = crc << 1
            crc &= 0xFFFF

    return crc


def verify_crc16(data: bytes, expected_crc: int) -> bool:
    """
    Verify CRC16 checksum.

    Args:
        data: Data to verify
        expected_crc: Expected CRC16 value

    Returns:
        True if checksum matches, False otherwise
    """
    calculated_crc = calculate_crc16(data)
    return calculated_crc == expected_crc
