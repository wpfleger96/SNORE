"""
QDataStream Compatibility Layer

Implements minimal Qt QDataStream binary format reading for OSCAR file parsing.
Supports Qt 4.6 format with little-endian byte order.
"""

import struct

from enum import IntEnum
from typing import Any, BinaryIO, cast


class QVariantType(IntEnum):
    """Qt QVariant type codes."""

    Invalid = 0
    Bool = 1
    Int = 2
    UInt = 3
    LongLong = 4
    ULongLong = 5
    Double = 6
    String = 10
    ByteArray = 12
    Date = 14
    Time = 15
    DateTime = 16
    UserType = 127  # User-defined types start at 127


class QDataStreamReader:
    """
    Minimal QDataStream reader for Qt 4.6 binary format.

    Reads binary data in Qt's QDataStream format with little-endian byte order.
    """

    def __init__(self, stream: BinaryIO):
        """
        Initialize QDataStream reader.

        Args:
            stream: Binary file-like object to read from
        """
        self.stream = stream
        self.byte_order = "<"  # Little-endian

    def read_bytes(self, count: int) -> bytes:
        """Read raw bytes from stream."""
        data = self.stream.read(count)
        if len(data) != count:
            raise EOFError(f"Expected {count} bytes, got {len(data)}")
        return data

    def read_bool(self) -> bool:
        """Read boolean (1 byte)."""
        return cast(bool, struct.unpack(f"{self.byte_order}?", self.read_bytes(1))[0])

    def read_int8(self) -> int:
        """Read signed 8-bit integer."""
        return cast(int, struct.unpack(f"{self.byte_order}b", self.read_bytes(1))[0])

    def read_uint8(self) -> int:
        """Read unsigned 8-bit integer."""
        return cast(int, struct.unpack(f"{self.byte_order}B", self.read_bytes(1))[0])

    def read_int16(self) -> int:
        """Read signed 16-bit integer."""
        return cast(int, struct.unpack(f"{self.byte_order}h", self.read_bytes(2))[0])

    def read_uint16(self) -> int:
        """Read unsigned 16-bit integer."""
        return cast(int, struct.unpack(f"{self.byte_order}H", self.read_bytes(2))[0])

    def read_int32(self) -> int:
        """Read signed 32-bit integer."""
        return cast(int, struct.unpack(f"{self.byte_order}i", self.read_bytes(4))[0])

    def read_uint32(self) -> int:
        """Read unsigned 32-bit integer."""
        return cast(int, struct.unpack(f"{self.byte_order}I", self.read_bytes(4))[0])

    def read_int64(self) -> int:
        """Read signed 64-bit integer."""
        return cast(int, struct.unpack(f"{self.byte_order}q", self.read_bytes(8))[0])

    def read_uint64(self) -> int:
        """Read unsigned 64-bit integer."""
        return cast(int, struct.unpack(f"{self.byte_order}Q", self.read_bytes(8))[0])

    def read_float(self) -> float:
        """Read 32-bit float."""
        return cast(float, struct.unpack(f"{self.byte_order}f", self.read_bytes(4))[0])

    def read_double(self) -> float:
        """Read 64-bit double."""
        return cast(float, struct.unpack(f"{self.byte_order}d", self.read_bytes(8))[0])

    def read_qstring(self) -> str | None:
        """
        Read QString (Qt string format).

        Format: 4-byte length (in bytes), then UTF-16 encoded string data.
        Length of 0xFFFFFFFF indicates null string.

        Returns:
            Decoded string or None for null string
        """
        length = self.read_uint32()

        if length == 0xFFFFFFFF:
            return None

        if length == 0:
            return ""

        data = self.read_bytes(length)

        # Decode UTF-16 (little-endian)
        try:
            return data.decode("utf-16-le")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="replace")

    def read_qvariant(self) -> Any:
        """
        Read QVariant (polymorphic Qt type).

        Format: 4-byte type code, then data based on type.

        Returns:
            Python value corresponding to the Qt type
        """
        type_code = self.read_uint32()
        is_null = self.read_bool()

        if is_null:
            return None

        if type_code == QVariantType.Bool:
            return self.read_bool()
        elif type_code == QVariantType.Int:
            return self.read_int32()
        elif type_code == QVariantType.UInt:
            return self.read_uint32()
        elif type_code == QVariantType.LongLong:
            return self.read_int64()
        elif type_code == QVariantType.ULongLong:
            return self.read_uint64()
        elif type_code == QVariantType.Double:
            return self.read_double()
        elif type_code == QVariantType.String:
            return self.read_qstring()
        elif type_code == QVariantType.ByteArray:
            length = self.read_uint32()
            if length == 0xFFFFFFFF:
                return None
            return self.read_bytes(length)
        else:
            import warnings

            warnings.warn(
                f"Skipping unsupported QVariant type: {type_code}", stacklevel=2
            )
            return None

    def skip_qhash_uint32_qvariant(self) -> None:
        """
        Skip QHash<quint32, QVariant> without parsing values.

        This is useful for skipping settings that contain unknown QVariant types.
        """
        count = self.read_uint32()

        for _ in range(count):
            self.read_uint32()
            type_code = self.read_uint32()
            is_null = self.read_bool()

            if is_null:
                continue

            if type_code == QVariantType.Bool:
                self.read_bool()
            elif type_code == QVariantType.Int:
                self.read_int32()
            elif type_code == QVariantType.UInt:
                self.read_uint32()
            elif type_code == QVariantType.LongLong:
                self.read_int64()
            elif type_code == QVariantType.ULongLong:
                self.read_uint64()
            elif type_code == QVariantType.Double:
                self.read_double()
            elif type_code == QVariantType.String:
                self.read_qstring()
            elif type_code == QVariantType.ByteArray:
                length = self.read_uint32()
                if length != 0xFFFFFFFF:
                    self.skip_bytes(length)

    def read_qhash_uint32_qvariant(self) -> dict[int, Any]:
        """
        Read QHash<quint32, QVariant>.

        Format: 4-byte count, then pairs of (key, value).
        Used for OSCAR channel settings.

        Returns:
            Dictionary mapping channel IDs to values
        """
        count = self.read_uint32()
        result = {}

        for _ in range(count):
            key = self.read_uint32()
            value = self.read_qvariant()
            result[key] = value

        return result

    def read_qhash_uint32_float(self) -> dict[int, float]:
        """
        Read QHash<quint32, EventDataType> where EventDataType is float.

        Format: 4-byte count, then pairs of (key, float value).
        Used for OSCAR statistics.

        Returns:
            Dictionary mapping channel IDs to float values
        """
        count = self.read_uint32()
        result = {}

        for _ in range(count):
            key = self.read_uint32()
            value = self.read_float()
            result[key] = value

        return result

    def read_qhash_uint32_double(self) -> dict[int, float]:
        """
        Read QHash<quint32, double>.

        Format: 4-byte count, then pairs of (key, double value).

        Returns:
            Dictionary mapping channel IDs to double values
        """
        count = self.read_uint32()
        result = {}

        for _ in range(count):
            key = self.read_uint32()
            value = self.read_double()
            result[key] = value

        return result

    def read_qhash_uint32_uint64(self) -> dict[int, int]:
        """
        Read QHash<quint32, quint64>.

        Format: 4-byte count, then pairs of (key, uint64 value).
        Used for timestamps per channel.

        Returns:
            Dictionary mapping channel IDs to uint64 values
        """
        count = self.read_uint32()
        result = {}

        for _ in range(count):
            key = self.read_uint32()
            value = self.read_uint64()
            result[key] = value

        return result

    def read_qhash_nested(self) -> dict[int, dict[int, int]]:
        """
        Read nested QHash<quint32, QHash<EventStoreType, EventStoreType>>.

        Used for value summaries (histogram of values per channel).

        Returns:
            Nested dictionary structure
        """
        outer_count = self.read_uint32()
        result = {}

        for _ in range(outer_count):
            outer_key = self.read_uint32()
            inner_count = self.read_uint32()
            inner_dict = {}

            for _ in range(inner_count):
                inner_key = self.read_int16()  # EventStoreType
                inner_value = self.read_int16()  # EventStoreType
                inner_dict[inner_key] = inner_value

            result[outer_key] = inner_dict

        return result

    def read_qhash_nested_time(self) -> dict[int, dict[int, int]]:
        """
        Read nested QHash<quint32, QHash<EventStoreType, quint32>>.

        Used for time summaries (time spent at each value per channel).

        Returns:
            Nested dictionary structure
        """
        outer_count = self.read_uint32()
        result = {}

        for _ in range(outer_count):
            outer_key = self.read_uint32()
            inner_count = self.read_uint32()
            inner_dict = {}

            for _ in range(inner_count):
                inner_key = self.read_int16()  # EventStoreType
                inner_value = self.read_uint32()  # quint32 time
                inner_dict[inner_key] = inner_value

            result[outer_key] = inner_dict

        return result

    def read_qlist_uint32(self) -> list[int]:
        """
        Read QList<quint32>.

        Format: 4-byte count, then array of uint32 values.
        Used for available channels list.

        Returns:
            List of channel IDs
        """
        count = self.read_uint32()
        result = []

        for _ in range(count):
            value = self.read_uint32()
            result.append(value)

        return result

    def read_qvector_int16(self) -> list[int]:
        """
        Read QVector<qint16> (EventStoreType array).

        Format: 4-byte count, then array of int16 values.

        Returns:
            List of int16 values
        """
        count = self.read_uint32()

        if count == 0:
            return []

        data = self.read_bytes(count * 2)
        result = list(struct.unpack(f"{self.byte_order}{count}h", data))

        return result

    def read_qvector_uint32(self) -> list[int]:
        """
        Read QVector<quint32> (time delta array).

        Format: 4-byte count, then array of uint32 values.

        Returns:
            List of uint32 values
        """
        count = self.read_uint32()

        if count == 0:
            return []

        data = self.read_bytes(count * 4)
        result = list(struct.unpack(f"{self.byte_order}{count}I", data))

        return result

    def skip_bytes(self, count: int) -> None:
        """Skip specified number of bytes."""
        self.stream.seek(count, 1)

    def tell(self) -> int:
        """Get current position in stream."""
        return self.stream.tell()

    def seek(self, position: int, whence: int = 0) -> None:
        """Seek to position in stream."""
        self.stream.seek(position, whence)
