"""Synthetic ResMed STR.edf files for slicing and trim tests."""

from __future__ import annotations

from pathlib import Path


def write_synthetic_str_edf(
    path: Path,
    start_date: str = "01.01.25",
    num_records: int = 30,
    num_signals: int = 2,
) -> None:
    """Create a minimal valid STR.edf for slicing tests.

    Each signal has 1 sample per record (int16), so record_size = num_signals * 2.
    Total file = 256 (header) + num_signals*256 (signal headers) + num_records * record_size.
    """
    header = bytearray(256)
    header[0:8] = b"0       "
    header[168:176] = f"{start_date:<8}".encode("ascii")
    header[176:184] = b"12.00.00"
    header_size = 256 + num_signals * 256
    header[184:192] = f"{header_size:<8}".encode("ascii")
    header[236:244] = f"{num_records:<8}".encode("ascii")
    header[244:252] = b"86400   "
    header[252:256] = f"{num_signals:<4}".encode("ascii")

    sig_headers = bytearray(num_signals * 256)
    # samples_per_record at offset num_signals * 216 within signal headers
    spr_offset = num_signals * 216
    for i in range(num_signals):
        sig_headers[spr_offset + i * 8 : spr_offset + (i + 1) * 8] = b"1       "
    # signal labels (16 bytes each, at the start)
    for i in range(num_signals):
        label = f"Signal{i:<10}".encode("ascii")[:16]
        sig_headers[i * 16 : (i + 1) * 16] = label

    record_size = num_signals * 2
    data = bytearray(num_records * record_size)
    for rec in range(num_records):
        for sig in range(num_signals):
            offset = rec * record_size + sig * 2
            data[offset : offset + 2] = (rec + sig).to_bytes(2, "little", signed=True)

    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(bytes(sig_headers))
        f.write(bytes(data))


def read_str_header(data: bytes) -> tuple[str, int]:
    """Return an STR.edf's ``(start date "dd.mm.yy", record count)``."""
    return data[168:176].decode("ascii").strip(), int(data[236:244].decode("ascii"))
