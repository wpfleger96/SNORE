"""Generic backup orchestrator for raw CPAP data files.

Delegates all file-layout knowledge to the DeviceParser. This service
handles path construction, serial validation, import-from-backup detection,
and error reporting — it never knows about manufacturer-specific files.

Writer lease
------------
``backup_via_parser`` is the single production path to ``parser.backup_raw_data()``.
It acquires the process-singleton shared writer lease for the entire file-copy
operation so that ``snore profile delete`` (which requires the exclusive lease)
is structurally unable to rename or delete the raw tree while a backup is in
progress.  The lease is held shared, so multiple concurrent backups in the same
process share one descriptor and never block each other.
"""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from snore.parsers.base import DeviceParser

from snore.parsers.base import RawFileManifest
from snore.services.writer_lease import get_writer_lease

logger = logging.getLogger(__name__)

_SERIAL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class InvalidSerialError(ValueError):
    """Device serial is unsafe to use as a backup path component."""


@dataclass
class BackupResult:
    """Result of a backup operation."""

    backup_root: Path
    """Device backup root path (e.g., ~/.snore/raw/<profile_id>/<serial>/)."""

    manifest: RawFileManifest | None = None
    """Manifest of files that were backed up. None if backup was skipped."""

    files_copied: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    skipped_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def was_skipped(self) -> bool:
        return self.manifest is None


class BackupService:
    """Generic backup orchestrator. Delegates file operations to parsers.

    The ``backup_root`` should be the profile-scoped directory
    (e.g. ``~/.snore/raw/<profile_id>/``).  The ``DEFAULT_RAW_BACKUP_DIR``
    constant is kept as the fallback for legacy single-user usage; multiuser
    callers must pass the profile-scoped root explicitly.
    """

    def __init__(self, backup_root: Path | None = None) -> None:
        from snore.constants import DEFAULT_RAW_BACKUP_DIR  # noqa: PLC0415

        self.backup_root = backup_root or DEFAULT_RAW_BACKUP_DIR

    def backup_via_parser(
        self,
        parser: DeviceParser,
        source_root: Path,
        device_serial: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> BackupResult:
        """Orchestrate backup for any parser that supports it.

        Acquires the shared writer lease for the entire operation.
        This is the ONLY production path to ``parser.backup_raw_data()``.

        Args:
            parser: The device parser for this data source.
            source_root: Path to the raw data (e.g., SD card root).
            device_serial: Device serial number.
            progress_callback: Optional callback for progress messages.

        Returns:
            BackupResult with the backup root path and manifest.

        Raises:
            ValueError: If the device serial is invalid.
            RuntimeError: If backup fails.
        """
        self._validate_serial(device_serial)
        dest_root = self.get_device_backup_root(device_serial)

        # Detect import-from-backup: source is already inside the backup dir
        try:
            if source_root.resolve().is_relative_to(self.backup_root.resolve()):
                reason = "Source is already within the backup directory"
                logger.debug(f"Skipping backup: {reason}")
                return BackupResult(
                    backup_root=dest_root,
                    skipped_reason=reason,
                )
        except ValueError:
            pass  # is_relative_to raises ValueError on Windows for different drives

        if progress_callback:
            progress_callback(f"Backing up raw files to {dest_root}")

        # Acquire the shared writer lease for the duration of the file copy.
        # The API server holds a lifetime shared lease; re-acquiring on the same
        # fd is idempotent and does not block.
        lease = get_writer_lease()
        with lease.shared():
            manifest = parser.backup_raw_data(
                source_root, dest_root, progress_callback=progress_callback
            )

        logger.debug(
            f"Backup complete: {manifest.files_copied} copied, "
            f"{manifest.files_skipped} skipped, {manifest.files_failed} failed, "
            f"{len(manifest.nights)} nights to {dest_root}"
        )
        if progress_callback:
            parts = [f"{manifest.files_copied} copied"]
            if manifest.files_skipped:
                parts.append(f"{manifest.files_skipped} skipped")
            if manifest.files_failed:
                parts.append(f"{manifest.files_failed} failed")
            progress_callback(
                f"Backed up {', '.join(parts)} ({len(manifest.nights)} nights)"
            )

        return BackupResult(
            backup_root=dest_root,
            manifest=manifest,
            files_copied=manifest.files_copied,
            files_skipped=manifest.files_skipped,
            files_failed=manifest.files_failed,
        )

    def get_device_backup_root(self, device_serial: str) -> Path:
        """Return the backup root path for a device serial."""
        self._validate_serial(device_serial)
        return self.backup_root / device_serial

    def has_backup(self, device_serial: str) -> bool:
        """Check if a backup exists for a device serial."""
        root = self.get_device_backup_root(device_serial)
        return root.is_dir()

    def list_backed_up_devices(self) -> list[str]:
        """Return list of device serial numbers that have backups."""
        if not self.backup_root.is_dir():
            return []
        return sorted(
            d.name
            for d in self.backup_root.iterdir()
            if d.is_dir() and _SERIAL_PATTERN.match(d.name)
        )

    def _validate_serial(self, device_serial: str) -> None:
        """Validate device serial for safe path construction."""
        if not device_serial:
            raise InvalidSerialError("Device serial number is empty")
        if not _SERIAL_PATTERN.match(device_serial):
            raise InvalidSerialError(
                f"Device serial '{device_serial}' contains invalid characters. "
                f"Expected alphanumeric, underscore, or hyphen."
            )
        # Bounds check: resolved path must be under backup_root
        resolved = (self.backup_root / device_serial).resolve()
        if not resolved.is_relative_to(self.backup_root.resolve()):
            raise InvalidSerialError(
                f"Device serial '{device_serial}' resolves to a path "
                f"outside the backup directory"
            )
