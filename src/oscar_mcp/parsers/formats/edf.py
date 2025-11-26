"""
EDF/EDF+ File Format Reader

Generic reader for European Data Format (EDF) and EDF+ files.
This is a standard format used by ResMed and some other medical devices.

EDF+ adds support for:
- Discontinuous recordings
- Time-stamped annotations
- Multiple data records per physical record

This module provides a high-level, easy-to-use interface over pyedflib.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging

import pyedflib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EDFHeader:
    """EDF file header information."""

    version: str
    patient_info: str
    recording_info: str
    start_datetime: datetime
    num_data_records: int
    record_duration: float  # seconds
    num_signals: int
    is_edf_plus: bool = False


@dataclass
class EDFSignalInfo:
    """Information about a single EDF signal/channel."""

    label: str  # Signal name
    transducer: str  # Transducer type
    physical_dimension: str  # Units (e.g., "cmH2O", "L/min")
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    prefiltering: str
    samples_per_record: int
    signal_index: int  # Index in EDF file

    # Computed values
    gain: float = 0.0
    offset: float = 0.0

    def __post_init__(self):
        """Calculate gain and offset for digital->physical conversion."""
        digital_range = self.digital_max - self.digital_min
        physical_range = self.physical_max - self.physical_min

        if digital_range > 0:
            self.gain = physical_range / digital_range
            self.offset = self.physical_min - (self.digital_min * self.gain)
        else:
            self.gain = 1.0
            self.offset = 0.0

    def digital_to_physical(self, digital_value: int) -> float:
        """Convert a digital value to physical units."""
        return (digital_value * self.gain) + self.offset


@dataclass
class EDFAnnotation:
    """An EDF+ annotation (event marker with optional duration)."""

    onset_time: float  # Seconds from recording start
    duration: Optional[float]  # Duration in seconds (if specified)
    annotations: List[str]  # List of annotation strings

    def to_datetime(self, recording_start: datetime) -> datetime:
        """Convert onset time to absolute datetime."""
        return recording_start + timedelta(seconds=self.onset_time)


class EDFReader:
    """
    High-level EDF/EDF+ file reader.

    Usage:
        with EDFReader("data.edf") as edf:
            header = edf.get_header()
            signals = edf.get_signal_info()

            # Read specific signal
            data = edf.read_signal("Flow Rate")

            # Read annotations
            annotations = edf.read_annotations()
    """

    def __init__(self, file_path: Path):
        """
        Initialize EDF reader.

        Args:
            file_path: Path to EDF or EDF+ file
        """
        self.file_path = Path(file_path)
        self._edf_file: Optional[pyedflib.EdfReader] = None
        self._header: Optional[EDFHeader] = None
        self._signals: Optional[Dict[str, EDFSignalInfo]] = None
        self._annotations: Optional[List[EDFAnnotation]] = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self):
        """Open the EDF file."""
        if self._edf_file is not None:
            return  # Already open

        if not self.file_path.exists():
            raise FileNotFoundError(f"EDF file not found: {self.file_path}")

        try:
            self._edf_file = pyedflib.EdfReader(str(self.file_path))
            logger.info(f"Opened EDF file: {self.file_path.name}")
        except Exception as e:
            error_msg = str(e).lower()

            # Check if this is a discontinuous file error
            if "discontinuous" in error_msg:
                # Verify it's actually discontinuous
                if is_discontinuous_edf(self.file_path):
                    raise ValueError(
                        f"Failed to open EDF file: {self.file_path.name} is a discontinuous "
                        f"EDF+ file (EDF+D format). This typically occurs when the CPAP mask "
                        f"was removed during recording. pyedflib cannot read discontinuous files."
                    ) from e

            # Check if this is a data records validation error (corrupted/truncated file)
            if (
                "number of datarecords" in error_msg
                or "not edf(+) or bdf(+) compliant" in error_msg
            ):
                raise ValueError(
                    f"Failed to open EDF file: {self.file_path.name}: the file is not EDF(+) "
                    f"or BDF(+) compliant (Number of Datarecords). This usually indicates a "
                    f"corrupted or truncated file where the actual data doesn't match the "
                    f"expected size from the header. The file may have been incompletely "
                    f"written (e.g., SD card removed during recording or device powered off). "
                    f"OSCAR may handle this file differently. Error: {e}"
                ) from e

            raise ValueError(f"Failed to open EDF file: {e}") from e

    def close(self):
        """Close the EDF file."""
        if self._edf_file is not None:
            self._edf_file.close()
            self._edf_file = None
            logger.debug(f"Closed EDF file: {self.file_path.name}")

    def get_header(self) -> EDFHeader:
        """
        Read EDF file header.

        Returns:
            EDFHeader with file metadata
        """
        if self._header is not None:
            return self._header

        if self._edf_file is None:
            self.open()

        # Read header fields
        start_datetime = self._edf_file.getStartdatetime()  # type: ignore[union-attr]

        # Get version from header, default to "0" if not present
        header_dict = self._edf_file.getHeader()  # type: ignore[union-attr]
        version = header_dict.get("version", "0")

        self._header = EDFHeader(
            version=version,
            patient_info=self._edf_file.getPatientName(),  # type: ignore[union-attr]
            recording_info=self._edf_file.getRecordingAdditional(),  # type: ignore[union-attr]
            start_datetime=start_datetime,
            num_data_records=self._edf_file.datarecords_in_file,  # type: ignore[union-attr]
            record_duration=self._edf_file.datarecord_duration,  # type: ignore[union-attr]
            num_signals=self._edf_file.signals_in_file,  # type: ignore[union-attr]
            is_edf_plus=self._edf_file.filetype == 1,  # type: ignore[union-attr]
        )

        return self._header

    def get_signal_info(self) -> Dict[str, EDFSignalInfo]:
        """
        Get information about all signals in the file.

        Returns:
            Dictionary mapping signal labels to EDFSignalInfo
        """
        if self._signals is not None:
            return self._signals

        if self._edf_file is None:
            self.open()

        self._signals = {}

        for i in range(self._edf_file.signals_in_file):  # type: ignore[union-attr]
            label = self._edf_file.getLabel(i).strip()  # type: ignore[union-attr]

            # Skip annotation channels (EDF+ specific)
            if label.startswith("EDF Annotations"):
                continue

            signal_info = EDFSignalInfo(
                label=label,
                transducer=self._edf_file.getTransducer(i).strip(),  # type: ignore[union-attr]
                physical_dimension=self._edf_file.getPhysicalDimension(i).strip(),  # type: ignore[union-attr]
                physical_min=self._edf_file.getPhysicalMinimum(i),  # type: ignore[union-attr]
                physical_max=self._edf_file.getPhysicalMaximum(i),  # type: ignore[union-attr]
                digital_min=self._edf_file.getDigitalMinimum(i),  # type: ignore[union-attr]
                digital_max=self._edf_file.getDigitalMaximum(i),  # type: ignore[union-attr]
                prefiltering=self._edf_file.getPrefilter(i).strip(),  # type: ignore[union-attr]
                samples_per_record=self._edf_file.samples_in_datarecord(i),  # type: ignore[union-attr]
                signal_index=i,
            )

            self._signals[label] = signal_info

        logger.info(f"Found {len(self._signals)} signals in {self.file_path.name}")
        return self._signals

    def list_signal_labels(self) -> List[str]:
        """
        Get list of all signal labels.

        Returns:
            List of signal names
        """
        signals = self.get_signal_info()
        return list(signals.keys())

    def has_signal(self, label: str) -> bool:
        """
        Check if a signal exists in the file.

        Args:
            label: Signal label to check

        Returns:
            True if signal exists
        """
        signals = self.get_signal_info()
        return label in signals

    def read_signal(
        self,
        label: str,
        start_sample: Optional[int] = None,
        num_samples: Optional[int] = None,
        physical_units: bool = True,
    ) -> Tuple[np.ndarray, EDFSignalInfo]:
        """
        Read data for a specific signal.

        Args:
            label: Signal label to read
            start_sample: Starting sample index (default: 0)
            num_samples: Number of samples to read (default: all)
            physical_units: If True, return physical values; if False, return digital values

        Returns:
            Tuple of (data array, signal info)

        Raises:
            KeyError: If signal not found

        Note:
            When reading ResMed EDF files, you may see harmless warnings like
            "read 0, less than X requested!!!" from pyedflib's underlying C library.
            These occur when the library encounters EDF+ annotation channels mixed
            with data channels. They do not indicate errors and can be safely ignored.
            This behavior matches OSCAR's EDF parsing (warnings are just hidden in GUI).
        """
        if self._edf_file is None:
            self.open()

        signals = self.get_signal_info()
        if label not in signals:
            raise KeyError(f"Signal '{label}' not found. Available: {list(signals.keys())}")

        signal_info = signals[label]
        signal_index = signal_info.signal_index

        # Read entire signal at once (pyedflib's readSignal returns all samples by default)
        data = self._edf_file.readSignal(signal_index, digital=not physical_units)  # type: ignore[union-attr]

        # Apply slicing if start_sample or num_samples specified
        if start_sample is not None or num_samples is not None:
            start = start_sample if start_sample is not None else 0
            end = (start + num_samples) if num_samples is not None else None
            data = data[start:end]

        logger.debug(f"Read {len(data)} samples from signal '{label}'")
        return data, signal_info

    def read_all_signals(
        self, physical_units: bool = True
    ) -> Dict[str, Tuple[np.ndarray, EDFSignalInfo]]:
        """
        Read data for all signals.

        Args:
            physical_units: If True, return physical values; if False, return digital values

        Returns:
            Dictionary mapping signal labels to (data, info) tuples
        """
        signals = self.get_signal_info()
        result = {}

        for label in signals:
            data, info = self.read_signal(label, physical_units=physical_units)
            result[label] = (data, info)

        return result

    def read_annotations(self) -> List[EDFAnnotation]:
        """
        Read EDF+ annotations (if present).

        Returns:
            List of EDFAnnotation objects
        """
        if self._annotations is not None:
            return self._annotations

        if self._edf_file is None:
            self.open()

        self._annotations = []

        # Check if this is EDF+ with annotations
        header = self.get_header()
        if not header.is_edf_plus:
            logger.debug("File is not EDF+, no annotations available")
            return self._annotations

        try:
            # Read annotations using pyedflib
            # Returns tuple of (onset_times, durations, texts) where each is an array
            annotations_tuple = self._edf_file.readAnnotations()  # type: ignore[union-attr]

            # pyedflib returns (onsets_array, durations_array, texts_array)
            if len(annotations_tuple) == 3:
                onsets, durations, texts = annotations_tuple

                # Zip them together and convert to our format
                for onset, duration, text in zip(onsets, durations, texts):
                    # text is a single string per annotation
                    annotation = EDFAnnotation(
                        onset_time=float(onset),
                        duration=float(duration) if duration > 0 else None,
                        annotations=[text] if text else [],
                    )
                    self._annotations.append(annotation)

                logger.info(f"Read {len(self._annotations)} annotations from {self.file_path.name}")
            else:
                logger.warning(
                    f"Unexpected annotation format from pyedflib: {type(annotations_tuple)}"
                )

        except Exception as e:
            logger.warning(f"Failed to read annotations: {e}")
            # Return empty list on error
            return self._annotations

        return self._annotations

    def get_sample_rate(self, label: str) -> float:
        """
        Get sample rate for a specific signal in Hz.

        Args:
            label: Signal label

        Returns:
            Sample rate in samples per second
        """
        signals = self.get_signal_info()
        if label not in signals:
            raise KeyError(f"Signal '{label}' not found")

        signal_info = signals[label]
        header = self.get_header()

        # Samples per record / record duration = samples per second
        return signal_info.samples_per_record / header.record_duration

    def get_duration(self) -> float:
        """
        Get total recording duration in seconds.

        Returns:
            Duration in seconds
        """
        header = self.get_header()
        return header.num_data_records * header.record_duration

    def get_timestamps(self, label: str, data: np.ndarray | None = None) -> np.ndarray:
        """
        Generate timestamp array for a signal.

        Args:
            label: Signal label
            data: Optional actual signal data to determine length.
                  If provided, uses len(data) instead of header calculation.
                  This handles cases where actual data length differs from
                  header metadata (e.g., discontinuous recordings).

        Returns:
            NumPy array of timestamps (seconds from start)
        """
        signals = self.get_signal_info()
        if label not in signals:
            raise KeyError(f"Signal '{label}' not found")

        signal_info = signals[label]
        header = self.get_header()

        # Calculate total samples
        if data is not None:
            # Use actual data length when provided
            total_samples = len(data)
        else:
            # Fall back to header calculation
            total_samples = signal_info.samples_per_record * header.num_data_records

        # Calculate sample interval
        sample_rate = self.get_sample_rate(label)
        sample_interval = 1.0 / sample_rate

        # Generate timestamps
        timestamps = np.arange(total_samples) * sample_interval

        return timestamps

    def __repr__(self) -> str:
        """Developer representation."""
        return f"<EDFReader file='{self.file_path.name}'>"

    def __str__(self) -> str:
        """Human-readable representation."""
        if self._header is None:
            return f"EDFReader: {self.file_path.name} (not opened)"

        header = self._header
        return (
            f"EDFReader: {self.file_path.name}\n"
            f"  Type: {'EDF+' if header.is_edf_plus else 'EDF'}\n"
            f"  Start: {header.start_datetime}\n"
            f"  Duration: {self.get_duration():.1f} seconds\n"
            f"  Signals: {header.num_signals}"
        )

    def is_discontinuous(self) -> bool:
        """
        Check if this EDF file is discontinuous (EDF+D format).

        Returns:
            True if file is discontinuous (EDF+D), False otherwise
        """
        return is_discontinuous_edf(self.file_path)


def is_discontinuous_edf(file_path: Path) -> bool:
    """
    Check if an EDF file is discontinuous (EDF+D format) by reading the header.

    Discontinuous EDF+ files occur when there are gaps in recording, such as
    when a CPAP user temporarily removes their mask during the night.

    Args:
        file_path: Path to the EDF file

    Returns:
        True if file is EDF+D (discontinuous), False for EDF+C (continuous) or regular EDF

    Note:
        This function reads the file header directly without using pyedflib,
        avoiding the error that occurs when pyedflib tries to open a discontinuous file.
    """
    if not file_path.exists():
        return False

    try:
        with open(file_path, "rb") as f:
            # EDF header is always 256 bytes
            header = f.read(256)
            if len(header) < 256:
                return False

            # Reserved field is at bytes 192-235 (44 bytes)
            # In EDF+, this field contains "EDF+C" for continuous or "EDF+D" for discontinuous
            reserved = header[192:236].decode("ascii", errors="ignore").strip()

            # Check for EDF+D marker
            return "EDF+D" in reserved

    except Exception as e:
        logger.debug(f"Could not check if {file_path} is discontinuous: {e}")
        return False


def get_edf_record_count(file_path: Path) -> int:
    """
    Get the number of data records from an EDF file header without opening with pyedflib.

    This allows us to check if a file has zero records (device on but not used)
    before attempting to open it with pyedflib, which rejects such files.

    OSCAR accepts files with 0 records (see edfparser.cpp:182-191 where the
    validation check is commented out), so we need to handle them gracefully.

    Args:
        file_path: Path to the EDF file

    Returns:
        Number of data records in the file, or -1 if cannot read header

    Note:
        Files with 0 records occur when device is turned on briefly but not used
        (e.g., daytime test, immediate shutdown). They have valid headers but no data.
    """
    if not file_path.exists():
        return -1

    try:
        with open(file_path, "rb") as f:
            # EDF header is always 256 bytes
            header = f.read(256)
            if len(header) < 256:
                return -1

            # Number of data records is at bytes 236-244 (8 ASCII digits)
            record_count_str = header[236:244].decode("ascii", errors="ignore").strip()

            try:
                return int(record_count_str)
            except ValueError:
                logger.warning(
                    f"Could not parse record count from {file_path}: '{record_count_str}'"
                )
                return -1

    except Exception as e:
        logger.debug(f"Could not read header from {file_path}: {e}")
        return -1


def parse_edf_annotations_raw(file_path: Path) -> List[EDFAnnotation]:
    """
    Parse EDF+ annotations directly from raw bytes (OSCAR-style implementation).

    This function reads EDF+ annotation records directly from the file using
    the EDF+ annotation format delimiters, similar to OSCAR's C++ implementation.
    Works for both continuous (EDF+C) and discontinuous (EDF+D) files.

    Args:
        file_path: Path to the EDF+ file

    Returns:
        List of EDFAnnotation objects extracted from the file

    Format:
        Annotations use special delimiter bytes:
        - \x14 (20): Separator between fields
        - \x15 (21): Duration marker
        - \x00 (0): End of annotation / padding

        Structure: +offset\x15duration\x14Annotation Text\x14\x00
        Example: +120.5\x1512.5\x14Obstructive apnea\x14\x00
    """

    annotations: List[EDFAnnotation] = []

    try:
        with open(file_path, "rb") as f:
            # Read EDF header (256 bytes)
            header_bytes = f.read(256)
            if len(header_bytes) < 256:
                raise ValueError("File too small to be valid EDF")

            # Parse header fields
            num_data_records = int(header_bytes[236:244].decode("ascii", errors="ignore").strip())
            num_signals = int(header_bytes[252:256].decode("ascii", errors="ignore").strip())

            # Read signal headers (256 bytes per signal)
            signal_labels = []
            samples_per_record = []

            for i in range(num_signals):
                label_bytes = f.read(16)
                signal_labels.append(label_bytes.decode("ascii", errors="ignore").strip())

            # Skip transducer, dimension, min/max fields (304 bytes per signal)
            f.seek(256 + num_signals * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80), 0)

            # Read samples per record for each signal
            for i in range(num_signals):
                samples_bytes = f.read(8)
                samples_per_record.append(
                    int(samples_bytes.decode("ascii", errors="ignore").strip())
                )

            # Skip reserved field (32 bytes per signal)
            f.seek(256 + num_signals * 256, 0)

            # Find the "EDF Annotations" signal
            anno_signal_idx = None
            for idx, label in enumerate(signal_labels):
                if "annotations" in label.lower():
                    anno_signal_idx = idx
                    break

            if anno_signal_idx is None:
                logger.debug(f"No annotation signal found in {file_path.name}")
                return annotations

            # Calculate data record size
            record_size = sum(samples_per_record[i] * 2 for i in range(num_signals))
            anno_samples = samples_per_record[anno_signal_idx]
            anno_size = anno_samples * 2  # 2 bytes per sample

            # Calculate offset to annotation data within each record
            anno_offset = sum(samples_per_record[i] * 2 for i in range(anno_signal_idx))

            # Read each data record and extract annotations
            for rec_num in range(num_data_records):
                # Seek to this record's annotation data
                record_start = 256 + num_signals * 256 + rec_num * record_size
                f.seek(record_start + anno_offset, 0)

                # Read annotation bytes
                anno_bytes = f.read(anno_size)

                # Parse annotations from this record
                record_annos = _parse_annotation_bytes(anno_bytes)
                annotations.extend(record_annos)

        logger.info(
            f"Parsed {len(annotations)} annotations from {file_path.name} using direct parsing"
        )
        return annotations

    except Exception as e:
        logger.error(f"Failed to parse annotations from {file_path.name}: {e}")
        return []


def _parse_annotation_bytes(data: bytes) -> List[EDFAnnotation]:
    """
    Parse annotation bytes using EDF+ delimiter format.

    Format: +offset\x15duration\x14Text\x14\x00
    - Offset: Required, starts with + or -, seconds from recording start
    - Duration: Optional, follows \x15 marker, in seconds
    - Text: One or more annotation texts separated by \x14
    - End: \x00 marks end of annotations
    """
    annotations = []
    pos = 0
    data_len = len(data)

    # Delimiter bytes
    ANNO_SEP = 20  # \x14
    ANNO_DUR = 21  # \x15
    ANNO_END = 0  # \x00

    while pos < data_len:
        # Skip null padding
        if data[pos] == ANNO_END:
            pos += 1
            continue

        # Check for onset time (starts with + or -)
        if data[pos] not in (ord("+"), ord("-")):
            pos += 1
            continue

        # Extract onset time
        onset_str = b""
        while pos < data_len and data[pos] not in (ANNO_SEP, ANNO_DUR, ANNO_END):
            onset_str += bytes([data[pos]])
            pos += 1

        if not onset_str:
            break

        try:
            onset_time = float(onset_str.decode("ascii", errors="ignore"))
        except ValueError:
            pos += 1
            continue

        # Check for optional duration
        duration = 0.0
        if pos < data_len and data[pos] == ANNO_DUR:
            pos += 1  # Skip duration marker
            dur_str = b""
            while pos < data_len and data[pos] not in (ANNO_SEP, ANNO_END):
                dur_str += bytes([data[pos]])
                pos += 1

            if dur_str:
                try:
                    duration = float(dur_str.decode("ascii", errors="ignore"))
                except ValueError:
                    pass

        # Extract annotation texts
        anno_texts = []
        while pos < data_len:
            if data[pos] == ANNO_SEP:
                pos += 1
                # Read text until next separator or end
                text = b""
                while pos < data_len and data[pos] not in (ANNO_SEP, ANNO_END):
                    text += bytes([data[pos]])
                    pos += 1

                if text:
                    decoded_text = text.decode("utf-8", errors="ignore").strip()
                    if decoded_text:
                        anno_texts.append(decoded_text)
            elif data[pos] == ANNO_END:
                pos += 1
                break
            else:
                pos += 1

        # Create annotation if we have text
        if anno_texts:
            annotation = EDFAnnotation(
                onset_time=onset_time,
                duration=duration if duration > 0 else None,
                annotations=anno_texts,
            )
            annotations.append(annotation)

    return annotations


def read_annotations_from_discontinuous(file_path: Path) -> List[EDFAnnotation]:
    """
    Read annotations from a discontinuous EDF+ file using MNE.

    This function uses MNE-Python as a fallback for reading discontinuous
    EDF+ files (EDF+D), which pyedflib cannot handle.

    Args:
        file_path: Path to the discontinuous EDF+ file

    Returns:
        List of EDFAnnotation objects extracted from the file

    Note:
        MNE is designed for EEG/MEG data but works for any EDF+ file.
        It handles discontinuous recordings by tracking actual timestamps.
    """
    try:
        import mne
        import warnings

        # Suppress MNE's verbose output
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            mne.set_log_level("ERROR")

            # Read the EDF+ file with MNE
            raw = mne.io.read_raw_edf(
                str(file_path),
                preload=False,  # Don't load data, just annotations
                stim_channel=None,  # Don't search for stimulus channel
                verbose=False,
            )

            # Extract annotations
            annotations = []
            for idx in range(len(raw.annotations)):
                onset = raw.annotations.onset[idx]
                duration = raw.annotations.duration[idx]
                description = raw.annotations.description[idx]

                # Convert to our EDFAnnotation format
                # MNE gives onset in seconds from file start
                annotation = EDFAnnotation(
                    onset_time=float(onset),
                    duration=float(duration) if duration else 0.0,
                    annotations=[description] if description else [],
                )
                annotations.append(annotation)

            logger.info(f"Read {len(annotations)} annotations from discontinuous file using MNE")
            return annotations

    except ImportError:
        logger.error("MNE library not installed - cannot read discontinuous EDF+ files")
        raise ValueError(
            "MNE library is required to read discontinuous EDF+ files. "
            "Install with: pip install mne"
        )
    except Exception as e:
        logger.error(f"Failed to read discontinuous file with MNE: {e}")
        raise ValueError(f"Failed to read discontinuous EDF+ file: {e}") from e


class EDFDiscontinuousReader:
    """
    Reader for discontinuous EDF+ (EDF+D) files using direct annotation parsing.

    This class provides a fallback for reading discontinuous EDF+ files,
    which occur when there are gaps in recording (e.g., CPAP mask removal).
    pyedflib cannot handle these files, so we parse annotations directly from raw bytes.
    """

    def __init__(self, file_path: Path):
        """Initialize the discontinuous reader."""
        self.file_path = file_path
        self._annotations = None
        self._header = None
        self._mne_raw = None  # Cache MNE raw object for signal reading

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self):
        """Open the discontinuous EDF+ file using direct parsing."""
        if self._annotations is not None:
            return  # Already open

        try:
            # Use our direct annotation parser
            self._annotations = parse_edf_annotations_raw(self.file_path)
            logger.info(f"Opened discontinuous EDF file: {self.file_path.name}")
        except Exception as e:
            raise ValueError(f"Failed to open discontinuous EDF+ file: {e}") from e

    def close(self):
        """Close the file."""
        self._annotations = None
        self._header = None
        if self._mne_raw is not None:
            self._mne_raw.close()
            self._mne_raw = None

    def read_annotations(self) -> List[EDFAnnotation]:
        """
        Read all annotations from the discontinuous file.

        Returns:
            List of EDFAnnotation objects
        """
        if self._annotations is None:
            raise ValueError("File not opened. Use 'with' statement or call open() first.")

        return self._annotations

    def get_header(self) -> EDFHeader:
        """
        Get header information from the discontinuous file.

        Returns:
            EDFHeader with file metadata

        Note:
            For discontinuous files, we only provide minimal header info.
            Full header parsing can be added if needed.
        """
        if self._annotations is None:
            raise ValueError("File not opened")

        if self._header is None:
            # Read basic header info directly
            try:
                with open(self.file_path, "rb") as f:
                    header_bytes = f.read(256)

                    # Parse start date/time (bytes 168-183)
                    date_str = header_bytes[168:176].decode("ascii", errors="ignore").strip()
                    time_str = header_bytes[176:184].decode("ascii", errors="ignore").strip()

                    # Parse date: dd.mm.yy format
                    day = int(date_str[0:2])
                    month = int(date_str[3:5])
                    year = int(date_str[6:8])
                    if year < 85:
                        year += 2000
                    else:
                        year += 1900

                    # Parse time: hh.mm.ss format
                    hour = int(time_str[0:2])
                    minute = int(time_str[3:5])
                    second = int(time_str[6:8])

                    start_datetime = datetime(year, month, day, hour, minute, second)

                    # Parse other header fields
                    num_records = int(
                        header_bytes[236:244].decode("ascii", errors="ignore").strip()
                    )
                    record_duration = float(
                        header_bytes[244:252].decode("ascii", errors="ignore").strip()
                    )
                    num_signals = int(
                        header_bytes[252:256].decode("ascii", errors="ignore").strip()
                    )

                    self._header = EDFHeader(
                        version="0",
                        patient_info="",
                        recording_info="",
                        start_datetime=start_datetime,
                        num_data_records=num_records,
                        record_duration=record_duration,
                        num_signals=num_signals,
                        is_edf_plus=True,
                    )
            except Exception as e:
                logger.warning(f"Could not parse header from {self.file_path.name}: {e}")
                # Return minimal header
                self._header = EDFHeader(
                    version="0",
                    patient_info="",
                    recording_info="",
                    start_datetime=datetime.now(),
                    num_data_records=0,
                    record_duration=1.0,
                    num_signals=0,
                    is_edf_plus=True,
                )

        return self._header

    def _ensure_mne_raw(self):
        """Lazy-load MNE raw object for signal reading."""
        if self._mne_raw is None:
            # Import MNE locally to avoid module-level interference with pyedflib
            import mne
            import os

            try:
                # Suppress both stdout and stderr at C-level to silence libsndfile warnings
                # These warnings are harmless but pollute the output with "read 0, less than X requested!!!"
                # Python-level redirection doesn't work because C libraries write directly to file descriptors
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                old_stdout_fd = os.dup(1)  # Save original stdout (fd=1)
                old_stderr_fd = os.dup(2)  # Save original stderr (fd=2)
                os.dup2(devnull_fd, 1)  # Redirect stdout to /dev/null
                os.dup2(devnull_fd, 2)  # Redirect stderr to /dev/null

                try:
                    self._mne_raw = mne.io.read_raw_edf(
                        str(self.file_path), preload=True, verbose=False
                    )
                finally:
                    # Restore stdout and stderr
                    os.dup2(old_stdout_fd, 1)
                    os.dup2(old_stderr_fd, 2)
                    os.close(old_stdout_fd)
                    os.close(old_stderr_fd)
                    os.close(devnull_fd)

                logger.debug(f"Successfully loaded {self.file_path.name} with MNE")
            except Exception as e:
                logger.error(
                    f"Failed to read discontinuous file {self.file_path.name} with MNE: {e}"
                )
                raise ValueError(f"Failed to read discontinuous file with MNE: {e}") from e

    def list_signal_labels(self) -> List[str]:
        """
        Get list of signal labels in the file.

        Returns:
            List of signal names
        """
        if self._annotations is None:
            raise ValueError("File not opened")

        self._ensure_mne_raw()
        return self._mne_raw.ch_names

    def read_signal(self, signal_label: str) -> Tuple[np.ndarray, EDFSignalInfo]:
        """
        Read signal data from the discontinuous file.

        Args:
            signal_label: Name of the signal to read

        Returns:
            Tuple of (data array, signal info) - matches EDFReader API
        """
        if self._annotations is None:
            raise ValueError("File not opened")

        self._ensure_mne_raw()

        # Find the channel index
        try:
            ch_idx = self._mne_raw.ch_names.index(signal_label)
        except ValueError:
            raise KeyError(f"Signal '{signal_label}' not found in file")

        # Get the data for this channel
        data = self._mne_raw.get_data(picks=[ch_idx])[0]  # Shape: (n_samples,)

        # Get channel information from MNE
        ch_info = self._mne_raw.info["chs"][ch_idx]
        sfreq = self._mne_raw.info["sfreq"]

        # Create EDFSignalInfo to match EDFReader API
        # MNE doesn't provide all EDF fields, so use reasonable defaults
        signal_info = EDFSignalInfo(
            label=signal_label,
            transducer="",  # Not available in MNE
            physical_dimension=ch_info.get("unit_name", ""),
            physical_min=float(np.min(data)) if len(data) > 0 else 0.0,
            physical_max=float(np.max(data)) if len(data) > 0 else 0.0,
            digital_min=-32768,  # Standard EDF values
            digital_max=32767,
            prefiltering="",  # Not available in MNE
            samples_per_record=int(sfreq),  # Approximate
            signal_index=ch_idx,
        )

        return data, signal_info

    def get_sample_rate(self, signal_label: str) -> float:
        """
        Get the sample rate for a specific signal.

        Args:
            signal_label: Name of the signal

        Returns:
            Sample rate in Hz
        """
        if self._annotations is None:
            raise ValueError("File not opened")

        self._ensure_mne_raw()

        # MNE typically returns the same sample rate for all channels
        # (or resamples to a common rate)
        return float(self._mne_raw.info["sfreq"])

    def __repr__(self) -> str:
        """Developer representation."""
        return f"<EDFDiscontinuousReader file='{self.file_path.name}'>"
