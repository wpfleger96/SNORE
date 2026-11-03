"""ResMed parser resolves each data root from its argument, not the last detect().

The parser registry shares one parser instance, and an import detects a parent
folder once and then parses every discovered root (or its raw backup copy) with
that same instance. Each parse must read the root it was given.
"""

import json
import shutil

from pathlib import Path

import pytest

from snore.parsers.registry import parser_registry
from snore.parsers.resmed_edf import ResmedEDFParser
from snore.services.import_service import ImportService

pytestmark = pytest.mark.parser

SERIALS = {"dev_a": "SERIAL-AAA", "dev_b": "SERIAL-BBB"}


def _copy_device_root(fixture_root: Path, dest: Path, serial: str) -> Path:
    """Copy the ResMed fixture root to dest with its serial rewritten."""
    shutil.copytree(fixture_root, dest)
    id_file = dest / "Identification.json"
    data = json.loads(id_file.read_text())
    data["FlowGenerator"]["IdentificationProfiles"]["Product"]["SerialNumber"] = serial
    id_file.write_text(json.dumps(data))
    return dest


@pytest.fixture
def two_device_parent(tmp_path: Path, resmed_fixture_path: Path) -> Path:
    """A parent folder holding two ResMed device roots with distinct serials."""
    parent = tmp_path / "parent"
    for name, serial in SERIALS.items():
        _copy_device_root(resmed_fixture_path, parent / name, serial)
    return parent


@pytest.fixture
def detected_parser(two_device_parent: Path) -> ResmedEDFParser:
    """One parser instance that has already detected both roots via the parent."""
    parser = ResmedEDFParser()
    detection = parser.detect(two_device_parent)
    assert detection.detected
    assert len(detection.metadata["all_roots"]) == 2
    return parser


def test_get_device_info_after_parent_detect_returns_each_roots_serial(
    detected_parser: ResmedEDFParser, two_device_parent: Path
) -> None:
    for name, serial in SERIALS.items():
        info = detected_parser.get_device_info(two_device_parent / name)
        assert info.serial_number == serial


@pytest.mark.parametrize("name", list(SERIALS))
def test_parse_sessions_after_parent_detect_reads_given_root(
    detected_parser: ResmedEDFParser, two_device_parent: Path, name: str
) -> None:
    root = two_device_parent / name

    sessions = list(detected_parser.parse_sessions(root, parallel=False))

    assert sessions
    assert {s.device_info.serial_number for s in sessions} == {SERIALS[name]}
    assert {s.raw_data_path for s in sessions} == {str(root)}


@pytest.mark.parametrize("name", list(SERIALS))
def test_parse_sessions_of_backup_copy_outside_parent_reads_backup(
    detected_parser: ResmedEDFParser,
    two_device_parent: Path,
    tmp_path: Path,
    name: str,
) -> None:
    serial = SERIALS[name]
    backup = tmp_path / "raw" / "1" / serial
    shutil.copytree(two_device_parent / name, backup)

    sessions = list(detected_parser.parse_sessions(backup, parallel=False))

    assert sessions
    assert {s.device_info.serial_number for s in sessions} == {serial}
    assert {s.raw_data_path for s in sessions} == {str(backup)}


def test_import_service_two_device_parent_parses_each_source_as_its_device(
    two_device_parent: Path,
) -> None:
    sources = [
        s
        for s in ImportService().detect_sources(two_device_parent)
        if s.parser_name == "resmed_edf"
    ]
    assert {s.device_serial for s in sources} == set(SERIALS.values())

    parser = next(
        p for p in parser_registry.list_parsers() if p.parser_id == "resmed_edf"
    )
    for source in sources:
        sessions = list(parser.parse_sessions(Path(source.root_path), parallel=False))
        assert sessions
        assert {s.device_info.serial_number for s in sessions} == {source.device_serial}
