"""ResMed parser resolves each data root from its argument, not the last detect().

The parser registry shares one parser instance, and an import detects a parent
folder once and then parses every discovered root with that same instance. Each
parse must read the root it was given.
"""

import json
import logging
import shutil

from pathlib import Path

import pytest

from sqlalchemy import func, select

from snore.database import models
from snore.database.session import init_database, session_scope
from snore.parsers.base import ParserError
from snore.parsers.resmed_edf import ResmedEDFParser
from snore.services.import_service import ImportService
from tests.integration.conftest import _make_profile

pytestmark = pytest.mark.parser

SERIALS = {"dev_a": "SERIAL-AAA", "dev_b": "SERIAL-BBB"}


def _copy_device_root(fixture_root: Path, dest: Path, serial: str) -> None:
    """Copy the ResMed fixture root to dest with its serial rewritten."""
    shutil.copytree(fixture_root, dest)
    id_file = dest / "Identification.json"
    data = json.loads(id_file.read_text())
    data["FlowGenerator"]["IdentificationProfiles"]["Product"]["SerialNumber"] = serial
    id_file.write_text(json.dumps(data))


async def _new_profile_id(db_path: Path) -> int:
    await init_database(str(db_path))
    async with session_scope() as db:
        profile = await _make_profile(db)
        return int(profile.id)


async def _sessions_per_serial() -> dict[str, int]:
    async with session_scope() as db:
        rows = (
            await db.execute(
                select(models.Device.serial_number, func.count(models.Session.id))
                .join(models.Session, models.Session.device_id == models.Device.id)
                .group_by(models.Device.serial_number)
            )
        ).all()
    return {serial: count for serial, count in rows}


@pytest.fixture
def two_device_parent(tmp_path: Path, resmed_fixture_path: Path) -> Path:
    """A parent folder holding two ResMed device roots with distinct serials."""
    parent = tmp_path / "parent"
    for name, serial in SERIALS.items():
        _copy_device_root(resmed_fixture_path, parent / name, serial)
    return parent


@pytest.fixture
def detected_parser(
    resmed_parser: ResmedEDFParser, two_device_parent: Path
) -> ResmedEDFParser:
    """One parser instance that has already detected both roots via the parent."""
    detection = resmed_parser.detect(two_device_parent)
    assert detection.detected
    assert len(detection.metadata["all_roots"]) == 2
    return resmed_parser


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


def test_get_device_info_on_two_device_parent_warns_and_uses_one_root(
    resmed_parser: ResmedEDFParser,
    two_device_parent: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="snore.parsers.resmed_edf"):
        info = resmed_parser.get_device_info(two_device_parent)

    assert info.serial_number in SERIALS.values()
    used = next(n for n, s in SERIALS.items() if s == info.serial_number)
    skipped = next(n for n in SERIALS if n != used)
    assert str(two_device_parent / skipped) in caplog.text
    assert "skipping" in caplog.text


def test_parser_on_empty_directory_reports_unknown_device_and_raises(
    resmed_parser: ResmedEDFParser, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    assert resmed_parser.get_device_info(empty).serial_number == "Unknown"
    with pytest.raises(ParserError):
        list(resmed_parser.parse_sessions(empty, parallel=False))


async def test_import_sources_two_device_parent_with_backup_stores_each_device_sessions(
    two_device_parent: Path, tmp_path: Path
) -> None:
    profile_id = await _new_profile_id(tmp_path / "snore.db")
    backup_root = tmp_path / "raw" / str(profile_id)
    svc = ImportService()
    sources = [
        s
        for s in svc.detect_sources(two_device_parent)
        if s.parser_name == "resmed_edf"
    ]
    assert {s.device_serial for s in sources} == set(SERIALS.values())

    result = await svc.import_sources(
        sources,
        profile_id=profile_id,
        backup=True,
        backup_root=backup_root,
        parallel=False,
    )

    assert result.total_imported == 2
    assert await _sessions_per_serial() == {serial: 1 for serial in SERIALS.values()}
    for serial in SERIALS.values():
        backed_up = json.loads(
            (backup_root / serial / "Identification.json").read_text()
        )
        product = backed_up["FlowGenerator"]["IdentificationProfiles"]["Product"]
        assert product["SerialNumber"] == serial


async def test_import_sources_with_backup_ignores_nights_only_in_archive(
    resmed_fixture_path: Path, tmp_path: Path
) -> None:
    """A night already in the backup archive but absent from the card is not parsed."""
    profile_id = await _new_profile_id(tmp_path / "snore.db")
    backup_root = tmp_path / "raw" / str(profile_id)
    serial = SERIALS["dev_a"]
    _copy_device_root(resmed_fixture_path, backup_root / serial, serial)
    card = tmp_path / "card"
    _copy_device_root(resmed_fixture_path, card, serial)
    shutil.rmtree(card / "DATALOG")
    (card / "DATALOG").mkdir()
    svc = ImportService()
    sources = [s for s in svc.detect_sources(card) if s.parser_name == "resmed_edf"]

    result = await svc.import_sources(
        sources,
        profile_id=profile_id,
        backup=True,
        backup_root=backup_root,
        parallel=False,
    )

    assert result.total_imported == 0
    assert await _sessions_per_serial() == {}
