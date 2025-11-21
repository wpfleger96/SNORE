"""Tests for ResMed parser smart detection functionality."""

import tempfile
from pathlib import Path

import pytest

from oscar_mcp.parsers.resmed_edf import ResmedEDFParser


class TestResmedSmartDetection:
    """Tests for ResMed parser's smart directory detection."""

    def test_detect_raw_sd_card_structure(self, resmed_fixture_path, resmed_parser):
        """Test detection of raw SD card data structure."""
        result = resmed_parser.detect(resmed_fixture_path)

        assert result.detected is True
        assert result.confidence >= 0.9
        assert result.metadata["structure_type"] == "raw_sd"
        assert result.metadata["data_root"] == str(resmed_fixture_path)
        assert result.metadata["profile_name"] is None

    def test_detect_oscar_profile_structure(self, tmp_path, resmed_parser):
        """Test detection of OSCAR profile directory structure."""
        oscar_path = tmp_path / "Profiles" / "testuser" / "ResMed_12345678" / "Backup"
        oscar_path.mkdir(parents=True)

        (oscar_path / "STR.edf").touch()
        (oscar_path / "DATALOG").mkdir()

        result = resmed_parser.detect(oscar_path)

        assert result.detected is True
        assert result.metadata["structure_type"] == "oscar_profile"
        assert result.metadata["profile_name"] == "testuser"
        assert result.metadata["device_serial"] == "12345678"
        assert result.metadata["data_root"] == str(oscar_path)

    def test_detect_from_deep_subdirectory(self, resmed_fixture_path, resmed_parser):
        """Test detection when pointing to DATALOG subdirectory (the original bug)."""
        datalog_path = resmed_fixture_path / "DATALOG"
        assert datalog_path.exists()

        result = resmed_parser.detect(datalog_path)

        assert result.detected is True
        assert result.metadata["data_root"] == str(resmed_fixture_path)
        assert "parent" in result.message.lower()

    def test_metadata_extraction(self, resmed_fixture_path, resmed_parser):
        """Test that all metadata fields are correctly extracted."""
        result = resmed_parser.detect(resmed_fixture_path)

        assert result.detected is True
        assert "data_root" in result.metadata
        assert "structure_type" in result.metadata
        assert "profile_name" in result.metadata
        assert "device_serial" in result.metadata

        assert result.metadata["structure_type"] in ["raw_sd", "oscar_profile"]

    def test_uses_discovered_root_for_parsing(self, resmed_fixture_path, resmed_parser):
        """Test that parse_sessions uses the discovered data root."""
        datalog_path = resmed_fixture_path / "DATALOG"

        detection = resmed_parser.detect(datalog_path)
        assert detection.detected

        sessions = list(resmed_parser.parse_sessions(datalog_path, limit=1))

        assert len(sessions) > 0
        assert sessions[0].device_info is not None
