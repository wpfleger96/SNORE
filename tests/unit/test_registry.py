"""Tests for parser registry multi-detection functionality."""

from unittest.mock import Mock

import pytest

from snore.parsers.base import ParserDetectionResult
from snore.parsers.registry import ParserRegistry


class TestMultiParserDetection:
    """Tests for ParserRegistry's detect_all_parsers method."""

    @pytest.fixture
    def registry(self):
        """Create a fresh parser registry."""
        return ParserRegistry()

    @pytest.fixture
    def mock_parser_high_confidence(self):
        """Create a mock parser that detects with high confidence."""
        parser = Mock()
        parser.parser_id = "mock_parser_high"
        parser.manufacturer = "MockVendor"
        parser.detect.return_value = ParserDetectionResult(
            detected=True,
            confidence=0.95,
            message="High confidence match",
            metadata={"data_root": "/path/to/data", "structure_type": "type_a"},
        )
        return parser

    @pytest.fixture
    def mock_parser_low_confidence(self):
        """Create a mock parser that detects with low confidence."""
        parser = Mock()
        parser.parser_id = "mock_parser_low"
        parser.manufacturer = "MockVendor"
        parser.detect.return_value = ParserDetectionResult(
            detected=True,
            confidence=0.70,
            message="Low confidence match",
            metadata={"data_root": "/path/to/data", "structure_type": "type_b"},
        )
        return parser

    @pytest.fixture
    def mock_parser_no_match(self):
        """Create a mock parser that doesn't detect."""
        parser = Mock()
        parser.parser_id = "mock_parser_none"
        parser.manufacturer = "MockVendor"
        parser.detect.return_value = ParserDetectionResult(
            detected=False,
            confidence=0.0,
            message="No match",
        )
        return parser

    def test_detect_all_parsers_single_match(
        self, registry, mock_parser_high_confidence, mock_parser_no_match, tmp_path
    ):
        """Test detect_all_parsers with single matching parser."""
        registry.register(mock_parser_high_confidence)
        registry.register(mock_parser_no_match)

        results = registry.detect_all_parsers(tmp_path)

        assert len(results) == 1
        assert results[0][0].parser_id == "mock_parser_high"
        assert results[0][1].confidence == 0.95

    def test_detect_all_parsers_sorted_results(
        self,
        registry,
        mock_parser_high_confidence,
        mock_parser_low_confidence,
        tmp_path,
    ):
        """Test that results are sorted by confidence (highest first)."""
        registry.register(mock_parser_low_confidence)
        registry.register(mock_parser_high_confidence)

        results = registry.detect_all_parsers(tmp_path)

        assert len(results) == 2
        assert results[0][0].parser_id == "mock_parser_high"
        assert results[0][1].confidence == 0.95
        assert results[1][0].parser_id == "mock_parser_low"
        assert results[1][1].confidence == 0.70

    def test_no_parsers_match(self, registry, mock_parser_no_match, tmp_path):
        """Test that empty list is returned when no parsers match."""
        registry.register(mock_parser_no_match)

        results = registry.detect_all_parsers(tmp_path)

        assert len(results) == 0

    def test_parser_exception_handling(self, registry, tmp_path):
        """Test that parser exceptions are handled and other parsers continue."""
        failing_parser = Mock()
        failing_parser.parser_id = "failing_parser"
        failing_parser.manufacturer = "FailVendor"
        failing_parser.detect.side_effect = Exception("Parser crashed")

        working_parser = Mock()
        working_parser.parser_id = "working_parser"
        working_parser.manufacturer = "GoodVendor"
        working_parser.detect.return_value = ParserDetectionResult(
            detected=True,
            confidence=0.90,
            message="Success",
            metadata={"data_root": "/path/to/data"},
        )

        registry.register(failing_parser)
        registry.register(working_parser)

        results = registry.detect_all_parsers(tmp_path)

        assert len(results) == 1
        assert results[0][0].parser_id == "working_parser"
