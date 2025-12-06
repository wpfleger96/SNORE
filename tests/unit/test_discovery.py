"""Tests for data root discovery functionality."""

from pathlib import Path

import pytest

from snore.parsers.discovery import DataRoot, DataRootFinder


class TestDataRootFinder:
    """Tests for the DataRootFinder class."""

    @pytest.fixture
    def finder(self):
        """Create a DataRootFinder instance."""
        return DataRootFinder()

    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator function."""

        def validator(path: Path) -> bool:
            return (path / "marker.txt").exists()

        return validator

    @pytest.fixture
    def mock_extractor(self):
        """Create a mock metadata extractor."""

        def extractor(path: Path) -> DataRoot:
            return DataRoot(
                path=path,
                structure_type="raw_sd",
                profile_name=None,
                device_serial="12345",
                confidence=0.9,
            )

        return extractor

    def test_find_root_from_exact_path(
        self, finder, mock_validator, mock_extractor, tmp_path
    ):
        """Test finding data root when given the exact root path."""
        root = tmp_path / "data"
        root.mkdir()
        (root / "marker.txt").touch()

        results = finder.find_data_roots(root, mock_validator, mock_extractor)

        assert len(results) == 1
        assert results[0].path == root
        assert results[0].device_serial == "12345"

    def test_find_root_from_child_path(
        self, finder, mock_validator, mock_extractor, tmp_path
    ):
        """Test finding data root when given a child directory path."""
        root = tmp_path / "data"
        root.mkdir()
        (root / "marker.txt").touch()

        child = root / "subdirectory"
        child.mkdir()

        results = finder.find_data_roots(child, mock_validator, mock_extractor)

        assert len(results) == 1
        assert results[0].path == root

    def test_find_root_from_parent_path(
        self, finder, mock_validator, mock_extractor, tmp_path
    ):
        """Test finding data root when given a parent directory path."""
        parent = tmp_path / "parent"
        parent.mkdir()

        root = parent / "data"
        root.mkdir()
        (root / "marker.txt").touch()

        results = finder.find_data_roots(
            parent, mock_validator, mock_extractor, max_levels_down=2
        )

        assert len(results) == 1
        assert results[0].path == root

    def test_multiple_roots_sorted_by_confidence(self, finder, tmp_path):
        """Test that multiple roots are sorted by confidence score."""
        root1 = tmp_path / "data1"
        root1.mkdir()
        (root1 / "marker.txt").touch()

        root2 = tmp_path / "data2"
        root2.mkdir()
        (root2 / "marker.txt").touch()

        def validator(path: Path) -> bool:
            return (path / "marker.txt").exists()

        def extractor(path: Path) -> DataRoot:
            confidence = 0.95 if path.name == "data2" else 0.85
            return DataRoot(
                path=path,
                structure_type="raw_sd",
                profile_name=None,
                device_serial="12345",
                confidence=confidence,
            )

        results = finder.find_data_roots(
            tmp_path, validator, extractor, max_levels_down=2
        )

        assert len(results) == 2
        assert results[0].path == root2
        assert results[0].confidence == 0.95
        assert results[1].path == root1
        assert results[1].confidence == 0.85

    def test_no_valid_root_found(
        self, finder, mock_validator, mock_extractor, tmp_path
    ):
        """Test that empty list is returned when no valid root is found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = finder.find_data_roots(empty_dir, mock_validator, mock_extractor)

        assert len(results) == 0

    def test_handles_permission_errors(self, finder, tmp_path):
        """Test that permission errors during traversal are handled gracefully."""
        root = tmp_path / "data"
        root.mkdir()
        (root / "marker.txt").touch()

        def validator(path: Path) -> bool:
            return (path / "marker.txt").exists()

        def extractor(path: Path) -> DataRoot:
            if "protected" in str(path):
                raise PermissionError("Access denied")
            return DataRoot(
                path=path,
                structure_type="raw_sd",
                profile_name=None,
                device_serial="12345",
                confidence=0.9,
            )

        results = finder.find_data_roots(root, validator, extractor)

        assert len(results) == 1
        assert results[0].path == root
