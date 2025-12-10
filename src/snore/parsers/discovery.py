"""Data root discovery for CPAP parsers."""

from collections.abc import Callable
from pathlib import Path

from snore.parsers.types import DataRoot

__all__ = ["DataRootFinder", "DataRoot"]


class DataRootFinder:
    """Generic data root discovery for CPAP parsers."""

    def find_data_roots(
        self,
        path: Path,
        validator_func: Callable[[Path], bool],
        metadata_extractor_func: Callable[[Path], DataRoot | None],
        max_levels_up: int = 5,
        max_levels_down: int = 3,
    ) -> list[DataRoot]:
        """
        Find all valid data roots from any starting path.

        Args:
            path: Starting path to search from
            validator_func: Function that returns True if path is a valid data root
            metadata_extractor_func: Function that extracts metadata from a valid root
            max_levels_up: Maximum parent directories to check
            max_levels_down: Maximum subdirectory levels to search

        Returns:
            List of DataRoot objects, sorted by confidence (highest first)
        """
        roots = []
        seen_paths = set()

        def add_root_if_valid(check_path: Path) -> None:
            normalized = check_path.resolve()
            if normalized in seen_paths:
                return
            seen_paths.add(normalized)

            if validator_func(check_path):
                root = metadata_extractor_func(check_path)
                if root is not None:
                    roots.append(root)

        add_root_if_valid(path)

        current = path
        for _ in range(max_levels_up):
            current = current.parent
            if current == current.parent:
                break
            add_root_if_valid(current)

        try:
            for subpath in path.rglob("*"):
                if subpath.is_dir():
                    depth = len(subpath.relative_to(path).parts)
                    if depth <= max_levels_down:
                        add_root_if_valid(subpath)
        except (PermissionError, OSError):
            pass

        return sorted(roots, key=lambda r: r.confidence, reverse=True)
