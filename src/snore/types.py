"""Core SNORE type definitions."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ShellConfig:
    """Configuration for a supported shell."""

    config_files: list[str]  # Relative to home, in priority order

    def get_config_candidates(self) -> list[Path]:
        """Get existing config file paths for this shell."""
        home = Path.home()
        return [home / cf for cf in self.config_files if (home / cf).exists()]
