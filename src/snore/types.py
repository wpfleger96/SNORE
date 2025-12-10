"""Core SNORE type definitions."""

from pathlib import Path

from pydantic import BaseModel, Field


class ShellConfig(BaseModel):
    """Configuration for a supported shell."""

    config_files: list[str] = Field(
        description="Config files relative to home, in priority order"
    )

    def get_config_candidates(self) -> list[Path]:
        """Get existing config file paths for this shell."""
        home = Path.home()
        return [home / cf for cf in self.config_files if (home / cf).exists()]
