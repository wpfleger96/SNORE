"""Configuration management for SNORE."""

import logging
import os
import tomllib

from pathlib import Path
from typing import Any

import tomli_w

logger = logging.getLogger(__name__)


def get_config_path() -> Path:
    """
    Get the path to the configuration file.

    Returns:
        Path to ~/.snore/config.toml
    """
    return Path.home() / ".snore" / "config.toml"


def load_config() -> dict[str, Any]:
    """
    Load configuration from TOML file.

    Returns:
        Configuration dictionary. Returns empty dict if file doesn't exist
        or is corrupted.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return {}

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        logger.warning("Treating config as empty. Fix or delete the file to resolve.")
        return {}


def save_config(config: dict[str, Any]) -> None:
    """
    Save configuration to TOML file using atomic write.

    Creates the parent directory if it doesn't exist.
    Uses temp file + rename for atomic operation.

    Args:
        config: Configuration dictionary to save

    Raises:
        PermissionError: If directory cannot be created or file cannot be written
    """
    config_path = get_config_path()

    config_dir = config_path.parent
    try:
        os.makedirs(config_dir, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot create config directory {config_dir}: {e}"
        ) from e

    temp_path = config_path.with_suffix(".toml.tmp")

    try:
        with open(temp_path, "wb") as f:
            tomli_w.dump(config, f)

        os.replace(temp_path, config_path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def get_default_profile() -> str | None:
    """
    Get the default profile username from config.

    Returns:
        Default profile username, or None if not set
    """
    config = load_config()
    default: str | None = config.get("profile", {}).get("default")
    return default


def set_default_profile(username: str) -> None:
    """
    Set the default profile username in config.

    Args:
        username: Profile username to set as default
    """
    config = load_config()

    if "profile" not in config:
        config["profile"] = {}

    config["profile"]["default"] = username
    save_config(config)


def unset_default_profile() -> None:
    """
    Remove the default profile setting from config.

    If this was the only setting in the profile section, removes the section.
    If config becomes empty, deletes the config file.
    """
    config = load_config()

    if "profile" in config and "default" in config["profile"]:
        del config["profile"]["default"]

        if not config["profile"]:
            del config["profile"]

        if not config:
            config_path = get_config_path()
            if config_path.exists():
                config_path.unlink()
        else:
            save_config(config)
