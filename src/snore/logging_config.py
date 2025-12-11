"""Centralized logging configuration for SNORE."""

import logging
import logging.config
import os
import sys

from pathlib import Path
from typing import Any

from snore.constants import (
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILE,
)

_logging_configured = False


def get_log_dir() -> Path:
    """
    Get log directory path, creating if needed.

    Returns:
        Path to log directory
    """
    log_dir = DEFAULT_LOG_DIR
    os.makedirs(log_dir, mode=0o700, exist_ok=True)
    return log_dir


def get_log_path() -> Path:
    """
    Get path to the active log file.

    Returns:
        Path to snore.log
    """
    return get_log_dir() / DEFAULT_LOG_FILE


def _get_user_logging_config() -> dict[str, Any]:
    """
    Load logging settings from config file.

    Returns:
        Dictionary with logging settings, or empty dict if not configured
    """
    try:
        from snore.config import load_config

        config = load_config()
        logging_config = config.get("logging", {})

        if isinstance(logging_config, dict):
            return logging_config
        return {}
    except Exception:
        return {}


def _build_logging_config(
    verbose: bool = False,
    console_format: str | None = None,
) -> dict[str, Any]:
    """
    Build the dictConfig configuration dictionary.

    Args:
        verbose: If True, set console to DEBUG level
        console_format: Override console format string

    Returns:
        Dictionary suitable for logging.config.dictConfig()
    """
    user_config = _get_user_logging_config()
    file_enabled = user_config.get("enabled", True)
    file_level = user_config.get("level", "DEBUG").upper()
    max_bytes = user_config.get("max_size_mb", 10) * 1024 * 1024
    backup_count = user_config.get("backup_count", DEFAULT_LOG_BACKUP_COUNT)

    log_file = get_log_path()

    console_fmt = (
        console_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {"format": console_fmt},
            "file": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG" if verbose else "INFO",
                "formatter": "console",
                "stream": "ext://sys.stderr",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console"],
        },
    }

    if file_enabled:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": file_level,
            "formatter": "file",
            "filename": str(log_file),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
        }
        config["root"]["handlers"].append("file")

    return config


def setup_logging(
    *,
    verbose: bool = False,
    console_format: str | None = None,
) -> None:
    """
    Configure logging for SNORE application.

    Uses dictConfig for robust configuration that properly handles
    loggers created before this function is called.

    Args:
        verbose: If True, set console to DEBUG level
        console_format: Override console format string. If None, uses full format.
    """
    global _logging_configured

    if _logging_configured:
        return

    try:
        config = _build_logging_config(verbose=verbose, console_format=console_format)
        logging.config.dictConfig(config)
    except Exception as e:
        sys.stderr.write(f"WARNING: Failed to configure logging: {e}\n")
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format=console_format or "%(levelname)s: %(message)s",
        )

    _logging_configured = True
