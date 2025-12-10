"""Update checking and application utilities."""

import json
import logging
import subprocess
import urllib.request

from dataclasses import dataclass

from .installer import (
    GITHUB_REPO,
    GITHUB_REPO_URL,
    PACKAGE_NAME,
    UV_NOT_FOUND_ERROR,
    get_tool_source,
    is_command_available,
)
from .version import get_package_version, is_newer

logger = logging.getLogger(__name__)


@dataclass
class UpdateInfo:
    """Information about available updates."""

    has_update: bool
    current_version: str
    latest_version: str
    source: str


def check_pypi_updates(
    package_name: str, current_version: str, timeout: int = 10
) -> UpdateInfo:
    """Check PyPI for newer version.

    Args:
        package_name: Package name on PyPI
        current_version: Currently installed version
        timeout: Request timeout in seconds (default: 10)

    Returns:
        UpdateInfo with update status
    """
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"

        req = urllib.request.Request(url)
        req.add_header("User-Agent", f"{package_name}/{current_version}")

        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode())

        latest_version = data["info"]["version"]
        has_update = is_newer(latest_version, current_version)

        return UpdateInfo(
            has_update=has_update,
            current_version=current_version,
            latest_version=latest_version,
            source="pypi",
        )

    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"PyPI check failed: {e}")
        return UpdateInfo(
            has_update=False,
            current_version=current_version,
            latest_version=current_version,
            source="pypi",
        )


def check_github_updates(
    repo: str, current_version: str, timeout: int = 10
) -> UpdateInfo:
    """Check GitHub tags for newer version.

    Args:
        repo: GitHub repository in format "owner/repo"
        current_version: Currently installed version
        timeout: Request timeout in seconds (default: 10)

    Returns:
        UpdateInfo with update status
    """
    try:
        url = f"https://api.github.com/repos/{repo}/tags"

        req = urllib.request.Request(url)
        req.add_header("User-Agent", f"snore/{current_version}")

        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode())

        if not data or len(data) == 0:
            return UpdateInfo(
                has_update=False,
                current_version=current_version,
                latest_version=current_version,
                source="github",
            )

        latest_tag = data[0]["name"]
        latest_version = latest_tag.lstrip("v")

        has_update = is_newer(latest_version, current_version)

        return UpdateInfo(
            has_update=has_update,
            current_version=current_version,
            latest_version=latest_version,
            source="github",
        )

    except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.debug(f"GitHub check failed: {e}")
        return UpdateInfo(
            has_update=False,
            current_version=current_version,
            latest_version=current_version,
            source="github",
        )


def check_tool_updates(timeout: int = 10) -> UpdateInfo | None:
    """Check for updates - auto-detect PyPI vs GitHub source.

    Args:
        timeout: Request timeout in seconds (default: 10)

    Returns:
        UpdateInfo if tool is installed and update check succeeds, None otherwise
    """
    try:
        current = get_package_version(PACKAGE_NAME)
    except Exception:
        return None

    source = get_tool_source(PACKAGE_NAME)

    if source == "github":
        return check_github_updates(GITHUB_REPO, current, timeout)
    else:
        return check_pypi_updates(PACKAGE_NAME, current, timeout)


def perform_update(force: bool = False) -> tuple[bool, str, bool]:
    """Upgrade SNORE from correct source.

    Args:
        force: Force reinstall even if already up to date

    Returns:
        Tuple of (success, message, was_upgraded)
        - success: Whether command succeeded
        - message: Human-readable status message
        - was_upgraded: True if package was actually upgraded
    """
    if not is_command_available("uv"):
        return False, UV_NOT_FOUND_ERROR, False

    source = get_tool_source(PACKAGE_NAME)

    if source == "github":
        cmd = ["uv", "tool", "install", "--force", "--reinstall", GITHUB_REPO_URL]
    elif source == "local":
        cmd = ["uv", "tool", "install", PACKAGE_NAME, "--force"]
    else:
        if force:
            cmd = ["uv", "tool", "install", PACKAGE_NAME, "--force"]
        else:
            cmd = ["uv", "tool", "upgrade", PACKAGE_NAME]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            output = result.stdout + result.stderr

            if (
                "Upgraded" in output
                or "Successfully installed" in output
                or "Installed" in output
            ):
                was_upgraded = True
            elif "Nothing to upgrade" in output or "already" in output.lower():
                was_upgraded = False
            else:
                was_upgraded = True

            return True, "Upgrade successful", was_upgraded

        error_msg = result.stderr.strip()
        if not error_msg:
            error_msg = "Upgrade failed with no error message"

        return False, error_msg, False

    except subprocess.TimeoutExpired:
        return False, "Upgrade timed out after 60 seconds", False
    except Exception as e:
        return False, f"Unexpected error: {e}", False
