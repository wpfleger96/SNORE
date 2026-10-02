"""Guard: the test session must never resolve paths under the real user home.

Fails if tests/conftest.py stops redirecting HOME before ``snore.constants``
is imported, which would let tests open the developer's real ~/.snore database.
"""

import os
import pwd

from pathlib import Path

import pytest

from snore import constants

REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


@pytest.mark.parametrize(
    "name",
    [
        "DEFAULT_DATABASE_PATH",
        "DEFAULT_RAW_BACKUP_DIR",
        "DEFAULT_VACUUM_PENDING_MARKER",
        "DEFAULT_DEPLOY_DEFERRED_MARKER",
        "DEFAULT_UPLOAD_SPOOL_DIR",
        "DEFAULT_LOG_DIR",
    ],
)
def test_snore_paths_resolve_outside_real_home(name: str) -> None:
    path = Path(getattr(constants, name)).resolve()
    assert not path.is_relative_to(REAL_HOME)


def test_home_is_not_real_home() -> None:
    assert not Path.home().resolve().is_relative_to(REAL_HOME)
