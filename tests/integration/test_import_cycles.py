"""Import every module in a fresh interpreter: first-import order exposes cycles (see #364)."""

import pkgutil
import subprocess
import sys

import pytest

import snore

MODULE_NAMES = [
    module.name
    for module in pkgutil.walk_packages(snore.__path__, "snore.")
    if ".migrations" not in module.name
]


@pytest.mark.parametrize("module_name", MODULE_NAMES, ids=MODULE_NAMES)
def test_module_imports_in_fresh_interpreter(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
