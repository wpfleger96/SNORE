"""Import every module in a fresh interpreter: first-import order exposes cycles (see #364)."""

import pkgutil
import subprocess
import sys

import pytest

import snore


def _raise(name: str) -> None:
    raise ImportError(name)


MODULE_NAMES = [
    module.name
    for module in pkgutil.walk_packages(snore.__path__, "snore.", onerror=_raise)
]


@pytest.mark.parametrize("module_name", MODULE_NAMES, ids=MODULE_NAMES)
def test_first_import_in_fresh_interpreter_succeeds(module_name: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; importlib.import_module(sys.argv[1])",
            module_name,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        pytest.fail(result.stderr, pytrace=False)
