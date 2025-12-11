import pytest


def pytest_collection_modifyitems(items):
    """Apply integration marker to all tests in this directory."""
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
