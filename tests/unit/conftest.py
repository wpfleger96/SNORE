import pytest


def pytest_collection_modifyitems(items):
    """Apply unit marker to all tests in this directory."""
    for item in items:
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
