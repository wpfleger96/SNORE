"""The agreement rule behind the nightly summary's respiratory values."""

import pytest

from snore.mcp.tools.summary import agreed_value

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([15.0], 15.0),
        ([15.0, 15.0, 15.0], 15.0),
        ([0.0, 0.0], 0.0),
        ([14.0, 16.0], None),
        ([15.0, None], None),
        ([None, 15.0], None),
        ([None, None], None),
        ([], None),
    ],
    ids=[
        "one-session",
        "all-agree",
        "agree-on-zero",
        "differ",
        "later-null",
        "first-null",
        "all-null",
        "no-statistics-rows",
    ],
)
def test_agreed_value_is_the_shared_non_null_value_or_none(
    values: list[float | None], expected: float | None
) -> None:
    assert agreed_value(values) == expected
