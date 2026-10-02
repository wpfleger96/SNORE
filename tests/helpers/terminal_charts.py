"""Assertion helpers for terminal (plotext braille) chart output."""


def has_drawn_braille(text: str) -> bool:
    """Return True if text contains a plotted braille point."""
    # U+2800 is the blank braille cell; any other braille char is a plotted point.
    return any("\u2801" <= ch <= "\u28ff" for ch in text)
