"""Colour for terminal output.

Colour only repeats what the words already say: ALERT is bold red, OK is cyan, WARMING UP is yellow,
column names are bold and example rows are dim. The 16 standard colours are used so the output
follows the reader's own terminal theme and stays legible on light and dark backgrounds.

Colour is on when output goes to a terminal. NO_COLOR turns it off, FORCE_COLOR turns it on,
and `--color always|never` on the command line overrides both.
"""
import os
import re
import sys

STYLES = {"alert": "1;31", "ok": "36", "warming_up": "33", "bold": "1", "dim": "2"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def enabled(mode="auto", stream=None):
    """Whether to colour output. mode is 'auto', 'always' or 'never'."""
    if mode == "always":
        return True
    if mode == "never":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = sys.stdout if stream is None else stream
    return bool(getattr(stream, "isatty", lambda: False)()) and os.environ.get("TERM") != "dumb"


def paint(text, style, on=True):
    """Wrap text in the escape codes for a style name from STYLES. Returns text unchanged when off."""
    return f"\x1b[{STYLES[style]}m{text}\x1b[0m" if on else str(text)


def strip(text):
    """Remove colour codes."""
    return ANSI.sub("", text)
