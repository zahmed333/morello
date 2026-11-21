"""Colour for terminal output.

Colour only repeats what the words already say: ALERT is bold red, OK is cyan, WARMING UP is
yellow, column names are bold and example rows are dim. The 16 standard colours are used, so
the output follows the reader's own terminal theme on light and dark backgrounds.

Colour is on when output goes to a terminal. NO_COLOR turns it off, FORCE_COLOR turns it on,
and `--color always|never` on the command line overrides both.
"""

from __future__ import annotations

import os
import re
import sys
from typing import TextIO

STYLES = {"alert": "1;31", "ok": "36", "warming_up": "33", "bold": "1", "dim": "2"}
ANSI_CODE = re.compile(r"\x1b\[[0-9;]*m")


def enabled(mode: str = "auto", stream: TextIO | None = None) -> bool:
    """Whether to colour output. mode is "auto", "always" or "never"."""
    if mode == "always":
        return True
    if mode == "never":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = sys.stdout if stream is None else stream
    is_terminal = bool(getattr(stream, "isatty", lambda: False)())
    return is_terminal and os.environ.get("TERM") != "dumb"


def paint(text: str, style: str, on: bool = True) -> str:
    """Wrap text in the codes of a style from STYLES. Returns the text unchanged when off."""
    return f"\x1b[{STYLES[style]}m{text}\x1b[0m" if on else str(text)


def strip(text: str) -> str:
    """Remove colour codes."""
    return ANSI_CODE.sub("", text)
