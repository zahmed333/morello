"""Draws a terminal session as an SVG in the brand colours.

    python terminal.py session.ansi out.svg

The input is plain text with the colour codes Morello prints (`morello --color always ...`).
A line starting with "$ " is drawn as a command. Text is converted to outlines of JetBrains Mono,
each glyph defined once and reused, so the image needs no fonts and stays small.
"""

import os
import re
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
GROUND, BAR, INK, DIM, ALERT, OK, WARM = (
    "#24090D",
    "#391217",
    "#EEF2FA",
    "#B7C5E0",
    "#EE9AA2",
    "#8DA5CA",
    "#E6C98A",
)
SIZE, LINE, PAD = 12.5, 20.0, 26.0
CODES = {"1;31": ("b", ALERT), "36": ("r", OK), "33": ("r", WARM), "1": ("b", INK), "2": ("r", DIM)}


def spans(line):
    """Split one line into (text, weight, colour) runs."""
    out, weight, colour = [], "r", INK
    for part in re.split(r"(\x1b\[[0-9;]*m)", line):
        m = re.fullmatch(r"\x1b\[([0-9;]*)m", part)
        if m:
            weight, colour = CODES.get(m.group(1), ("r", INK))
        elif part:
            out.append((part, weight, colour))
    return out


def render(text, title="morello"):
    fonts = {
        w: TTFont(f"{HERE}/fonts/JetBrainsMono-{n}.ttf")
        for w, n in (("r", "Regular"), ("b", "Bold"))
    }
    upm = fonts["r"]["head"].unitsPerEm
    k = SIZE / upm
    adv = fonts["r"]["hmtx"]["a"][0] * k
    defs, body, used = [], [], {}
    lines = text.rstrip("\n").split("\n")
    cols = max(len(re.sub(r"\x1b\[[0-9;]*m", "", ln)) for ln in lines)
    w = round(PAD * 2 + cols * adv)
    top = 44.0
    h = round(top + len(lines) * LINE + PAD - 4)
    for i, ln in enumerate(lines):
        x, y = PAD, top + (i + 0.72) * LINE
        runs = spans(ln)
        if ln.startswith("$ "):
            runs = [("$ ", "r", DIM), (ln[2:], "b", INK)]
        for run, weight, colour in runs:
            uses = []
            for ch in run:
                if ch != " ":
                    tt = fonts[weight]
                    name = tt.getBestCmap().get(ord(ch))
                    if name is None:
                        raise SystemExit(f"JetBrains Mono has no glyph for {ch!r}")
                    gid = used.get((weight, name))
                    if gid is None:
                        gid = used[(weight, name)] = f"{weight}{len(used)}"
                        pen = SVGPathPen(
                            tt.getGlyphSet(), ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
                        )
                        tt.getGlyphSet()[name].draw(pen)
                        defs.append(f'<path id="{gid}" d="{pen.getCommands()}"/>')
                    uses.append(f'<use href="#{gid}" x="{x / k:.0f}"/>')
                x += adv
            if uses:
                body.append(
                    f'<g fill="{colour}" transform="translate(0 {y:.1f}) scale({k:.5f} -{k:.5f})">{"".join(uses)}</g>'
                )
    dots = "".join(
        f'<circle cx="{PAD + 16 * i:g}" cy="18" r="5" fill="{c}"/>'
        for i, c in enumerate((ALERT, WARM, OK))
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
        f'aria-label="Terminal session: {title}"><defs>{"".join(defs)}</defs>'
        f'<rect width="{w}" height="{h}" rx="12" fill="{GROUND}"/>'
        f'<path d="M0 12a12 12 0 0 1 12-12h{w - 24}a12 12 0 0 1 12 12v24H0Z" fill="{BAR}"/>{dots}{"".join(body)}</svg>\n'
    )


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    svg = render(Path(src).read_text())
    Path(dst).write_text(svg)
    try:
        import cairosvg

        cairosvg.svg2png(
            bytestring=svg.encode(), write_to=os.path.splitext(dst)[0] + ".png", scale=2
        )
    except ImportError:
        pass
