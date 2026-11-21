"""Regenerates every Morello brand file from the two font files (Space Grotesk SemiBold, Inter Medium).
Text is converted to outlines, so the SVGs need no fonts installed.
Needs: fonttools, uharfbuzz; cairosvg for the PNG copies."""

import os
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("MORELLO_FONTS", f"{HERE}/fonts")
OUT = os.environ.get("MORELLO_OUT", HERE)
os.makedirs(f"{OUT}/png", exist_ok=True)
WINE, SKY, MIST, PAPER, SKY_DEEP = "#391217", "#8DA5CA", "#DDE6F7", "#F6F8FC", "#4F658B"
TAG = "Know when your data stops making sense."


class Face:
    def __init__(self, name):
        path = f"{SRC}/{name}.ttf"
        self.tt = TTFont(path)
        self.gs = self.tt.getGlyphSet()
        self.upm = self.tt["head"].unitsPerEm
        self.hb = hb.Font(hb.Face(Path(path).read_bytes()))

    def text(self, s, size, x=0.0, y=0.0, tracking=0.0):
        buf = hb.Buffer()
        buf.add_str(s)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf)
        k = size / self.upm
        pen_x, parts = x, []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            name = self.tt.getGlyphName(info.codepoint)
            p = SVGPathPen(self.gs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
            self.gs[name].draw(
                TransformPen(p, (k, 0, 0, -k, pen_x + pos.x_offset * k, y - pos.y_offset * k))
            )
            parts.append(p.getCommands())
            pen_x += pos.x_advance * k + tracking
        return " ".join(parts), pen_x - x - tracking


def g(body, x=0.0, y=0.0, s=1.0):
    return f'<g transform="translate({x:.2f} {y:.2f}) scale({s:.4f})">{body}</g>'


def mark(a, b, leg=23.0):
    """A lowercase m: two arches side by side on three stems. The first arch and the middle stem
    are one colour; the second arch is the odd one. Box: x 0..51, y 0..(15 + leg), stroke 9."""
    y0 = 15.0
    y1 = y0 + leg

    def arch(x, colour):
        return (
            f'<path d="M{x} {y1:g}V{y0:g}a10.5 10.5 0 0 1 21 0V{y1:g}" '
            f'fill="none" stroke="{colour}" stroke-width="9"/>'
        )

    return (
        arch(4.5, a)
        + arch(25.5, b)
        + f'<path d="M25.5 {y1:g}V{y0:g}" fill="none" stroke="{a}" stroke-width="9"/>'
    )


def svg(w, h, body, bg=None, label="Morello"):
    rect = f'<rect width="{w:g}" height="{h:g}" fill="{bg}"/>' if bg else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" width="{w:g}" height="{h:g}" '
        f'role="img" aria-label="{label}">{rect}{body}</svg>\n'
    )


def write(name, s):
    Path(OUT, name).write_text(s)


grotesk, inter = Face("SpaceGrotesk-SemiBold"), Face("Inter-Medium")

# mark alone, centred in a 64 box: light grounds get wine + sky, dark grounds get sky + paper
for tag, a, b in (("wine", WINE, SKY), ("sky", SKY, PAPER)):
    write(f"mark-{tag}.svg", svg(64, 64, g(mark(a, b), 6.5, 13)))
# app icon and favicon: the mark on a solid tile
write(
    "icon-wine.svg",
    svg(
        64,
        64,
        f'<rect width="64" height="64" rx="14" fill="{WINE}"/>'
        + g(mark(SKY, PAPER), 11.6, 16.8, 0.8),
    ),
)
write(
    "icon-sky.svg",
    svg(
        64,
        64,
        f'<rect width="64" height="64" rx="14" fill="{SKY}"/>'
        + g(mark(WINE, PAPER), 11.6, 16.8, 0.8),
    ),
)

# lockup: the mark stands on the wordmark's baseline
d, tw = grotesk.text("morello", 42, 0, 0, tracking=-0.5)
BASE = 49.0
for tag, a, b, ink in (("wine", WINE, SKY, WINE), ("sky", SKY, PAPER, SKY)):
    body = (
        g(mark(a, b), 2, BASE - 38)
        + f'<path fill="{ink}" transform="translate(69 {BASE:g})" d="{d}"/>'
    )
    write(f"logo-{tag}.svg", svg(69 + tw + 4, 64, body))
    write(
        f"wordmark-{tag}.svg",
        svg(tw + 4, 48, f'<path fill="{ink}" transform="translate(2 37)" d="{d}"/>'),
    )


def scene(w, h, big, tag_size, pad):
    """Mist ground, a sky slab on the right. The m stands on the bottom edge of the slab and its
    stems run off the frame: one wine arch, one paper arch."""
    sw = h * 0.78
    sx = w - sw
    k = sw * 0.62 / 51.0
    top = h * 0.36
    big_m = g(mark(WINE, PAPER, leg=(h - top) / k), sx + (sw - 51 * k) / 2, top, k)
    slab = f'<rect x="{sx:.1f}" y="0" width="{sw:.1f}" height="{h}" fill="{SKY}"/>'
    nd, _ = grotesk.text("morello", big, 0, 0, tracking=-big * 0.015)
    td, _ = inter.text(TAG, tag_size, 0, 0)
    words = (
        f'<path fill="{WINE}" transform="translate({pad} {h - pad - tag_size * 2.0:.1f})" d="{nd}"/>'
        f'<path fill="{SKY_DEEP}" transform="translate({pad + big * 0.04:.1f} {h - pad:.1f})" d="{td}"/>'
    )
    return svg(w, h, slab + big_m + words, bg=MIST, label=f"Morello. {TAG}")


write("social-card.svg", scene(1280, 640, 140, 30, 72))
write("banner.svg", scene(1280, 320, 90, 22, 44))

try:
    import cairosvg

    for n, wpx in (
        ("logo-wine", 1200),
        ("logo-sky", 1200),
        ("mark-wine", 512),
        ("mark-sky", 512),
        ("icon-wine", 512),
        ("icon-sky", 512),
        ("banner", 1280),
        ("social-card", 1280),
    ):
        cairosvg.svg2png(url=f"{OUT}/{n}.svg", write_to=f"{OUT}/png/{n}.png", output_width=wpx)
    for sz in (32, 180):
        cairosvg.svg2png(
            url=f"{OUT}/icon-wine.svg", write_to=f"{OUT}/png/favicon-{sz}.png", output_width=sz
        )
except ImportError:
    print("cairosvg not installed: SVGs written, PNG copies skipped")
print(f"built; wordmark width {tw:.1f}")
