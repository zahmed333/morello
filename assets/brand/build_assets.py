"""Regenerates every Morello brand file from the two font files (Sora SemiBold, Inter Medium).
Text is converted to outlines, so the SVGs need no fonts installed.
Needs: fonttools, uharfbuzz; cairosvg for the PNG copies."""
import os
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f"{HERE}/fonts"
WINE, SKY, MIST, PAPER, SKY_DEEP = "#391217", "#8DA5CA", "#DDE6F7", "#F6F8FC", "#4F658B"


class Face:
    def __init__(self, name):
        path = f"{SRC}/{name}.ttf"
        self.tt = TTFont(path); self.gs = self.tt.getGlyphSet(); self.upm = self.tt["head"].unitsPerEm
        self.hb = hb.Font(hb.Face(open(path, "rb").read()))

    def text(self, s, size, x=0.0, y=0.0, tracking=0.0):
        buf = hb.Buffer(); buf.add_str(s); buf.guess_segment_properties(); hb.shape(self.hb, buf)
        k = size / self.upm; pen_x, parts = x, []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            name = self.tt.getGlyphName(info.codepoint)
            p = SVGPathPen(self.gs, ntos=lambda v: ("%.2f" % v).rstrip("0").rstrip("."))
            self.gs[name].draw(TransformPen(p, (k, 0, 0, -k, pen_x + pos.x_offset * k, y - pos.y_offset * k)))
            parts.append(p.getCommands()); pen_x += pos.x_advance * k + tracking
        return " ".join(parts), pen_x - x - tracking


def g(body, x=0.0, y=0.0, s=1.0):
    return f'<g transform="translate({x:.2f} {y:.2f}) scale({s:.4f})">{body}</g>'


def mark_pair(a, b, stem):
    """Two cherries on one stalk. They should match and one does not."""
    return (f'<path d="M36 9C30 18 22 24 19 33M36 9c2 10 6 18 10 25" fill="none" stroke="{stem}" stroke-width="3.6" stroke-linecap="round"/>'
            f'<path d="M36 9c6-6 14-6 19-3-4 6-12 8-19 3Z" fill="{stem}"/>'
            f'<circle cx="18" cy="45" r="13" fill="{a}"/><circle cx="47" cy="46" r="13" fill="{b}"/>')




def svg(w, h, body, bg=None, label="Morello"):
    rect = f'<rect width="{w:g}" height="{h:g}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" width="{w:g}" height="{h:g}" '
            f'role="img" aria-label="{label}">{rect}{body}</svg>\n')


OUT = HERE; os.makedirs(f"{OUT}/png", exist_ok=True)
sora, inter = Face("Sora-SemiBold"), Face("Inter-Medium")
TAG = "Know when your data stops making sense."


def write(name, s):
    open(f"{OUT}/{name}", "w").write(s)


# mark alone: light grounds get wine + sky, dark grounds get sky + paper
for tag, a, b, stem in (("wine", WINE, SKY, WINE), ("sky", SKY, PAPER, SKY)):
    write(f"mark-{tag}.svg", svg(64, 64, g(mark_pair(a, b, stem), -0.5, 1.5)))
# app icon and favicon: the mark on a solid tile
write("icon-wine.svg", svg(64, 64, f'<rect width="64" height="64" rx="14" fill="{WINE}"/>' + g(mark_pair(SKY, PAPER, SKY), 8.6, 9.6, 0.72)))
write("icon-sky.svg", svg(64, 64, f'<rect width="64" height="64" rx="14" fill="{SKY}"/>' + g(mark_pair(WINE, PAPER, WINE), 8.6, 9.6, 0.72)))

# lockup and wordmark
d, tw = sora.text("morello", 40, 0, 0, tracking=-0.6)
for tag, a, b, ink in (("wine", WINE, SKY, WINE), ("sky", SKY, PAPER, SKY)):
    body = g(mark_pair(a, b, ink), 0, 1.5) + f'<path fill="{ink}" transform="translate(76 47.5)" d="{d}"/>'
    write(f"logo-{tag}.svg", svg(76 + tw + 4, 64, body))
    write(f"wordmark-{tag}.svg", svg(tw + 4, 48, f'<path fill="{ink}" transform="translate(2 37)" d="{d}"/>'))


def scene(w, h, big, tag_size, pad):
    """Mist ground, a sky slab on the right. A stalk comes in from the top edge of the slab and
    carries two cherries: one wine, one paper."""
    sw = h * 0.78; sx = w - sw; k = h / 78.0
    slab = f'<rect x="{sx:.1f}" y="0" width="{sw:.1f}" height="{h}" fill="{SKY}"/>'
    pair = mark_pair(WINE, PAPER, WINE).replace("M36 9C30 18", "M36 -8C31 10").replace("M36 9c2 10 6 18 10 25", "M36 -8c1 16 5 30 10 42")
    pair = pair.split('<path d="M36 9c6-6')[0] + pair[pair.index("<circle"):]          # no leaf: the stalk leaves the frame
    fruit = g(pair, sx + sw / 2 - 32.5 * k, h * 0.5 - 30 * k, k)
    nd, _ = sora.text("morello", big, 0, 0, tracking=-big * 0.02)
    td, _ = inter.text(TAG, tag_size, 0, 0)
    words = (f'<path fill="{WINE}" transform="translate({pad} {h - pad - tag_size * 2.0:.1f})" d="{nd}"/>'
             f'<path fill="{SKY_DEEP}" transform="translate({pad + big * 0.04:.1f} {h - pad:.1f})" d="{td}"/>')
    return svg(w, h, slab + fruit + words, bg=MIST, label=f"Morello. {TAG}")


write("social-card.svg", scene(1280, 640, 136, 30, 72))
write("banner.svg", scene(1280, 320, 88, 22, 44))

try:
    import cairosvg
    for n, wpx in (("logo-wine", 1200), ("logo-sky", 1200), ("mark-wine", 512), ("mark-sky", 512), ("icon-wine", 512),
                   ("icon-sky", 512), ("banner", 1280), ("social-card", 1280)):
        cairosvg.svg2png(url=f"{OUT}/{n}.svg", write_to=f"{OUT}/png/{n}.png", output_width=wpx)
    for sz in (32, 180):
        cairosvg.svg2png(url=f"{OUT}/icon-wine.svg", write_to=f"{OUT}/png/favicon-{sz}.png", output_width=sz)
except ImportError:
    print("cairosvg not installed: SVGs written, PNG copies skipped")
print(sorted(os.listdir(OUT)), "wordmark width %.1f" % tw)
