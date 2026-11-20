# Morello brand

The brand book, tokens for light and dark themes, and the logo files live in the Morello design
system. This folder is the copy that ships with the code.

A morello is a dark sour cherry, which is where the wine colour comes from. The mark is a lowercase
m made of two arches side by side. The arches should match, and the second one is the wrong
colour, which is the kind of row the tool looks for.

## Colours

| name | hex | use |
|---|---|---|
| wine | `#391217` | primary; text on light grounds, large blocks, the mark |
| sky | `#8DA5CA` | secondary; the odd arch, large blocks, the reversed mark. Not for text on light grounds (2.4:1) |
| mist | `#DDE6F7` | banner and card grounds |
| paper | `#F6F8FC` | page ground, text on wine, the odd arch on dark grounds |
| muted | `#4F658B` | secondary text and links on light grounds (5.5:1) |
| alert | `#9E2A33` | the word ALERT on light grounds (7.0:1) |

Wine on paper is 15.5:1, wine on sky 6.6:1, paper on wine 15.5:1.

## Type

- Display and wordmark: Space Grotesk SemiBold (`fonts/SpaceGrotesk-SemiBold.woff2`).
- Text: Inter. Code: JetBrains Mono.

Space Grotesk and Inter are licensed under the SIL Open Font License 1.1 (`fonts/OFL.txt`). The wordmark
and the tagline in the banner are those fonts converted to outlines.

## Files

| file | use |
|---|---|
| `logo-wine.svg`, `logo-sky.svg` | mark and name; wine ink for light grounds, sky ink for wine or dark grounds |
| `mark-*.svg`, `wordmark-*.svg` | either half on its own |
| `icon-*.svg`, `png/favicon-*.png` | app icon and favicon |
| `banner.svg` | top of a README or docs page (1280 x 320) |
| `social-card.svg` | link previews and title slides (1280 x 640) |
| `png/` | raster copies |
| `build_assets.py` | regenerates every file from the two font files in `fonts/` |

## Rules

- The wordmark, the package and the command are lowercase: `morello`. In a sentence the product
  is Morello.
- Light grounds: wine arch on the left, sky arch on the right. Wine and dark grounds: sky on
  the left, paper on the right. Do not swap sides or make both arches the same colour.
- Do not rotate, outline or add effects to the mark.
- Clear space: the width of one arch on every side. Minimum size: mark 16px tall, lockup 20px.
- In the lockup the mark stands on the wordmark's baseline.
- Flat colours, no gradients, no shadows.

## Name

The two brand colours were sampled from a reference image chosen by the owner. The name has not
been checked for trademark conflicts. Arm runs a processor research programme called Morello
(CHERI capability hardware), which is a different field from data tooling.
