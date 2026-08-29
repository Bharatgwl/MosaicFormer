from __future__ import annotations

import math
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "RESULT"
PNG_PATH = RESULT_DIR / "mosaicformer_architecture_diagram.png"
SVG_PATH = RESULT_DIR / "mosaicformer_architecture_diagram.svg"

W, H = 1200, 1780
BG = "#f7f7f4"
INK = "#111111"
SOFT_GREEN = "#e8f5e3"
PINK = "#f5ddf5"
BLUE = "#d7edf5"
YELLOW = "#fff7d9"
ORANGE = "#ffe7bd"
PURPLE = "#e6defb"
GRAY = "#f2f2f2"


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(46, True)
FONT_BIG = font(30, True)
FONT_MED = font(26, True)
FONT_TEXT = font(24)
FONT_SMALL = font(18)


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
            f'<rect width="{W}" height="{H}" fill="{BG}"/>',
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#111"/></marker></defs>',
        ]

    def rect(self, xy, fill, stroke=INK, radius=14, width=2):
        x1, y1, x2, y2 = xy
        self.parts.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
            f'rx="{radius}" ry="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def line(self, points, width=3, arrow=True):
        d = " ".join(f"{x},{y}" for x, y in points)
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<polyline points="{d}" fill="none" stroke="{INK}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{marker}/>'
        )

    def text(self, x, y, text, size=24, bold=False, anchor="middle"):
        weight = "700" if bold else "400"
        for i, line in enumerate(text.split("\n")):
            self.parts.append(
                f'<text x="{x}" y="{y + i * (size + 6)}" font-family="Arial, Segoe UI, sans-serif" '
                f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{INK}">{escape(line)}</text>'
            )

    def circle(self, cx, cy, r, fill=BG, width=2):
        self.parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{INK}" stroke-width="{width}"/>'
        )

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def draw_centered(draw, box, text, fill=INK, fnt=FONT_TEXT, line_gap=4):
    x1, y1, x2, y2 = box
    max_chars = max(8, int((x2 - x1) / (fnt.size * 0.55)))
    lines: list[str] = []
    for raw in text.split("\n"):
        lines.extend(wrap(raw, width=max_chars) or [""])
    line_heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total_h = sum(line_heights) + line_gap * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2
    for line, line_h in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        x = x1 + ((x2 - x1) - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, fill=fill, font=fnt)
        y += line_h + line_gap


def rounded_box(draw, svg, box, label, fill, fnt=FONT_TEXT, radius=14):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=INK, width=2)
    svg.rect(box, fill, radius=radius)
    draw_centered(draw, box, label, fnt=fnt)
    x1, y1, x2, y2 = box
    svg.text((x1 + x2) // 2, y1 + (y2 - y1) // 2 - 8, label, size=fnt.size, bold=fnt in {FONT_TITLE, FONT_BIG, FONT_MED})


def arrow(draw, svg, points, width=4):
    draw.line(points, fill=INK, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 14
    left = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
    right = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
    draw.polygon([(x2, y2), left, right], fill=INK)
    svg.line(points, width=width)


def plus(draw, svg, cx, cy, r=28):
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=BG, outline=INK, width=2)
    draw.line((cx - r + 8, cy, cx + r - 8, cy), fill=INK, width=2)
    draw.line((cx, cy - r + 8, cx, cy + r - 8), fill=INK, width=2)
    svg.circle(cx, cy, r)
    svg.line([(cx - r + 8, cy), (cx + r - 8, cy)], width=2, arrow=False)
    svg.line([(cx, cy - r + 8), (cx, cy + r - 8)], width=2, arrow=False)


def create_diagram() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    svg = Svg()

    draw.text((W // 2, 34), "MosaicFormer / MosaicTransformer Block", fill=INK, font=FONT_TITLE, anchor="mm")
    svg.text(W // 2, 52, "MosaicFormer / MosaicTransformer Block", size=46, bold=True)
    draw.text((W // 2, 92), "Local detail + compressed memory + learned gate", fill=INK, font=FONT_TEXT, anchor="mm")
    svg.text(W // 2, 104, "Local detail + compressed memory + learned gate", size=24)

    # Bottom input flow.
    rounded_box(draw, svg, (420, 1610, 780, 1690), "Input token IDs", PINK, FONT_MED)
    arrow(draw, svg, [(600, 1570), (600, 1610)])
    plus(draw, svg, 600, 1530)
    rounded_box(draw, svg, (350, 1420, 850, 1498), "Token embedding + position embedding", PINK, FONT_TEXT)
    arrow(draw, svg, [(600, 1498), (600, 1502)])
    arrow(draw, svg, [(600, 1558), (600, 1570)])
    draw.text((810, 1535), "position", fill=INK, font=FONT_SMALL, anchor="lm")
    svg.text(810, 1542, "position", size=18, anchor="start")

    rounded_box(draw, svg, (390, 1298, 810, 1372), "Split sequence into chunks", BLUE, FONT_TEXT)
    arrow(draw, svg, [(600, 1372), (600, 1420)])

    # Main block.
    block = (250, 465, 950, 1270)
    draw.rounded_rectangle(block, radius=24, fill=SOFT_GREEN, outline=INK, width=2)
    svg.rect(block, SOFT_GREEN, radius=24)
    draw.text((985, 840), "Repeated\nMosaic\nlayer", fill=INK, font=FONT_MED, anchor="lm", spacing=6)
    svg.text(985, 815, "Repeated\nMosaic\nlayer", size=26, bold=True, anchor="start")

    rounded_box(draw, svg, (385, 1160, 815, 1238), "Local Causal Attention\nnearby exact words", PINK, FONT_TEXT)
    rounded_box(draw, svg, (385, 1020, 815, 1098), "Add & Norm", YELLOW, FONT_TEXT)
    rounded_box(draw, svg, (350, 860, 850, 938), "Memory Cross-Attention\nread old summary notes", PURPLE, FONT_TEXT)
    rounded_box(draw, svg, (385, 720, 815, 798), "Gated Fusion\nmix local + memory", BLUE, FONT_TEXT)
    rounded_box(draw, svg, (385, 580, 815, 658), "Add & Norm", YELLOW, FONT_TEXT)
    rounded_box(draw, svg, (410, 510, 790, 570), "Feed Forward", ORANGE, FONT_TEXT)

    arrow(draw, svg, [(600, 1298), (600, 1238)])
    arrow(draw, svg, [(600, 1160), (600, 1098)])
    arrow(draw, svg, [(600, 1020), (600, 938)])
    arrow(draw, svg, [(600, 860), (600, 798)])
    arrow(draw, svg, [(600, 720), (600, 658)])
    arrow(draw, svg, [(600, 580), (600, 570)])

    # Residual side arrows.
    arrow(draw, svg, [(815, 1198), (895, 1198), (895, 1058), (815, 1058)])
    arrow(draw, svg, [(815, 758), (895, 758), (895, 618), (815, 618)])

    # Memory side.
    rounded_box(draw, svg, (60, 1010, 300, 1095), "Previous\nchunk states", GRAY, FONT_TEXT)
    rounded_box(draw, svg, (55, 840, 305, 930), "Memory Compressor\n64 notes -> 8 notes", ORANGE, FONT_SMALL)
    rounded_box(draw, svg, (55, 685, 305, 775), "Memory Bank\nstored summaries", BLUE, FONT_TEXT)
    arrow(draw, svg, [(180, 1010), (180, 930)])
    arrow(draw, svg, [(180, 840), (180, 775)])
    arrow(draw, svg, [(305, 730), (350, 900)])
    draw.text((48, 808), "old context", fill=INK, font=FONT_SMALL, anchor="lm")
    svg.text(48, 815, "old context", size=18, anchor="start")

    # Compressor output from current block to memory bank.
    rounded_box(draw, svg, (70, 510, 300, 600), "New memory\nfor later chunks", BLUE, FONT_SMALL)
    arrow(draw, svg, [(385, 550), (300, 550)])
    arrow(draw, svg, [(185, 600), (185, 685)])

    # Top output flow.
    rounded_box(draw, svg, (395, 350, 805, 415), "Final LayerNorm", YELLOW, FONT_TEXT)
    rounded_box(draw, svg, (430, 255, 770, 320), "Linear LM Head", BLUE, FONT_TEXT)
    rounded_box(draw, svg, (455, 165, 745, 230), "Softmax", PURPLE, FONT_TEXT)
    arrow(draw, svg, [(600, 510), (600, 415)])
    arrow(draw, svg, [(600, 350), (600, 320)])
    arrow(draw, svg, [(600, 255), (600, 230)])
    arrow(draw, svg, [(600, 165), (600, 135)])
    draw.text((600, 130), "Output probabilities", fill=INK, font=FONT_MED, anchor="ms")
    svg.text(600, 128, "Output probabilities", size=26, bold=True)

    # Explanatory notes.
    notes = [
        ("Local path", "nearby detail", 965, 1198),
        ("Memory path", "old summaries", 965, 900),
        ("Gate", "choose mix", 965, 755),
    ]
    for title, body, x, y in notes:
        draw.text((x, y), title, fill=INK, font=FONT_SMALL, anchor="lm")
        draw.text((x, y + 25), body, fill=INK, font=FONT_SMALL, anchor="lm")
        svg.text(x, y + 8, title, size=18, bold=True, anchor="start")
        svg.text(x, y + 34, body, size=18, anchor="start")

    image.save(PNG_PATH)
    svg.save(SVG_PATH)
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    create_diagram()
