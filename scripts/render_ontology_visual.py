from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "src" / "agent" / "tools" / "graph_ontology_v2.json"
OUTPUT_PATH = ROOT / "artifacts" / "ontology_visualization.png"

W, H = 1920, 1080
FONT_PATH = ROOT / "src" / "streamlit" / "source" / "강원교육튼튼.ttf"
FONT_BOLD_PATH = FONT_PATH


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
    return ImageFont.truetype(str(path), size=size)


def wrap_text(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False)


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    fill: str,
    fnt: ImageFont.FreeTypeFont,
    spacing: int = 8,
) -> None:
    x1, y1, x2, y2 = box
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total) / 2
    for line, line_h in zip(lines, heights):
        line_w = draw.textlength(line, font=fnt)
        draw.text((x1 + ((x2 - x1) - line_w) / 2, y), line, fill=fill, font=fnt)
        y += line_h + spacing


def rounded_node(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    size: tuple[int, int],
    title: str,
    subtitle: str,
    color: str,
    border: str,
) -> tuple[int, int, int, int]:
    cx, cy = center
    w, h = size
    box = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
    shadow = (box[0] + 8, box[1] + 10, box[2] + 8, box[3] + 10)
    draw.rounded_rectangle(shadow, radius=28, fill=(19, 31, 49, 36))
    draw.rounded_rectangle(box, radius=28, fill=color, outline=border, width=4)
    title_f = font(30, bold=True)
    sub_f = font(19)
    title_w = draw.textlength(title, font=title_f)
    draw.text((cx - title_w / 2, box[1] + 30), title, fill="#142033", font=title_f)
    desc_lines = wrap_text(subtitle, 20)
    draw_text_center(draw, (box[0] + 34, box[1] + 90, box[2] - 34, box[3] - 20), desc_lines, "#334155", sub_f, 6)
    return box


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    label: str,
    label_offset: tuple[int, int] = (0, 0),
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=5)
    angle = math.atan2(ey - sy, ex - sx)
    length = 22
    spread = math.radians(28)
    p1 = (ex - length * math.cos(angle - spread), ey - length * math.sin(angle - spread))
    p2 = (ex - length * math.cos(angle + spread), ey - length * math.sin(angle + spread))
    draw.polygon([(ex, ey), p1, p2], fill=color)

    mid = ((sx + ex) // 2 + label_offset[0], (sy + ey) // 2 + label_offset[1])
    label_f = font(21, bold=True)
    pad_x, pad_y = 16, 8
    tw = draw.textlength(label, font=label_f)
    th = draw.textbbox((0, 0), label, font=label_f)[3]
    label_box = (mid[0] - tw / 2 - pad_x, mid[1] - th / 2 - pad_y, mid[0] + tw / 2 + pad_x, mid[1] + th / 2 + pad_y)
    draw.rounded_rectangle(label_box, radius=14, fill="#FFFFFF", outline="#D9E2EC", width=2)
    draw.text((mid[0] - tw / 2, mid[1] - th / 2 - 2), label, fill="#243B53", font=label_f)


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, outline: str) -> int:
    x, y = xy
    f = font(20, bold=True)
    w = int(draw.textlength(text, font=f)) + 36
    h = 48
    draw.rounded_rectangle((x, y, x + w, y + h), radius=22, fill=fill, outline=outline, width=2)
    draw.text((x + 18, y + 12), text, fill="#243B53", font=f)
    return w


def main() -> None:
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    nodes = ontology["nodes"]
    relationships = ontology["relationships"]

    img = Image.new("RGB", (W, H), "#F7FAFC")
    draw = ImageDraw.Draw(img, "RGBA")

    for y in range(H):
        r = int(247 - y * 0.02)
        g = int(250 - y * 0.015)
        b = int(252 - y * 0.005)
        draw.line((0, y, W, y), fill=(r, g, b, 255))

    title_f = font(58, bold=True)
    sub_f = font(28)
    title = "기업 지식그래프 온톨로지"
    subtitle = "기업, 지역, 업종, 뉴스 데이터를 연결하는 발표용 개념 구조"
    draw.text((96, 70), title, fill="#102A43", font=title_f)
    draw.text((100, 146), subtitle, fill="#52606D", font=sub_f)

    stat_f = font(26, bold=True)
    small_f = font(18)
    stats = [
        (f"{len(nodes)}개", "노드 타입", 190),
        (f"{len(relationships)}개", "관계 타입", 190),
        ("증거 기반", "evidence/source 추적", 250),
    ]
    x = 1168
    for value, label, card_w in stats:
        draw.rounded_rectangle((x, 76, x + card_w, 168), radius=18, fill="#FFFFFF", outline="#D9E2EC", width=2)
        draw.text((x + 24, 94), value, fill="#102A43", font=stat_f)
        draw.text((x + 24, 130), label, fill="#627D98", font=small_f)
        x += card_w + 24

    colors = {
        "ParentCompany": ("#D9EAFD", "#4A90E2"),
        "SubsidiaryCompany": ("#DFF5E1", "#3BA66B"),
        "Region": ("#FFF1C9", "#D69E2E"),
        "Section": ("#F3E5F5", "#8E44AD"),
        "News": ("#FFE2DC", "#E76F51"),
    }

    positions = {
        "ParentCompany": (960, 520),
        "SubsidiaryCompany": (300, 520),
        "Region": (960, 795),
        "Section": (1620, 520),
        "News": (960, 260),
    }
    sizes = {
        "ParentCompany": (390, 188),
        "SubsidiaryCompany": (400, 170),
        "Region": (300, 136),
        "Section": (330, 154),
        "News": (340, 154),
    }

    boxes = {}
    for node, pos in positions.items():
        fill, border = colors[node]
        boxes[node] = rounded_node(draw, pos, sizes[node], node, nodes[node]["description"], fill, border)

    arrow(draw, (765, 520), (500, 520), "#3BA66B", "HAS_SUBSIDIARY", (0, -52))
    arrow(draw, (1155, 520), (1455, 520), "#8E44AD", "IN_INDUSTRY", (0, -52))
    arrow(draw, (960, 614), (960, 727), "#D69E2E", "LOCATED_IN", (-155, 18))
    arrow(draw, (960, 337), (960, 426), "#E76F51", "RELATED_TO", (130, 0))

    draw.arc((785, 388, 1135, 652), start=212, end=508, fill="#4A90E2", width=5)
    draw.polygon([(786, 586), (810, 579), (801, 604)], fill="#4A90E2")
    arrow_label_f = font(21, bold=True)
    label = "AFFILIATED_WITH"
    tw = draw.textlength(label, font=arrow_label_f)
    draw.rounded_rectangle((1260 - tw / 2 - 18, 657, 1260 + tw / 2 + 18, 704), radius=14, fill="#FFFFFF", outline="#B7D4F8", width=3)
    draw.text((1260 - tw / 2, 667), label, fill="#1D4F91", font=arrow_label_f)

    schema_y = 908
    draw.text((96, schema_y), "주요 속성 예시", fill="#102A43", font=font(30, bold=True))
    samples = [
        ("ParentCompany", "crno, name, address, aliases, homepage, representatives"),
        ("SubsidiaryCompany", "name, business_content, domestic, region"),
        ("News", "date, title, summary, publisher, url, embedding"),
    ]
    card_x = 330
    card_w = 460
    for node, text in samples:
        fill, border = colors[node]
        draw.rounded_rectangle((card_x, schema_y - 22, card_x + card_w, schema_y + 84), radius=18, fill="#FFFFFF", outline="#D9E2EC", width=2)
        pill(draw, (card_x + 22, schema_y - 2), node, fill, border)
        prop_f = font(16)
        for i, line in enumerate(wrap_text(text, 44)):
            draw.text((card_x + 22, schema_y + 44 + i * 20), line, fill="#52606D", font=prop_f)
        card_x += card_w + 34

    legend_y = 1022
    draw.line((96, 995, 1824, 995), fill="#D9E2EC", width=2)
    draw.text((96, legend_y), "관계 속성 공통: evidence, source_case, source_row", fill="#52606D", font=font(23))
    draw.text((1375, legend_y), "source: graph_ontology_v2.json", fill="#829AB1", font=font(21))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PATH, quality=96)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
