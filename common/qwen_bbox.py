"""Qwen3-VL bbox parsing / visualization helpers."""
from __future__ import annotations

import json
import re

from PIL import Image, ImageDraw, ImageFont

LABEL_ALIASES = {
    "pot": "inner_pot",
    "inner pot": "inner_pot",
    "inner_pot": "inner_pot",
    "innerpot": "inner_pot",
    "lid": "lid",
    "pot lid": "lid",
    "cover": "lid",
    "body": "body",
    "pot body": "body",
    "outer shell": "body",
    "rice cooker body": "body",
    "cooker body": "body",
}

BATCH_PROMPT = (
    "The scene is a photography area enclosed by several black boards. "
    "These black boards, the background, and the patterned cloth on the ground "
    "are only the boundary/backdrop — do NOT detect them. "
    "Detect the following rice-cooker parts IF visible in this image:\n"
    "- lid: the circular metallic pot lid with a knob\n"
    "- body: the outer rice cooker shell/housing (exterior case, NOT the inner pot)\n"
    "- inner_pot: the black removable inner cooking pot bowl\n"
    "If a part is NOT visible in this view, omit it. "
    'Return ONLY a JSON array: [{"bbox_2d":[x1,y1,x2,y2],"label":"lid"|"body"|"inner_pot"}]. '
    "Coordinates normalized 0-1000. No other text."
)

COLORS = [
    "#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE", "#FF2D55",
    "#5AC8FA", "#FFCC00", "#00C7BE", "#A2845E", "#8E8E93", "#30B0C7",
]


def normalize_label(label: str) -> str:
    key = label.strip().lower().replace("-", "_")
    return LABEL_ALIASES.get(key, key.replace(" ", "_"))


def parse_boxes(text: str) -> list[dict]:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    payload = m.group(1) if m else text
    start, end = payload.find("["), payload.rfind("]")
    if start != -1 and end > start:
        payload = payload[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        data = []
        for obj in re.findall(r"\{[^{}]*\}", payload):
            try:
                data.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
    boxes = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bb = item.get("bbox_2d") or item.get("bbox") or item.get("box_2d")
        if bb is None or len(bb) != 4:
            continue
        boxes.append({"bbox_2d": [float(v) for v in bb],
                      "label": normalize_label(str(item.get("label", "object")))})
    return boxes


def to_pixels(box, width, height):
    x1, y1, x2, y2 = box
    ax1, ax2 = sorted((x1 / 1000 * width, x2 / 1000 * width))
    ay1, ay2 = sorted((y1 / 1000 * height, y2 / 1000 * height))
    return int(ax1), int(ay1), int(ax2), int(ay2)


def visualize(image, boxes, out_path):
    img = image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)
    lw = max(2, round(min(W, H) / 300))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  size=max(14, round(min(W, H) / 45)))
    except OSError:
        font = ImageFont.load_default()
    for i, box in enumerate(boxes):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = to_pixels(box["bbox_2d"], W, H)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=lw)
        label = box["label"]
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle((x1, ty, x1 + tw + 8, ty + th + 6), fill=color)
        draw.text((x1 + 4, ty + 2), label, fill="white", font=font)
    img.save(out_path)
    return out_path
