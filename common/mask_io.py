"""Shared helpers for normalized mask pipeline (bbox.json + palette PNG)."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image

VIEW_NAMES = ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6"]

PART_COLORS: dict[str, list[int]] = {
    "lid": [255, 59, 48],
    "body": [52, 199, 89],
    "inner_pot": [0, 122, 255],
}

DEFAULT_PARTS = ["lid", "body", "inner_pot"]
DEFAULT_PROMPTS = {
    "lid": ["lid"],
    "body": ["rice cooker body", "cooker body without lid"],
    "inner_pot": ["inner pot", "metal bowl", "black pot"],
}


def parts_meta(parts: list[str] | None = None) -> dict[str, dict[str, Any]]:
    parts = parts or DEFAULT_PARTS
    return {
        name: {"id": i + 1, "color": PART_COLORS[name]}
        for i, name in enumerate(parts)
    }


def part_id_map(parts: list[str] | None = None) -> dict[str, int]:
    return {name: meta["id"] for name, meta in parts_meta(parts).items()}


def frame_path(frames_dir: str, layout: str, timestamp: str, view: str) -> str:
    if layout == "normalized":
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(frames_dir, view, f"{timestamp}{ext}")
            if os.path.exists(path):
                return path
        return os.path.join(frames_dir, view, f"{timestamp}.jpg")
    return os.path.join(frames_dir, timestamp, f"{view}.png")


def list_timestamps(frames_dir: str, layout: str) -> list[str]:
    if layout == "normalized":
        view_dir = os.path.join(frames_dir, VIEW_NAMES[0])
        if not os.path.isdir(view_dir):
            raise FileNotFoundError(view_dir)
        ts = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(view_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        )
        return ts
    return sorted(d for d in os.listdir(frames_dir) if d.isdigit())


def bbox_json_path(masks_dir: str) -> str:
    return os.path.join(masks_dir, "bbox.json")


def load_bbox_json(masks_dir: str) -> dict:
    path = bbox_json_path(masks_dir)
    if not os.path.exists(path):
        return {"parts": parts_meta(), "frames": {}}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("parts", parts_meta())
    data.setdefault("frames", {})
    return data


def save_bbox_json(masks_dir: str, data: dict) -> None:
    os.makedirs(masks_dir, exist_ok=True)
    path = bbox_json_path(masks_dir)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def update_bbox_frame(
    masks_dir: str,
    timestamp: str,
    view: str,
    image_path: str,
    image_size: list[int],
    parts: list[dict],
    parts_list: list[str] | None = None,
) -> None:
    data = load_bbox_json(masks_dir)
    data["parts"] = parts_meta(parts_list)
    data["frames"].setdefault(timestamp, {})
    data["frames"][timestamp][view] = {
        "image_path": image_path,
        "image_size": image_size,
        "parts": parts,
    }
    save_bbox_json(masks_dir, data)


def qwen_bboxes_from_json(masks_dir: str, timestamp: str, view: str) -> dict[str, list[float]]:
    data = load_bbox_json(masks_dir)
    view_data = data.get("frames", {}).get(timestamp, {}).get(view, {})
    out: dict[str, list[float]] = {}
    for item in view_data.get("parts", []):
        label = item.get("label")
        bb = item.get("bbox_2d")
        if label and bb and len(bb) == 4:
            out[str(label)] = [float(v) for v in bb]
    return out


def build_palette(parts: list[str] | None = None) -> list[int]:
    """Flat RGB list (256*3) for PIL palette mode; index 0 = background."""
    flat = [0, 0, 0]
    for name in (parts or DEFAULT_PARTS):
        r, g, b = PART_COLORS[name]
        flat.extend([r, g, b])
    while len(flat) < 256 * 3:
        flat.append(0)
    return flat[: 256 * 3]


def masks_to_label_map(
    part_masks: dict[str, np.ndarray],
    parts: list[str] | None = None,
) -> np.ndarray:
    """Merge boolean part masks into uint8 indexed label map (later parts overwrite)."""
    parts = parts or DEFAULT_PARTS
    ids = part_id_map(parts)
    any_mask = next(iter(part_masks.values()))
    label = np.zeros(any_mask.shape, dtype=np.uint8)
    for name in parts:
        m = part_masks.get(name)
        if m is not None and m.any():
            label[m.astype(bool)] = ids[name]
    return label


def save_palette_png(label: np.ndarray, out_path: str, parts: list[str] | None = None) -> None:
    if label.dtype != np.uint8:
        label = label.astype(np.uint8)
    img = Image.fromarray(label, mode="P")
    img.putpalette(build_palette(parts))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
