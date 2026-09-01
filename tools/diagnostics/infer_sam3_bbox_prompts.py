#!/usr/bin/env python3
"""Visualize bounded box prompts or run SAM3.1 instance-box inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


MAX_VIEWS = 16
MAX_PARTS = 16
MAX_SOURCE_PIXELS = 4096 * 4096


def _load_spec(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    frame_id = str(raw["frame_id"])
    if not frame_id.isdigit():
        raise ValueError("frame_id must be numeric")
    views = raw.get("views", {})
    parts = raw.get("parts", {})
    if not 1 <= len(views) <= MAX_VIEWS:
        raise ValueError(f"spec must contain 1..{MAX_VIEWS} views")
    if not 1 <= len(parts) <= MAX_PARTS:
        raise ValueError(f"spec must contain 1..{MAX_PARTS} parts")
    part_ids = [int(value["id"]) for value in parts.values()]
    if any(value < 1 or value > 255 for value in part_ids):
        raise ValueError("part IDs must be in [1, 255]")
    if len(part_ids) != len(set(part_ids)):
        raise ValueError("part IDs must be unique")
    order = raw.get("front_to_back", list(parts))
    if set(order) != set(parts) or len(order) != len(parts):
        raise ValueError("front_to_back must contain every part exactly once")
    for part, values in parts.items():
        color = values.get("color_rgb")
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError(f"invalid color for {part}")
        if any(int(channel) < 0 or int(channel) > 255 for channel in color):
            raise ValueError(f"invalid color for {part}")
    for view, values in views.items():
        boxes = values.get("boxes_normalized_1000", {})
        if set(boxes) != set(parts):
            raise ValueError(f"{view} must provide exactly one box per part")
        for part, box in boxes.items():
            if len(box) != 4:
                raise ValueError(f"invalid box for {view}/{part}")
            x1, y1, x2, y2 = (float(value) for value in box)
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError(f"invalid normalized box for {view}/{part}: {box}")
    return raw


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size
        )
    except OSError:
        return ImageFont.load_default()


def _frame_path(spec: dict[str, Any], view: str) -> Path:
    root = Path(spec["frames_root"]).resolve()
    frame_id = str(spec["frame_id"])
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = root / view / f"{frame_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(root / view / f"{frame_id}.jpg")


def _pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        round(x1 / 1000.0 * width),
        round(y1 / 1000.0 * height),
        round(x2 / 1000.0 * width),
        round(y2 / 1000.0 * height),
    )


def _draw_boxes(image: Image.Image, spec: dict[str, Any], view: str) -> Image.Image:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    width, height = preview.size
    line_width = max(3, round(min(width, height) / 240))
    font = _font(max(18, round(min(width, height) / 42)))
    boxes = spec["views"][view]["boxes_normalized_1000"]
    for part, part_spec in spec["parts"].items():
        color = tuple(int(value) for value in part_spec["color_rgb"])
        x1, y1, x2, y2 = _pixels(boxes[part], width, height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text_box = draw.textbbox((0, 0), part, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        top = max(0, y1 - text_height - 10)
        draw.rectangle((x1, top, x1 + text_width + 12, top + text_height + 8), fill=color)
        draw.text((x1 + 6, top + 3), part, fill=(255, 255, 255), font=font)
    return preview


def _grid(images: list[tuple[str, Image.Image]], output: Path) -> None:
    tile_width, tile_height = 640, 360
    grid = Image.new("RGB", (tile_width * len(images), tile_height), (0, 0, 0))
    for index, (_view, image) in enumerate(images):
        tile = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        grid.paste(tile, (index * tile_width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output, quality=95)


def _visualize_boxes(spec: dict[str, Any], output_root: Path) -> None:
    previews = []
    report: dict[str, Any] = {"frame_id": spec["frame_id"], "views": {}}
    for view in spec["views"]:
        path = _frame_path(spec, view)
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.width * image.height > MAX_SOURCE_PIXELS:
            raise ValueError(f"source image exceeds pixel budget: {path}")
        preview = _draw_boxes(image, spec, view)
        destination = output_root / "bbox_previews" / f"{view}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        preview.save(destination, quality=95)
        previews.append((view, preview))
        report["views"][view] = {
            "frame": str(path),
            "size": [image.width, image.height],
            "boxes_normalized_1000": spec["views"][view]["boxes_normalized_1000"],
            "boxes_pixels": {
                part: list(_pixels(box, image.width, image.height))
                for part, box in spec["views"][view]["boxes_normalized_1000"].items()
            },
        }
    _grid(previews, output_root / "bbox_grid.jpg")
    (output_root / "bbox_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"bbox visualization -> {output_root / 'bbox_grid.jpg'}", flush=True)


def _save_palette_mask(path: Path, labels: np.ndarray, spec: dict[str, Any]) -> None:
    palette = [0] * (256 * 3)
    for values in spec["parts"].values():
        offset = int(values["id"]) * 3
        palette[offset : offset + 3] = [int(value) for value in values["color_rgb"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(labels.astype(np.uint8), mode="P")
    image.putpalette(palette)
    image.save(path, format="PNG", optimize=True)


def _infer(
    spec: dict[str, Any],
    output_root: Path,
    checkpoint: Path,
    minimum_pixels: int,
) -> None:
    import torch

    from common.masking.sam import (
        build_sam31_instance_box_processor,
        predict_image_mask,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SAM3 bbox inference requires exactly one visible CUDA device")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    processor = build_sam31_instance_box_processor(str(checkpoint))
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    previews: list[tuple[str, Image.Image]] = []
    report: dict[str, Any] = {
        "method": "sam31_instance_box",
        "frame_id": spec["frame_id"],
        "checkpoint": str(checkpoint.resolve()),
        "visible_cuda_devices": int(torch.cuda.device_count()),
        "views": {},
    }
    for view in spec["views"]:
        path = _frame_path(spec, view)
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.width * image.height > MAX_SOURCE_PIXELS:
            raise ValueError(f"source image exceeds pixel budget: {path}")
        with autocast:
            state = processor.set_image(image)
        masks: dict[str, np.ndarray] = {}
        details: dict[str, Any] = {}
        boxes = spec["views"][view]["boxes_normalized_1000"]
        for part in spec["parts"]:
            mask, info = predict_image_mask(
                processor,
                state,
                [part.replace("_", " ")],
                boxes[part],
                confidence=0.0,
                autocast=autocast,
                minimum_pixels=minimum_pixels,
                prompt_mode="instance_box",
            )
            if mask is None:
                raise RuntimeError(f"SAM3 returned an empty mask for {view}/{part}: {info}")
            masks[part] = np.asarray(mask, dtype=bool)
            mask_path = output_root / "part_masks" / part / view / f"{spec['frame_id']}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            details[part] = {**info, "mask": str(mask_path)}
            print(
                f"{view}/{part}: pixels={info['pixels']} score={info.get('score')}",
                flush=True,
            )

        labels = np.zeros((image.height, image.width), dtype=np.uint8)
        for part in reversed(spec["front_to_back"]):
            labels[masks[part]] = int(spec["parts"][part]["id"])
        palette_path = output_root / "masks" / view / f"{spec['frame_id']}.png"
        _save_palette_mask(palette_path, labels, spec)

        overlay = np.asarray(image).astype(np.float32)
        for part in reversed(spec["front_to_back"]):
            mask = masks[part]
            color = np.asarray(spec["parts"][part]["color_rgb"], dtype=np.float32)
            overlay[mask] = overlay[mask] * 0.55 + color * 0.45
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        for part in spec["front_to_back"]:
            contours, _ = cv2.findContours(
                masks[part].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                overlay,
                contours,
                -1,
                tuple(int(value) for value in spec["parts"][part]["color_rgb"]),
                3,
            )
        preview = Image.fromarray(overlay)
        draw = ImageDraw.Draw(preview)
        draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
        draw.text(
            (9, 8),
            f"{view} | SAM3.1 instance-box | frame {spec['frame_id']}",
            fill=(255, 255, 255),
            font=_font(18),
        )
        preview_path = output_root / "sam3_previews" / f"{view}.jpg"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(preview_path, quality=95)
        previews.append((view, preview))
        report["views"][view] = {
            "frame": str(path),
            "palette_mask": str(palette_path),
            "parts": details,
        }
    _grid(previews, output_root / "sam3_grid.jpg")
    (output_root / "sam3_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"SAM3 visualization -> {output_root / 'sam3_grid.jpg'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("bbox", "infer"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--minimum-pixels", type=int, default=500)
    args = parser.parse_args()
    if args.minimum_pixels < 1:
        raise ValueError("minimum-pixels must be positive")
    spec = _load_spec(args.spec.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "bbox":
        _visualize_boxes(spec, output_root)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for infer")
        _infer(spec, output_root, args.checkpoint.resolve(), args.minimum_pixels)


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/diagnostics/infer_sam3_bbox_prompts.py")
    main()
