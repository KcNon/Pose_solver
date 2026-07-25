#!/usr/bin/env python
"""Create cropped RGBA ReconViaGen inputs from original RGB and SAM masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def image_path(root: Path, timestamp: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = root / f"{timestamp}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing RGB frame {timestamp} under {root}")


def crop_rgba(rgb: np.ndarray, mask: np.ndarray, padding: float) -> Image.Image:
    ys, xs = np.nonzero(mask)
    if len(xs) < 500:
        raise RuntimeError(f"mask too small: {len(xs)} pixels")
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(np.ceil(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)))
    side = max(side, 32)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, top = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    src_x1, src_y1 = max(left, 0), max(top, 0)
    src_x2, src_y2 = min(left + side, rgb.shape[1]), min(top + side, rgb.shape[0])
    dst_x1, dst_y1 = src_x1 - left, src_y1 - top
    dst_x2, dst_y2 = dst_x1 + src_x2 - src_x1, dst_y1 + src_y2 - src_y1
    rgba[dst_y1:dst_y2, dst_x1:dst_x2, :3] = rgb[src_y1:src_y2, src_x1:src_x2]
    rgba[dst_y1:dst_y2, dst_x1:dst_x2, 3] = (
        mask[src_y1:src_y2, src_x1:src_x2].astype(np.uint8) * 255
    )
    return Image.fromarray(rgba, "RGBA")


def contact_sheet(paths: list[Path], output: Path) -> None:
    tiles = []
    for path in paths:
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        alpha = rgba[..., 3:4].astype(np.float32) / 255
        checker = np.full(rgba.shape[:2] + (3,), 220, dtype=np.uint8)
        comp = (rgba[..., :3] * alpha + checker * (1 - alpha)).astype(np.uint8)
        tile = cv2.resize(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR), (240, 240))
        cv2.putText(tile, path.stem, (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)
        tiles.append(tile)
    cols = 4
    blank = np.zeros_like(tiles[0])
    rows = [
        np.hstack(tiles[index:index + cols] + [blank] * (cols - len(tiles[index:index + cols])))
        for index in range(0, len(tiles), cols)
    ]
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[3]
            / "configs"
            / "reconviagen_objects.json"
        ),
    )
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--padding", type=float, default=0.1)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_root = Path(config["output_root"]).resolve()
    output_root = Path(config["rgba_root"]).resolve()
    selected = args.parts or list(config["parts"])
    unknown = set(selected).difference(config["parts"])
    if unknown:
        raise ValueError(f"unknown parts: {sorted(unknown)}")
    manifest = {"method": "qwen_per_frame_then_sam_alpha", "parts": {}}
    for part in selected:
        spec = config["parts"][part]
        output_dir = output_root / part
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        records = []
        frames_dir = source_root / part / "frames" / part
        masks_dir = source_root / part / "masks"
        for timestamp in spec["rgba_timestamps"]:
            source = image_path(frames_dir, timestamp)
            mask_path = masks_dir / timestamp / f"{part}.png"
            rgb = np.asarray(Image.open(source).convert("RGB"))
            labels = np.asarray(Image.open(mask_path))
            mask = labels == int(spec["id"])
            if spec.get("postprocess") == "convex_hull":
                points = cv2.findNonZero(mask.astype(np.uint8))
                if points is None:
                    raise RuntimeError(f"empty mask: {mask_path}")
                hull = cv2.convexHull(points)
                repaired = np.zeros(mask.shape, dtype=np.uint8)
                cv2.fillConvexPoly(repaired, hull, 1)
                mask = repaired.astype(bool)
            rgba = crop_rgba(rgb, mask, args.padding)
            destination = output_dir / f"{timestamp}.png"
            rgba.save(destination)
            written.append(destination)
            records.append({
                "timestamp": timestamp,
                "rgb": str(source),
                "mask": str(mask_path),
                "rgba": str(destination),
                "mask_pixels": int(mask.sum()),
                "rgba_size": list(rgba.size),
            })
        contact_sheet(written, output_dir / "contact_sheet.jpg")
        manifest["parts"][part] = {"count": len(written), "frames": records}
        print(f"{part}: wrote {len(written)} RGBA inputs -> {output_dir}")
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
