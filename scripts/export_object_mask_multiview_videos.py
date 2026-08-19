#!/usr/bin/env python3
"""Render mask overlays for all GX views and encode one grid video per object."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image


def natural_object_key(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def render_tile(frame_path: Path, mask_path: Path, label: str, size: tuple[int, int]) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read frame: {frame_path}")

    with Image.open(mask_path) as mask_image:
        if mask_image.mode != "P":
            raise RuntimeError(f"Expected a P-mode PNG, got {mask_image.mode}: {mask_path}")
        index_mask = np.asarray(mask_image, dtype=np.uint8)
        raw_palette = mask_image.getpalette()
        if raw_palette is None:
            raise RuntimeError(f"P-mode mask has no palette: {mask_path}")
        # Pillow palettes can be shorter than 256 entries. Pad unused entries safely.
        raw_palette = raw_palette + [0] * (768 - len(raw_palette))
        palette_rgb = np.asarray(raw_palette[:768], dtype=np.uint8).reshape(256, 3)

    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    index_mask = cv2.resize(index_mask, size, interpolation=cv2.INTER_NEAREST)
    foreground = index_mask != 0
    if foreground.any():
        # Preserve every palette index's original RGB color (OpenCV frames are BGR).
        palette_bgr = palette_rgb[:, ::-1]
        palette_image = palette_bgr[index_mask]
        pixels = frame[foreground].astype(np.float32)
        colors = palette_image[foreground].astype(np.float32)
        frame[foreground] = np.clip(pixels * 0.52 + colors * 0.48, 0, 255).astype(np.uint8)

        # Draw each instance's boundary in its exact palette color.
        kernel = np.ones((3, 3), np.uint8)
        for instance_id in np.unique(index_mask[foreground]):
            instance = (index_mask == instance_id).astype(np.uint8) * 255
            contour = cv2.morphologyEx(instance, cv2.MORPH_GRADIENT, kernel) > 0
            frame[contour] = palette_bgr[instance_id]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(frame, (8, 8), (22 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, label, (15, 15 + text_h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def render_object(
    object_dir: Path,
    output_dir: Path,
    fps: float,
    crf: int,
    preset: str,
    encoder_threads: int,
    tile_size: tuple[int, int],
    max_frames: int | None,
) -> dict[str, object]:
    views = sorted(p.name for p in (object_dir / "masks").glob("GX*") if p.is_dir())
    if not views:
        raise RuntimeError(f"No GX mask directories found in {object_dir}")
    if len(views) != 8:
        raise RuntimeError(f"Expected 8 GX views in {object_dir}, found {len(views)}")

    frame_ids = [p.stem for p in sorted((object_dir / "masks" / views[0]).glob("*.png"))]
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise RuntimeError(f"No mask frames found in {object_dir}")

    for view in views:
        for frame_id in frame_ids:
            frame_path = object_dir / "frames" / view / f"{frame_id}.jpg"
            mask_path = object_dir / "masks" / view / f"{frame_id}.png"
            if not frame_path.is_file() or not mask_path.is_file():
                raise RuntimeError(f"Missing matching frame/mask pair: {view}/{frame_id}")

    tile_w, tile_h = tile_size
    width, height = tile_w * 4, tile_h * 2
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{object_dir.name}_8view_mask_overlay.mp4"
    temp_path = output_dir / f".{final_path.stem}.encoding.mp4"

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-threads", str(encoder_threads),
        "-movflags", "+faststart", str(temp_path),
    ]
    started = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index, frame_id in enumerate(frame_ids, start=1):
            tiles = [
                render_tile(
                    object_dir / "frames" / view / f"{frame_id}.jpg",
                    object_dir / "masks" / view / f"{frame_id}.png",
                    view,
                    tile_size,
                )
                for view in views
            ]
            grid = np.vstack((np.hstack(tiles[:4]), np.hstack(tiles[4:])))
            process.stdin.write(grid.tobytes())
            if index == 1 or index % 50 == 0 or index == len(frame_ids):
                elapsed = time.monotonic() - started
                print(
                    f"[{object_dir.name}] {index}/{len(frame_ids)} frames "
                    f"({index / max(elapsed, 1e-6):.1f} fps render throughput)",
                    flush=True,
                )
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {object_dir.name} with status {return_code}")
    os.replace(temp_path, final_path)
    return {
        "object": object_dir.name,
        "file": final_path.name,
        "views": views,
        "frames": len(frame_ids),
        "fps": fps,
        "duration_seconds": len(frame_ids) / fps,
        "resolution": f"{width}x{height}",
        "codec": "H.264/libx264",
        "crf": crf,
        "preset": preset,
        "encoder_threads": encoder_threads,
        "size_bytes": final_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/objects-0805"))
    parser.add_argument(
        "--output", type=Path, default=Path("exports/objects-0805_mask_overlay_multiview")
    )
    parser.add_argument("--objects", nargs="*", help="Optional object directory names")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoder-threads", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=360)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    selected = set(args.objects or [])
    object_dirs = sorted(args.input.glob("Object-*"), key=natural_object_key)
    object_dirs = [p for p in object_dirs if p.name != "Object-1"]
    if selected:
        object_dirs = [p for p in object_dirs if p.name in selected]
    if not object_dirs:
        parser.error("No matching objects found")

    worker_count = max(1, min(args.workers, len(object_dirs)))
    print(
        f"Rendering {len(object_dirs)} objects with {worker_count} object-level workers "
        f"and {args.encoder_threads} x264 threads per worker",
        flush=True,
    )
    results = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                render_object,
                object_dir,
                args.output,
                args.fps,
                args.crf,
                args.preset,
                args.encoder_threads,
                (args.tile_width, args.tile_height),
                args.max_frames,
            ): object_dir.name
            for object_dir in object_dirs
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{result['object']}] complete: {result['file']}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    results.sort(key=lambda result: int(str(result["object"]).split("-", 1)[1]))

    manifest = {
        "description": "Eight-view GX videos with original P-mode palette mask colors in a 4x2 grid",
        "mask_rendering": "index 0 transparent; each nonzero index uses its PNG palette RGB color",
        "source": str(args.input),
        "excluded": ["Object-1"],
        "videos": results,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} videos and {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
