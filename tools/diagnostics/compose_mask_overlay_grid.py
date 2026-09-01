#!/usr/bin/env python3
"""Compose synchronized frame/mask sequences into a bounded review video."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MAX_VIEWS = 16
MAX_FRAMES = 10_000
MAX_OUTPUT_PIXELS = 4096 * 4096


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--tile-height", type=int, default=270)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument(
        "--mask-color-bgr",
        type=int,
        nargs=3,
        default=(40, 40, 255),
        metavar=("B", "G", "R"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report destination. Omit to write only the video.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[int, int]:
    if not 1 <= len(args.views) <= MAX_VIEWS:
        raise ValueError(f"views must contain 1..{MAX_VIEWS} names")
    if len(args.views) != len(set(args.views)):
        raise ValueError("views must be unique")
    if args.start_frame < 0:
        raise ValueError("start-frame must be nonnegative")
    if not 1 <= args.frame_count <= MAX_FRAMES:
        raise ValueError(f"frame-count must be in [1, {MAX_FRAMES}]")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if not 1 <= args.columns <= MAX_VIEWS:
        raise ValueError(f"columns must be in [1, {MAX_VIEWS}]")
    if args.tile_width <= 0 or args.tile_height <= 0:
        raise ValueError("tile dimensions must be positive")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("mask-alpha must be in [0, 1]")
    if any(value < 0 or value > 255 for value in args.mask_color_bgr):
        raise ValueError("mask color channels must be in [0, 255]")
    rows = math.ceil(len(args.views) / args.columns)
    width = args.columns * args.tile_width
    height = rows * args.tile_height
    if width * height > MAX_OUTPUT_PIXELS:
        raise ValueError("output grid exceeds the 4096x4096 pixel budget")
    if width % 2 or height % 2:
        raise ValueError("output dimensions must be even for yuv420p encoding")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("output must use the .mp4 suffix")
    return width, height


def _requested_paths(
    root: Path, views: list[str], start: int, count: int, suffix: str
) -> list[list[Path]]:
    return [
        [root / view / f"{frame_id:06d}{suffix}" for frame_id in range(start, start + count)]
        for view in views
    ]


def _validate_inputs(frame_paths: list[list[Path]], mask_paths: list[list[Path]]) -> None:
    missing = [
        str(path)
        for paths in (*frame_paths, *mask_paths)
        for path in paths
        if not path.is_file()
    ]
    if missing:
        preview = missing[:20]
        suffix = f" (and {len(missing) - 20} more)" if len(missing) > 20 else ""
        raise FileNotFoundError(f"missing synchronized inputs: {preview}{suffix}")


def _read_mask(
    path: Path, fallback_color: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    with Image.open(path) as image:
        labels = np.asarray(image)
        mode = image.mode
        if mode == "P":
            if labels.ndim != 2:
                raise ValueError(f"invalid P-mode mask shape {labels.shape}: {path}")
            palette = image.getpalette()
            if palette is None:
                raise ValueError(f"P-mode mask has no palette: {path}")
            # PIL resolves the palette indices to their declared RGB colors.
            # OpenCV/video frames are BGR, so reverse the final channel here.
            mask_bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            palette_digest = hashlib.sha256(bytes(palette)).hexdigest()
            return labels != 0, mask_bgr, mode, palette_digest
    if labels.ndim == 2:
        active = labels != 0
    elif labels.ndim == 3:
        active = np.any(labels != 0, axis=2)
    else:
        raise ValueError(f"unsupported mask array shape {labels.shape}: {path}")
    mask_bgr = np.empty((*active.shape, 3), dtype=np.uint8)
    mask_bgr[:] = np.rint(fallback_color).astype(np.uint8)
    return active, mask_bgr, mode, None


def _tile(
    frame_path: Path,
    mask_path: Path,
    *,
    size: tuple[int, int],
    alpha: float,
    color: np.ndarray,
    label: str,
) -> tuple[np.ndarray, int, str, str | None]:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"could not decode frame: {frame_path}")
    mask, mask_bgr, mask_mode, palette_digest = _read_mask(mask_path, color)
    if mask.shape != frame.shape[:2]:
        raise ValueError(
            f"frame/mask size mismatch for {frame_path.name}: "
            f"{frame.shape[:2]} vs {mask.shape}"
        )
    tile = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    active = cv2.resize(
        mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    palette_tile = cv2.resize(mask_bgr, size, interpolation=cv2.INTER_NEAREST)
    if np.any(active):
        foreground = tile[active].astype(np.float32)
        tile[active] = np.rint(
            foreground * (1.0 - alpha)
            + palette_tile[active].astype(np.float32) * alpha
        ).astype(np.uint8)
    cv2.rectangle(tile, (0, 0), (size[0], 31), (0, 0, 0), thickness=-1)
    cv2.putText(
        tile,
        label,
        (9, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile, int(np.count_nonzero(mask)), mask_mode, palette_digest


def main() -> None:
    args = _parser().parse_args()
    output_width, output_height = _validate_args(args)
    frames_root = args.frames_root.resolve()
    masks_root = args.masks_root.resolve()
    output = args.output.resolve()
    report = args.report.resolve() if args.report is not None else None
    if output.exists() or (report is not None and report.exists()):
        raise FileExistsError("output/report already exists; choose a new destination")
    frame_paths = _requested_paths(
        frames_root, args.views, args.start_frame, args.frame_count, ".jpg"
    )
    mask_paths = _requested_paths(
        masks_root, args.views, args.start_frame, args.frame_count, ".png"
    )
    _validate_inputs(frame_paths, mask_paths)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    if partial.exists():
        raise FileExistsError(f"partial output already exists: {partial}")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{output_width}x{output_height}",
        "-r",
        str(args.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    overlay_pixels = {view: 0 for view in args.views}
    mask_modes = {view: set() for view in args.views}
    palette_digests = {view: set() for view in args.views}
    rows = math.ceil(len(args.views) / args.columns)
    color = np.asarray(args.mask_color_bgr, dtype=np.float32)
    try:
        for offset in range(args.frame_count):
            frame_id = args.start_frame + offset
            grid = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            for index, view in enumerate(args.views):
                tile, pixels, mask_mode, palette_digest = _tile(
                    frame_paths[index][offset],
                    mask_paths[index][offset],
                    size=(args.tile_width, args.tile_height),
                    alpha=args.mask_alpha,
                    color=color,
                    label=f"{view}   frame {frame_id:06d}",
                )
                row, column = divmod(index, args.columns)
                y0 = row * args.tile_height
                x0 = column * args.tile_width
                grid[y0 : y0 + args.tile_height, x0 : x0 + args.tile_width] = tile
                overlay_pixels[view] += pixels
                mask_modes[view].add(mask_mode)
                if palette_digest is not None:
                    palette_digests[view].add(palette_digest)
            process.stdin.write(grid.tobytes())
            if offset == 0 or (offset + 1) % 50 == 0 or offset + 1 == args.frame_count:
                print(f"encoded {offset + 1}/{args.frame_count} frames", flush=True)
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited {return_code}: {stderr.strip()}")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        partial.unlink(missing_ok=True)
        raise

    capture = cv2.VideoCapture(str(partial))
    frames_written = 0
    while frames_written <= args.frame_count:
        ok, _frame = capture.read()
        if not ok:
            break
        frames_written += 1
    encoded_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    encoded_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if (
        frames_written != args.frame_count
        or encoded_width != output_width
        or encoded_height != output_height
        or abs(encoded_fps - args.fps) > 1e-3
    ):
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "encoded video validation failed: "
            f"frames={frames_written}, size={encoded_width}x{encoded_height}, "
            f"fps={encoded_fps}"
        )
    partial.replace(output)
    if report is not None:
        report.write_text(
            json.dumps(
                {
                "frames_root": str(frames_root),
                "masks_root": str(masks_root),
                "views": args.views,
                "frame_range": [
                    args.start_frame,
                    args.start_frame + args.frame_count - 1,
                ],
                "frames_written": frames_written,
                "fps": encoded_fps,
                "duration_seconds": frames_written / encoded_fps,
                "layout": {"columns": args.columns, "rows": rows},
                "tile_size": [args.tile_width, args.tile_height],
                "output_size": [output_width, output_height],
                "mask_alpha": args.mask_alpha,
                "mask_color_bgr": list(args.mask_color_bgr),
                "mask_color_mapping": "P-mode palette RGB converted to BGR",
                "mask_modes_by_view": {
                    view: sorted(values) for view, values in mask_modes.items()
                },
                "distinct_palette_digests_by_view": {
                    view: sorted(values)
                    for view, values in palette_digests.items()
                },
                "source_mask_pixels_by_view": overlay_pixels,
                "output": str(output),
                "size_bytes": output.stat().st_size,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/diagnostics/compose_mask_overlay_grid.py")
    main()
