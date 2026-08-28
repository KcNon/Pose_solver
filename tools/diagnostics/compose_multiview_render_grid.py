#!/usr/bin/env python
"""Compose bounded per-view render videos into a synchronized grid."""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import write_json


MAX_VIEWS = 16
MAX_EXPECTED_FRAMES = 100_000
MAX_OUTPUT_PIXELS = 4096 * 4096


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--video-name", default="overlay.mp4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--tile-height", type=int, default=270)
    args = parser.parse_args()

    views = [str(value) for value in args.views]
    if not views or len(views) > MAX_VIEWS or len(views) != len(set(views)):
        raise ValueError(f"views must contain 1..{MAX_VIEWS} unique names")
    if not 1 <= args.expected_frames <= MAX_EXPECTED_FRAMES:
        raise ValueError(
            f"expected-frames must be in [1, {MAX_EXPECTED_FRAMES}]"
        )
    if args.columns < 1 or args.columns > MAX_VIEWS:
        raise ValueError(f"columns must be in [1, {MAX_VIEWS}]")
    rows = int(math.ceil(len(views) / args.columns))
    output_size = (args.columns * args.tile_width, rows * args.tile_height)
    if min(output_size) <= 0 or output_size[0] * output_size[1] > MAX_OUTPUT_PIXELS:
        raise ValueError("requested grid output is empty or exceeds 4096x4096")

    input_root = Path(args.input_root).resolve()
    paths = [input_root / view / args.video_name for view in views]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing grid inputs: {missing}")

    fps_values = []
    frame_counts = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open grid input video: {path}")
        fps_values.append(capture.get(cv2.CAP_PROP_FPS))
        frame_counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
    if any(value <= 0 for value in fps_values):
        raise RuntimeError(f"invalid input fps values: {fps_values}")
    if max(fps_values) - min(fps_values) > 1e-3:
        raise RuntimeError(f"input videos have mismatched fps: {fps_values}")
    if any(value != args.expected_frames for value in frame_counts):
        raise RuntimeError(
            "input videos have unexpected frame counts: "
            f"{dict(zip(views, frame_counts))}"
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for reliable multi-view grids")
    filters = []
    labels = []
    layout = []
    for index, view in enumerate(views):
        escaped = view.replace("\\", "\\\\").replace(":", "\\:")
        label = f"tile{index}"
        filters.append(
            f"[{index}:v]scale={args.tile_width}:{args.tile_height}:"
            "flags=area,"
            "drawbox=x=0:y=0:w=iw:h=32:color=black:t=fill,"
            f"drawtext=text='{escaped}':x=9:y=7:fontsize=18:"
            f"fontcolor=white[{label}]"
        )
        labels.append(f"[{label}]")
        row, column = divmod(index, args.columns)
        layout.append(
            f"{column * args.tile_width}_{row * args.tile_height}"
        )
    filters.append(
        "".join(labels)
        + f"xstack=inputs={len(views)}:layout={'|'.join(layout)}:"
        + "fill=black[grid]"
    )
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        command.extend(("-i", str(path)))
    command.extend((
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[grid]",
        "-frames:v",
        str(args.expected_frames),
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
    ))
    subprocess.run(command, check=True)

    capture = cv2.VideoCapture(str(partial))
    frames_written = 0
    while frames_written < args.expected_frames:
        ok, _frame = capture.read()
        if not ok:
            break
        frames_written += 1
    extra_frame, _frame = capture.read()
    capture.release()
    if frames_written != args.expected_frames or extra_frame:
        raise RuntimeError(
            f"grid encoded {frames_written} frames; "
            f"expected exactly {args.expected_frames}"
        )

    partial.replace(output)
    report = {
        "video_name": args.video_name,
        "views": views,
        "layout": {"columns": args.columns, "rows": rows},
        "tile_size": [args.tile_width, args.tile_height],
        "output_size": list(output_size),
        "fps": float(fps_values[0]),
        "expected_frames": args.expected_frames,
        "frames_written": frames_written,
        "encoder": "ffmpeg_libx264_xstack",
        "inputs": [str(path) for path in paths],
        "output": str(output),
    }
    write_json(Path(args.report), report)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/diagnostics/compose_multiview_render_grid.py")
    main()
