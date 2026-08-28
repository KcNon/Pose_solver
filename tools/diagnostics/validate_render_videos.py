#!/usr/bin/env python
"""Decode a bounded set of render videos and verify exact stream properties."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import write_json


MAX_VIDEOS = 32
MAX_TOTAL_FRAMES = 100_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-width", type=int, required=True)
    parser.add_argument("--expected-height", type=int, required=True)
    parser.add_argument("--expected-fps", type=float, required=True)
    parser.add_argument("--fps-tolerance", type=float, default=1e-3)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    paths = [Path(value).resolve() for value in args.videos]
    if not paths or len(paths) > MAX_VIDEOS or len(paths) != len(set(paths)):
        raise ValueError(f"videos must contain 1..{MAX_VIDEOS} unique paths")
    if args.expected_frames < 1:
        raise ValueError("expected-frames must be positive")
    if args.expected_frames * len(paths) > MAX_TOTAL_FRAMES:
        raise ValueError(
            f"requested decode exceeds {MAX_TOTAL_FRAMES} total frames"
        )
    if args.expected_width < 1 or args.expected_height < 1:
        raise ValueError("expected dimensions must be positive")
    if args.expected_fps <= 0 or args.fps_tolerance < 0:
        raise ValueError("expected fps must be positive and tolerance nonnegative")

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing videos: {missing}")

    results = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = 0
        try:
            while frames <= args.expected_frames:
                ok, _frame = capture.read()
                if not ok:
                    break
                frames += 1
        finally:
            capture.release()
        errors = []
        if frames != args.expected_frames:
            errors.append(f"frames={frames}, expected={args.expected_frames}")
        if width != args.expected_width or height != args.expected_height:
            errors.append(
                f"size={width}x{height}, "
                f"expected={args.expected_width}x{args.expected_height}"
            )
        if abs(fps - args.expected_fps) > args.fps_tolerance:
            errors.append(f"fps={fps}, expected={args.expected_fps}")
        if errors:
            raise RuntimeError(f"{path}: {'; '.join(errors)}")
        results.append(
            {
                "path": str(path),
                "frames_decoded": frames,
                "width": width,
                "height": height,
                "fps": fps,
                "size_bytes": path.stat().st_size,
            }
        )
        print(f"validated {path.name}: {frames} frames", flush=True)

    write_json(
        Path(args.report),
        {
            "video_count": len(results),
            "expected_frames": args.expected_frames,
            "expected_size": [args.expected_width, args.expected_height],
            "expected_fps": args.expected_fps,
            "videos": results,
        },
    )


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/diagnostics/validate_render_videos.py")
    main()
