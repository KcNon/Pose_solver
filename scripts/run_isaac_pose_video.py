#!/usr/bin/env python3
"""Render a solved trajectory as a kinematic multi-view Isaac Sim video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--rt-subframes", type=int, default=1)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.resolve()
    runtime_root = args.runtime_root.resolve()
    output_root = args.output_dir.resolve()
    manifest = json.loads(
        (asset_root / "manifest.json").read_text(encoding="utf-8")
    )
    trajectory_value = Path(manifest["inputs"]["trajectory"])
    trajectory_path = (
        trajectory_value
        if trajectory_value.is_absolute()
        else PROJECT_ROOT / trajectory_value
    )
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": args.width,
            "height": args.height,
            "renderer": "RayTracedLighting",
            "multi_gpu": False,
            "sync_loads": True,
            "limit_cpu_threads": 16,
        }
    )
    try:
        from common.isaac_video import render_complete_pose_video

        report = render_complete_pose_video(
            args,
            app,
            asset_root,
            runtime_root,
            output_root,
            manifest,
            trajectory,
        )
        print(f"Isaac pose video: {report['output_video']}", flush=True)
        print(
            f"Frames: {report['frame_count']} | "
            f"duration: {report['duration_s']:.2f}s",
            flush=True,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
