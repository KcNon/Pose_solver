#!/usr/bin/env python3
"""Render the complete 05-style force-driven PhysX trajectory video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "experiments/data_1/simulation_assets"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "experiments/data_1/isaac_runtime_proxy_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments/data_1/isaac_video_physics_complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--controller-frequency", type=float, default=16.0)
    parser.add_argument("--blocked-error-m", type=float, default=0.02)
    parser.add_argument("--rt-subframes", type=int, default=1)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.resolve()
    runtime_root = args.runtime_root.resolve()
    output_root = args.output_dir.resolve()
    manifest = json.loads(
        (asset_root / "manifest.json").read_text(encoding="utf-8")
    )
    trajectory_path = PROJECT_ROOT / manifest["inputs"]["trajectory"]
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
        from common.isaac_physics_video import render_complete_physics_video

        report = render_complete_physics_video(
            args,
            app,
            asset_root,
            runtime_root,
            output_root,
            manifest,
            trajectory,
        )
        print(f"Complete physics video: {report['output_video']}", flush=True)
        print(
            f"Frames: {report['frame_count']} | "
            f"duration: {report['duration_s']:.2f}s | "
            f"first blocked: {report['first_blocked_frame']}",
            flush=True,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
