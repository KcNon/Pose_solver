#!/usr/bin/env python3
"""Render the complete 05-style force-driven PhysX trajectory video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ASSET_ROOT = (
    PROJECT_ROOT / "experiments/data_1/simulation_assets_scale_calibrated"
)
DEFAULT_RUNTIME_ROOT = (
    PROJECT_ROOT / "experiments/data_1/isaac_runtime_scale_calibrated"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments/data_1/"
    "isaac_video_scale_calibrated_assembly_bounce_fixed_final"
)


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
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Run CPU PhysX and write JSON/USD without loading a renderer.",
    )
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

    app_config = {
        "headless": True,
        "width": args.width,
        "height": args.height,
        "multi_gpu": False,
        "sync_loads": True,
        "limit_cpu_threads": 16,
    }
    if args.no_capture:
        required_extensions = [
            "omni.physx",
            "omni.physx.tensors",
            "omni.physx.fabric",
            "omni.warp.core",
            "usdrt.scenegraph",
            "omni.kit.telemetry",
            "omni.kit.loop",
            "omni.kit.usd.mdl",
            "omni.usd.metrics.assembler.ui",
            "omni.hydra.usdrt_delegate",
            "isaacsim.core.experimental.prims",
            "isaacsim.asset.importer.urdf",
        ]
        app_config["extra_args"] = [
            "--/app/exts/folders=['apps','exts','extscache','extsUser','extsDeprecated']",
            *[
                value
                for extension in required_extensions
                for value in ("--enable", extension)
            ],
        ]
        app = SimulationApp(app_config, experience=None)
    else:
        app_config["renderer"] = "RayTracedLighting"
        app = SimulationApp(app_config)
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
        if report["output_video"]:
            print(f"Complete physics video: {report['output_video']}", flush=True)
        else:
            print("Physics-only report completed (capture disabled).", flush=True)
        print(
            f"Frames: {report['frame_count']} | "
            f"duration: {report['duration_s']:.2f}s | "
            f"first blocked: {report['first_blocked_frame']} | "
            "validation: "
            f"{'PASS' if report['physics_validation']['passed'] else 'FAIL'}",
            flush=True,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
