#!/usr/bin/env python3
"""Run the formal Isaac Sim trajectory replay and insertion validation.

Use Isaac Sim's ``python.sh``. Asset conversion is handled separately by
``export_simulation_assets.py``.
"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--runtime-output-dir",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
    )
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-drop", action="store_true")
    parser.add_argument(
        "--place-release-only",
        action="store_true",
        help=(
            "Disable replay, controllers, preload, and fixed joints; place the "
            "inserted part just above its corrected assembly pose and release it."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Run the standardized assembly validator from the unmodified visual "
            "assembly pose. No target correction, axis alignment, controller, "
            "preload, fixed joint, or trajectory rewrite is allowed."
        ),
    )
    parser.add_argument(
        "--physics-refine-only",
        action="store_true",
        help=(
            "Search a bounded neighborhood around the visual assembly pose, "
            "using contact and a flexible dip-tube proxy to write a physically "
            "projected trajectory. ICP observations are not changed in place."
        ),
    )
    parser.add_argument(
        "--refinement-video-fps",
        type=float,
        default=5.0,
        help="Capture rate for the selected physical-refinement rollout.",
    )
    parser.add_argument("--trial-limit", type=int)
    parser.add_argument("--align-insert-up-axis", action="store_true")
    parser.add_argument(
        "--capture", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclusive_modes = sum(
        bool(value)
        for value in (
            args.validate_only,
            args.place_release_only,
            args.physics_refine_only,
        )
    )
    if exclusive_modes > 1:
        raise ValueError(
            "choose only one of --validate-only, --place-release-only, and "
            "--physics-refine-only"
        )
    if args.refinement_video_fps <= 0.0:
        raise ValueError("--refinement-video-fps must be positive")
    asset_root = args.asset_root.resolve()
    runtime_root = args.runtime_output_dir.resolve()
    manifest_path = asset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Asset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.validate_only and manifest.get("assembly_interface") is None:
        raise ValueError(
            "--validate-only requires a declared assembly_interface; generic "
            "mesh geometry is insufficient to infer a functional connector"
        )

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
        from common.isaac_runtime import run_insertion

        run_insertion(args, app, asset_root, runtime_root, manifest)
    finally:
        app.close()


if __name__ == "__main__":
    main()
