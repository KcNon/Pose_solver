#!/usr/bin/env python3
"""Verify frozen-pose scale candidates with Isaac PhysX contact manifolds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--scale-report", required=True, type=Path)
    parser.add_argument("--runtime-output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    runtime_root = (
        args.runtime_output_dir or asset_root
    ).resolve()
    manifest = json.loads(
        (asset_root / "manifest.json").read_text(encoding="utf-8")
    )
    scale_report = json.loads(
        args.scale_report.resolve().read_text(encoding="utf-8")
    )

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting",
            "multi_gpu": False,
            "sync_loads": True,
            "limit_cpu_threads": 16,
        }
    )
    try:
        from common.isaac_scale_query import run_frozen_scale_queries

        run_frozen_scale_queries(
            args,
            app,
            asset_root,
            runtime_root,
            manifest,
            scale_report,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
