#!/usr/bin/env python
"""Run the reusable pose stages with resumable file-level checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json
from common.pose_config import validate_pose_config


def _run(command: list[str], expected: Path, force: bool) -> None:
    if expected.exists() and not force:
        print(f"[resume] {expected}", flush=True)
        return
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not expected.exists():
        raise RuntimeError(
            f"stage completed without expected output: {expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "preflight", "states", "solve", "review", "render"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--render-view")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = validate_pose_config(
        load_json(config_path),
        check_paths=True,
    )
    output = Path(config["output_root"])
    python = sys.executable
    print(
        f"preflight: {len(config['views'])} views, "
        f"{len(config['parts'])} parts, "
        f"frames {config['frames']['start']}..{config['frames']['end']}",
        flush=True,
    )
    if args.stage == "preflight":
        return

    if args.stage in {"all", "states"}:
        _run(
            [
                python,
                "-u",
                "tools/diagnostics/detect_part_states.py",
                "--config",
                str(config_path),
            ],
            output / "diagnostics" / "part_states.json",
            args.force,
        )
    if args.stage in {"all", "solve"}:
        command = [
            python,
            "-u",
            "tools/stages/pose/solve_multiview_pose.py",
            "--config",
            str(config_path),
        ]
        calibration = output / "pose" / "calibration.json"
        if calibration.exists() and not args.force_calibration:
            command.append("--reuse-calibration")
        _run(
            command,
            output / "pose" / "trajectory.json",
            args.force or args.force_calibration,
        )
    if args.stage in {"all", "review"} and not args.skip_review:
        _run(
            [
                python,
                "-u",
                "tools/diagnostics/export_multiview_pose_review.py",
                "--config",
                str(config_path),
                "--trajectory",
                str(output / "pose" / "trajectory.json"),
                "--output-root",
                str(output),
            ],
            output / "diagnostics" / "multiview_metrics.json",
            args.force,
        )
    if args.stage in {"all", "render"} and not args.skip_render:
        view = (
            args.render_view
            or config.get("render", {}).get("primary_view")
            or config["views"][0]
        )
        if view not in config["views"]:
            raise ValueError(f"unknown render view: {view}")
        _run(
            [
                python,
                "-u",
                "scripts/render_multiview_pose.py",
                "--config",
                str(config_path),
                "--trajectory",
                str(output / "pose" / "trajectory.json"),
                "--output-root",
                str(output),
                "--view",
                view,
            ],
            output / "render" / view / "overlay.mp4",
            args.force,
        )
    print(f"complete: {output / 'pose' / 'trajectory.json'}", flush=True)


if __name__ == "__main__":
    main()
