#!/usr/bin/env python
"""Run FoundationPose-hybrid proposal, six-view evaluation and safe selection."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], expected: Path, force: bool) -> None:
    if expected.exists() and not force:
        print(f"[resume] {expected}", flush=True)
        return
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not expected.exists():
        raise RuntimeError(f"stage completed without expected output: {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pose_multiview_111_v4.json")
    parser.add_argument("--quality-config", default="configs/pose_multiview_111_quality.json")
    parser.add_argument("--baseline", default=(
        "experiments/three_part_multiview_111f/outputs_v6_global_pose/accepted_final"))
    parser.add_argument("--output-root", default=(
        "experiments/three_part_multiview_111f/outputs_v7_foundationpose_hybrid"))
    parser.add_argument("--spot-env", default="spot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    config = (ROOT / args.config).resolve()
    quality = (ROOT / args.quality_config).resolve()
    baseline = (ROOT / args.baseline).resolve()
    output = (ROOT / args.output_root).resolve()
    candidates = output / "candidates.json"
    evaluated = output / "evaluated"
    final = output / "accepted_final"

    run([
        "conda", "run", "-n", args.spot_env, "python",
        "scripts/generate_foundationpose_hybrid_candidates.py",
        "--config", str(quality), "--trajectory", str(baseline / "pose/trajectory.json"),
        "--output", str(candidates),
    ], candidates, args.force)
    run([
        python, "scripts/evaluate_foundationpose_hybrid.py",
        "--config", str(config), "--quality-config", str(quality),
        "--candidates", str(candidates), "--output-root", str(evaluated),
    ], evaluated / "pose/trajectory.json", args.force)
    run([
        python, "scripts/export_multiview_pose_review.py",
        "--config", str(config), "--trajectory", str(evaluated / "pose/trajectory.json"),
        "--output-root", str(evaluated),
    ], evaluated / "diagnostics/multiview_metrics.json", args.force)
    run([
        python, "scripts/select_pose_multimetric_final.py",
        "--config", str(config), "--quality-config", str(quality),
        "--baseline", str(baseline / "pose/trajectory.json"),
        "--baseline-metrics", str(baseline / "diagnostics/multiview_metrics.json"),
        "--candidate", str(evaluated / "pose/trajectory.json"),
        "--candidate-metrics", str(evaluated / "diagnostics/multiview_metrics.json"),
        "--output-root", str(final),
    ], final / "pose/trajectory.json", args.force)
    print(f"complete: {final / 'pose/trajectory.json'}", flush=True)


if __name__ == "__main__":
    main()
