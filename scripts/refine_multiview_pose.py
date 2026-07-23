#!/usr/bin/env python
"""Run the complete V6 multi-view pose refinement and acceptance pipeline."""
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
        "experiments/three_part_multiview_111f/outputs_v4_lid_se3"))
    parser.add_argument("--output-root", default=(
        "experiments/three_part_multiview_111f/outputs_v6_global_pose"))
    parser.add_argument("--force", action="store_true", help="recompute completed stages")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    config = str((ROOT / args.config).resolve())
    quality = str((ROOT / args.quality_config).resolve())
    baseline = (ROOT / args.baseline).resolve()
    output = (ROOT / args.output_root).resolve()
    body = output / "body_global"
    inner = output / "body_inner"
    lid = output / "lid_tilt"
    final = output / "accepted_final"

    run([python, "scripts/refine_body_global_yaw.py", "--config", config,
         "--quality-config", quality, "--trajectory", str(baseline / "pose/trajectory.json"),
         "--output-root", str(body)], body / "pose/trajectory.json", args.force)
    run([python, "scripts/refine_inner_assembly_prior.py", "--config", config,
         "--quality-config", quality, "--baseline", str(baseline / "pose/trajectory.json"),
         "--trajectory", str(body / "pose/trajectory.json"), "--output-root", str(inner)],
        inner / "pose/trajectory.json", args.force)
    run([python, "scripts/refine_lid_multiview_se3.py", "--config", config,
         "--trajectory", str(inner / "pose/trajectory.json"), "--output-root", str(lid)],
        lid / "pose/trajectory.json", args.force)
    run([python, "scripts/export_multiview_pose_review.py", "--config", config,
         "--trajectory", str(lid / "pose/trajectory.json"), "--output-root", str(lid)],
        lid / "diagnostics/multiview_metrics.json", args.force)
    run([python, "scripts/select_pose_multimetric_final.py", "--config", config,
         "--quality-config", quality, "--baseline", str(baseline / "pose/trajectory.json"),
         "--baseline-metrics", str(baseline / "diagnostics/multiview_metrics.json"),
         "--candidate", str(lid / "pose/trajectory.json"), "--candidate-metrics",
         str(lid / "diagnostics/multiview_metrics.json"), "--output-root", str(final)],
        final / "pose/trajectory.json", args.force)
    if not args.skip_render:
        run([python, "scripts/render_multiview_pose.py", "--config", config,
             "--trajectory", str(final / "pose/trajectory.json"), "--output-root", str(final),
             "--width", "640", "--height", "360"],
            final / "render/2-3/mesh_axes.mp4", args.force)
    print(f"complete: {final / 'pose/trajectory.json'}")


if __name__ == "__main__":
    main()

