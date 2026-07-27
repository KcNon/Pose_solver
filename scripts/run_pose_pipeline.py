#!/usr/bin/env python
"""Run the reusable pose stages with resumable file-level checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.calibration_cache import build_calibration_fingerprint
from common.pose_autoconfig import resolve_pose_config
from common.pose_config import validate_pose_config
from common.stage_cache import (
    checkpoint_matches,
    stage_fingerprint,
    write_checkpoint,
)


def _run(
    command: list[str],
    expected: Path,
    force: bool,
    *,
    content_files: tuple[Path, ...] = (),
    stat_paths: tuple[Path, ...] = (),
) -> None:
    fingerprint = stage_fingerprint(
        command=command,
        content_files=content_files,
        stat_paths=stat_paths,
    )
    if not force and checkpoint_matches(expected, fingerprint):
        print(f"[resume] {expected}", flush=True)
        return
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not expected.exists():
        raise RuntimeError(
            f"stage completed without expected output: {expected}"
        )
    write_checkpoint(expected, fingerprint)


def _calibration_cache_matches(config: dict, calibration_path: Path) -> bool:
    """Return true only when an existing calibration matches its inputs.

    The solver deliberately refuses stale calibration files.  Checking the
    same fingerprint here keeps orchestration resumable without blindly
    passing ``--reuse-calibration`` merely because a file happens to exist.
    """

    if not calibration_path.exists():
        return False
    cached = load_json(calibration_path).get("input_fingerprint", {}).get(
        "sha256"
    )
    cloud_root = Path(config.get(
        "point_cloud_root",
        Path(config["output_root"]) / "parts_ply" / config["recon_backend"],
    ))
    current = build_calibration_fingerprint(
        config,
        cloud_root=cloud_root,
        mesh_dir=Path(config["mesh_dir"]),
    ).get("sha256")
    return bool(cached and cached == current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "preflight",
            "states",
            "solve",
            "refine",
            "constrain",
            "review",
            "render",
        ),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--render-view")
    args = parser.parse_args()

    source_config_path = Path(args.config).resolve()
    source_config = load_json(source_config_path)
    automation_enabled = bool(
        source_config.get("automation", {}).get("enabled", False)
    )
    config = validate_pose_config(
        source_config,
        check_paths=True,
        allow_auto=automation_enabled,
    )
    config_path = source_config_path
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

    state_path = output / "diagnostics" / "part_states.json"
    if args.stage in {"all", "states"} or (
        automation_enabled
        and args.stage in {"solve", "refine", "constrain", "review", "render"}
        and not state_path.exists()
    ):
        _run(
            [
                python,
                "-u",
                "tools/diagnostics/detect_part_states.py",
                "--config",
                str(source_config_path),
            ],
            state_path,
            args.force,
            content_files=(source_config_path,),
            stat_paths=(
                Path(config["masks_dir"]),
                Path(config.get("point_cloud_root", output / "parts_ply")),
            ),
        )
    if automation_enabled:
        if not state_path.exists():
            raise RuntimeError(
                "automatic pose config requires part_states.json"
            )
        resolved, audit = resolve_pose_config(
            source_config,
            load_json(state_path),
        )
        resolved_path = output / "automation" / "resolved_pose_config.json"
        write_json(resolved_path, resolved)
        write_json(
            output / "automation" / "pose_autoconfig_report.json",
            audit,
        )
        config_path = resolved_path
        config = validate_pose_config(resolved, check_paths=True)
        print(f"resolved pose config -> {resolved_path}", flush=True)
    if args.stage in {"all", "solve"}:
        command = [
            python,
            "-u",
            "tools/stages/pose/solve_multiview_pose.py",
            "--config",
            str(config_path),
        ]
        calibration = output / "pose" / "calibration.json"
        reuse_calibration = (
            not args.force_calibration
            and _calibration_cache_matches(config, calibration)
        )
        if reuse_calibration:
            command.append("--reuse-calibration")
        elif calibration.exists() and not args.force_calibration:
            print(
                f"[invalidate] stale calibration cache: {calibration}",
                flush=True,
            )
        _run(
            command,
            output / "pose" / "trajectory.json",
            args.force or args.force_calibration,
            content_files=(
                config_path,
                state_path,
                Path(config.get("depth_gauge_path", config_path)),
            ),
            stat_paths=(
                Path(config["masks_dir"]),
                Path(config["mesh_dir"]),
                Path(config.get("point_cloud_root", output / "parts_ply")),
            ),
        )
    baseline_trajectory = output / "pose" / "trajectory.json"
    refined_trajectory = output / "pose" / "trajectory_render_refined.json"
    refinement_enabled = bool(
        config.get("render_loss_refinement", {}).get("enabled", False)
    )
    if args.stage in {"all", "refine"} and refinement_enabled:
        _run(
            [
                python,
                "-u",
                "tools/stages/pose/refine_pose_render_loss.py",
                "--config",
                str(config_path),
                "--trajectory",
                str(baseline_trajectory),
                "--output-trajectory",
                str(refined_trajectory),
                "--report",
                str(output / "diagnostics" / "render_loss_refinement.json"),
            ],
            refined_trajectory,
            args.force,
            content_files=(config_path, baseline_trajectory),
            stat_paths=(Path(config["masks_dir"]),),
        )
    render_refined_trajectory = (
        refined_trajectory
        if refinement_enabled and refined_trajectory.exists()
        else baseline_trajectory
    )
    constrained_trajectory = (
        output / "pose" / "trajectory_collision_refined.json"
    )
    constraints_enabled = bool(
        config.get("trajectory_constraints", {}).get("enabled", False)
    )
    if args.stage in {"all", "constrain"} and constraints_enabled:
        _run(
            [
                python,
                "-u",
                "tools/stages/pose/enforce_trajectory_constraints.py",
                "--config",
                str(config_path),
                "--trajectory",
                str(render_refined_trajectory),
                "--output-trajectory",
                str(constrained_trajectory),
                "--report",
                str(output / "diagnostics" / "trajectory_constraints.json"),
            ],
            constrained_trajectory,
            args.force,
            content_files=(
                config_path,
                render_refined_trajectory,
                Path(
                    config["trajectory_constraints"].get(
                        "geometry_proxy_config",
                        config["trajectory_constraints"].get(
                            "collision_proxy_config"
                        ),
                    )
                ),
            ),
            stat_paths=(Path(config["masks_dir"]),),
        )
    active_trajectory = (
        constrained_trajectory
        if constraints_enabled and constrained_trajectory.exists()
        else render_refined_trajectory
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
                str(active_trajectory),
                "--output-root",
                str(output),
            ],
            output / "diagnostics" / "multiview_metrics.json",
            args.force,
            content_files=(config_path, active_trajectory),
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
                str(active_trajectory),
                "--output-root",
                str(output),
                "--view",
                view,
            ],
            output / "render" / view / "overlay.mp4",
            args.force,
            content_files=(config_path, active_trajectory),
        )
    print(f"complete: {active_trajectory}", flush=True)


if __name__ == "__main__":
    main()
