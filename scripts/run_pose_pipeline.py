#!/usr/bin/env python
"""Internal pose adapter used by :mod:`pose_solver`."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.calibration_cache import build_calibration_fingerprint
from common.pose_autoconfig import resolve_pose_config
from common.pose_config import validate_pose_config
from common.support_plane import validate_part_support_window
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


def _stage_is_current(
    command: list[str],
    expected: Path,
    *,
    content_files: tuple[Path, ...] = (),
    stat_paths: tuple[Path, ...] = (),
) -> bool:
    """Return whether ``expected`` was produced from the current inputs.

    Merely checking that an optional downstream trajectory exists can select
    output from a previous solve.  Reusing the exact stage fingerprint here
    makes trajectory selection obey the same freshness rule as stage resume.
    """

    return checkpoint_matches(
        expected,
        stage_fingerprint(
            command=command,
            content_files=content_files,
            stat_paths=stat_paths,
        ),
    )


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


def main(argv: Sequence[str] | None = None) -> Path | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-root",
        help="isolated output root; writes a resolved source copy there",
    )
    parser.add_argument(
        "--point-cloud-root",
        help=(
            "isolated point-cloud override; requires --output-root so an A/B "
            "run cannot overwrite the source experiment"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "preflight",
            "states",
            "solve",
            "refine",
            "constrain",
            "stabilize",
            "review",
            "render",
        ),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument(
        "--calibration",
        help=(
            "explicit calibration.json for an isolated tracking A/B run; "
            "the solver still validates its input fingerprint"
        ),
    )
    parser.add_argument(
        "--force-reuse-calibration",
        action="store_true",
        help=(
            "intentionally reuse --calibration after point-cloud inputs "
            "change; use only when both clouds share a verified world rig"
        ),
    )
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--render-view")
    args = parser.parse_args(argv)

    if args.point_cloud_root and not args.output_root:
        parser.error("--point-cloud-root requires --output-root")
    if args.force_reuse_calibration and not args.calibration:
        parser.error("--force-reuse-calibration requires --calibration")

    source_config_path = Path(args.config).resolve()
    source_config = load_json(source_config_path)
    if args.output_root:
        source_config = dict(source_config)
        source_config["output_root"] = str(Path(args.output_root).resolve())
        if args.point_cloud_root:
            source_config["point_cloud_root"] = str(
                Path(args.point_cloud_root).resolve()
            )
            source_config["point_cloud_variant"] = (
                Path(args.point_cloud_root).resolve().name
            )
        source_config_path = (
            Path(source_config["output_root"])
            / "runtime"
            / "source_pose_config.json"
        )
        write_json(source_config_path, source_config)
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
        return None

    state_path = output / "diagnostics" / "part_states.json"
    state_command = [
        python,
        "-u",
        "tools/diagnostics/detect_part_states.py",
        "--config",
        str(source_config_path),
    ]
    state_option_names = {
        "minimum_mask_pixels": "min-px",
        "displacement_enter_px": "disp-hi",
        "displacement_exit_px": "disp-lo",
        "displacement_force_enter_px": "disp-force-hi",
        "area_enter_log_ratio": "area-hi",
        "area_exit_log_ratio": "area-lo",
        "area_force_enter_log_ratio": "area-force-hi",
        "surface_enter_mm": "surf-hi-mm",
        "surface_exit_mm": "surf-lo-mm",
        "surface_force_min_mm": "surf-force-min-mm",
        "enter_dwell_frames": "dwell-on",
        "exit_dwell_frames": "dwell-off",
        "motion_lag_frames": "motion-lag",
        "occlusion_ratio": "occlusion-ratio",
        "assembled_tolerance_m": "assembled-tol-m",
    }
    for key, option in state_option_names.items():
        value = source_config.get("state_detection", {}).get(key)
        if value is not None:
            state_command.extend((f"--{option}", str(value)))
    if args.stage in {"all", "states"} or (
        automation_enabled
        and args.stage in {
            "solve",
            "refine",
            "constrain",
            "stabilize",
            "review",
            "render",
        }
    ):
        _run(
            state_command,
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
        support_settings = config.get("automation", {})
        reference_report = audit["parts"][config["reference_part"]].get(
            "stable_reference_window"
        )
        if (
            support_settings.get("validate_table_support", True)
            and reference_report is not None
        ):
            cloud_root = Path(config.get(
                "point_cloud_root",
                output / "parts_ply" / config["recon_backend"],
            ))
            support = validate_part_support_window(
                config,
                part=config["reference_part"],
                frames=[int(value) for value in reference_report["window"]],
                cloud_root=cloud_root,
                representative_frame=int(
                    reference_report["representative_frame"]
                ),
                maximum_contact_gap_m=float(
                    support_settings.get("maximum_table_contact_gap_m", 0.05)
                ),
                maximum_gap_mad_m=float(
                    support_settings.get("maximum_table_gap_mad_m", 0.015)
                ),
            )
            write_json(
                output / "automation" / "stable_reference_support.json",
                support,
            )
            if support.get("table_plane", {}).get("accepted", False):
                config["support_plane"] = support["table_plane"]
                resolved["support_plane"] = support["table_plane"]
                write_json(resolved_path, resolved)
            if (
                support_settings.get("require_table_support", True)
                and not support.get("accepted", False)
            ):
                raise RuntimeError(
                    "stable reference window failed table-support validation: "
                    f"{support.get('reason')}"
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
        calibration_source = (
            Path(args.calibration).resolve() if args.calibration else None
        )
        if calibration_source is not None:
            command.extend(("--calibration", str(calibration_source)))
            if args.force_reuse_calibration:
                command.append("--force-reuse-calibration")
        else:
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
                *((calibration_source,) if calibration_source else ()),
                *(
                    (Path(config["depth_gauge_path"]),)
                    if config.get("depth_gauge_path")
                    else ()
                ),
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
    refinement_command = [
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
    ]
    refinement_content_files = (config_path, baseline_trajectory)
    refinement_stat_paths = (Path(config["masks_dir"]),)
    if args.stage in {"all", "refine"} and refinement_enabled:
        _run(
            refinement_command,
            refined_trajectory,
            args.force,
            content_files=refinement_content_files,
            stat_paths=refinement_stat_paths,
        )
    render_refined_trajectory = (
        refined_trajectory
        if (
            refinement_enabled
            and _stage_is_current(
                refinement_command,
                refined_trajectory,
                content_files=refinement_content_files,
                stat_paths=refinement_stat_paths,
            )
        )
        else baseline_trajectory
    )
    coaxial_projected_trajectory = (
        output / "pose" / "trajectory_coaxial_projected.json"
    )
    coaxial_projection_enabled = any(
        bool(
            part_settings.get("coaxial_constraint", {}).get(
                "enabled", False
            )
            and part_settings.get("coaxial_constraint", {}).get(
                "final_project_all_frames", True
            )
        )
        for part_settings in config.get("render_loss_refinement", {})
        .get("parts", {})
        .values()
    )
    coaxial_projection_command = [
        python,
        "-u",
        "tools/stages/pose/project_coaxial_trajectory.py",
        "--config",
        str(config_path),
        "--trajectory",
        str(render_refined_trajectory),
        "--output-trajectory",
        str(coaxial_projected_trajectory),
        "--report",
        str(output / "diagnostics" / "coaxial_projection.json"),
    ]
    coaxial_projection_content_files = (
        config_path,
        render_refined_trajectory,
    )
    if args.stage in {"all", "refine"} and coaxial_projection_enabled:
        _run(
            coaxial_projection_command,
            coaxial_projected_trajectory,
            args.force,
            content_files=coaxial_projection_content_files,
        )
    if (
        coaxial_projection_enabled
        and _stage_is_current(
            coaxial_projection_command,
            coaxial_projected_trajectory,
            content_files=coaxial_projection_content_files,
        )
    ):
        render_refined_trajectory = coaxial_projected_trajectory
    observed_assembly_trajectory = (
        output / "pose" / "trajectory_observed_assembly.json"
    )
    observed_assembly_enabled = bool(
        config.get("observed_assembly_regularization", {}).get(
            "enabled", False
        )
    )
    observed_assembly_command = [
        python,
        "-u",
        "tools/stages/pose/regularize_observed_assembly.py",
        "--config",
        str(config_path),
        "--trajectory",
        str(render_refined_trajectory),
        "--output-trajectory",
        str(observed_assembly_trajectory),
        "--report",
        str(output / "diagnostics" / "observed_assembly_regularization.json"),
    ]
    observed_assembly_content_files = (
        config_path,
        render_refined_trajectory,
    )
    observed_assembly_stat_paths = (Path(config["masks_dir"]),)
    if args.stage in {"all", "refine"} and observed_assembly_enabled:
        _run(
            observed_assembly_command,
            observed_assembly_trajectory,
            args.force,
            content_files=observed_assembly_content_files,
            stat_paths=observed_assembly_stat_paths,
        )
    if (
        observed_assembly_enabled
        and _stage_is_current(
            observed_assembly_command,
            observed_assembly_trajectory,
            content_files=observed_assembly_content_files,
            stat_paths=observed_assembly_stat_paths,
        )
    ):
        render_refined_trajectory = observed_assembly_trajectory
    constrained_trajectory = (
        output / "pose" / "trajectory_collision_refined.json"
    )
    constraints_enabled = bool(
        config.get("trajectory_constraints", {}).get("enabled", False)
    )
    constraint_command = [
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
    ]
    constraint_proxy = config.get("trajectory_constraints", {}).get(
        "geometry_proxy_config",
        config.get("trajectory_constraints", {}).get(
            "collision_proxy_config"
        ),
    )
    constraint_content_files = (
        config_path,
        render_refined_trajectory,
        *((Path(constraint_proxy),) if constraint_proxy else ()),
    )
    constraint_stat_paths = (Path(config["masks_dir"]),)
    if args.stage in {"all", "constrain"} and constraints_enabled:
        _run(
            constraint_command,
            constrained_trajectory,
            args.force,
            content_files=constraint_content_files,
            stat_paths=constraint_stat_paths,
        )
    pre_stabilized_trajectory = (
        constrained_trajectory
        if (
            constraints_enabled
            and _stage_is_current(
                constraint_command,
                constrained_trajectory,
                content_files=constraint_content_files,
                stat_paths=constraint_stat_paths,
            )
        )
        else render_refined_trajectory
    )
    stabilized_trajectory = output / "pose" / "trajectory_final.json"
    stabilization_enabled = bool(
        config.get("static_pose_consensus", {}).get("parts")
    )
    stabilization_command = [
        python,
        "-u",
        "tools/stages/pose/stabilize_static_pose.py",
        "--config",
        str(config_path),
        "--trajectory",
        str(pre_stabilized_trajectory),
        "--output-trajectory",
        str(stabilized_trajectory),
        "--report",
        str(output / "diagnostics" / "static_pose_consensus.json"),
    ]
    stabilization_content_files = (config_path, pre_stabilized_trajectory)
    if args.stage in {"all", "stabilize"} and stabilization_enabled:
        _run(
            stabilization_command,
            stabilized_trajectory,
            args.force,
            content_files=stabilization_content_files,
        )
    stabilized_active_trajectory = (
        stabilized_trajectory
        if (
            stabilization_enabled
            and _stage_is_current(
                stabilization_command,
                stabilized_trajectory,
                content_files=stabilization_content_files,
            )
        )
        else pre_stabilized_trajectory
    )
    active_trajectory = stabilized_active_trajectory
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
                "tools/stages/pose/render_multiview_pose.py",
                "--config",
                str(config_path),
                "--trajectory",
                str(active_trajectory),
                "--output-root",
                str(output),
                "--view",
                view,
                "--width",
                str(int(config.get("render", {}).get("width", 1280))),
                "--height",
                str(int(config.get("render", {}).get("height", 720))),
            ],
            output / "render" / view / "overlay.mp4",
            args.force,
            content_files=(config_path, active_trajectory),
        )
    print(f"complete: {active_trajectory}", flush=True)
    return active_trajectory
