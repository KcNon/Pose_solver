#!/usr/bin/env python
"""Refine assembled inner_pot inside a tight body-relative prior basin."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.gicp import multiscale_gicp, voxel_unique
from common.io_utils import load_json, write_json
from common.mesh_align import read_ply_xyz
from common.pose_refinement import cap_pose_delta, cloud_metrics, sample_canonical
from common.pose_transforms import similarity_from_rigid
from common.trajectory_io import write_trajectory_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pose_multiview_111_v4.json"))
    parser.add_argument("--quality-config", default=str(
        ROOT / "configs/pose_multiview_111_quality.json"))
    parser.add_argument("--baseline", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v4_lid_se3/pose/trajectory.json"))
    parser.add_argument("--trajectory", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v6_global_pose/body_global/pose/trajectory.json"))
    parser.add_argument("--output-root", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v6_global_pose/body_inner"))
    parser.add_argument("--assembled-start", type=int, default=40)
    parser.add_argument("--cloud-frames", type=int, nargs="*", default=[40, 50, 60, 70, 80, 90, 100])
    parser.add_argument("--max-translation-mm", type=float, default=12.0)
    parser.add_argument("--max-rotation-deg", type=float, default=12.0)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    quality = load_json(Path(args.quality_config))
    baseline = load_json(Path(args.baseline))
    source_path = Path(args.trajectory).resolve()
    source = load_json(source_path)
    output = Path(args.output_root).resolve()
    part = "inner_pot"
    scale = float(source["scales"][part])
    origin = np.asarray(source["raw_mesh_origins"][part], float)
    mesh = trimesh.load(Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh")
    canonical = sample_canonical(mesh, scale, origin, 30000, 3301)

    clouds, used = [], []
    root = Path(quality["point_cloud_root"])
    for frame in args.cloud_frames:
        path = root / f"{frame:06d}" / f"{part}.ply"
        if path.exists():
            points = read_ply_xyz(str(path))
            if len(points) >= 100:
                clouds.append(points); used.append(frame)
    observed = voxel_unique(np.concatenate(clouds), 0.003)
    maximum = int(quality["registration"]["max_points"])
    if len(observed) > maximum:
        observed = observed[np.random.default_rng(3302).choice(
            len(observed), maximum, replace=False)]
    observed = np.ascontiguousarray(observed, dtype=np.float64)

    # The accepted assembled relationship is the prior, independent of any
    # global correction to the body world pose.
    base_key = f"{args.assembled_start:06d}"
    T_WB_old = np.asarray(baseline["frames"][base_key]["parts"]["body"]["T_world_from_part"], float)
    T_WI_old = np.asarray(baseline["frames"][base_key]["parts"][part]["T_world_from_part"], float)
    prior = np.linalg.inv(T_WB_old) @ T_WI_old
    T_WB = np.asarray(source["frames"][base_key]["parts"]["body"]["T_world_from_part"], float)
    prior_world = T_WB @ prior

    world_to_part, gicp = multiscale_gicp(
        observed, canonical, np.linalg.inv(prior_world), quality["registration"])
    raw_world = np.linalg.inv(world_to_part)
    raw_relative = np.linalg.inv(T_WB) @ raw_world
    delta = np.linalg.inv(prior) @ raw_relative
    bounded_delta, cap_report = cap_pose_delta(
        delta, args.max_translation_mm / 1000.0, args.max_rotation_deg)
    refined_relative = prior @ bounded_delta
    refined_world = T_WB @ refined_relative
    prior_metrics = cloud_metrics(observed, canonical, prior_world)
    refined_metrics = cloud_metrics(observed, canonical, refined_world)
    accepted = (refined_metrics["fitness_8mm"] >= prior_metrics["fitness_8mm"]
                and refined_metrics["median_nn_m"] <= prior_metrics["median_nn_m"])
    selected_world = refined_world if accepted else prior_world
    selected_relative = refined_relative if accepted else prior

    result = copy.deepcopy(source)
    result.setdefault("provenance", {})["inner_assembly_prior"] = str(
        output / "diagnostics/inner_assembly_prior.json")
    previous = {name: None for name in result["parts"]}
    for key in sorted(result["frames"], key=int):
        frame = result["frames"][key]
        if int(key) >= args.assembled_start:
            record = frame["parts"][part]
            record["T_world_from_part"] = selected_world.tolist()
            record["S_world_from_raw_mesh"] = similarity_from_rigid(
                selected_world, scale, origin).tolist()
            record["source"] = "inner_assembly_prior_bounded_gicp"
            record["assembly_prior_bounded"] = True
        body_pose = np.asarray(frame["parts"]["body"]["T_world_from_part"], float)
        for name in result["parts"]:
            record = frame["parts"][name]
            pose = np.asarray(record["T_world_from_part"], float)
            rel = np.linalg.inv(body_pose) @ pose
            record["T_body_from_part"] = rel.tolist()
            record["translation_body_m"] = rel[:3, 3].tolist()
            record["quaternion_body_xyzw"] = Rotation.from_matrix(rel[:3, :3]).as_quat().tolist()
            if previous[name] is None:
                record["translation_step_m"] = record["rotation_step_deg"] = 0.0
            else:
                step = np.linalg.inv(previous[name]) @ pose
                record["translation_step_m"] = float(np.linalg.norm(step[:3, 3]))
                record["rotation_step_deg"] = float(np.degrees(
                    Rotation.from_matrix(step[:3, :3]).magnitude()))
            previous[name] = pose

    report = {
        "schema_version": 1, "source_trajectory": str(source_path),
        "assembled_start": args.assembled_start, "cloud_frames": used,
        "prior_T_body_from_inner": prior.tolist(),
        "max_translation_mm": args.max_translation_mm,
        "max_rotation_deg": args.max_rotation_deg,
        "bounded_correction": cap_report, "gicp": gicp,
        "prior_cloud_metrics": prior_metrics,
        "refined_cloud_metrics": refined_metrics,
        "accepted_bounded_refinement": bool(accepted),
        "selected_T_body_from_inner": selected_relative.tolist(),
    }
    write_json(output / "diagnostics/inner_assembly_prior.json", report)
    write_json(output / "pose/trajectory.json", result)
    write_trajectory_csv(result, output / "pose/trajectory.csv")
    print(f"bounded GICP accepted={accepted}; used frames={used}")
    print(f"wrote {output / 'pose/trajectory.json'}")


if __name__ == "__main__":
    main()
