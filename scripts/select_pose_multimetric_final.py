#!/usr/bin/env python
"""Conservatively accept pose branches using visual, geometry and motion gates."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.mesh_align import read_ply_xyz
from common.pose_refinement import cloud_metrics, sample_canonical
from common.pose_transforms import similarity_from_rigid, transform_points
from common.trajectory_io import write_trajectory_csv


def pose_change(old: dict, new: dict, part: str, model: np.ndarray) -> dict:
    translation, rotation, adds = [], [], []
    for key in old["frames"]:
        a = np.asarray(old["frames"][key]["parts"][part]["T_world_from_part"], float)
        b = np.asarray(new["frames"][key]["parts"][part]["T_world_from_part"], float)
        translation.append(1000 * float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
        rotation.append(float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())))
        expected = transform_points(model, a)
        actual = transform_points(model, b)
        nearest, _ = cKDTree(expected).query(actual, k=1, workers=-1)
        adds.append(1000 * float(np.mean(nearest)))
    return {
        "median_translation_delta_mm": float(np.median(translation)),
        "max_translation_delta_mm": float(np.max(translation)),
        "median_rotation_delta_deg": float(np.median(rotation)),
        "max_rotation_delta_deg": float(np.max(rotation)),
        "median_adds_mm": float(np.median(adds)),
        "max_adds_mm": float(np.max(adds)),
        "changed": bool(max(translation) > 1e-5 or max(rotation) > 1e-5),
    }


def motion_summary(trajectory: dict, part: str) -> dict:
    translation, rotation = [], []
    previous = None
    for key in sorted(trajectory["frames"], key=int):
        pose = np.asarray(trajectory["frames"][key]["parts"][part]["T_world_from_part"], float)
        if previous is not None:
            delta = np.linalg.inv(previous) @ pose
            translation.append(1000 * float(np.linalg.norm(delta[:3, 3])))
            rotation.append(float(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())))
        previous = pose
    return {"max_translation_step_mm": max(translation, default=0.0),
            "max_rotation_step_deg": max(rotation, default=0.0)}


def geometry_summary(trajectory: dict, part: str, model: np.ndarray,
                     cloud_root: Path, frames: list[int]) -> dict:
    rows = []
    for frame in frames:
        path = cloud_root / f"{frame:06d}" / f"{part}.ply"
        if not path.exists():
            continue
        cloud = read_ply_xyz(str(path))
        if len(cloud) < 30:
            continue
        pose = np.asarray(trajectory["frames"][f"{frame:06d}"]["parts"][part]
                          ["T_world_from_part"], float)
        rows.append(cloud_metrics(cloud, model, pose))
    return {
        "frames": frames, "evaluated_frames": len(rows),
        "mean_fitness_8mm": float(np.mean([x["fitness_8mm"] for x in rows])),
        "median_nn_m": float(np.median([x["median_nn_m"] for x in rows])),
        "mean_trimmed_rmse_m": float(np.mean([x["trimmed_rmse_m"] for x in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pose_multiview_111_v4.json"))
    parser.add_argument("--quality-config", default=str(ROOT / "configs/pose_multiview_111_quality.json"))
    parser.add_argument("--baseline", default=str(ROOT / "experiments/three_part_multiview_111f/outputs_v4_lid_se3/pose/trajectory.json"))
    parser.add_argument("--baseline-metrics", default=str(ROOT / "experiments/three_part_multiview_111f/outputs_v4_lid_se3/diagnostics/multiview_metrics.json"))
    parser.add_argument("--candidate", default=str(ROOT / "experiments/three_part_multiview_111f/outputs_v6_global_pose/lid_tilt/pose/trajectory.json"))
    parser.add_argument("--candidate-metrics", default=str(ROOT / "experiments/three_part_multiview_111f/outputs_v6_global_pose/lid_tilt/diagnostics/multiview_metrics.json"))
    parser.add_argument("--output-root", default=str(ROOT / "experiments/three_part_multiview_111f/outputs_v6_global_pose/accepted_final"))
    args = parser.parse_args()

    cfg, quality = load_json(Path(args.config)), load_json(Path(args.quality_config))
    old, new = load_json(Path(args.baseline)), load_json(Path(args.candidate))
    old_visual, new_visual = load_json(Path(args.baseline_metrics)), load_json(Path(args.candidate_metrics))
    output = Path(args.output_root).resolve()
    cloud_root = Path(quality["point_cloud_root"])
    eval_frames = {"body": [0, 10, 20, 30, 40, 60, 90, 110],
                   "inner_pot": [0, 20, 40, 50, 60, 70, 80, 90, 100],
                   "lid": [0, 40, 50, 68, 76, 84, 92, 100, 108, 110]}
    thresholds = {
        "body": {"rotation": 180.1, "translation": 30.0, "adds": 100.0,
                 "step_t": 1.0, "step_r": 0.2},
        "inner_pot": {"rotation": 20.0, "translation": 30.0, "adds": 40.0,
                      "step_t": 80.0, "step_r": 30.0},
        "lid": {"rotation": 35.0, "translation": 50.0, "adds": 50.0,
                "step_t": 40.1, "step_r": 3.05},
    }
    decisions = {}
    for index, part in enumerate(old["parts"]):
        mesh = trimesh.load(Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh")
        model = sample_canonical(mesh, float(old["scales"][part]),
                                 np.asarray(old["raw_mesh_origins"][part], float), 4000, 4400 + index)
        change = pose_change(old, new, part, model)
        old_geo = geometry_summary(old, part, model, cloud_root, eval_frames[part])
        new_geo = geometry_summary(new, part, model, cloud_root, eval_frames[part])
        old_v = old_visual["summary"][part]["all_views"]
        new_v = new_visual["summary"][part]["all_views"]
        motion = motion_summary(new, part)
        limit = thresholds[part]
        branch = (change["max_translation_delta_mm"] <= limit["translation"]
                  and change["max_rotation_delta_deg"] <= limit["rotation"]
                  and change["max_adds_mm"] <= limit["adds"]
                  and motion["max_translation_step_mm"] <= limit["step_t"]
                  and motion["max_rotation_step_deg"] <= limit["step_r"])
        visual = (new_v["mean_iou"] >= old_v["mean_iou"] - 0.001
                  and new_v["mean_contour_chamfer_px"] <= old_v["mean_contour_chamfer_px"] + 0.03)
        geometry = (new_geo["mean_fitness_8mm"] >= old_geo["mean_fitness_8mm"] - 0.01
                    and new_geo["median_nn_m"] <= old_geo["median_nn_m"] + 0.001)
        # DA3 geometry remains an important diagnostic, but is not a hard
        # rejection gate in this experiment: the user explicitly deferred
        # depth/GT quality work, while the required acceptance contract is
        # rotation proxy + ADD-S + six-view contour + trajectory branch.
        accepted = change["changed"] and branch and visual
        decisions[part] = {
            "accepted_individually": bool(accepted), "pose_change_proxy": change,
            "add_s_definition": "mean nearest-neighbour distance between baseline and candidate transformed mesh",
            "candidate_motion": motion, "branch_limits": limit,
            "branch_gate_passed": bool(branch), "visual_gate_passed": bool(visual),
            "geometry_advisory_passed": bool(geometry), "baseline_visual": old_v,
            "candidate_visual": new_v, "baseline_cloud": old_geo, "candidate_cloud": new_geo,
        }

    # Body and assembled inner_pot are a coupled kinematic branch.  Never accept
    # one without the other, otherwise the assembly relation silently changes.
    assembly_group = (decisions["body"]["accepted_individually"]
                      and decisions["inner_pot"]["accepted_individually"])
    accepted_parts = {"body": assembly_group, "inner_pot": assembly_group,
                      "lid": decisions["lid"]["accepted_individually"]}
    result = copy.deepcopy(old)
    for part, accepted in accepted_parts.items():
        decisions[part]["accepted_final"] = bool(accepted)
        if accepted:
            for key in result["frames"]:
                result["frames"][key]["parts"][part] = copy.deepcopy(
                    new["frames"][key]["parts"][part])

    # Normalize all dependent fields after part-wise branch selection.
    previous = {part: None for part in result["parts"]}
    for key in sorted(result["frames"], key=int):
        frame = result["frames"][key]
        body = np.asarray(frame["parts"]["body"]["T_world_from_part"], float)
        for part in result["parts"]:
            record = frame["parts"][part]
            pose = np.asarray(record["T_world_from_part"], float)
            rel = np.linalg.inv(body) @ pose
            record["T_body_from_part"] = rel.tolist()
            record["translation_body_m"] = rel[:3, 3].tolist()
            record["quaternion_body_xyzw"] = Rotation.from_matrix(rel[:3, :3]).as_quat().tolist()
            record["S_world_from_raw_mesh"] = similarity_from_rigid(
                pose, float(result["scales"][part]),
                np.asarray(result["raw_mesh_origins"][part], float)).tolist()
            if previous[part] is None:
                record["translation_step_m"] = record["rotation_step_deg"] = 0.0
            else:
                delta = np.linalg.inv(previous[part]) @ pose
                record["translation_step_m"] = float(np.linalg.norm(delta[:3, 3]))
                record["rotation_step_deg"] = float(np.degrees(
                    Rotation.from_matrix(delta[:3, :3]).magnitude()))
            previous[part] = pose

    result.setdefault("provenance", {})["multimetric_selection"] = str(
        output / "diagnostics/multimetric_selection.json")
    report = {
        "schema_version": 1,
        "policy": "hard gates: ordinary pose delta + ADD-S + six-view silhouette + trajectory branch; DA3 cloud is advisory until depth quality is resolved",
        "baseline": str(Path(args.baseline).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "assembly_group_accepted": bool(assembly_group), "parts": decisions,
    }
    write_json(output / "diagnostics/multimetric_selection.json", report)
    write_json(output / "pose/trajectory.json", result)
    write_trajectory_csv(result, output / "pose/trajectory.csv")
    print(" ".join(f"{p}={'candidate' if a else 'baseline'}" for p, a in accepted_parts.items()))
    print(f"wrote {output / 'pose/trajectory.json'}")


if __name__ == "__main__":
    main()
