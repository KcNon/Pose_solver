#!/usr/bin/env python
"""Six-view score, GICP-refine and conservatively apply hybrid FP candidates."""
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

from common.gicp import multiscale_gicp
from common.io_utils import load_json, write_json
from common.mesh_align import read_ply_xyz
from common.pose_refinement import (
    cap_pose_delta,
    cloud_metrics,
    limit_pose_velocity,
    sample_canonical,
)
from common.pose_transforms import similarity_from_rigid
from common.trajectory_io import write_trajectory_csv
from scripts.refine_lid_multiview_se3 import FrameObjective, interpolate_corrections


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(a.T @ b).magnitude()))


def compact_evaluation(value: dict) -> dict:
    return {
        "score": float(value["score"]),
        "observing_views": int(value["observing_views"]),
        "mean_iou": float(np.mean([item["iou"] for item in value["per_view"].values()])),
        "mean_silhouette_chamfer_px": float(np.mean([
            item["silhouette_chamfer_px"] for item in value["per_view"].values()])),
        "mean_rgb_edge_chamfer_px": float(np.mean([
            item["rgb_edge_chamfer_px"] for item in value["per_view"].values()])),
        "per_view": value["per_view"],
    }


def joint_score(visual: dict, geometry: dict | None) -> float:
    if geometry is None:
        return float(visual["score"])
    return float(visual["score"] + 0.08 * geometry["fitness_8mm"]
                 - 2.0 * geometry["median_nn_m"])


def recompute_records(trajectory: dict) -> None:
    reference = trajectory["reference_part"]
    previous = {part: None for part in trajectory["parts"]}
    for key in sorted(trajectory["frames"], key=int):
        frame = trajectory["frames"][key]
        body = np.asarray(frame["parts"][reference]["T_world_from_part"], float)
        for part in trajectory["parts"]:
            record = frame["parts"][part]
            pose = np.asarray(record["T_world_from_part"], float)
            relative = np.linalg.inv(body) @ pose
            record["T_body_from_part"] = relative.tolist()
            record["translation_body_m"] = relative[:3, 3].tolist()
            record["quaternion_body_xyzw"] = Rotation.from_matrix(
                relative[:3, :3]).as_quat().tolist()
            record["S_world_from_raw_mesh"] = similarity_from_rigid(
                pose, float(trajectory["scales"][part]),
                np.asarray(trajectory["raw_mesh_origins"][part], float)).tolist()
            if previous[part] is None:
                record["translation_step_m"] = 0.0
                record["rotation_step_deg"] = 0.0
            else:
                delta = np.linalg.inv(previous[part]) @ pose
                record["translation_step_m"] = float(np.linalg.norm(delta[:3, 3]))
                record["rotation_step_deg"] = float(np.degrees(
                    Rotation.from_matrix(delta[:3, :3]).magnitude()))
            previous[part] = pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pose_multiview_111_v4.json"))
    parser.add_argument("--quality-config", default=str(
        ROOT / "configs/pose_multiview_111_quality.json"))
    parser.add_argument("--candidates", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v7_foundationpose_hybrid/candidates.json"))
    parser.add_argument("--output-root", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v7_foundationpose_hybrid/evaluated"))
    parser.add_argument("--top-gicp", type=int, default=3)
    parser.add_argument("--min-score-gain", type=float, default=0.003)
    parser.add_argument("--max-gicp-translation-mm", type=float, default=20.0)
    parser.add_argument("--max-gicp-rotation-deg", type=float, default=20.0)
    parser.add_argument("--max-lid-branch-deg", type=float, default=45.0)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    quality = load_json(Path(args.quality_config))
    proposal_report = load_json(Path(args.candidates))
    source_path = Path(proposal_report["trajectory"]).resolve()
    source = load_json(source_path)
    output = Path(args.output_root).resolve()
    cloud_root = Path(quality["point_cloud_root"])
    options = cfg["lid_refinement"]
    width, height = map(int, options["resolution"])
    reports = {}
    accepted: dict[str, dict[int, np.ndarray]] = {part: {} for part in source["parts"]}

    # One renderer instance amortizes EGL setup across all keyframes.
    from common.mesh_render import SceneRenderer
    with SceneRenderer(width, height) as renderer:
        for target_key, target in proposal_report["targets"].items():
            part, frame = target["part"], int(target["frame"])
            timestamp = f"{frame:06d}"
            mesh = trimesh.load(Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh")
            scale = float(source["scales"][part])
            origin = np.asarray(source["raw_mesh_origins"][part], float)
            canonical = sample_canonical(mesh, scale, origin, 16000, 8100 + frame)
            cloud_path = cloud_root / timestamp / f"{part}.ply"
            cloud = read_ply_xyz(str(cloud_path)) if cloud_path.exists() else None
            if cloud is not None and len(cloud) < 30:
                cloud = None
            objective = FrameObjective(
                cfg, frame, mesh, scale, origin, renderer, width, height,
                int(options["min_mask_pixels"]),
                float(options["silhouette_chamfer_weight"]),
                float(options["rgb_edge_chamfer_weight"]), part=part,
            )
            initial = np.asarray(target["initial_T_world_from_part"], float)
            baseline_visual = compact_evaluation(objective.evaluate(initial))
            baseline_geometry = cloud_metrics(cloud, canonical, initial) if cloud is not None else None
            baseline_joint = joint_score(baseline_visual, baseline_geometry)
            rows = []
            for index, candidate in enumerate(target["candidates"]):
                pose = np.asarray(candidate["T_world_from_part"], float)
                visual = compact_evaluation(objective.evaluate(pose))
                geometry = cloud_metrics(cloud, canonical, pose) if cloud is not None else None
                rows.append({
                    "stage": candidate["source"], "candidate_index": index,
                    "source_view": candidate.get("view"),
                    "rotation_from_initial_deg": rotation_error_deg(
                        initial[:3, :3], pose[:3, :3]),
                    "translation_from_initial_mm": float(
                        1000 * np.linalg.norm(pose[:3, 3] - initial[:3, 3])),
                    "visual": visual, "geometry": geometry,
                    "joint_score": joint_score(visual, geometry), "pose": pose,
                })
            rows.sort(key=lambda item: item["joint_score"], reverse=True)

            gicp_rows = []
            if cloud is not None:
                for row in rows[:max(0, args.top_gicp)]:
                    world_to_part, gicp = multiscale_gicp(
                        cloud, canonical, np.linalg.inv(row["pose"]), quality["registration"])
                    raw_refined = np.linalg.inv(world_to_part)
                    correction = raw_refined @ np.linalg.inv(row["pose"])
                    bounded, cap = cap_pose_delta(
                        correction, args.max_gicp_translation_mm / 1000.0,
                        args.max_gicp_rotation_deg)
                    pose = bounded @ row["pose"]
                    visual = compact_evaluation(objective.evaluate(pose))
                    geometry = cloud_metrics(cloud, canonical, pose)
                    gicp_rows.append({
                        "stage": "foundationpose_then_gicp",
                        "parent_candidate_index": row["candidate_index"],
                        "source_view": row["source_view"],
                        "rotation_from_initial_deg": rotation_error_deg(
                            initial[:3, :3], pose[:3, :3]),
                        "translation_from_initial_mm": float(
                            1000 * np.linalg.norm(pose[:3, 3] - initial[:3, 3])),
                        "visual": visual, "geometry": geometry,
                        "joint_score": joint_score(visual, geometry),
                        "gicp": gicp, "bounded_correction": cap, "pose": pose,
                    })
            ranked = sorted(rows + gicp_rows, key=lambda item: item["joint_score"], reverse=True)
            selected = ranked[0]
            score_gain = selected["joint_score"] - baseline_joint
            geometry_passed = True
            if baseline_geometry is not None and selected["geometry"] is not None:
                geometry_passed = (
                    selected["geometry"]["fitness_8mm"] >= baseline_geometry["fitness_8mm"] - 0.01
                    and selected["geometry"]["median_nn_m"] <= baseline_geometry["median_nn_m"] + 0.001
                )
            branch_limit = args.max_lid_branch_deg if part == "lid" else 180.1
            branch_passed = selected["rotation_from_initial_deg"] <= branch_limit
            accepted_frame = bool(
                score_gain >= args.min_score_gain
                and selected["visual"]["observing_views"] >= 3
                and geometry_passed and branch_passed
                and selected["translation_from_initial_mm"] <= args.max_gicp_translation_mm + 1e-6
            )
            if accepted_frame:
                accepted[part][frame] = selected["pose"]

            def serial(row: dict) -> dict:
                return {key: value for key, value in row.items() if key != "pose"}

            reports[target_key] = {
                "part": part, "frame": frame,
                "baseline": {"visual": baseline_visual, "geometry": baseline_geometry,
                             "joint_score": baseline_joint},
                "selected": serial(selected), "score_gain": float(score_gain),
                "accepted": accepted_frame,
                "gates": {"geometry": bool(geometry_passed), "branch": bool(branch_passed),
                          "branch_limit_deg": branch_limit,
                          "min_score_gain": args.min_score_gain},
                "top_candidates": [serial(row) for row in ranked[:10]],
            }
            print(f"{target_key}: gain={score_gain:+.5f} "
                  f"dr={selected['rotation_from_initial_deg']:.1f}deg "
                  f"dt={selected['translation_from_initial_mm']:.1f}mm "
                  f"accepted={accepted_frame}", flush=True)

    result = copy.deepcopy(source)
    if accepted["body"]:
        # Static body: use the strongest accepted keyframe measurement.
        frame = max(accepted["body"], key=lambda value: reports[f"body:{value:06d}"]["score_gain"])
        pose = accepted["body"][frame]
        for record in result["frames"].values():
            record["parts"]["body"]["T_world_from_part"] = pose.tolist()
            record["parts"]["body"]["source"] = "foundationpose_hybrid_static"

    lid_targets = sorted(
        int(value["frame"]) for value in proposal_report["targets"].values()
        if value["part"] == "lid")
    lid_velocity_report = None
    if len(lid_targets) >= 2:
        key_base = {
            frame: np.asarray(source["frames"][f"{frame:06d}"]["parts"]["lid"]
                              ["T_world_from_part"], float)
            for frame in lid_targets
        }
        key_refined = {frame: accepted["lid"].get(frame, key_base[frame]) for frame in lid_targets}
        all_base = {
            frame: np.asarray(source["frames"][f"{frame:06d}"]["parts"]["lid"]
                              ["T_world_from_part"], float)
            for frame in range(lid_targets[0], lid_targets[-1] + 1)
        }
        motion = interpolate_corrections(
            key_base, key_refined, all_base, lid_targets[0], lid_targets[-1])
        motion, lid_velocity_report = limit_pose_velocity(
            motion,
            float(options.get("max_translation_step_m", 0.04)),
            float(options.get("max_rotation_step_deg", 3.0)),
        )
        for key, frame_record in result["frames"].items():
            frame = int(key)
            if frame < lid_targets[0]:
                pose = key_refined[lid_targets[0]]
            elif frame > lid_targets[-1]:
                pose = key_refined[lid_targets[-1]]
            else:
                pose = motion[frame]
            frame_record["parts"]["lid"]["T_world_from_part"] = pose.tolist()
            frame_record["parts"]["lid"]["source"] = (
                "foundationpose_hybrid_keyframe" if frame in accepted["lid"]
                else "foundationpose_hybrid_interpolated")

    result.setdefault("provenance", {})["foundationpose_hybrid"] = str(
        output / "diagnostics/foundationpose_hybrid.json")
    result["provenance"]["derived_from_trajectory"] = str(source_path)
    recompute_records(result)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "policy": "FP global rotation candidates; six-view visual+RGB-edge+quality-cloud score; bounded GICP; conservative branch gates",
        "source_trajectory": str(source_path),
        "candidate_report": str(Path(args.candidates).resolve()),
        "accepted_keyframes": {part: sorted(frames) for part, frames in accepted.items()},
        "lid_motion_velocity_gate": lid_velocity_report,
        "targets": reports,
    }
    write_json(output / "diagnostics/foundationpose_hybrid.json", summary)
    write_json(output / "pose/trajectory.json", result)
    write_trajectory_csv(result, output / "pose/trajectory.csv")
    print(f"wrote {output / 'pose/trajectory.json'}", flush=True)


if __name__ == "__main__":
    main()
