#!/usr/bin/env python
"""Find the static body orientation with a 360-degree proposal search.

Every yaw proposal is scored against all six silhouettes.  The best global
basin is then refined with the project's existing multi-scale GICP using a
multi-frame, six-view quality cloud.  The source trajectory is never changed.
"""
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

from common.io_utils import load_json, write_json
from common.gicp import multiscale_gicp, voxel_unique
from common.mesh_align import read_ply_xyz
from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon, scale_intrinsics
from common.pose_refinement import (
    aggregate_mask_targets, cloud_metrics, make_mask_comparison,
    sample_canonical, silhouette_metrics,
)
from common.pose_transforms import axis_rotation_degrees, similarity_from_rigid
from common.trajectory_io import write_trajectory_csv


def load_fused(root: Path, frames: list[int], part: str, maximum: int,
               seed: int) -> tuple[np.ndarray, dict]:
    clouds, used = [], []
    for frame in frames:
        path = root / f"{frame:06d}" / f"{part}.ply"
        if not path.exists():
            continue
        cloud = read_ply_xyz(str(path))
        if len(cloud) >= 30:
            clouds.append(cloud)
            used.append(frame)
    if not clouds:
        raise RuntimeError(f"no {part} clouds below {root}")
    points = voxel_unique(np.concatenate(clouds), 0.003)
    if len(points) > maximum:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), maximum, replace=False)]
    return np.ascontiguousarray(points, dtype=np.float64), {
        "requested_frames": frames, "used_frames": used, "points": int(len(points))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pose_multiview_111_v4.json"))
    parser.add_argument("--quality-config", default=str(
        ROOT / "configs/pose_multiview_111_quality.json"))
    parser.add_argument("--trajectory", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v4_lid_se3/pose/trajectory.json"))
    parser.add_argument("--output-root", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v6_global_pose/body_global"))
    parser.add_argument("--yaw-step-deg", type=float, default=15.0)
    parser.add_argument("--axis", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    parser.add_argument("--cloud-frames", type=int, nargs="*", default=[0, 10, 20, 30, 40, 60, 90, 110])
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=50)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    quality_cfg = load_json(Path(args.quality_config))
    source_path = Path(args.trajectory).resolve()
    source = load_json(source_path)
    output = Path(args.output_root).resolve()
    body = source["reference_part"]
    first_key = sorted(source["frames"])[0]
    base_pose = np.asarray(source["frames"][first_key]["parts"][body]["T_world_from_part"], float)
    scale = float(source["scales"][body])
    origin = np.asarray(source["raw_mesh_origins"][body], float)
    mesh = trimesh.load(Path(cfg["mesh_dir"]) / f"{body}.glb", force="mesh")
    canonical = sample_canonical(mesh, scale, origin, 30000, 2301)
    observed, cloud_info = load_fused(
        Path(quality_cfg["point_cloud_root"]), args.cloud_frames, body,
        int(quality_cfg["registration"]["max_points"]), 2302)

    targets = aggregate_mask_targets(cfg, body, args.target_start, args.target_end, 3,
                                     args.width, args.height)
    recon = load_recon(cfg, f"{args.target_start:06d}", backend=cfg["recon_backend"])
    cameras = [(scale_intrinsics(recon["intrinsics"][i], recon["depth_hw"],
                                 (args.height, args.width)), recon["extrinsics"][i])
               for i in range(len(cfg["views"]))]

    def render_score(renderer: SceneRenderer, pose: np.ndarray) -> dict:
        similarity = similarity_from_rigid(pose, scale, origin)
        rendered, per_view = [], {}
        for view, target, (K, E) in zip(cfg["views"], targets, cameras):
            mask = renderer.render_seg([(body, mesh, similarity)], K, E)[body]
            rendered.append(mask)
            iou, chamfer, score = silhouette_metrics(mask, target)
            per_view[view] = {"iou": iou, "chamfer_px": chamfer, "score": score}
        return {
            "mean_iou": float(np.mean([x["iou"] for x in per_view.values()])),
            "mean_chamfer_px": float(np.mean([x["chamfer_px"] for x in per_view.values()])),
            "mean_score": float(np.mean([x["score"] for x in per_view.values()])),
            "per_view": per_view, "rendered": rendered,
        }

    axis = np.asarray(args.axis, float)
    proposal_rows = []
    with SceneRenderer(args.width, args.height) as renderer:
        for angle in np.arange(-180.0, 180.0, args.yaw_step_deg):
            pose = base_pose @ axis_rotation_degrees(axis, float(angle))
            sil = render_score(renderer, pose)
            geom = cloud_metrics(observed, canonical, pose)
            # Silhouette selects orientation; the cloud term breaks close ties.
            joint = sil["mean_score"] + 0.08 * geom["fitness_8mm"] - 2.0 * geom["median_nn_m"]
            proposal_rows.append({
                "yaw_delta_deg": float(angle), "pose": pose, "silhouette": sil,
                "cloud": geom, "proposal_score": float(joint),
            })
            print(f"yaw {angle:+6.1f}: score={joint:.5f} iou={sil['mean_iou']:.4f} "
                  f"fit8={geom['fitness_8mm']:.3f}", flush=True)
        basin = max(proposal_rows, key=lambda x: x["proposal_score"])

        # Existing GICP maps observed-world source into the canonical model.
        world_to_part, gicp_report = multiscale_gicp(
            observed, canonical, np.linalg.inv(basin["pose"]), quality_cfg["registration"])
        refined_pose = np.linalg.inv(world_to_part)
        refined_sil = render_score(renderer, refined_pose)
        refined_geom = cloud_metrics(observed, canonical, refined_pose)
        refined_joint = (refined_sil["mean_score"] + 0.08 * refined_geom["fitness_8mm"]
                         - 2.0 * refined_geom["median_nn_m"])
        refined = {"pose": refined_pose, "silhouette": refined_sil, "cloud": refined_geom,
                   "proposal_score": float(refined_joint)}
        # Do not allow a local optimizer to leave the selected visual basin.
        selected = refined if refined_joint >= basin["proposal_score"] - 0.002 else basin
        baseline_sil = next(x["silhouette"] for x in proposal_rows
                            if abs(x["yaw_delta_deg"]) < 1e-8)

    chosen_pose = selected["pose"]
    chosen_similarity = similarity_from_rigid(chosen_pose, scale, origin)
    result = copy.deepcopy(source)
    result.setdefault("provenance", {})["body_global_yaw"] = str(
        output / "diagnostics/body_global_yaw.json")
    result["provenance"]["derived_from_trajectory"] = str(source_path)
    for frame in result["frames"].values():
        record = frame["parts"][body]
        record["T_world_from_part"] = chosen_pose.tolist()
        record["S_world_from_raw_mesh"] = chosen_similarity.tolist()
        record["source"] = "body_360_yaw_multiview_gicp"
    # Recompute every body-relative record after changing the reference frame.
    previous = {part: None for part in result["parts"]}
    for key in sorted(result["frames"], key=int):
        frame = result["frames"][key]
        for part in result["parts"]:
            record = frame["parts"][part]
            pose = np.asarray(record["T_world_from_part"], float)
            relative = np.linalg.inv(chosen_pose) @ pose
            record["T_body_from_part"] = relative.tolist()
            record["translation_body_m"] = relative[:3, 3].tolist()
            record["quaternion_body_xyzw"] = Rotation.from_matrix(relative[:3, :3]).as_quat().tolist()
            if previous[part] is None:
                record["translation_step_m"] = record["rotation_step_deg"] = 0.0
            else:
                delta = np.linalg.inv(previous[part]) @ pose
                record["translation_step_m"] = float(np.linalg.norm(delta[:3, 3]))
                record["rotation_step_deg"] = float(np.degrees(
                    Rotation.from_matrix(delta[:3, :3]).magnitude()))
            previous[part] = pose

    def serial(row: dict) -> dict:
        return {"yaw_delta_deg": row.get("yaw_delta_deg"),
                "proposal_score": row["proposal_score"], "silhouette": {
                    k: v for k, v in row["silhouette"].items() if k != "rendered"},
                "cloud": row["cloud"]}

    report = {
        "schema_version": 1, "source_trajectory": str(source_path),
        "coverage_deg": 360.0, "yaw_step_deg": args.yaw_step_deg,
        "proposal_count": len(proposal_rows), "six_view_scoring": True,
        "cloud": cloud_info, "baseline": serial(next(
            x for x in proposal_rows if abs(x["yaw_delta_deg"]) < 1e-8)),
        "global_basin": serial(basin),
        "gicp_refined": {**serial(refined), "gicp": gicp_report},
        "selected_stage": "gicp_refined" if selected is refined else "global_proposal",
        "T_world_from_body": chosen_pose.tolist(),
        "proposals": [serial(x) for x in proposal_rows],
    }
    write_json(output / "diagnostics/body_global_yaw.json", report)
    write_json(output / "pose/trajectory.json", result)
    write_trajectory_csv(result, output / "pose/trajectory.csv")
    make_mask_comparison(targets, baseline_sil["rendered"],
                         selected["silhouette"]["rendered"], cfg["views"],
                         output / "diagnostics/body_global_yaw_comparison.jpg")
    print(f"selected {basin['yaw_delta_deg']:+.1f} deg basin; stage={report['selected_stage']}")
    print(f"wrote {output / 'pose/trajectory.json'}")


if __name__ == "__main__":
    main()
