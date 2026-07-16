#!/usr/bin/env python
"""Refine the static body yaw against an aggregate six-view silhouette.

The current body calibration is inherited from a validated single-view run.
This script keeps its translation, scale, roll, and pitch fixed, and searches
only rotation around the body's local vertical axis.  It writes a derived
trajectory; the source trajectory is never modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon, scale_intrinsics
from common.pose_transforms import axis_rotation_degrees, similarity_from_rigid


def aggregate_targets(cfg: dict, part: str, start: int, end: int,
                      min_support: int, width: int, height: int) -> list[np.ndarray]:
    part_id = int(cfg["part_ids"][part])
    result = []
    kernel = np.ones((3, 3), np.uint8)
    for view in cfg["views"]:
        support = np.zeros((height, width), np.uint16)
        for frame in range(start, end + 1):
            labels = np.asarray(Image.open(
                Path(cfg["masks_dir"]) / f"{frame:06d}" / f"{view}.png"))
            mask = cv2.resize((labels == part_id).astype(np.uint8), (width, height),
                              interpolation=cv2.INTER_NEAREST)
            support += mask
        target = support >= min_support
        # Close small temporal segmentation gaps without altering the outer
        # asymmetric outline that carries yaw information.
        target = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_CLOSE,
                                  kernel, iterations=1).astype(bool)
        result.append(target)
    return result


def edge(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)


def silhouette_metrics(rendered: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    union = np.logical_or(rendered, target).sum()
    value_iou = float(np.logical_and(rendered, target).sum() / union) if union else 1.0
    er, et = edge(rendered), edge(target)
    if not er.any() or not et.any():
        return value_iou, 100.0, value_iou - 1.0
    dt_target = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 3)
    dt_render = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = 0.5 * (float(dt_target[er].mean()) + float(dt_render[et].mean()))
    score = value_iou - 0.01 * chamfer
    return value_iou, chamfer, score


def make_comparison(targets: list[np.ndarray], baseline: list[np.ndarray],
                    refined: list[np.ndarray], views: list[str], out: Path) -> None:
    rows = []
    for label, rendered_set in (("baseline", baseline), ("refined", refined)):
        panels = []
        for view, target, rendered in zip(views, targets, rendered_set):
            image = np.zeros((*target.shape, 3), np.uint8)
            image[target] = (55, 55, 210)       # target-only red (BGR)
            image[rendered] = (210, 80, 45)     # render-only blue
            image[target & rendered] = (225, 225, 225)
            cv2.putText(image, f"{label} {view}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(panels))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])


def write_csv(trajectory: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "part", "state", "source", "observing_views",
                         "tx", "ty", "tz", "qx", "qy", "qz", "qw",
                         "translation_step_m", "rotation_step_deg"])
        for key, frame in trajectory["frames"].items():
            for part in trajectory["parts"]:
                record = frame["parts"][part]
                writer.writerow([
                    int(key), part, record["state"], record["source"],
                    record["observing_views"], *record["translation_body_m"],
                    *record["quaternion_body_xyzw"], record["translation_step_m"],
                    record["rotation_step_deg"],
                ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_multiview_111.json"))
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--output-root", default=str(
        ROOT / "experiments" / "three_part_multiview_111f" / "outputs_v3_body_yaw"))
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=50)
    parser.add_argument("--min-support-frames", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--coarse-range-deg", type=float, default=45.0)
    parser.add_argument("--coarse-step-deg", type=float, default=2.0)
    parser.add_argument("--fine-radius-deg", type=float, default=2.0)
    parser.add_argument("--fine-step-deg", type=float, default=0.25)
    parser.add_argument("--axis", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_json(config_path)
    source_path = (Path(args.trajectory).resolve() if args.trajectory else
                   Path(cfg["output_root"]) / "pose" / "trajectory.json")
    trajectory = load_json(source_path)
    body = cfg["reference_part"]
    first = trajectory["frames"][sorted(trajectory["frames"])[0]]["parts"][body]
    base_pose = np.asarray(first["T_world_from_part"], float)
    scale = float(trajectory["scales"][body])
    origin = np.asarray(trajectory["raw_mesh_origins"][body], float)
    mesh = trimesh.load(Path(cfg["mesh_dir"]) / f"{body}.glb", force="mesh")
    targets = aggregate_targets(
        cfg, body, args.target_start, args.target_end, args.min_support_frames,
        args.width, args.height)

    recon = load_recon(cfg, f"{args.target_start:06d}", backend=cfg["recon_backend"])
    cameras = [
        (scale_intrinsics(recon["intrinsics"][index], recon["depth_hw"],
                          (args.height, args.width)), recon["extrinsics"][index])
        for index in range(len(cfg["views"]))
    ]
    cache: dict[float, dict] = {}
    with SceneRenderer(args.width, args.height) as renderer:
        def evaluate(angle: float) -> dict:
            key = round(float(angle), 6)
            if key in cache:
                return cache[key]
            pose = base_pose @ axis_rotation_degrees(np.asarray(args.axis), key)
            similarity = similarity_from_rigid(pose, scale, origin)
            rendered = []
            per_view = {}
            for view, target, (K, E) in zip(cfg["views"], targets, cameras):
                mask = renderer.render_seg([(body, mesh, similarity)], K, E)[body]
                rendered.append(mask)
                value_iou, chamfer, score = silhouette_metrics(mask, target)
                per_view[view] = {"iou": value_iou, "chamfer_px": chamfer, "score": score}
            result = {
                "yaw_delta_deg": key,
                "mean_iou": float(np.mean([v["iou"] for v in per_view.values()])),
                "mean_chamfer_px": float(np.mean([v["chamfer_px"] for v in per_view.values()])),
                "mean_score": float(np.mean([v["score"] for v in per_view.values()])),
                "per_view": per_view,
                "pose": pose,
                "similarity": similarity,
                "rendered": rendered,
            }
            cache[key] = result
            print(f"yaw {key:+6.2f} deg | IoU {result['mean_iou']:.4f} | "
                  f"edge {result['mean_chamfer_px']:.2f}px | score {result['mean_score']:.4f}",
                  flush=True)
            return result

        coarse = np.arange(-args.coarse_range_deg, args.coarse_range_deg + 0.5 * args.coarse_step_deg,
                           args.coarse_step_deg)
        coarse_results = [evaluate(float(angle)) for angle in coarse]
        coarse_best = max(coarse_results, key=lambda value: value["mean_score"])
        fine = np.arange(coarse_best["yaw_delta_deg"] - args.fine_radius_deg,
                         coarse_best["yaw_delta_deg"] + args.fine_radius_deg
                         + 0.5 * args.fine_step_deg, args.fine_step_deg)
        fine_results = [evaluate(float(angle)) for angle in fine]
        best = max(fine_results, key=lambda value: value["mean_score"])
        baseline = evaluate(0.0)

    output = Path(args.output_root).resolve()
    comparison_path = output / "diagnostics" / "body_yaw_comparison.jpg"
    make_comparison(targets, baseline["rendered"], best["rendered"], cfg["views"], comparison_path)

    refined = copy.deepcopy(trajectory)
    refined["config"] = str(config_path)
    refined.setdefault("provenance", {})["derived_from_trajectory"] = str(source_path)
    refined["provenance"]["body_yaw_refinement"] = str(
        output / "diagnostics" / "body_yaw_refinement.json")
    T_WB = best["pose"]
    S_WB = best["similarity"]
    for frame in refined["frames"].values():
        for part in refined["parts"]:
            record = frame["parts"][part]
            if part == body:
                record["T_world_from_part"] = T_WB.tolist()
                record["S_world_from_raw_mesh"] = S_WB.tolist()
                record["source"] = "body_multiview_yaw_refined"
                record["translation_step_m"] = 0.0
                record["rotation_step_deg"] = 0.0
            T_WP = np.asarray(record["T_world_from_part"], float)
            T_BP = np.linalg.inv(T_WB) @ T_WP
            record["T_body_from_part"] = T_BP.tolist()
            record["translation_body_m"] = T_BP[:3, 3].tolist()
            record["quaternion_body_xyzw"] = Rotation.from_matrix(T_BP[:3, :3]).as_quat().tolist()

    serializable_search = []
    for value in sorted(cache.values(), key=lambda item: item["yaw_delta_deg"]):
        serializable_search.append({key: val for key, val in value.items()
                                    if key not in ("pose", "similarity", "rendered")})
    report = {
        "config": str(config_path),
        "source_trajectory": str(source_path),
        "output_trajectory": str(output / "pose" / "trajectory.json"),
        "part": body,
        "local_yaw_axis": list(args.axis),
        "target_frames": [args.target_start, args.target_end],
        "min_support_frames": args.min_support_frames,
        "resolution": [args.width, args.height],
        "baseline": {key: value for key, value in baseline.items()
                     if key not in ("pose", "similarity", "rendered")},
        "best": {key: value for key, value in best.items()
                 if key not in ("pose", "similarity", "rendered")},
        "T_world_from_body": T_WB.tolist(),
        "S_world_from_body_raw": S_WB.tolist(),
        "search": serializable_search,
    }
    write_json(output / "diagnostics" / "body_yaw_refinement.json", report)
    write_json(output / "pose" / "trajectory.json", refined)
    write_csv(refined, output / "pose" / "trajectory.csv")
    print(f"best yaw delta: {best['yaw_delta_deg']:+.2f} deg")
    print(f"mean IoU: {baseline['mean_iou']:.4f} -> {best['mean_iou']:.4f}")
    print(f"mean edge distance: {baseline['mean_chamfer_px']:.2f} -> "
          f"{best['mean_chamfer_px']:.2f} px")
    print(f"wrote {output / 'pose' / 'trajectory.json'}")


if __name__ == "__main__":
    main()
