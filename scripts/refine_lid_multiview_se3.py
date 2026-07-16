#!/usr/bin/env python
"""Refine lid SE(3) at keyframes using six-view silhouette and RGB edges.

The input trajectory is never modified.  A small coordinate search measures
translation and local XYZ rotation at selected frames, reports per-axis
rotation sensitivity, smooths the measured corrections, and interpolates them
over the complete lid motion.  Static poses before/after the motion inherit the
corresponding refined endpoint.

RGB is used through illumination-robust Canny features inside the visible lid
mask.  This preserves information from the handle and textured internal
structure without assuming rendered and captured brightness are identical.
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
from common.pose_transforms import similarity_from_rigid


def mask_edge(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)).astype(bool)


def distance_to(binary: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~binary).astype(np.uint8), cv2.DIST_L2, 3)


def capped_mean(values: np.ndarray, cap: float = 20.0) -> float:
    return float(np.mean(np.minimum(values, cap))) if len(values) else cap


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
                writer.writerow([int(key), part, record["state"], record["source"],
                                 record["observing_views"], *record["translation_body_m"],
                                 *record["quaternion_body_xyzw"], record["translation_step_m"],
                                 record["rotation_step_deg"]])


class FrameObjective:
    def __init__(self, cfg: dict, frame: int, mesh: trimesh.Trimesh, scale: float,
                 origin: np.ndarray, renderer: SceneRenderer, width: int, height: int,
                 min_pixels: int, chamfer_weight: float, feature_weight: float):
        self.cfg = cfg
        self.frame = frame
        self.mesh = mesh
        self.scale = scale
        self.origin = origin
        self.renderer = renderer
        self.width, self.height = width, height
        self.chamfer_weight = chamfer_weight
        self.feature_weight = feature_weight
        self.cache: dict[tuple[float, ...], dict] = {}
        timestamp = f"{frame:06d}"
        recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
        self.views = []
        for index, view in enumerate(cfg["views"]):
            labels = np.asarray(Image.open(Path(cfg["masks_dir"]) / timestamp / f"{view}.png"))
            target = cv2.resize((labels == int(cfg["part_ids"]["lid"])).astype(np.uint8),
                                (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            if int(target.sum()) < min_pixels:
                continue
            source = cv2.imread(str(Path(cfg["frames_dir"]) / view / f"{timestamp}.jpg"))
            if source is None:
                raise FileNotFoundError(Path(cfg["frames_dir"]) / view / f"{timestamp}.jpg")
            source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            interior = cv2.erode(target.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            rgb_edge = (cv2.Canny(gray, 55, 130) > 0) & interior
            K = scale_intrinsics(recon["intrinsics"][index], recon["depth_hw"], (height, width))
            touches = (target[0].any() or target[-1].any() or target[:, 0].any()
                       or target[:, -1].any())
            self.views.append({
                "name": view, "target": target, "target_edge": mask_edge(target),
                "target_dt": distance_to(mask_edge(target)), "rgb_edge": rgb_edge,
                "rgb_dt": distance_to(rgb_edge) if rgb_edge.any() else None,
                "K": K, "E": recon["extrinsics"][index],
                "weight": (0.55 if touches else 1.0),
            })

    def evaluate(self, pose: np.ndarray) -> dict:
        key = tuple(np.round(pose[:3, :].ravel(), 7))
        if key in self.cache:
            return self.cache[key]
        similarity = similarity_from_rigid(pose, self.scale, self.origin)
        per_view = {}
        weighted_score = 0.0
        total_weight = 0.0
        for item in self.views:
            rgb, depth = self.renderer.render([(self.mesh, similarity)], item["K"], item["E"])
            predicted = depth > 0
            target = item["target"]
            union = np.logical_or(predicted, target).sum()
            iou = float(np.logical_and(predicted, target).sum() / union) if union else 1.0
            pred_edge = mask_edge(predicted)
            if pred_edge.any() and item["target_edge"].any():
                pred_dt = distance_to(pred_edge)
                silhouette_chamfer = 0.5 * (
                    capped_mean(item["target_dt"][pred_edge])
                    + capped_mean(pred_dt[item["target_edge"]]))
            else:
                silhouette_chamfer = 20.0

            rendered_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            pred_interior = cv2.erode(predicted.astype(np.uint8),
                                      np.ones((5, 5), np.uint8)).astype(bool)
            rendered_edge = (cv2.Canny(rendered_gray, 35, 105) > 0) & pred_interior
            if (item["rgb_dt"] is not None and item["rgb_edge"].sum() >= 12
                    and rendered_edge.sum() >= 12):
                rendered_dt = distance_to(rendered_edge)
                feature_chamfer = 0.5 * (
                    capped_mean(item["rgb_dt"][rendered_edge])
                    + capped_mean(rendered_dt[item["rgb_edge"]]))
            else:
                feature_chamfer = 10.0
            score = (iou - self.chamfer_weight * silhouette_chamfer
                     - self.feature_weight * feature_chamfer)
            weight = item["weight"]
            weighted_score += weight * score
            total_weight += weight
            per_view[item["name"]] = {
                "iou": iou,
                "silhouette_chamfer_px": silhouette_chamfer,
                "rgb_edge_chamfer_px": feature_chamfer,
                "score": score,
                "weight": weight,
                "mask_pixels": int(target.sum()),
            }
        result = {
            "score": weighted_score / max(total_weight, 1e-9),
            "observing_views": len(self.views),
            "per_view": per_view,
        }
        self.cache[key] = result
        return result


def perturb(pose: np.ndarray, axis: int, value: float, rotation_axis: bool) -> np.ndarray:
    result = pose.copy()
    if rotation_axis:
        vector = np.zeros(3)
        vector[axis] = np.deg2rad(value)
        result[:3, :3] = result[:3, :3] @ Rotation.from_rotvec(vector).as_matrix()
    else:
        # Translation axes are in world coordinates because the six camera
        # projections directly constrain the world-space center.
        result[axis, 3] += value
    return result


def coordinate_search(objective: FrameObjective, base: np.ndarray,
                      rotation_steps: list[float], translation_steps: list[float]) -> tuple[np.ndarray, list[dict]]:
    current = base.copy()
    trace = []
    for level, (rstep, tstep) in enumerate(zip(rotation_steps, translation_steps)):
        # Alternate translation and rotation; a second rotation pass reduces
        # ordering bias while keeping the number of EGL renders bounded.
        for kind, axes in (("translation", range(3)), ("rotation", range(3)),
                           ("rotation", range(3))):
            for axis in axes:
                step = rstep if kind == "rotation" else tstep
                candidates = [perturb(current, axis, sign * step, kind == "rotation")
                              for sign in (-1.0, 0.0, 1.0)]
                measured = [objective.evaluate(value) for value in candidates]
                best_index = int(np.argmax([value["score"] for value in measured]))
                current = candidates[best_index]
                trace.append({
                    "level": level, "kind": kind, "axis": "xyz"[axis], "step": step,
                    "candidate_scores": [value["score"] for value in measured],
                    "selected": best_index - 1,
                })
    return current, trace


def smooth_keyframe_corrections(base_poses: dict[int, np.ndarray],
                                measured: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    frames = sorted(measured)
    translations = np.stack([measured[f][:3, 3] - base_poses[f][:3, 3] for f in frames])
    rotvecs = np.stack([
        Rotation.from_matrix(base_poses[f][:3, :3].T @ measured[f][:3, :3]).as_rotvec()
        for f in frames
    ])
    for values in (translations, rotvecs):
        original = values.copy()
        for i in range(1, len(values) - 1):
            values[i] = 0.25 * original[i - 1] + 0.5 * original[i] + 0.25 * original[i + 1]
    result = {}
    for i, frame in enumerate(frames):
        pose = base_poses[frame].copy()
        pose[:3, 3] += translations[i]
        pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(rotvecs[i]).as_matrix()
        result[frame] = pose
    return result


def interpolate_corrections(key_base: dict[int, np.ndarray],
                            key_refined: dict[int, np.ndarray],
                            all_base: dict[int, np.ndarray],
                            start: int, end: int) -> dict[int, np.ndarray]:
    """Interpolate measured corrections while preserving the original path."""
    keys = sorted(key_refined)
    translation_delta = np.stack([
        key_refined[f][:3, 3] - key_base[f][:3, 3] for f in keys
    ])
    rotation_delta = np.stack([
        Rotation.from_matrix(key_base[f][:3, :3].T @ key_refined[f][:3, :3]).as_rotvec()
        for f in keys
    ])
    result = {}
    for frame in range(start, end + 1):
        pose = all_base[frame].copy()
        dt = np.asarray([np.interp(frame, keys, translation_delta[:, axis])
                         for axis in range(3)])
        dr = np.asarray([np.interp(frame, keys, rotation_delta[:, axis])
                         for axis in range(3)])
        pose[:3, 3] += dt
        pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(dr).as_matrix()
        result[frame] = pose
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_multiview_111_v4.json"))
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--keyframes", type=int, nargs="*", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_json(config_path)
    source_path = Path(args.trajectory or cfg["base_trajectory"]).resolve()
    output = Path(args.output_root or cfg["output_root"]).resolve()
    source = load_json(source_path)
    options = cfg["lid_refinement"]
    keyframes = sorted(args.keyframes or [int(x) for x in options["keyframes"]])
    width, height = map(int, options["resolution"])
    mesh = trimesh.load(Path(cfg["mesh_dir"]) / "lid.glb", force="mesh")
    scale = float(source["scales"]["lid"])
    origin = np.asarray(source["raw_mesh_origins"]["lid"], float)
    base_poses = {
        frame: np.asarray(source["frames"][f"{frame:06d}"]["parts"]["lid"]["T_world_from_part"], float)
        for frame in keyframes
    }

    raw_measured = {}
    frame_reports = {}
    with SceneRenderer(width, height) as renderer:
        for index, frame in enumerate(keyframes):
            print(f"\nkeyframe {frame:03d} ({index + 1}/{len(keyframes)})", flush=True)
            objective = FrameObjective(
                cfg, frame, mesh, scale, origin, renderer, width, height,
                int(options["min_mask_pixels"]),
                float(options["silhouette_chamfer_weight"]),
                float(options["rgb_edge_chamfer_weight"]),
            )
            base_eval = objective.evaluate(base_poses[frame])
            measured, trace = coordinate_search(
                objective, base_poses[frame],
                [float(x) for x in options["rotation_steps_deg"]],
                [float(x) for x in options["translation_steps_m"]],
            )
            best_eval = objective.evaluate(measured)
            sensitivity = {}
            sensitivity_deg = float(options["sensitivity_deg"])
            for axis in range(3):
                minus = objective.evaluate(perturb(measured, axis, -sensitivity_deg, True))["score"]
                plus = objective.evaluate(perturb(measured, axis, sensitivity_deg, True))["score"]
                sensitivity["xyz"[axis]] = {
                    "delta_deg": sensitivity_deg,
                    "score_drop": float(best_eval["score"] - 0.5 * (minus + plus)),
                    "minus_score": minus, "plus_score": plus,
                }
            raw_correction = Rotation.from_matrix(
                base_poses[frame][:3, :3].T @ measured[:3, :3]).as_rotvec()
            observability_threshold = float(
                options.get("rotation_observability_min_score_drop", 0.0))
            gated_correction = raw_correction.copy()
            observable_axes = {}
            for axis, name in enumerate("xyz"):
                observable_axes[name] = sensitivity[name]["score_drop"] >= observability_threshold
                if not observable_axes[name]:
                    gated_correction[axis] = 0.0
            accepted = measured.copy()
            accepted[:3, :3] = base_poses[frame][:3, :3] @ Rotation.from_rotvec(
                gated_correction).as_matrix()
            accepted_eval = objective.evaluate(accepted)
            raw_measured[frame] = accepted
            frame_reports[f"{frame:06d}"] = {
                "base_score": base_eval["score"],
                "unconstrained_refined_score": best_eval["score"],
                "refined_score": accepted_eval["score"],
                "score_gain": accepted_eval["score"] - base_eval["score"],
                "translation_correction_m": (accepted[:3, 3] - base_poses[frame][:3, 3]).tolist(),
                "translation_correction_norm_m": float(np.linalg.norm(
                    accepted[:3, 3] - base_poses[frame][:3, 3])),
                "unconstrained_rotation_correction_rotvec_deg": np.rad2deg(
                    raw_correction).tolist(),
                "rotation_correction_deg": float(np.rad2deg(
                    np.linalg.norm(gated_correction))),
                "rotation_correction_rotvec_deg": np.rad2deg(gated_correction).tolist(),
                "axis_sensitivity": sensitivity,
                "observable_rotation_axes": observable_axes,
                "observability_threshold": observability_threshold,
                "observing_views": accepted_eval["observing_views"],
                "base_per_view": base_eval["per_view"],
                "unconstrained_refined_per_view": best_eval["per_view"],
                "refined_per_view": accepted_eval["per_view"],
                "search_trace": trace,
            }
            print(f"score {base_eval['score']:.5f} -> {accepted_eval['score']:.5f} "
                  f"(raw {best_eval['score']:.5f}) | "
                  f"dt={frame_reports[f'{frame:06d}']['translation_correction_norm_m']*1000:.1f}mm "
                  f"dr={frame_reports[f'{frame:06d}']['rotation_correction_deg']:.1f}deg", flush=True)

    measured = smooth_keyframe_corrections(base_poses, raw_measured)
    motion_start, motion_end = keyframes[0], keyframes[-1]
    all_base = {
        frame: np.asarray(source["frames"][f"{frame:06d}"]["parts"]["lid"]
                          ["T_world_from_part"], float)
        for frame in range(motion_start, motion_end + 1)
    }
    motion_poses = interpolate_corrections(
        base_poses, measured, all_base, motion_start, motion_end)
    refined = copy.deepcopy(source)
    refined["config"] = str(config_path)
    refined.setdefault("provenance", {})["derived_from_trajectory"] = str(source_path)
    refined["provenance"]["lid_multiview_se3_refinement"] = str(
        output / "diagnostics" / "lid_se3_refinement.json")
    refined["provenance"]["point_cloud_variant_for_future_depth_terms"] = cfg.get("point_cloud_variant")
    T_WB = np.asarray(refined["frames"][sorted(refined["frames"])[0]]["parts"]["body"]
                      ["T_world_from_part"], float)
    previous = {part: None for part in refined["parts"]}
    for key, frame_record in refined["frames"].items():
        frame = int(key)
        lid_record = frame_record["parts"]["lid"]
        if frame < motion_start:
            pose = measured[motion_start]
            source_label, measurement = "lid_se3_static_endpoint", "static_propagated"
        elif frame > motion_end:
            pose = measured[motion_end]
            source_label, measurement = "lid_se3_static_endpoint", "static_propagated"
        else:
            pose = motion_poses[frame]
            if frame in measured:
                source_label, measurement = "lid_multiview_se3", "measured_keyframe"
            else:
                source_label, measurement = "lid_multiview_se3_interpolated", "interpolated"
        lid_record["T_world_from_part"] = pose.tolist()
        lid_record["S_world_from_raw_mesh"] = similarity_from_rigid(pose, scale, origin).tolist()
        lid_record["source"] = source_label
        lid_record["pose_measurement"] = measurement
        if key in frame_reports:
            drops = [value["score_drop"] for value in frame_reports[key]["axis_sensitivity"].values()]
            lid_record["rotation_observability"] = {
                "min_axis_score_drop_5deg": float(min(drops)),
                "mean_axis_score_drop_5deg": float(np.mean(drops)),
                "observable_axes": frame_reports[key]["observable_rotation_axes"],
            }

        for part in refined["parts"]:
            record = frame_record["parts"][part]
            T_WP = np.asarray(record["T_world_from_part"], float)
            T_BP = np.linalg.inv(T_WB) @ T_WP
            record["T_body_from_part"] = T_BP.tolist()
            record["translation_body_m"] = T_BP[:3, 3].tolist()
            record["quaternion_body_xyzw"] = Rotation.from_matrix(T_BP[:3, :3]).as_quat().tolist()
            if previous[part] is None:
                record["translation_step_m"] = 0.0
                record["rotation_step_deg"] = 0.0
            else:
                delta = np.linalg.inv(previous[part]) @ T_WP
                record["translation_step_m"] = float(np.linalg.norm(delta[:3, 3]))
                record["rotation_step_deg"] = float(np.rad2deg(
                    Rotation.from_matrix(delta[:3, :3]).magnitude()))
            previous[part] = T_WP

    gains = [value["score_gain"] for value in frame_reports.values()]
    report = {
        "config": str(config_path), "source_trajectory": str(source_path),
        "output_trajectory": str(output / "pose" / "trajectory.json"),
        "objective": "six-view full silhouette + textured RGB edges",
        "resolution": [width, height], "keyframes": keyframes,
        "rotation_steps_deg": options["rotation_steps_deg"],
        "translation_steps_m": options["translation_steps_m"],
        "rotation_observability_min_score_drop": options.get(
            "rotation_observability_min_score_drop", 0.0),
        "summary": {
            "mean_score_gain": float(np.mean(gains)),
            "median_score_gain": float(np.median(gains)),
            "improved_keyframes": int(np.sum(np.asarray(gains) > 0)),
        },
        "frames": frame_reports,
    }
    write_json(output / "diagnostics" / "lid_se3_refinement.json", report)
    write_json(output / "pose" / "trajectory.json", refined)
    write_csv(refined, output / "pose" / "trajectory.csv")
    print(f"\nwrote {output / 'pose' / 'trajectory.json'}")


if __name__ == "__main__":
    main()
