#!/usr/bin/env python
"""Solve one body-relative 6D-pose trajectory from synchronized multi-view clouds.

The solver is intentionally object-agnostic.  Scene knowledge (static/dynamic
ranges, anchor frames, symmetry axes) lives in JSON.  Mesh scale is calibrated
once; every exported per-frame pose is rigid and body-relative.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import small_gicp
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.mesh_align import align_mesh_to_cloud, read_ply_xyz
from common.normalized_recon import load_recon, scale_intrinsics


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def decompose_similarity(S: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    scale = float(np.cbrt(max(np.linalg.det(S[:3, :3]), 1e-12)))
    return scale, S[:3, :3] / scale, S[:3, 3].copy()


def similarity(scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    S = np.eye(4)
    S[:3, :3] = scale * R
    S[:3, 3] = t
    return S


def rigid_from_similarity(S: np.ndarray, origin_raw: np.ndarray) -> np.ndarray:
    scale, R, t = decompose_similarity(S)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = scale * (R @ origin_raw) + t
    return T


def similarity_from_rigid(T: np.ndarray, scale: float, origin_raw: np.ndarray) -> np.ndarray:
    S = np.eye(4)
    S[:3, :3] = scale * T[:3, :3]
    S[:3, 3] = T[:3, 3] - scale * (T[:3, :3] @ origin_raw)
    return S


def voxel_unique(points: np.ndarray, voxel: float = 0.002) -> np.ndarray:
    if not len(points):
        return points
    cells = np.floor(points / voxel).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def load_cloud(root: Path, frame: int, part: str) -> np.ndarray | None:
    path = root / f"{frame:06d}" / f"{part}.ply"
    if not path.exists():
        return None
    points = read_ply_xyz(str(path))
    return points if len(points) >= 30 else None


def fused_cloud(root: Path, frames: list[int], part: str, max_points: int,
                seed: int) -> np.ndarray:
    clouds = [load_cloud(root, frame, part) for frame in frames]
    clouds = [cloud for cloud in clouds if cloud is not None]
    if not clouds:
        raise RuntimeError(f"no cloud for {part} in frames {frames}")
    points = voxel_unique(np.concatenate(clouds), 0.002)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points


def subsample(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(points) <= maximum:
        return np.ascontiguousarray(points, dtype=np.float64)
    index = np.random.default_rng(seed).choice(len(points), maximum, replace=False)
    return np.ascontiguousarray(points[index], dtype=np.float64)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def transform_angle(T: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(T[:3, :3]).magnitude()))


def pair_quality(source: np.ndarray, target: np.ndarray, T: np.ndarray) -> dict:
    aligned = transform_points(source, T)
    distances, _ = cKDTree(target).query(aligned, k=1)
    threshold = 0.008
    inliers = distances <= threshold
    return {
        "fitness_8mm": float(inliers.mean()),
        "inlier_rmse_m": float(np.sqrt(np.mean(distances[inliers] ** 2))) if inliers.any() else None,
        "median_nn_m": float(np.median(distances)),
        "trimmed_rmse_m": float(np.sqrt(np.mean(np.sort(distances)[:max(30, int(0.8 * len(distances)))] ** 2))),
        "n_source": int(len(source)),
        "n_target": int(len(target)),
    }


def multiscale_gicp(source: np.ndarray, target: np.ndarray, init: np.ndarray,
                    cfg: dict) -> tuple[np.ndarray, dict]:
    T = np.asarray(init, dtype=np.float64).copy()
    stages = []
    for voxel, max_dist in zip(cfg["voxel_sizes_m"], cfg["max_correspondence_m"]):
        target_pp, target_tree = small_gicp.preprocess_points(
            target, downsampling_resolution=float(voxel), num_neighbors=20, num_threads=1)
        source_pp, _ = small_gicp.preprocess_points(
            source, downsampling_resolution=float(voxel), num_neighbors=20, num_threads=1)
        result = small_gicp.align(
            target_pp, source_pp, target_tree, init_T_target_source=T,
            registration_type="GICP", max_correspondence_distance=float(max_dist),
            max_iterations=int(cfg["max_iterations"]), translation_epsilon=1e-6,
            num_threads=1)
        T = np.asarray(result.T_target_source, dtype=np.float64)
        stages.append({
            "voxel_m": float(voxel), "max_correspondence_m": float(max_dist),
            "iterations": int(result.iterations), "converged": bool(result.converged),
            "error": float(result.error),
        })
    quality = pair_quality(source, target, T)
    quality["stages"] = stages
    return T, quality


def best_pair_registration(source: np.ndarray, target: np.ndarray,
                           previous: np.ndarray | None, cfg: dict,
                           seed: int) -> tuple[np.ndarray, dict]:
    maximum = int(cfg["max_points"])
    source = subsample(source, maximum, seed)
    target = subsample(target, maximum, seed + 1)
    centroid = np.eye(4)
    centroid[:3, 3] = target.mean(0) - source.mean(0)
    candidates = [("identity", np.eye(4)), ("centroid", centroid)]
    if previous is not None:
        candidates.insert(0, ("previous", previous))

    # A cheap nearest-neighbour score selects the most plausible basin before GICP.
    tree = cKDTree(target)
    scored = []
    probe = subsample(source, min(3000, len(source)), seed + 2)
    for name, init in candidates:
        distances, _ = tree.query(transform_points(probe, init), k=1)
        scored.append((float(np.median(distances)), name, init))
    _, init_name, init = min(scored, key=lambda item: item[0])
    T, quality = multiscale_gicp(source, target, init, cfg)
    quality["init"] = init_name
    quality["translation_m"] = float(np.linalg.norm(T[:3, 3]))
    quality["rotation_deg"] = transform_angle(T)
    quality["candidate_median_nn_m"] = {name: score for score, name, _ in scored}
    return T, quality


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    return T


def symmetry_align_pose(T: np.ndarray, reference: np.ndarray, axis_raw: np.ndarray) -> np.ndarray:
    best = None
    for angle in np.deg2rad(np.arange(0, 360, 5)):
        candidate = T @ axis_rotation(axis_raw, float(angle))
        error = Rotation.from_matrix(reference[:3, :3].T @ candidate[:3, :3]).magnitude()
        if best is None or error < best[0]:
            best = (float(error), candidate)
    return best[1]


def interpolate_correction(delta: np.ndarray, u: float) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(
        Rotation.from_matrix(delta[:3, :3]).as_rotvec() * u).as_matrix()
    value[:3, 3] = delta[:3, 3] * u
    return value


def smooth_poses(poses: dict[int, np.ndarray], start: int, end: int,
                 passes: int = 2) -> None:
    for _ in range(passes):
        old = {frame: poses[frame].copy() for frame in range(start, end + 1)}
        for frame in range(start + 1, end):
            weights = np.asarray([1.0, 4.0, 1.0])
            translations = np.stack([old[frame - 1][:3, 3], old[frame][:3, 3], old[frame + 1][:3, 3]])
            rotations = Rotation.from_matrix(np.stack([
                old[frame - 1][:3, :3], old[frame][:3, :3], old[frame + 1][:3, :3]]))
            T = old[frame].copy()
            T[:3, 3] = np.average(translations, axis=0, weights=weights)
            T[:3, :3] = rotations.mean(weights=weights).as_matrix()
            poses[frame] = T


def track_model_translation(part: str, mesh: trimesh.Trimesh, scale: float,
                            origin_raw: np.ndarray, start: int, end: int,
                            start_pose: np.ndarray, end_pose: np.ndarray,
                            cloud_root: Path, seed: int = 0) -> tuple[dict[int, np.ndarray], dict]:
    """Track a near-axisymmetric object with a mesh-gated robust translation ICP.

    The lid remains almost horizontal in this sequence.  Locking its orientation
    to the well-observed table anchor prevents the hand-contaminated depth inside
    the mask from manufacturing large rotations.
    """
    rng = np.random.default_rng(seed)
    raw, _ = trimesh.sample.sample_surface(mesh, 30000)
    canonical = scale * (np.asarray(raw, float) - origin_raw)
    if len(canonical) > 18000:
        canonical = canonical[rng.choice(len(canonical), 18000, replace=False)]
    R = start_pose[:3, :3].copy()
    final = end_pose.copy()
    final[:3, :3] = R
    poses = {start: start_pose.copy()}
    records = {}
    velocity = np.zeros(3)
    for frame in range(start + 1, end + 1):
        cloud = load_cloud(cloud_root, frame, part)
        predicted = poses[frame - 1].copy()
        predicted[:3, :3] = R
        predicted[:3, 3] += np.clip(velocity, -0.035, 0.035)
        if cloud is None:
            poses[frame] = predicted
            records[f"{frame:06d}"] = {"status": "inferred", "n_gated": 0}
            continue
        cloud = subsample(cloud, 18000, seed + frame)
        model_world = canonical @ R.T + predicted[:3, 3]
        tree = cKDTree(model_world)
        delta = np.zeros(3)
        n_gated = 0
        median_distance = None
        for _ in range(12):
            moved = cloud + delta
            distances, indices = tree.query(moved, k=1)
            gate = distances < 0.065
            if gate.sum() < 100:
                break
            threshold = np.quantile(distances[gate], 0.60)
            keep = gate & (distances <= threshold)
            residual = model_world[indices[keep]] - moved[keep]
            increment = np.median(residual, axis=0)
            delta += 0.75 * increment
            n_gated = int(keep.sum())
            median_distance = float(np.median(distances[keep]))
            if np.linalg.norm(increment) < 2e-4:
                break
        correction = -delta
        norm = float(np.linalg.norm(correction))
        if norm > 0.065 or n_gated < 100:
            correction[:] = 0.0
            accepted = False
        else:
            accepted = True
        current = predicted.copy()
        current[:3, 3] += correction
        new_velocity = current[:3, 3] - poses[frame - 1][:3, 3]
        velocity = 0.65 * velocity + 0.35 * new_velocity
        poses[frame] = current
        records[f"{frame:06d}"] = {
            "status": "measured" if accepted else "motion_prior",
            "n_gated": n_gated, "median_distance_m": median_distance,
            "correction_m": correction.tolist(),
        }

    # The independently fitted final anchor fixes accumulated translation drift;
    # orientation stays in the continuous symmetry representative from the start.
    drift = final[:3, 3] - poses[end][:3, 3]
    for frame in range(start, end + 1):
        u = (frame - start) / max(end - start, 1)
        poses[frame][:3, 3] += u * drift
    smooth_poses(poses, start, end, passes=3)
    poses[start] = start_pose.copy()
    poses[end] = final.copy()
    return poses, records


def project_bbox(points_part: np.ndarray, center_world: np.ndarray,
                 R_world_part: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Project a fixed-orientation mesh and return xmin,ymin,xmax,ymax."""
    world = points_part @ R_world_part.T + center_world
    camera = world @ E[:, :3].T + E[:, 3]
    uvw = camera @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return np.concatenate([uv.min(axis=0), uv.max(axis=0)])


def lid_mask_bboxes(mask_root: Path, frame: int, part_id: int,
                    views: list[str]) -> list[tuple[np.ndarray, tuple[int, int]] | None]:
    """Load full-resolution per-view boxes without treating clipped edges as data."""
    from PIL import Image

    result = []
    for view in views:
        labels = np.asarray(Image.open(mask_root / f"{frame:06d}" / f"{view}.png"))
        y, x = np.where(labels == part_id)
        if len(x) < 1000:
            result.append(None)
            continue
        bbox = np.asarray([x.min(), y.min(), x.max(), y.max()], dtype=float)
        result.append((bbox, labels.shape))
    return result


def track_mask_bbox_translation(
    part: str,
    mesh: trimesh.Trimesh,
    scale: float,
    origin_raw: np.ndarray,
    start: int,
    end: int,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    mask_root: Path,
    part_id: int,
    views: list[str],
    cfg: dict,
) -> tuple[dict[int, np.ndarray], dict]:
    """Track a fixed-orientation lid from its six-view silhouette envelopes.

    Depth inside a correct lid mask can still belong to an occluding hand.  The
    four silhouette envelope lines, unlike a mask centroid, remain useful when
    the middle of the object is occluded.  Known mesh extent supplies depth even
    when the lid leaves most views.  Per-view envelope bias is calibrated at the
    two clean anchor poses and interpolated over the motion.
    """
    canonical = scale * (np.asarray(mesh.vertices, float) - origin_raw)
    R = start_pose[:3, :3].copy()
    final = end_pose.copy()
    final[:3, :3] = R
    backend = cfg["recon_backend"]

    camera_cache: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    bbox_cache: dict[int, list[tuple[np.ndarray, tuple[int, int]] | None]] = {}

    def observations(frame: int):
        if frame not in bbox_cache:
            boxes = lid_mask_bboxes(mask_root, frame, part_id, views)
            recon = load_recon(cfg, f"{frame:06d}", backend=backend)
            cameras = []
            for index, box in enumerate(boxes):
                shape = box[1] if box is not None else (1080, 1920)
                K = scale_intrinsics(recon["intrinsics"][index], recon["depth_hw"], shape)
                cameras.append((K, recon["extrinsics"][index]))
            bbox_cache[frame] = boxes
            camera_cache[frame] = cameras
        return bbox_cache[frame], camera_cache[frame]

    def anchor_bias(anchor_pose: np.ndarray, frames: list[int]) -> np.ndarray:
        per_view: list[list[np.ndarray]] = [[] for _ in views]
        for frame in frames:
            boxes, cameras = observations(frame)
            for index, item in enumerate(boxes):
                if item is None:
                    continue
                observed, _ = item
                predicted = project_bbox(canonical, anchor_pose[:3, 3], R, *cameras[index])
                per_view[index].append(predicted - observed)
        return np.stack([
            np.median(values, axis=0) if values else np.zeros(4)
            for values in per_view
        ])

    tracker_cfg = cfg["states"][part].get("mask_bbox_tracking", {})
    start_bias_frames = [int(x) for x in tracker_cfg.get("start_bias_frames", [start])]
    end_bias_frames = [int(x) for x in tracker_cfg.get("end_bias_frames", [end])]
    bias_start = anchor_bias(start_pose, start_bias_frames)
    bias_end = anchor_bias(final, end_bias_frames)

    poses = {start: start_pose.copy()}
    records = {}
    velocity = np.zeros(3)
    for frame in range(start + 1, end + 1):
        boxes, cameras = observations(frame)
        u = (frame - start) / max(end - start, 1)
        bias = (1.0 - u) * bias_start + u * bias_end
        previous = poses[frame - 1][:3, 3]
        predicted_motion = previous + np.clip(velocity, -0.035, 0.035)
        linear_prior = (1.0 - u) * start_pose[:3, 3] + u * final[:3, 3]
        n_edges = 0

        def residual(center: np.ndarray) -> np.ndarray:
            nonlocal n_edges
            values = []
            edge_count = 0
            for index, item in enumerate(boxes):
                if item is None:
                    continue
                observed, (height, width) = item
                predicted = project_bbox(canonical, center, R, *cameras[index])
                # An envelope touching the image boundary is censored, not an
                # object edge; retain all other sides of the same view.
                valid = np.asarray([
                    observed[0] > 1, observed[1] > 1,
                    observed[2] < width - 2, observed[3] < height - 2,
                ])
                values.extend((predicted[valid] - observed[valid] - bias[index, valid]).tolist())
                edge_count += int(valid.sum())
            n_edges = edge_count
            # Weak motion priors stabilize the short interval where the lid is
            # almost entirely outside five cameras.  Silhouette edges dominate.
            values.extend((3.0 * (center - predicted_motion) / 0.015).tolist())
            values.extend((0.35 * (center - linear_prior) / 0.03).tolist())
            return np.asarray(values)

        result = least_squares(
            residual, predicted_motion, loss="soft_l1", f_scale=10.0,
            max_nfev=80, xtol=1e-7, ftol=1e-7,
        )
        center = result.x
        step = center - previous
        if np.linalg.norm(step) > 0.06:
            center = previous + step * (0.06 / np.linalg.norm(step))
        velocity = 0.65 * velocity + 0.35 * (center - previous)
        pose = start_pose.copy()
        pose[:3, 3] = center
        poses[frame] = pose
        edge_residual = residual(center)[:n_edges]
        records[f"{frame:06d}"] = {
            "status": "measured" if n_edges >= 4 else "motion_prior",
            "valid_silhouette_edges": n_edges,
            "median_edge_error_px": (float(np.median(np.abs(edge_residual)))
                                     if len(edge_residual) else None),
            "translation_step_m": float(np.linalg.norm(center - previous)),
        }

    # Smooth only translations; the selected symmetry representative is fixed.
    smooth_poses(poses, start, end, passes=2)
    for pose in poses.values():
        pose[:3, :3] = R
    poses[start] = start_pose.copy()
    poses[end] = final.copy()
    records["calibration"] = {
        "start_bias_frames": start_bias_frames,
        "end_bias_frames": end_bias_frames,
        "start_edge_bias_px": bias_start.tolist(),
        "end_edge_bias_px": bias_end.tolist(),
    }
    return poses, records


def calibration_frames(state: dict, anchor: int) -> list[int]:
    windows = state.get("anchor_windows", {})
    return [int(x) for x in windows.get(str(anchor), [anchor])]


def fit_anchor(mesh: trimesh.Trimesh, cloud_root: Path, state: dict,
               anchor: int, part: str, seed: int) -> dict:
    frames = calibration_frames(state, anchor)
    cloud = fused_cloud(cloud_root, frames, part, 40000, seed)
    fit = align_mesh_to_cloud(mesh, cloud, n_mesh_sample=40000, n_obs_max=16000,
                              coarse_iters=30, fine_iters=100, seed=seed)
    return {
        "frame": anchor, "frames": frames, "n_cloud_points": int(len(cloud)),
        "S_world_from_raw": fit["T_mesh_to_world"], "scale": float(fit["scale"]),
        "fit_rmse_m": float(fit["fit_rmse"]), "icp_cost": float(fit["icp_cost"]),
    }


def solve_dynamic(part: str, start: int, end: int, start_pose: np.ndarray,
                  end_pose: np.ndarray, cloud_root: Path, reg_cfg: dict,
                  axis_raw: np.ndarray | None) -> tuple[dict[int, np.ndarray], dict]:
    clouds = {frame: load_cloud(cloud_root, frame, part) for frame in range(start, end + 1)}
    observed = [frame for frame, cloud in clouds.items() if cloud is not None]
    if observed[0] != start or observed[-1] != end:
        raise RuntimeError(f"{part}: dynamic endpoints must have clouds, got {observed[0]}..{observed[-1]}")

    poses: dict[int, np.ndarray] = {start: start_pose.copy()}
    registrations = {}
    previous_pair = None
    previous_frame = start
    for frame in observed[1:]:
        gap = frame - previous_frame
        pair, quality = best_pair_registration(
            clouds[frame], clouds[previous_frame], previous_pair, reg_cfg,
            seed=1000 + frame)
        # Reject catastrophic fits.  A centroid-only bridge is safer and remains auditable.
        rejected = (quality["translation_m"] > 0.12 * gap
                    or quality["rotation_deg"] > 30.0 * gap
                    or quality["fitness_8mm"] < 0.15)
        if rejected:
            # Partial clouds from different cameras can have centroids tens of
            # centimetres apart although the rigid object moved smoothly.  Reuse
            # the last accepted motion instead of turning that visibility change
            # into object motion.  Matrix power spans a short missing interval.
            pair = (np.linalg.matrix_power(previous_pair, gap)
                    if previous_pair is not None else np.eye(4))
            quality["rejected"] = True
            quality["fallback"] = "constant_velocity" if previous_pair is not None else "identity"
        else:
            quality["rejected"] = False
            previous_pair = pair
        current = np.linalg.inv(pair) @ poses[previous_frame]
        if axis_raw is not None and reg_cfg.get("symmetry_lock", True):
            current = symmetry_align_pose(current, poses[previous_frame], axis_raw)
        poses[frame] = current
        registrations[f"{frame:06d}_to_{previous_frame:06d}"] = {
            "source": frame, "target": previous_frame,
            "T_target_from_source": pair.tolist(), "quality": quality,
        }
        print(f"{part} {frame:03d}->{previous_frame:03d} "
              f"fitness={quality['fitness_8mm']:.3f} "
              f"t={quality['translation_m']:.3f}m r={quality['rotation_deg']:.1f}deg "
              f"rejected={quality['rejected']}", flush=True)
        previous_frame = frame

    # Interpolate only frames with no multi-view cloud between measured poses.
    for left, right in zip(observed[:-1], observed[1:]):
        if right == left + 1:
            continue
        rotations = Rotation.from_matrix(np.stack([poses[left][:3, :3], poses[right][:3, :3]]))
        slerp = Slerp([0.0, 1.0], rotations)
        for frame in range(left + 1, right):
            u = (frame - left) / (right - left)
            T = np.eye(4)
            T[:3, :3] = slerp([u]).as_matrix()[0]
            T[:3, 3] = (1.0 - u) * poses[left][:3, 3] + u * poses[right][:3, 3]
            poses[frame] = T

    # Distribute endpoint drift across the measured trajectory.
    predicted_end = poses[end]
    if axis_raw is not None:
        end_pose = symmetry_align_pose(end_pose, predicted_end, axis_raw)
    delta = end_pose @ np.linalg.inv(predicted_end)
    for frame in range(start, end + 1):
        u = (frame - start) / max(end - start, 1)
        poses[frame] = interpolate_correction(delta, u) @ poses[frame]
    poses[start] = start_pose.copy()
    poses[end] = end_pose.copy()
    smooth_poses(poses, start, end, passes=2)
    poses[start] = start_pose.copy()
    poses[end] = end_pose.copy()
    return poses, registrations


def pose_record(T_WP: np.ndarray, T_WB: np.ndarray, S_WP: np.ndarray,
                state: str, source: str, n_views: int) -> dict:
    T_BP = np.linalg.inv(T_WB) @ T_WP
    quaternion = Rotation.from_matrix(T_BP[:3, :3]).as_quat()
    return {
        "state": state, "source": source, "observing_views": n_views,
        "T_world_from_part": T_WP.tolist(),
        "T_body_from_part": T_BP.tolist(),
        "S_world_from_raw_mesh": S_WP.tolist(),
        "translation_body_m": T_BP[:3, 3].tolist(),
        "quaternion_body_xyzw": quaternion.tolist(),
    }


def count_observing_views(mask_root: Path, frame: int, part_id: int, views: list[str]) -> int:
    from PIL import Image
    total = 0
    for view in views:
        labels = np.asarray(Image.open(mask_root / f"{frame:06d}" / f"{view}.png"))
        total += int(np.any(labels == part_id))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_multiview_111.json"))
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--reuse-calibration", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_json(config_path)
    output = Path(cfg["output_root"])
    cloud_root = output / "parts_ply" / cfg["recon_backend"]
    calibration_path = output / "pose" / "calibration.json"
    mesh_dir = Path(cfg["mesh_dir"])
    meshes = {part: trimesh.load(mesh_dir / f"{part}.glb", force="mesh") for part in cfg["parts"]}
    origins = {part: np.asarray(meshes[part].centroid, float) for part in cfg["parts"]}

    if args.reuse_calibration and calibration_path.exists():
        calibration_json = load_json(calibration_path)
        anchors = {
            part: {int(frame): np.asarray(value["S_world_from_raw"], float)
                   for frame, value in values.items()}
            for part, values in calibration_json["anchors"].items()
        }
        scales = {part: float(value) for part, value in calibration_json["scales"].items()}
    else:
        anchor_info: dict[str, dict] = {}
        anchors: dict[str, dict[int, np.ndarray]] = {}
        scales: dict[str, float] = {}
        for index, part in enumerate(cfg["parts"]):
            state = cfg["states"][part]
            frames = ([int(state["calibration_frames"][len(state["calibration_frames"]) // 2])]
                      if part == cfg["reference_part"] else [int(x) for x in state["anchor_frames"]])
            anchors[part] = {}
            anchor_info[part] = {}
            fits = []
            for anchor in frames:
                if part == cfg["reference_part"]:
                    local_state = dict(state)
                    local_state["anchor_windows"] = {str(anchor): state["calibration_frames"]}
                else:
                    local_state = state
                fit = fit_anchor(meshes[part], cloud_root, local_state, anchor, part,
                                 seed=31 + 10 * index + anchor)
                fits.append(fit)
                anchor_info[part][str(anchor)] = {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in fit.items()
                }
                print(f"calibration {part}@{anchor}: scale={fit['scale']:.6f} "
                      f"rmse={fit['fit_rmse_m']:.5f}", flush=True)
            scale = float(state.get("scale_prior", np.median([fit["scale"] for fit in fits])))
            scales[part] = scale
            for fit in fits:
                _, R, t = decompose_similarity(fit["S_world_from_raw"])
                anchors[part][fit["frame"]] = similarity(scale, R, t)
                anchor_info[part][str(fit["frame"])]["S_world_from_raw"] = anchors[part][fit["frame"]].tolist()
                anchor_info[part][str(fit["frame"])]["fixed_scale"] = scale
        calibration_json = {
            "config": str(config_path), "coordinate_convention": "world-to-camera extrinsics; column transforms",
            "scales": scales, "raw_mesh_origins": {p: origins[p].tolist() for p in origins},
            "anchors": anchor_info,
        }
        save_json(calibration_path, calibration_json)

    # A validated trajectory may provide a stronger assembly calibration than a
    # fresh partial-cloud similarity fit.  The mechanism is optional and keeps
    # the general solver reusable for scenes without such a prior.
    prior_cache = {}
    for part in cfg["parts"]:
        state = cfg["states"][part]
        prior_path = state.get("prior_trajectory")
        if prior_path:
            prior_cache[part] = load_json(Path(prior_path))
    if cfg["states"][cfg["reference_part"]].get("method") == "similarity_prior":
        body_prior = prior_cache[cfg["reference_part"]]
        body = cfg["reference_part"]
        S = np.asarray(body_prior["S_world_from_body_raw"], float)
        scales[body] = float(body_prior["body_scale"])
        anchors[body] = {start_frame if 'start_frame' in locals() else 0: S}
    for part in cfg["parts"]:
        if cfg["states"][part].get("method") == "trajectory_prior":
            prior = prior_cache[part]
            scales[part] = float(prior["inner_scale"])

    if args.calibrate_only:
        return

    body = cfg["reference_part"]
    body_anchor = next(iter(anchors[body].values()))
    T_WB = rigid_from_similarity(body_anchor, origins[body])
    start_frame, end_frame = int(cfg["frames"]["start"]), int(cfg["frames"]["end"])
    world_poses: dict[str, dict[int, np.ndarray]] = {}
    all_registrations = {}

    body_poses = {frame: T_WB.copy() for frame in range(start_frame, end_frame + 1)}
    world_poses[body] = body_poses
    for part in cfg["parts"]:
        if part == body:
            continue
        state_cfg = cfg["states"][part]
        part_poses: dict[int, np.ndarray] = {}
        part_regs = {}
        axis = np.asarray(state_cfg["symmetry_axis_raw"], float) if "symmetry_axis_raw" in state_cfg else None
        method = state_cfg.get("method", "cloud_registration")
        if method == "trajectory_prior":
            prior = prior_cache[part]
            prior_frames = prior["frames"]
            last_key = sorted(prior_frames)[-1]
            for frame in range(start_frame, end_frame + 1):
                key = f"{frame:06d}"
                source_key = key if key in prior_frames else last_key
                S = np.asarray(prior_frames[source_key]["S_world_from_inner_raw"], float)
                part_poses[frame] = rigid_from_similarity(S, origins[part])
            part_regs["prior"] = {"path": state_cfg["prior_trajectory"],
                                  "available_through": int(last_key)}
        for dynamic_start, dynamic_end in ([] if method == "trajectory_prior" else state_cfg["dynamic_ranges"]):
            dynamic_start, dynamic_end = int(dynamic_start), int(dynamic_end)
            anchor_frames = sorted(anchors[part])
            start_anchor = min(anchor_frames, key=lambda value: abs(value - dynamic_start))
            end_anchor = min(anchor_frames, key=lambda value: abs(value - dynamic_end))
            T_start = rigid_from_similarity(anchors[part][start_anchor], origins[part])
            T_end = rigid_from_similarity(anchors[part][end_anchor], origins[part])
            if method == "model_tracking":
                poses, registrations = track_model_translation(
                    part, meshes[part], scales[part], origins[part],
                    dynamic_start, dynamic_end, T_start, T_end, cloud_root,
                    seed=700 + dynamic_start)
            elif method == "mask_bbox_tracking":
                poses, registrations = track_mask_bbox_translation(
                    part, meshes[part], scales[part], origins[part],
                    dynamic_start, dynamic_end, T_start, T_end,
                    Path(cfg["masks_dir"]), int(cfg["part_ids"][part]),
                    cfg["views"], cfg)
            else:
                poses, registrations = solve_dynamic(
                    part, dynamic_start, dynamic_end, T_start, T_end,
                    cloud_root, cfg["registration"], axis)
            part_poses.update(poses)
            part_regs.update(registrations)
        for static_start, static_end in ([] if method == "trajectory_prior" else state_cfg["static_ranges"]):
            static_start, static_end = int(static_start), int(static_end)
            # Reuse an already-solved boundary pose when a static range touches a
            # dynamic range.  This preserves the chosen symmetry representative
            # and prevents an invisible yaw jump at the state transition.
            boundary = next((frame for frame in (static_start, static_end)
                             if frame in part_poses), None)
            if boundary is not None:
                T_static = part_poses[boundary].copy()
            else:
                anchor_frame = min(anchors[part], key=lambda value: abs(value - (static_start + static_end) / 2))
                T_static = rigid_from_similarity(anchors[part][anchor_frame], origins[part])
            for frame in range(static_start, static_end + 1):
                if frame not in part_poses:
                    part_poses[frame] = T_static.copy()
        missing = sorted(set(range(start_frame, end_frame + 1)) - set(part_poses))
        if missing:
            raise RuntimeError(f"{part}: uncovered frames {missing}")
        world_poses[part] = part_poses
        all_registrations[part] = part_regs

    mask_root = Path(cfg["masks_dir"])
    trajectory = {
        "config": str(config_path),
        "conventions": {
            "T_world_from_part": "rigid pose of the canonical part frame; origin is raw mesh centroid",
            "T_body_from_part": "inv(T_world_from_body) @ T_world_from_part",
            "S_world_from_raw_mesh": "render transform for raw GLB; includes fixed per-part scale",
            "quaternion": "xyzw",
        },
        "parts": cfg["parts"], "reference_part": body,
        "scales": scales, "raw_mesh_origins": {p: origins[p].tolist() for p in origins},
        "frames": {},
    }
    csv_rows = []
    previous = {part: None for part in cfg["parts"]}
    for frame in range(start_frame, end_frame + 1):
        key = f"{frame:06d}"
        trajectory["frames"][key] = {"parts": {}}
        for part in cfg["parts"]:
            T_WP = world_poses[part][frame]
            S_WP = similarity_from_rigid(T_WP, scales[part], origins[part])
            views = count_observing_views(mask_root, frame, int(cfg["part_ids"][part]), cfg["views"])
            if part == body:
                state, source = "static", "body_anchor"
            else:
                in_dynamic = any(int(a) <= frame <= int(b) for a, b in cfg["states"][part]["dynamic_ranges"])
                if views == 0:
                    state, source = "inferred_unobservable", "interpolation"
                elif in_dynamic:
                    if cfg["states"][part].get("method") == "trajectory_prior":
                        state, source = "moving", "validated_trajectory_prior"
                    elif cfg["states"][part].get("method") == "model_tracking":
                        state, source = "moving", "multiview_mesh_gated_tracking"
                    elif cfg["states"][part].get("method") == "mask_bbox_tracking":
                        state, source = "moving", "multiview_silhouette_tracking"
                    else:
                        state, source = "moving", "multiview_cloud_registration"
                else:
                    source = ("validated_trajectory_prior"
                              if cfg["states"][part].get("method") == "trajectory_prior"
                              else "static_anchor")
                    state = "static"
            record = pose_record(T_WP, T_WB, S_WP, state, source, views)
            if previous[part] is None:
                step_m, step_deg = 0.0, 0.0
            else:
                delta = np.linalg.inv(previous[part]) @ T_WP
                step_m, step_deg = float(np.linalg.norm(delta[:3, 3])), transform_angle(delta)
            record["translation_step_m"] = step_m
            record["rotation_step_deg"] = step_deg
            trajectory["frames"][key]["parts"][part] = record
            q = record["quaternion_body_xyzw"]
            t = record["translation_body_m"]
            csv_rows.append([frame, part, state, source, views, *t, *q, step_m, step_deg])
            previous[part] = T_WP

    pose_dir = output / "pose"
    save_json(pose_dir / "trajectory.json", trajectory)
    save_json(pose_dir / "pair_registrations.json", all_registrations)
    with open(pose_dir / "trajectory.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "part", "state", "source", "observing_views",
                         "tx", "ty", "tz", "qx", "qy", "qz", "qw",
                         "translation_step_m", "rotation_step_deg"])
        writer.writerows(csv_rows)
    print(f"wrote {pose_dir / 'trajectory.json'}", flush=True)


if __name__ == "__main__":
    main()
