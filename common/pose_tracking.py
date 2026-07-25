"""Reusable pose tracking strategies for synchronized multi-view sequences."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from common.cloud_io import read_ply_xyz
from common.gicp import multiscale_gicp, subsample, transform_angle, voxel_unique
from common.normalized_recon import load_recon, recon_npz_path, scale_intrinsics
from common.pose_transforms import transform_points
from common.symmetry import SymmetrySpec, resolve_symmetric_pose


def load_part_cloud(
    root: Path, frame: int, part: str, minimum_points: int = 30
) -> np.ndarray | None:
    path = root / f"{frame:06d}" / f"{part}.ply"
    if not path.exists():
        return None
    points = read_ply_xyz(path)
    return points if len(points) >= minimum_points else None


def fuse_part_clouds(
    root: Path,
    frames: list[int],
    part: str,
    max_points: int,
    seed: int,
) -> np.ndarray:
    clouds = [load_part_cloud(root, frame, part) for frame in frames]
    clouds = [cloud for cloud in clouds if cloud is not None]
    if not clouds:
        raise RuntimeError(f"no cloud for {part} in frames {frames}")
    points = voxel_unique(np.concatenate(clouds), 0.002)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points


def align_symmetric_pose(
    pose: np.ndarray,
    reference: np.ndarray,
    axis_raw: np.ndarray,
    step_degrees: float = 5.0,
) -> np.ndarray:
    """Compatibility wrapper for continuous axial symmetry.

    New code should construct a :class:`common.symmetry.SymmetrySpec` and call
    :func:`common.symmetry.resolve_symmetric_pose` directly.
    """
    symmetry = SymmetrySpec(
        axis_raw=tuple(np.asarray(axis_raw, dtype=np.float64)),
        equivalence="continuous_axial",
        candidate_step_deg=float(step_degrees),
    )
    return resolve_symmetric_pose(
        pose,
        reference,
        symmetry,
        include_observation_ambiguities=False,
    ).pose


def interpolate_transform(delta: np.ndarray, fraction: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(
        Rotation.from_matrix(delta[:3, :3]).as_rotvec() * fraction
    ).as_matrix()
    result[:3, 3] = delta[:3, 3] * fraction
    return result


def smooth_pose_sequence(
    poses: dict[int, np.ndarray],
    start: int,
    end: int,
    passes: int = 2,
) -> None:
    for _ in range(passes):
        old = {frame: poses[frame].copy() for frame in range(start, end + 1)}
        for frame in range(start + 1, end):
            weights = np.asarray([1.0, 4.0, 1.0])
            translations = np.stack([
                old[frame - 1][:3, 3],
                old[frame][:3, 3],
                old[frame + 1][:3, 3],
            ])
            rotations = Rotation.from_matrix(np.stack([
                old[frame - 1][:3, :3],
                old[frame][:3, :3],
                old[frame + 1][:3, :3],
            ]))
            current = old[frame].copy()
            current[:3, 3] = np.average(
                translations, axis=0, weights=weights
            )
            current[:3, :3] = rotations.mean(weights=weights).as_matrix()
            poses[frame] = current


def _best_pair_registration(
    source: np.ndarray,
    target: np.ndarray,
    previous: np.ndarray | None,
    config: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    maximum = int(config["max_points"])
    source = subsample(source, maximum, seed)
    target = subsample(target, maximum, seed + 1)
    centroid = np.eye(4)
    centroid[:3, 3] = target.mean(0) - source.mean(0)
    candidates = [("identity", np.eye(4)), ("centroid", centroid)]
    if previous is not None:
        candidates.insert(0, ("previous", previous))
    tree = cKDTree(target)
    probe = subsample(source, min(3000, len(source)), seed + 2)
    scored = []
    for name, initial in candidates:
        distances, _ = tree.query(transform_points(probe, initial), k=1)
        scored.append((float(np.median(distances)), name, initial))
    _, initial_name, initial = min(scored, key=lambda item: item[0])
    transform, quality = multiscale_gicp(source, target, initial, config)
    quality["init"] = initial_name
    quality["translation_m"] = float(np.linalg.norm(transform[:3, 3]))
    quality["rotation_deg"] = transform_angle(transform)
    quality["candidate_median_nn_m"] = {
        name: score for score, name, _initial in scored
    }
    return transform, quality


def track_cloud_registration(
    part: str,
    start: int,
    end: int,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    cloud_root: Path,
    registration_config: dict,
    symmetry: SymmetrySpec | None,
) -> tuple[dict[int, np.ndarray], dict]:
    clouds = {
        frame: load_part_cloud(cloud_root, frame, part)
        for frame in range(start, end + 1)
    }
    observed = [frame for frame, cloud in clouds.items() if cloud is not None]
    if not observed or observed[0] != start or observed[-1] != end:
        bounds = "none" if not observed else f"{observed[0]}..{observed[-1]}"
        raise RuntimeError(
            f"{part}: dynamic endpoints must have clouds, got {bounds}"
        )
    poses: dict[int, np.ndarray] = {start: start_pose.copy()}
    registrations = {}
    previous_pair = None
    previous_frame = start
    for frame in observed[1:]:
        gap = frame - previous_frame
        pair, quality = _best_pair_registration(
            clouds[frame],
            clouds[previous_frame],
            previous_pair,
            registration_config,
            seed=1000 + frame,
        )
        rejected = (
            quality["translation_m"] > 0.12 * gap
            or quality["rotation_deg"] > 30.0 * gap
            or quality["fitness_8mm"] < 0.15
        )
        if rejected:
            pair = (
                np.linalg.matrix_power(previous_pair, gap)
                if previous_pair is not None
                else np.eye(4)
            )
            quality["rejected"] = True
            quality["fallback"] = (
                "constant_velocity" if previous_pair is not None else "identity"
            )
        else:
            quality["rejected"] = False
            previous_pair = pair
        current = np.linalg.inv(pair) @ poses[previous_frame]
        if symmetry is not None and registration_config.get(
            "symmetry_lock", True
        ):
            current = resolve_symmetric_pose(
                current,
                poses[previous_frame],
                symmetry,
                include_observation_ambiguities=False,
            ).pose
        poses[frame] = current
        registrations[f"{frame:06d}_to_{previous_frame:06d}"] = {
            "source": frame,
            "target": previous_frame,
            "T_target_from_source": pair.tolist(),
            "quality": quality,
        }
        print(
            f"{part} {frame:03d}->{previous_frame:03d} "
            f"fitness={quality['fitness_8mm']:.3f} "
            f"t={quality['translation_m']:.3f}m "
            f"r={quality['rotation_deg']:.1f}deg "
            f"rejected={quality['rejected']}",
            flush=True,
        )
        previous_frame = frame
    for left, right in zip(observed[:-1], observed[1:]):
        if right == left + 1:
            continue
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(
                np.stack([poses[left][:3, :3], poses[right][:3, :3]])
            ),
        )
        for frame in range(left + 1, right):
            fraction = (frame - left) / (right - left)
            pose = np.eye(4)
            pose[:3, :3] = slerp([fraction]).as_matrix()[0]
            pose[:3, 3] = (
                (1.0 - fraction) * poses[left][:3, 3]
                + fraction * poses[right][:3, 3]
            )
            poses[frame] = pose
    predicted_end = poses[end]
    if symmetry is not None:
        end_pose = resolve_symmetric_pose(
            end_pose,
            predicted_end,
            symmetry,
            include_observation_ambiguities=False,
        ).pose
    delta = end_pose @ np.linalg.inv(predicted_end)
    for frame in range(start, end + 1):
        fraction = (frame - start) / max(end - start, 1)
        poses[frame] = interpolate_transform(delta, fraction) @ poses[frame]
    poses[start] = start_pose.copy()
    poses[end] = end_pose.copy()
    smooth_pose_sequence(poses, start, end, passes=2)
    poses[start] = start_pose.copy()
    poses[end] = end_pose.copy()
    return poses, registrations


def track_model_translation(
    part: str,
    mesh: trimesh.Trimesh,
    scale: float,
    origin_raw: np.ndarray,
    start: int,
    end: int,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    cloud_root: Path,
    seed: int = 0,
) -> tuple[dict[int, np.ndarray], dict]:
    """Track a near-axisymmetric object using translation-only ICP."""
    rng = np.random.default_rng(seed)
    raw, _ = trimesh.sample.sample_surface(mesh, 30000)
    canonical = scale * (np.asarray(raw, float) - origin_raw)
    if len(canonical) > 18000:
        canonical = canonical[
            rng.choice(len(canonical), 18000, replace=False)
        ]
    rotation = start_pose[:3, :3].copy()
    final = end_pose.copy()
    final[:3, :3] = rotation
    poses = {start: start_pose.copy()}
    records = {}
    velocity = np.zeros(3)
    for frame in range(start + 1, end + 1):
        cloud = load_part_cloud(cloud_root, frame, part)
        predicted = poses[frame - 1].copy()
        predicted[:3, :3] = rotation
        predicted[:3, 3] += np.clip(velocity, -0.035, 0.035)
        if cloud is None:
            poses[frame] = predicted
            records[f"{frame:06d}"] = {"status": "inferred", "n_gated": 0}
            continue
        cloud = subsample(cloud, 18000, seed + frame)
        model_world = canonical @ rotation.T + predicted[:3, 3]
        tree = cKDTree(model_world)
        correction_accumulator = np.zeros(3)
        n_gated = 0
        median_distance = None
        for _ in range(12):
            moved = cloud + correction_accumulator
            distances, indices = tree.query(moved, k=1)
            gate = distances < 0.065
            if gate.sum() < 100:
                break
            threshold = np.quantile(distances[gate], 0.60)
            keep = gate & (distances <= threshold)
            residual = model_world[indices[keep]] - moved[keep]
            increment = np.median(residual, axis=0)
            correction_accumulator += 0.75 * increment
            n_gated = int(keep.sum())
            median_distance = float(np.median(distances[keep]))
            if np.linalg.norm(increment) < 2e-4:
                break
        correction = -correction_accumulator
        if np.linalg.norm(correction) > 0.065 or n_gated < 100:
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
            "n_gated": n_gated,
            "median_distance_m": median_distance,
            "correction_m": correction.tolist(),
        }
    drift = final[:3, 3] - poses[end][:3, 3]
    for frame in range(start, end + 1):
        fraction = (frame - start) / max(end - start, 1)
        poses[frame][:3, 3] += fraction * drift
    smooth_pose_sequence(poses, start, end, passes=3)
    poses[start] = start_pose.copy()
    poses[end] = final.copy()
    return poses, records


def _project_bbox(
    points_part: np.ndarray,
    center_world: np.ndarray,
    rotation_world_part: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    world = points_part @ rotation_world_part.T + center_world
    camera = world @ extrinsic[:, :3].T + extrinsic[:, 3]
    uvw = camera @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return np.concatenate([uv.min(axis=0), uv.max(axis=0)])


def _load_mask_bboxes(
    mask_root: Path,
    frame: int,
    part_id: int,
    views: list[str],
) -> list[tuple[np.ndarray, tuple[int, int]] | None]:
    from PIL import Image

    result = []
    for view in views:
        labels = np.asarray(
            Image.open(mask_root / f"{frame:06d}" / f"{view}.png")
        )
        rows, columns = np.where(labels == part_id)
        if len(columns) < 1000:
            result.append(None)
            continue
        bbox = np.asarray([
            columns.min(), rows.min(), columns.max(), rows.max()
        ], dtype=float)
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
    config: dict,
) -> tuple[dict[int, np.ndarray], dict]:
    """Track translation from multi-view envelopes and anchor orientations."""
    envelope = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
    if len(envelope) > 12000:
        envelope = envelope[
            np.random.default_rng(1701 + start).choice(
                len(envelope), 12000, replace=False
            )
        ]
    canonical = scale * (envelope - origin_raw)
    final = end_pose.copy()
    rotation_slerp = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(
            np.stack([start_pose[:3, :3], final[:3, :3]])
        ),
    )

    def rotation_at(fraction: float) -> np.ndarray:
        return rotation_slerp(
            [float(np.clip(fraction, 0.0, 1.0))]
        ).as_matrix()[0]

    backend = config["recon_backend"]
    initial_recon = load_recon(config, f"{start:06d}", backend=backend)
    depth_hw = initial_recon["depth_hw"]
    camera_cache = {}
    bbox_cache = {}

    def observations(frame: int):
        if frame not in bbox_cache:
            boxes = _load_mask_bboxes(mask_root, frame, part_id, views)
            with np.load(
                recon_npz_path(config, f"{frame:06d}", backend)
            ) as data:
                intrinsics = np.asarray(
                    data["intrinsic"]
                    if "intrinsic" in data.files
                    else data["intrinsics"],
                    dtype=np.float64,
                )
                extrinsics = np.asarray(
                    data["extrinsic"]
                    if "extrinsic" in data.files
                    else data["extrinsics"],
                    dtype=np.float64,
                )
            cameras = []
            for index, box in enumerate(boxes):
                shape = box[1] if box is not None else (1080, 1920)
                cameras.append((
                    scale_intrinsics(intrinsics[index], depth_hw, shape),
                    extrinsics[index],
                ))
            bbox_cache[frame] = boxes
            camera_cache[frame] = cameras
        return bbox_cache[frame], camera_cache[frame]

    def anchor_bias(anchor_pose: np.ndarray, frames: list[int]) -> np.ndarray:
        per_view = [[] for _ in views]
        for frame in frames:
            boxes, cameras = observations(frame)
            for index, item in enumerate(boxes):
                if item is None:
                    continue
                observed, _shape = item
                predicted = _project_bbox(
                    canonical,
                    anchor_pose[:3, 3],
                    anchor_pose[:3, :3],
                    *cameras[index],
                )
                per_view[index].append(predicted - observed)
        return np.stack([
            np.median(values, axis=0) if values else np.zeros(4)
            for values in per_view
        ])

    tracker_config = config["states"][part].get("mask_bbox_tracking", {})
    tracker_config = tracker_config.get("ranges", {}).get(
        f"{start}-{end}", tracker_config
    )
    start_bias_frames = [
        int(value)
        for value in tracker_config.get("start_bias_frames", [start])
    ]
    end_bias_frames = [
        int(value)
        for value in tracker_config.get("end_bias_frames", [end])
    ]
    bias_start = anchor_bias(start_pose, start_bias_frames)
    bias_end = anchor_bias(final, end_bias_frames)
    poses = {start: start_pose.copy()}
    records = {}
    velocity = np.zeros(3)
    rotations = {start: start_pose[:3, :3].copy()}
    velocity_cap = float(
        tracker_config.get("prediction_velocity_cap_m", 0.15)
    )
    motion_weight = float(tracker_config.get("motion_prior_weight", 0.5))
    motion_sigma = float(
        tracker_config.get("motion_prior_sigma_m", 0.08)
    )
    linear_weight = float(tracker_config.get("linear_prior_weight", 0.1))
    linear_sigma = float(
        tracker_config.get("linear_prior_sigma_m", 0.1)
    )
    maximum_step = float(
        tracker_config.get("max_translation_step_m", 0.35)
    )
    for frame in range(start + 1, end + 1):
        boxes, cameras = observations(frame)
        fraction = (frame - start) / max(end - start, 1)
        current_rotation = rotation_at(fraction)
        rotations[frame] = current_rotation
        bias = (1.0 - fraction) * bias_start + fraction * bias_end
        previous = poses[frame - 1][:3, 3]
        predicted_motion = previous + np.clip(
            velocity, -velocity_cap, velocity_cap
        )
        linear_prior = (
            (1.0 - fraction) * start_pose[:3, 3]
            + fraction * final[:3, 3]
        )
        edge_count = 0

        def residual(center: np.ndarray) -> np.ndarray:
            nonlocal edge_count
            values = []
            current_edge_count = 0
            for index, item in enumerate(boxes):
                if item is None:
                    continue
                observed, (height, width) = item
                predicted = _project_bbox(
                    canonical, center, current_rotation, *cameras[index]
                )
                valid = np.asarray([
                    observed[0] > 1,
                    observed[1] > 1,
                    observed[2] < width - 2,
                    observed[3] < height - 2,
                ])
                values.extend(
                    (
                        predicted[valid]
                        - observed[valid]
                        - bias[index, valid]
                    ).tolist()
                )
                current_edge_count += int(valid.sum())
            edge_count = current_edge_count
            values.extend(
                (
                    motion_weight
                    * (center - predicted_motion)
                    / motion_sigma
                ).tolist()
            )
            values.extend(
                (
                    linear_weight
                    * (center - linear_prior)
                    / linear_sigma
                ).tolist()
            )
            return np.asarray(values)

        result = least_squares(
            residual,
            predicted_motion,
            loss="soft_l1",
            f_scale=10.0,
            max_nfev=80,
            xtol=1e-7,
            ftol=1e-7,
        )
        center = result.x
        step = center - previous
        if np.linalg.norm(step) > maximum_step:
            center = previous + step * (
                maximum_step / np.linalg.norm(step)
            )
        velocity = 0.65 * velocity + 0.35 * (center - previous)
        pose = start_pose.copy()
        pose[:3, :3] = current_rotation
        pose[:3, 3] = center
        poses[frame] = pose
        edge_residual = residual(center)[:edge_count]
        records[f"{frame:06d}"] = {
            "status": "measured" if edge_count >= 4 else "motion_prior",
            "valid_silhouette_edges": edge_count,
            "median_edge_error_px": (
                float(np.median(np.abs(edge_residual)))
                if len(edge_residual)
                else None
            ),
            "translation_step_m": float(np.linalg.norm(center - previous)),
        }
    smooth_pose_sequence(poses, start, end, passes=2)
    for frame, pose in poses.items():
        pose[:3, :3] = rotations[frame]
    poses[start] = start_pose.copy()
    poses[end] = final.copy()
    records["calibration"] = {
        "start_bias_frames": start_bias_frames,
        "end_bias_frames": end_bias_frames,
        "start_edge_bias_px": bias_start.tolist(),
        "end_edge_bias_px": bias_end.tolist(),
    }
    return poses, records
