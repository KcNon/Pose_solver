"""Shared geometry, image objectives and bounded-pose constraints."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from common.pose_transforms import transform_points


def sample_canonical(mesh: trimesh.Trimesh, scale: float, origin: np.ndarray,
                     count: int, seed: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(
        mesh, count, seed=np.random.default_rng(seed)
    )
    return scale * (np.asarray(points, float) - origin)


def cloud_metrics(observed: np.ndarray, canonical: np.ndarray, pose: np.ndarray) -> dict:
    local = transform_points(observed, np.linalg.inv(pose))
    distances, _ = cKDTree(canonical).query(local, k=1, workers=-1)
    keep = np.sort(distances)[:max(30, int(0.8 * len(distances)))]
    return {
        "fitness_8mm": float(np.mean(distances <= 0.008)),
        "median_nn_m": float(np.median(distances)),
        "trimmed_rmse_m": float(np.sqrt(np.mean(keep ** 2))),
    }


def aggregate_mask_targets(cfg: dict, part: str, start: int, end: int,
                           min_support: int, width: int, height: int) -> list[np.ndarray]:
    part_id = int(cfg["part_ids"][part])
    targets = []
    kernel = np.ones((3, 3), np.uint8)
    for view in cfg["views"]:
        support = np.zeros((height, width), np.uint16)
        for frame in range(start, end + 1):
            labels = np.asarray(Image.open(
                Path(cfg["masks_dir"]) / f"{frame:06d}" / f"{view}.png"))
            support += cv2.resize((labels == part_id).astype(np.uint8), (width, height),
                                  interpolation=cv2.INTER_NEAREST)
        target = cv2.morphologyEx((support >= min_support).astype(np.uint8),
                                  cv2.MORPH_CLOSE, kernel, iterations=1)
        targets.append(target.astype(bool))
    return targets


def silhouette_metrics(rendered: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    union = np.logical_or(rendered, target).sum()
    iou = float(np.logical_and(rendered, target).sum() / union) if union else 1.0
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.morphologyEx(rendered.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    et = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    if not er.any() or not et.any():
        return iou, 100.0, iou - 1.0
    dt_t = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 3)
    dt_r = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = 0.5 * (float(dt_t[er].mean()) + float(dt_r[et].mean()))
    return iou, chamfer, iou - 0.01 * chamfer


def make_mask_comparison(targets: list[np.ndarray], baseline: list[np.ndarray],
                         refined: list[np.ndarray], views: list[str], out: Path) -> None:
    rows = []
    for label, rendered_set in (("baseline", baseline), ("refined", refined)):
        panels = []
        for view, target, rendered in zip(views, targets, rendered_set):
            image = np.zeros((*target.shape, 3), np.uint8)
            image[target] = (55, 55, 210)
            image[rendered] = (210, 80, 45)
            image[target & rendered] = (225, 225, 225)
            cv2.putText(image, f"{label} {view}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(panels))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])


def cap_pose_delta(delta: np.ndarray, max_translation_m: float,
                   max_rotation_deg: float) -> tuple[np.ndarray, dict]:
    result = np.eye(4)
    dt = np.asarray(delta[:3, 3], float)
    dr = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
    t_scale = min(1.0, max_translation_m / max(float(np.linalg.norm(dt)), 1e-12))
    r_deg = float(np.degrees(np.linalg.norm(dr)))
    r_scale = min(1.0, max_rotation_deg / max(r_deg, 1e-12))
    result[:3, 3] = t_scale * dt
    result[:3, :3] = Rotation.from_rotvec(r_scale * dr).as_matrix()
    return result, {
        "raw_translation_mm": float(1000 * np.linalg.norm(dt)),
        "raw_rotation_deg": r_deg, "translation_scale": t_scale,
        "rotation_scale": r_scale,
    }


def limit_pose_velocity(poses: dict[int, np.ndarray], max_translation_m: float,
                        max_rotation_deg: float) -> tuple[dict[int, np.ndarray], dict]:
    result, limited = {}, []
    raw_t, raw_r, final_t, final_r = [], [], [], []
    previous = None
    for frame in sorted(poses):
        pose = poses[frame].copy()
        if previous is not None:
            dt = pose[:3, 3] - previous[:3, 3]
            dr = Rotation.from_matrix(previous[:3, :3].T @ pose[:3, :3]).as_rotvec()
            dt_norm = float(np.linalg.norm(dt))
            dr_deg = float(np.degrees(np.linalg.norm(dr)))
            raw_t.append(dt_norm); raw_r.append(dr_deg)
            ts = min(1.0, max_translation_m / max(dt_norm, 1e-12))
            rs = min(1.0, max_rotation_deg / max(dr_deg, 1e-12))
            if ts < 1.0 or rs < 1.0:
                limited.append(frame)
            pose[:3, 3] = previous[:3, 3] + ts * dt
            pose[:3, :3] = previous[:3, :3] @ Rotation.from_rotvec(rs * dr).as_matrix()
            final_t.append(float(np.linalg.norm(pose[:3, 3] - previous[:3, 3])))
            final_r.append(float(np.degrees(Rotation.from_matrix(
                previous[:3, :3].T @ pose[:3, :3]).magnitude())))
        result[frame] = pose
        previous = pose
    return result, {
        "max_translation_step_m": max_translation_m, "max_rotation_step_deg": max_rotation_deg,
        "limited_frames": limited, "raw_max_translation_step_m": max(raw_t, default=0.0),
        "raw_max_rotation_step_deg": max(raw_r, default=0.0),
        "final_max_translation_step_m": max(final_t, default=0.0),
        "final_max_rotation_step_deg": max(final_r, default=0.0),
    }


def smooth_pose_ranges(
    poses: dict[int, np.ndarray],
    ranges: list[tuple[int, int]] | list[list[int]],
    *,
    passes: int = 2,
    fixed_frames: set[int] | None = None,
) -> dict[int, np.ndarray]:
    """Smooth configured dynamic ranges while keeping their neighbours fixed.

    The fixed frame immediately before and after each range prevents an
    independently refined moving interval from introducing a jump at either
    stable boundary.  Missing boundary frames are handled by leaving the
    available endpoint fixed.
    """

    result = {
        int(frame): np.asarray(pose, dtype=np.float64).copy()
        for frame, pose in poses.items()
    }
    if passes <= 0 or not result:
        return result
    available = set(result)
    fixed = {int(frame) for frame in (fixed_frames or set())}
    for raw_start, raw_end in ranges:
        start, end = int(raw_start), int(raw_end)
        if end < start:
            raise ValueError(f"invalid smoothing range [{start}, {end}]")
        for _ in range(int(passes)):
            old = {frame: pose.copy() for frame, pose in result.items()}
            for frame in range(start, end + 1):
                if frame in fixed:
                    continue
                if not {frame - 1, frame, frame + 1}.issubset(available):
                    continue
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
                result[frame][:3, 3] = np.average(
                    translations,
                    axis=0,
                    weights=np.asarray([1.0, 4.0, 1.0]),
                )
                result[frame][:3, :3] = rotations.mean(
                    weights=np.asarray([1.0, 4.0, 1.0])
                ).as_matrix()
    return result


def interpolate_untrusted_pose_frames(
    poses: dict[int, np.ndarray],
    ranges: list[tuple[int, int]] | list[list[int]],
    *,
    trusted_frames: set[int],
) -> tuple[dict[int, np.ndarray], list[int]]:
    """Replace rejected/missing measurements between trusted temporal poses.

    A rejected render refinement is evidence that the measurement is weak; it
    must not become a fixed smoothing anchor.  This fills such frames by SE(3)
    interpolation between the nearest accepted measurements (or the fixed
    static neighbours of the dynamic range) before the normal local smoother
    is applied.
    """

    result = {
        int(frame): np.asarray(pose, dtype=np.float64).copy()
        for frame, pose in poses.items()
    }
    available = set(result)
    trusted = {int(frame) for frame in trusted_frames if int(frame) in available}
    replaced: list[int] = []
    for raw_start, raw_end in ranges:
        start, end = int(raw_start), int(raw_end)
        if end < start:
            raise ValueError(f"invalid interpolation range [{start}, {end}]")
        local_trusted = {
            frame for frame in trusted if start <= frame <= end
        }
        if start - 1 in available:
            local_trusted.add(start - 1)
        if end + 1 in available:
            local_trusted.add(end + 1)
        ordered = sorted(local_trusted)
        if not ordered:
            continue
        source = {frame: result[frame].copy() for frame in ordered}
        for frame in range(start, end + 1):
            if frame not in available or frame in local_trusted:
                continue
            left = max((value for value in ordered if value < frame), default=None)
            right = min((value for value in ordered if value > frame), default=None)
            if left is None and right is None:
                continue
            if left is None:
                result[frame] = source[right].copy()
            elif right is None:
                result[frame] = source[left].copy()
            else:
                fraction = (frame - left) / (right - left)
                pose = np.eye(4, dtype=np.float64)
                pose[:3, 3] = (
                    (1.0 - fraction) * source[left][:3, 3]
                    + fraction * source[right][:3, 3]
                )
                rotations = Rotation.from_matrix(np.stack([
                    source[left][:3, :3], source[right][:3, :3]
                ]))
                pose[:3, :3] = Slerp([0.0, 1.0], rotations)(
                    [fraction]
                ).as_matrix()[0]
                result[frame] = pose
            replaced.append(frame)
    return result, sorted(set(replaced))
