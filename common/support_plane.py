"""Deterministic table-plane and stable part-support validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from common.geom import backproject_view
from common.normalized_recon import load_recon
from common.pose_tracking import load_part_cloud


def estimate_local_table_plane(
    cfg: dict[str, Any], frame: int, *, seed: int = 9021
) -> dict[str, Any]:
    """Fit the dominant background plane in a ring around all part masks."""

    timestamp = f"{int(frame):06d}"
    recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    camera_centres = []
    for view_index, view in enumerate(cfg["views"]):
        mask_path = Path(cfg["masks_dir"]) / timestamp / f"{view}.png"
        if not mask_path.exists():
            continue
        labels = np.asarray(Image.open(mask_path))
        height, width = recon["depth_hw"]
        labels = cv2.resize(
            labels.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        foreground = labels > 0
        near = cv2.dilate(
            foreground.astype(np.uint8), np.ones((17, 17), np.uint8)
        ) > 0
        far = cv2.dilate(
            foreground.astype(np.uint8), np.ones((61, 61), np.uint8)
        ) > 0
        annulus = far & ~near
        depth = recon["depth"][view_index]
        confidence = recon["conf"][view_index]
        valid = (
            annulus
            & np.isfinite(depth)
            & (depth > 1e-3)
            & np.isfinite(confidence)
        )
        if valid.any():
            valid &= confidence >= np.quantile(confidence[valid], 0.35)
        if int(valid.sum()) > 2500:
            indexes = np.flatnonzero(valid)
            keep = rng.choice(indexes, 2500, replace=False)
            valid[:] = False
            valid.flat[keep] = True
        if valid.any():
            cloud, _ = backproject_view(
                depth,
                recon["intrinsics"][view_index],
                recon["extrinsics"][view_index],
                mask=valid,
            )
            points.append(cloud)
        extrinsic = recon["extrinsics"][view_index]
        camera_centres.append(
            -extrinsic[:3, :3].T @ extrinsic[:3, 3]
        )
    if not points:
        return {"accepted": False, "reason": "no_local_background_points"}
    cloud = np.concatenate(points, axis=0).astype(np.float64)
    if len(cloud) > 16000:
        cloud = cloud[rng.choice(len(cloud), 16000, replace=False)]
    best_inliers = np.zeros(len(cloud), dtype=bool)
    threshold = 0.008
    for _ in range(700):
        sample = cloud[rng.choice(len(cloud), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        inliers = np.abs((cloud - sample[0]) @ normal) < threshold
        if int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
    if int(best_inliers.sum()) < 500:
        return {
            "accepted": False,
            "reason": "insufficient_plane_inliers",
            "points": int(len(cloud)),
            "inliers": int(best_inliers.sum()),
        }
    support = cloud[best_inliers]
    centroid = support.mean(axis=0)
    _, _, vh = np.linalg.svd(support - centroid, full_matrices=False)
    normal = vh[-1]
    cameras = np.asarray(camera_centres)
    if len(cameras) and float(np.median((cameras - centroid) @ normal)) < 0.0:
        normal = -normal
    signed = (support - centroid) @ normal
    rmse = float(np.sqrt(np.mean(np.square(signed))))
    accepted = bool(len(support) / len(cloud) >= 0.20 and rmse <= 0.012)
    return {
        "accepted": accepted,
        "method": "local_mask_annulus_multiview_ransac_svd",
        "frame": int(frame),
        "normal_world": normal.tolist(),
        "point_world": centroid.tolist(),
        "offset": float(-np.dot(normal, centroid)),
        "points": int(len(cloud)),
        "inliers": int(len(support)),
        "inlier_fraction": float(len(support) / len(cloud)),
        "rmse_m": rmse,
        "threshold_m": threshold,
    }


def validate_part_support_window(
    cfg: dict[str, Any],
    *,
    part: str,
    frames: list[int],
    cloud_root: Path,
    representative_frame: int,
    maximum_contact_gap_m: float = 0.05,
    maximum_gap_mad_m: float = 0.015,
) -> dict[str, Any]:
    """Check that a stable-window cloud remains at a constant table height."""

    plane = estimate_local_table_plane(cfg, representative_frame)
    if not plane.get("accepted", False):
        return {
            "accepted": False,
            "reason": "table_plane_not_observable",
            "table_plane": plane,
        }
    normal = np.asarray(plane["normal_world"], dtype=np.float64)
    point = np.asarray(plane["point_world"], dtype=np.float64)
    rows = []
    for frame in frames:
        cloud = load_part_cloud(cloud_root, int(frame), part)
        if cloud is None or len(cloud) < 100:
            continue
        signed = (np.asarray(cloud, dtype=np.float64) - point) @ normal
        rows.append({
            "frame": int(frame),
            "bottom_gap_m": float(np.quantile(signed, 0.02)),
            "median_height_m": float(np.median(signed)),
        })
    if len(rows) < max(2, len(frames) // 2):
        return {
            "accepted": False,
            "reason": "insufficient_window_clouds",
            "table_plane": plane,
            "frames": rows,
        }
    gaps = np.asarray([row["bottom_gap_m"] for row in rows])
    median_gap = float(np.median(gaps))
    gap_mad = float(np.median(np.abs(gaps - median_gap)))
    accepted = bool(
        abs(median_gap) <= float(maximum_contact_gap_m)
        and gap_mad <= float(maximum_gap_mad_m)
    )
    return {
        "accepted": accepted,
        "reason": None if accepted else "unstable_or_detached_from_table_plane",
        "table_plane": plane,
        "frames": rows,
        "median_bottom_gap_m": median_gap,
        "bottom_gap_mad_m": gap_mad,
        "maximum_contact_gap_m": float(maximum_contact_gap_m),
        "maximum_gap_mad_m": float(maximum_gap_mad_m),
    }
