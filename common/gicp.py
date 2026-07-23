"""Reusable point-cloud registration primitives."""
from __future__ import annotations

import numpy as np
import small_gicp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from common.pose_transforms import transform_points


def voxel_unique(points: np.ndarray, voxel: float = 0.002) -> np.ndarray:
    if not len(points):
        return points
    cells = np.floor(points / voxel).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def subsample(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(points) <= maximum:
        return np.ascontiguousarray(points, dtype=np.float64)
    index = np.random.default_rng(seed).choice(len(points), maximum, replace=False)
    return np.ascontiguousarray(points[index], dtype=np.float64)


def transform_angle(T: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(T[:3, :3]).magnitude()))


def pair_quality(source: np.ndarray, target: np.ndarray, T: np.ndarray) -> dict:
    distances, _ = cKDTree(target).query(transform_points(source, T), k=1)
    inliers = distances <= 0.008
    keep = np.sort(distances)[:max(30, int(0.8 * len(distances)))]
    return {
        "fitness_8mm": float(inliers.mean()),
        "inlier_rmse_m": float(np.sqrt(np.mean(distances[inliers] ** 2))) if inliers.any() else None,
        "median_nn_m": float(np.median(distances)),
        "trimmed_rmse_m": float(np.sqrt(np.mean(keep ** 2))),
        "n_source": int(len(source)), "n_target": int(len(target)),
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

