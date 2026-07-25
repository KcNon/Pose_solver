"""Dependency-light point-cloud I/O and nearest-neighbour diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_ply_xyz(path: str | Path) -> np.ndarray:
    """Read xyz vertices from the ASCII PLY files produced by this project."""
    points = []
    with Path(path).open(encoding="ascii") as stream:
        in_data = False
        for line in stream:
            if line.strip() == "end_header":
                in_data = True
                continue
            if not in_data:
                continue
            values = line.split()
            if len(values) >= 3:
                points.append([float(values[0]), float(values[1]), float(values[2])])
    return np.asarray(points, dtype=np.float64)


def write_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if len(colors) != len(points):
            raise ValueError("point/color count mismatch")
    with output.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            stream.write(
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            )
        stream.write("end_header\n")
        if colors is None:
            for point in points:
                stream.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n"
                )
        else:
            for point, color in zip(points, colors):
                stream.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]}\n"
                )


def nearest_neighbor_rmse(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    *,
    max_points: int = 4000,
    seed: int = 0,
) -> float:
    """Subsampled source-to-target nearest-neighbour RMSE."""
    source = np.ascontiguousarray(
        np.asarray(source, dtype=np.float64).reshape(-1, 3)
    )
    target = np.ascontiguousarray(
        np.asarray(target, dtype=np.float64).reshape(-1, 3)
    )
    if len(source) < 3 or len(target) < 3:
        return float("inf")
    rng = np.random.default_rng(seed)
    if len(source) > max_points:
        source = source[rng.choice(len(source), max_points, replace=False)]
    if len(target) > max_points:
        target = target[rng.choice(len(target), max_points, replace=False)]
    aligned = (
        source @ np.asarray(transform[:3, :3], dtype=np.float64).T
        + np.asarray(transform[:3, 3], dtype=np.float64)
    )
    distances, _ = cKDTree(target).query(aligned, k=1, workers=-1)
    return float(np.sqrt(np.mean(distances * distances)))
