"""Shared rigid/similarity transform conventions used by the pose pipeline."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 column-vector transform to row-major 3D points."""
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def decompose_similarity(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return uniform scale, rotation, and translation from a 4x4 similarity."""
    transform = np.asarray(transform, dtype=np.float64)
    determinant = float(np.linalg.det(transform[:3, :3]))
    if determinant <= 0.0:
        raise ValueError("Similarity transform must have a positive determinant")
    scale = float(np.cbrt(determinant))
    rotation = transform[:3, :3] / scale
    return scale, rotation, transform[:3, 3].copy()


def similarity(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 uniform similarity transform."""
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = float(scale) * np.asarray(rotation, dtype=np.float64)
    value[:3, 3] = np.asarray(translation, dtype=np.float64)
    return value


def rigid_from_similarity(transform: np.ndarray, raw_mesh_origin: np.ndarray) -> np.ndarray:
    """Convert raw-mesh similarity into the canonical part-frame rigid pose."""
    scale, rotation, translation = decompose_similarity(transform)
    origin = np.asarray(raw_mesh_origin, dtype=np.float64)
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = scale * (rotation @ origin) + translation
    return value


def similarity_from_rigid(
    transform: np.ndarray,
    scale: float,
    raw_mesh_origin: np.ndarray,
) -> np.ndarray:
    """Convert a canonical part-frame rigid pose into raw-mesh similarity."""
    transform = np.asarray(transform, dtype=np.float64)
    origin = np.asarray(raw_mesh_origin, dtype=np.float64)
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = float(scale) * transform[:3, :3]
    value[:3, 3] = (
        transform[:3, 3]
        - float(scale) * (transform[:3, :3] @ origin)
    )
    return value


def axis_rotation(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    """Return a 4x4 rotation around ``axis`` by a radian angle."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("Rotation axis must be non-zero")
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_rotvec(
        axis / norm * float(angle_radians)
    ).as_matrix()
    return value


def axis_rotation_degrees(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Return a 4x4 rotation around ``axis`` by a degree angle."""
    return axis_rotation(axis, np.deg2rad(float(angle_degrees)))
