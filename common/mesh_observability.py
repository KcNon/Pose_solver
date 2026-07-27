"""Geometry/texture observability metadata inferred from an object mesh."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) < 4:
        raise ValueError(f"mesh has insufficient geometry: {path}")
    return loaded


def _points(mesh: trimesh.Trimesh, maximum: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    values = np.concatenate((vertices, centers), axis=0)
    if len(values) > maximum:
        indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
        values = values[indices]
    return values - values.mean(axis=0)


def _rotation_error(
    points: np.ndarray,
    tree: cKDTree,
    axis: np.ndarray,
    angle_deg: float,
    scale: float,
) -> float:
    rotated = Rotation.from_rotvec(
        np.deg2rad(angle_deg) * axis
    ).apply(points)
    distance, _ = tree.query(rotated, k=1)
    return float(np.quantile(distance, 0.75) / max(scale, 1e-12))


def mesh_has_texture(mesh: trimesh.Trimesh) -> bool:
    visual = getattr(mesh, "visual", None)
    if visual is None or getattr(visual, "kind", None) != "texture":
        return False
    material = getattr(visual, "material", None)
    return bool(
        material is not None
        and (
            getattr(material, "image", None) is not None
            or getattr(material, "baseColorTexture", None) is not None
        )
    )


def infer_mesh_observability(
    path: str | Path,
    *,
    maximum_points: int = 12000,
    continuous_tolerance: float = 0.025,
    cyclic_tolerance: float = 0.018,
) -> dict[str, Any]:
    """Infer conservative geometric symmetry candidates from raw coordinates.

    This is a proposal generator, not semantic ground truth.  Explicit config
    remains authoritative, especially when a textured object's canonical
    "front" matters downstream.
    """

    mesh_path = Path(path)
    mesh = _load_mesh(mesh_path)
    points = _points(mesh, maximum_points)
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = eigenvectors[:, order].T
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    tree = cKDTree(points)
    angles = (30.0, 45.0, 60.0, 90.0, 120.0, 180.0)
    candidates = []
    for axis in axes:
        axis = axis / np.linalg.norm(axis)
        errors = {
            str(int(angle)): _rotation_error(
                points, tree, axis, angle, scale
            )
            for angle in angles
        }
        candidates.append({
            "axis_raw": axis.tolist(),
            "errors": errors,
            "continuous_score": max(errors["30"], errors["45"]),
        })
    best = min(candidates, key=lambda row: row["continuous_score"])
    symmetry: dict[str, Any] = {"equivalence": "none"}
    confidence = "low"
    if float(best["continuous_score"]) <= continuous_tolerance:
        symmetry = {
            "equivalence": "continuous_axial",
            "axis_raw": best["axis_raw"],
            "candidate_step_deg": 15.0,
        }
        confidence = (
            "high"
            if float(best["continuous_score"]) <= 0.5 * continuous_tolerance
            else "medium"
        )
    else:
        # Prefer the highest supported generator order.  The tested orders are
        # intentionally small; accidental near-symmetry is otherwise common
        # on reconstructed meshes.
        for cyclic_order in (6, 4, 3, 2):
            angle = int(round(360 / cyclic_order))
            if float(best["errors"].get(str(angle), 1.0)) <= cyclic_tolerance:
                symmetry = {
                    "equivalence": "cyclic",
                    "axis_raw": best["axis_raw"],
                    "discrete_order": cyclic_order,
                }
                confidence = "medium"
                break
    ratios = (
        eigenvalues / max(float(eigenvalues[0]), 1e-12)
    ).tolist()
    return {
        "mesh": str(mesh_path),
        "symmetry": symmetry,
        "confidence": confidence,
        "has_texture": mesh_has_texture(mesh),
        "principal_variance_ratios": ratios,
        "axis_candidates": candidates,
        "method": "pca_axes_rotated_surface_p75_nn_v1",
    }
