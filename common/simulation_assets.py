"""Reusable helpers for exporting solved part poses as simulation assets.

The pose solver represents every part with a canonical frame whose origin is
the centroid of the raw generated mesh.  The raw GLB is mapped to that frame by
uniformly scaling it and subtracting ``raw_mesh_origins[part]``.  This module
keeps that convention explicit so URDF, USD, and the renderer all agree.
"""
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from trimesh.exchange.obj import export_obj


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_flat_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh scene has no geometry: {path}")
        mesh = loaded.to_geometry()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Expected a triangle mesh in {path}, got {type(loaded).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh is empty: {path}")
    return mesh


def canonicalize_mesh(mesh: trimesh.Trimesh, scale: float, raw_origin: Iterable[float]) -> trimesh.Trimesh:
    result = mesh.copy()
    origin = np.asarray(raw_origin, dtype=np.float64)
    result.vertices = (np.asarray(result.vertices, dtype=np.float64) - origin[None, :]) * float(scale)
    return result


def canonical_from_raw_matrix(scale: float, raw_origin: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= float(scale)
    matrix[:3, 3] = -float(scale) * np.asarray(raw_origin, dtype=np.float64)
    return matrix


def export_textured_obj(mesh: trimesh.Trimesh, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_text, sidecars = export_obj(mesh, return_texture=True, mtl_name=f"{stem}.mtl")
    obj_path = output_dir / f"{stem}.obj"
    obj_path.write_text(obj_text, encoding="utf-8")
    for name, payload in sidecars.items():
        sidecar = output_dir / name
        if isinstance(payload, str):
            sidecar.write_text(payload, encoding="utf-8")
        else:
            sidecar.write_bytes(payload)
    return obj_path


def export_collision_obj(mesh: trimesh.Trimesh, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obj_text = export_obj(
        mesh,
        include_color=False,
        include_texture=False,
        include_normals=True,
        return_texture=False,
    )
    output_path.write_text(obj_text, encoding="utf-8")


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = Rotation.from_matrix(a).inv() * Rotation.from_matrix(b)
    return float(np.degrees(relative.magnitude()))


def _mad(values: np.ndarray) -> float:
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def robust_average_pose(
    poses: list[np.ndarray],
    frame_ids: list[str],
    max_translation_residual_m: float = 0.03,
    max_rotation_residual_deg: float = 15.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not poses:
        raise ValueError("Cannot average an empty pose list")
    translations = np.stack([pose[:3, 3] for pose in poses])
    rotations = Rotation.from_matrix(np.stack([pose[:3, :3] for pose in poses]))

    translation_center = np.median(translations, axis=0)
    rotation_center = rotations.mean()
    translation_residuals = np.linalg.norm(translations - translation_center[None, :], axis=1)
    rotation_residuals_deg = np.degrees((rotation_center.inv() * rotations).magnitude())

    trans_limit = min(
        float(max_translation_residual_m),
        float(np.median(translation_residuals) + 3.5 * max(_mad(translation_residuals), 1e-6)),
    )
    rot_limit = min(
        float(max_rotation_residual_deg),
        float(np.median(rotation_residuals_deg) + 3.5 * max(_mad(rotation_residuals_deg), 1e-3)),
    )
    keep = (translation_residuals <= trans_limit) & (rotation_residuals_deg <= rot_limit)
    if not np.any(keep):
        keep[:] = True

    kept_translations = translations[keep]
    kept_rotations = Rotation.from_matrix(np.stack([poses[i][:3, :3] for i in np.flatnonzero(keep)]))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = kept_rotations.mean().as_matrix()
    result[:3, 3] = np.median(kept_translations, axis=0)

    final_translation_residuals = np.linalg.norm(translations - result[:3, 3][None, :], axis=1)
    final_rotation_residuals = np.array(
        [rotation_error_deg(result[:3, :3], pose[:3, :3]) for pose in poses], dtype=np.float64
    )
    stats = {
        "candidate_frames": frame_ids,
        "used_frames": [frame_ids[i] for i in np.flatnonzero(keep)],
        "rejected_frames": [frame_ids[i] for i in np.flatnonzero(~keep)],
        "translation_residual_m": {
            "median": float(np.median(final_translation_residuals)),
            "p95": float(np.percentile(final_translation_residuals, 95)),
            "max": float(np.max(final_translation_residuals)),
        },
        "rotation_residual_deg": {
            "median": float(np.median(final_rotation_residuals)),
            "p95": float(np.percentile(final_rotation_residuals, 95)),
            "max": float(np.max(final_rotation_residuals)),
        },
    }
    return result, stats


def select_part_poses(
    trajectory: dict[str, Any],
    part: str,
    selection: dict[str, Any],
) -> tuple[list[np.ndarray], list[str]]:
    start, end = selection.get("frame_range", [0, 10**9])
    required_state = selection.get("required_state")
    min_views = int(selection.get("min_observing_views", 0))
    poses: list[np.ndarray] = []
    frame_ids: list[str] = []
    for frame_id in sorted(trajectory["frames"], key=int):
        index = int(frame_id)
        if index < int(start) or index > int(end):
            continue
        part_data = trajectory["frames"][frame_id]["parts"].get(part)
        if part_data is None:
            continue
        if required_state is not None and part_data.get("state") != required_state:
            continue
        if int(part_data.get("observing_views", 0)) < min_views:
            continue
        if "T_body_from_part" not in part_data:
            continue
        poses.append(np.asarray(part_data["T_body_from_part"], dtype=np.float64))
        frame_ids.append(frame_id)
    return poses, frame_ids


def state_runs(trajectory: dict[str, Any], part: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    active_state: str | None = None
    start = previous = None
    for frame_id in sorted(trajectory["frames"], key=int):
        state = trajectory["frames"][frame_id]["parts"][part].get("state", "unknown")
        index = int(frame_id)
        if state != active_state:
            if active_state is not None:
                runs.append({"state": active_state, "start": start, "end": previous})
            active_state = state
            start = index
        previous = index
    if active_state is not None:
        runs.append({"state": active_state, "start": start, "end": previous})
    return runs


def matrix_to_xyz_rpy(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    xyz = matrix[:3, 3]
    rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=False)
    return xyz, rpy


def _numbers(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def box_inertia(mass: float, extents: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in extents)
    return (
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def _append_link(
    robot: ET.Element,
    part: str,
    visual_mesh: str,
    collision_mesh: str,
    mass: float,
    extents: Iterable[float],
) -> None:
    link = ET.SubElement(robot, "link", {"name": part})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass:.10g}"})
    ixx, iyy, izz = box_inertia(mass, extents)
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{ixx:.10g}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{iyy:.10g}",
            "iyz": "0",
            "izz": f"{izz:.10g}",
        },
    )
    visual = ET.SubElement(link, "visual", {"name": f"{part}_visual"})
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": visual_mesh, "scale": "1 1 1"})
    collision = ET.SubElement(link, "collision", {"name": f"{part}_collision"})
    ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    collision_geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(collision_geometry, "mesh", {"filename": collision_mesh, "scale": "1 1 1"})


def write_urdf(
    path: Path,
    robot_name: str,
    parts: list[str],
    part_info: dict[str, dict[str, Any]],
    fixed_transforms: dict[str, np.ndarray] | None = None,
    root_part: str | None = None,
) -> None:
    robot = ET.Element("robot", {"name": robot_name})
    for part in parts:
        visual_rel = f"../meshes/visual/{part}/{part}.obj"
        collision_rel = f"../meshes/collision/{part}.obj"
        _append_link(
            robot,
            part,
            visual_rel,
            collision_rel,
            float(part_info[part]["mass_kg"]),
            part_info[part]["canonical_extents_m"],
        )
    if fixed_transforms:
        if root_part is None:
            raise ValueError("root_part is required when fixed_transforms are provided")
        for child, transform in fixed_transforms.items():
            if child == root_part:
                continue
            xyz, rpy = matrix_to_xyz_rpy(transform)
            joint = ET.SubElement(robot, "joint", {"name": f"{root_part}_to_{child}", "type": "fixed"})
            ET.SubElement(joint, "parent", {"link": root_part})
            ET.SubElement(joint, "child", {"link": child})
            ET.SubElement(joint, "origin", {"xyz": _numbers(xyz), "rpy": _numbers(rpy)})
    ET.indent(robot, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def aabb_overlap(bounds_a: np.ndarray, bounds_b: np.ndarray) -> dict[str, Any]:
    low = np.maximum(bounds_a[0], bounds_b[0])
    high = np.minimum(bounds_a[1], bounds_b[1])
    extents = np.maximum(high - low, 0.0)
    return {
        "overlap_extents_m": extents.tolist(),
        "overlap_volume_m3": float(np.prod(extents)),
        "has_aabb_overlap": bool(np.all(extents > 0.0)),
    }


def vertex_surface_distance_summary(
    body_vertices: np.ndarray,
    inserted_vertices: np.ndarray,
) -> dict[str, float]:
    """Unsigned vertex-to-vertex diagnostic; deliberately not called clearance."""
    tree = cKDTree(np.asarray(body_vertices, dtype=np.float64))
    distances, _ = tree.query(np.asarray(inserted_vertices, dtype=np.float64), workers=-1)
    return {
        "min": float(np.min(distances)),
        "p01": float(np.percentile(distances, 1)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def rotation_align_vectors(source: Iterable[float], target: Iterable[float]) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    source_array /= np.linalg.norm(source_array)
    target_array /= np.linalg.norm(target_array)
    cross = np.cross(source_array, target_array)
    dot = float(np.clip(np.dot(source_array, target_array), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(source_array[0]) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source_array, trial)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * math.pi).as_matrix()
    axis = cross / np.linalg.norm(cross)
    return Rotation.from_rotvec(axis * math.acos(dot)).as_matrix()
