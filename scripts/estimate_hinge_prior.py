#!/usr/bin/env python3
"""Estimate a canonical revolute-joint prior from two reconstructed states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


UP_ROTATIONS = {
    "Z": np.eye(3),
    "-Z": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64),
    "X": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    "-X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    "Y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    "-Y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
}


def load_flat_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Expected a non-empty triangle mesh: {path}")
    return mesh


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def fit_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, singular_values, vt = np.linalg.svd(target_zero.T @ source_zero / len(source))
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    variance = np.sum(source_zero * source_zero) / len(source)
    scale = np.trace(np.diag(singular_values) @ sign) / max(variance, 1e-12)
    translation = target_center - scale * rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def trimmed_similarity_icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    trim_fraction: float,
    iterations: int,
    maximum_distance: float,
) -> tuple[np.ndarray, dict]:
    tree = cKDTree(target)
    transform = np.eye(4, dtype=np.float64)
    history = []
    for iteration in range(iterations):
        transformed = transform_points(source, transform)
        distances, indices = tree.query(transformed, k=1, workers=-1)
        threshold = min(float(np.quantile(distances, trim_fraction)), maximum_distance)
        selected = distances <= threshold
        if selected.sum() < 100:
            raise RuntimeError("Too few stationary correspondences survived trimming")
        updated = fit_similarity(source[selected], target[indices[selected]])
        delta = float(np.max(np.abs(updated - transform)))
        transform = updated
        history.append(
            {
                "iteration": iteration,
                "selected": int(selected.sum()),
                "median_distance": float(np.median(distances)),
                "trim_threshold": threshold,
                "transform_delta": delta,
            }
        )
        if iteration >= 10 and delta < 1e-6:
            break
    return transform, {"iterations": history}


def distance_summary(source: np.ndarray, target: np.ndarray) -> dict:
    distances = cKDTree(target).query(source, k=1, workers=-1)[0]
    ordered = np.sort(distances)
    keep = max(1, int(0.8 * len(ordered)))
    return {
        "median": float(np.median(distances)),
        "p90": float(np.quantile(distances, 0.9)),
        "trimmed_rmse_80": float(np.sqrt(np.mean(ordered[:keep] ** 2))),
    }


def plucker_to_axis_point(plucker: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(plucker[:3], dtype=np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    point = np.cross(np.asarray(plucker[3:], dtype=np.float64), axis)
    return axis, point


def prediction_axis_raw(
    prediction_dir: Path,
    up_dir: str,
) -> tuple[np.ndarray, np.ndarray, dict, trimesh.Trimesh]:
    prediction = np.load(prediction_dir / "eval" / "pred.npz")
    proxy = load_flat_mesh(prediction_dir / "eval" / "pred.obj")
    labels = np.asarray(prediction["face_part_ids"], dtype=np.int64)
    revolute = np.flatnonzero(np.asarray(prediction["is_part_revolute"], dtype=bool))
    if len(revolute) != 1:
        raise ValueError(
            f"Expected exactly one revolute part in {prediction_dir}, got {revolute.tolist()}"
        )
    part_id = int(revolute[0])
    axis_normalized, point_normalized = plucker_to_axis_point(
        prediction["revolute_plucker"][part_id]
    )
    up_rotation = UP_ROTATIONS[up_dir]
    rotated_bounds = np.vstack(
        (
            (np.asarray(proxy.vertices) @ up_rotation.T).min(axis=0),
            (np.asarray(proxy.vertices) @ up_rotation.T).max(axis=0),
        )
    )
    center = rotated_bounds.mean(axis=0)
    scale = float(np.max(rotated_bounds[1] - rotated_bounds[0]))
    axis_raw = up_rotation.T @ axis_normalized
    axis_raw /= np.linalg.norm(axis_raw)
    point_raw = up_rotation.T @ (scale * point_normalized + center)
    face_mask = labels == part_id
    moving_mesh = proxy.submesh([face_mask], append=True, repair=False)
    metadata = {
        "prediction_dir": str(prediction_dir),
        "part_id": part_id,
        "moving_face_fraction": float(face_mask.mean()),
        "axis_raw": axis_raw.tolist(),
        "point_raw": point_raw.tolist(),
        "range_radians": np.asarray(prediction["revolute_range"][part_id]).tolist(),
    }
    return axis_raw, point_raw, metadata, moving_mesh


def consensus_axis(
    axes: list[np.ndarray],
    points: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict]:
    reference = axes[0]
    aligned = [axis if np.dot(axis, reference) >= 0 else -axis for axis in axes]
    axis = np.sum(aligned, axis=0)
    axis /= np.linalg.norm(axis)
    system = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for candidate_axis, candidate_point in zip(aligned, points):
        projection = np.eye(3) - np.outer(candidate_axis, candidate_axis)
        system += projection
        rhs += projection @ candidate_point
    point = np.linalg.lstsq(system, rhs, rcond=None)[0]
    point -= axis * np.dot(axis, point)
    angles = [
        float(np.degrees(np.arccos(np.clip(abs(np.dot(axis, value)), -1.0, 1.0))))
        for value in axes
    ]
    line_offsets = []
    for candidate_axis, candidate_point in zip(aligned, points):
        offset = point - candidate_point
        line_offsets.append(float(np.linalg.norm(np.cross(offset, candidate_axis))))
    return axis, point, {
        "maximum_axis_deviation_degrees": max(angles),
        "axis_deviations_degrees": angles,
        "line_offset_distances": line_offsets,
    }


def rotation_about_axis(
    points: np.ndarray,
    axis: np.ndarray,
    point: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    rotation = Rotation.from_rotvec(axis * np.deg2rad(angle_degrees)).as_matrix()
    return (points - point) @ rotation.T + point


def make_axis_mesh(axis: np.ndarray, point: np.ndarray, length: float) -> list[trimesh.Trimesh]:
    cylinder = trimesh.creation.cylinder(radius=0.006 * length, height=length, sections=48)
    align = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis)
    cylinder.apply_transform(align)
    cylinder.apply_translation(point)
    cylinder.visual.face_colors = [0, 122, 255, 255]
    pivot = trimesh.creation.icosphere(subdivisions=2, radius=0.018 * length)
    pivot.apply_translation(point)
    pivot.visual.face_colors = [255, 204, 0, 255]
    return [cylinder, pivot]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-body", type=Path, required=True)
    parser.add_argument("--open-head", type=Path, required=True)
    parser.add_argument("--close-mesh", type=Path, required=True)
    parser.add_argument("--close-prediction", type=Path, action="append", required=True)
    parser.add_argument("--up-dir", choices=tuple(UP_ROTATIONS), default="Z")
    parser.add_argument("--stationary-z-max", type=float, default=0.08)
    parser.add_argument("--samples", type=int, default=120000)
    parser.add_argument("--fit-samples", type=int, default=40000)
    parser.add_argument("--trim-fraction", type=float, default=0.65)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    open_body = load_flat_mesh(args.open_body.expanduser().resolve())
    open_head = load_flat_mesh(args.open_head.expanduser().resolve())
    close_mesh = load_flat_mesh(args.close_mesh.expanduser().resolve())

    open_points, _ = trimesh.sample.sample_surface(open_body, args.samples, seed=rng)
    close_points, _ = trimesh.sample.sample_surface(close_mesh, args.samples, seed=rng)
    open_stationary = open_points[open_points[:, 2] < args.stationary_z_max]
    close_stationary = close_points[close_points[:, 2] < args.stationary_z_max]
    if len(open_stationary) > args.fit_samples:
        open_stationary = open_stationary[
            rng.choice(len(open_stationary), args.fit_samples, replace=False)
        ]
    if len(close_stationary) > args.fit_samples:
        close_stationary = close_stationary[
            rng.choice(len(close_stationary), args.fit_samples, replace=False)
        ]
    close_to_open, icp = trimmed_similarity_icp(
        close_stationary,
        open_stationary,
        trim_fraction=args.trim_fraction,
        iterations=args.iterations,
        maximum_distance=0.12,
    )
    transformed_close = transform_points(close_stationary, close_to_open)
    forward = distance_summary(transformed_close, open_stationary)
    backward = distance_summary(open_stationary, transformed_close)
    matrix = close_to_open[:3, :3]
    state_scale = float(np.cbrt(np.linalg.det(matrix)))
    state_rotation = matrix / state_scale

    raw_axes, raw_points, candidate_reports, moving_meshes = [], [], [], []
    for path in args.close_prediction:
        resolved = path.expanduser().resolve()
        axis, point, metadata, moving_mesh = prediction_axis_raw(resolved, args.up_dir)
        raw_axes.append(axis)
        raw_points.append(point)
        candidate_reports.append(metadata)
        moving_meshes.append(moving_mesh)
    close_axis, close_point, consensus = consensus_axis(raw_axes, raw_points)
    open_axis = state_rotation @ close_axis
    open_axis /= np.linalg.norm(open_axis)
    open_point = transform_points(close_point[None, :], close_to_open)[0]

    # Estimate the state angle from the largest predicted moving surface.  The
    # estimate is reported but gated because independent reconstructions may
    # contain different visible surfaces.
    selected_index = int(
        np.argmax([row["moving_face_fraction"] for row in candidate_reports])
    )
    moving_points, _ = trimesh.sample.sample_surface(
        moving_meshes[selected_index], 20000, seed=rng
    )
    moving_points = transform_points(moving_points, close_to_open)
    head_points, _ = trimesh.sample.sample_surface(open_head, 100000, seed=rng)
    head_tree = cKDTree(head_points)
    angle_scores = []
    for angle in np.linspace(-180.0, 180.0, 181):
        rotated = rotation_about_axis(moving_points, open_axis, open_point, float(angle))
        distances = np.sort(head_tree.query(rotated, k=1, workers=-1)[0])
        keep = max(1, int(0.8 * len(distances)))
        score = float(np.sqrt(np.mean(distances[:keep] ** 2)))
        angle_scores.append((score, float(angle)))
    coarse_score, coarse_angle = min(angle_scores)
    fine_scores = []
    for angle in np.linspace(coarse_angle - 2.0, coarse_angle + 2.0, 41):
        rotated = rotation_about_axis(moving_points, open_axis, open_point, float(angle))
        distances = np.sort(head_tree.query(rotated, k=1, workers=-1)[0])
        keep = max(1, int(0.8 * len(distances)))
        fine_scores.append((float(np.sqrt(np.mean(distances[:keep] ** 2))), float(angle)))
    angle_score, angle = min(fine_scores)
    object_diagonal = float(np.linalg.norm(np.vstack((open_body.bounds, open_head.bounds)).ptp(axis=0)))
    angle_accepted = angle_score / max(object_diagonal, 1e-12) <= 0.05

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "joint_type": "revolute",
        "parent": "body",
        "child": "head",
        "canonical_mesh_state": "open",
        "joint_zero_state": "closed",
        "axis_canonical_open": open_axis.tolist(),
        "point_canonical_open": open_point.tolist(),
        "axis_close_raw": close_axis.tolist(),
        "point_close_raw": close_point.tolist(),
        "close_to_canonical_open_similarity": close_to_open.tolist(),
        "state_alignment": {
            "scale": state_scale,
            "rotation": state_rotation.tolist(),
            "translation": close_to_open[:3, 3].tolist(),
            "stationary_z_max": args.stationary_z_max,
            "open_to_close_surface": backward,
            "close_to_open_surface": forward,
            "icp_last": icp["iterations"][-1],
        },
        "axis_consensus": consensus,
        "particulate_candidates": candidate_reports,
        "open_angle_estimate": {
            "degrees_from_closed_to_open": angle,
            "trimmed_rmse_80": angle_score,
            "rmse_over_object_diagonal": angle_score / max(object_diagonal, 1e-12),
            "accepted": angle_accepted,
            "usage": "initialization_only" if angle_accepted else "rejected_use_video_optimization",
        },
        "quality_gates": {
            "state_alignment_pass": max(
                forward["trimmed_rmse_80"], backward["trimmed_rmse_80"]
            ) / max(object_diagonal, 1e-12) < 0.03,
            "axis_consensus_pass": consensus["maximum_axis_deviation_degrees"] < 20.0,
            "angle_pass": angle_accepted,
        },
    }
    (output_dir / "hinge_prior.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    body_review = open_body.copy()
    body_review.visual.face_colors = [52, 199, 89, 255]
    head_review = open_head.copy()
    head_review.visual.face_colors = [255, 59, 48, 255]
    length = 0.5 * object_diagonal
    trimesh.Scene(
        [body_review, head_review, *make_axis_mesh(open_axis, open_point, length)]
    ).export(output_dir / "hinge_axis_review.glb")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
