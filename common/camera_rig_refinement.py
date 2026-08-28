"""Bounded local refinement of a fixed camera rig from static-part clouds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RigCorrespondence:
    """Fixed cross-view matches expressed in the initial world gauge."""

    view_a: int
    view_b: int
    points_a: np.ndarray
    points_b: np.ndarray
    rays_a: np.ndarray
    rays_b: np.ndarray
    frame: int

    def __post_init__(self) -> None:
        count = len(self.points_a)
        if count == 0:
            raise ValueError("a correspondence batch cannot be empty")
        for name in ("points_b", "rays_a", "rays_b"):
            value = np.asarray(getattr(self, name))
            if value.shape != (count, 3):
                raise ValueError(f"{name} must have shape ({count}, 3)")


def connected_components(
    view_count: int,
    edge_counts: dict[tuple[int, int], int],
) -> list[list[int]]:
    """Return camera-graph components, largest/most-supported first."""

    neighbours = [set() for _ in range(view_count)]
    for (left, right), count in edge_counts.items():
        if count <= 0:
            continue
        neighbours[left].add(right)
        neighbours[right].add(left)
    unseen = set(range(view_count))
    result: list[list[int]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        component = []
        unseen.remove(root)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(neighbours[current]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        result.append(sorted(component))

    def support(component: list[int]) -> int:
        selected = set(component)
        return sum(
            count
            for edge, count in edge_counts.items()
            if set(edge).issubset(selected)
        )

    return sorted(result, key=lambda item: (-len(item), -support(item), item))


def choose_anchor(
    active_views: Iterable[int],
    edge_counts: dict[tuple[int, int], int],
) -> int:
    """Choose the best-supported camera as the fixed gauge anchor."""

    active = sorted(set(int(value) for value in active_views))
    if not active:
        raise ValueError("at least one active view is required")
    degree = {view: 0 for view in active}
    for (left, right), count in edge_counts.items():
        if left in degree and right in degree:
            degree[left] += int(count)
            degree[right] += int(count)
    return min(active, key=lambda view: (-degree[view], view))


def _variable_views(active_views: Iterable[int], anchor: int) -> list[int]:
    active = sorted(set(int(value) for value in active_views))
    if anchor not in active:
        raise ValueError("anchor must be active")
    return [view for view in active if view != anchor]


def optimize_depth_corrections(
    correspondences: list[RigCorrespondence],
    *,
    view_count: int,
    active_views: Iterable[int],
    anchor: int,
    data_sigma_m: float = 0.005,
    prior_sigma_m: float = 0.015,
    maximum_correction_m: float = 0.03,
    max_nfev: int = 60,
) -> tuple[np.ndarray, dict]:
    """Estimate one additive along-ray depth correction per camera."""

    variables = _variable_views(active_views, anchor)
    index = {view: offset for offset, view in enumerate(variables)}
    if not correspondences or not variables:
        return np.zeros(view_count, np.float64), {
            "success": True,
            "message": "nothing to optimize",
            "nfev": 0,
            "cost": 0.0,
        }

    def unpack(parameters: np.ndarray) -> np.ndarray:
        correction = np.zeros(view_count, np.float64)
        for view, offset in index.items():
            correction[view] = parameters[offset]
        return correction

    def residual(parameters: np.ndarray) -> np.ndarray:
        correction = unpack(parameters)
        values = []
        for batch in correspondences:
            left = batch.points_a + correction[batch.view_a] * batch.rays_a
            right = batch.points_b + correction[batch.view_b] * batch.rays_b
            values.append(((left - right) / data_sigma_m).reshape(-1))
        values.append(parameters / prior_sigma_m)
        return np.concatenate(values)

    bound = float(maximum_correction_m)
    result = least_squares(
        residual,
        np.zeros(len(variables), np.float64),
        bounds=(-bound, bound),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    return unpack(result.x), {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
    }


def optimize_pose_corrections(
    correspondences: list[RigCorrespondence],
    depth_corrections_m: np.ndarray,
    *,
    view_count: int,
    active_views: Iterable[int],
    anchor: int,
    data_sigma_m: float = 0.005,
    rotation_prior_deg: float = 1.0,
    translation_prior_m: float = 0.01,
    maximum_rotation_deg: float = 3.0,
    maximum_translation_m: float = 0.03,
    max_nfev: int = 60,
) -> tuple[np.ndarray, dict]:
    """Estimate bounded world-frame SE(3) corrections for each camera cloud."""

    variables = _variable_views(active_views, anchor)
    index = {view: offset for offset, view in enumerate(variables)}
    deltas = np.tile(np.eye(4, dtype=np.float64), (view_count, 1, 1))
    if not correspondences or not variables:
        return deltas, {
            "success": True,
            "message": "nothing to optimize",
            "nfev": 0,
            "cost": 0.0,
        }
    depth_corrections_m = np.asarray(depth_corrections_m, np.float64)
    if depth_corrections_m.shape != (view_count,):
        raise ValueError("depth corrections must have one value per view")

    maximum_rotation_rad = np.radians(maximum_rotation_deg)

    def bounded_vector(value: np.ndarray, maximum: float) -> np.ndarray:
        """Project a three-vector onto a radial hard limit."""

        norm = float(np.linalg.norm(value))
        if norm <= maximum or norm <= 1e-12:
            return value
        return value * (maximum / norm)

    def unpack(parameters: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        rotations = [np.eye(3, dtype=np.float64) for _ in range(view_count)]
        translations = np.zeros((view_count, 3), np.float64)
        for view, offset in index.items():
            start = 6 * offset
            rotation_vector = bounded_vector(
                parameters[start:start + 3], maximum_rotation_rad
            )
            rotations[view] = Rotation.from_rotvec(
                rotation_vector
            ).as_matrix()
            translations[view] = bounded_vector(
                parameters[start + 3:start + 6], maximum_translation_m
            )
        return rotations, translations

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotations, translations = unpack(parameters)
        values = []
        for batch in correspondences:
            left = (
                batch.points_a
                + depth_corrections_m[batch.view_a] * batch.rays_a
            ) @ rotations[batch.view_a].T + translations[batch.view_a]
            right = (
                batch.points_b
                + depth_corrections_m[batch.view_b] * batch.rays_b
            ) @ rotations[batch.view_b].T + translations[batch.view_b]
            values.append(((left - right) / data_sigma_m).reshape(-1))
        priors = []
        for view in variables:
            priors.extend(
                Rotation.from_matrix(rotations[view]).as_rotvec()
                / np.radians(rotation_prior_deg)
            )
            priors.extend(translations[view] / translation_prior_m)
        values.append(np.asarray(priors, np.float64))
        return np.concatenate(values)

    lower = np.tile(
        np.asarray(
            [-maximum_rotation_rad] * 3
            + [-maximum_translation_m] * 3,
            np.float64,
        ),
        len(variables),
    )
    upper = -lower
    result = least_squares(
        residual,
        np.zeros(6 * len(variables), np.float64),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        x_scale="jac",
    )
    rotations, translations = unpack(result.x)
    for view in variables:
        deltas[view, :3, :3] = rotations[view]
        deltas[view, :3, 3] = translations[view]
    return deltas, {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
    }


def corrected_extrinsics(
    extrinsics: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    """Convert cloud-space corrections into corrected world-to-camera poses."""

    extrinsics = np.asarray(extrinsics, np.float64)
    deltas = np.asarray(deltas, np.float64)
    view_count = len(extrinsics)
    ext44 = np.tile(np.eye(4, dtype=np.float64), (view_count, 1, 1))
    ext44[:, :3, :4] = extrinsics[:, :3, :4]
    corrected = ext44 @ np.linalg.inv(deltas)
    return corrected[:, :3, :4]


def correspondence_metrics(
    correspondences: list[RigCorrespondence],
    depth_corrections_m: np.ndarray,
    deltas: np.ndarray,
    *,
    inlier_threshold_m: float = 0.01,
) -> dict:
    """Evaluate fixed matches without changing their membership."""

    depth_corrections_m = np.asarray(depth_corrections_m, np.float64)
    deltas = np.asarray(deltas, np.float64)
    per_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    all_distances = []
    for batch in correspondences:
        left = (
            batch.points_a
            + depth_corrections_m[batch.view_a] * batch.rays_a
        ) @ deltas[batch.view_a, :3, :3].T + deltas[batch.view_a, :3, 3]
        right = (
            batch.points_b
            + depth_corrections_m[batch.view_b] * batch.rays_b
        ) @ deltas[batch.view_b, :3, :3].T + deltas[batch.view_b, :3, 3]
        distances = np.linalg.norm(left - right, axis=1)
        pair = tuple(sorted((batch.view_a, batch.view_b)))
        per_pair.setdefault(pair, []).append(distances)
        all_distances.append(distances)

    def summarize(values: np.ndarray) -> dict:
        return {
            "samples": int(len(values)),
            "median_m": float(np.median(values)),
            "p90_m": float(np.quantile(values, 0.9)),
            "mean_m": float(np.mean(values)),
            "inlier_ratio": float(np.mean(values <= inlier_threshold_m)),
        }

    if not all_distances:
        return {
            "samples": 0,
            "median_m": None,
            "p90_m": None,
            "mean_m": None,
            "inlier_ratio": None,
            "pairs": [],
        }
    combined = np.concatenate(all_distances)
    summary = summarize(combined)
    summary["pairs"] = [
        {
            "views": list(pair),
            **summarize(np.concatenate(values)),
        }
        for pair, values in sorted(per_pair.items())
    ]
    return summary
