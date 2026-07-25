"""Unified pose equivalence and observation-ambiguity handling."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from common.pose_transforms import axis_rotation


_EQUIVALENCE_TYPES = {"none", "continuous_axial", "cyclic"}
_AMBIGUITIES = {"axis_flip"}


def _unit(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("symmetry axis must be non-zero")
    return value / norm


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = (
        np.asarray(first, dtype=np.float64)[:3, :3].T
        @ np.asarray(second, dtype=np.float64)[:3, :3]
    )
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def _perpendicular_axis(axis: np.ndarray) -> np.ndarray:
    basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(axis)))]
    return _unit(np.cross(axis, basis))


@dataclass(frozen=True)
class SymmetrySpec:
    """Geometric pose equivalence plus non-equivalent observation ambiguities."""

    axis_raw: tuple[float, float, float] | None = None
    equivalence: str = "none"
    discrete_order: int | None = None
    observation_ambiguities: tuple[str, ...] = ()
    candidate_step_deg: float = 30.0
    continuity_weight: float = 1.0

    def __post_init__(self) -> None:
        equivalence = str(self.equivalence)
        if equivalence not in _EQUIVALENCE_TYPES:
            raise ValueError(
                f"unknown symmetry equivalence {equivalence!r}; "
                f"expected {sorted(_EQUIVALENCE_TYPES)}"
            )
        unknown = set(self.observation_ambiguities).difference(_AMBIGUITIES)
        if unknown:
            raise ValueError(f"unknown observation ambiguities: {sorted(unknown)}")
        requires_axis = (
            equivalence != "none" or bool(self.observation_ambiguities)
        )
        if requires_axis and self.axis_raw is None:
            raise ValueError("axial symmetry/ambiguity requires axis_raw")
        if self.axis_raw is not None:
            normalized = tuple(float(value) for value in _unit(self.axis_raw))
            object.__setattr__(self, "axis_raw", normalized)
        if equivalence == "cyclic":
            if self.discrete_order is None or int(self.discrete_order) < 2:
                raise ValueError("cyclic symmetry requires discrete_order >= 2")
            object.__setattr__(self, "discrete_order", int(self.discrete_order))
        if float(self.candidate_step_deg) <= 0.0:
            raise ValueError("candidate_step_deg must be positive")
        if float(self.continuity_weight) < 0.0:
            raise ValueError("continuity_weight must be non-negative")

    @property
    def axis(self) -> np.ndarray | None:
        return (
            None
            if self.axis_raw is None
            else np.asarray(self.axis_raw, dtype=np.float64)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "equivalence": self.equivalence,
            "axis_raw": (
                None if self.axis_raw is None else list(self.axis_raw)
            ),
            "discrete_order": self.discrete_order,
            "observation_ambiguities": list(
                self.observation_ambiguities
            ),
            "candidate_step_deg": float(self.candidate_step_deg),
            "continuity_weight": float(self.continuity_weight),
        }


@dataclass(frozen=True)
class SymmetryResolution:
    pose: np.ndarray
    equivalent_transform: np.ndarray
    axial_angle_deg: float
    axis_flipped: bool
    continuity_error_deg: float
    observable_axis_error_deg: float | None
    selection_source: str
    candidate_score: float | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "equivalent_transform": self.equivalent_transform.tolist(),
            "axial_angle_deg": float(self.axial_angle_deg),
            "axis_flipped": bool(self.axis_flipped),
            "continuity_error_deg": float(self.continuity_error_deg),
            "observable_axis_error_deg": (
                None
                if self.observable_axis_error_deg is None
                else float(self.observable_axis_error_deg)
            ),
            "selection_source": self.selection_source,
            "candidate_score": self.candidate_score,
        }


def symmetry_spec_from_state(state: Mapping[str, Any]) -> SymmetrySpec:
    """Read the new schema, with compatibility for the former split fields."""
    configured = state.get("symmetry")
    if configured is not None:
        configured = dict(configured)
        return SymmetrySpec(
            axis_raw=(
                None
                if configured.get("axis_raw") is None
                else tuple(configured["axis_raw"])
            ),
            equivalence=str(configured.get("equivalence", "none")),
            discrete_order=configured.get("discrete_order"),
            observation_ambiguities=tuple(
                configured.get("observation_ambiguities", ())
            ),
            candidate_step_deg=float(
                configured.get("candidate_step_deg", 30.0)
            ),
            continuity_weight=float(
                configured.get("continuity_weight", 1.0)
            ),
        )

    appearance = dict(state.get("appearance", {}))
    axis = state.get(
        "symmetry_axis_raw", appearance.get("symmetry_axis_raw")
    )
    mode = str(appearance.get("candidate_mode", "none"))
    if axis is None:
        return SymmetrySpec()
    equivalence = (
        "continuous_axial"
        if mode in {"axial", "axial_and_flip"} or mode == "none"
        else "none"
    )
    ambiguities = (
        ("axis_flip",)
        if mode in {"axis_flip", "axial_and_flip"}
        else ()
    )
    return SymmetrySpec(
        axis_raw=tuple(axis),
        equivalence=equivalence,
        observation_ambiguities=ambiguities,
        candidate_step_deg=float(appearance.get("angle_step_deg", 30.0)),
        continuity_weight=float(
            appearance.get("continuity_weight", 1.0)
        ),
    )


def symmetry_candidates(
    symmetry: SymmetrySpec,
    *,
    angle_step_deg: float | None = None,
    include_observation_ambiguities: bool = True,
) -> list[dict[str, Any]]:
    """Return local rotation candidates for appearance/depth scoring."""
    axis = symmetry.axis
    if symmetry.equivalence == "continuous_axial":
        step = float(angle_step_deg or symmetry.candidate_step_deg)
        count = max(1, int(round(360.0 / step)))
        angles = [index * 360.0 / count for index in range(count)]
    elif symmetry.equivalence == "cyclic":
        angles = [
            index * 360.0 / int(symmetry.discrete_order)
            for index in range(int(symmetry.discrete_order))
        ]
    else:
        angles = [0.0]

    branches = [("same", np.eye(4, dtype=np.float64))]
    if (
        include_observation_ambiguities
        and "axis_flip" in symmetry.observation_ambiguities
    ):
        branches.append(
            (
                "flipped",
                axis_rotation(_perpendicular_axis(axis), math.pi),
            )
        )

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for branch_name, branch in branches:
        for angle_deg in angles:
            axial = (
                np.eye(4, dtype=np.float64)
                if axis is None
                else axis_rotation(axis, math.radians(angle_deg))
            )
            local = branch @ axial
            key = tuple(np.round(local[:3, :3], decimals=9).ravel())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "label": f"{branch_name}:axis_{angle_deg:.3f}",
                    "axis_flipped": branch_name == "flipped",
                    "axis_angle_deg": float(angle_deg),
                    "local_transform": local,
                }
            )
    return candidates


def _closest_continuous_angle(
    pose_rotation: np.ndarray,
    reference_rotation: np.ndarray,
    axis: np.ndarray,
) -> float:
    """Analytic right-multiplied axial angle minimizing SO(3) distance."""
    relative = reference_rotation.T @ pose_rotation
    x, y, z = axis
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    alpha = float(np.trace(relative) - axis @ relative @ axis)
    beta = float(np.trace(relative @ skew))
    return math.atan2(beta, alpha)


def axis_direction_error_deg(
    first: np.ndarray,
    second: np.ndarray,
    axis_raw: np.ndarray,
) -> float:
    axis = _unit(axis_raw)
    first_axis = _unit(
        np.asarray(first, dtype=np.float64)[:3, :3] @ axis
    )
    second_axis = _unit(
        np.asarray(second, dtype=np.float64)[:3, :3] @ axis
    )
    cosine = float(np.clip(np.dot(first_axis, second_axis), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def resolve_symmetric_pose(
    measured_pose: np.ndarray,
    reference_pose: np.ndarray,
    symmetry: SymmetrySpec,
    *,
    candidate_scorer: Callable[[np.ndarray], float] | None = None,
    angle_step_deg: float | None = None,
    include_observation_ambiguities: bool = False,
) -> SymmetryResolution:
    """Choose one pose representative using evidence, then continuity.

    Continuous axial equivalence is solved analytically when no visual/depth
    scorer is supplied. A scorer triggers finite sampling because appearance
    can make the otherwise-equivalent axial angle semantically observable.
    Observation ambiguities such as ``axis_flip`` are excluded by default and
    only become candidates when explicitly requested.
    """
    measured = np.asarray(measured_pose, dtype=np.float64)
    reference = np.asarray(reference_pose, dtype=np.float64)
    if measured.shape != (4, 4) or reference.shape != (4, 4):
        raise ValueError("measured_pose and reference_pose must be 4x4")

    axis = symmetry.axis
    candidates: list[dict[str, Any]]
    if (
        candidate_scorer is None
        and symmetry.equivalence == "continuous_axial"
    ):
        branches = [("same", np.eye(4, dtype=np.float64))]
        if (
            include_observation_ambiguities
            and "axis_flip" in symmetry.observation_ambiguities
        ):
            branches.append(
                (
                    "flipped",
                    axis_rotation(_perpendicular_axis(axis), math.pi),
                )
            )
        candidates = []
        for branch_name, branch in branches:
            base = measured @ branch
            angle = _closest_continuous_angle(
                base[:3, :3], reference[:3, :3], axis
            )
            local = branch @ axis_rotation(axis, angle)
            candidates.append(
                {
                    "local_transform": local,
                    "axis_flipped": branch_name == "flipped",
                    "axis_angle_deg": math.degrees(angle) % 360.0,
                }
            )
    else:
        candidates = symmetry_candidates(
            symmetry,
            angle_step_deg=angle_step_deg,
            include_observation_ambiguities=include_observation_ambiguities,
        )

    scored = []
    for candidate in candidates:
        pose = measured @ candidate["local_transform"]
        continuity = _rotation_error_deg(reference, pose)
        evidence = (
            None
            if candidate_scorer is None
            else float(candidate_scorer(pose))
        )
        objective = (
            -continuity
            if evidence is None
            else evidence - symmetry.continuity_weight * continuity / 180.0
        )
        scored.append((objective, -continuity, candidate, pose, evidence))
    _, neg_continuity, selected, pose, evidence = max(
        scored, key=lambda item: (item[0], item[1])
    )
    observable_axis_error = (
        None
        if axis is None
        else axis_direction_error_deg(reference, pose, axis)
    )
    return SymmetryResolution(
        pose=pose,
        equivalent_transform=np.asarray(
            selected["local_transform"], dtype=np.float64
        ),
        axial_angle_deg=float(selected["axis_angle_deg"]),
        axis_flipped=bool(selected["axis_flipped"]),
        continuity_error_deg=float(-neg_continuity),
        observable_axis_error_deg=observable_axis_error,
        selection_source=(
            "candidate_score" if candidate_scorer is not None else "continuity"
        ),
        candidate_score=evidence,
    )


def symmetry_rotation_distance_deg(
    first: np.ndarray,
    second: np.ndarray,
    symmetry: SymmetrySpec,
) -> float:
    """SO(3) distance after removing geometric symmetry equivalence only."""
    resolved = resolve_symmetric_pose(
        second,
        first,
        symmetry,
        include_observation_ambiguities=False,
    )
    return resolved.continuity_error_deg
