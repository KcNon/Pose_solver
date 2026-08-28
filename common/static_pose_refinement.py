"""Utilities for auditing and applying independently refined static poses."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from common.render_loss_refinement import world_pose_delta_vector
from common.trajectory_io import refresh_trajectory_derived_fields


def containing_static_range(
    config: dict[str, Any], part: str, frame: int
) -> tuple[int, int] | None:
    """Return the configured static interval containing ``frame``."""

    for start, end in config["states"][part].get("static_ranges", []):
        start, end = int(start), int(end)
        if start <= int(frame) <= end:
            return start, end
    return None


def align_pose_to_support_plane(
    pose: np.ndarray,
    vertices_part_m: np.ndarray,
    support_plane: dict[str, Any],
    *,
    contact_quantile: float = 0.005,
    maximum_shift_m: float = 0.02,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Translate a static rigid pose so its robust bottom touches the table.

    The input vertices must already be centered and expressed in metric part
    coordinates.  Rotation is intentionally left untouched: silhouettes
    determine orientation, while the observed support plane removes the weak
    depth/height degree of freedom.
    """

    transform = np.asarray(pose, dtype=np.float64).copy()
    vertices = np.asarray(vertices_part_m, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("pose must be a finite 4x4 matrix")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices_part_m must be a non-empty Nx3 array")
    quantile = float(contact_quantile)
    if not 0.0 <= quantile < 0.5:
        raise ValueError("contact_quantile must be in [0, 0.5)")
    normal = np.asarray(support_plane["normal_world"], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= 1e-12:
        raise ValueError("support plane normal must be finite and non-zero")
    normal /= normal_norm
    point = np.asarray(support_plane["point_world"], dtype=np.float64)
    world_vertices = (
        vertices @ transform[:3, :3].T + transform[:3, 3]
    )
    signed_distance = (world_vertices - point) @ normal
    bottom_gap = float(np.quantile(signed_distance, quantile))
    shift = -bottom_gap
    accepted = bool(
        np.isfinite(shift) and abs(shift) <= float(maximum_shift_m)
    )
    if accepted:
        transform[:3, 3] += shift * normal
    return transform, {
        "accepted": accepted,
        "contact_quantile": quantile,
        "bottom_gap_before_m": bottom_gap,
        "translation_along_normal_m": shift if accepted else 0.0,
        "bottom_gap_after_m": 0.0 if accepted else bottom_gap,
        "maximum_shift_m": float(maximum_shift_m),
        "reason": (
            None if accepted else "required_table_shift_exceeds_limit"
        ),
    }


def merge_static_pose_refinements(
    config: dict[str, Any],
    baseline: dict[str, Any],
    refinements: dict[str, dict[str, Any]],
    *,
    frame: int,
    propagate: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge one accepted per-part static pose into a canonical trajectory.

    Each refinement trajectory is expected to differ from ``baseline`` only
    through its requested part.  The caller may either keep the result as a
    one-frame visual diagnostic or propagate it across the containing static
    interval after manual/automatic anchor validation.
    """

    timestamp = f"{int(frame):06d}"
    if timestamp not in baseline["frames"]:
        raise ValueError(f"baseline does not contain frame {timestamp}")
    result = copy.deepcopy(baseline)
    audit: dict[str, Any] = {
        "frame": int(frame),
        "propagate": bool(propagate),
        "parts": {},
    }
    unknown = sorted(set(refinements).difference(baseline["parts"]))
    if unknown:
        raise ValueError(f"unknown refinement parts: {unknown}")
    for part, refined in refinements.items():
        if list(refined.get("parts", [])) != list(baseline["parts"]):
            raise ValueError(f"{part}: refinement parts do not match baseline")
        if timestamp not in refined.get("frames", {}):
            raise ValueError(f"{part}: refinement lacks frame {timestamp}")
        source_pose = np.asarray(
            refined["frames"][timestamp]["parts"][part]["T_world_from_part"],
            dtype=np.float64,
        )
        baseline_pose = np.asarray(
            baseline["frames"][timestamp]["parts"][part]["T_world_from_part"],
            dtype=np.float64,
        )
        if source_pose.shape != (4, 4) or not np.all(np.isfinite(source_pose)):
            raise ValueError(f"{part}: refined pose must be a finite 4x4 matrix")
        delta = world_pose_delta_vector(baseline_pose, source_pose)
        static_range = containing_static_range(config, part, int(frame))
        if propagate and static_range is None:
            raise ValueError(
                f"{part}: frame {frame} is not inside a configured static range"
            )
        apply_range = static_range if propagate else (int(frame), int(frame))
        applied = []
        for target_frame in range(apply_range[0], apply_range[1] + 1):
            target_id = f"{target_frame:06d}"
            if target_id not in result["frames"]:
                continue
            record = result["frames"][target_id]["parts"][part]
            record["T_world_from_part"] = source_pose.tolist()
            record["source"] = (
                str(record.get("source", "pose"))
                + "+full_silhouette_static_anchor"
            )
            applied.append(target_frame)
        audit["parts"][part] = {
            "static_range": list(static_range) if static_range else None,
            "applied_range": list(apply_range),
            "applied_frames": len(applied),
            "translation_delta_m": float(np.linalg.norm(delta[:3])),
            "rotation_delta_deg": float(np.degrees(np.linalg.norm(delta[3:]))),
        }
    refresh_trajectory_derived_fields(result)
    return result, audit
