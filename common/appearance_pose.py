from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from common.mask_io import frame_path
from common.multiview_quality import mask_area_quality
from common.normalized_recon import load_recon, scale_intrinsics
from common.pose_refinement import silhouette_metrics
from common.pose_tracking import load_part_cloud
from common.pose_transforms import rigid_from_similarity, similarity_from_rigid
from common.pose_transforms import axis_rotation_degrees
from common.symmetry import (
    SymmetrySpec,
    axis_direction_error_deg,
    symmetry_candidates,
    symmetry_rotation_distance_deg,
    symmetry_spec_from_state,
)


def candidate_local_rotations(
    axis: np.ndarray,
    mode: str,
    angle_step_deg: float = 30.0,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for the former appearance candidate API."""
    mode = str(mode).lower()
    if mode not in {"none", "axial", "axis_flip", "axial_and_flip"}:
        raise ValueError(f"Unsupported appearance candidate mode: {mode}")
    if angle_step_deg <= 0.0 or angle_step_deg > 360.0:
        raise ValueError("angle_step_deg must be in (0, 360]")
    symmetry = SymmetrySpec(
        axis_raw=tuple(np.asarray(axis, dtype=np.float64)),
        equivalence=(
            "continuous_axial"
            if mode in {"axial", "axial_and_flip"}
            else "none"
        ),
        observation_ambiguities=(
            ("axis_flip",)
            if mode in {"axis_flip", "axial_and_flip"}
            else ()
        ),
        candidate_step_deg=float(angle_step_deg),
    )
    return symmetry_candidates(symmetry)


def rotation_distance_deg(transform_a: np.ndarray, transform_b: np.ndarray) -> float:
    relative = np.asarray(transform_a, dtype=np.float64)[:3, :3].T @ np.asarray(
        transform_b, dtype=np.float64
    )[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def carry_pose_orientation(
    current_pose: np.ndarray,
    previous_pose: np.ndarray,
) -> np.ndarray:
    """Use the current centre with the previous anchor's orientation."""

    carried = np.asarray(current_pose, dtype=np.float64).copy()
    carried[:3, :3] = np.asarray(previous_pose, dtype=np.float64)[:3, :3]
    return carried


def align_pose_axis(
    pose: np.ndarray,
    axis_raw: np.ndarray,
    target_world: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Minimally rotate a rigid pose so a raw-mesh axis faces a world axis.

    Translation is intentionally preserved: the rigid pose is defined at the
    canonical part origin, so this creates an orientation hypothesis around
    that origin.  The later table-contact pass remains responsible for the
    small translation along the support-plane normal.
    """

    value = np.asarray(pose, dtype=np.float64).copy()
    source = np.asarray(axis_raw, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("axis_raw and target_world must be 3-vectors")
    if source_norm <= 1e-12 or target_norm <= 1e-12:
        raise ValueError("axis_raw and target_world must be non-zero")
    source /= source_norm
    target /= target_norm
    source_world = value[:3, :3] @ source
    cross = np.cross(source_world, target)
    dot = float(np.clip(np.dot(source_world, target), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    angle = math.acos(dot)
    if cross_norm <= 1e-12:
        if dot > 0.0:
            correction = np.eye(3, dtype=np.float64)
        else:
            trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(source_world[0])) > 0.9:
                trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            rotation_axis = np.cross(source_world, trial)
            rotation_axis /= np.linalg.norm(rotation_axis)
            correction, _ = cv2.Rodrigues(rotation_axis * angle)
    else:
        correction, _ = cv2.Rodrigues(cross / cross_norm * angle)
    value[:3, :3] = correction @ value[:3, :3]
    return value, math.degrees(angle)


def rigid_core_mask(
    mask: np.ndarray,
    *,
    opening_pixels: int,
    keep_largest: bool,
) -> np.ndarray:
    """Return the rigid silhouette core without changing source masks.

    This auxiliary mask prevents a thin cable, strap, or segmentation whisker
    from determining an otherwise rigid body's yaw.  If morphology removes
    everything, the original mask is returned so scoring remains defined.
    """

    original = np.asarray(mask, dtype=bool)
    if opening_pixels < 0:
        raise ValueError("opening_pixels must be non-negative")
    value = original.astype(np.uint8)
    if opening_pixels > 0:
        size = 2 * int(opening_pixels) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        value = cv2.morphologyEx(value, cv2.MORPH_OPEN, kernel)
    if keep_largest and int(value.sum()) > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            value, connectivity=8
        )
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            value = (labels == largest).astype(np.uint8)
    result = value.astype(bool)
    return result if int(result.sum()) > 0 else original.copy()


def table_yaw_angles(step_deg: float) -> list[float]:
    """Return a deterministic, non-duplicated full turn of yaw candidates."""

    if not 0.0 < float(step_deg) <= 360.0:
        raise ValueError("table_yaw_step_deg must be in (0, 360]")
    count = max(1, int(math.ceil(360.0 / float(step_deg))))
    return [360.0 * index / count for index in range(count)]


def _is_static_transition(
    frame_a: int,
    frame_b: int,
    static_ranges: list[list[int]] | list[tuple[int, int]],
) -> bool:
    # Anchors normally sit on the first/last dynamic frame, while the interval
    # between them is represented as a static range.  Treat those one-frame
    # boundary offsets as static too, otherwise an axial/flip ambiguity can
    # change by 180 degrees across an object that did not move for 100 frames.
    return any(
        int(start) <= frame_a + 1 and frame_b - 1 <= int(end)
        for start, end in static_ranges
    )


def select_candidate_chain(
    anchor_frames: list[int],
    candidate_rows: list[list[dict[str, Any]]],
    *,
    transition_weight: float,
    max_rotation_deg_per_frame: float,
    max_translation_m_per_frame: float | None = None,
    static_ranges: list[list[int]] | list[tuple[int, int]] | None = None,
    dynamic_ranges: list[list[int]] | list[tuple[int, int]] | None = None,
    transition_axis: np.ndarray | None = None,
    transition_symmetry: SymmetrySpec | None = None,
    hard_rotation_rate: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    """Select orientation hypotheses with a generic bounded-motion prior."""

    def selectable_score(row: dict[str, Any]) -> float:
        score = float(row.get("selection_score", row["score"]))
        # Semantic direction gates are hard. Relative silhouette quality is
        # encoded in ``selection_score`` as a large but finite penalty:
        # deleting every visually weak candidate can make a valid temporal
        # chain infeasible after a real manipulation. The selected weak anchor
        # is still rejected locally below and falls back to its geometry pose.
        if (
            row.get("semantic_candidate_gate_passed") is False
            or score <= -1.0e11
        ):
            return -np.inf
        return score

    if len(anchor_frames) != len(candidate_rows) or not anchor_frames:
        raise ValueError("anchor_frames and candidate_rows must be non-empty and have equal length")
    static_ranges = static_ranges or []
    dynamic_ranges = dynamic_ranges or []
    transition_axis = (
        None
        if transition_axis is None
        else np.asarray(transition_axis, dtype=np.float64)
    )
    if transition_axis is not None and transition_symmetry is not None:
        raise ValueError(
            "transition_axis and transition_symmetry are mutually exclusive"
        )
    if max_rotation_deg_per_frame <= 0:
        raise ValueError("max_rotation_deg_per_frame must be positive")

    scores: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    if any(not rows for rows in candidate_rows):
        raise RuntimeError("every anchor must have at least one orientation candidate")
    scores.append(np.asarray([
        selectable_score(row)
        for row in candidate_rows[0]
    ], dtype=np.float64))
    if not np.isfinite(scores[0]).any():
        raise RuntimeError(
            "no orientation candidate passes the configured candidate gates "
            f"at anchor {anchor_frames[0]}"
        )
    parents.append(np.full(len(candidate_rows[0]), -1, dtype=np.int32))
    transitions: list[list[list[float]]] = []

    for anchor_index in range(1, len(anchor_frames)):
        previous_rows = candidate_rows[anchor_index - 1]
        current_rows = candidate_rows[anchor_index]
        current_scores = np.full(len(current_rows), -np.inf, dtype=np.float64)
        current_parents = np.full(len(current_rows), -1, dtype=np.int32)
        transition_matrix: list[list[float]] = []
        frame_gap = max(1, anchor_frames[anchor_index] - anchor_frames[anchor_index - 1])
        if _is_static_transition(
            anchor_frames[anchor_index - 1], anchor_frames[anchor_index], static_ranges
        ):
            frame_gap = 1
        elif dynamic_ranges:
            # Stable waiting time between an anchor and the actual manipulation
            # cannot be used to justify a faster rotation. Count only frames in
            # declared motion intervals between the two anchors.
            motion_frames = sum(
                max(
                    0,
                    min(anchor_frames[anchor_index], int(end))
                    - max(anchor_frames[anchor_index - 1], int(start))
                    + 1,
                )
                for start, end in dynamic_ranges
            )
            frame_gap = max(1, motion_frames)

        for current_index, current in enumerate(current_rows):
            penalties: list[float] = []
            for previous_index, previous in enumerate(previous_rows):
                previous_pose = previous.get("transition_pose", previous["pose"])
                current_pose = current.get("transition_pose", current["pose"])
                if transition_symmetry is not None:
                    angle_deg = symmetry_rotation_distance_deg(
                        previous_pose,
                        current_pose,
                        transition_symmetry,
                    )
                elif transition_axis is None:
                    angle_deg = rotation_distance_deg(previous_pose, current_pose)
                else:
                    angle_deg = axis_direction_error_deg(
                        previous_pose,
                        current_pose,
                        transition_axis,
                    )
                normalized_rate = angle_deg / frame_gap / max_rotation_deg_per_frame
                penalty = (
                    float("inf")
                    if hard_rotation_rate and normalized_rate > 1.0
                    else float(transition_weight)
                    * normalized_rate
                    * normalized_rate
                )
                if max_translation_m_per_frame is not None:
                    if max_translation_m_per_frame <= 0.0:
                        raise ValueError(
                            "max_translation_m_per_frame must be positive"
                        )
                    translation = float(np.linalg.norm(
                        np.asarray(previous_pose)[:3, 3]
                        - np.asarray(current_pose)[:3, 3]
                    ))
                    normalized_translation = (
                        translation
                        / frame_gap
                        / max_translation_m_per_frame
                    )
                    penalty += (
                        float(transition_weight)
                        * normalized_translation
                        * normalized_translation
                    )
                penalties.append(penalty)
                value = (
                    scores[-1][previous_index]
                    + selectable_score(current)
                    - penalty
                )
                if value > current_scores[current_index]:
                    current_scores[current_index] = value
                    current_parents[current_index] = previous_index
            transition_matrix.append(penalties)
        transitions.append(transition_matrix)
        if not np.isfinite(current_scores).any():
            raise RuntimeError(
                "no orientation candidate satisfies the configured hard "
                f"rotation rate at anchor {anchor_frames[anchor_index]}"
            )
        scores.append(current_scores)
        parents.append(current_parents)

    selected = [0] * len(anchor_frames)
    selected[-1] = int(np.argmax(scores[-1]))
    for anchor_index in range(len(anchor_frames) - 1, 0, -1):
        selected[anchor_index - 1] = int(parents[anchor_index][selected[anchor_index]])

    diagnostics = {
        "selected_indices": selected,
        "path_score": float(scores[-1][selected[-1]]),
        "dynamic_programming_scores": [row.tolist() for row in scores],
        "transition_penalties": transitions,
        "transition_metric": (
            "geometric_symmetry_quotient"
            if transition_symmetry is not None
            else (
                "full_rotation"
                if transition_axis is None
                else "declared_axis_direction"
            )
        ),
        "max_translation_m_per_frame": max_translation_m_per_frame,
    }
    return selected, diagnostics


def _load_mask(path: Path, part_id: int, width: int, height: int) -> np.ndarray:
    mask = np.asarray(Image.open(path), dtype=np.uint8) == int(part_id)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def _load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _texture_edges(rgb: np.ndarray, mask: np.ndarray, erosion_pixels: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 140) > 0
    if erosion_pixels > 0:
        size = 2 * erosion_pixels + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        interior = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    else:
        interior = mask
    return edges & interior


def _symmetric_edge_chamfer(
    observed_edges: np.ndarray,
    rendered_edges: np.ndarray,
    cap_pixels: float,
) -> float | None:
    if int(observed_edges.sum()) < 4 or int(rendered_edges.sum()) < 4:
        return None
    observed_distance = cv2.distanceTransform((~observed_edges).astype(np.uint8), cv2.DIST_L2, 3)
    rendered_distance = cv2.distanceTransform((~rendered_edges).astype(np.uint8), cv2.DIST_L2, 3)
    forward = float(observed_distance[rendered_edges].mean())
    backward = float(rendered_distance[observed_edges].mean())
    return min(float(cap_pixels), 0.5 * (forward + backward))


def _masked_photometric_correlation(
    observed_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
    *,
    erosion_pixels: int,
    blur_sigma: float,
    minimum_pixels: int,
) -> float | None:
    """Lighting-normalized dense texture agreement on mutually visible pixels.

    Silhouette and binary edges cannot distinguish a 180-degree rotation of a
    nearly symmetric part.  The rendered and observed colors can.  Per-channel
    centering/scaling removes most global illumination and exposure changes;
    light Gaussian smoothing makes the score tolerant to small registration
    errors without erasing the part's spatial texture layout.
    """

    overlap = np.asarray(observed_mask, dtype=bool) & np.asarray(
        rendered_mask, dtype=bool
    )
    if erosion_pixels > 0:
        size = 2 * int(erosion_pixels) + 1
        overlap = cv2.erode(
            overlap.astype(np.uint8),
            np.ones((size, size), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    if int(overlap.sum()) < int(minimum_pixels):
        return None

    observed = cv2.cvtColor(
        np.asarray(observed_rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB
    ).astype(np.float32)
    rendered = cv2.cvtColor(
        np.asarray(rendered_rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB
    ).astype(np.float32)
    if blur_sigma > 0.0:
        observed = cv2.GaussianBlur(observed, (0, 0), float(blur_sigma))
        rendered = cv2.GaussianBlur(rendered, (0, 0), float(blur_sigma))

    correlations: list[float] = []
    # L carries printed/shaded texture; a/b carry color texture.  Ignore a
    # channel when either image is effectively uniform, because its normalized
    # correlation would only amplify quantization noise.
    for channel in range(3):
        left = observed[..., channel][overlap].astype(np.float64)
        right = rendered[..., channel][overlap].astype(np.float64)
        left -= np.median(left)
        right -= np.median(right)
        left_scale = float(np.sqrt(np.mean(left * left)))
        right_scale = float(np.sqrt(np.mean(right * right)))
        if left_scale < 2.0 or right_scale < 2.0:
            continue
        correlation = float(np.mean(
            (left / left_scale) * (right / right_scale)
        ))
        correlations.append(float(np.clip(correlation, -1.0, 1.0)))
    if not correlations:
        return None
    return float(np.mean(correlations))


def _resolve_anchor_evidence(
    appearance_cfg: dict[str, Any],
    anchor_frames: list[int],
) -> dict[int, list[int]]:
    raw = appearance_cfg.get("anchor_evidence_frames", {})
    resolved: dict[int, list[int]] = {}
    for frame in anchor_frames:
        values = raw.get(str(frame), raw.get(frame, [frame]))
        if isinstance(values, int):
            values = [values]
        resolved[frame] = [int(value) for value in values]
    return resolved


def _table_support_evidence(
    cfg: dict[str, Any],
    part: str,
    evidence_by_anchor: dict[int, list[int]],
    *,
    maximum_observed_bottom_gap_m: float,
) -> tuple[dict[int, bool], np.ndarray | None, np.ndarray | None]:
    plane = cfg.get("support_plane", {})
    cloud_root = cfg.get("point_cloud_root")
    if not plane.get("accepted", False) or not cloud_root:
        return {}, None, None
    normal = np.asarray(plane["normal_world"], dtype=np.float64)
    point = np.asarray(plane["point_world"], dtype=np.float64)
    applicable: dict[int, bool] = {}
    for anchor, frames in evidence_by_anchor.items():
        gaps = []
        for frame in frames:
            cloud = load_part_cloud(Path(cloud_root), int(frame), part)
            if cloud is None:
                continue
            signed = (np.asarray(cloud, dtype=np.float64) - point) @ normal
            gaps.append(float(np.quantile(signed, 0.02)))
        applicable[anchor] = bool(
            gaps
            and float(np.median(gaps)) <= maximum_observed_bottom_gap_m
        )
    return applicable, normal, point


def _table_support_score(
    vertices_raw: np.ndarray,
    similarity: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
    *,
    contact_quantile: float,
    gap_cap_m: float,
    penetration_tolerance_m: float,
) -> tuple[float, dict[str, float]]:
    world = (
        np.asarray(vertices_raw, dtype=np.float64)
        @ np.asarray(similarity, dtype=np.float64)[:3, :3].T
        + np.asarray(similarity, dtype=np.float64)[:3, 3]
    )
    signed = (world - plane_point) @ plane_normal
    contact_gap = float(np.quantile(signed, contact_quantile))
    lower_gap = float(np.quantile(signed, 0.001))
    gap_penalty = min(1.0, abs(contact_gap) / max(gap_cap_m, 1e-6))
    penetration = max(0.0, -lower_gap - penetration_tolerance_m)
    penetration_penalty = min(
        1.0, penetration / max(gap_cap_m, 1e-6)
    )
    score = -gap_penalty - penetration_penalty
    return float(score), {
        "contact_gap_m": contact_gap,
        "lower_gap_m": lower_gap,
        "gap_penalty": float(gap_penalty),
        "penetration_penalty": float(penetration_penalty),
    }


def normalize_similarity_to_support_plane(
    vertices_raw: np.ndarray,
    similarity: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
    *,
    contact_quantile: float,
    maximum_shift_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove table-normal translation before comparing orientation basins.

    Registration hypotheses can describe the same visible orientation with
    different table-normal translations.  Penalizing their absolute support
    gap before removing that nuisance translation can make a geometrically
    upside-down basin win merely because one of its vertices already touches
    the table.  Normalize every applicable candidate to the same robust
    contact height first; silhouette/texture then compare orientation rather
    than registration height.
    """

    transform = np.asarray(similarity, dtype=np.float64).copy()
    vertices = np.asarray(vertices_raw, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    point = np.asarray(plane_point, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("similarity must be a finite 4x4 matrix")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices_raw must be a non-empty Nx3 array")
    if normal_norm <= 1e-12 or not np.isfinite(normal_norm):
        raise ValueError("plane_normal must be finite and non-zero")
    quantile = float(contact_quantile)
    if not 0.0 <= quantile < 0.5:
        raise ValueError("contact_quantile must be in [0, 0.5)")
    normal /= normal_norm
    world = vertices @ transform[:3, :3].T + transform[:3, 3]
    signed = (world - point) @ normal
    gap_before = float(np.quantile(signed, quantile))
    shift = -gap_before
    accepted = bool(
        np.isfinite(shift) and abs(shift) <= float(maximum_shift_m)
    )
    if accepted:
        transform[:3, 3] += shift * normal
    return transform, {
        "accepted": accepted,
        "contact_quantile": quantile,
        "contact_gap_before_m": gap_before,
        "translation_along_normal_m": shift if accepted else 0.0,
        "contact_gap_after_m": 0.0 if accepted else gap_before,
        "maximum_shift_m": float(maximum_shift_m),
        "reason": None if accepted else "required_table_shift_exceeds_limit",
    }


def _range_pairs(values: list[Any]) -> list[list[int]]:
    pairs: list[list[int]] = []
    for item in values:
        if isinstance(item, dict):
            pairs.append([int(item["start"]), int(item["end"])])
        else:
            pairs.append([int(item[0]), int(item[1])])
    return pairs


def refine_anchor_orientations(
    *,
    cfg: dict[str, Any],
    state_cfg: dict[str, Any],
    part: str,
    mesh: Any,
    scale: float,
    origin: np.ndarray,
    anchors: dict[int, np.ndarray],
    anchor_hypotheses: dict[int, list[dict[str, Any]]] | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Resolve ambiguous anchor orientations from texture, masks and motion.

    The implementation is deliberately agnostic to semantic class names. All
    behavior is controlled by per-state object properties in the pipeline JSON.
    """

    appearance_cfg = dict(state_cfg.get("appearance", {}))
    if not appearance_cfg.get("enabled", False):
        return anchors, {"enabled": False}

    from common.mesh_render import SceneRenderer

    symmetry = symmetry_spec_from_state(state_cfg)
    candidate_symmetry_config = appearance_cfg.get("candidate_symmetry")
    candidate_symmetry = (
        symmetry
        if candidate_symmetry_config is None
        else symmetry_spec_from_state(
            {"symmetry": dict(candidate_symmetry_config)}
        )
    )
    candidates = symmetry_candidates(candidate_symmetry)
    width, height = [int(value) for value in appearance_cfg.get("resolution", [240, 135])]
    erosion_pixels = int(appearance_cfg.get("texture_erosion_pixels", 3))
    min_mask_pixels = int(appearance_cfg.get("min_mask_pixels", 80))
    edge_cap_pixels = float(appearance_cfg.get("edge_chamfer_cap_pixels", 30.0))
    silhouette_weight = float(appearance_cfg.get("silhouette_weight", 0.25))
    texture_weight = float(appearance_cfg.get("texture_weight", 1.0))
    photometric_weight = float(
        appearance_cfg.get("photometric_weight", 0.75)
    )
    photometric_erosion = int(
        appearance_cfg.get("photometric_erosion_pixels", 2)
    )
    photometric_blur_sigma = float(
        appearance_cfg.get("photometric_blur_sigma", 1.5)
    )
    photometric_minimum_pixels = int(
        appearance_cfg.get("photometric_minimum_pixels", 60)
    )
    rigid_core_weight = float(
        appearance_cfg.get("rigid_core_silhouette_weight", 0.0)
    )
    rigid_core_opening_pixels = int(
        appearance_cfg.get("rigid_core_opening_pixels", 0)
    )
    rigid_core_keep_largest = bool(
        appearance_cfg.get("rigid_core_keep_largest", True)
    )
    if rigid_core_weight < 0.0:
        raise ValueError("rigid_core_silhouette_weight must be non-negative")
    minimum_full_pixels = int(
        appearance_cfg.get("minimum_full_mask_pixels", 1)
    )
    maximum_area_ratio = float(
        appearance_cfg.get("maximum_mask_area_ratio", 4.0)
    )
    trim_worst_views = int(appearance_cfg.get("trim_worst_views", 0))
    worst_view_weight = float(appearance_cfg.get("worst_view_weight", 0.25))
    table_support_weight = float(
        appearance_cfg.get("table_support_weight", 0.3)
    )
    table_support_contact_quantile = float(
        appearance_cfg.get("table_support_contact_quantile", 0.01)
    )
    table_support_gap_cap_m = float(
        appearance_cfg.get("table_support_gap_cap_m", 0.02)
    )
    table_support_penetration_tolerance_m = float(
        appearance_cfg.get("table_support_penetration_tolerance_m", 0.005)
    )
    normalize_table_height = bool(
        appearance_cfg.get("normalize_table_height_before_scoring", True)
    )
    table_height_maximum_shift_m = float(
        appearance_cfg.get("table_height_maximum_shift_m", 0.05)
    )
    support_facing_axis_raw = appearance_cfg.get("support_facing_axis_raw")
    if support_facing_axis_raw is not None:
        support_facing_axis_raw = np.asarray(
            support_facing_axis_raw, dtype=np.float64
        )
        support_facing_norm = float(np.linalg.norm(support_facing_axis_raw))
        if support_facing_axis_raw.shape != (3,) or support_facing_norm <= 1e-12:
            raise ValueError(
                f"{part}: appearance.support_facing_axis_raw must be a "
                "non-zero 3-vector"
            )
        support_facing_axis_raw = support_facing_axis_raw / support_facing_norm
    minimum_support_facing_alignment = float(
        appearance_cfg.get("minimum_support_facing_alignment", 0.0)
    )
    if not -1.0 <= minimum_support_facing_alignment <= 1.0:
        raise ValueError(
            "appearance.minimum_support_facing_alignment must be in [-1, 1]"
        )
    opening_axis_raw = appearance_cfg.get("opening_axis_raw")
    if opening_axis_raw is not None:
        opening_axis_raw = np.asarray(opening_axis_raw, dtype=np.float64)
        opening_axis_norm = float(np.linalg.norm(opening_axis_raw))
        if opening_axis_raw.shape != (3,) or opening_axis_norm <= 1e-12:
            raise ValueError(
                f"{part}: appearance.opening_axis_raw must be a non-zero "
                "3-vector"
            )
        opening_axis_raw = opening_axis_raw / opening_axis_norm
    minimum_opening_up_alignment = float(
        appearance_cfg.get("minimum_opening_up_alignment", 0.0)
    )
    maximum_opening_up_alignment = float(
        appearance_cfg.get("maximum_opening_up_alignment", 1.0)
    )
    if not (
        -1.0 <= minimum_opening_up_alignment
        <= maximum_opening_up_alignment <= 1.0
    ):
        raise ValueError(
            "appearance opening alignment bounds must satisfy "
            "-1 <= minimum_opening_up_alignment <= "
            "maximum_opening_up_alignment <= 1"
        )

    frames_root = Path(cfg["frames_dir"])
    mask_root = Path(cfg["masks_dir"])
    views = [str(value) for value in appearance_cfg.get("views", cfg["views"])]
    view_indices = {view: cfg["views"].index(view) for view in views}
    part_id = int(cfg["part_ids"][part])
    anchor_frames = sorted(anchors)
    evidence_by_anchor = _resolve_anchor_evidence(appearance_cfg, anchor_frames)
    support_applicable, support_normal, support_point = _table_support_evidence(
        cfg,
        part,
        evidence_by_anchor,
        maximum_observed_bottom_gap_m=float(
            appearance_cfg.get(
                "table_support_maximum_observed_bottom_gap_m", 0.05
            )
        ),
    )
    configured_support_ranges = (
        cfg.get("automation", {}).get("table_support_ranges", {}).get(part)
    )
    semantic_support_applicable = {
        anchor_frame: bool(
            support_applicable.get(anchor_frame, False)
            or (
                configured_support_ranges is not None
                and any(
                    int(start) <= anchor_frame <= int(end)
                    for start, end in configured_support_ranges
                )
            )
        )
        for anchor_frame in anchor_frames
    }
    table_yaw_enabled = bool(
        appearance_cfg.get("table_yaw_search", False)
        and candidate_symmetry.equivalence == "none"
        and support_normal is not None
    )
    table_yaw_frames = set(anchor_frames)
    if (
        table_yaw_enabled
        and appearance_cfg.get("table_yaw_first_static_interval_only", True)
    ):
        table_yaw_frames = set()
        for start, end in _range_pairs(state_cfg.get("static_ranges", [])):
            inside = {
                frame for frame in anchor_frames if start <= frame <= end
            }
            if inside:
                table_yaw_frames = inside
                break
    yaw_angles = table_yaw_angles(
        float(appearance_cfg.get("table_yaw_step_deg", 30.0))
    )
    vertices_raw = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices_raw) > 20000:
        vertices_raw = vertices_raw[:: max(1, len(vertices_raw) // 20000)][
            :20000
        ]

    observations: dict[int, list[dict[str, Any]]] = {}
    observation_quality: dict[int, list[dict[str, Any]]] = {}
    for anchor_frame in anchor_frames:
        rows: list[dict[str, Any]] = []
        quality_rows: list[dict[str, Any]] = []
        for timestamp in evidence_by_anchor[anchor_frame]:
            recon = load_recon(cfg, f"{timestamp:06d}", backend=cfg["recon_backend"])
            labels_by_view: dict[str, np.ndarray] = {}
            for view in views:
                mask_path = mask_root / f"{timestamp:06d}" / f"{view}.png"
                if mask_path.exists():
                    labels_by_view[view] = np.asarray(
                        Image.open(mask_path), dtype=np.uint8
                    )
            quality = mask_area_quality(
                labels_by_view,
                part_id,
                minimum_pixels=minimum_full_pixels,
                maximum_area_ratio=maximum_area_ratio,
            )
            quality_rows.append({"timestamp": timestamp, **quality})
            for view in views:
                image_path = Path(
                    frame_path(
                        str(frames_root),
                        cfg.get("frames_layout", "normalized"),
                        f"{timestamp:06d}",
                        view,
                    )
                )
                mask_path = mask_root / f"{timestamp:06d}" / f"{view}.png"
                if not image_path.exists() or not mask_path.exists():
                    continue
                if not quality["views"].get(view, {}).get("valid", False):
                    continue
                target = _load_mask(mask_path, part_id, width, height)
                if int(target.sum()) < min_mask_pixels:
                    continue
                rgb = _load_rgb(image_path, width, height)
                view_index = view_indices[view]
                intrinsics = scale_intrinsics(
                    recon["intrinsics"][view_index],
                    recon["depth_hw"],
                    (height, width),
                )
                rows.append(
                    {
                        "timestamp": timestamp,
                        "view": view,
                        "target": target,
                        "target_rigid_core": rigid_core_mask(
                            target,
                            opening_pixels=rigid_core_opening_pixels,
                            keep_largest=rigid_core_keep_largest,
                        ),
                        "observed_rgb": rgb,
                        "observed_edges": _texture_edges(
                            rgb, target, erosion_pixels=erosion_pixels
                        ),
                        "intrinsics": intrinsics,
                        "world_from_camera": recon["extrinsics"][view_index],
                    }
                )
        observations[anchor_frame] = rows
        observation_quality[anchor_frame] = quality_rows

    renderer = SceneRenderer(
        width=width,
        height=height,
        cache_mesh_resources=True,
    )
    all_candidate_rows: list[list[dict[str, Any]]] = []
    per_anchor_report: dict[str, Any] = {}
    try:
        for anchor_index, anchor_frame in enumerate(anchor_frames):
            scored_rows: list[dict[str, Any]] = []
            hypotheses = list(
                (anchor_hypotheses or {}).get(anchor_frame)
                or [{"label": "registration", "similarity": anchors[anchor_frame]}]
            )
            if (
                anchor_index > 0
                and bool(appearance_cfg.get(
                    "include_previous_anchor_orientation_candidate", True
                ))
            ):
                previous_frame = anchor_frames[anchor_index - 1]
                current_rigid = rigid_from_similarity(
                    np.asarray(anchors[anchor_frame], dtype=np.float64),
                    origin,
                )
                previous_rigid = rigid_from_similarity(
                    np.asarray(anchors[previous_frame], dtype=np.float64),
                    origin,
                )
                carried_rigid = carry_pose_orientation(
                    current_rigid,
                    previous_rigid,
                )
                hypotheses.append({
                    "label": f"temporal_hold_{previous_frame}",
                    "similarity": similarity_from_rigid(
                        carried_rigid,
                        scale,
                        origin,
                    ),
                })
            if (
                bool(appearance_cfg.get(
                    "generate_support_aligned_candidates", True
                ))
                and support_facing_axis_raw is not None
                and support_normal is not None
                and semantic_support_applicable.get(anchor_frame, False)
            ):
                expanded_hypotheses: list[dict[str, Any]] = []
                for hypothesis in hypotheses:
                    expanded_hypotheses.append(hypothesis)
                    rigid = rigid_from_similarity(
                        np.asarray(hypothesis["similarity"], dtype=np.float64),
                        origin,
                    )
                    aligned_rigid, correction_deg = align_pose_axis(
                        rigid,
                        support_facing_axis_raw,
                        -support_normal,
                    )
                    if correction_deg > 1e-4:
                        expanded_hypotheses.append({
                            "label": (
                                f"{hypothesis['label']}|support_aligned"
                            ),
                            "similarity": similarity_from_rigid(
                                aligned_rigid, scale, origin
                            ),
                            "semantic_support_aligned": True,
                            "semantic_support_correction_deg": float(
                                correction_deg
                            ),
                        })
                hypotheses = expanded_hypotheses
            for hypothesis in hypotheses:
                rigid_base = rigid_from_similarity(
                    np.asarray(hypothesis["similarity"], dtype=np.float64),
                    origin,
                )
                anchor_candidates = candidates
                if (
                    table_yaw_enabled
                    and anchor_frame in table_yaw_frames
                    and semantic_support_applicable.get(anchor_frame, False)
                ):
                    anchor_candidates = [
                        {
                            **candidate,
                            "label": (
                                f"{candidate['label']}|world_yaw_"
                                f"{angle_deg:.3f}"
                            ),
                            "world_yaw_deg": float(angle_deg),
                        }
                        for candidate in candidates
                        for angle_deg in yaw_angles
                    ]
                for candidate in anchor_candidates:
                    pose = rigid_base @ candidate["local_transform"]
                    world_yaw_deg = float(candidate.get("world_yaw_deg", 0.0))
                    if abs(world_yaw_deg) > 1e-12:
                        world_yaw = axis_rotation_degrees(
                            support_normal, world_yaw_deg
                        )[:3, :3]
                        pose = pose.copy()
                        pose[:3, :3] = world_yaw @ pose[:3, :3]
                    similarity_transform = similarity_from_rigid(
                        pose, scale, origin
                    )
                    table_height_normalization = None
                    if (
                        normalize_table_height
                        and support_applicable.get(anchor_frame, False)
                        and support_normal is not None
                        and support_point is not None
                    ):
                        (
                            normalized_similarity,
                            table_height_normalization,
                        ) = normalize_similarity_to_support_plane(
                            vertices_raw,
                            similarity_transform,
                            support_normal,
                            support_point,
                            contact_quantile=(
                                table_support_contact_quantile
                            ),
                            maximum_shift_m=(
                                table_height_maximum_shift_m
                            ),
                        )
                        if table_height_normalization["accepted"]:
                            translation_delta = (
                                normalized_similarity[:3, 3]
                                - similarity_transform[:3, 3]
                            )
                            pose = pose.copy()
                            pose[:3, 3] += translation_delta
                            similarity_transform = normalized_similarity
                    support_facing_alignment = None
                    if (
                        support_facing_axis_raw is not None
                        and support_normal is not None
                        and semantic_support_applicable.get(anchor_frame, False)
                    ):
                        # The configured raw-mesh vector points toward the
                        # physical support side (the underside).  A table's
                        # accepted normal points away from the support, so the
                        # two directions should be antiparallel.  Silhouette
                        # and depth cannot distinguish a 180-degree flip of a
                        # nearly symmetric part; this explicit semantic bit is
                        # therefore a gate, not a weak visual heuristic.
                        support_facing_alignment = float(np.dot(
                            pose[:3, :3] @ support_facing_axis_raw,
                            -support_normal,
                        ))
                    opening_up_alignment = None
                    if (
                        opening_axis_raw is not None
                        and support_normal is not None
                        and semantic_support_applicable.get(anchor_frame, False)
                    ):
                        # An opening is a directional semantic feature, not a
                        # support face.  It must point into the table's upper
                        # hemisphere.  Unlike support alignment, this never
                        # creates or rotates a hypothesis: mask/cloud/render
                        # evidence still determines the exact slant angle.
                        opening_up_alignment = float(np.dot(
                            pose[:3, :3] @ opening_axis_raw,
                            support_normal,
                        ))
                    observation_scores: list[dict[str, Any]] = []
                    for observation in observations[anchor_frame]:
                        rendered_rgb, rendered_depth = renderer.render(
                            [(mesh, similarity_transform)],
                            observation["intrinsics"],
                            observation["world_from_camera"],
                        )
                        predicted = rendered_depth > 0.0
                        (
                            silhouette_iou,
                            silhouette_chamfer,
                            silhouette_score,
                        ) = silhouette_metrics(predicted, observation["target"])
                        predicted_rigid_core = rigid_core_mask(
                            predicted,
                            opening_pixels=rigid_core_opening_pixels,
                            keep_largest=rigid_core_keep_largest,
                        )
                        (
                            rigid_core_iou,
                            rigid_core_chamfer,
                            rigid_core_score,
                        ) = silhouette_metrics(
                            predicted_rigid_core,
                            observation["target_rigid_core"],
                        )
                        rendered_edges = _texture_edges(
                            rendered_rgb,
                            predicted,
                            erosion_pixels=erosion_pixels,
                        )
                        edge_chamfer = _symmetric_edge_chamfer(
                            observation["observed_edges"],
                            rendered_edges,
                            edge_cap_pixels,
                        )
                        texture_score = (
                            -edge_cap_pixels
                            if edge_chamfer is None
                            else -float(edge_chamfer)
                        ) / edge_cap_pixels
                        photometric_correlation = (
                            _masked_photometric_correlation(
                                observation["observed_rgb"],
                                rendered_rgb,
                                observation["target"],
                                predicted,
                                erosion_pixels=photometric_erosion,
                                blur_sigma=photometric_blur_sigma,
                                minimum_pixels=photometric_minimum_pixels,
                            )
                            if photometric_weight > 0.0
                            else None
                        )
                        score = (
                            silhouette_weight * float(silhouette_score)
                            + rigid_core_weight * float(rigid_core_score)
                            + texture_weight * texture_score
                            + photometric_weight
                            * (
                                0.0
                                if photometric_correlation is None
                                else photometric_correlation
                            )
                        )
                        observation_scores.append(
                            {
                                "timestamp": int(observation["timestamp"]),
                                "view": observation["view"],
                                "score": float(score),
                                "silhouette_iou": float(silhouette_iou),
                                "silhouette_chamfer_px": float(
                                    silhouette_chamfer
                                ),
                                "rigid_core_iou": float(rigid_core_iou),
                                "rigid_core_chamfer_px": float(
                                    rigid_core_chamfer
                                ),
                                "texture_edge_chamfer_px": (
                                    None
                                    if edge_chamfer is None
                                    else float(edge_chamfer)
                                ),
                                "photometric_correlation": (
                                    None
                                    if photometric_correlation is None
                                    else float(photometric_correlation)
                                ),
                            }
                        )
                    ordered_scores = sorted(
                        observation_scores,
                        key=lambda row: float(row["score"]),
                        reverse=True,
                    )
                    kept_scores = (
                        ordered_scores[:-trim_worst_views]
                        if trim_worst_views > 0
                        and len(ordered_scores) > trim_worst_views + 1
                        else ordered_scores
                    )
                    aggregate = (
                        float(np.mean([row["score"] for row in kept_scores]))
                        if kept_scores
                        else -1e6
                    )
                    if observation_scores:
                        aggregate += worst_view_weight * min(
                            float(row["score"]) for row in observation_scores
                        )
                    table_support_score = None
                    table_support_report = None
                    if (
                        table_support_weight > 0.0
                        and support_applicable.get(anchor_frame, False)
                        and support_normal is not None
                        and support_point is not None
                    ):
                        (
                            table_support_score,
                            table_support_report,
                        ) = _table_support_score(
                            vertices_raw,
                            similarity_transform,
                            support_normal,
                            support_point,
                            contact_quantile=table_support_contact_quantile,
                            gap_cap_m=table_support_gap_cap_m,
                            penetration_tolerance_m=(
                                table_support_penetration_tolerance_m
                            ),
                        )
                        aggregate += (
                            table_support_weight * table_support_score
                        )
                    mean_iou = (
                        float(
                            np.mean(
                                [row["silhouette_iou"] for row in kept_scores]
                            )
                        )
                        if kept_scores
                        else 0.0
                    )
                    mean_silhouette_chamfer = (
                        float(
                            np.mean(
                                [
                                    row["silhouette_chamfer_px"]
                                    for row in kept_scores
                                ]
                            )
                        )
                        if kept_scores
                        else 100.0
                    )
                    mean_rigid_core_iou = (
                        float(np.mean([
                            row["rigid_core_iou"] for row in kept_scores
                        ]))
                        if kept_scores
                        else 0.0
                    )
                    edge_values = [
                        row["texture_edge_chamfer_px"]
                        for row in kept_scores
                        if row["texture_edge_chamfer_px"] is not None
                    ]
                    photometric_values = [
                        row["photometric_correlation"]
                        for row in kept_scores
                        if row["photometric_correlation"] is not None
                    ]
                    scored_rows.append(
                        {
                            "label": (
                                f"{hypothesis['label']}|{candidate['label']}"
                            ),
                            "hypothesis": str(hypothesis["label"]),
                            "semantic_support_aligned": bool(
                                hypothesis.get(
                                    "semantic_support_aligned", False
                                )
                            ),
                            "semantic_support_correction_deg": hypothesis.get(
                                "semantic_support_correction_deg"
                            ),
                            "axis_flipped": bool(candidate["axis_flipped"]),
                            "axis_angle_deg": float(candidate["axis_angle_deg"]),
                            "world_yaw_deg": float(
                                candidate.get("world_yaw_deg", 0.0)
                            ),
                            "pose": pose,
                            "score": aggregate,
                            "mean_silhouette_iou": mean_iou,
                            "worst_silhouette_iou": (
                                float(min(
                                    row["silhouette_iou"]
                                    for row in observation_scores
                                ))
                                if observation_scores
                                else 0.0
                            ),
                            "mean_silhouette_chamfer_px": (
                                mean_silhouette_chamfer
                            ),
                            "mean_rigid_core_iou": mean_rigid_core_iou,
                            "mean_texture_edge_chamfer_px": (
                                float(np.mean(edge_values))
                                if edge_values
                                else None
                            ),
                            "mean_photometric_correlation": (
                                float(np.mean(photometric_values))
                                if photometric_values
                                else None
                            ),
                            "table_support_score": table_support_score,
                            "table_support": table_support_report,
                            "table_height_normalization": (
                                table_height_normalization
                            ),
                            "support_facing_alignment": (
                                support_facing_alignment
                            ),
                            "opening_up_alignment": opening_up_alignment,
                            "observations": observation_scores,
                            "aggregated_views": [
                                row["view"] for row in kept_scores
                            ],
                        }
                    )
            for row in scored_rows:
                support_alignment = row.get("support_facing_alignment")
                row["support_facing_gate_passed"] = bool(
                    support_facing_axis_raw is None
                    or support_alignment is None
                    or float(support_alignment)
                    >= minimum_support_facing_alignment
                )
                opening_alignment = row.get("opening_up_alignment")
                row["opening_direction_gate_passed"] = bool(
                    opening_axis_raw is None
                    or opening_alignment is None
                    or (
                        minimum_opening_up_alignment
                        <= float(opening_alignment)
                        <= maximum_opening_up_alignment
                    )
                )
            semantic_eligible_rows = [
                row
                for row in scored_rows
                if row["support_facing_gate_passed"]
                and row["opening_direction_gate_passed"]
            ]
            best_candidate_iou = max(
                (
                    float(row["mean_silhouette_iou"])
                    for row in semantic_eligible_rows
                ),
                default=0.0,
            )
            maximum_iou_drop = appearance_cfg.get(
                "maximum_candidate_iou_drop", 0.04
            )
            relative_iou_gate_penalty = float(
                appearance_cfg.get("relative_iou_gate_penalty", 1.0)
            )
            minimum_iou_ratio = appearance_cfg.get(
                "minimum_candidate_iou_ratio_to_best"
            )
            for row in scored_rows:
                candidate_iou = float(row["mean_silhouette_iou"])
                silhouette_gate_passed = bool(
                    (
                        maximum_iou_drop is None
                        or candidate_iou
                        >= best_candidate_iou - float(maximum_iou_drop)
                    )
                    and (
                        minimum_iou_ratio is None
                        or candidate_iou
                        >= best_candidate_iou * float(minimum_iou_ratio)
                    )
                )
                support_facing_gate_passed = bool(
                    row["support_facing_gate_passed"]
                )
                opening_direction_gate_passed = bool(
                    row["opening_direction_gate_passed"]
                )
                semantic_gate_passed = bool(
                    support_facing_gate_passed
                    and opening_direction_gate_passed
                )
                gate_passed = bool(
                    silhouette_gate_passed and semantic_gate_passed
                )
                row["silhouette_candidate_gate_passed"] = (
                    silhouette_gate_passed
                )
                row["semantic_candidate_gate_passed"] = (
                    semantic_gate_passed
                )
                row["best_candidate_iou"] = best_candidate_iou
                row["candidate_gate_passed"] = gate_passed
                row["selection_score"] = (
                    float(row["score"])
                    - (
                        0.0
                        if silhouette_gate_passed
                        else relative_iou_gate_penalty
                    )
                    if semantic_gate_passed
                    else -1.0e12
                )
            # A visually weak candidate is allowed to keep the temporal chain
            # feasible, but it is not applied below: that anchor falls back to
            # the geometry pose. The chain must therefore score transitions
            # against the pose that will actually be written. Otherwise a
            # rejected candidate can hide a near-180-degree discontinuity
            # between the geometry fallback and the next accepted anchor.
            fallback_rigid = rigid_from_similarity(
                np.asarray(anchors[anchor_frame], dtype=np.float64),
                origin,
            )
            minimum_iou = float(appearance_cfg.get("minimum_mean_iou", 0.0))
            minimum_worst_iou = float(
                appearance_cfg.get("minimum_worst_view_iou", 0.0)
            )
            minimum_observations = int(
                appearance_cfg.get("minimum_observations", 1)
            )
            for row in scored_rows:
                will_apply = bool(
                    bool(row.get("silhouette_candidate_gate_passed", False))
                    and bool(row.get("support_facing_gate_passed", False))
                    and bool(row.get("opening_direction_gate_passed", False))
                    and float(row["mean_silhouette_iou"]) >= minimum_iou
                    and float(row["worst_silhouette_iou"])
                    >= minimum_worst_iou
                    and len(row["observations"]) >= minimum_observations
                )
                row["transition_pose"] = (
                    row["pose"] if will_apply else fallback_rigid
                )
            all_candidate_rows.append(scored_rows)
            per_anchor_report[str(anchor_frame)] = {
                "evidence_frames": evidence_by_anchor[anchor_frame],
                "observation_count": len(observations[anchor_frame]),
                "observation_quality": observation_quality[anchor_frame],
                "table_support_applicable": bool(
                    support_applicable.get(anchor_frame, False)
                ),
                "semantic_support_applicable": bool(
                    semantic_support_applicable.get(anchor_frame, False)
                ),
                "candidates": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"pose", "transition_pose", "observations"}
                    }
                    for row in scored_rows
                ],
            }
            print(
                f"appearance {part}@{anchor_frame}: "
                f"observations={len(observations[anchor_frame])} "
                f"candidates={len(scored_rows)}",
                flush=True,
            )
    finally:
        renderer.close()

    selected, chain_report = select_candidate_chain(
        anchor_frames,
        all_candidate_rows,
        transition_weight=float(appearance_cfg.get("transition_weight", 0.5)),
        max_rotation_deg_per_frame=float(
            appearance_cfg.get("max_rotation_deg_per_frame", 10.0)
        ),
        max_translation_m_per_frame=float(
            appearance_cfg.get("max_translation_m_per_frame", 0.05)
        ),
        static_ranges=_range_pairs(state_cfg.get("static_ranges", [])),
        dynamic_ranges=_range_pairs(state_cfg.get("dynamic_ranges", [])),
        transition_symmetry=symmetry,
        hard_rotation_rate=bool(
            appearance_cfg.get("hard_rotation_rate", True)
        ),
    )
    updated: dict[int, np.ndarray] = {}
    for anchor_index, anchor_frame in enumerate(anchor_frames):
        chosen = all_candidate_rows[anchor_index][selected[anchor_index]]
        minimum_iou = float(appearance_cfg.get("minimum_mean_iou", 0.0))
        minimum_worst_iou = float(
            appearance_cfg.get("minimum_worst_view_iou", 0.0)
        )
        minimum_observations = int(
            appearance_cfg.get("minimum_observations", 1)
        )
        acceptable = bool(
            bool(chosen.get("silhouette_candidate_gate_passed", False))
            and bool(chosen.get("support_facing_gate_passed", False))
            and bool(chosen.get("opening_direction_gate_passed", False))
            and float(chosen["mean_silhouette_iou"]) >= minimum_iou
            and float(chosen["worst_silhouette_iou"]) >= minimum_worst_iou
            and len(chosen["observations"]) >= minimum_observations
        )
        if acceptable:
            updated[anchor_frame] = similarity_from_rigid(
                chosen["pose"], scale, origin
            )
            selected_report = {
                key: value
                for key, value in chosen.items()
                if key not in {"pose", "transition_pose", "observations"}
            }
            selected_report["appearance_accepted"] = True
        else:
            # Sparse first appearances often expose a part in only one or two
            # cameras.  Reject that visual decision locally and retain the
            # geometry anchor; later well-observed anchors and render loss can
            # still constrain the trajectory without a manual keyframe.
            updated[anchor_frame] = anchors[anchor_frame]
            selected_report = {
                "appearance_accepted": False,
                "fallback": "geometry_anchor",
                "reason": (
                    "support_facing_gate_failed"
                    if not chosen.get("support_facing_gate_passed", False)
                    else (
                        "opening_direction_gate_failed"
                        if not chosen.get(
                            "opening_direction_gate_passed", False
                        )
                        else (
                            "relative_silhouette_gate_failed"
                            if not chosen.get(
                                "silhouette_candidate_gate_passed", False
                            )
                            else (
                                "insufficient_observations"
                                if len(chosen["observations"])
                                < minimum_observations
                                else (
                                    "worst_view_iou_below_threshold"
                                    if float(chosen["worst_silhouette_iou"])
                                    < minimum_worst_iou
                                    else "silhouette_iou_below_threshold"
                                )
                            )
                        )
                    )
                ),
                "candidate_label": chosen["label"],
                "mean_silhouette_iou": float(
                    chosen["mean_silhouette_iou"]
                ),
                "observation_count": len(chosen["observations"]),
                "minimum_observations": minimum_observations,
                "minimum_mean_iou": minimum_iou,
                "worst_silhouette_iou": float(
                    chosen["worst_silhouette_iou"]
                ),
                "minimum_worst_view_iou": minimum_worst_iou,
            }
        per_anchor_report[str(anchor_frame)]["selected"] = selected_report

    return updated, {
        "enabled": True,
        "part": part,
        "symmetry": symmetry.as_dict(),
        "candidate_symmetry": candidate_symmetry.as_dict(),
        "resolution": [width, height],
        "table_yaw_search": {
            "enabled": table_yaw_enabled,
            "frames": sorted(table_yaw_frames) if table_yaw_enabled else [],
            "angles_deg": yaw_angles if table_yaw_enabled else [0.0],
        },
        "rigid_core_silhouette": {
            "weight": rigid_core_weight,
            "opening_pixels": rigid_core_opening_pixels,
            "keep_largest": rigid_core_keep_largest,
        },
        "anchors": per_anchor_report,
        "chain": chain_report,
    }
