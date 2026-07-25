from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from common.mask_io import frame_path
from common.normalized_recon import load_recon, scale_intrinsics
from common.pose_refinement import silhouette_metrics
from common.pose_transforms import rigid_from_similarity, similarity_from_rigid
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


def _is_static_transition(
    frame_a: int,
    frame_b: int,
    static_ranges: list[list[int]] | list[tuple[int, int]],
) -> bool:
    return any(int(start) <= frame_a and frame_b <= int(end) for start, end in static_ranges)


def select_candidate_chain(
    anchor_frames: list[int],
    candidate_rows: list[list[dict[str, Any]]],
    *,
    transition_weight: float,
    max_rotation_deg_per_frame: float,
    static_ranges: list[list[int]] | list[tuple[int, int]] | None = None,
    transition_axis: np.ndarray | None = None,
    transition_symmetry: SymmetrySpec | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Select orientation hypotheses with a generic bounded-motion prior."""

    if len(anchor_frames) != len(candidate_rows) or not anchor_frames:
        raise ValueError("anchor_frames and candidate_rows must be non-empty and have equal length")
    static_ranges = static_ranges or []
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
    scores.append(np.asarray([float(row["score"]) for row in candidate_rows[0]], dtype=np.float64))
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

        for current_index, current in enumerate(current_rows):
            penalties: list[float] = []
            for previous_index, previous in enumerate(previous_rows):
                if transition_symmetry is not None:
                    angle_deg = symmetry_rotation_distance_deg(
                        previous["pose"],
                        current["pose"],
                        transition_symmetry,
                    )
                elif transition_axis is None:
                    angle_deg = rotation_distance_deg(previous["pose"], current["pose"])
                else:
                    angle_deg = axis_direction_error_deg(
                        previous["pose"],
                        current["pose"],
                        transition_axis,
                    )
                normalized_rate = angle_deg / frame_gap / max_rotation_deg_per_frame
                penalty = float(transition_weight) * normalized_rate * normalized_rate
                penalties.append(penalty)
                value = scores[-1][previous_index] + float(current["score"]) - penalty
                if value > current_scores[current_index]:
                    current_scores[current_index] = value
                    current_parents[current_index] = previous_index
            transition_matrix.append(penalties)
        transitions.append(transition_matrix)
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
    if symmetry.axis is None:
        raise ValueError(
            f"{part}: appearance refinement requires a symmetry axis"
        )
    candidates = symmetry_candidates(symmetry)
    width, height = [int(value) for value in appearance_cfg.get("resolution", [240, 135])]
    erosion_pixels = int(appearance_cfg.get("texture_erosion_pixels", 3))
    min_mask_pixels = int(appearance_cfg.get("min_mask_pixels", 80))
    edge_cap_pixels = float(appearance_cfg.get("edge_chamfer_cap_pixels", 30.0))
    silhouette_weight = float(appearance_cfg.get("silhouette_weight", 0.25))
    texture_weight = float(appearance_cfg.get("texture_weight", 1.0))

    frames_root = Path(cfg["frames_dir"])
    mask_root = Path(cfg["masks_dir"])
    views = [str(value) for value in appearance_cfg.get("views", cfg["views"])]
    view_indices = {view: cfg["views"].index(view) for view in views}
    part_id = int(cfg["part_ids"][part])
    anchor_frames = sorted(anchors)
    evidence_by_anchor = _resolve_anchor_evidence(appearance_cfg, anchor_frames)

    observations: dict[int, list[dict[str, Any]]] = {}
    for anchor_frame in anchor_frames:
        rows: list[dict[str, Any]] = []
        for timestamp in evidence_by_anchor[anchor_frame]:
            recon = load_recon(cfg, f"{timestamp:06d}", backend=cfg["recon_backend"])
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
                        "observed_edges": _texture_edges(
                            rgb, target, erosion_pixels=erosion_pixels
                        ),
                        "intrinsics": intrinsics,
                        "world_from_camera": recon["extrinsics"][view_index],
                    }
                )
        observations[anchor_frame] = rows

    renderer = SceneRenderer(width=width, height=height)
    all_candidate_rows: list[list[dict[str, Any]]] = []
    per_anchor_report: dict[str, Any] = {}
    try:
        for anchor_frame in anchor_frames:
            rigid_base = rigid_from_similarity(anchors[anchor_frame], origin)
            scored_rows: list[dict[str, Any]] = []
            for candidate in candidates:
                pose = rigid_base @ candidate["local_transform"]
                similarity_transform = similarity_from_rigid(pose, scale, origin)
                observation_scores: list[dict[str, Any]] = []
                for observation in observations[anchor_frame]:
                    rendered_rgb, rendered_depth = renderer.render(
                        [(mesh, similarity_transform)],
                        observation["intrinsics"],
                        observation["world_from_camera"],
                    )
                    predicted = rendered_depth > 0.0
                    silhouette_iou, silhouette_chamfer, silhouette_score = silhouette_metrics(
                        predicted, observation["target"]
                    )
                    rendered_edges = _texture_edges(
                        rendered_rgb, predicted, erosion_pixels=erosion_pixels
                    )
                    edge_chamfer = _symmetric_edge_chamfer(
                        observation["observed_edges"], rendered_edges, edge_cap_pixels
                    )
                    texture_score = (
                        -edge_cap_pixels if edge_chamfer is None else -float(edge_chamfer)
                    ) / edge_cap_pixels
                    score = (
                        silhouette_weight
                        * float(silhouette_score)
                        + texture_weight * texture_score
                    )
                    observation_scores.append(
                        {
                            "timestamp": int(observation["timestamp"]),
                            "view": observation["view"],
                            "score": float(score),
                            "silhouette_iou": float(silhouette_iou),
                            "silhouette_chamfer_px": float(silhouette_chamfer),
                            "texture_edge_chamfer_px": (
                                None if edge_chamfer is None else float(edge_chamfer)
                            ),
                        }
                    )
                aggregate = (
                    float(np.mean([row["score"] for row in observation_scores]))
                    if observation_scores
                    else -1e6
                )
                mean_iou = (
                    float(np.mean([row["silhouette_iou"] for row in observation_scores]))
                    if observation_scores
                    else 0.0
                )
                mean_silhouette_chamfer = (
                    float(
                        np.mean(
                            [
                                row["silhouette_chamfer_px"]
                                for row in observation_scores
                            ]
                        )
                    )
                    if observation_scores
                    else 100.0
                )
                edge_values = [
                    row["texture_edge_chamfer_px"]
                    for row in observation_scores
                    if row["texture_edge_chamfer_px"] is not None
                ]
                scored_rows.append(
                    {
                        "label": candidate["label"],
                        "axis_flipped": bool(candidate["axis_flipped"]),
                        "axis_angle_deg": float(candidate["axis_angle_deg"]),
                        "pose": pose,
                        "score": aggregate,
                        "mean_silhouette_iou": mean_iou,
                        "mean_silhouette_chamfer_px": mean_silhouette_chamfer,
                        "mean_texture_edge_chamfer_px": (
                            float(np.mean(edge_values)) if edge_values else None
                        ),
                        "observations": observation_scores,
                    }
                )
            all_candidate_rows.append(scored_rows)
            per_anchor_report[str(anchor_frame)] = {
                "evidence_frames": evidence_by_anchor[anchor_frame],
                "observation_count": len(observations[anchor_frame]),
                "candidates": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"pose", "observations"}
                    }
                    for row in scored_rows
                ],
            }
    finally:
        renderer.close()

    selected, chain_report = select_candidate_chain(
        anchor_frames,
        all_candidate_rows,
        transition_weight=float(appearance_cfg.get("transition_weight", 0.5)),
        max_rotation_deg_per_frame=float(
            appearance_cfg.get("max_rotation_deg_per_frame", 10.0)
        ),
        static_ranges=_range_pairs(state_cfg.get("static_ranges", [])),
        transition_symmetry=symmetry,
    )
    updated: dict[int, np.ndarray] = {}
    for anchor_index, anchor_frame in enumerate(anchor_frames):
        chosen = all_candidate_rows[anchor_index][selected[anchor_index]]
        updated[anchor_frame] = similarity_from_rigid(chosen["pose"], scale, origin)
        per_anchor_report[str(anchor_frame)]["selected"] = {
            key: value
            for key, value in chosen.items()
            if key not in {"pose", "observations"}
        }

    return updated, {
        "enabled": True,
        "part": part,
        "symmetry": symmetry.as_dict(),
        "resolution": [width, height],
        "anchors": per_anchor_report,
        "chain": chain_report,
    }
