#!/usr/bin/env python
"""Refine moving-part SE(3) poses with multi-view mesh render losses."""
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.exact_render_refinement import ExactMultiViewRenderObjective
from common.mesh_render import SceneRenderer
from common.multiview_quality import (
    cloud_supported_view_quality,
    mask_area_quality,
    temporal_mask_area_references,
)
from common.normalized_recon import load_recon
from common.occlusion_masks import known_occluder_mask
from common.pose_config import validate_pose_config
from common.pose_refinement import (
    interpolate_untrusted_pose_frames,
    sample_canonical,
    smooth_pose_ranges,
)
from common.pose_validation import validate_trajectory
from common.pose_visualization import camera_from_recon
from common.render_loss_refinement import (
    MultiViewRenderObjective,
    RenderObservation,
    clamp_pose_step,
    coarse_reacquire_pose,
    refine_pose_coordinate_search,
    world_pose_delta_vector,
)
from common.symmetry import symmetry_spec_from_state
from common.static_pose_refinement import align_pose_to_support_plane
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    write_trajectory_files,
)
from common.trajectory_constraints import (
    pairwise_alignment_metrics,
    project_coaxial_pose,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_ranges(
    state: dict[str, Any],
    part_config: dict[str, Any],
) -> list[tuple[int, int]]:
    values = part_config.get("ranges", state.get("dynamic_ranges", []))
    return [(int(item[0]), int(item[1])) for item in values]


def frame_in_ranges(frame: int, values: list[list[int]]) -> bool:
    return any(int(start) <= frame <= int(end) for start, end in values)


def other_part_touch_ratio(
    target_mask: np.ndarray,
    other_part_mask: np.ndarray,
    *,
    dilation_pixels: int = 2,
) -> float:
    """Return how much of a modal target boundary touches another part.

    A clean static silhouette can constrain the full mesh without any depth
    interpretation.  When another labelled part touches a large fraction of
    its boundary, however, that view is modal/occluded and must not be used by
    a full-silhouette anchor objective.  Measuring contact at the target
    boundary avoids comparing raw pixel areas across cameras.
    """

    target = np.asarray(target_mask, dtype=bool)
    other = np.asarray(other_part_mask, dtype=bool)
    if target.shape != other.shape:
        raise ValueError("target and other-part masks must have matching shapes")
    boundary = cv2.morphologyEx(
        target.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    boundary_pixels = int(boundary.sum())
    if boundary_pixels == 0:
        return 1.0
    dilation_pixels = max(0, int(dilation_pixels))
    if dilation_pixels:
        size = 2 * dilation_pixels + 1
        other = cv2.dilate(
            other.astype(np.uint8),
            np.ones((size, size), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    return float(np.logical_and(boundary, other).sum() / boundary_pixels)


def canonical_axis_origin(
    settings: dict[str, Any],
    role: str,
    part: str,
    trajectory: dict[str, Any],
) -> np.ndarray:
    metric_key = f"{role}_axis_origin_part_m"
    if metric_key in settings:
        return np.asarray(settings[metric_key], dtype=np.float64)
    raw_key = f"{role}_axis_origin_raw"
    if raw_key not in settings:
        return np.zeros(3, dtype=np.float64)
    return float(trajectory["scales"][part]) * (
        np.asarray(settings[raw_key], dtype=np.float64)
        - np.asarray(trajectory["raw_mesh_origins"][part], dtype=np.float64)
    )


def load_observations(
    cfg: dict[str, Any],
    timestamp: str,
    part: str,
    refinement: dict[str, Any],
    *,
    observation_state: str | None = None,
) -> list[RenderObservation]:
    width, height = [
        int(value) for value in refinement.get("resolution", [160, 90])
    ]
    part_id = int(cfg["part_ids"][part])
    minimum_pixels = int(refinement.get("min_mask_pixels", 30))
    recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
    observations = []
    labels_by_view = {
        view: np.asarray(
            Image.open(Path(cfg["masks_dir"]) / timestamp / f"{view}.png")
        )
        for view in cfg["views"]
        if (Path(cfg["masks_dir"]) / timestamp / f"{view}.png").exists()
    }
    view_quality = dict(cfg.get("view_quality", {}))
    view_quality_parts = view_quality.pop("parts", {})
    view_quality.update(dict(view_quality_parts.get(part, {})))
    reference_cache = getattr(load_observations, "_area_reference_cache", {})
    reference_key = (
        str(Path(cfg["masks_dir"]).resolve()),
        int(part_id),
        tuple(str(view) for view in cfg["views"]),
        tuple(int(cfg["frames"][key]) for key in ("start", "end")),
        repr(sorted(view_quality.items())),
    )
    if reference_key not in reference_cache:
        reference_cache[reference_key] = temporal_mask_area_references(
            Path(cfg["masks_dir"]),
            part_id,
            [str(view) for view in cfg["views"]],
            (int(cfg["frames"]["start"]), int(cfg["frames"]["end"])),
            view_quality,
        )
        load_observations._area_reference_cache = reference_cache
    quality = mask_area_quality(
        labels_by_view,
        part_id,
        minimum_pixels=int(
            refinement.get("minimum_full_mask_pixels", 1)
        ),
        maximum_area_ratio=float(
            refinement.get("maximum_mask_area_ratio", 4.0)
        ),
        minimum_area_ratio=float(
            refinement.get("minimum_mask_area_ratio", 0.0)
        ),
        reference_areas=reference_cache[reference_key],
        presence_minimum_pixels=int(
            refinement.get("presence_minimum_full_mask_pixels", minimum_pixels)
        ),
    )
    trusted_views = set(cfg["views"])
    depth_trusted_views = set(cfg["views"])
    cloud_gate = None
    mask_only_fallback = False
    if refinement.get("use_cloud_supported_view_gate", False):
        summary_path = Path(
            refinement.get(
                "quality_cloud_summary",
                Path(cfg["point_cloud_root"]) / "quality_cloud_summary.json",
            )
        )
        cache = getattr(load_observations, "_quality_cloud_cache", {})
        cache_key = str(summary_path.resolve())
        if cache_key not in cache:
            cache[cache_key] = load_json(summary_path).get("frames", {})
            load_observations._quality_cloud_cache = cache
        frame_cloud = cache[cache_key].get(timestamp, {}).get(part, {})
        cloud_gate = cloud_supported_view_quality(
            frame_cloud,
            list(cfg["views"]),
            minimum_supported_points=int(
                refinement.get("minimum_supported_points_per_view", 30)
            ),
            minimum_support_fraction=float(
                refinement.get("minimum_view_support_fraction", 0.25)
            ),
        )
        cloud_trusted_views = {
            view
            for view, row in cloud_gate["views"].items()
            if row["valid"]
        }
        depth_trusted_views = cloud_trusted_views
        mask_only_fallback = bool(
            observation_state == "visible_cloud_unreliable"
            and refinement.get(
                "fallback_to_mask_views_when_cloud_unreliable", False
            )
            and len(cloud_trusted_views)
            < int(refinement.get("mask_fallback_minimum_cloud_views", 3))
        )
        # In mask-primary mode point-cloud quality controls only the optional
        # metric depth term.  Accurate masks remain valid pose observations.
        # The legacy behaviour is retained for older configs.
        if (
            not refinement.get("mask_primary", False)
            and not mask_only_fallback
        ):
            trusted_views = cloud_trusted_views
    for view_index, view in enumerate(cfg["views"]):
        if not quality["views"].get(view, {}).get("valid", False):
            continue
        if view not in trusted_views:
            continue
        labels = labels_by_view[view]
        target = cv2.resize(
            (labels == part_id).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if int(target.sum()) < minimum_pixels:
            continue
        other_parts = cv2.resize(
            known_occluder_mask(labels, part_id).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        maximum_touch_ratio = refinement.get(
            "maximum_other_part_touch_ratio"
        )
        if maximum_touch_ratio is not None and other_part_touch_ratio(
            target,
            other_parts,
            dilation_pixels=int(
                refinement.get("other_part_touch_dilation_pixels", 2)
            ),
        ) > float(maximum_touch_ratio):
            continue
        intrinsics, extrinsics = camera_from_recon(
            recon, view_index, (height, width)
        )
        observed_depth = cv2.resize(
            recon["depth"][view_index].astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        depth_loss_enabled = bool(view in depth_trusted_views)
        if mask_only_fallback and refinement.get(
            "disable_depth_for_mask_only_fallback", True
        ):
            depth_loss_enabled = False
        known_occluder = None
        if refinement.get("known_part_occlusion_aware", True):
            known_occluder = cv2.resize(
                known_occluder_mask(
                    labels,
                    part_id,
                    refinement.get("known_occluder_labels"),
                ).astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        observations.append(
            RenderObservation(
                view=view,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                target_mask=target,
                observed_depth=observed_depth,
                depth_loss_enabled=depth_loss_enabled,
                known_occluder_mask=known_occluder,
            )
        )
    return observations


def compact_metric(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "loss",
            "worst_view_loss",
            "mean_iou",
            "worst_view_iou",
            "mean_contour_chamfer_px",
            "mean_target_coverage",
        )
    }


def exact_holdout_gate(
    baseline: dict[str, Any] | None,
    selected: dict[str, Any] | None,
    settings: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Apply the independent-view gate again after exact triangle polishing."""

    if baseline is None or selected is None:
        return True, {
            "evaluated": False,
            "loss_degradation": 0.0,
            "failures": [],
        }
    degradation = float(selected["loss"] - baseline["loss"])
    maximum = float(
        settings.get(
            "exact_maximum_holdout_degradation",
            settings.get("maximum_holdout_degradation", 0.015),
        )
    )
    minimum_iou = float(
        settings.get(
            "exact_minimum_holdout_iou",
            settings.get("minimum_holdout_iou", 0.0),
        )
    )
    failures = []
    if degradation > maximum + 1e-12:
        failures.append("holdout_loss_degradation")
    if float(selected.get("mean_iou", 0.0)) < minimum_iou:
        failures.append("holdout_iou_below_minimum")
    return not failures, {
        "evaluated": True,
        "loss_degradation": degradation,
        "maximum_loss_degradation": maximum,
        "selected_mean_iou": float(selected.get("mean_iou", 0.0)),
        "minimum_mean_iou": minimum_iou,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-trajectory", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--parts", nargs="+", default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="debugging limit across all parts; omitted runs every configured frame",
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help=(
            "optimize only these independent diagnostic frames; temporal "
            "post-smoothing is disabled"
        ),
    )
    parser.add_argument(
        "--static-anchor-search",
        action="store_true",
        help=(
            "diagnose/refine explicitly requested clean static frames with "
            "full-mask exact-triangle SE(3); disables depth-based occlusion "
            "exemptions and permits configured tracking anchors"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    trajectory_path = Path(args.trajectory).resolve()
    output_path = Path(args.output_trajectory).resolve()
    report_path = (
        Path(args.report).resolve()
        if args.report
        else output_path.parents[1] / "diagnostics/render_loss_refinement.json"
    )
    cfg = validate_pose_config(load_json(config_path), check_paths=True)
    refinement = dict(cfg.get("render_loss_refinement", {}))
    if args.static_anchor_search:
        if args.frames is None:
            raise ValueError("--static-anchor-search requires --frames")
        refinement.update({
            "resolution": [320, 180],
            "occlusion_aware": False,
            # Known rigid-part masks are valid modal occluders.  DA3 depth
            # occlusion is disabled because it made the old static score look
            # artificially good by hiding misprojected mesh pixels.
            "known_part_occlusion_aware": True,
            "maximum_other_part_touch_ratio": 0.75,
            "other_part_touch_dilation_pixels": 2,
            "mask_primary": True,
            "use_cloud_supported_view_gate": False,
            "exact_triangle_refinement": True,
            "translation_steps_m": [0.012, 0.006, 0.003, 0.0015],
            "rotation_steps_deg": [8.0, 4.0, 2.0, 1.0],
            "maximum_translation_delta_m": 0.04,
            "maximum_rotation_delta_deg": 30.0,
            "exact_translation_steps_m": [0.008, 0.004, 0.002, 0.001],
            "exact_rotation_steps_deg": [6.0, 3.0, 1.5, 0.75],
            "exact_maximum_translation_delta_m": 0.03,
            "exact_maximum_rotation_delta_deg": 24.0,
            "static_table_alignment": True,
            "static_table_contact_quantile": 0.005,
            "static_table_maximum_shift_m": 0.02,
            "minimum_improvement": 0.0,
            "prior_weight": 0.005,
            "temporal_weight": 0.0,
        })
        weights = dict(refinement.get("weights", {}))
        weights["depth"] = 0.0
        refinement["weights"] = weights
        cfg["render_loss_refinement"] = refinement
    if not refinement.get("enabled", False):
        raise ValueError("render_loss_refinement.enabled must be true")
    baseline = load_json(trajectory_path)
    trajectory = copy.deepcopy(baseline)
    requested_parts = list(args.parts or refinement.get("parts", {}).keys())
    unknown = sorted(set(requested_parts).difference(trajectory["parts"]))
    if unknown:
        raise ValueError(f"unknown refinement parts: {unknown}")

    optimize_views = [
        str(value)
        for value in refinement.get("optimize_views", cfg["views"])
    ]
    holdout_views = [
        str(value)
        for value in refinement.get("holdout_views", [])
    ]
    overlap = sorted(set(optimize_views).intersection(holdout_views))
    if overlap:
        raise ValueError(f"optimize_views and holdout_views overlap: {overlap}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "multi_view_sampled_mesh_render_loss_coordinate_search",
        "config": str(config_path),
        "trajectory_input": str(trajectory_path),
        "trajectory_input_sha256": sha256_file(trajectory_path),
        "trajectory_output": str(output_path),
        "resolution": refinement.get("resolution", [160, 90]),
        "optimize_views": optimize_views,
        "holdout_views": holdout_views,
        "parts": {},
    }
    timeline_fps = float(cfg.get("frames", {}).get("fps", 0.0) or 0.0)
    correction_fps = float(refinement.get("full_view_correction_fps", 0.0) or 0.0)
    frame_stride = int(refinement.get("full_view_stride", 0) or 0)
    if frame_stride <= 0:
        frame_stride = (
            max(1, int(round(timeline_fps / correction_fps)))
            if timeline_fps > 0.0 and correction_fps > 0.0
            else 1
        )
    report["timeline_fps"] = timeline_fps or None
    report["full_view_correction_fps"] = correction_fps or None
    report["full_view_stride"] = frame_stride
    processed_frames = 0
    for part_index, part in enumerate(requested_parts):
        part_config = dict(refinement.get("parts", {}).get(part, {}))
        if not part_config.get("enabled", True):
            continue
        # Part-specific settings are allowed to override observation quality
        # gates and objective weights as well as coordinate-search steps.  In
        # particular, a large, cleanly segmented body can use all mask views
        # even when its reconstructed depths disagree, while a tiny occluded
        # insert still needs the cloud-supported camera gate.
        part_refinement = dict(refinement)
        part_refinement.update(part_config)
        state = cfg["states"][part]
        configured_anchor_frames = {
            int(value) for value in state.get("tracking_anchor_frames", [])
        }
        protected_anchor_frames = (
            configured_anchor_frames
            if bool(part_refinement.get("protect_tracking_anchors", True))
            and not args.static_anchor_search
            else set()
        )
        ranges = (
            [(frame, frame) for frame in sorted(set(args.frames))]
            if args.frames is not None
            else configured_ranges(state, part_config)
        )
        if not ranges:
            continue
        raw_mesh = trimesh.load(
            Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh"
        )
        exact_enabled = bool(
            part_refinement.get("exact_triangle_refinement", False)
        )
        exact_renderer = None
        if exact_enabled:
            exact_width, exact_height = [
                int(value)
                for value in part_refinement.get(
                    "resolution", [320, 180]
                )
            ]
            exact_renderer = SceneRenderer(
                exact_width,
                exact_height,
                cache_mesh_resources=True,
            )
        points = sample_canonical(
            raw_mesh,
            float(baseline["scales"][part]),
            np.asarray(baseline["raw_mesh_origins"][part], dtype=np.float64),
            count=int(part_refinement.get("surface_points", 30000)),
            seed=int(part_refinement.get("seed", 1701)) + part_index,
        )
        metric_vertices = (
            np.asarray(raw_mesh.vertices, dtype=np.float64)
            - np.asarray(
                baseline["raw_mesh_origins"][part], dtype=np.float64
            )
        ) * float(baseline["scales"][part])
        symmetry = symmetry_spec_from_state(state)
        symmetry_axis = (
            symmetry.axis
            if symmetry.equivalence == "continuous_axial"
            else None
        )
        part_report: dict[str, Any] = {
            "ranges": [list(item) for item in ranges],
            "surface_points": int(len(points)),
            "frames": {},
        }
        accepted_count = 0
        part_optimize_views = [
            str(value)
            for value in part_config.get("optimize_views", optimize_views)
        ]
        part_holdout_views = [
            str(value)
            for value in part_config.get("holdout_views", holdout_views)
        ]
        minimum_optimize_views = int(
            part_config.get(
                "minimum_optimize_views",
                refinement.get("minimum_optimize_views", 3),
            )
        )
        minimum_holdout_views = int(
            part_config.get(
                "minimum_holdout_views",
                refinement.get("minimum_holdout_views", 0),
            )
        )
        temporal_delta = np.zeros(6, dtype=np.float64)
        last_accepted_pose = None
        last_accepted_frame = None
        previous_coaxial_relative = None
        previous_coaxial_frame = None
        coaxial_boundaries = {
            int(value)
            for interval in part_refinement.get(
                "coaxial_constraint", {}
            ).get("ranges", [])
            for value in interval
        }
        for range_start, range_end in ranges:
            temporal_delta[:] = 0.0
            for frame in range(range_start, range_end + 1):
                if args.frames is not None and frame not in set(args.frames):
                    continue
                if frame in protected_anchor_frames:
                    part_report["frames"][f"{frame:06d}"] = {
                        "accepted": False,
                        "skip_reason": "protected_stable_anchor",
                    }
                    continue
                if (
                    args.frames is None
                    and frame
                    not in (
                        {range_start, range_end}
                        | coaxial_boundaries
                        | {
                            value
                            for value in (
                                args.start_frame,
                                args.end_frame,
                            )
                            if value is not None
                        }
                    )
                    and (frame - range_start) % frame_stride != 0
                ):
                    continue
                if args.start_frame is not None and frame < args.start_frame:
                    continue
                if args.end_frame is not None and frame > args.end_frame:
                    continue
                if (
                    args.max_frames is not None
                    and processed_frames >= args.max_frames
                ):
                    break
                timestamp = f"{frame:06d}"
                if timestamp not in baseline["frames"]:
                    continue
                record = baseline["frames"][timestamp]["parts"][part]
                if (
                    record.get("pose_valid") is False
                    or int(record.get("observing_views", 0)) <= 0
                ):
                    part_report["frames"][timestamp] = {
                        "accepted": False,
                        "skip_reason": "unobservable",
                    }
                    continue
                observations = load_observations(
                    cfg,
                    timestamp,
                    part,
                    part_refinement,
                    observation_state=str(
                        record.get("observation_state", "")
                    ),
                )
                available = {item.view for item in observations}
                effective_optimize = [
                    view for view in part_optimize_views if view in available
                ]
                effective_holdout = [
                    view for view in part_holdout_views if view in available
                ]
                if (
                    args.static_anchor_search
                    and not effective_holdout
                    and len(effective_optimize) > minimum_optimize_views
                ):
                    # Preserve an independent camera gate even when the
                    # configured holdout view is rejected as inter-part
                    # occluded.  Selection is deterministic and never leaks
                    # this view back into the optimize set.
                    replacement_holdout = effective_optimize[-1]
                    effective_optimize = effective_optimize[:-1]
                    effective_holdout = [replacement_holdout]
                if len(effective_optimize) < minimum_optimize_views:
                    part_report["frames"][timestamp] = {
                        "accepted": False,
                        "skip_reason": "insufficient_views",
                        "available_views": sorted(available),
                    }
                    continue
                if len(effective_holdout) < minimum_holdout_views:
                    part_report["frames"][timestamp] = {
                        "accepted": False,
                        "skip_reason": "insufficient_holdout_views",
                        "available_views": sorted(available),
                        "effective_holdout_views": effective_holdout,
                    }
                    continue
                objective = MultiViewRenderObjective(
                    points, observations, part_refinement
                )
                tracking_initial = np.asarray(
                    record["T_world_from_part"], dtype=np.float64
                )
                initial = tracking_initial.copy()
                reacquire_report = None
                reacquired = False
                initial_visual = objective.evaluate(
                    initial, effective_optimize
                )
                if (
                    part_refinement.get("global_reacquire_enabled", False)
                    and float(initial_visual.get("mean_iou", 0.0))
                    < float(
                        part_refinement.get(
                            "global_reacquire_iou_threshold", 0.25
                        )
                    )
                ):
                    translation_prior_max_gap = int(
                        part_refinement.get(
                            "global_reacquire_translation_prior_max_gap_frames", 2
                        )
                    )
                    translation_reference_pose = (
                        last_accepted_pose
                        if last_accepted_frame is not None
                        and frame - last_accepted_frame <= translation_prior_max_gap
                        else None
                    )
                    candidate, reacquire_report = coarse_reacquire_pose(
                        objective,
                        initial,
                        views=effective_optimize,
                        translation_radii_m=[
                            float(value)
                            for value in part_refinement.get(
                                "global_reacquire_translation_radii_m",
                                [0.08, 0.04],
                            )
                        ],
                        rotation_angles_deg=[
                            float(value)
                            for value in part_refinement.get(
                                "global_reacquire_rotation_angles_deg",
                                [60.0, 30.0],
                            )
                        ],
                        symmetry_axis_part=symmetry_axis,
                        optimize_rotation=bool(
                            part_config.get("optimize_rotation", True)
                        ),
                        alternating_passes=int(
                            part_refinement.get(
                                "global_reacquire_alternating_passes", 1
                            )
                        ),
                        rotation_reference_pose=last_accepted_pose,
                        rotation_prior_weight=float(
                            part_refinement.get(
                                "global_reacquire_rotation_prior_weight", 0.0
                            )
                        ),
                        rotation_prior_scale_deg=float(
                            part_refinement.get(
                                "global_reacquire_rotation_prior_scale_deg", 90.0
                            )
                        ),
                        translation_reference_pose=translation_reference_pose,
                        translation_prior_weight=float(
                            part_refinement.get(
                                "global_reacquire_translation_prior_weight", 0.0
                            )
                        ),
                        translation_prior_scale_m=float(
                            part_refinement.get(
                                "global_reacquire_translation_prior_scale_m", 0.04
                            )
                        ),
                    )
                    reacquired = bool(
                        float(reacquire_report["loss_improvement"])
                        >= float(
                            part_refinement.get(
                                "global_reacquire_minimum_improvement", 0.02
                            )
                        )
                    )
                    if reacquired:
                        initial = candidate
                validation_config = state.get("validation", {})
                previous_pose = (
                    np.asarray(
                        trajectory["frames"][f"{frame - 1:06d}"]["parts"][
                            part
                        ]["T_world_from_part"],
                        dtype=np.float64,
                    )
                    if f"{frame - 1:06d}" in trajectory["frames"]
                    else None
                )
                if reacquired and part_refinement.get(
                    "global_reacquire_relax_temporal_boundary", True
                ):
                    previous_pose = None
                next_pose = (
                    np.asarray(
                        baseline["frames"][f"{frame + 1:06d}"]["parts"][
                            part
                        ]["T_world_from_part"],
                        dtype=np.float64,
                    )
                    if frame == range_end
                    and f"{frame + 1:06d}" in baseline["frames"]
                    else None
                )
                selected, frame_report = refine_pose_coordinate_search(
                    objective,
                    initial,
                    optimize_views=effective_optimize,
                    holdout_views=effective_holdout,
                    translation_steps_m=[
                        float(value)
                        for value in part_config.get(
                            "translation_steps_m",
                            refinement.get(
                                "translation_steps_m",
                                [0.006, 0.003, 0.0015],
                            ),
                        )
                    ],
                    rotation_steps_deg=[
                        float(value)
                        for value in part_config.get(
                            "rotation_steps_deg",
                            refinement.get(
                                "rotation_steps_deg", [2.0, 1.0, 0.5]
                            ),
                        )
                    ],
                    symmetry_axis_part=symmetry_axis,
                    optimize_rotation=bool(
                        part_config.get("optimize_rotation", True)
                    ),
                    maximum_translation_delta_m=float(
                        part_config.get(
                            "maximum_translation_delta_m",
                            refinement.get(
                                "maximum_translation_delta_m", 0.015
                            ),
                        )
                    ),
                    maximum_rotation_delta_deg=float(
                        part_config.get(
                            "maximum_rotation_delta_deg",
                            refinement.get(
                                "maximum_rotation_delta_deg", 4.0
                            ),
                        )
                    ),
                    minimum_improvement=(
                        0.0
                        if reacquired
                        else float(
                            part_refinement.get(
                                "minimum_improvement", 0.002
                            )
                        )
                    ),
                    maximum_holdout_degradation=float(
                        part_refinement.get(
                            "maximum_holdout_degradation", 0.015
                        )
                    ),
                    minimum_refined_iou=float(
                        part_refinement.get("minimum_refined_iou", 0.0)
                    ),
                    minimum_refined_target_coverage=float(
                        part_refinement.get(
                            "minimum_refined_target_coverage", 0.0
                        )
                    ),
                    minimum_holdout_iou=float(
                        part_refinement.get("minimum_holdout_iou", 0.0)
                    ),
                    minimum_per_view_iou=float(
                        part_refinement.get("minimum_per_view_iou", 0.0)
                    ),
                    maximum_worst_view_loss=float(
                        part_refinement.get(
                            "maximum_worst_view_loss", 1.0e9
                        )
                    ),
                    previous_pose=previous_pose,
                    next_pose=next_pose,
                    maximum_step_translation_m=float(
                        validation_config.get(
                            "max_translation_step_m", float("inf")
                        )
                    ),
                    maximum_step_rotation_deg=float(
                        validation_config.get(
                            "max_rotation_step_deg", float("inf")
                        )
                    ),
                    prior_weight=float(
                        part_config.get(
                            "prior_weight", refinement.get("prior_weight", 0.03)
                        )
                    ),
                    temporal_delta_reference=temporal_delta,
                    temporal_weight=float(
                        part_config.get(
                            "temporal_weight",
                            refinement.get("temporal_weight", 0.02),
                        )
                    ),
                )
                accepted = bool(frame_report["accepted"])
                exact_report = None
                # Coaxial refinement is optional and only configured for a
                # subset of assembly tasks.  The accepted-pose bookkeeping
                # below is shared by exact and point-render paths, so its gate
                # must have an explicit false value for ordinary parts.
                coaxial_active = False
                if exact_renderer is not None:
                    exact_objective = ExactMultiViewRenderObjective(
                        raw_mesh,
                        float(baseline["scales"][part]),
                        np.asarray(
                            baseline["raw_mesh_origins"][part],
                            dtype=np.float64,
                        ),
                        observations,
                        part_refinement,
                        exact_renderer,
                    )
                    exact_baseline = exact_objective.evaluate(
                        tracking_initial, effective_optimize
                    )
                    exact_baseline_holdout = (
                        exact_objective.evaluate(
                            tracking_initial, effective_holdout
                        )
                        if effective_holdout
                        else None
                    )
                    point_proposal = np.asarray(
                        frame_report["best_pose_before_gates"],
                        dtype=np.float64,
                    )
                    coaxial = dict(
                        part_refinement.get("coaxial_constraint", {})
                    )
                    coaxial_active = bool(
                        coaxial.get("enabled", False)
                        and frame_in_ranges(
                            frame, coaxial.get("ranges", [])
                        )
                    )
                    coaxial_report = None
                    reference_world = None
                    reference_origin = None
                    moving_origin = None
                    if coaxial_active:
                        reference_part = str(coaxial["reference_part"])
                        reference_world = np.asarray(
                            trajectory["frames"][timestamp]["parts"]
                            [reference_part]["T_world_from_part"],
                            dtype=np.float64,
                        )
                        reference_origin = canonical_axis_origin(
                            coaxial,
                            "reference",
                            reference_part,
                            baseline,
                        )
                        moving_origin = canonical_axis_origin(
                            coaxial, "moving", part, baseline
                        )
                        relative_proposal = (
                            np.linalg.inv(reference_world) @ point_proposal
                        )
                        twist_angles = [
                            float(value)
                            for value in coaxial.get(
                                "twist_offsets_deg",
                                list(range(0, 360, 30)),
                            )
                        ]
                        axial_offsets = [
                            float(value)
                            for value in coaxial.get(
                                "axial_offsets_m",
                                [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03],
                            )
                        ]
                        candidates = []
                        best_world = None
                        best_data = None
                        best_score = None
                        for twist_deg in twist_angles:
                            for axial_delta in axial_offsets:
                                candidate_relative = project_coaxial_pose(
                                    relative_proposal,
                                    reference_axis=coaxial[
                                        "reference_axis_part"
                                    ],
                                    moving_axis=coaxial["moving_axis_part"],
                                    reference_axis_origin_m=reference_origin,
                                    moving_axis_origin_m=moving_origin,
                                    allow_axis_flip=bool(
                                        coaxial.get("allow_axis_flip", False)
                                    ),
                                    target_axis_offset_m=float(
                                        coaxial.get(
                                            "target_axis_offset_m", 0.0
                                        )
                                    ),
                                    twist_rad=np.deg2rad(twist_deg),
                                    axial_delta_m=axial_delta,
                                )
                                candidate_world = (
                                    reference_world @ candidate_relative
                                )
                                candidate_data = objective.evaluate(
                                    candidate_world, effective_optimize
                                )
                                temporal_rotation_deg = 0.0
                                temporal_translation_m = 0.0
                                temporal_penalty = 0.0
                                temporal_violation = 0.0
                                if previous_coaxial_relative is not None:
                                    candidate_delta = world_pose_delta_vector(
                                        previous_coaxial_relative,
                                        candidate_relative,
                                    )
                                    temporal_translation_m = float(
                                        np.linalg.norm(candidate_delta[:3])
                                    )
                                    temporal_rotation_deg = float(
                                        np.degrees(
                                            np.linalg.norm(candidate_delta[3:])
                                        )
                                    )
                                    frame_gap = max(
                                        frame
                                        - int(previous_coaxial_frame),
                                        1,
                                    )
                                    rotation_limit = frame_gap * float(
                                        coaxial.get(
                                            "maximum_rotation_step_deg_per_frame",
                                            20.0,
                                        )
                                    )
                                    translation_limit = frame_gap * float(
                                        coaxial.get(
                                            "maximum_translation_step_m_per_frame",
                                            0.02,
                                        )
                                    )
                                    normalized_rotation = (
                                        temporal_rotation_deg
                                        / max(rotation_limit, 1e-6)
                                    )
                                    normalized_translation = (
                                        temporal_translation_m
                                        / max(translation_limit, 1e-6)
                                    )
                                    temporal_penalty = float(
                                        coaxial.get("temporal_weight", 0.12)
                                    ) * (
                                        normalized_rotation**2
                                        + normalized_translation**2
                                    )
                                    temporal_violation = max(
                                        normalized_rotation - 1.0,
                                        normalized_translation - 1.0,
                                        0.0,
                                    )
                                selection_score = (
                                    float(candidate_data["loss"])
                                    + temporal_penalty
                                    + 100.0 * temporal_violation**2
                                )
                                candidates.append(
                                    {
                                        "twist_offset_deg": twist_deg,
                                        "axial_offset_m": axial_delta,
                                        "loss": float(candidate_data["loss"]),
                                        "mean_iou": float(
                                            candidate_data["mean_iou"]
                                        ),
                                        "selection_score": selection_score,
                                        "temporal_rotation_deg": (
                                            temporal_rotation_deg
                                        ),
                                        "temporal_translation_m": (
                                            temporal_translation_m
                                        ),
                                        "temporal_violation": temporal_violation,
                                    }
                                )
                                if (
                                    best_score is None
                                    or selection_score < best_score
                                ):
                                    best_world = candidate_world
                                    best_data = candidate_data
                                    best_score = selection_score
                        if best_world is None or best_data is None:
                            raise RuntimeError(
                                "coaxial candidate search produced no pose"
                            )
                        point_proposal = best_world
                        seed_relative = (
                            np.linalg.inv(reference_world) @ point_proposal
                        )
                        seed_alignment = pairwise_alignment_metrics(
                            seed_relative,
                            reference_axis=coaxial[
                                "reference_axis_part"
                            ],
                            moving_axis=coaxial["moving_axis_part"],
                            allow_axis_flip=bool(
                                coaxial.get("allow_axis_flip", False)
                            ),
                            reference_axis_origin_m=reference_origin,
                            moving_axis_origin_m=moving_origin,
                        )
                        best_candidate = min(
                            candidates,
                            key=lambda item: item["selection_score"],
                        )
                        coaxial_report = {
                            "hard": bool(coaxial.get("hard", True)),
                            "reference_part": reference_part,
                            "candidate_evaluations": len(candidates),
                            "best_twist_offset_deg": best_candidate[
                                "twist_offset_deg"
                            ],
                            "best_axial_offset_m": best_candidate[
                                "axial_offset_m"
                            ],
                            "temporal_rotation_deg": best_candidate[
                                "temporal_rotation_deg"
                            ],
                            "temporal_translation_m": best_candidate[
                                "temporal_translation_m"
                            ],
                            "temporal_violation": best_candidate[
                                "temporal_violation"
                            ],
                            "seed_alignment": seed_alignment,
                        }
                    proposal_data = exact_objective.evaluate(
                        point_proposal, effective_optimize
                    )
                    exact_seed = (
                        point_proposal
                        if coaxial_active
                        and bool(coaxial.get("hard", True))
                        else point_proposal
                        if float(proposal_data["loss"])
                        < float(exact_baseline["loss"])
                        else tracking_initial
                    )
                    exact_selected, exact_search = (
                        refine_pose_coordinate_search(
                            exact_objective,
                            exact_seed,
                            optimize_views=effective_optimize,
                            holdout_views=[],
                            translation_steps_m=[
                                float(value)
                                for value in (
                                    coaxial.get(
                                        "exact_translation_steps_m",
                                        part_refinement.get(
                                            "exact_translation_steps_m",
                                            [0.004, 0.002],
                                        ),
                                    )
                                    if coaxial_active
                                    else part_refinement.get(
                                        "exact_translation_steps_m",
                                        [0.004, 0.002],
                                    )
                                )
                            ],
                            rotation_steps_deg=[
                                float(value)
                                for value in (
                                    coaxial.get(
                                        "exact_rotation_steps_deg",
                                        part_refinement.get(
                                            "exact_rotation_steps_deg",
                                            [2.0, 1.0],
                                        ),
                                    )
                                    if coaxial_active
                                    else part_refinement.get(
                                        "exact_rotation_steps_deg",
                                        [2.0, 1.0],
                                    )
                                )
                            ],
                            symmetry_axis_part=symmetry_axis,
                            optimize_rotation=bool(
                                part_config.get("optimize_rotation", True)
                            ),
                            maximum_translation_delta_m=float(
                                coaxial.get(
                                    "exact_maximum_translation_delta_m",
                                    part_refinement.get(
                                        "exact_maximum_translation_delta_m",
                                        0.012,
                                    ),
                                )
                                if coaxial_active
                                else part_refinement.get(
                                    "exact_maximum_translation_delta_m", 0.012
                                )
                            ),
                            maximum_rotation_delta_deg=float(
                                coaxial.get(
                                    "exact_maximum_rotation_delta_deg",
                                    part_refinement.get(
                                        "exact_maximum_rotation_delta_deg",
                                        6.0,
                                    ),
                                )
                                if coaxial_active
                                else part_refinement.get(
                                    "exact_maximum_rotation_delta_deg", 6.0
                                )
                            ),
                            minimum_improvement=0.0,
                            maximum_holdout_degradation=0.0,
                            minimum_refined_iou=float(
                                part_refinement.get(
                                    "exact_minimum_refined_iou", 0.0
                                )
                            ),
                            minimum_refined_target_coverage=float(
                                part_refinement.get(
                                    "exact_minimum_target_coverage", 0.0
                                )
                            ),
                            minimum_per_view_iou=float(
                                part_refinement.get(
                                    "exact_minimum_per_view_iou", 0.0
                                )
                            ),
                            prior_weight=0.0,
                            temporal_weight=0.0,
                        )
                    )
                    table_support_report = None
                    if (
                        args.static_anchor_search
                        and part_refinement.get(
                            "static_table_alignment", True
                        )
                    ):
                        support_plane = cfg.get("support_plane", {})
                        if not support_plane.get("accepted", False):
                            raise ValueError(
                                "static anchor search requires an accepted "
                                "support_plane when table alignment is enabled"
                            )
                        exact_selected, table_support_report = (
                            align_pose_to_support_plane(
                                exact_selected,
                                metric_vertices,
                                support_plane,
                                contact_quantile=float(
                                    part_refinement.get(
                                        "static_table_contact_quantile", 0.005
                                    )
                                ),
                                maximum_shift_m=float(
                                    part_refinement.get(
                                        "static_table_maximum_shift_m", 0.02
                                    )
                                ),
                            )
                        )
                    exact_final = exact_objective.evaluate(
                        exact_selected, effective_optimize
                    )
                    exact_final_holdout = (
                        exact_objective.evaluate(
                            exact_selected, effective_holdout
                        )
                        if effective_holdout
                        else None
                    )
                    exact_holdout_accepted, exact_holdout_report = (
                        exact_holdout_gate(
                            exact_baseline_holdout,
                            exact_final_holdout,
                            part_refinement,
                        )
                    )
                    exact_improvement = float(
                        exact_baseline["loss"] - exact_final["loss"]
                    )
                    if coaxial_active and bool(coaxial.get("hard", True)):
                        selected_relative = (
                            np.linalg.inv(reference_world) @ exact_selected
                        )
                        selected_alignment = pairwise_alignment_metrics(
                            selected_relative,
                            reference_axis=coaxial[
                                "reference_axis_part"
                            ],
                            moving_axis=coaxial["moving_axis_part"],
                            allow_axis_flip=bool(
                                coaxial.get("allow_axis_flip", False)
                            ),
                            reference_axis_origin_m=reference_origin,
                            moving_axis_origin_m=moving_origin,
                        )
                        coaxial_report["selected_alignment"] = (
                            selected_alignment
                        )
                        accepted = bool(
                            exact_search["accepted"]
                            and exact_holdout_accepted
                            and float(exact_final["loss"])
                            <= float(exact_baseline["loss"])
                            + float(
                                coaxial.get(
                                    "maximum_visual_loss_degradation", 0.10
                                )
                            )
                            and float(selected_alignment["axis_angle_deg"])
                            <= float(
                                coaxial.get("maximum_axis_angle_deg", 3.0)
                            )
                            and float(selected_alignment["axis_offset_m"])
                            <= float(
                                coaxial.get("maximum_axis_offset_m", 0.006)
                            )
                        )
                    else:
                        accepted = bool(
                            exact_search["accepted"]
                            and (
                                table_support_report is None
                                or table_support_report["accepted"]
                            )
                            and exact_holdout_accepted
                            and exact_improvement
                            >= float(
                                part_refinement.get(
                                    "exact_minimum_improvement", 0.001
                                )
                            )
                        )
                    selected = (
                        exact_selected if accepted else tracking_initial
                    )
                    total_delta = world_pose_delta_vector(
                        tracking_initial, selected
                    )
                    frame_report["translation_delta_m"] = (
                        total_delta[:3].tolist()
                    )
                    frame_report["translation_delta_norm_m"] = float(
                        np.linalg.norm(total_delta[:3])
                    )
                    frame_report["rotation_delta_deg"] = float(
                        np.degrees(np.linalg.norm(total_delta[3:]))
                    )
                    exact_report = {
                        "accepted": accepted,
                        "baseline": compact_metric(exact_baseline),
                        "baseline_holdout": (
                            compact_metric(exact_baseline_holdout)
                            if exact_baseline_holdout is not None
                            else None
                        ),
                        "point_proposal": compact_metric(proposal_data),
                        "selected": compact_metric(exact_final),
                        "selected_holdout": (
                            compact_metric(exact_final_holdout)
                            if exact_final_holdout is not None
                            else None
                        ),
                        "holdout_gate": exact_holdout_report,
                        "loss_improvement": exact_improvement,
                        "search_evaluations": exact_search["evaluations"],
                        "coaxial_constraint": coaxial_report,
                        "table_support": table_support_report,
                    }
                continuity_fallback = False
                if accepted:
                    target_record = trajectory["frames"][timestamp]["parts"][
                        part
                    ]
                    target_record["T_world_from_part"] = selected.tolist()
                    target_record["source"] = (
                        str(target_record.get("source", "pose"))
                        + ("+global_reacquire" if reacquired else "")
                        + "+render_loss"
                        + ("+exact_triangle" if exact_report else "")
                    )
                    temporal_delta = world_pose_delta_vector(
                        tracking_initial, selected
                    )
                    last_accepted_pose = selected.copy()
                    last_accepted_frame = frame
                    if coaxial_active:
                        previous_coaxial_relative = (
                            np.linalg.inv(reference_world) @ selected
                        )
                        previous_coaxial_frame = frame
                    accepted_count += 1
                else:
                    if previous_pose is not None:
                        guarded = clamp_pose_step(
                            previous_pose,
                            initial,
                            maximum_translation_m=float(
                                validation_config.get(
                                    "max_translation_step_m", float("inf")
                                )
                            ),
                            maximum_rotation_deg=float(
                                validation_config.get(
                                    "max_rotation_step_deg", float("inf")
                                )
                            ),
                        )
                        guard_delta = world_pose_delta_vector(
                            initial, guarded
                        )
                        continuity_fallback = bool(
                            np.linalg.norm(guard_delta) > 1e-9
                        )
                        if continuity_fallback:
                            target_record = trajectory["frames"][timestamp][
                                "parts"
                            ][part]
                            target_record["T_world_from_part"] = guarded.tolist()
                            target_record["source"] = (
                                str(target_record.get("source", "pose"))
                                + "+continuity_guard"
                            )
                            temporal_delta = guard_delta
                        else:
                            temporal_delta *= float(part_config.get(
                                "rejected_temporal_decay",
                                refinement.get("rejected_temporal_decay", 0.5),
                            ))
                    else:
                        temporal_delta *= float(part_config.get(
                            "rejected_temporal_decay",
                            refinement.get("rejected_temporal_decay", 0.5),
                        ))
                part_report["frames"][timestamp] = {
                    "accepted": accepted,
                    "continuity_fallback": continuity_fallback,
                    "available_views": sorted(available),
                    "evaluations": frame_report["evaluations"],
                    "translation_delta_m": frame_report[
                        "translation_delta_m"
                    ],
                    "translation_delta_norm_m": frame_report[
                        "translation_delta_norm_m"
                    ],
                    "rotation_delta_deg": frame_report[
                        "rotation_delta_deg"
                    ],
                    "optimize_loss_improvement": frame_report[
                        "optimize_loss_improvement"
                    ],
                    "holdout_loss_degradation": frame_report[
                        "holdout_loss_degradation"
                    ],
                    "absolute_gate_failures": frame_report[
                        "absolute_gate_failures"
                    ],
                    "global_reacquired": reacquired,
                    "exact_triangle": exact_report,
                    "global_reacquire": (
                        None
                        if reacquire_report is None
                        else {
                            "evaluations": reacquire_report["evaluations"],
                            "loss_improvement": reacquire_report[
                                "loss_improvement"
                            ],
                            "translation_delta_norm_m": reacquire_report[
                                "translation_delta_norm_m"
                            ],
                            "rotation_delta_deg": reacquire_report[
                                "rotation_delta_deg"
                            ],
                            "rotation_prior": reacquire_report[
                                "rotation_prior"
                            ],
                            "translation_prior": reacquire_report[
                                "translation_prior"
                            ],
                            "baseline": compact_metric(
                                reacquire_report["baseline"]
                            ),
                            "selected": compact_metric(
                                reacquire_report["selected"]
                            ),
                        }
                    ),
                    "baseline_optimize": compact_metric(
                        frame_report["baseline_optimize"]
                    ),
                    "refined_optimize": compact_metric(
                        frame_report["refined_optimize"]
                    ),
                    "baseline_holdout": compact_metric(
                        frame_report["baseline_holdout"]
                    ),
                    "refined_holdout": compact_metric(
                        frame_report["refined_holdout"]
                    ),
                }
                processed_frames += 1
                print(
                    f"{part} {timestamp}: accepted={accepted} "
                    f"opt Δ={frame_report['optimize_loss_improvement']:+.4f} "
                    f"holdout Δ={frame_report['holdout_loss_degradation']:+.4f} "
                    f"t={1000.0 * frame_report['translation_delta_norm_m']:.1f}mm "
                    f"r={frame_report['rotation_delta_deg']:.2f}deg",
                    flush=True,
                )
            if (
                args.max_frames is not None
                and processed_frames >= args.max_frames
            ):
                break
        post_smoothing_passes = int(
            part_config.get(
                "post_smoothing_passes",
                refinement.get("post_smoothing_passes", 0),
            )
        )
        if post_smoothing_passes > 0 and args.frames is None:
            post_ranges = []
            for range_start, range_end in ranges:
                clipped_start = max(
                    range_start,
                    args.start_frame
                    if args.start_frame is not None
                    else range_start,
                )
                clipped_end = min(
                    range_end,
                    args.end_frame
                    if args.end_frame is not None
                    else range_end,
                )
                if clipped_start <= clipped_end:
                    post_ranges.append((clipped_start, clipped_end))
            part_poses = {
                int(timestamp): np.asarray(
                    frame_record["parts"][part]["T_world_from_part"],
                    dtype=np.float64,
                )
                for timestamp, frame_record in trajectory["frames"].items()
            }
            trusted_frames = {
                int(timestamp)
                for timestamp, frame_report in part_report["frames"].items()
                if frame_report.get("accepted", False)
            }.union(protected_anchor_frames)
            part_poses, interpolated_frames = (
                interpolate_untrusted_pose_frames(
                    part_poses,
                    post_ranges,
                    trusted_frames=trusted_frames,
                )
            )
            smoothing_fixed_frames = set(protected_anchor_frames)
            if part_config.get(
                "post_smoothing_preserve_trusted",
                refinement.get("post_smoothing_preserve_trusted", False),
            ):
                smoothing_fixed_frames.update(trusted_frames)
            smoothed = smooth_pose_ranges(
                part_poses,
                post_ranges,
                passes=post_smoothing_passes,
                fixed_frames=smoothing_fixed_frames,
            )
            smoothed_frames = {
                frame
                for range_start, range_end in post_ranges
                for frame in range(range_start, range_end + 1)
                if frame in smoothed
            }
            for frame in sorted(smoothed_frames):
                target_record = trajectory["frames"][f"{frame:06d}"]["parts"][
                    part
                ]
                target_record["T_world_from_part"] = smoothed[frame].tolist()
                target_record["source"] = (
                    str(target_record.get("source", "pose"))
                    + "+temporal_smoothing"
                )
            part_report["post_smoothing"] = {
                "passes": post_smoothing_passes,
                "frames": len(smoothed_frames),
                "fixed_boundary_policy": "adjacent_static_frames",
                "interpolated_untrusted_frames": interpolated_frames,
                "fixed_protected_anchor_frames": sorted(
                    protected_anchor_frames
                ),
                "fixed_trusted_frames": sorted(
                    smoothing_fixed_frames.difference(protected_anchor_frames)
                ),
            }

        frame_rows = list(part_report["frames"].values())
        evaluated = [
            row for row in frame_rows if "baseline_optimize" in row
        ]
        accepted_rows = [row for row in evaluated if row["accepted"]]

        def selected_mean_iou(split: str) -> tuple[float | None, float | None]:
            rows = [
                row
                for row in evaluated
                if row[f"baseline_{split}"].get("mean_iou") is not None
                and row[f"refined_{split}"].get("mean_iou") is not None
            ]
            if not rows:
                return None, None
            baseline_mean = float(
                np.mean(
                    [row[f"baseline_{split}"]["mean_iou"] for row in rows]
                )
            )
            selected_mean = float(
                np.mean(
                    [
                        (
                            row[f"refined_{split}"]["mean_iou"]
                            if row["accepted"]
                            else row[f"baseline_{split}"]["mean_iou"]
                        )
                        for row in rows
                    ]
                )
            )
            return baseline_mean, selected_mean

        baseline_optimize_iou, selected_optimize_iou = selected_mean_iou(
            "optimize"
        )
        baseline_holdout_iou, selected_holdout_iou = selected_mean_iou(
            "holdout"
        )
        part_report["summary"] = {
            "evaluated_frames": len(evaluated),
            "accepted_frames": accepted_count,
            "acceptance_rate": (
                float(accepted_count / len(evaluated)) if evaluated else None
            ),
            "baseline_optimize_mean_iou": baseline_optimize_iou,
            "selected_optimize_mean_iou": selected_optimize_iou,
            "baseline_holdout_mean_iou": baseline_holdout_iou,
            "selected_holdout_mean_iou": selected_holdout_iou,
            "accepted_translation_delta_median_m": (
                float(
                    np.median(
                        [
                            row["translation_delta_norm_m"]
                            for row in accepted_rows
                        ]
                    )
                )
                if accepted_rows
                else None
            ),
            "accepted_translation_delta_max_m": (
                float(
                    max(
                        row["translation_delta_norm_m"]
                        for row in accepted_rows
                    )
                )
                if accepted_rows
                else None
            ),
            "accepted_rotation_delta_median_deg": (
                float(
                    np.median(
                        [row["rotation_delta_deg"] for row in accepted_rows]
                    )
                )
                if accepted_rows
                else None
            ),
            "accepted_rotation_delta_max_deg": (
                float(
                    max(row["rotation_delta_deg"] for row in accepted_rows)
                )
                if accepted_rows
                else None
            ),
        }
        if exact_renderer is not None:
            exact_renderer.close()
        report["parts"][part] = part_report
        if (
            args.max_frames is not None
            and processed_frames >= args.max_frames
        ):
            break

    refresh_trajectory_derived_fields(trajectory)
    trajectory_validation, validation_failures = validate_trajectory(
        cfg, trajectory, enforce_assembly=False
    )
    report["trajectory_validation"] = trajectory_validation
    if validation_failures:
        report["summary"] = {
            "processed_frames": processed_frames,
            "accepted_frames": int(
                sum(
                    item["summary"]["accepted_frames"]
                    for item in report["parts"].values()
                )
            ),
            "validation_passed": False,
        }
        write_json(report_path, report)
        raise RuntimeError("; ".join(validation_failures))
    trajectory.setdefault("refinements", []).append(
        {
            "method": report["method"],
            "input": str(trajectory_path),
            "report": str(report_path),
        }
    )
    write_trajectory_files(trajectory, output_path)
    report["trajectory_output_sha256"] = sha256_file(output_path)
    report["summary"] = {
        "processed_frames": processed_frames,
        "accepted_frames": int(
            sum(
                item["summary"]["accepted_frames"]
                for item in report["parts"].values()
            )
        ),
        "validation_passed": True,
    }
    write_json(report_path, report)
    print(f"trajectory -> {output_path}", flush=True)
    print(f"report -> {report_path}", flush=True)


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/stages/pose/refine_pose_render_loss.py")
    main()
