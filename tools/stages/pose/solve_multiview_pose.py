#!/usr/bin/env python
"""Solve a body-relative 6D-pose trajectory from synchronized multi-view data.

This file is the orchestration layer.  Tracking strategies, symmetry handling,
trajectory normalization, and validation live in ``common/``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.appearance_pose import refine_anchor_orientations
from common.calibration_cache import build_calibration_fingerprint
from common.io_utils import load_json, write_json
from common.mesh_align import align_mesh_to_cloud
from common.multiview_quality import (
    cloud_supported_view_quality,
    mask_area_quality,
    part_visibility_quality,
    temporal_mask_area_references,
)
from common.pose_config import validate_pose_config
from common.pose_tracking import (
    fuse_part_clouds,
    load_part_cloud,
    track_cloud_registration,
    track_cloud_registration_reverse,
    track_anchor_relative_registration,
    track_mask_bbox_translation,
    track_model_translation,
)
from common.pose_transforms import (
    rigid_from_similarity,
    similarity_from_rigid,
)
from common.pose_refinement import sample_canonical
from common.render_loss_refinement import (
    MultiViewRenderObjective,
    RenderObservation,
    refine_pose_coordinate_search,
)
from common.symmetry import resolve_symmetric_pose, symmetry_spec_from_state
from common.trajectory_constraints import interpolate_pose
from common.pose_validation import validate_world_poses
from common.scale_diagnostics import (
    earliest_static_interval_scale_fits,
    select_anchor_scale,
)
from common.silhouette_scale_calibration import (
    build_render_observations,
    calibrate_part_scale,
)
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    world_pose_record,
    write_trajectory_files,
)


def _calibration_frames(state: dict, anchor: int) -> list[int]:
    windows = state.get("anchor_windows", {})
    return [int(value) for value in windows.get(str(anchor), [anchor])]


def _fit_anchor(
    mesh: trimesh.Trimesh,
    cloud_root: Path,
    state: dict,
    anchor: int,
    part: str,
    seed: int,
    fixed_scale: float | None = None,
) -> dict:
    frames = _calibration_frames(state, anchor)
    cloud = fuse_part_clouds(cloud_root, frames, part, 40000, seed)
    fit = align_mesh_to_cloud(
        mesh,
        cloud,
        n_mesh_sample=40000,
        n_obs_max=16000,
        coarse_iters=30,
        fine_iters=100,
        seed=seed,
        fixed_scale=fixed_scale,
        return_candidates=fixed_scale is not None,
    )
    return {
        "frame": anchor,
        "frames": frames,
        "n_cloud_points": int(len(cloud)),
        "S_world_from_raw": fit["T_mesh_to_world"],
        "scale": float(fit["scale"]),
        "fit_rmse_m": float(fit["fit_rmse"]),
        "icp_cost": float(fit["icp_cost"]),
        "candidate_fits": fit.get("candidate_fits", []),
    }


def _view_quality_settings(config: dict, part: str) -> dict:
    settings = dict(config.get("view_quality", {}))
    per_part = settings.pop("parts", {})
    settings.update(dict(per_part.get(part, {})))
    return settings


def _part_visibility_report(
    mask_root: Path,
    frame: int,
    part_id: int,
    views: list[str],
    *,
    settings: dict,
    cloud_row: dict | None,
    reference_areas: dict[str, float] | None = None,
) -> dict:
    from PIL import Image

    masks = {}
    for view in views:
        path = mask_root / f"{frame:06d}" / f"{view}.png"
        if path.exists():
            masks[view] = np.asarray(Image.open(path))
    report = mask_area_quality(
        masks,
        part_id,
        minimum_pixels=int(settings.get("minimum_full_mask_pixels", 800)),
        maximum_area_ratio=float(
            settings.get("maximum_mask_area_ratio", 4.0)
        ),
        minimum_area_ratio=float(
            settings.get("minimum_mask_area_ratio", 0.0)
        ),
        reference_areas=reference_areas,
        presence_minimum_pixels=int(
            settings.get(
                "presence_minimum_full_mask_pixels",
                settings.get("minimum_full_mask_pixels", 800),
            )
        ),
    )
    require_cloud = bool(
        settings.get("use_cloud_supported_view_gate", False)
    )
    cloud_report = None
    if require_cloud:
        cloud_report = cloud_supported_view_quality(
            cloud_row,
            views,
            minimum_supported_points=int(
                settings.get("minimum_supported_points_per_view", 30)
            ),
            minimum_support_fraction=float(
                settings.get("minimum_view_support_fraction", 0.25)
            ),
        )
    result = part_visibility_quality(
        report,
        views,
        cloud_report=cloud_report,
        require_cloud_support=require_cloud,
        minimum_pose_views=int(settings.get("minimum_pose_views", 1)),
        minimum_visibility_views=int(
            settings.get("minimum_visibility_views", 1)
        ),
    )
    result["mask_quality"] = report
    if cloud_report is not None:
        result["cloud_quality"] = cloud_report
    return result


def _serialize_fit(fit: dict) -> dict:
    """Convert fit diagnostics, including retained global candidates, to JSON."""

    return {
        key: (
            [
                {
                    candidate_key: (
                        candidate_value.tolist()
                        if isinstance(candidate_value, np.ndarray)
                        else candidate_value
                    )
                    for candidate_key, candidate_value in candidate.items()
                }
                for candidate in value
            ]
            if key == "candidate_fits"
            else value.tolist()
            if isinstance(value, np.ndarray)
            else value
        )
        for key, value in fit.items()
    }


def _align_similarity_to_table(
    mesh: trimesh.Trimesh,
    similarity: np.ndarray,
    plane: dict,
    *,
    maximum_shift_m: float,
) -> tuple[np.ndarray, dict]:
    """Place the complete mesh's robust bottom on an observed support plane."""

    transform = np.asarray(similarity, dtype=np.float64).copy()
    normal = np.asarray(plane["normal_world"], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    point = np.asarray(plane["point_world"], dtype=np.float64)
    vertices = (
        np.asarray(mesh.vertices, dtype=np.float64) @ transform[:3, :3].T
        + transform[:3, 3]
    )
    signed = (vertices - point) @ normal
    bottom_gap = float(np.quantile(signed, 0.005))
    shift = -bottom_gap
    accepted = bool(abs(shift) <= float(maximum_shift_m))
    if accepted:
        transform[:3, 3] += shift * normal
    return transform, {
        "accepted": accepted,
        "bottom_gap_before_m": bottom_gap,
        "translation_along_table_normal_m": shift if accepted else 0.0,
        "maximum_shift_m": float(maximum_shift_m),
        "reason": None if accepted else "required_table_shift_exceeds_limit",
    }


def _select_stable_interval_anchor_winners(
    config: dict,
    anchors: dict[str, dict[int, np.ndarray]],
    anchor_info: dict,
    appearance_reports: dict,
) -> dict:
    """Keep one visually strongest tracking anchor per static interval."""

    selection = {}
    for part in config["parts"]:
        selection[part] = []
        state = config["states"][part]
        for static_start, static_end in state.get("static_ranges", []):
            group = sorted(
                frame for frame in anchors[part]
                if int(static_start) <= frame <= int(static_end)
            )
            if len(group) <= 1:
                continue

            def score(frame: int) -> tuple[float, float, float, int]:
                selected = (
                    appearance_reports.get(part, {})
                    .get("anchors", {})
                    .get(str(frame), {})
                    .get("selected", {})
                )
                mean_iou = float(selected.get("mean_silhouette_iou", -1.0))
                worst_iou = float(selected.get("worst_silhouette_iou", -1.0))
                render_report = anchor_info[part][str(frame)].get(
                    "anchor_render_refinement_post_scale",
                    anchor_info[part][str(frame)].get(
                        "anchor_render_refinement", {}
                    ),
                )
                if render_report.get("accepted", False):
                    refined = render_report.get("refined_optimize", {})
                    mean_iou = float(refined.get("mean_iou", mean_iou))
                    worst_iou = float(
                        refined.get("worst_view_iou", worst_iou)
                    )
                rmse = float(
                    anchor_info[part][str(frame)].get("fit_rmse_m", 1.0)
                )
                return mean_iou, worst_iou, -rmse, -frame

            winner = max(group, key=score)
            losers = [frame for frame in group if frame != winner]
            selection[part].append({
                "static_range": [int(static_start), int(static_end)],
                "candidates": group,
                "winner": winner,
                "winner_score": list(score(winner)),
            })
            for frame in losers:
                anchors[part].pop(frame, None)
                row = anchor_info[part][str(frame)]
                row["usable_for_tracking"] = False
                row["reliable"] = False
                row["rejection_reason"] = (
                    "redundant_static_interval_candidate_lower_render_score"
                )
                row["static_interval_winner"] = winner
    return selection


def _add_static_consensus_hypotheses(
    state: dict,
    anchors: dict[int, np.ndarray],
    hypotheses: dict[int, list[dict]],
    anchor_info: dict,
) -> None:
    """Let every static candidate reuse its interval's strongest geometry fit."""

    for static_start, static_end in state.get("static_ranges", []):
        group = sorted(
            frame for frame in anchors
            if int(static_start) <= frame <= int(static_end)
        )
        if len(group) <= 1:
            continue
        consensus = min(
            group,
            key=lambda frame: float(
                anchor_info[str(frame)].get("fit_rmse_m", float("inf"))
            ),
        )
        consensus_rows = list(hypotheses[consensus])
        for frame in group:
            if frame == consensus:
                continue
            # Propagate the complete PCA-basin hypothesis bank.  Copying only
            # the consensus anchor's default transform makes a correct
            # non-default basin unavailable at the other stable frames; the
            # temporal chain can then be forced into a wrong but continuous
            # 180-degree orientation.
            existing = {str(row.get("label")) for row in hypotheses[frame]}
            for consensus_row in consensus_rows:
                label = (
                    f"static_consensus_{consensus}:"
                    f"{consensus_row['label']}"
                )
                if label in existing:
                    continue
                hypotheses[frame].append({
                    "label": label,
                    "similarity": np.asarray(
                        consensus_row["similarity"], dtype=np.float64
                    ).copy(),
                })
                existing.add(label)


def _refine_anchor_poses_with_render_loss(
    config: dict,
    meshes: dict[str, trimesh.Trimesh],
    origins: dict[str, np.ndarray],
    scales: dict[str, float],
    anchors: dict[str, dict[int, np.ndarray]],
    *,
    seed_offset: int,
) -> dict:
    """Refine stable anchors before they are propagated into a trajectory."""

    automation = config.get("automation", {})
    anchor_settings = dict(
        automation.get("anchor_render_refinement", {})
    )
    if not anchor_settings.get("enabled", True):
        return {"enabled": False}
    settings = dict(config.get("render_loss_refinement", {}))
    settings.update(anchor_settings)
    reports: dict[str, dict] = {}
    for part_index, part in enumerate(config["parts"]):
        part_settings = dict(settings)
        part_settings.update(
            anchor_settings.get("parts", {}).get(part, {})
        )
        optimize_views = [
            str(value)
            for value in part_settings.get("optimize_views", config["views"])
        ]
        holdout_views = [
            str(value) for value in part_settings.get("holdout_views", [])
        ]
        points = sample_canonical(
            meshes[part],
            float(scales[part]),
            np.asarray(origins[part], dtype=np.float64),
            count=int(part_settings.get("surface_points", 30000)),
            seed=int(part_settings.get("seed", 1701))
            + seed_offset + part_index,
        )
        symmetry = symmetry_spec_from_state(config["states"][part])
        symmetry_axis = (
            symmetry.axis
            if symmetry.equivalence == "continuous_axial"
            else None
        )
        configured_evidence = dict(
            config["states"][part]
            .get("appearance", {})
            .get("anchor_evidence_frames", {})
        )
        configured_evidence.update(
            part_settings.get("anchor_evidence_frames", {})
        )
        part_report = {}
        for frame, similarity in sorted(anchors[part].items()):
            evidence_frames = [
                int(value)
                for value in configured_evidence.get(str(frame), [frame])
            ]
            observations: list[RenderObservation] = []
            optimize_observations: list[str] = []
            holdout_observations: list[str] = []
            available_views: set[str] = set()
            for evidence_frame in evidence_frames:
                for row in build_render_observations(
                    config, part, evidence_frame, part_settings
                ):
                    alias = f"{evidence_frame:06d}:{row.view}"
                    observations.append(RenderObservation(
                        view=alias,
                        intrinsics=row.intrinsics,
                        extrinsics=row.extrinsics,
                        target_mask=row.target_mask,
                        observed_depth=row.observed_depth,
                    ))
                    available_views.add(row.view)
                    if row.view in optimize_views:
                        optimize_observations.append(alias)
                    if row.view in holdout_views:
                        holdout_observations.append(alias)
            minimum_optimize_views = int(
                part_settings.get("minimum_optimize_views", 3)
            )
            minimum_holdout_views = int(
                part_settings.get("minimum_holdout_views", 0)
            )
            if len(available_views.intersection(optimize_views)) < (
                minimum_optimize_views
            ):
                part_report[str(frame)] = {
                    "accepted": False,
                    "skip_reason": "insufficient_views",
                    "available_views": sorted(available_views),
                    "evidence_frames": evidence_frames,
                }
                continue
            if len(available_views.intersection(holdout_views)) < (
                minimum_holdout_views
            ):
                part_report[str(frame)] = {
                    "accepted": False,
                    "skip_reason": "insufficient_holdout_views",
                    "available_views": sorted(available_views),
                    "evidence_frames": evidence_frames,
                }
                continue
            objective = MultiViewRenderObjective(
                points, observations, part_settings
            )
            initial = rigid_from_similarity(similarity, origins[part])
            is_reference = part == config["reference_part"]
            orientation_constraints = []
            appearance_settings = config["states"][part].get(
                "appearance", {}
            )
            opening_axis_raw = appearance_settings.get("opening_axis_raw")
            support_plane = config.get("support_plane", {})
            if (
                opening_axis_raw is not None
                and support_plane.get("accepted", False)
            ):
                orientation_constraints.append({
                    "label": "opening_up",
                    "axis_part": opening_axis_raw,
                    "target_world": support_plane["normal_world"],
                    "minimum_alignment": appearance_settings.get(
                        "minimum_opening_up_alignment", 0.0
                    ),
                    "maximum_alignment": appearance_settings.get(
                        "maximum_opening_up_alignment", 1.0
                    ),
                })
            selected, report = refine_pose_coordinate_search(
                objective,
                initial,
                optimize_views=optimize_observations,
                holdout_views=holdout_observations,
                translation_steps_m=[
                    float(value)
                    for value in part_settings.get(
                        "anchor_translation_steps_m",
                        [0.020, 0.010, 0.005, 0.002],
                    )
                ],
                rotation_steps_deg=[
                    float(value)
                    for value in part_settings.get(
                        "anchor_rotation_steps_deg", [15.0, 10.0, 5.0, 2.0, 1.0]
                    )
                ],
                symmetry_axis_part=symmetry_axis,
                optimize_rotation=bool(
                    part_settings.get("anchor_optimize_rotation", True)
                ),
                maximum_translation_delta_m=float(
                    part_settings.get(
                        (
                            "anchor_reference_maximum_translation_delta_m"
                            if is_reference
                            else "anchor_maximum_translation_delta_m"
                        ),
                        0.05 if is_reference else 0.08,
                    )
                ),
                maximum_rotation_delta_deg=float(
                    part_settings.get(
                        (
                            "anchor_reference_maximum_rotation_delta_deg"
                            if is_reference
                            else "anchor_maximum_rotation_delta_deg"
                        ),
                        15.0 if is_reference else 45.0,
                    )
                ),
                minimum_improvement=float(
                    part_settings.get("anchor_minimum_improvement", 0.003)
                ),
                maximum_holdout_degradation=float(
                    part_settings.get(
                        "anchor_maximum_holdout_degradation", 0.02
                    )
                ),
                minimum_refined_iou=float(
                    part_settings.get("minimum_refined_iou", 0.0)
                ),
                minimum_refined_target_coverage=float(
                    part_settings.get(
                        "minimum_refined_target_coverage", 0.0
                    )
                ),
                minimum_holdout_iou=float(
                    part_settings.get("minimum_holdout_iou", 0.0)
                ),
                minimum_per_view_iou=float(
                    part_settings.get("minimum_per_view_iou", 0.0)
                ),
                maximum_worst_view_loss=float(
                    part_settings.get("maximum_worst_view_loss", 1.0e9)
                ),
                prior_weight=float(
                    part_settings.get("anchor_prior_weight", 0.01)
                ),
                temporal_weight=0.0,
                orientation_constraints=orientation_constraints,
            )
            report["evidence_frames"] = evidence_frames
            report["observation_count"] = len(observations)
            report["available_views"] = sorted(available_views)
            if report["accepted"]:
                anchors[part][frame] = similarity_from_rigid(
                    selected, scales[part], origins[part]
                )
            part_report[str(frame)] = report
            print(
                f"anchor render {part}@{frame}: "
                f"accepted={report['accepted']} "
                f"iou={report['baseline_optimize']['mean_iou']:.3f}->"
                f"{report['refined_optimize']['mean_iou']:.3f} "
                f"t={1000.0 * report['translation_delta_norm_m']:.1f}mm "
                f"r={report['rotation_delta_deg']:.1f}deg",
                flush=True,
            )
        reports[part] = part_report
    return {"enabled": True, "parts": reports}


def _fit_calibration(
    config: dict,
    config_path: Path,
    cloud_root: Path,
    mesh_dir: Path,
    meshes: dict[str, trimesh.Trimesh],
    origins: dict[str, np.ndarray],
    input_fingerprint: dict,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, float], dict]:
    anchor_info = {}
    anchors = {}
    scales = {}
    scale_consensus = {}
    anchor_hypotheses: dict[str, dict[int, list[dict]]] = {}
    moving_reference = bool(
        config.get("automation", {}).get("allow_moving_reference", False)
    )
    for index, part in enumerate(config["parts"]):
        state = config["states"][part]
        configured_scale = (
            None
            if state.get("scale_prior") is None
            else float(state["scale_prior"])
        )
        if part == config["reference_part"] and not moving_reference:
            calibration = state["calibration_frames"]
            frames = [int(calibration[len(calibration) // 2])]
        else:
            frames = [int(value) for value in state["anchor_frames"]]
        anchors[part] = {}
        anchor_hypotheses[part] = {}
        anchor_info[part] = {}
        fits = []
        for anchor in frames:
            local_state = state
            if part == config["reference_part"] and not moving_reference:
                local_state = dict(state)
                local_state["anchor_windows"] = {
                    str(anchor): state["calibration_frames"]
                }
            try:
                fit = _fit_anchor(
                    meshes[part],
                    cloud_root,
                    local_state,
                    anchor,
                    part,
                    seed=31 + 10 * index + anchor,
                    fixed_scale=configured_scale,
                )
            except RuntimeError as error:
                if not str(error).startswith("no cloud for "):
                    raise
                anchor_info[part][str(anchor)] = {
                    "frame": int(anchor),
                    "frames": _calibration_frames(local_state, anchor),
                    "reliable": False,
                    "rejection_reason": "no_quality_cloud",
                    "error": str(error),
                }
                print(
                    f"rejecting {part}@{anchor}: no quality cloud",
                    flush=True,
                )
                continue
            fits.append(fit)
            anchor_info[part][str(anchor)] = _serialize_fit(fit)
            print(
                f"calibration {part}@{anchor}: "
                f"scale={fit['scale']:.6f} rmse={fit['fit_rmse_m']:.5f}",
                flush=True,
            )
        if not fits:
            raise RuntimeError(
                f"{part}: no calibration anchor has a quality point cloud"
            )
        if configured_scale is not None:
            scale = float(configured_scale)
            consensus = {
                "method": "configured_scale_prior",
                "selected_scale": scale,
                "candidate_count": len(fits),
            }
        else:
            scale_fits = fits
            scale_fit_selection = {
                "method": "all_available_scale_anchors",
                "selected_frames": [int(fit["frame"]) for fit in fits],
                "excluded_frames": [],
                "fallback_to_all_fits": False,
            }
            if config.get("automation", {}).get(
                "scale_consensus_first_static_interval", False
            ):
                scale_fits, scale_fit_selection = (
                    earliest_static_interval_scale_fits(
                        fits,
                        state.get("static_ranges", []),
                    )
                )
            scale, consensus = select_anchor_scale(scale_fits)
            consensus["fit_selection"] = scale_fit_selection
        scales[part] = scale
        scale_consensus[part] = consensus
        for free_fit in fits:
            anchor = int(free_fit["frame"])
            local_state = state
            if part == config["reference_part"] and not moving_reference:
                local_state = dict(state)
                local_state["anchor_windows"] = {
                    str(anchor): state["calibration_frames"]
                }
            fixed_fit = _fit_anchor(
                meshes[part],
                cloud_root,
                local_state,
                anchor,
                part,
                seed=1031 + 10 * index + anchor,
                fixed_scale=scale,
            )
            transform = np.asarray(
                fixed_fit["S_world_from_raw"], dtype=np.float64
            )
            anchors[part][anchor] = transform
            anchor_info[part][str(anchor)] = {
                **_serialize_fit(fixed_fit),
                "fixed_scale": scale,
                "free_scale_fit": _serialize_fit(free_fit),
            }
            maximum_fit_rmse = float(
                state.get(
                    "maximum_anchor_fit_rmse_m",
                    config.get("maximum_anchor_fit_rmse_m", 0.05),
                )
            )
            reliable = bool(fixed_fit["fit_rmse_m"] <= maximum_fit_rmse)
            anchor_info[part][str(anchor)]["reliable"] = reliable
            anchor_info[part][str(anchor)]["geometry_quality_passed"] = reliable
            anchor_info[part][str(anchor)]["usable_for_tracking"] = True
            if not reliable:
                anchor_info[part][str(anchor)]["warning"] = (
                    "weak_geometry_fit_retained_for_multiview_selection"
                )
            anchor_info[part][str(anchor)][
                "maximum_anchor_fit_rmse_m"
            ] = maximum_fit_rmse
            anchor_hypotheses[part][anchor] = [
                {
                    "label": str(candidate["label"]),
                    "similarity": np.asarray(
                        candidate["T_mesh_to_world"], dtype=np.float64
                    ),
                }
                for candidate in fixed_fit.get("candidate_fits", [])
            ] or [{"label": "registration", "similarity": transform}]
            print(
                f"fixed-scale {part}@{anchor}: scale={scale:.6f} "
                f"rmse={fixed_fit['fit_rmse_m']:.5f} "
                f"candidates={len(anchor_hypotheses[part][anchor])}",
                flush=True,
            )

        weak_geometry = {
            int(frame)
            for frame, row in anchor_info[part].items()
            if not row.get("reliable", True)
        }
        if weak_geometry:
            print(
                f"retaining weak {part} anchors for visual/temporal selection: "
                f"{sorted(weak_geometry)}",
                flush=True,
            )
        if not anchors[part]:
            raise RuntimeError(
                f"{part}: every anchor failed fixed-scale registration quality"
            )

        # A static interval supplies a strong hypothesis: the part should keep
        # the preceding anchor pose.  The appearance chain still decides it
        # jointly with the independently registered multi-view candidates.
        if state.get("appearance", {}).get(
            "allow_static_consensus_hypotheses", True
        ):
            _add_static_consensus_hypotheses(
                state,
                anchors[part],
                anchor_hypotheses[part],
                anchor_info[part],
            )

    scale_reports = {}
    scale_settings = dict(
        config.get("automation", {}).get("silhouette_scale_calibration", {})
    )
    if scale_settings.get("enabled", False):
        render_settings = dict(config.get("render_loss_refinement", {}))
        render_settings.update(scale_settings)
        for part_index, part in enumerate(config["parts"]):
            part_scale_settings = dict(render_settings)
            part_scale_settings.update(
                scale_settings.get("parts", {}).get(part, {})
            )
            if (
                config["states"][part].get("scale_prior") is not None
                and part_scale_settings.get(
                    "treat_scale_prior_as_configured", True
                )
            ):
                part_scale_settings["configured_scale_prior"] = True
            scales[part], anchors[part], scale_reports[part] = calibrate_part_scale(
                cfg=config,
                part=part,
                mesh=meshes[part],
                raw_origin=origins[part],
                base_scale=scales[part],
                anchors=anchors[part],
                settings=part_scale_settings,
                seed=int(scale_settings.get("seed", 8111)) + part_index,
            )
            for frame, transform in anchors[part].items():
                anchor_info[part][str(frame)]["S_world_from_raw"] = (
                    transform.tolist()
                )
                anchor_info[part][str(frame)]["fixed_scale"] = scales[part]
            scale_report = scale_reports[part]
            selected_scale_factor = float(
                scale_report.get("selected_scale_factor", 1.0)
            )
            print(
                f"silhouette scale {part}: "
                f"factor={selected_scale_factor:.3f} "
                f"scale={scales[part]:.6f} "
                f"accepted={scale_report['accepted']}"
                + (
                    f" reason={scale_report['reason']}"
                    if scale_report.get("reason")
                    else ""
                ),
                flush=True,
            )
            if (
                scale_settings.get("require_quality_gate", False)
                and not scale_report.get("quality_passed", False)
            ):
                candidates = scale_report.get("candidates", [])
                selected_index = int(scale_report.get("selected_index", 0))
                selected_row = (
                    candidates[selected_index]
                    if 0 <= selected_index < len(candidates)
                    else {}
                )
                raise RuntimeError(
                    f"{part}: silhouette scale calibration failed absolute "
                    f"quality gate (reason={scale_report.get('reason')}, "
                    f"optimize_iou={selected_row.get('optimize_iou')})"
                )

            # Scale and rigid registration are coupled for a partial cloud.
            # The silhouette stage preserves the old part-centre pose while it
            # changes scale, but the old rotation/translation were fitted with
            # a differently sized mesh.  Refit every surviving anchor at the
            # selected scale before asking appearance to choose a PCA basin.
            # Otherwise appearance can replace the orientation after scale was
            # scored and invalidate the very silhouettes that selected it.
            if scale_reports[part].get("accepted", False):
                state = config["states"][part]
                refitted_anchors: dict[int, np.ndarray] = {}
                refitted_hypotheses: dict[int, list[dict]] = {}
                for anchor in sorted(anchors[part]):
                    local_state = state
                    if part == config["reference_part"] and not moving_reference:
                        local_state = dict(state)
                        local_state["anchor_windows"] = {
                            str(anchor): state["calibration_frames"]
                        }
                    fixed_fit = _fit_anchor(
                        meshes[part],
                        cloud_root,
                        local_state,
                        anchor,
                        part,
                        seed=2031 + 10 * part_index + anchor,
                        fixed_scale=scales[part],
                    )
                    transform = np.asarray(
                        fixed_fit["S_world_from_raw"], dtype=np.float64
                    )
                    maximum_fit_rmse = float(
                        state.get(
                            "maximum_anchor_fit_rmse_m",
                            config.get("maximum_anchor_fit_rmse_m", 0.05),
                        )
                    )
                    reliable = bool(
                        fixed_fit["fit_rmse_m"] <= maximum_fit_rmse
                    )
                    previous = anchor_info[part][str(anchor)]
                    refit_info = {
                        **_serialize_fit(fixed_fit),
                        "fixed_scale": scales[part],
                        "free_scale_fit": previous.get("free_scale_fit"),
                        "pre_scale_refit": previous,
                        "reliable": reliable,
                        "geometry_quality_passed": reliable,
                        "usable_for_tracking": True,
                        "maximum_anchor_fit_rmse_m": maximum_fit_rmse,
                        "registration_stage": "post_silhouette_scale_refit",
                    }
                    if not reliable:
                        # Re-fitting a partial cloud after changing scale can
                        # jump into a different PCA basin. Keep the already
                        # silhouette-calibrated pose and its hypotheses when
                        # this optional refit is weak; do not destroy a motion
                        # bracket that the tracker requires.
                        anchor_info[part][str(anchor)] = {
                            **previous,
                            "post_scale_refit_attempt": refit_info,
                            "post_scale_refit_accepted": False,
                            "usable_for_tracking": True,
                            "warning": "weak_post_scale_refit_preserved_prior_anchor",
                        }
                        print(
                            f"preserving pre-refit {part}@{anchor}: "
                            f"rmse={fixed_fit['fit_rmse_m']:.5f}",
                            flush=True,
                        )
                        refitted_anchors[anchor] = anchors[part][anchor]
                        refitted_hypotheses[anchor] = anchor_hypotheses[part][anchor]
                        continue
                    refit_info["post_scale_refit_accepted"] = True
                    anchor_info[part][str(anchor)] = refit_info
                    refitted_anchors[anchor] = transform
                    refitted_hypotheses[anchor] = [
                        {
                            "label": str(candidate["label"]),
                            "similarity": np.asarray(
                                candidate["T_mesh_to_world"],
                                dtype=np.float64,
                            ),
                        }
                        for candidate in fixed_fit.get("candidate_fits", [])
                    ] or [
                        {"label": "registration", "similarity": transform}
                    ]
                    print(
                        f"post-scale {part}@{anchor}: "
                        f"scale={scales[part]:.6f} "
                        f"rmse={fixed_fit['fit_rmse_m']:.5f} "
                        f"candidates={len(refitted_hypotheses[anchor])}",
                        flush=True,
                    )
                if not refitted_anchors:
                    raise RuntimeError(
                        f"{part}: every anchor failed post-scale registration"
                    )
                anchors[part] = refitted_anchors
                anchor_hypotheses[part] = refitted_hypotheses

                if state.get("appearance", {}).get(
                    "allow_static_consensus_hypotheses", True
                ):
                    _add_static_consensus_hypotheses(
                        state,
                        anchors[part],
                        anchor_hypotheses[part],
                        anchor_info[part],
                    )
    # Orientation candidates must be rendered at the visually calibrated
    # scale.  Evaluating texture/silhouette hypotheses at the often inflated
    # partial-cloud scale can select the wrong basin or reject every basin.
    appearance_reports = {}
    for part in config["parts"]:
        anchors[part], report = refine_anchor_orientations(
            cfg=config,
            state_cfg=config["states"][part],
            part=part,
            mesh=meshes[part],
            scale=scales[part],
            origin=origins[part],
            anchors=anchors[part],
            anchor_hypotheses=anchor_hypotheses[part],
        )
        appearance_reports[part] = report
        for frame, transform in anchors[part].items():
            anchor_info[part][str(frame)][
                "S_world_from_raw"
            ] = transform.tolist()
            selected = report.get("anchors", {}).get(str(frame), {}).get(
                "selected"
            )
            if selected is not None:
                anchor_info[part][str(frame)][
                    "appearance_selection"
                ] = selected
        rejected_visual_anchors = {
            int(frame)
            for frame, value in report.get("anchors", {}).items()
            if not value.get("selected", {}).get(
                "appearance_accepted", True
            )
        }
        if rejected_visual_anchors:
            for frame in rejected_visual_anchors:
                anchor_info[part][str(frame)]["appearance_quality_passed"] = False
                anchor_info[part][str(frame)]["usable_for_tracking"] = True
                anchor_info[part][str(frame)]["warning"] = (
                    "appearance_gate_failed_anchor_retained_for_bracketing"
                )
            report["weak_anchors_retained"] = sorted(rejected_visual_anchors)
            print(
                f"retaining visually weak {part} anchors for bracketing: "
                f"{sorted(rejected_visual_anchors)}",
                flush=True,
            )
    anchor_render_reports = _refine_anchor_poses_with_render_loss(
        config,
        meshes,
        origins,
        scales,
        anchors,
        seed_offset=100,
    )
    for part in config["parts"]:
        for frame, transform in anchors[part].items():
            anchor_info[part][str(frame)]["S_world_from_raw"] = (
                transform.tolist()
            )
            report = (
                anchor_render_reports.get("parts", {})
                .get(part, {})
                .get(str(frame))
            )
            if report is not None:
                anchor_info[part][str(frame)][
                    "anchor_render_refinement"
                ] = report

    # The first scale pass only supplies a usable size for orientation and
    # rigid render refinement.  Select one absolute pose per detected static
    # interval before the final scale pass: otherwise a redundant anchor in a
    # different 90/180-degree basin can corrupt the supposedly final metric
    # size even though that anchor is rejected immediately afterward.
    if config.get("automation", {}).get(
        "select_stable_interval_anchor_winner", True
    ):
        stable_interval_selection = _select_stable_interval_anchor_winners(
            config, anchors, anchor_info, appearance_reports
        )
    else:
        stable_interval_selection = {
            "enabled": False,
            "reason": "disabled_by_config",
        }

    # Scale is evaluated once more after orientation and rigid anchor
    # refinement.  Before that point a wrong PCA basin can make the correct
    # scale look worse.  This second pass freezes the now selected SE(3) pose,
    # keeps the independent holdout/cross-frame gates, and never re-enters ICP.
    post_anchor_scale_reports = {}
    post_anchor_scale_changed = False
    if (
        scale_settings.get("enabled", False)
        and scale_settings.get("post_anchor_render_pass", True)
    ):
        render_settings = dict(config.get("render_loss_refinement", {}))
        render_settings.update(scale_settings)
        for part_index, part in enumerate(config["parts"]):
            part_settings = dict(render_settings)
            part_settings.update(
                scale_settings.get("parts", {}).get(part, {})
            )
            if (
                config["states"][part].get("scale_prior") is not None
                and part_settings.get(
                    "treat_scale_prior_as_configured", True
                )
            ):
                part_settings["configured_scale_prior"] = True
            old_scale = scales[part]
            (
                scales[part],
                anchors[part],
                post_anchor_scale_reports[part],
            ) = calibrate_part_scale(
                cfg=config,
                part=part,
                mesh=meshes[part],
                raw_origin=origins[part],
                base_scale=old_scale,
                anchors=anchors[part],
                settings=part_settings,
                seed=int(scale_settings.get("seed", 8111)) + 100 + part_index,
            )
            post_anchor_scale_changed |= bool(
                post_anchor_scale_reports[part].get("accepted", False)
            )
            for frame, transform in anchors[part].items():
                anchor_info[part][str(frame)]["S_world_from_raw"] = (
                    transform.tolist()
                )
                anchor_info[part][str(frame)]["fixed_scale"] = scales[part]
            selected_scale_factor = float(
                post_anchor_scale_reports[part].get(
                    "selected_scale_factor", 1.0
                )
            )
            print(
                f"post-anchor scale {part}: "
                f"factor={selected_scale_factor:.3f} "
                f"scale={scales[part]:.6f} "
                f"accepted={post_anchor_scale_reports[part]['accepted']}",
                flush=True,
            )

    anchor_render_post_scale_reports = {"enabled": False}
    if post_anchor_scale_changed:
        anchor_render_post_scale_reports = (
            _refine_anchor_poses_with_render_loss(
                config,
                meshes,
                origins,
                scales,
                anchors,
                seed_offset=200,
            )
        )
        for part in config["parts"]:
            for frame, transform in anchors[part].items():
                anchor_info[part][str(frame)]["S_world_from_raw"] = (
                    transform.tolist()
                )
                report = (
                    anchor_render_post_scale_reports.get("parts", {})
                    .get(part, {})
                    .get(str(frame))
                )
                if report is not None:
                    anchor_info[part][str(frame)][
                        "anchor_render_refinement_post_scale"
                    ] = report

    table_alignment = {}
    table_plane = config.get("support_plane")
    table_settings = config.get("automation", {})
    reference_part = config["reference_part"]
    configured_support_parts = table_settings.get("align_parts_to_table")
    support_parts = (
        [str(part) for part in configured_support_parts]
        if configured_support_parts is not None
        else (
            [reference_part]
            if table_settings.get("align_reference_to_table", True)
            else []
        )
    )
    configured_support_ranges = table_settings.get("table_support_ranges", {})
    if (
        table_plane
        and table_plane.get("accepted", False)
        and support_parts
    ):
        unknown_support_parts = sorted(set(support_parts).difference(anchors))
        if unknown_support_parts:
            raise ValueError(
                "align_parts_to_table contains unknown parts: "
                f"{unknown_support_parts}"
            )
        for support_part in support_parts:
            table_alignment[support_part] = {}
            for frame, transform in list(anchors[support_part].items()):
                support_ranges = configured_support_ranges.get(support_part)
                if support_ranges is not None and not any(
                    int(start) <= frame <= int(end)
                    for start, end in support_ranges
                ):
                    table_alignment[support_part][str(frame)] = {
                        "accepted": False,
                        "reason": "anchor_outside_configured_table_support_ranges",
                        "configured_ranges": [
                            [int(start), int(end)]
                            for start, end in support_ranges
                        ],
                    }
                    continue
                aligned, alignment = _align_similarity_to_table(
                    meshes[support_part],
                    transform,
                    table_plane,
                    maximum_shift_m=float(
                        table_settings.get(
                            "maximum_table_alignment_m", 0.03
                        )
                    ),
                )
                table_alignment[support_part][str(frame)] = alignment
                if alignment["accepted"]:
                    anchors[support_part][frame] = aligned
                    anchor_info[support_part][str(frame)][
                        "S_world_from_raw"
                    ] = aligned.tolist()

    # Render refinement and table alignment both change translation, so both
    # can re-introduce scale/depth compensation after an earlier scale pass.
    # Finish with a frozen-pose exact-mesh pass after those constraints.  For
    # table-supported anchors every candidate is shifted along the plane
    # normal to preserve its already established contact, making scale the
    # only remaining degree of freedom.  No SE(3) optimization follows.
    final_pose_scale_reports = {}
    if (
        scale_settings.get("enabled", False)
        and scale_settings.get("final_pose_scale_pass", True)
    ):
        render_settings = dict(config.get("render_loss_refinement", {}))
        render_settings.update(scale_settings)
        for part_index, part in enumerate(config["parts"]):
            part_settings = dict(render_settings)
            part_settings.update(
                scale_settings.get("parts", {}).get(part, {})
            )
            contact_frames = [
                int(frame)
                for frame, report in table_alignment.get(part, {}).items()
                if report.get("accepted", False)
            ]
            if contact_frames:
                part_settings["support_plane"] = table_plane
                part_settings["support_contact_frames"] = contact_frames
            if (
                config["states"][part].get("scale_prior") is not None
                and part_settings.get(
                    "treat_scale_prior_as_configured", True
                )
            ):
                part_settings["configured_scale_prior"] = True
            old_scale = scales[part]
            (
                scales[part],
                anchors[part],
                final_pose_scale_reports[part],
            ) = calibrate_part_scale(
                cfg=config,
                part=part,
                mesh=meshes[part],
                raw_origin=origins[part],
                base_scale=old_scale,
                anchors=anchors[part],
                settings=part_settings,
                seed=int(scale_settings.get("seed", 8111)) + 300 + part_index,
            )
            for frame, transform in anchors[part].items():
                anchor_info[part][str(frame)]["S_world_from_raw"] = (
                    transform.tolist()
                )
                anchor_info[part][str(frame)]["fixed_scale"] = scales[part]
            selected_scale_factor = float(
                final_pose_scale_reports[part].get(
                    "selected_scale_factor", 1.0
                )
            )
            print(
                f"final-pose scale {part}: "
                f"factor={selected_scale_factor:.3f} "
                f"scale={scales[part]:.6f} "
                f"accepted={final_pose_scale_reports[part]['accepted']}",
                flush=True,
            )
    calibration = {
        "config": str(config_path),
        "coordinate_convention": (
            "world-to-camera extrinsics; column transforms"
        ),
        "point_cloud_root": str(cloud_root),
        "point_cloud_variant": config.get(
            "point_cloud_variant", config["recon_backend"]
        ),
        "depth_gauge_path": config.get("depth_gauge_path"),
        "scales": scales,
        "scale_consensus": scale_consensus,
        "raw_mesh_origins": {
            part: origins[part].tolist() for part in origins
        },
        "anchors": anchor_info,
        "appearance_refinement": appearance_reports,
        "anchor_render_refinement": anchor_render_reports,
        "post_anchor_render_scale_calibration": post_anchor_scale_reports,
        "anchor_render_refinement_post_scale": (
            anchor_render_post_scale_reports
        ),
        "final_pose_scale_calibration": final_pose_scale_reports,
        "stable_interval_anchor_selection": stable_interval_selection,
        "silhouette_scale_calibration": scale_reports,
        "table_alignment": table_alignment,
        "input_fingerprint": input_fingerprint,
    }
    return anchors, scales, calibration


def _load_calibration(path: Path) -> tuple[dict, dict, dict]:
    calibration = load_json(path)
    anchors = {
        part: {
            int(frame): np.asarray(value["S_world_from_raw"], dtype=float)
            for frame, value in values.items()
            if value.get(
                "usable_for_tracking", value.get("reliable", True)
            )
        }
        for part, values in calibration["anchors"].items()
    }
    scales = {
        part: float(value)
        for part, value in calibration["scales"].items()
    }
    return anchors, scales, calibration


def _solve_part(
    part: str,
    config: dict,
    mesh: trimesh.Trimesh,
    scale: float,
    origin: np.ndarray,
    anchors: dict[int, np.ndarray],
    cloud_root: Path,
    frame_range: tuple[int, int],
    prior: dict | None,
    visibility: dict[int, dict] | None = None,
) -> tuple[dict[int, np.ndarray], dict]:
    start_frame, end_frame = frame_range
    state = config["states"][part]
    method = state.get("method", "cloud_registration")
    poses = {}
    registrations = {}
    symmetry = symmetry_spec_from_state(state)
    geometric_symmetry = (
        symmetry if symmetry.equivalence != "none" else None
    )
    if method == "anchor_relative_registration":
        rigid_anchors = {
            frame: rigid_from_similarity(transform, origin)
            for frame, transform in anchors.items()
        }
        registration = dict(config["registration"])
        registration.update(state.get("anchor_relative_registration", {}))
        registration["state_static_ranges"] = state.get(
            "static_ranges", []
        )
        registration["state_dynamic_ranges"] = state.get(
            "dynamic_ranges", []
        )
        return track_anchor_relative_registration(
            part,
            mesh,
            scale,
            origin,
            start_frame,
            end_frame,
            rigid_anchors,
            cloud_root,
            registration,
            observable_frames={
                int(frame)
                for frame, row in (visibility or {}).items()
                if bool(
                    row.get("tracking_valid", row.get("pose_valid", True))
                )
            } if visibility is not None else None,
        )
    if method == "trajectory_prior":
        if prior is None:
            raise ValueError(f"{part}: trajectory_prior requires prior_trajectory")
        prior_frames = prior["frames"]
        last_key = sorted(prior_frames)[-1]
        for frame in range(start_frame, end_frame + 1):
            key = f"{frame:06d}"
            source_key = key if key in prior_frames else last_key
            transform = np.asarray(
                prior_frames[source_key]["S_world_from_inner_raw"], dtype=float
            )
            poses[frame] = rigid_from_similarity(transform, origin)
        registrations["prior"] = {
            "path": state["prior_trajectory"],
            "available_through": int(last_key),
        }
        return poses, registrations

    for dynamic_start, dynamic_end in state["dynamic_ranges"]:
        dynamic_start, dynamic_end = int(dynamic_start), int(dynamic_end)
        requested_dynamic_start, requested_dynamic_end = (
            dynamic_start,
            dynamic_end,
        )
        cloud_frames: list[int] | None = None
        if method == "cloud_registration":
            cloud_frames = [
                frame
                for frame in range(dynamic_start, dynamic_end + 1)
                if load_part_cloud(cloud_root, frame, part) is not None
            ]
            if cloud_frames:
                dynamic_start, dynamic_end = (
                    cloud_frames[0],
                    cloud_frames[-1],
                )
                if (dynamic_start, dynamic_end) != (
                    requested_dynamic_start,
                    requested_dynamic_end,
                ):
                    registrations[
                        f"{requested_dynamic_start:06d}-"
                        f"{requested_dynamic_end:06d}:endpoint_trim"
                    ] = {
                        "requested_range": [
                            requested_dynamic_start,
                            requested_dynamic_end,
                        ],
                        "tracked_range": [dynamic_start, dynamic_end],
                        "reason": "missing_quality_cloud_at_endpoint",
                    }
        anchor_frames = sorted(anchors)
        before = [frame for frame in anchor_frames if frame <= dynamic_start]
        after = [frame for frame in anchor_frames if frame >= dynamic_end]
        start_anchor = max(before) if before else None
        end_anchor = min(after) if after else None
        if start_anchor is None and end_anchor is None:
            raise RuntimeError(
                f"{part}: no reliable anchor brackets dynamic range "
                f"{dynamic_start}..{dynamic_end}"
            )
        start_pose = (
            None
            if start_anchor is None
            else rigid_from_similarity(anchors[start_anchor], origin)
        )
        end_pose = (
            None
            if end_anchor is None
            else rigid_from_similarity(anchors[end_anchor], origin)
        )
        if method == "cloud_registration" and not cloud_frames:
            if start_pose is None and end_pose is None:
                raise RuntimeError(
                    f"{part}: no cloud or reliable pose for dynamic range "
                    f"{requested_dynamic_start}..{requested_dynamic_end}"
                )
            count = max(requested_dynamic_end - requested_dynamic_start, 1)
            for frame in range(
                requested_dynamic_start, requested_dynamic_end + 1
            ):
                if start_pose is None:
                    pose = end_pose
                elif end_pose is None:
                    pose = start_pose
                else:
                    pose = interpolate_pose(
                        start_pose,
                        end_pose,
                        (frame - requested_dynamic_start) / count,
                    )
                poses[frame] = pose.copy()
            registrations[
                f"{requested_dynamic_start:06d}-"
                f"{requested_dynamic_end:06d}:no_cloud"
            ] = {"reason": "no_quality_cloud_in_dynamic_range"}
            continue
        if start_pose is None:
            if method == "mask_bbox_tracking":
                segment, segment_report = track_mask_bbox_translation(
                    part,
                    mesh,
                    scale,
                    origin,
                    dynamic_start,
                    dynamic_end,
                    None,
                    end_pose,
                    Path(config["masks_dir"]),
                    int(config["part_ids"][part]),
                    config["views"],
                    config,
                )
                poses.update(segment)
                registrations.update(segment_report)
                continue
            # Absolute silhouette trackers require a start pose.  When the
            # first visible frame failed anchor quality, recover that short
            # prefix from pairwise clouds in reverse from the reliable end.
            segment, segment_report = track_cloud_registration_reverse(
                part,
                dynamic_start,
                dynamic_end,
                end_pose,
                cloud_root,
                config["registration"],
                geometric_symmetry,
            )
            poses.update(segment)
            registrations.update(segment_report)
            if segment:
                first_pose = segment[min(segment)]
                last_pose = segment[max(segment)]
                for frame in range(
                    requested_dynamic_start, dynamic_start
                ):
                    poses.setdefault(frame, first_pose.copy())
                for frame in range(
                    dynamic_end + 1, requested_dynamic_end + 1
                ):
                    poses.setdefault(frame, last_pose.copy())
            continue
        if end_pose is None:
            raise RuntimeError(
                f"{part}: no reliable anchor after dynamic range "
                f"{dynamic_start}..{dynamic_end}"
            )
        if geometric_symmetry is not None and method in {
            "model_tracking", "mask_bbox_tracking"
        }:
            end_pose = resolve_symmetric_pose(
                end_pose,
                start_pose,
                geometric_symmetry,
                include_observation_ambiguities=False,
            ).pose
        if method == "model_tracking":
            segment, segment_report = track_model_translation(
                part,
                mesh,
                scale,
                origin,
                dynamic_start,
                dynamic_end,
                start_pose,
                end_pose,
                cloud_root,
                seed=700 + dynamic_start,
            )
        elif method == "mask_bbox_tracking":
            segment, segment_report = track_mask_bbox_translation(
                part,
                mesh,
                scale,
                origin,
                dynamic_start,
                dynamic_end,
                start_pose,
                end_pose,
                Path(config["masks_dir"]),
                int(config["part_ids"][part]),
                config["views"],
                config,
            )
        elif method == "cloud_registration":
            segment, segment_report = track_cloud_registration(
                part,
                dynamic_start,
                dynamic_end,
                start_pose,
                end_pose,
                cloud_root,
                config["registration"],
                geometric_symmetry,
            )
        else:
            raise ValueError(f"{part}: unknown tracking method {method!r}")
        if method == "cloud_registration" and segment:
            endpoint_key = (
                f"{dynamic_start:06d}-{dynamic_end:06d}:endpoint_correction"
            )
            endpoint_report = segment_report.get(endpoint_key, {})
            if (
                endpoint_report.get("rotation_rejected", False)
                and endpoint_report.get("propagate_to_anchor", True)
                and end_anchor is not None
            ):
                resolved_end = segment[max(segment)]
                anchors[end_anchor] = similarity_from_rigid(
                    resolved_end,
                    scale,
                    origin,
                )
                endpoint_report["propagated_to_anchor_frame"] = int(
                    end_anchor
                )
                endpoint_report["propagation_method"] = (
                    "replace_anchor_rotation_with_pairwise_tracked_branch"
                )
        poses.update(segment)
        registrations.update(segment_report)
        if method == "cloud_registration" and segment:
            first_pose = segment[min(segment)]
            last_pose = segment[max(segment)]
            for frame in range(requested_dynamic_start, dynamic_start):
                poses.setdefault(frame, first_pose.copy())
            for frame in range(dynamic_end + 1, requested_dynamic_end + 1):
                poses.setdefault(frame, last_pose.copy())

    for static_start, static_end in state["static_ranges"]:
        static_start, static_end = int(static_start), int(static_end)
        boundary = next(
            (
                frame
                for frame in (static_start, static_end)
                if frame in poses
            ),
            None,
        )
        if boundary is not None:
            static_pose = poses[boundary].copy()
        else:
            center = (static_start + static_end) / 2
            anchor = min(anchors, key=lambda value: abs(value - center))
            static_pose = rigid_from_similarity(anchors[anchor], origin)
        for frame in range(static_start, static_end + 1):
            poses.setdefault(frame, static_pose.copy())
    missing = sorted(
        set(range(start_frame, end_frame + 1)).difference(poses)
    )
    if missing:
        raise RuntimeError(f"{part}: uncovered frames {missing}")
    if geometric_symmetry is not None and config["registration"].get(
        "symmetry_lock", True
    ):
        for frame in range(start_frame + 1, end_frame + 1):
            poses[frame] = resolve_symmetric_pose(
                poses[frame],
                poses[frame - 1],
                geometric_symmetry,
                include_observation_ambiguities=False,
            ).pose
    return poses, registrations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--force-reuse-calibration", action="store_true")
    parser.add_argument(
        "--calibration",
        help="explicit calibration.json to reuse (useful for isolated reruns)",
    )
    parser.add_argument("--output-root", help="override config output_root")
    parser.add_argument(
        "--point-cloud-root",
        help="override config point_cloud_root for an isolated calibration run",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = validate_pose_config(load_json(config_path), check_paths=True)
    output = Path(args.output_root or config["output_root"])
    cloud_root = Path(
        args.point_cloud_root
        or config.get(
            "point_cloud_root",
            output / "parts_ply" / config["recon_backend"],
        )
    ).resolve()
    mesh_dir = Path(config["mesh_dir"])
    calibration_path = (
        Path(args.calibration).resolve()
        if args.calibration
        else output / "pose" / "calibration.json"
    )
    meshes = {
        part: trimesh.load(mesh_dir / f"{part}.glb", force="mesh")
        for part in config["parts"]
    }
    origins = {
        part: np.asarray(meshes[part].centroid, dtype=float)
        for part in config["parts"]
    }
    fingerprint = build_calibration_fingerprint(
        config, cloud_root=cloud_root, mesh_dir=mesh_dir
    )
    if args.calibration and not calibration_path.exists():
        raise FileNotFoundError(calibration_path)
    if (args.reuse_calibration or args.calibration) and calibration_path.exists():
        anchors, scales, calibration = _load_calibration(calibration_path)
        cached = calibration.get("input_fingerprint", {}).get("sha256")
        current = fingerprint["sha256"]
        if cached != current and not args.force_reuse_calibration:
            raise RuntimeError(
                "Calibration inputs changed or the cache predates "
                "fingerprinting. Run without --reuse-calibration, or pass "
                "--force-reuse-calibration intentionally. "
                f"cached={cached!r}, current={current!r}"
            )
        for part in config["parts"]:
            selected = config["states"][part].get("tracking_anchor_frames")
            if selected is None:
                continue
            selected_frames = {int(value) for value in selected}
            anchors[part] = {
                frame: transform
                for frame, transform in anchors[part].items()
                if frame in selected_frames
            }
            if not anchors[part]:
                raise RuntimeError(
                    f"{part}: tracking_anchor_frames selected no calibrated anchor"
                )
    else:
        anchors, scales, calibration = _fit_calibration(
            config,
            config_path,
            cloud_root,
            mesh_dir,
            meshes,
            origins,
            fingerprint,
        )
        write_json(calibration_path, calibration)

    start_frame = int(config["frames"]["start"])
    end_frame = int(config["frames"]["end"])
    priors = {}
    for part in config["parts"]:
        path = config["states"][part].get("prior_trajectory")
        if path:
            priors[part] = load_json(Path(path))
    body = config["reference_part"]
    moving_reference = bool(
        config.get("automation", {}).get("allow_moving_reference", False)
    )
    if config["states"][body].get("method") == "similarity_prior":
        prior = priors[body]
        scales[body] = float(prior["body_scale"])
        anchors[body] = {
            start_frame: np.asarray(
                prior["S_world_from_body_raw"], dtype=float
            )
        }
    for part in config["parts"]:
        if config["states"][part].get("method") == "trajectory_prior":
            scales[part] = float(priors[part]["inner_scale"])
    if args.calibrate_only:
        return

    mask_root = Path(config["masks_dir"])
    quality_summary_frames: dict = {}
    quality_paths = {
        str(
            Path(
                _view_quality_settings(config, part).get(
                    "quality_cloud_summary",
                    cloud_root / "quality_cloud_summary.json",
                )
            ).resolve()
        )
        for part in config["parts"]
        if _view_quality_settings(config, part).get(
            "use_cloud_supported_view_gate", False
        )
    }
    if len(quality_paths) > 1:
        raise ValueError(
            "pose visibility currently requires one shared "
            "quality_cloud_summary across parts"
        )
    if quality_paths:
        quality_path = Path(next(iter(quality_paths)))
        if not quality_path.exists():
            raise FileNotFoundError(
                "cloud-supported visibility was requested but its summary "
                f"does not exist: {quality_path}"
            )
        quality_summary_frames = load_json(quality_path).get("frames", {})
    temporal_area_references = {
        part: temporal_mask_area_references(
            mask_root,
            int(config["part_ids"][part]),
            list(config["views"]),
            (start_frame, end_frame),
            _view_quality_settings(config, part),
        )
        for part in config["parts"]
    }
    visibility_by_part: dict[str, dict[int, dict]] = {}
    for part in config["parts"]:
        settings = _view_quality_settings(config, part)
        visibility_by_part[part] = {
            frame: _part_visibility_report(
                mask_root,
                frame,
                int(config["part_ids"][part]),
                list(config["views"]),
                settings=settings,
                cloud_row=quality_summary_frames.get(
                    f"{frame:06d}", {}
                ).get(part),
                reference_areas=temporal_area_references[part],
            )
            for frame in range(start_frame, end_frame + 1)
        }
    write_json(
        output / "diagnostics" / "part_visibility.json",
        {
            "method": (
                "per_view_temporal_mask_and_cross_view_depth_support"
            ),
            "temporal_mask_area_references": temporal_area_references,
            "parts": {
                part: {
                    f"{frame:06d}": row
                    for frame, row in rows.items()
                }
                for part, rows in visibility_by_part.items()
            },
        },
    )

    world_poses = {}
    if not moving_reference:
        body_pose = rigid_from_similarity(
            next(iter(anchors[body].values())), origins[body]
        )
        world_poses[body] = {
            frame: body_pose.copy()
            for frame in range(start_frame, end_frame + 1)
        }
    registration_reports = {}
    solve_parts = [
        part
        for part in config["parts"]
        if not (part == body and not moving_reference)
    ]

    def solve_one(part: str) -> tuple[dict[int, np.ndarray], dict]:
        return _solve_part(
            part,
            config,
            meshes[part],
            scales[part],
            origins[part],
            anchors[part],
            cloud_root,
            (start_frame, end_frame),
            priors.get(part),
            visibility_by_part.get(part),
        )

    tracking_workers = max(
        1,
        int(config.get("registration", {}).get("parallel_parts", 1)),
    )
    if tracking_workers > 1 and len(solve_parts) > 1:
        workers = min(tracking_workers, len(solve_parts))
        print(
            f"tracking {len(solve_parts)} parts with {workers} workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {
                executor.submit(solve_one, part): part for part in solve_parts
            }
            for future in as_completed(pending):
                part = pending[future]
                poses, report = future.result()
                world_poses[part] = poses
                registration_reports[part] = report
                print(f"tracking complete: {part}", flush=True)
    else:
        for part in solve_parts:
            poses, report = solve_one(part)
            world_poses[part] = poses
            registration_reports[part] = report

    validation, failures = validate_world_poses(config, world_poses)
    write_json(output / "diagnostics/trajectory_validation.json", validation)
    if failures:
        raise RuntimeError("; ".join(failures))

    detected_path = output / "diagnostics" / "part_states.json"
    detected = (
        load_json(detected_path)["parts"] if detected_path.exists() else {}
    )
    trajectory = {
        "config": str(config_path),
        "provenance": {
            "recon_backend": config["recon_backend"],
            "point_cloud_root": str(cloud_root),
            "point_cloud_variant": config.get(
                "point_cloud_variant", config["recon_backend"]
            ),
            "depth_gauge_path": config.get("depth_gauge_path"),
            "timeline_fps": config.get("frames", {}).get("fps"),
            "timeline_policy": "all_configured_frames_no_pose_subsampling",
        },
        "conventions": {
            "T_world_from_part": (
                "rigid pose of the canonical part frame; "
                "origin is raw mesh centroid"
            ),
            "T_body_from_part": (
                "inv(T_world_from_body) @ T_world_from_part"
            ),
            "S_world_from_raw_mesh": (
                "render transform for raw GLB; includes fixed per-part scale"
            ),
            "quaternion": "xyzw",
        },
        "parts": config["parts"],
        "reference_part": body,
        "scales": scales,
        "raw_mesh_origins": {
            part: origins[part].tolist() for part in origins
        },
        "frames": {},
    }
    for frame in range(start_frame, end_frame + 1):
        key = f"{frame:06d}"
        trajectory["frames"][key] = {"parts": {}}
        for part in config["parts"]:
            visibility = visibility_by_part[part][frame]
            reliable_views = list(visibility["reliable_views"])
            mask_visible_views = list(visibility["mask_visible_views"])
            observing_views = len(mask_visible_views)
            tracking_valid = bool(visibility["tracking_valid"])
            pose_valid = bool(visibility["render_valid"])
            observation_state = str(visibility["observation_state"])
            state_config = config["states"][part]
            method = state_config.get("method", "cloud_registration")
            if not pose_valid:
                state, source = "out_of_frame", "multiview_mask_absence"
            elif not tracking_valid:
                state = observation_state
                source = "visible_pose_prediction_without_reliable_cloud"
            elif part == body and not moving_reference:
                state, source = "static", "body_anchor"
            else:
                moving = any(
                    int(start) <= frame <= int(end)
                    for start, end in state_config["dynamic_ranges"]
                )
                if moving:
                    sources = {
                        "trajectory_prior": "validated_trajectory_prior",
                        "model_tracking": "multiview_mesh_gated_tracking",
                        "mask_bbox_tracking": "multiview_silhouette_tracking",
                        "cloud_registration": "multiview_cloud_registration",
                        "anchor_relative_registration": (
                            "bidirectional_anchor_relative_multiview_registration"
                            if len(anchors[part]) >= 2
                            else "independent_anchor_relative_multiview_registration"
                        ),
                    }
                    state, source = "moving", sources[method]
                else:
                    state = "static"
                    source = (
                        "validated_trajectory_prior"
                        if method == "trajectory_prior"
                        else "static_anchor"
                    )
            record = world_pose_record(
                world_poses[part][frame],
                state=state,
                source=source,
                observing_views=observing_views,
            )
            record["pose_valid"] = pose_valid
            record["tracking_valid"] = tracking_valid
            record["observation_state"] = observation_state
            record["visible_views"] = mask_visible_views
            record["reliable_views"] = reliable_views
            record["tracking_observing_views"] = len(reliable_views)
            record["visibility_source"] = (
                "per_view_temporal_mask_and_cross_view_depth_support"
                if visibility["require_cloud_support"]
                else "per_view_temporal_mask_quality"
            )
            detected_entry = (
                detected.get(part, {}).get("states", {}).get(key)
            )
            if detected_entry:
                record["detected_state"] = detected_entry["state"]
            trajectory["frames"][key]["parts"][part] = record
    refresh_trajectory_derived_fields(trajectory)

    pose_dir = output / "pose"
    write_trajectory_files(trajectory, pose_dir / "trajectory.json")
    write_json(pose_dir / "pair_registrations.json", registration_reports)
    print(f"wrote {pose_dir / 'trajectory.json'}", flush=True)


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/stages/pose/solve_multiview_pose.py")
    main()
