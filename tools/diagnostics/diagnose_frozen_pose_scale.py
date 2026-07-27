#!/usr/bin/env python3
"""Scan per-part uniform scale while keeping every observed SE(3) pose fixed."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.normalized_recon import load_recon
from common.pose_refinement import sample_canonical
from common.pose_visualization import camera_from_recon
from common.render_loss_refinement import (
    MultiViewRenderObjective,
    RenderObservation,
)
from common.scale_diagnostics import (
    aggregate_visual_metrics,
    pareto_indices,
)
from common.simulation_assets import create_collision_proxy
from common.trajectory_constraints import (
    CylindricalContainer,
    SampledSurface,
    evaluate_insertion_trajectory,
    evaluate_surface_contact_trajectory,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_surface(
    mesh: trimesh.Trimesh, count: int, seed: int
) -> SampledSurface:
    points, face_indices = trimesh.sample.sample_surface(
        mesh, max(int(count), 1000), seed=int(seed)
    )
    return SampledSurface(
        np.asarray(points, dtype=np.float64),
        np.asarray(mesh.face_normals, dtype=np.float64)[face_indices],
    )


def scaled_surface(surface: SampledSurface, factor: float) -> SampledSurface:
    return SampledSurface(
        np.asarray(surface.points, dtype=np.float64) * float(factor),
        np.asarray(surface.normals, dtype=np.float64),
    )


def load_observations(
    pose_cfg: dict[str, Any],
    timestamp: str,
    part: str,
    settings: dict[str, Any],
) -> list[RenderObservation]:
    width, height = [
        int(value) for value in settings.get("resolution", [160, 90])
    ]
    part_id = int(pose_cfg["part_ids"][part])
    minimum_pixels = int(settings.get("min_mask_pixels", 30))
    recon = load_recon(
        pose_cfg, timestamp, backend=pose_cfg["recon_backend"]
    )
    observations: list[RenderObservation] = []
    for view_index, view in enumerate(pose_cfg["views"]):
        labels = np.asarray(
            Image.open(
                Path(pose_cfg["masks_dir"]) / timestamp / f"{view}.png"
            )
        )
        target = cv2.resize(
            (labels == part_id).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if int(target.sum()) < minimum_pixels:
            continue
        intrinsics, extrinsics = camera_from_recon(
            recon, view_index, (height, width)
        )
        observed_depth = cv2.resize(
            recon["depth"][view_index].astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        observations.append(
            RenderObservation(
                view=str(view),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                target_mask=target,
                observed_depth=observed_depth,
            )
        )
    return observations


def frame_ids(
    relation: dict[str, Any],
    key: str,
    fallback: str,
) -> list[int]:
    if key in relation:
        return sorted({int(value) for value in relation[key]})
    start, end = map(int, relation[fallback])
    return list(range(start, end + 1))


def compact_render(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "loss",
            "mean_iou",
            "mean_contour_chamfer_px",
            "mean_target_coverage",
            "views",
        )
    }


def evaluate_visual(
    trajectory: dict[str, Any],
    part: str,
    points: np.ndarray,
    observations: dict[int, list[RenderObservation]],
    render: dict[str, Any],
    optimize_views: list[str],
    holdout_views: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    optimize_rows = []
    holdout_rows = []
    frames: dict[str, Any] = {}
    for frame, frame_observations in observations.items():
        pose = np.asarray(
            trajectory["frames"][f"{frame:06d}"]["parts"][part][
                "T_world_from_part"
            ],
            dtype=np.float64,
        )
        objective = MultiViewRenderObjective(
            points, frame_observations, render
        )
        optimize = objective.evaluate(pose, optimize_views)
        holdout = objective.evaluate(pose, holdout_views)
        optimize_rows.append(optimize)
        if holdout["views"]:
            holdout_rows.append(holdout)
        frames[f"{frame:06d}"] = {
            "optimize": compact_render(optimize),
            "holdout": compact_render(holdout),
        }
    return (
        aggregate_visual_metrics(optimize_rows),
        aggregate_visual_metrics(holdout_rows),
        frames,
    )


def relative_poses(
    trajectory: dict[str, Any],
    reference_part: str,
    moving_part: str,
    frames: list[int],
) -> dict[int, np.ndarray]:
    values = {}
    for frame in frames:
        records = trajectory["frames"][f"{frame:06d}"]["parts"]
        reference = np.asarray(
            records[reference_part]["T_world_from_part"], dtype=np.float64
        )
        moving = np.asarray(
            records[moving_part]["T_world_from_part"], dtype=np.float64
        )
        values[frame] = np.linalg.inv(reference) @ moving
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--relations", nargs="+")
    args = parser.parse_args()

    config_path = resolve_path(args.config).resolve()
    config = load_json(config_path)
    pose_config_path = resolve_path(config["pose_config"]).resolve()
    trajectory_path = resolve_path(config["trajectory"]).resolve()
    proxy_config_path = resolve_path(config["geometry_proxy_config"]).resolve()
    pose_cfg = load_json(pose_config_path)
    trajectory = load_json(trajectory_path)
    proxy_cfg = load_json(proxy_config_path)
    proxies = proxy_cfg["collision_proxies"]
    report_path = (
        resolve_path(args.report)
        if args.report
        else resolve_path(config["output_report"])
    ).resolve()

    render = dict(config.get("render_objective", {}))
    optimize_views = [
        str(value)
        for value in render.get("optimize_views", pose_cfg["views"])
    ]
    holdout_views = [
        str(value) for value in render.get("holdout_views", [])
    ]
    requested = set(args.relations or [])
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "frozen_observed_pose_uniform_scale_pareto",
        "trajectory_mutated": False,
        "pose_semantics": {
            "observed_pose": "immutable input T_world_from_part",
            "candidate_change": (
                "uniform geometry scale about the existing part-frame origin"
            ),
            "physics_pose_projection": "not produced",
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "pose_config": str(pose_config_path),
            "trajectory": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "geometry_proxy_config": str(proxy_config_path),
        },
        "render_objective": render,
        "relations": {},
    }

    for relation_index, relation in enumerate(config["relations"]):
        name = str(relation["name"])
        if requested and name not in requested:
            continue
        relation_type = str(relation.get("type", "pairwise_contact"))
        reference_part = str(
            relation.get("reference_part", relation.get("container", ""))
        )
        moving_part = str(relation["moving_part"])
        factors = sorted(
            {float(value) for value in relation["moving_scale_factors"]}
        )
        if not factors or any(value <= 0.0 for value in factors):
            raise ValueError(f"{name}: moving_scale_factors must be positive")
        visual_frames = frame_ids(
            relation, "visual_frames", "frame_range"
        )
        geometry_frames = frame_ids(
            relation, "geometry_frames", "frame_range"
        )
        missing = [
            frame
            for frame in sorted(set(visual_frames + geometry_frames))
            if f"{frame:06d}" not in trajectory["frames"]
        ]
        if missing:
            raise ValueError(f"{name}: frames absent from trajectory: {missing}")

        raw_mesh = trimesh.load(
            Path(pose_cfg["mesh_dir"]) / f"{moving_part}.glb",
            force="mesh",
        )
        base_scale = float(trajectory["scales"][moving_part])
        base_points = sample_canonical(
            raw_mesh,
            base_scale,
            np.asarray(
                trajectory["raw_mesh_origins"][moving_part],
                dtype=np.float64,
            ),
            count=int(render.get("surface_points", 30000)),
            seed=int(render.get("seed", 9107)) + relation_index,
        )
        observations = {
            frame: load_observations(
                pose_cfg, f"{frame:06d}", moving_part, render
            )
            for frame in visual_frames
        }
        relative = relative_poses(
            trajectory,
            reference_part,
            moving_part,
            geometry_frames,
        )
        dummy = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        reference_proxy, reference_proxy_info = create_collision_proxy(
            dummy, proxies[reference_part]
        )
        moving_proxy, moving_proxy_info = create_collision_proxy(
            dummy, proxies[moving_part]
        )
        surface_count = int(relation.get("surface_points", 6000))
        seed = int(relation.get("seed", 7127))
        moving_surface = sample_surface(
            moving_proxy, surface_count, seed + 1
        )
        reference_surface = sample_surface(
            reference_proxy, surface_count, seed
        )

        rows: list[dict[str, Any]] = []
        for factor in factors:
            optimize, holdout, visual_details = evaluate_visual(
                trajectory,
                moving_part,
                base_points * factor,
                observations,
                render,
                optimize_views,
                holdout_views,
            )
            if relation_type == "insert_into":
                container = CylindricalContainer.from_spec(
                    proxies[reference_part]
                )
                geometry = evaluate_insertion_trajectory(
                    relative,
                    moving_surface.points * factor,
                    container,
                    substeps=int(relation.get("continuous_substeps", 1)),
                    entry_center_radius_m=float(
                        relation["entry_center_radius_m"]
                    ),
                    contact_tolerance_m=float(
                        relation.get("contact_tolerance_m", 0.001)
                    ),
                    entry_frame=int(
                        relation.get(
                            "entry_frame", min(geometry_frames)
                        )
                    ),
                )
            elif relation_type == "pairwise_contact":
                geometry = evaluate_surface_contact_trajectory(
                    relative,
                    scaled_surface(moving_surface, factor),
                    reference_surface,
                    relation,
                )
            else:
                raise ValueError(
                    f"{name}: unsupported relation type {relation_type!r}"
                )
            row = {
                "scale_factor": factor,
                "absolute_scale": base_scale * factor,
                "visual_loss": optimize["visual_loss"],
                "visual_mean_iou": optimize["mean_iou"],
                "visual_mean_contour_chamfer_px": optimize[
                    "mean_contour_chamfer_px"
                ],
                "holdout_visual_loss": holdout["visual_loss"],
                "holdout_mean_iou": holdout["mean_iou"],
                "max_penetration_m": float(
                    geometry["max_penetration_m"]
                ),
                "max_contact_gap_violation_m": float(
                    geometry.get("max_contact_gap_violation_m", 0.0)
                ),
                "rms_penetration_m": float(
                    geometry.get(
                        "rms_penetration_m",
                        np.sqrt(
                            np.mean(
                                np.square(
                                    [
                                        sample["rms_penetration_m"]
                                        for sample in geometry.get(
                                            "samples", []
                                        )
                                    ]
                                )
                            )
                        )
                        if geometry.get("samples")
                        else 0.0,
                    )
                ),
                "visual": {
                    "optimize": optimize,
                    "holdout": holdout,
                    "frames": visual_details,
                },
                "geometry": geometry,
            }
            row["physical_violation_m"] = max(
                row["max_penetration_m"],
                row["max_contact_gap_violation_m"],
            )
            rows.append(row)
            print(
                f"{name} scale={factor:.4f}: "
                f"visual={row['visual_loss']:.5f} "
                f"holdout={row['holdout_visual_loss']:.5f} "
                f"penetration={1000.0 * row['max_penetration_m']:.2f}mm",
                flush=True,
            )

        baseline_index = min(
            range(len(rows)),
            key=lambda index: abs(rows[index]["scale_factor"] - 1.0),
        )
        baseline = rows[baseline_index]
        optimize_limit = float(
            relation.get("maximum_optimize_loss_degradation", 0.03)
        )
        holdout_limit = float(
            relation.get("maximum_holdout_loss_degradation", 0.035)
        )
        eligible = [
            index
            for index, row in enumerate(rows)
            if (
                row["visual_loss"]
                <= baseline["visual_loss"] + optimize_limit + 1e-12
                and (
                    not np.isfinite(row["holdout_visual_loss"])
                    or not np.isfinite(baseline["holdout_visual_loss"])
                    or row["holdout_visual_loss"]
                    <= baseline["holdout_visual_loss"]
                    + holdout_limit
                    + 1e-12
                )
            )
        ]
        selected_index = min(
            eligible or [baseline_index],
            key=lambda index: (
                rows[index]["physical_violation_m"],
                rows[index]["visual_loss"],
                abs(rows[index]["scale_factor"] - 1.0),
            ),
        )
        pareto = pareto_indices(
            rows, ("visual_loss", "physical_violation_m")
        )
        relation_report = {
            "type": relation_type,
            "reference_part": reference_part,
            "moving_part": moving_part,
            "base_absolute_scale": base_scale,
            "visual_frames": visual_frames,
            "geometry_frames": geometry_frames,
            "reference_scale_factor": 1.0,
            "reference_proxy": reference_proxy_info,
            "moving_proxy": moving_proxy_info,
            "isaac_acceptance": {
                "maximum_cpu_physical_violation_m": float(
                    relation.get(
                        "isaac_acceptance", {}
                    ).get("maximum_cpu_physical_violation_m", 0.002)
                ),
                "maximum_isaac_penetration_m": float(
                    relation.get(
                        "isaac_acceptance", {}
                    ).get("maximum_isaac_penetration_m", 0.002)
                ),
                "maximum_isaac_positive_separation_m": float(
                    relation.get(
                        "isaac_acceptance", {}
                    ).get(
                        "maximum_isaac_positive_separation_m", 0.004
                    )
                ),
                "require_physx_contact": bool(
                    relation.get(
                        "isaac_acceptance", {}
                    ).get("require_physx_contact", True)
                ),
            },
            "selection": {
                "policy": (
                    "minimum penetration inside independent optimize/holdout "
                    "visual-loss gates; pose remains frozen"
                ),
                "baseline_index": baseline_index,
                "eligible_indices": eligible,
                "pareto_indices": pareto,
                "selected_index": selected_index,
                "selected_scale_factor": rows[selected_index][
                    "scale_factor"
                ],
                "selected_absolute_scale": rows[selected_index][
                    "absolute_scale"
                ],
                "maximum_optimize_loss_degradation": optimize_limit,
                "maximum_holdout_loss_degradation": holdout_limit,
                "requires_isaac_confirmation": True,
            },
            "candidates": rows,
        }
        report["relations"][name] = relation_report

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(f"Frozen-pose scale report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
