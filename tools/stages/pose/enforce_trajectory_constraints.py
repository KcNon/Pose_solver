#!/usr/bin/env python
"""Enforce continuous, configuration-driven geometric trajectory constraints."""
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
from common.normalized_recon import load_recon
from common.pose_config import validate_pose_config
from common.pose_refinement import sample_canonical
from common.pose_validation import validate_trajectory
from common.pose_visualization import camera_from_recon
from common.render_loss_refinement import (
    MultiViewRenderObjective,
    RenderObservation,
)
from common.simulation_assets import create_collision_proxy
from common.trajectory_constraints import (
    CylindricalContainer,
    SampledSurface,
    evaluate_insertion_trajectory,
    evaluate_surface_contact_trajectory,
    interpolate_pose,
    pose_delta,
    refine_insert_trajectory,
    refine_surface_contact_trajectory,
)
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    write_trajectory_files,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sample_proxy_points(
    proxy: trimesh.Trimesh, count: int, seed: int
) -> np.ndarray:
    values = [
        np.asarray(proxy.vertices, dtype=np.float64),
        np.asarray(proxy.triangles_center, dtype=np.float64),
    ]
    if count > 0:
        sampled, _ = trimesh.sample.sample_surface(
            proxy, int(count), seed=int(seed)
        )
        values.append(np.asarray(sampled, dtype=np.float64))
    return np.vstack(values)


def sample_proxy_surface(
    proxy: trimesh.Trimesh, count: int, seed: int
) -> SampledSurface:
    points, face_indices = trimesh.sample.sample_surface(
        proxy, max(int(count), 1000), seed=int(seed)
    )
    normals = np.asarray(proxy.face_normals, dtype=np.float64)[face_indices]
    return SampledSurface(points, normals)


def load_render_observations(
    cfg: dict[str, Any],
    timestamp: str,
    part: str,
    gate: dict[str, Any],
) -> list[RenderObservation]:
    width, height = [int(value) for value in gate.get("resolution", [160, 90])]
    part_id = int(cfg["part_ids"][part])
    minimum_pixels = int(gate.get("min_mask_pixels", 30))
    recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
    observations = []
    for view_index, view in enumerate(cfg["views"]):
        mask_path = Path(cfg["masks_dir"]) / timestamp / f"{view}.png"
        labels = np.asarray(Image.open(mask_path))
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
                view=view,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                target_mask=target,
                observed_depth=observed_depth,
            )
        )
    return observations


def compact_render_metric(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "loss",
            "mean_iou",
            "mean_contour_chamfer_px",
            "mean_target_coverage",
        )
    }


def render_gate_pose(
    cfg: dict[str, Any],
    part: str,
    frame: int,
    initial_world: np.ndarray,
    proposed_world: np.ndarray,
    canonical_points: np.ndarray,
    gate: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Line-search toward a feasible pose while bounding reprojection damage."""
    if not gate.get("enabled", True):
        return proposed_world, {"enabled": False, "accepted_alpha": 1.0}
    observations = load_render_observations(
        cfg, f"{frame:06d}", part, gate
    )
    optimize_views = [
        str(value) for value in gate.get("optimize_views", cfg["views"])
    ]
    holdout_views = [str(value) for value in gate.get("holdout_views", [])]
    available = {item.view for item in observations}
    effective_optimize = [
        view for view in optimize_views if view in available
    ]
    effective_holdout = [
        view for view in holdout_views if view in available
    ]
    if len(effective_optimize) < int(gate.get("minimum_optimize_views", 3)):
        return proposed_world, {
            "enabled": True,
            "skipped": True,
            "reason": "insufficient_views",
            "available_views": sorted(available),
            "accepted_alpha": 1.0,
        }
    objective = MultiViewRenderObjective(
        canonical_points, observations, gate
    )
    baseline_opt = objective.evaluate(initial_world, effective_optimize)
    baseline_holdout = (
        objective.evaluate(initial_world, effective_holdout)
        if effective_holdout
        else {"loss": 0.0, "mean_iou": None}
    )
    maximum_optimize = float(
        gate.get("maximum_optimize_degradation", 0.03)
    )
    maximum_holdout = float(
        gate.get("maximum_holdout_degradation", 0.02)
    )
    trials = []
    selected = initial_world
    accepted_alpha = 0.0
    selected_opt = baseline_opt
    selected_holdout = baseline_holdout
    for alpha in [
        float(value) for value in gate.get("line_search_alphas", [1.0, 0.75, 0.5, 0.25])
    ]:
        candidate = interpolate_pose(initial_world, proposed_world, alpha)
        candidate_opt = objective.evaluate(candidate, effective_optimize)
        candidate_holdout = (
            objective.evaluate(candidate, effective_holdout)
            if effective_holdout
            else baseline_holdout
        )
        optimize_degradation = float(
            candidate_opt["loss"] - baseline_opt["loss"]
        )
        holdout_degradation = float(
            candidate_holdout["loss"] - baseline_holdout["loss"]
        )
        accepted = bool(
            optimize_degradation <= maximum_optimize
            and (
                not effective_holdout
                or holdout_degradation <= maximum_holdout
            )
        )
        trials.append(
            {
                "alpha": alpha,
                "accepted": accepted,
                "optimize_loss_degradation": optimize_degradation,
                "holdout_loss_degradation": holdout_degradation,
            }
        )
        if accepted:
            selected = candidate
            accepted_alpha = alpha
            selected_opt = candidate_opt
            selected_holdout = candidate_holdout
            break
    return selected, {
        "enabled": True,
        "skipped": False,
        "available_views": sorted(available),
        "accepted_alpha": accepted_alpha,
        "baseline_optimize": compact_render_metric(baseline_opt),
        "selected_optimize": compact_render_metric(selected_opt),
        "baseline_holdout": compact_render_metric(baseline_holdout),
        "selected_holdout": compact_render_metric(selected_holdout),
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-trajectory", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--relations", nargs="+", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    trajectory_path = Path(args.trajectory).resolve()
    output_path = Path(args.output_trajectory).resolve()
    report_path = (
        Path(args.report).resolve()
        if args.report
        else output_path.parents[1]
        / "diagnostics"
        / "trajectory_constraints.json"
    )
    cfg = validate_pose_config(load_json(config_path), check_paths=True)
    settings = dict(cfg.get("trajectory_constraints", {}))
    if not settings.get("enabled", False):
        raise ValueError("trajectory_constraints.enabled must be true")
    geometry_config_path = resolve_project_path(
        settings.get(
            "geometry_proxy_config",
            settings.get("collision_proxy_config"),
        )
    )
    geometry_config = load_json(geometry_config_path)
    proxies = geometry_config["collision_proxies"]
    baseline = load_json(trajectory_path)
    trajectory = copy.deepcopy(baseline)

    requested = set(args.relations or [])
    relations = [
        relation
        for relation in settings.get("relations", [])
        if not requested or str(relation["name"]) in requested
    ]
    if requested.difference(str(item["name"]) for item in relations):
        raise ValueError(
            f"unknown relations: {sorted(requested.difference(str(item['name']) for item in relations))}"
        )
    report: dict[str, Any] = {
        "schema_version": 2,
        "method": "continuous_pairwise_geometric_constraints",
        "config": str(config_path),
        "trajectory_input": str(trajectory_path),
        "trajectory_input_sha256": sha256_file(trajectory_path),
        "geometry_proxy_config": str(geometry_config_path),
        "relations": {},
    }

    for relation_index, relation in enumerate(relations):
        # A relation is a transaction.  Approximate proxies or incompatible
        # image/geometric evidence must never leave a partially corrected
        # trajectory behind when that relation ultimately fails validation.
        relation_input_trajectory = copy.deepcopy(trajectory)
        relation_type = str(relation.get("type", "insert_into"))
        if relation_type not in {"insert_into", "pairwise_contact"}:
            raise ValueError(
                f"unsupported trajectory constraint type: {relation.get('type')}"
            )
        name = str(relation["name"])
        reference_part = str(
            relation.get("container", relation.get("reference_part", ""))
        )
        moving_part = str(relation["moving_part"])
        unknown = {reference_part, moving_part}.difference(trajectory["parts"])
        if unknown:
            raise ValueError(f"{name}: unknown parts {sorted(unknown)}")
        start, end = map(int, relation["frame_range"])
        frames = [
            frame
            for frame in range(start, end + 1)
            if f"{frame:06d}" in trajectory["frames"]
        ]
        if len(frames) < 2:
            raise ValueError(f"{name}: frame range contains fewer than two poses")
        default_seed = int(settings.get("seed", 3191))
        stable_offset = int(
            hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16
        ) % 1_000_000
        relation_seed = int(
            relation.get("seed", default_seed + stable_offset)
        )

        dummy = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        reference_proxy, reference_proxy_info = create_collision_proxy(
            dummy, proxies[reference_part]
        )
        moving_proxy, moving_proxy_info = create_collision_proxy(
            dummy, proxies[moving_part]
        )
        relative_initial = {}
        for frame in frames:
            records = trajectory["frames"][f"{frame:06d}"]["parts"]
            reference_world = np.asarray(
                records[reference_part]["T_world_from_part"],
                dtype=np.float64,
            )
            moving_world = np.asarray(
                records[moving_part]["T_world_from_part"],
                dtype=np.float64,
            )
            relative_initial[frame] = (
                np.linalg.inv(reference_world) @ moving_world
            )

        geometry_metadata: dict[str, Any]
        if relation_type == "insert_into":
            points = sample_proxy_points(
                moving_proxy,
                int(relation.get("surface_points", 2000)),
                relation_seed,
            )
            container = CylindricalContainer.from_spec(proxies[reference_part])
            proposed, optimization = refine_insert_trajectory(
                relative_initial, points, container, relation
            )

            def evaluate_selected(
                values: dict[int, np.ndarray],
            ) -> dict[str, Any]:
                return evaluate_insertion_trajectory(
                    values,
                    points,
                    container,
                    substeps=int(relation.get("continuous_substeps", 8)),
                    entry_center_radius_m=float(
                        relation["entry_center_radius_m"]
                    ),
                    contact_tolerance_m=float(
                        relation.get("contact_tolerance_m", 0.001)
                    ),
                    entry_frame=optimization["entry_frame"],
                )

            geometry_metadata = {
                "reference_proxy": proxies[reference_part],
                "moving_proxy": moving_proxy_info,
                "collision_sample_points": int(len(points)),
                "entry_frame": optimization["entry_frame"],
            }
        else:
            surface_count = int(relation.get("surface_points", 6000))
            reference_surface = sample_proxy_surface(
                reference_proxy,
                surface_count,
                relation_seed,
            )
            moving_surface = sample_proxy_surface(
                moving_proxy,
                surface_count,
                relation_seed + 1,
            )
            proposed, optimization = refine_surface_contact_trajectory(
                relative_initial,
                moving_surface,
                reference_surface,
                relation,
            )

            def evaluate_selected(
                values: dict[int, np.ndarray],
            ) -> dict[str, Any]:
                return evaluate_surface_contact_trajectory(
                    values,
                    moving_surface,
                    reference_surface,
                    relation,
                )

            geometry_metadata = {
                "reference_proxy": reference_proxy_info,
                "moving_proxy": moving_proxy_info,
                "reference_surface_points": int(
                    len(reference_surface.points)
                ),
                "moving_surface_points": int(len(moving_surface.points)),
            }
        gate = dict(settings.get("render_gate", {}))
        gate.update(relation.get("render_gate", {}))
        raw_mesh = trimesh.load(
            Path(cfg["mesh_dir"]) / f"{moving_part}.glb", force="mesh"
        )
        canonical_points = sample_canonical(
            raw_mesh,
            float(baseline["scales"][moving_part]),
            np.asarray(
                baseline["raw_mesh_origins"][moving_part], dtype=np.float64
            ),
            count=int(gate.get("surface_points", 30000)),
            seed=relation_seed + 100,
        )
        selected_relative = {}
        visual_gates = {}
        for frame in frames:
            key = f"{frame:06d}"
            reference_world = np.asarray(
                trajectory["frames"][key]["parts"][reference_part][
                    "T_world_from_part"
                ],
                dtype=np.float64,
            )
            initial_world = reference_world @ relative_initial[frame]
            proposed_world = reference_world @ proposed[frame]
            selected_world, gate_report = render_gate_pose(
                cfg,
                moving_part,
                frame,
                initial_world,
                proposed_world,
                canonical_points,
                gate,
            )
            selected_relative[frame] = (
                np.linalg.inv(reference_world) @ selected_world
            )
            visual_gates[key] = gate_report
            delta = pose_delta(initial_world, selected_world)
            if np.linalg.norm(delta) > 1e-10:
                record = trajectory["frames"][key]["parts"][moving_part]
                record["T_world_from_part"] = selected_world.tolist()
                record["source"] = (
                    str(record.get("source", "pose"))
                    + "+continuous_geometric_constraint"
                )

        terminal_delta = (
            selected_relative[frames[-1]]
            @ np.linalg.inv(relative_initial[frames[-1]])
        )
        propagate_end = int(relation.get("propagate_through_frame", end))
        propagation_gate = dict(gate)
        propagation_gate.update(
            relation.get("propagation_render_gate", {})
        )
        propagation_gate_frames = [
            int(value)
            for value in propagation_gate.get("frames", [])
            if end < int(value) <= propagate_end
            and f"{int(value):06d}" in trajectory["frames"]
        ]
        propagation_gate_reports = {}
        propagation_alpha = 1.0
        if propagation_gate.get("enabled", False):
            for frame in propagation_gate_frames:
                key = f"{frame:06d}"
                records = trajectory["frames"][key]["parts"]
                reference_world = np.asarray(
                    records[reference_part]["T_world_from_part"],
                    dtype=np.float64,
                )
                initial_world = np.asarray(
                    records[moving_part]["T_world_from_part"],
                    dtype=np.float64,
                )
                initial_relative = (
                    np.linalg.inv(reference_world) @ initial_world
                )
                proposed_world = (
                    reference_world @ terminal_delta @ initial_relative
                )
                _, item = render_gate_pose(
                    cfg,
                    moving_part,
                    frame,
                    initial_world,
                    proposed_world,
                    canonical_points,
                    propagation_gate,
                )
                propagation_gate_reports[key] = item
                propagation_alpha = min(
                    propagation_alpha,
                    float(item.get("accepted_alpha", 1.0)),
                )
        if propagation_alpha < 1.0:
            terminal_delta = interpolate_pose(
                np.eye(4, dtype=np.float64),
                terminal_delta,
                propagation_alpha,
            )
            final_frame = frames[-1]
            selected_relative[final_frame] = (
                terminal_delta @ relative_initial[final_frame]
            )
            key = f"{final_frame:06d}"
            reference_world = np.asarray(
                trajectory["frames"][key]["parts"][reference_part][
                    "T_world_from_part"
                ],
                dtype=np.float64,
            )
            trajectory["frames"][key]["parts"][moving_part][
                "T_world_from_part"
            ] = (
                reference_world @ selected_relative[final_frame]
            ).tolist()

        after = evaluate_selected(selected_relative)
        validation_start, validation_end = map(
            int, relation.get("validation_frame_range", [start, end])
        )
        validation_relative = {
            frame: value
            for frame, value in selected_relative.items()
            if validation_start <= frame <= validation_end
        }
        if not validation_relative:
            raise ValueError(
                f"{name}: validation_frame_range contains no evaluated poses"
            )
        validation_after = evaluate_selected(validation_relative)
        propagated = []
        for frame in range(end + 1, propagate_end + 1):
            key = f"{frame:06d}"
            if key not in trajectory["frames"]:
                continue
            records = trajectory["frames"][key]["parts"]
            reference_world = np.asarray(
                records[reference_part]["T_world_from_part"], dtype=np.float64
            )
            initial_world = np.asarray(
                records[moving_part]["T_world_from_part"], dtype=np.float64
            )
            initial_relative = np.linalg.inv(reference_world) @ initial_world
            selected_world = (
                reference_world @ terminal_delta @ initial_relative
            )
            delta = pose_delta(initial_world, selected_world)
            if np.linalg.norm(delta) <= 1e-10:
                continue
            record = trajectory["frames"][key]["parts"][moving_part]
            record["T_world_from_part"] = selected_world.tolist()
            record["source"] = (
                str(record.get("source", "pose"))
                + "+continuous_geometric_constraint_propagated"
            )
            propagated.append(frame)

        selected_corrections = {}
        for frame in frames:
            delta = pose_delta(relative_initial[frame], selected_relative[frame])
            selected_corrections[f"{frame:06d}"] = {
                "translation_delta_m": delta[:3].tolist(),
                "translation_delta_norm_m": float(np.linalg.norm(delta[:3])),
                "rotation_delta_deg": float(
                    np.degrees(np.linalg.norm(delta[3:]))
                ),
            }
        limit = float(
            relation.get("maximum_allowed_penetration_m", 0.001)
        )
        relation_passed = validation_after["max_penetration_m"] <= limit
        if relation_type == "pairwise_contact":
            relation_passed = bool(
                relation_passed
                and validation_after["max_contact_gap_violation_m"] <= 1e-9
                and validation_after["max_axis_angle_violation_deg"] <= 1e-9
                and validation_after["max_axis_offset_violation_m"] <= 1e-9
            )
        required = bool(relation.get("required", True))
        apply_on_failure = bool(relation.get("apply_on_failure", False))
        applied = bool(relation_passed or apply_on_failure)
        if not applied:
            trajectory = relation_input_trajectory
        report["relations"][name] = {
            "type": relation_type,
            "reference_part": reference_part,
            "moving_part": moving_part,
            "frame_range": [start, end],
            "propagated_frames": propagated,
            **geometry_metadata,
            "optimizer_evaluations": optimization["evaluations"],
            "before": optimization["before"],
            "proposed_after": optimization["proposed_after"],
            "after_render_gate": after,
            "validation_frame_range": [
                validation_start,
                validation_end,
            ],
            "validation_after_render_gate": validation_after,
            "selected_corrections": selected_corrections,
            "render_gates": visual_gates,
            "propagation_render_gate": {
                "enabled": bool(propagation_gate.get("enabled", False)),
                "frames": propagation_gate_frames,
                "accepted_alpha": propagation_alpha,
                "frame_reports": propagation_gate_reports,
            },
            "maximum_allowed_penetration_m": limit,
            "required": required,
            "applied": applied,
            "rollback_reason": (
                None
                if applied
                else "relation validation failed"
            ),
            "passed": relation_passed,
        }
        print(
            f"{name} [{relation_type}]: penetration "
            f"{1000.0 * optimization['before']['max_penetration_m']:.2f}mm -> "
            f"{1000.0 * after['max_penetration_m']:.2f}mm; "
            f"violating samples {optimization['before']['violating_samples']} -> "
            f"{after['violating_samples']}",
            flush=True,
        )

    refresh_trajectory_derived_fields(trajectory)
    validation, validation_failures = validate_trajectory(cfg, trajectory)
    relation_failures = [
        f"{name}: unresolved continuous penetration"
        for name, value in report["relations"].items()
        if value["required"] and not value["passed"]
    ]
    failures = validation_failures + relation_failures
    report["trajectory_validation"] = validation
    report["summary"] = {
        "relations": len(report["relations"]),
        "passed_relations": sum(
            int(value["passed"]) for value in report["relations"].values()
        ),
        "validation_passed": not failures,
        "failures": failures,
    }
    trajectory.setdefault("refinements", []).append(
        {
            "method": report["method"],
            "input": str(trajectory_path),
            "report": str(report_path),
        }
    )
    write_trajectory_files(trajectory, output_path)
    report["trajectory_output"] = str(output_path)
    report["trajectory_output_sha256"] = sha256_file(output_path)
    write_json(report_path, report)
    if failures and settings.get("fail_on_unresolved", True):
        raise RuntimeError("; ".join(failures))
    print(f"trajectory -> {output_path}", flush=True)
    print(f"report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
