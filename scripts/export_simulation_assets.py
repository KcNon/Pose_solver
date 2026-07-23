#!/usr/bin/env python3
"""Export pose_solver meshes and trajectories as URDF-ready simulation assets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common.simulation_assets import (
    aabb_overlap,
    canonical_from_raw_matrix,
    canonicalize_mesh,
    export_collision_obj,
    export_textured_obj,
    load_flat_mesh,
    robust_average_pose,
    select_part_poses,
    sha256_file,
    state_runs,
    vertex_surface_distance_summary,
    write_urdf,
)
from common.io_utils import load_json, write_json
from common.pose_transforms import transform_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def validate_config(config: dict[str, Any], trajectory: dict[str, Any]) -> None:
    required = {"trajectory", "meshes", "output_dir", "reference_part", "assembly_targets", "simulation"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    trajectory_parts = set(trajectory["parts"])
    if config["reference_part"] not in trajectory_parts:
        raise ValueError("reference_part is not present in the trajectory")
    unknown = sorted(set(config["meshes"]) - trajectory_parts)
    if unknown:
        raise ValueError(f"Configured meshes are not present in the trajectory: {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/simulation_assets.json")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_json(config_path)
    trajectory_path = resolve_path(project_root, config["trajectory"])
    trajectory = load_json(trajectory_path)
    validate_config(config, trajectory)
    output_root = (args.output_dir or resolve_path(project_root, config["output_dir"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    part_info: dict[str, dict[str, Any]] = {}
    canonical_meshes = {}
    transform_checks = {}
    for part, mesh_value in config["meshes"].items():
        source_path = resolve_path(project_root, mesh_value)
        raw_mesh = load_flat_mesh(source_path)
        scale = float(trajectory["scales"][part])
        raw_origin = trajectory["raw_mesh_origins"][part]
        canonical_mesh = canonicalize_mesh(raw_mesh, scale, raw_origin)
        canonical_meshes[part] = canonical_mesh

        visual_path = export_textured_obj(canonical_mesh, output_root / "meshes/visual" / part, part)
        collision_path = output_root / "meshes/collision" / f"{part}.obj"
        export_collision_obj(canonical_mesh, collision_path)

        # ``split`` materializes a new mesh for every connected component and
        # becomes prohibitively expensive for dense reconstruction meshes.
        # ``body_count`` computes the same QA quantity from sparse vertex
        # adjacency without duplicating any geometry.
        connected_components = int(canonical_mesh.body_count)
        part_info[part] = {
            "source_mesh": str(source_path.relative_to(project_root)),
            "source_sha256": sha256_file(source_path),
            "visual_mesh": str(visual_path.relative_to(output_root)),
            "collision_mesh": str(collision_path.relative_to(output_root)),
            "scale": scale,
            "raw_mesh_origin": list(map(float, raw_origin)),
            "T_part_from_raw_mesh": canonical_from_raw_matrix(scale, raw_origin).tolist(),
            "canonical_bounds_m": np.asarray(canonical_mesh.bounds, dtype=float).tolist(),
            "canonical_extents_m": np.asarray(canonical_mesh.extents, dtype=float).tolist(),
            "vertices": int(len(canonical_mesh.vertices)),
            "faces": int(len(canonical_mesh.faces)),
            "connected_components": connected_components,
            "watertight": bool(canonical_mesh.is_watertight),
            "mass_kg": float(config.get("mass_kg", {}).get(part, 1.0)),
            "mass_source": "configured_assumption",
        }

        expected = canonical_from_raw_matrix(scale, raw_origin)
        errors = []
        for frame_id, frame in trajectory["frames"].items():
            part_data = frame["parts"][part]
            actual = np.asarray(part_data["S_world_from_raw_mesh"], dtype=np.float64)
            pose = np.asarray(part_data["T_world_from_part"], dtype=np.float64)
            errors.append(float(np.max(np.abs(actual - pose @ expected))))
        transform_checks[part] = {
            "max_abs_matrix_error": float(max(errors)),
            "passed": bool(max(errors) < 1e-7),
        }

    reference_part = config["reference_part"]
    assembled_transforms: dict[str, np.ndarray] = {reference_part: np.eye(4)}
    assembly_stats: dict[str, Any] = {
        reference_part: {"used_frames": "identity_reference", "candidate_frames": "identity_reference"}
    }
    for part, selection in config["assembly_targets"].items():
        poses, frame_ids = select_part_poses(trajectory, part, selection)
        if not poses:
            raise RuntimeError(f"No poses selected for assembly target {part}: {selection}")
        transform, stats = robust_average_pose(
            poses,
            frame_ids,
            max_translation_residual_m=float(selection.get("max_translation_residual_m", 0.03)),
            max_rotation_residual_deg=float(selection.get("max_rotation_residual_deg", 15.0)),
        )
        assembled_transforms[part] = transform
        assembly_stats[part] = stats

    urdf_dir = output_root / "urdf"
    for part in config["meshes"]:
        write_urdf(
            urdf_dir / f"{part}.urdf",
            robot_name=part,
            parts=[part],
            part_info=part_info,
        )
    display_parts = list(config.get("display_parts", config["meshes"].keys()))
    write_urdf(
        urdf_dir / "rice_cooker_display.urdf",
        robot_name="rice_cooker_display",
        parts=display_parts,
        part_info=part_info,
        fixed_transforms={part: assembled_transforms[part] for part in display_parts},
        root_part=reference_part,
    )

    simulation = config["simulation"]
    container_part = simulation["container_part"]
    inserted_part = simulation["inserted_part"]
    target = assembled_transforms[inserted_part]
    body_mesh = canonical_meshes[container_part]
    inserted_world_vertices = transform_points(canonical_meshes[inserted_part].vertices, target)
    inserted_bounds = np.stack((inserted_world_vertices.min(axis=0), inserted_world_vertices.max(axis=0)))
    geometry_report = {
        "schema_version": 1,
        "note": (
            "AABB overlap and vertex-to-vertex distances are diagnostics, not collision or signed-clearance tests. "
            "Authoritative contact/penetration checks are produced by Isaac Sim."
        ),
        "transform_consistency": transform_checks,
        "mesh_quality": {
            part: {
                key: part_info[part][key]
                for key in ("vertices", "faces", "connected_components", "watertight", "canonical_bounds_m", "canonical_extents_m")
            }
            for part in part_info
        },
        "assembly_pose_statistics": assembly_stats,
        "assembled_aabb": aabb_overlap(np.asarray(body_mesh.bounds), inserted_bounds),
        "inserted_to_container_vertex_distance_m": vertex_surface_distance_summary(
            np.asarray(body_mesh.vertices), inserted_world_vertices
        ),
    }

    replay_start, replay_end = simulation.get("observed_replay_frame_range", [0, 10**9])
    replay_frames = []
    for frame_id in sorted(trajectory["frames"], key=int):
        index = int(frame_id)
        if index < int(replay_start) or index > int(replay_end):
            continue
        data = trajectory["frames"][frame_id]["parts"][inserted_part]
        replay_frames.append(
            {
                "frame_id": frame_id,
                "state": data.get("state", "unknown"),
                "source": data.get("source"),
                "observing_views": data.get("observing_views"),
                "T_body_from_part": data["T_body_from_part"],
            }
        )

    write_json(output_root / "qa/geometry_report.json", geometry_report)
    write_json(
        output_root / "qa/insertion_trajectory_body.json",
        {
            "reference_part": reference_part,
            "inserted_part": inserted_part,
            "frames": replay_frames,
        },
    )

    manifest = {
        "schema_version": 1,
        "asset_name": config.get("asset_name", "rice_cooker"),
        "units": "meter",
        "coordinate_convention": {
            "handedness": "right",
            "part_frame": trajectory["conventions"]["T_world_from_part"],
            "urdf_mesh_origin": "canonical part frame; visual and collision origins are zero",
            "quaternion": "xyzw in pose_solver; wxyz in Isaac runtime",
        },
        "inputs": {
            "config": str(config_path.relative_to(project_root)),
            "config_sha256": sha256_file(config_path),
            "trajectory": str(trajectory_path.relative_to(project_root)),
            "trajectory_sha256": sha256_file(trajectory_path),
        },
        "reference_part": reference_part,
        "parts": part_info,
        "states": {part: state_runs(trajectory, part) for part in trajectory["parts"]},
        "assembled_T_body_from_part": {part: matrix.tolist() for part, matrix in assembled_transforms.items()},
        "assembly_pose_statistics": assembly_stats,
        "simulation": {
            **simulation,
            "collision_policy": {
                container_part: "static triangle mesh; preserve cavity",
                inserted_part: "dynamic convex decomposition in Isaac Sim",
            },
        },
        "outputs": {
            "display_urdf": "urdf/rice_cooker_display.urdf",
            "independent_urdfs": {part: f"urdf/{part}.urdf" for part in part_info},
            "geometry_report": "qa/geometry_report.json",
            "observed_replay": "qa/insertion_trajectory_body.json",
            "runtime_preflight": "qa/isaac_runtime_preflight.json",
            "assembly_preview": "qa/assembly_preview_axes.png",
            "isaac_report": "qa/isaac_insertion_report.json",
        },
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(
        output_root / "stages/export.complete.json",
        {
            "stage": "export",
            "status": "complete",
            "trajectory_sha256": manifest["inputs"]["trajectory_sha256"],
            "config_sha256": manifest["inputs"]["config_sha256"],
            "transform_consistency_passed": bool(all(item["passed"] for item in transform_checks.values())),
        },
    )
    print(f"Exported simulation assets to {output_root}")
    print(f"Manifest: {output_root / 'manifest.json'}")
    print(f"Geometry QA: {output_root / 'qa/geometry_report.json'}")


if __name__ == "__main__":
    main()
