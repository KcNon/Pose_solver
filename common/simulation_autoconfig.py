"""Derive a conservative Isaac asset config from pose/state artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def final_state_run(
    rows: dict[str, dict[str, Any]],
    states: set[str],
) -> tuple[int, int] | None:
    selected = sorted(
        int(frame)
        for frame, row in rows.items()
        if str(row.get("state")) in states
        and int(row.get("observing_views", 0)) > 0
    )
    if not selected:
        return None
    runs: list[list[int]] = []
    for frame in selected:
        if runs and frame == runs[-1][-1] + 1:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    run = runs[-1]
    return run[0], run[-1]


def generate_simulation_config(
    *,
    trajectory_path: Path,
    state_report: dict[str, Any],
    mesh_dir: Path,
    output_dir: Path,
    asset_name: str,
    assembly_window_frames: int = 30,
) -> dict[str, Any]:
    from common.io_utils import load_json

    trajectory = load_json(trajectory_path)
    parts = list(trajectory["parts"])
    reference = str(trajectory["reference_part"])
    moving_parts = [part for part in parts if part != reference]
    if not moving_parts:
        raise ValueError("Isaac collision validation requires a moving part")

    meshes: dict[str, str] = {}
    masses: dict[str, float] = {}
    raw_extents: dict[str, np.ndarray] = {}
    for part in parts:
        path = (mesh_dir / f"{part}.glb").absolute()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh = trimesh.load(path, force="mesh")
        raw_extents[part] = np.asarray(mesh.extents, dtype=float)
        canonical_extents = raw_extents[part] * float(
            trajectory["scales"][part]
        )
        # AABB volume with a conservative 25% fill fraction and 650 kg/m^3
        # density.  This is explicitly an assumption, bounded for stability.
        mass = 650.0 * 0.25 * float(np.prod(canonical_extents))
        masses[part] = float(np.clip(mass, 0.05, 5.0))
        meshes[part] = str(path)

    assembly_targets: dict[str, Any] = {}
    assembly_starts: list[int] = []
    moving_starts: list[int] = []
    for part in moving_parts:
        report = state_report["parts"][part]
        rows = report["states"]
        run = final_state_run(rows, {"assembled"})
        if run is None:
            run = final_state_run(rows, {"static"})
        if run is None:
            raise RuntimeError(f"{part}: no visible final static state")
        run_start, run_end = run
        target_start = max(run_start, run_end - assembly_window_frames + 1)
        assembly_starts.append(run_start)
        assembly_targets[part] = {
            "frame_range": [target_start, run_end],
            "required_state": "static",
            "min_observing_views": 1,
            "max_translation_residual_m": 0.02,
            "max_rotation_residual_deg": 10.0,
        }
        detected = report.get("detected_moving_ranges", [])
        if detected:
            moving_starts.append(int(detected[-1][0]))

    up_axis_index = int(np.argmax(raw_extents[reference]))
    up_axis = [0.0, 0.0, 0.0]
    up_axis[up_axis_index] = 1.0
    replay_start = min(moving_starts or assembly_starts)
    replay_end = min(
        max(assembly_starts) + 5,
        max(int(frame) for frame in trajectory["frames"]),
    )
    return {
        "asset_name": asset_name,
        "trajectory": str(trajectory_path.resolve()),
        "meshes": meshes,
        "output_dir": str(output_dir.resolve()),
        "reference_part": reference,
        "display_parts": parts,
        "mass_kg": masses,
        "collision_proxies": {
            # Recon meshes are often non-manifold and cannot be cooked as a
            # dynamic SDF. A compound shell of watertight convex cells is a
            # deterministic, category-agnostic fallback which also preserves
            # observed openings instead of filling the entire convex hull.
            part: {
                "type": "voxel_shell",
                "resolution": 24,
                "parameter_source": "mesh_fit",
                "confidence": "low",
            }
            for part in moving_parts
        },
        "assembly_targets": assembly_targets,
        "simulation": {
            "container_part": reference,
            "inserted_part": moving_parts[0],
            "dynamic_collision_approximation": "convexDecomposition",
            "observed_overlap_carving": {
                "enabled": True,
                "penetration_tolerance_m": 0.001,
                "minimum_reference_vertices": 5,
                "maximum_removed_fraction": 0.45,
            },
            "floor_clearance_m": 0.03,
            "up_axis_body": up_axis,
            "align_insert_up_axis": False,
            "observed_replay_frame_range": [replay_start, replay_end],
            "physics_hz": 240,
            "settle_seconds": 3.0,
            "drop_height_m": 0.03,
            "static_friction": 0.4,
            "dynamic_friction": 0.3,
            "restitution": 0.0,
            "contact_offset_m": 0.001,
            "rest_offset_m": 0.0,
            "assembly_validation": {
                "allow_pose_mutation": False,
                "release_height_m": 0.0,
                "settle_seconds": 3.0,
                "contact_window_seconds": 1.0,
                "minimum_contact_fraction": 0.1,
                "maximum_contact_gap_seconds": 0.12,
                "maximum_lateral_error_m": 0.005,
                "maximum_axial_error_m": 0.008,
                "maximum_tilt_error_deg": 5.0,
                "maximum_final_linear_speed_mps": 0.01,
                "maximum_final_angular_speed_radps": 0.1,
                "translation_levels_m": [0.001, 0.002, 0.005],
                "tilt_levels_deg": [1.0, 3.0, 5.0],
            },
            "settled_contact_control": {
                "enabled": True,
                "states": ["static"],
                "frequency_radps": 6.0,
                "damping_ratio": 2.5,
                "maximum_position_error_m": 0.01,
            },
            "success_translation_m": 0.02,
            "success_rotation_deg": 15.0,
            "success_linear_speed_mps": 0.03,
            "success_angular_speed_radps": 0.15,
            "trials": [],
            "mass_note": (
                "Automatically estimated from canonical AABB volume; no "
                "measured object mass was supplied."
            ),
        },
    }
