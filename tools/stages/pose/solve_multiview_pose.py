#!/usr/bin/env python
"""Solve a body-relative 6D-pose trajectory from synchronized multi-view data.

This file is the orchestration layer.  Tracking strategies, symmetry handling,
trajectory normalization, and validation live in ``common/``.
"""
from __future__ import annotations

import argparse
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
from common.pose_config import validate_pose_config
from common.pose_tracking import (
    align_symmetric_pose,
    fuse_part_clouds,
    track_cloud_registration,
    track_mask_bbox_translation,
    track_model_translation,
)
from common.pose_transforms import (
    decompose_similarity,
    rigid_from_similarity,
    similarity,
)
from common.pose_validation import validate_world_poses
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
    )
    return {
        "frame": anchor,
        "frames": frames,
        "n_cloud_points": int(len(cloud)),
        "S_world_from_raw": fit["T_mesh_to_world"],
        "scale": float(fit["scale"]),
        "fit_rmse_m": float(fit["fit_rmse"]),
        "icp_cost": float(fit["icp_cost"]),
    }


def _count_observing_views(
    mask_root: Path,
    frame: int,
    part_id: int,
    views: list[str],
) -> int:
    from PIL import Image

    return sum(
        int(np.any(
            np.asarray(
                Image.open(mask_root / f"{frame:06d}" / f"{view}.png")
            ) == part_id
        ))
        for view in views
    )


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
    for index, part in enumerate(config["parts"]):
        state = config["states"][part]
        if part == config["reference_part"]:
            calibration = state["calibration_frames"]
            frames = [int(calibration[len(calibration) // 2])]
        else:
            frames = [int(value) for value in state["anchor_frames"]]
        anchors[part] = {}
        anchor_info[part] = {}
        fits = []
        for anchor in frames:
            local_state = state
            if part == config["reference_part"]:
                local_state = dict(state)
                local_state["anchor_windows"] = {
                    str(anchor): state["calibration_frames"]
                }
            fit = _fit_anchor(
                meshes[part],
                cloud_root,
                local_state,
                anchor,
                part,
                seed=31 + 10 * index + anchor,
            )
            fits.append(fit)
            anchor_info[part][str(anchor)] = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in fit.items()
            }
            print(
                f"calibration {part}@{anchor}: "
                f"scale={fit['scale']:.6f} rmse={fit['fit_rmse_m']:.5f}",
                flush=True,
            )
        scale = float(
            state.get("scale_prior", np.median([fit["scale"] for fit in fits]))
        )
        scales[part] = scale
        for fit in fits:
            _raw_scale, rotation, translation = decompose_similarity(
                fit["S_world_from_raw"]
            )
            transform = similarity(scale, rotation, translation)
            anchors[part][fit["frame"]] = transform
            anchor_info[part][str(fit["frame"])][
                "S_world_from_raw"
            ] = transform.tolist()
            anchor_info[part][str(fit["frame"])]["fixed_scale"] = scale

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
        "raw_mesh_origins": {
            part: origins[part].tolist() for part in origins
        },
        "anchors": anchor_info,
        "appearance_refinement": appearance_reports,
        "input_fingerprint": input_fingerprint,
    }
    return anchors, scales, calibration


def _load_calibration(path: Path) -> tuple[dict, dict, dict]:
    calibration = load_json(path)
    anchors = {
        part: {
            int(frame): np.asarray(value["S_world_from_raw"], dtype=float)
            for frame, value in values.items()
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
) -> tuple[dict[int, np.ndarray], dict]:
    start_frame, end_frame = frame_range
    state = config["states"][part]
    method = state.get("method", "cloud_registration")
    poses = {}
    registrations = {}
    axis = (
        np.asarray(state["symmetry_axis_raw"], dtype=float)
        if "symmetry_axis_raw" in state
        else None
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
        anchor_frames = sorted(anchors)
        start_anchor = min(
            anchor_frames, key=lambda value: abs(value - dynamic_start)
        )
        end_anchor = min(
            anchor_frames, key=lambda value: abs(value - dynamic_end)
        )
        start_pose = rigid_from_similarity(anchors[start_anchor], origin)
        end_pose = rigid_from_similarity(anchors[end_anchor], origin)
        if axis is not None and method in {
            "model_tracking", "mask_bbox_tracking"
        }:
            end_pose = align_symmetric_pose(end_pose, start_pose, axis)
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
                axis,
            )
        else:
            raise ValueError(f"{part}: unknown tracking method {method!r}")
        poses.update(segment)
        registrations.update(segment_report)

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
    if axis is not None and config["registration"].get("symmetry_lock", True):
        for frame in range(start_frame + 1, end_frame + 1):
            poses[frame] = align_symmetric_pose(
                poses[frame], poses[frame - 1], axis
            )
    return poses, registrations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "pose_data_1_8view.json"),
    )
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--force-reuse-calibration", action="store_true")
    parser.add_argument(
        "--calibration",
        help="explicit calibration.json to reuse (useful for isolated reruns)",
    )
    parser.add_argument("--output-root", help="override config output_root")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = validate_pose_config(load_json(config_path), check_paths=True)
    output = Path(args.output_root or config["output_root"])
    cloud_root = Path(config.get(
        "point_cloud_root",
        output / "parts_ply" / config["recon_backend"],
    ))
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

    body_pose = rigid_from_similarity(
        next(iter(anchors[body].values())), origins[body]
    )
    world_poses = {
        body: {
            frame: body_pose.copy()
            for frame in range(start_frame, end_frame + 1)
        }
    }
    registration_reports = {}
    for part in config["parts"]:
        if part == body:
            continue
        poses, report = _solve_part(
            part,
            config,
            meshes[part],
            scales[part],
            origins[part],
            anchors[part],
            cloud_root,
            (start_frame, end_frame),
            priors.get(part),
        )
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
    mask_root = Path(config["masks_dir"])
    for frame in range(start_frame, end_frame + 1):
        key = f"{frame:06d}"
        trajectory["frames"][key] = {"parts": {}}
        for part in config["parts"]:
            observing_views = _count_observing_views(
                mask_root,
                frame,
                int(config["part_ids"][part]),
                config["views"],
            )
            state_config = config["states"][part]
            method = state_config.get("method", "cloud_registration")
            if part == body:
                state, source = "static", "body_anchor"
            else:
                moving = any(
                    int(start) <= frame <= int(end)
                    for start, end in state_config["dynamic_ranges"]
                )
                if observing_views == 0:
                    state, source = "inferred_unobservable", "interpolation"
                elif moving:
                    sources = {
                        "trajectory_prior": "validated_trajectory_prior",
                        "model_tracking": "multiview_mesh_gated_tracking",
                        "mask_bbox_tracking": "multiview_silhouette_tracking",
                        "cloud_registration": "multiview_cloud_registration",
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
    main()
