"""Validation for the reusable multi-view pose configuration contract."""
from __future__ import annotations

from pathlib import Path


TRACKING_METHODS = {
    "cloud_registration",
    "model_tracking",
    "mask_bbox_tracking",
    "trajectory_prior",
    "similarity_prior",
}


def validate_pose_config(config: dict, *, check_paths: bool = False) -> dict:
    required = {
        "frames",
        "views",
        "parts",
        "part_ids",
        "reference_part",
        "states",
        "registration",
        "mesh_dir",
        "masks_dir",
        "output_root",
        "recon_backend",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"pose config is missing keys: {sorted(missing)}")
    parts = [str(part) for part in config["parts"]]
    views = [str(view) for view in config["views"]]
    if not parts or len(parts) != len(set(parts)):
        raise ValueError("parts must be non-empty and unique")
    if not views or len(views) != len(set(views)):
        raise ValueError("views must be non-empty and unique")
    if config["reference_part"] not in parts:
        raise ValueError("reference_part must be present in parts")
    if set(config["part_ids"]) != set(parts):
        raise ValueError("part_ids must contain every configured part exactly once")
    ids = [int(config["part_ids"][part]) for part in parts]
    if any(value <= 0 or value >= 256 for value in ids):
        raise ValueError("part IDs must be in [1,255]")
    if len(ids) != len(set(ids)):
        raise ValueError("part IDs must be unique")
    start = int(config["frames"]["start"])
    end = int(config["frames"]["end"])
    if start < 0 or end < start:
        raise ValueError(f"invalid frame range: {start}..{end}")
    unknown_states = set(config["states"]).difference(parts)
    missing_states = set(parts).difference(config["states"])
    if unknown_states or missing_states:
        raise ValueError(
            "states must contain every part exactly once; "
            f"unknown={sorted(unknown_states)}, missing={sorted(missing_states)}"
        )
    for part in parts:
        state = config["states"][part]
        method = state.get("method", "cloud_registration")
        if method not in TRACKING_METHODS:
            raise ValueError(f"{part}: unsupported method {method!r}")
        ranges = [
            tuple(map(int, values))
            for key in ("static_ranges", "dynamic_ranges")
            for values in state.get(key, [])
        ]
        for range_start, range_end in ranges:
            if range_start > range_end:
                raise ValueError(
                    f"{part}: invalid state range {range_start}..{range_end}"
                )
        if part != config["reference_part"] and method != "trajectory_prior":
            covered = {
                frame
                for range_start, range_end in ranges
                for frame in range(range_start, range_end + 1)
            }
            uncovered = sorted(set(range(start, end + 1)).difference(covered))
            if uncovered:
                raise ValueError(
                    f"{part}: state ranges do not cover frames "
                    f"{uncovered[0]}..{uncovered[-1]}"
                )
        if method in {"trajectory_prior", "similarity_prior"} and not state.get(
            "prior_trajectory"
        ):
            raise ValueError(f"{part}: {method} requires prior_trajectory")
    registration = config["registration"]
    voxels = registration.get("voxel_sizes_m", [])
    distances = registration.get("max_correspondence_m", [])
    if not voxels or len(voxels) != len(distances):
        raise ValueError(
            "registration voxel_sizes_m and max_correspondence_m must have "
            "the same non-zero length"
        )
    if check_paths:
        for key in ("mesh_dir", "masks_dir"):
            path = Path(config[key])
            if not path.exists():
                raise FileNotFoundError(f"{key}: {path}")
        for part in parts:
            mesh = Path(config["mesh_dir"]) / f"{part}.glb"
            if not mesh.exists():
                raise FileNotFoundError(mesh)
    return config
