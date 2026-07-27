"""Validation for the reusable multi-view pose configuration contract."""
from __future__ import annotations

from pathlib import Path

from common.symmetry import symmetry_spec_from_state


TRACKING_METHODS = {
    "cloud_registration",
    "model_tracking",
    "mask_bbox_tracking",
    "trajectory_prior",
    "similarity_prior",
}


def validate_pose_config(
    config: dict,
    *,
    check_paths: bool = False,
    allow_auto: bool = False,
) -> dict:
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
        try:
            symmetry_spec_from_state(state)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{part}: invalid symmetry: {error}") from error
        method = state.get("method", "cloud_registration")
        if allow_auto and method == "auto":
            method = "cloud_registration"
        if method not in TRACKING_METHODS:
            raise ValueError(f"{part}: unsupported method {method!r}")
        ranges = []
        for key in ("static_ranges", "dynamic_ranges"):
            values = state.get(key, [])
            if allow_auto and (
                values is None
                or values == "auto"
            ):
                continue
            ranges.extend(tuple(map(int, pair)) for pair in values)
        for range_start, range_end in ranges:
            if range_start > range_end:
                raise ValueError(
                    f"{part}: invalid state range {range_start}..{range_end}"
                )
        if (
            part != config["reference_part"]
            and method != "trajectory_prior"
            and not allow_auto
        ):
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
    constraints = config.get("trajectory_constraints", {})
    if constraints.get("enabled", False):
        proxy_config = constraints.get(
            "geometry_proxy_config",
            constraints.get("collision_proxy_config"),
        )
        if not proxy_config:
            raise ValueError(
                "trajectory_constraints requires geometry_proxy_config"
            )
        relations = constraints.get("relations", [])
        if not relations:
            raise ValueError(
                "trajectory_constraints requires at least one relation"
            )
        names = [str(relation.get("name", "")) for relation in relations]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "trajectory constraint relation names must be non-empty and unique"
            )
        for relation in relations:
            name = str(relation["name"])
            relation_type = str(relation.get("type", "insert_into"))
            if relation_type not in {"insert_into", "pairwise_contact"}:
                raise ValueError(
                    f"{name}: unsupported trajectory constraint type"
                )
            reference_part = relation.get(
                "container", relation.get("reference_part", "")
            )
            relation_parts = {
                str(reference_part),
                str(relation.get("moving_part", "")),
            }
            unknown = relation_parts.difference(parts)
            if unknown:
                raise ValueError(
                    f"{name}: trajectory constraint has unknown parts "
                    f"{sorted(unknown)}"
                )
            relation_start, relation_end = map(
                int, relation["frame_range"]
            )
            if (
                relation_start < start
                or relation_end > end
                or relation_start >= relation_end
            ):
                raise ValueError(
                    f"{name}: invalid trajectory constraint frame range "
                    f"{relation_start}..{relation_end}"
                )
            if (
                relation_type == "insert_into"
                and float(relation["entry_center_radius_m"]) <= 0.0
            ):
                raise ValueError(
                    f"{name}: entry_center_radius_m must be positive"
                )
            if relation_type == "pairwise_contact":
                contact_start = int(
                    relation.get("contact_start_frame", relation_start)
                )
                if not relation_start <= contact_start <= relation_end:
                    raise ValueError(
                        f"{name}: contact_start_frame must lie in frame_range"
                    )
                if (
                    float(relation.get("surface_points", 6000)) < 1000
                    or float(relation.get("near_field_m", 0.03)) <= 0.0
                    or float(
                        relation.get("maximum_contact_gap_m", 0.004)
                    )
                    <= 0.0
                ):
                    raise ValueError(
                        f"{name}: invalid pairwise contact sampling/distance settings"
                    )
            validation_start, validation_end = map(
                int,
                relation.get(
                    "validation_frame_range",
                    [relation_start, relation_end],
                ),
            )
            if (
                validation_start < relation_start
                or validation_end > relation_end
                or validation_start > validation_end
            ):
                raise ValueError(
                    f"{name}: validation_frame_range must lie in frame_range"
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
        if constraints.get("enabled", False):
            proxy_config = Path(
                constraints.get(
                    "geometry_proxy_config",
                    constraints.get("collision_proxy_config"),
                )
            )
            if not proxy_config.exists():
                raise FileNotFoundError(proxy_config)
    return config
