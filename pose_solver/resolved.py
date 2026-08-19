"""Generate internal stage configs from the single source contract."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common.io_utils import load_json, write_json
from pose_solver.artifacts import ArtifactLayout
from pose_solver.config import PipelineConfig, StageConfig, deep_merge


def _compatibility_base(stage: StageConfig) -> dict[str, Any]:
    return (
        deepcopy(load_json(stage.compatibility_config))
        if stage.compatibility_config
        else {}
    )


def _model(config: PipelineConfig, name: str) -> Any:
    if name not in config.models:
        raise ValueError(
            f"models.{name} is required to run this stage; alternatively set "
            "the stage to mode='reuse' with an artifact path"
        )
    return config.models[name]


def frame_config(config: PipelineConfig) -> dict[str, Any]:
    """Build the synchronized extraction config when videos are supplied."""

    if not config.videos:
        raise ValueError("input.videos is required to extract frames")
    raw = {
        "frames_dir": str(config.frames_dir),
        "views": list(config.views),
        "videos": {
            view: str(path) for view, path in config.videos.items()
        },
        "sample_fps": config.input_options.get("sample_fps", "source"),
        "sync_offsets_s": config.input_options.get("sync_offsets_s", {}),
        "reference_trim_s": config.input_options.get("reference_trim_s", 0.0),
    }
    for key in ("duration_s", "allow_frame_duplication"):
        if key in config.input_options:
            raw[key] = config.input_options[key]
    return raw


def mask_config(config: PipelineConfig, layout: ArtifactLayout) -> dict[str, Any]:
    raw = {
        "frames_dir": str(config.frames_dir),
        "work_root": str(layout.mask_work),
        "output_root": str(layout.mask_output),
        "views": list(config.views),
        "parts": {part.name: part.mask_dict() for part in config.parts},
        "occlusion_order": [part.name for part in config.parts],
        "mask_postprocess": {
            "enabled": True,
            "close_kernel": 5,
            "fill_holes": True,
        },
        "automation": {
            "discovery": {
                "stride": 10,
                "minimum_views": min(3, len(config.views)),
                "consecutive_scans": 2,
                "stop_when_resolved": True,
                "stop_when_refined_resolved": True,
            },
            "repair": {
                "enabled": True,
                "apply": True,
                "padding_frames": 2,
                "maximum_jobs_per_part": 3,
            },
        },
        "qwen_python": str(_model(config, "qwen_python")),
        "sam_python": str(_model(config, "sam_python")),
        "qwen_model": str(_model(config, "qwen_model")),
        "sam_ckpt": str(_model(config, "sam_checkpoint")),
        "runtime": {
            "qwen_gpu": config.devices[0],
            "sam_gpu": config.devices[-1],
        },
    }
    da3_dir = config.input_options.get("depth_dir")
    if da3_dir:
        raw.update({
            "recon_backend": "da3_self_cond",
            "da3_self_cond_dir": str(Path(da3_dir).expanduser().resolve()),
        })
    result = deep_merge(
        deep_merge(_compatibility_base(config.mask), raw),
        config.mask.overrides,
    )
    # Data identity is owned by the source contract, never by an algorithm
    # override or a compatibility preset.
    for key in ("frames_dir", "work_root", "output_root", "views", "runtime"):
        result[key] = deepcopy(raw[key])
    for name, values in raw["parts"].items():
        result.setdefault("parts", {}).setdefault(name, {}).update(values)
    if set(result["parts"]) != set(raw["parts"]):
        result["parts"] = {
            name: result["parts"][name] for name in raw["parts"]
        }
    return result


def resolved_part_starts(
    config: PipelineConfig,
    layout: ArtifactLayout,
) -> dict[str, int]:
    resolved_mask = layout.mask_work / "manifests" / "resolved_mask_config.json"
    if resolved_mask.exists():
        raw = load_json(resolved_mask)
        return {
            name: int(raw["parts"][name]["start_frame"])
            for name in config.part_names
        }
    starts = {}
    for part in config.parts:
        if part.appearance_hint == "auto":
            raise RuntimeError(
                f"{part.name}: appearance_hint is still automatic; run mask "
                "discovery before depth or pose"
            )
        starts[part.name] = int(part.appearance_hint)
    return starts


def masks_dir(config: PipelineConfig, layout: ArtifactLayout) -> Path:
    return (
        config.mask.artifact
        if config.mask.mode == "reuse"
        else layout.mask_output / "masks"
    )


def point_cloud_root(config: PipelineConfig, layout: ArtifactLayout) -> Path:
    if config.depth.mode == "reuse":
        assert config.depth.artifact is not None
        return config.depth.artifact
    compatibility = _compatibility_base(config.depth)
    settings = deep_merge(compatibility, config.depth.overrides)
    quality = settings.get("quality_cloud", {})
    if quality.get("enabled", True):
        variant = str(quality.get("variant", "da3_self_cond_quality"))
    else:
        backend = str(settings.get("recon_backend", "da3_self_cond"))
        tag = settings.get("point_cloud_tag")
        variant = backend if not tag else f"{backend}_{tag}"
    return layout.depth_output / "parts_ply" / variant


def depth_config(
    config: PipelineConfig,
    layout: ArtifactLayout,
    starts: dict[str, int],
) -> dict[str, Any]:
    depth_dir = config.input_options.get("depth_dir")
    raw = {
        "frames_dir": str(config.frames_dir),
        "masks_dir": str(masks_dir(config, layout)),
        "mesh_dir": str(config.mesh_dir),
        "output_root": str(layout.depth_output),
        "point_cloud_output_root": str(layout.depth_output),
        "recon_backend": "da3_self_cond",
        "da3_self_cond_dir": str(
            Path(depth_dir).expanduser().resolve()
            if depth_dir
            else layout.depth_output / "da3-self-cond"
        ),
        "views": list(config.views),
        "parts": config.part_names,
        "part_ids": config.part_ids,
        "part_start_frames": starts,
        "frames": {"start": config.frame_start, "end": config.frame_end},
        "depth_pipeline": {
            "da3_python": str(_model(config, "da3_python")),
            "camera_frames": [
                config.frame_start,
                (config.frame_start + config.frame_end) // 2,
                config.frame_end,
            ],
            "process_res": 756,
            "process_res_method": "upper_bound_resize",
            "full_w": 1920,
            "full_h": 1080,
            "point_stride": 1,
        },
        "quality_cloud": {
            "enabled": True,
            "variant": "da3_self_cond_quality",
            "mask_resize_mode": "coverage",
            "mask_coverage_threshold": 0.08,
            "stride": 1,
            "fusion_voxel_mm": 0.0,
        },
    }
    result = deep_merge(
        deep_merge(_compatibility_base(config.depth), raw),
        config.depth.overrides,
    )
    for key in (
        "frames_dir", "masks_dir", "mesh_dir", "output_root",
        "point_cloud_output_root", "views", "parts", "part_ids",
        "part_start_frames", "frames",
    ):
        result[key] = deepcopy(raw[key])
    quality = result.get("quality_cloud", {})
    if quality.get("enabled", False):
        variant = str(quality.get("variant", "da3_self_cond_quality"))
        quality["point_cloud_root"] = str(
            layout.depth_output / "parts_ply" / variant
        )
    return result


def pose_config(
    config: PipelineConfig,
    layout: ArtifactLayout,
    starts: dict[str, int],
) -> dict[str, Any]:
    depth_dir = config.input_options.get("depth_dir")
    raw = {
        "frames_dir": str(config.frames_dir),
        "masks_dir": str(masks_dir(config, layout)),
        "mesh_dir": str(config.mesh_dir),
        "output_root": str(layout.pose_output),
        "point_cloud_output_root": str(layout.depth_output),
        "point_cloud_root": str(point_cloud_root(config, layout)),
        "point_cloud_variant": point_cloud_root(config, layout).name,
        "recon_backend": "da3_self_cond",
        "views": list(config.views),
        "parts": config.part_names,
        "part_ids": config.part_ids,
        "part_start_frames": starts,
        "reference_part": config.reference_part,
        "frames": {"start": config.frame_start, "end": config.frame_end},
        "states": {
            part.name: {
                "method": "auto",
                "static_ranges": "auto",
                "dynamic_ranges": "auto",
                "validation": {
                    "max_translation_step_m": 0.08,
                    "max_rotation_step_deg": 25.0,
                    "fail_on_violation": False,
                },
            }
            for part in config.parts
        },
        "automation": {
            "enabled": True,
            "use_detected_states": True,
            "lock_detected_static_pose": True,
            "allow_moving_reference": True,
            "infer_reference_part": not any(part.reference for part in config.parts),
            "infer_calibration_frames": True,
            "infer_anchors": True,
            "infer_appearance_evidence": True,
            "infer_symmetry": True,
            "minimum_observing_views": min(4, len(config.views)),
        },
        "backprojection": {
            "conf_mode": "adaptive",
            "conf_quantile": 0.5,
            "stride": 1,
            "max_points": 100000,
        },
        "registration": {
            "max_points": 20000,
            "voxel_sizes_m": [0.010, 0.005, 0.0025],
            "max_correspondence_m": [0.040, 0.025, 0.015],
            "max_iterations": 80,
            "minimum_fitness": 0.20,
            "maximum_median_nn_m": 0.020,
            "symmetry_lock": True,
        },
    }
    if depth_dir:
        raw["da3_self_cond_dir"] = str(Path(depth_dir).expanduser().resolve())
    compatibility = _compatibility_base(config.pose)
    if compatibility:
        # Compatibility presets carry only the proven algorithm policy. The
        # source config below still owns data identity and artifact routing.
        contract_keys = (
            "frames_dir", "masks_dir", "mesh_dir", "output_root",
            "point_cloud_output_root", "point_cloud_root",
            "point_cloud_variant", "recon_backend", "views", "parts",
            "part_ids", "part_start_frames", "reference_part", "frames",
        )
        contract = {key: deepcopy(raw[key]) for key in contract_keys}
        if "da3_self_cond_dir" in raw:
            contract["da3_self_cond_dir"] = raw["da3_self_cond_dir"]
        result = deep_merge(compatibility, contract)
    else:
        result = deepcopy(raw)
    result = deep_merge(result, config.pose.overrides)
    for key in (
        "frames_dir", "masks_dir", "mesh_dir", "output_root",
        "point_cloud_output_root", "point_cloud_root", "point_cloud_variant",
        "views", "parts", "part_ids", "part_start_frames", "frames",
    ):
        result[key] = deepcopy(raw[key])
    # A declared physical reference is part of the source contract. With no
    # declaration, automation may replace this provisional value.
    result["reference_part"] = raw["reference_part"]
    return result


def write_resolved(path: Path, value: dict[str, Any]) -> Path:
    write_json(path, value)
    return path
