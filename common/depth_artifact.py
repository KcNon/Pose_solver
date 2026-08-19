"""Versioned metadata and resume validation for DA3 prediction artifacts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


DEPTH_ARTIFACT_SCHEMA_VERSION = 1
REQUIRED_PREDICTION_ARRAYS = {
    "images",
    "depth",
    "depth_conf",
    "extrinsic",
    "intrinsic",
    "world_points_from_depth",
    "view_names",
}
METADATA_KEYS = {
    "pose_solver_depth_schema_version",
    "process_res",
    "process_res_method",
    "source_image_hw",
    "processed_image_hw",
    "camera_frames",
    "model_dir",
    "use_ray_pose",
    "ref_view_strategy",
}


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def prediction_metadata(
    *,
    process_res: int,
    process_res_method: str,
    source_image_hw: tuple[int, int],
    processed_image_hw: tuple[int, int],
    camera_frames: Sequence[str],
    model_dir: str | Path,
    use_ray_pose: bool,
    ref_view_strategy: str,
) -> dict[str, np.ndarray]:
    """Return serializable metadata arrays stored alongside DA3 tensors."""

    return {
        "pose_solver_depth_schema_version": np.asarray(
            DEPTH_ARTIFACT_SCHEMA_VERSION, dtype=np.int32
        ),
        "process_res": np.asarray(process_res, dtype=np.int32),
        "process_res_method": np.asarray(str(process_res_method)),
        "source_image_hw": np.asarray(source_image_hw, dtype=np.int32),
        "processed_image_hw": np.asarray(processed_image_hw, dtype=np.int32),
        "camera_frames": np.asarray([str(value) for value in camera_frames]),
        "model_dir": np.asarray(str(Path(model_dir).resolve())),
        "use_ray_pose": np.asarray(bool(use_ray_pose)),
        "ref_view_strategy": np.asarray(str(ref_view_strategy)),
    }


def prediction_compatibility(
    artifact: Mapping[str, Any],
    *,
    views: Sequence[str],
    process_res: int,
    process_res_method: str,
    source_image_hw: tuple[int, int],
    camera_frames: Sequence[str],
    model_dir: str | Path,
    use_ray_pose: bool,
    ref_view_strategy: str,
    allow_legacy_shape_resume: bool = False,
) -> tuple[bool, str]:
    """Validate that an existing prediction matches the requested DA3 run."""

    files = set(getattr(artifact, "files", artifact.keys()))
    missing = sorted(REQUIRED_PREDICTION_ARRAYS.difference(files))
    if missing:
        return False, f"missing arrays: {missing}"
    try:
        depth = np.asarray(artifact["depth"])
        images = np.asarray(artifact["images"])
        stored_views = [str(value) for value in artifact["view_names"].tolist()]
        for key in REQUIRED_PREDICTION_ARRAYS:
            _ = np.asarray(artifact[key]).shape
    except Exception as exc:
        return False, f"unreadable arrays: {exc}"
    if depth.ndim not in {3, 4}:
        return False, f"unexpected depth shape {depth.shape}"
    if stored_views != [str(value) for value in views]:
        return False, "view names or ordering changed"
    if int(depth.shape[0]) != len(views):
        return False, "view count changed"
    depth_hw = tuple(int(value) for value in depth.shape[1:3])
    if images.ndim != 4 or tuple(images.shape[-2:]) != depth_hw:
        return False, "processed image/depth resolution mismatch"

    missing_metadata = sorted(METADATA_KEYS.difference(files))
    if missing_metadata:
        if not allow_legacy_shape_resume:
            return False, f"missing metadata: {missing_metadata}"
        if str(process_res_method) != "upper_bound_resize":
            return False, "legacy artifact cannot verify resize method"
        if max(depth_hw) != int(process_res):
            return False, "legacy artifact resolution changed"
        source_aspect = float(source_image_hw[1]) / float(source_image_hw[0])
        depth_aspect = float(depth_hw[1]) / float(depth_hw[0])
        if abs(depth_aspect / source_aspect - 1.0) > 0.05:
            return False, "legacy artifact aspect ratio changed"
        return True, "legacy shape-compatible artifact"

    expected = prediction_metadata(
        process_res=process_res,
        process_res_method=process_res_method,
        source_image_hw=source_image_hw,
        processed_image_hw=depth_hw,
        camera_frames=camera_frames,
        model_dir=model_dir,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
    )
    for key, wanted in expected.items():
        actual = _scalar(artifact[key])
        desired = _scalar(wanted)
        if actual != desired:
            return False, f"metadata mismatch for {key}: {actual!r} != {desired!r}"
    return True, "metadata-compatible artifact"
