"""View-level mask quality gates shared by pose stages.

The pose pipeline must not equate "the label exists" with "this camera is a
useful observation".  Thin parts can become small after resizing while a
tracking failure can label an entire neighbouring object.  These helpers keep
the decision in full-resolution image space and reject gross cross-view area
outliers before clouds or render losses are aggregated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def temporal_mask_area_references(
    mask_root: Path,
    part_id: int,
    views: list[str],
    frame_range: tuple[int, int],
    settings: Mapping[str, Any],
) -> dict[str, float]:
    """Build fixed-rig per-camera mask-area baselines.

    Cameras in a multi-view rig commonly have different focal lengths and
    object distances, so raw mask areas must not be compared across cameras.
    The median is computed independently for every camera over one or more
    configured clean reference intervals.
    """

    if str(settings.get("mask_area_reference_mode", "cross_view")) != (
        "per_view_temporal"
    ):
        return {}
    from PIL import Image

    configured_ranges = settings.get(
        "temporal_reference_ranges", [list(frame_range)]
    )
    minimum = int(settings.get("presence_minimum_full_mask_pixels", 1))
    values: dict[str, list[int]] = {str(view): [] for view in views}
    for first, last in configured_ranges:
        first = max(int(first), int(frame_range[0]))
        last = min(int(last), int(frame_range[1]))
        for frame in range(first, last + 1):
            for view in views:
                path = Path(mask_root) / f"{frame:06d}" / f"{view}.png"
                if not path.exists():
                    continue
                labels = np.asarray(Image.open(path))
                area = int(np.count_nonzero(labels == int(part_id)))
                if area >= minimum:
                    values[str(view)].append(area)
    references = {
        view: float(np.median(samples))
        for view, samples in values.items()
        if samples
    }
    missing = sorted(set(str(view) for view in views).difference(references))
    if missing:
        raise ValueError(
            "per-view temporal mask references contain no positive samples "
            f"for views: {missing}"
        )
    return references


def mask_area_quality(
    masks: Mapping[str, np.ndarray],
    part_id: int,
    *,
    minimum_pixels: int = 1,
    maximum_area_ratio: float = 4.0,
    minimum_area_ratio: float = 0.0,
    reference_areas: Mapping[str, float] | None = None,
    presence_minimum_pixels: int | None = None,
) -> dict[str, Any]:
    """Return robust per-view validity from full-resolution label masks.

    The default legacy mode compares synchronized cameras.  When
    ``reference_areas`` is supplied, each camera is instead compared with its
    own temporal reference area.  The latter is the safe mode for fixed rigs:
    focal length, crop and viewpoint can make two valid synchronized masks
    differ greatly in raw pixel area.

    ``presence_minimum_pixels`` is deliberately separate from the quality
    threshold.  A mask can prove that a part remains in frame even when it is
    too partial or noisy to update a 3D pose.
    """

    if minimum_pixels < 1:
        raise ValueError("minimum_pixels must be positive")
    if presence_minimum_pixels is None:
        presence_minimum_pixels = minimum_pixels
    if int(presence_minimum_pixels) < 1:
        raise ValueError("presence_minimum_pixels must be positive")
    if maximum_area_ratio <= 1.0:
        raise ValueError("maximum_area_ratio must be greater than one")
    if minimum_area_ratio < 0.0 or minimum_area_ratio >= 1.0:
        raise ValueError("minimum_area_ratio must be in [0, 1)")
    areas = {
        str(view): int(np.count_nonzero(np.asarray(labels) == int(part_id)))
        for view, labels in masks.items()
    }
    positive = [area for area in areas.values() if area > 0]
    median = float(np.median(positive)) if positive else 0.0
    temporal_references = {
        str(view): float(value)
        for view, value in dict(reference_areas or {}).items()
        if np.isfinite(float(value)) and float(value) > 0.0
    }
    views: dict[str, dict[str, Any]] = {}
    for view, area in areas.items():
        reference = temporal_references.get(view)
        denominator = reference if reference is not None else median
        ratio = float(area / denominator) if denominator > 0.0 else 0.0
        reasons = []
        if area < minimum_pixels:
            reasons.append("too_few_pixels")
        ratio_source = (
            "per_view_temporal" if reference is not None else "cross_view"
        )
        if denominator > 0.0 and ratio > maximum_area_ratio:
            reasons.append(
                "area_above_per_view_temporal_ratio"
                if reference is not None
                else "area_above_cross_view_ratio"
            )
        if (
            denominator > 0.0
            and minimum_area_ratio > 0.0
            and ratio < minimum_area_ratio
        ):
            reasons.append(
                "area_below_per_view_temporal_ratio"
                if reference is not None
                else "area_below_cross_view_ratio"
            )
        views[view] = {
            "valid": not reasons,
            "present": area >= int(presence_minimum_pixels),
            "pixels": area,
            "area_to_median": ratio,
            "area_ratio": ratio,
            "area_ratio_source": ratio_source,
            "reference_pixels": reference,
            "reasons": reasons,
        }
    return {
        "part_id": int(part_id),
        "median_positive_pixels": median,
        "minimum_pixels": int(minimum_pixels),
        "presence_minimum_pixels": int(presence_minimum_pixels),
        "maximum_area_ratio": float(maximum_area_ratio),
        "minimum_area_ratio": float(minimum_area_ratio),
        "area_reference_mode": (
            "per_view_temporal" if temporal_references else "cross_view"
        ),
        "reference_areas": temporal_references,
        "present_view_count": sum(row["present"] for row in views.values()),
        "valid_view_count": sum(row["valid"] for row in views.values()),
        "views": views,
    }


def valid_mask_views(
    masks: Mapping[str, np.ndarray],
    part_id: int,
    **kwargs: Any,
) -> tuple[list[str], dict[str, Any]]:
    """Return valid view names and the complete auditable quality report."""

    report = mask_area_quality(masks, part_id, **kwargs)
    return [
        view for view, row in report["views"].items() if row["valid"]
    ], report


def cloud_supported_view_quality(
    cloud_row: Mapping[str, Any] | None,
    views: list[str],
    *,
    minimum_supported_points: int = 30,
    minimum_support_fraction: float = 0.25,
) -> dict[str, Any]:
    """Gate cameras using cross-view-supported depth samples.

    Balanced masks alone do not prove that a camera sees the intended part:
    during hand occlusion a plausible label can still carry inconsistent
    depth.  The quality-cloud stage already records per-camera support from
    the other synchronized views.  Converting it into a view gate lets pose
    stages use only genuinely corroborated cameras.
    """

    if minimum_supported_points < 1:
        raise ValueError("minimum_supported_points must be positive")
    if not 0.0 <= minimum_support_fraction <= 1.0:
        raise ValueError("minimum_support_fraction must be in [0, 1]")
    source = dict(cloud_row or {})
    depth_rows = list(source.get("views", []))
    mask_rows = dict(source.get("mask_quality", {}).get("views", {}))
    gate = dict(source.get("quality_gate", {}) or {})
    frame_status = source.get("status")
    if "passed" in gate:
        frame_accepted: bool | None = bool(gate["passed"])
    elif frame_status is not None:
        frame_accepted = str(frame_status) == "ok"
    else:
        # Some focused diagnostics provide only per-view rows.  Preserve that
        # supported legacy use, but never reinterpret an explicit rejection as
        # an accepted pose frame.
        frame_accepted = None
    result: dict[str, dict[str, Any]] = {}
    for index, view in enumerate(views):
        depth = dict(depth_rows[index]) if index < len(depth_rows) else {}
        candidate = int(depth.get("candidate_points", 0) or 0)
        supported = int(depth.get("supported_points", 0) or 0)
        fraction = float(supported / candidate) if candidate > 0 else 0.0
        reasons = []
        if not mask_rows.get(view, {}).get("valid", True):
            reasons.append("mask_quality_failed")
        if supported < int(minimum_supported_points):
            reasons.append("too_few_cross_view_supported_points")
        if fraction < float(minimum_support_fraction):
            reasons.append("cross_view_support_fraction_below_minimum")
        result[str(view)] = {
            "valid": not reasons,
            "candidate_points": candidate,
            "supported_points": supported,
            "support_fraction": fraction,
            "reasons": reasons,
        }
    return {
        "available": bool(depth_rows),
        "frame_status": frame_status,
        "frame_accepted": frame_accepted,
        "frame_rejection_reasons": list(gate.get("reasons", [])),
        "minimum_supported_points": int(minimum_supported_points),
        "minimum_support_fraction": float(minimum_support_fraction),
        "valid_view_count": sum(row["valid"] for row in result.values()),
        "views": result,
    }


def part_visibility_quality(
    mask_report: Mapping[str, Any],
    views: list[str],
    *,
    cloud_report: Mapping[str, Any] | None = None,
    require_cloud_support: bool = False,
    minimum_pose_views: int = 1,
    minimum_visibility_views: int = 1,
) -> dict[str, Any]:
    """Combine silhouette and 3D support into an auditable visibility state.

    ``observing_views`` used to mean only that a label contained enough
    pixels.  That is unsafe around exits and hand occlusions: a tracker can
    leave a plausible-looking label on the hand or neighbouring part.  This
    helper keeps per-camera visibility and global pose validity separate.

    When cross-view support is explicitly required, missing support evidence
    fails closed.  Falling back to the mask in that case would recreate the
    exact false-presence failure this gate is intended to prevent.
    """

    if minimum_pose_views < 1:
        raise ValueError("minimum_pose_views must be positive")
    if minimum_visibility_views < 1:
        raise ValueError("minimum_visibility_views must be positive")
    mask_rows = dict(mask_report.get("views", {}))
    cloud = dict(cloud_report or {})
    cloud_rows = dict(cloud.get("views", {}))
    cloud_available = bool(cloud.get("available", False))
    cloud_frame_accepted = cloud.get("frame_accepted")
    rows: dict[str, dict[str, Any]] = {}
    mask_visible_views: list[str] = []
    visible_views: list[str] = []
    for view in views:
        mask_row = dict(mask_rows.get(view, {}))
        cloud_row = dict(cloud_rows.get(view, {}))
        reasons: list[str] = []
        mask_present = bool(
            mask_row.get(
                "present",
                int(mask_row.get("pixels", 0) or 0)
                >= int(mask_report.get("minimum_pixels", 1)),
            )
        )
        if mask_present:
            mask_visible_views.append(str(view))
        mask_valid = bool(mask_row.get("valid", False))
        if not mask_valid:
            reasons.extend(
                str(value)
                for value in mask_row.get("reasons", ["mask_missing"])
            )
        if require_cloud_support:
            if not cloud_available:
                reasons.append("cross_view_support_unavailable")
            elif not bool(cloud_row.get("valid", False)):
                reasons.extend(
                    str(value)
                    for value in cloud_row.get(
                        "reasons", ["cross_view_support_failed"]
                    )
                )
        valid = not reasons
        if valid:
            visible_views.append(str(view))
        rows[str(view)] = {
            "valid": valid,
            "mask_present": mask_present,
            "mask_valid": mask_valid,
            "cloud_supported": (
                bool(cloud_row.get("valid", False))
                if cloud_available
                else None
            ),
            "mask_pixels": int(mask_row.get("pixels", 0) or 0),
            "supported_points": int(
                cloud_row.get("supported_points", 0) or 0
            ),
            "support_fraction": float(
                cloud_row.get("support_fraction", 0.0) or 0.0
            ),
            "reasons": reasons,
        }
    frame_reasons: list[str] = []
    if (
        require_cloud_support
        and cloud_available
        and cloud_frame_accepted is False
    ):
        frame_reasons.append("cloud_frame_quality_failed")
        frame_reasons.extend(
            str(value)
            for value in cloud.get("frame_rejection_reasons", [])
        )
    pose_valid = (
        len(visible_views) >= int(minimum_pose_views)
        and not frame_reasons
    )
    render_valid = len(mask_visible_views) >= int(minimum_visibility_views)
    if not mask_visible_views:
        observation_state = "out_of_frame"
    elif pose_valid:
        observation_state = "observed_reliable"
    elif require_cloud_support:
        observation_state = "visible_cloud_unreliable"
    else:
        observation_state = "occluded_or_partial"
    return {
        "pose_valid": pose_valid,
        "tracking_valid": pose_valid,
        "render_valid": render_valid,
        "observation_state": observation_state,
        "observing_views": len(visible_views),
        "visible_views": visible_views,
        "reliable_views": visible_views,
        "mask_visible_views": mask_visible_views,
        "mask_visible_view_count": len(mask_visible_views),
        "minimum_pose_views": int(minimum_pose_views),
        "minimum_visibility_views": int(minimum_visibility_views),
        "require_cloud_support": bool(require_cloud_support),
        "cloud_support_available": cloud_available,
        "cloud_frame_accepted": cloud_frame_accepted,
        "frame_reasons": frame_reasons,
        "views": rows,
    }
