"""Utilities for removing thin deformable appendages from rigid-pose evidence.

The operations in this module are deliberately geometric and object agnostic.
Object-specific choices (image-space thickness and a mesh-space halfspace) live
in experiment configuration, while the original masks and meshes remain
untouched.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


AXES = {"x": 0, "y": 1, "z": 2}


def thick_core_region(
    mask: np.ndarray,
    *,
    erosion_radius: int,
    restore_radius: int,
    minimum_core_pixels: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep thick regions of a binary mask and restore their outer boundary.

    Erosion removes appendages whose local half-thickness is below
    ``erosion_radius``.  The surviving cores are then dilated inside the
    original mask.  A restore radius larger than the erosion radius recovers
    the rigid component's silhouette without reconnecting a long thin tube.

    If no sufficiently large core survives, the result is empty.  Treating
    that observation as unavailable is safer than allowing a thin fragment to
    determine a six-degree-of-freedom pose.
    """

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if erosion_radius < 1:
        raise ValueError("erosion_radius must be positive")
    if restore_radius < erosion_radius:
        raise ValueError("restore_radius must be at least erosion_radius")
    if minimum_core_pixels < 1:
        raise ValueError("minimum_core_pixels must be positive")

    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * erosion_radius + 1, 2 * erosion_radius + 1),
    )
    eroded = cv2.erode(source.astype(np.uint8), erosion_kernel) > 0
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        eroded.astype(np.uint8), connectivity=8
    )
    retained_core = np.zeros_like(source)
    retained_components = 0
    for label in range(1, component_count):
        pixels = int(stats[label, cv2.CC_STAT_AREA])
        if pixels < minimum_core_pixels:
            continue
        retained_core |= labels == label
        retained_components += 1

    if not np.any(retained_core):
        return np.zeros_like(source), {
            "source_pixels": int(source.sum()),
            "core_pixels": 0,
            "output_pixels": 0,
            "retained_fraction": 0.0,
            "retained_components": 0,
            "status": "no_rigid_core",
        }

    restore_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * restore_radius + 1, 2 * restore_radius + 1),
    )
    restored = cv2.dilate(retained_core.astype(np.uint8), restore_kernel) > 0
    result = source & restored
    source_pixels = int(source.sum())
    output_pixels = int(result.sum())
    return result, {
        "source_pixels": source_pixels,
        "core_pixels": int(retained_core.sum()),
        "output_pixels": output_pixels,
        "retained_fraction": (
            float(output_pixels / source_pixels) if source_pixels else 0.0
        ),
        "retained_components": retained_components,
        "status": "ok" if output_pixels else "no_rigid_core",
    }


def pose_guided_rigid_region(
    source_mask: np.ndarray,
    rendered_rigid_mask: np.ndarray,
    *,
    dilation_radius: int,
    minimum_output_pixels: int = 25,
    minimum_render_overlap: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep observed pixels supported by a dilated rigid-mesh projection.

    This complements thickness filtering when a flexible appendage projects
    wider than a thin rigid trigger. The projection supplies only a spatial
    prior: final boundaries still come from the observed mask. A low-overlap
    pose is rejected instead of silently producing a truncated observation.
    """

    source = np.asarray(source_mask, dtype=bool)
    rendered = np.asarray(rendered_rigid_mask, dtype=bool)
    if source.shape != rendered.shape:
        raise ValueError("source and rendered masks must have the same shape")
    radius = int(dilation_radius)
    if radius < 0:
        raise ValueError("dilation_radius must be non-negative")
    if radius:
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        support = cv2.dilate(rendered.astype(np.uint8), kernel).astype(bool)
    else:
        support = rendered
    output = source & support
    overlap_denominator = max(
        1, min(int(source.sum()), int(rendered.sum()))
    )
    render_overlap = float((source & rendered).sum() / overlap_denominator)
    accepted = bool(
        int(output.sum()) >= int(minimum_output_pixels)
        and render_overlap >= float(minimum_render_overlap)
    )
    if not accepted:
        output = np.zeros_like(source)
    return output, {
        "source_pixels": int(source.sum()),
        "rendered_pixels": int(rendered.sum()),
        "output_pixels": int(output.sum()),
        "retained_fraction": (
            float(output.sum() / source.sum()) if source.any() else 0.0
        ),
        "render_overlap": render_overlap,
        "status": "ok" if accepted else "rejected_pose_guide",
    }


def halfspace_face_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    axis: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    """Select mesh faces whose centroids lie inside an axis-aligned slab."""

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if axis not in AXES:
        raise ValueError(f"axis must be one of {sorted(AXES)}")
    if minimum is None and maximum is None:
        raise ValueError("at least one halfspace bound is required")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum must not exceed maximum")

    coordinate = points[triangles].mean(axis=1)[:, AXES[axis]]
    keep = np.ones(len(triangles), dtype=bool)
    if minimum is not None:
        keep &= coordinate >= float(minimum)
    if maximum is not None:
        keep &= coordinate <= float(maximum)
    return keep
