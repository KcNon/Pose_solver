"""Per-frame depth gauge anchored on a static reference part.

The recon depth of each frame carries a per-view global shift of up to ~9 mm
while the pixel-level residual is sub-millimetre (measured by
``tools/diagnostics/diagnose_depth_stability.py``). Anchoring every frame's depth to
the temporal median depth of the static reference part removes that shift
before backprojection, so the per-part clouds inherit the residual noise
floor instead of the frame-to-frame drift.

The gauge only needs the reference part to be static and visible in most
frames of each view; pixels are compared against their own temporal median,
so partial occlusion in any single frame is tolerated.
"""
from __future__ import annotations

import cv2
import numpy as np

from common.backproject_utils import load_palette_masks
from common.normalized_recon import load_recon


def reference_part(cfg: dict, configured: str | None = None) -> str:
    """Resolve the static gauge part without assuming an object taxonomy."""

    selected = configured or cfg.get("reference_part")
    if selected:
        return str(selected)
    parts = cfg.get("parts", [])
    names = list(parts) if isinstance(parts, (list, dict)) else []
    if not names:
        raise ValueError(
            "depth gauge requires reference_part or a non-empty parts list"
        )
    return str(names[0])


def compute_depth_gauge(
    cfg: dict,
    backend: str,
    timestamps: list[str],
    part: str | None = None,
    *,
    min_support: int = 30,
    min_pixels: int = 300,
    erode: int = 2,
) -> dict:
    """Estimate one additive depth shift per frame and view.

    ``min_support`` is the minimum number of frames a pixel must be visible in
    before its temporal median counts as reference.  Frames whose visible
    overlap with the reference falls below ``min_pixels`` get an interpolated
    shift instead of a measured one.
    """
    part = reference_part(cfg, part)
    kernel = np.ones((3, 3), np.uint8)

    depth_stack = None
    valid_stack = None
    for index, timestamp in enumerate(timestamps):
        recon = load_recon(cfg, timestamp, backend=backend)
        depth = recon["depth"]
        if depth_stack is None:
            depth_stack = np.empty((len(timestamps),) + depth.shape, np.float32)
            valid_stack = np.zeros((len(timestamps),) + depth.shape, bool)
        depth_stack[index] = depth
        masks = load_palette_masks(
            cfg["masks_dir"], timestamp, [part], recon["depth_hw"],
            views=cfg.get("views"),
        )[part]
        for v in range(depth.shape[0]):
            mask = cv2.erode(masks[v].astype(np.uint8), kernel, iterations=erode).astype(bool)
            valid_stack[index, v] = mask & np.isfinite(depth[v]) & (depth[v] > 1e-3)

    n_frames, n_views = depth_stack.shape[:2]
    shifts = np.full((n_frames, n_views), np.nan)
    n_used = np.zeros((n_frames, n_views), int)
    for v in range(n_views):
        masked = np.where(valid_stack[:, v], depth_stack[:, v], np.nan)
        support = valid_stack[:, v].sum(axis=0)
        reference = np.nanmedian(masked, axis=0)
        usable = (support >= min_support) & np.isfinite(reference)
        for f in range(n_frames):
            overlap = valid_stack[f, v] & usable
            n_used[f, v] = int(overlap.sum())
            if n_used[f, v] >= min_pixels:
                shifts[f, v] = float(np.median(depth_stack[f, v][overlap] - reference[overlap]))

    # Bridge frames without a reliable measurement by linear interpolation.
    interpolated = np.zeros_like(shifts, bool)
    for v in range(n_views):
        series = shifts[:, v]
        missing = np.isnan(series)
        if missing.all():
            shifts[:, v] = 0.0
            interpolated[:, v] = True
            continue
        if missing.any():
            index = np.arange(n_frames)
            shifts[missing, v] = np.interp(index[missing], index[~missing], series[~missing])
            interpolated[missing, v] = True

    return {
        "part": part,
        "backend": backend,
        "min_support": min_support,
        "min_pixels": min_pixels,
        "shift_std_mm": [float(np.std(shifts[:, v]) * 1000) for v in range(n_views)],
        "shift_range_mm": [[float(shifts[:, v].min() * 1000), float(shifts[:, v].max() * 1000)]
                           for v in range(n_views)],
        "frames": {
            timestamp: {
                "shift_m": [float(s) for s in shifts[f]],
                "n_pixels": [int(n) for n in n_used[f]],
                "interpolated": [bool(b) for b in interpolated[f]],
            }
            for f, timestamp in enumerate(timestamps)
        },
    }


def compute_view_bias(
    cfg: dict,
    backend: str,
    timestamps: list[str],
    part: str | None = None,
    gauge: dict | None = None,
    *,
    iterations: int = 6,
    damping: float = 0.6,
    max_points: int = 4000,
    seed: int = 0,
) -> dict:
    """Estimate one constant additive depth bias per view from cross-view overlap.

    The temporal gauge aligns each view to its own history; it cannot see that
    two cameras place the same static surface 10 mm apart.  For each view the
    signed along-ray offset to the other views' fused surface is measured and
    the per-view biases are relaxed jointly.  Biases are zero-meaned: the
    common component is a global scale/depth choice that the downstream anchor
    calibration absorbs anyway.
    """
    from scipy.spatial import cKDTree

    from common.geom import backproject_view

    part = reference_part(cfg, part)
    rng = np.random.default_rng(seed)

    per_frame = []
    for timestamp in timestamps:
        recon = load_recon(cfg, timestamp, backend=backend)
        depth = recon["depth"]
        if gauge is not None:
            depth = apply_depth_gauge(depth, gauge, timestamp)
        masks = load_palette_masks(
            cfg["masks_dir"], timestamp, [part], recon["depth_hw"],
            views=cfg.get("views"),
        )[part]
        clouds, rays = [], []
        for v in range(depth.shape[0]):
            mask = masks[v] & np.isfinite(depth[v]) & (depth[v] > 1e-3)
            if mask.sum() < 200:
                clouds.append(None)
                rays.append(None)
                continue
            points, _ = backproject_view(depth[v], recon["intrinsics"][v],
                                         recon["extrinsics"][v], mask=mask, color=None)
            if len(points) > max_points:
                points = points[rng.choice(len(points), max_points, replace=False)]
            E = recon["extrinsics"][v]
            center = -E[:3, :3].T @ E[:3, 3]
            direction = points - center
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            clouds.append(points)
            rays.append(direction)
        live = [v for v, c in enumerate(clouds) if c is not None]
        if len(live) < 3:
            continue
        bias = np.zeros(len(clouds))
        for _ in range(iterations):
            for v in live:
                moved = {u: clouds[u] + bias[u] * rays[u] for u in live}
                reference = np.concatenate([moved[u] for u in live if u != v])
                tree = cKDTree(reference)
                source = moved[v]
                _, index = tree.query(source, k=1)
                signed = np.einsum("ij,ij->i", reference[index] - source, rays[v])
                bias[v] += damping * float(np.median(signed))
            bias -= bias.mean()
        per_frame.append(bias)

    if not per_frame:
        raise RuntimeError(f"no frame had >=3 views of {part} for cross-view bias")
    stacked = np.stack(per_frame)
    bias = np.median(stacked, axis=0)
    bias -= bias.mean()
    return {
        "part": part,
        "n_frames_used": int(len(per_frame)),
        "view_bias_m": [float(b) for b in bias],
        "per_frame_spread_mm": [float(np.std(stacked[:, v]) * 1000)
                                for v in range(stacked.shape[1])],
    }


def load_depth_gauge(path: str) -> dict:
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_depth_gauge(depth: np.ndarray, gauge: dict, timestamp: str) -> np.ndarray:
    """Return depth with the per-view frame shift (and cross-view bias, when
    calibrated) removed; zeros stay zero."""
    entry = gauge["frames"].get(timestamp)
    if entry is None:
        raise KeyError(f"depth gauge has no entry for frame {timestamp}")
    shifts = np.asarray(entry["shift_m"], np.float32)
    if len(shifts) != depth.shape[0]:
        raise ValueError(f"gauge has {len(shifts)} views, depth has {depth.shape[0]}")
    if "view_bias_m" in gauge:
        shifts = shifts + np.asarray(gauge["view_bias_m"], np.float32)
    corrected = depth - shifts[:, None, None]
    return np.where(depth > 1e-3, corrected, depth)
