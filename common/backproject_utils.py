"""Backproject helpers: palette masks + part cloud fusion."""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

from common.geom import backproject_view
from common.mask_io import VIEW_NAMES, part_id_map, parts_meta, view_names


def load_palette_masks(
    masks_root: str,
    timestamp: str,
    parts: list[str],
    depth_hw: tuple[int, int],
    views: list[str] | None = None,
    part_ids: dict[str, int] | None = None,
    resize_mode: str = "nearest",
    coverage_threshold: float = 0.25,
    coverage_parts: list[str] | None = None,
) -> dict[str, list[np.ndarray]]:
    """Load per-view boolean masks resized to depth resolution.

    ``coverage`` downsamples each part independently with area integration and
    assigns a target pixel to the part with greatest coverage.  It preserves
    thin structures that can disappear under categorical nearest-neighbour
    sampling while keeping the resulting part masks mutually exclusive.
    """
    h, w = depth_hw
    if resize_mode not in {"nearest", "coverage", "hybrid"}:
        raise ValueError(
            "mask resize_mode must be 'nearest', 'coverage', or 'hybrid'"
        )
    if not 0.0 < float(coverage_threshold) <= 1.0:
        raise ValueError("mask coverage_threshold must be in (0, 1]")
    selected_coverage_parts = set(coverage_parts or parts)
    unknown_coverage_parts = selected_coverage_parts.difference(parts)
    if unknown_coverage_parts:
        raise ValueError(
            f"unknown coverage mask parts: {sorted(unknown_coverage_parts)}"
        )
    ids = (
        {part: int(part_ids[part]) for part in parts}
        if part_ids is not None
        else part_id_map(parts)
    )
    colors = {name: meta["color"] for name, meta in parts_meta(parts).items()}
    out: dict[str, list[np.ndarray]] = {p: [] for p in parts}
    for vname in (views or VIEW_NAMES):
        path = os.path.join(masks_root, timestamp, f"{vname}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        # PIL preserves the integer indices of mode-P masks.  OpenCV expands
        # those files to BGR and can alias distinct labels whose display
        # palette happens to reuse a legacy colour (for example blade/lid).
        image = np.asarray(Image.open(path))
        if image.ndim == 2:
            full_masks = [image == ids[part] for part in parts]
        else:
            full_masks = []
            for part in parts:
                rgb = np.array(colors[part], dtype=np.uint8)
                full_masks.append(np.all(image[:, :, :3] == rgb, axis=2))
        nearest = [
            cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for mask in full_masks
        ]
        if resize_mode == "nearest":
            resized = nearest
        else:
            interpolation = (
                cv2.INTER_AREA
                if h <= image.shape[0] and w <= image.shape[1]
                else cv2.INTER_LINEAR
            )
            coverage = np.stack([
                cv2.resize(
                    mask.astype(np.float32),
                    (w, h),
                    interpolation=interpolation,
                )
                for mask in full_masks
            ])
            if resize_mode == "coverage":
                winner = np.argmax(coverage, axis=0)
                maximum = np.max(coverage, axis=0)
                resized = [
                    (winner == index) & (
                        maximum >= float(coverage_threshold)
                    )
                    for index in range(len(parts))
                ]
            else:
                # Preserve every categorical nearest-neighbour assignment, then
                # let selected thin parts claim only otherwise-background cells.
                resized = [mask.copy() for mask in nearest]
                occupied = np.logical_or.reduce(resized)
                selected = [
                    index for index, part in enumerate(parts)
                    if part in selected_coverage_parts
                ]
                if selected:
                    selected_coverage = coverage[selected]
                    winner = np.argmax(selected_coverage, axis=0)
                    maximum = np.max(selected_coverage, axis=0)
                    eligible = (~occupied) & (
                        maximum >= float(coverage_threshold)
                    )
                    for local_index, part_index in enumerate(selected):
                        resized[part_index] |= eligible & (
                            winner == local_index
                        )
        for part, mask in zip(parts, resized):
            out[part].append(mask)
    return out


def conf_threshold(
    conf: np.ndarray,
    mask: np.ndarray,
    mode: str,
    global_thr: float,
    quantile: float,
) -> float:
    """Return confidence cutoff for one view + part mask."""
    geometric = mask & np.isfinite(conf) & (conf > 0)
    values = conf[geometric]
    if len(values) == 0:
        return global_thr
    if mode == "adaptive":
        return float(np.quantile(values, quantile))
    return global_thr


def fuse_part_cloud(
    depth: np.ndarray,
    img: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    conf: np.ndarray,
    part_masks: list[np.ndarray],
    *,
    conf_mode: str = "global",
    conf_quantile: float = 0.25,
    global_conf_thr: float | None = None,
    stride: int = 2,
    max_pts: int = 80000,
    seed: int = 0,
    views: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Fuse one part across all views. Returns pts, cols, per-view stats."""
    if global_conf_thr is None:
        global_conf_thr = float(np.median(conf))
    all_pts, all_cols = [], []
    stats: dict[str, int] = {}
    ordered_views = views or VIEW_NAMES
    if len(ordered_views) != depth.shape[0]:
        raise ValueError(
            f"configured {len(ordered_views)} views but reconstruction has "
            f"{depth.shape[0]}"
        )
    for v in range(depth.shape[0]):
        m = part_masks[v].astype(bool)
        m &= np.isfinite(depth[v]) & (depth[v] > 1e-3)
        thr = conf_threshold(conf[v], part_masks[v], conf_mode, global_conf_thr, conf_quantile)
        m &= conf[v] >= thr
        sub = np.zeros_like(m)
        sub[::stride, ::stride] = True
        m &= sub
        stats[ordered_views[v]] = int(m.sum())
        if m.sum() == 0:
            continue
        pts, cols = backproject_view(depth[v], K[v], E[v], mask=m, color=img[v])
        all_pts.append(pts)
        all_cols.append(cols)
    if not all_pts:
        return np.empty((0, 3), np.float32), None, stats
    pts = np.concatenate(all_pts, 0)
    cols = np.concatenate(all_cols, 0)
    if len(pts) > max_pts:
        idx = np.random.default_rng(seed).choice(len(pts), max_pts, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols, stats


def load_recon_colors(recon: dict, cfg: dict, timestamp: str) -> np.ndarray:
    """Per-view RGB uint8 at depth resolution."""
    depth_hw = recon["depth_hw"]
    h, w = depth_hw
    if recon.get("images") is not None:
        raw = recon["images"]
        if raw.ndim == 4 and raw.shape[1] == 3:  # NCHW
            colors = []
            for v in range(raw.shape[0]):
                chw = raw[v]
                if chw.max() <= 1.5:
                    rgb = (np.transpose(chw, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
                else:
                    rgb = np.transpose(chw, (1, 2, 0)).astype(np.uint8)
                if rgb.shape[:2] != (h, w):
                    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
                colors.append(rgb)
            return np.stack(colors, axis=0)
    # fallback: resize full-res frames
    from common.mask_io import frame_path

    colors = []
    for vname in view_names(cfg):
        path = frame_path(cfg["frames_dir"], cfg.get("frames_layout", "normalized"), timestamp, vname)
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        colors.append(small)
    return np.stack(colors, axis=0)
