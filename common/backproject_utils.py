"""Backproject helpers: palette masks + part cloud fusion."""
from __future__ import annotations

import os

import cv2
import numpy as np

from common.geom import backproject_view
from common.mask_io import PART_COLORS, VIEW_NAMES, part_id_map


def load_palette_masks(
    masks_root: str,
    timestamp: str,
    parts: list[str],
    depth_hw: tuple[int, int],
) -> dict[str, list[np.ndarray]]:
    """Load per-view boolean masks resized to depth resolution."""
    h, w = depth_hw
    ids = part_id_map(parts)
    out: dict[str, list[np.ndarray]] = {p: [] for p in parts}
    for vname in VIEW_NAMES:
        path = os.path.join(masks_root, timestamp, f"{vname}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"cannot read {path}")
        if image.ndim == 2:
            small = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
            for part in parts:
                out[part].append(small == ids[part])
        else:
            small = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
            for part in parts:
                bgr = np.array(PART_COLORS[part][::-1], dtype=np.uint8)
                out[part].append(np.all(small[:, :, :3] == bgr, axis=2))
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
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Fuse one part across all views. Returns pts, cols, per-view stats."""
    if global_conf_thr is None:
        global_conf_thr = float(np.median(conf))
    all_pts, all_cols = [], []
    stats: dict[str, int] = {}
    for v in range(depth.shape[0]):
        m = part_masks[v].astype(bool)
        m &= np.isfinite(depth[v]) & (depth[v] > 1e-3)
        thr = conf_threshold(conf[v], part_masks[v], conf_mode, global_conf_thr, conf_quantile)
        m &= conf[v] >= thr
        sub = np.zeros_like(m)
        sub[::stride, ::stride] = True
        m &= sub
        stats[VIEW_NAMES[v]] = int(m.sum())
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
    for vname in VIEW_NAMES:
        path = frame_path(cfg["frames_dir"], cfg.get("frames_layout", "normalized"), timestamp, vname)
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        colors.append(small)
    return np.stack(colors, axis=0)
