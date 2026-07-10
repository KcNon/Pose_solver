"""Load normalized recon (multi-backend) + mask/frame paths."""
from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

from common.mask_io import VIEW_NAMES, frame_path, part_id_map

BACKEND_DIRS = {
    "da3_vggt_cond": "da3_vggt_cond_dir",
    "da3_self_cond": "da3_self_cond_dir",
    "vggt_omega": "vggt_omega_dir",
}

SAMPLE_SEGMENTS = [
    list(range(31, 41, 3)),
    list(range(67, 89, 3)),
    list(range(98, 111, 3)),
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def output_root(cfg: dict) -> str:
    return cfg.get("output_root", os.path.join(ROOT, "outputs", "normalized"))


def masks_dir(cfg: dict) -> str:
    return cfg.get("masks_dir", os.path.join(output_root(cfg), "masks"))


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_backend(cfg: dict, backend: str | None = None) -> str:
    backend = backend or cfg.get("recon_backend", "da3_self_cond")
    if backend not in BACKEND_DIRS:
        raise ValueError(f"unknown backend {backend!r}, choose from {list(BACKEND_DIRS)}")
    key = BACKEND_DIRS[backend]
    if key not in cfg:
        raise KeyError(f"pipeline missing {key} for backend {backend}")
    return backend


def recon_dir(cfg: dict, backend: str | None = None) -> str:
    backend = resolve_backend(cfg, backend)
    return cfg[BACKEND_DIRS[backend]]


def recon_npz_path(cfg: dict, timestamp: str, backend: str | None = None) -> str:
    return os.path.join(recon_dir(cfg, backend), timestamp, "predictions.npz")


def parts_ply_dir(cfg: dict, backend: str | None = None) -> str:
    backend = resolve_backend(cfg, backend)
    return os.path.join(output_root(cfg), "parts_ply", backend)


def icp_out_dir(cfg: dict, backend: str | None = None) -> str:
    backend = resolve_backend(cfg, backend)
    return os.path.join(output_root(cfg), "icp", backend)


def proj_vis_dir(cfg: dict, backend: str | None = None) -> str:
    backend = resolve_backend(cfg, backend)
    return os.path.join(output_root(cfg), "proj_vis", backend)


def sample_timestamps(segments: list[list[int]] | None = None) -> list[str]:
    segments = segments or SAMPLE_SEGMENTS
    out: list[str] = []
    for seg in segments:
        for i in seg:
            out.append(f"{i:06d}")
    return out


def sample_segments(segments: list[list[int]] | None = None) -> list[list[str]]:
    segments = segments or SAMPLE_SEGMENTS
    return [[f"{i:06d}" for i in seg] for seg in segments]


def load_recon(cfg: dict, timestamp: str, backend: str | None = None) -> dict[str, Any]:
    path = recon_npz_path(cfg, timestamp, backend)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path)
    depth = d["depth"]
    if depth.ndim == 4:
        depth = depth[..., 0]
    conf = d["depth_conf"] if "depth_conf" in d.files else d["conf"]
    K = d["intrinsic"] if "intrinsic" in d.files else d["intrinsics"]
    E = d["extrinsic"] if "extrinsic" in d.files else d["extrinsics"]
    images = d["image"] if "image" in d.files else d.get("images")
    return {
        "path": path,
        "backend": resolve_backend(cfg, backend),
        "depth": depth,
        "conf": conf,
        "intrinsics": K,
        "extrinsics": E,
        "images": images,
        "n_views": depth.shape[0],
        "depth_hw": tuple(depth.shape[1:3]),
    }


def load_palette_masks(masks_dir: str, timestamp: str, parts: list[str]) -> dict[str, list[np.ndarray]]:
    ids = part_id_map(parts)
    inv = {v: k for k, v in ids.items()}
    out: dict[str, list[np.ndarray]] = {p: [] for p in parts}
    ts_dir = os.path.join(masks_dir, timestamp)
    for vname in VIEW_NAMES:
        path = os.path.join(ts_dir, f"{vname}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        label = np.array(Image.open(path))
        for pid, pname in inv.items():
            out[pname].append(label == pid)
    return out


def load_frame_colors(cfg: dict, timestamp: str, depth_hw: tuple[int, int]) -> np.ndarray:
    frames_dir = cfg["frames_dir"]
    h, w = depth_hw
    colors = []
    for vname in VIEW_NAMES:
        path = frame_path(frames_dir, "normalized", timestamp, vname)
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        colors.append(small)
    return np.stack(colors, axis=0)


def load_recon_colors(cfg: dict, timestamp: str, recon: dict[str, Any]) -> np.ndarray:
    if recon.get("images") is not None:
        imgs = np.asarray(recon["images"])
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        if imgs.dtype != np.uint8:
            imgs = np.clip(imgs, 0, 255).astype(np.uint8)
        return imgs
    return load_frame_colors(cfg, timestamp, recon["depth_hw"])


def scale_intrinsics(K: np.ndarray, from_hw: tuple[int, int], to_hw: tuple[int, int]) -> np.ndarray:
    Hf, Wf = from_hw
    Ht, Wt = to_hw
    K2 = np.asarray(K, dtype=np.float64).copy()
    sx, sy = Wt / Wf, Ht / Hf
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def resize_depth_to(depth: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def load_view_bundle(cfg: dict, timestamp: str, backend: str | None = None) -> dict[str, Any]:
    recon = load_recon(cfg, timestamp, backend=backend)
    frames = []
    for vname in VIEW_NAMES:
        path = frame_path(cfg["frames_dir"], "normalized", timestamp, vname)
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    frames = np.stack(frames, axis=0)
    H, W = frames.shape[1:3]
    depth_hw = recon["depth_hw"]
    K_scaled = np.stack([
        scale_intrinsics(recon["intrinsics"][v], depth_hw, (H, W))
        for v in range(recon["n_views"])
    ], axis=0)
    depth_scaled = np.stack([
        resize_depth_to(recon["depth"][v], (H, W))
        for v in range(recon["n_views"])
    ], axis=0)
    return {
        "backend": recon["backend"],
        "images": frames,
        "depth": depth_scaled,
        "intrinsics": K_scaled,
        "extrinsics": recon["extrinsics"],
    }


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    pts, cols = [], None
    col_buf = []
    has_color = False
    with open(path, encoding="ascii") as f:
        header = True
        for line in f:
            if header:
                if "property uchar red" in line:
                    has_color = True
                if line.strip() == "end_header":
                    header = False
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if has_color and len(parts) >= 6:
                col_buf.append([int(parts[3]), int(parts[4]), int(parts[5])])
    if col_buf:
        cols = np.asarray(col_buf, dtype=np.uint8)
    return np.asarray(pts, dtype=np.float32), cols
