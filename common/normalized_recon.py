"""Load normalized-layout recon, masks, and output paths under outputs/normalized/."""
from __future__ import annotations

import json
import os
from typing import Any, Literal

import cv2
import numpy as np

from common.mask_io import frame_path, list_timestamps, view_names

ReconBackend = Literal["da3_self_cond", "da3_vggt_cond", "vggt_omega"]
SUPPORTED_BACKENDS = ("da3_self_cond", "da3_vggt_cond", "vggt_omega")

BACKEND_DIR_KEYS = {
    "da3_self_cond": "da3_self_cond_dir",
    "da3_vggt_cond": "da3_vggt_cond_dir",
    "vggt_omega": "vggt_omega_dir",
}


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def output_root(cfg: dict) -> str:
    return cfg.get(
        "output_root",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "normalized"),
    )


def resolve_backend(cfg: dict, override: str | None = None) -> ReconBackend:
    backend = (override or cfg.get("recon_backend", "da3_self_cond")).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}")
    return backend  # type: ignore[return-value]


def recon_dir(cfg: dict, backend: ReconBackend | None = None) -> str:
    backend = resolve_backend(cfg, backend)
    key = BACKEND_DIR_KEYS[backend]
    if key not in cfg:
        raise KeyError(f"{key} missing from pipeline config")
    return cfg[key]


def parts_ply_dir(cfg: dict, backend: ReconBackend | None = None) -> str:
    return os.path.join(output_root(cfg), "parts_ply", resolve_backend(cfg, backend))


def icp_out_dir(cfg: dict, backend: ReconBackend | None = None) -> str:
    return os.path.join(output_root(cfg), "icp", resolve_backend(cfg, backend))


def proj_vis_dir(cfg: dict, backend: ReconBackend | None = None) -> str:
    return os.path.join(output_root(cfg), "proj_vis", resolve_backend(cfg, backend))


def masks_dir(cfg: dict) -> str:
    return cfg["masks_dir"]


def all_timestamps(cfg: dict) -> list[str]:
    return list_timestamps(
        cfg["frames_dir"],
        cfg.get("frames_layout", "normalized"),
        view_names(cfg),
    )


def sample_timestamps() -> list[str]:
    """Every 3rd frame in segments 31-40, 67-88, 98-110."""
    out: list[str] = []
    for start, end in ((31, 40), (67, 88), (98, 110)):
        for i in range(start, end + 1, 3):
            out.append(f"{i:06d}")
    return out


def recon_npz_path(cfg: dict, timestamp: str, backend: ReconBackend | None = None) -> str:
    return os.path.join(recon_dir(cfg, backend), timestamp, "predictions.npz")


def load_recon(cfg: dict, timestamp: str, backend: ReconBackend | None = None) -> dict[str, Any]:
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
    images = d["image"] if "image" in d.files else (d["images"] if "images" in d.files else None)
    return {
        "path": path,
        "backend": resolve_backend(cfg, backend),
        "depth": depth.astype(np.float32),
        "conf": conf.astype(np.float32),
        "intrinsics": K.astype(np.float64),
        "extrinsics": E.astype(np.float64),
        "images": images,
        "n_views": int(depth.shape[0]),
        "depth_hw": tuple(depth.shape[1:3]),
    }


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


def load_fullres_frames(cfg: dict, timestamp: str) -> np.ndarray:
    frames = []
    for vname in view_names(cfg):
        path = frame_path(cfg["frames_dir"], cfg.get("frames_layout", "normalized"), timestamp, vname)
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def load_view_bundle(cfg: dict, timestamp: str, backend: ReconBackend | None = None) -> dict[str, Any]:
    recon = load_recon(cfg, timestamp, backend=backend)
    frames = load_fullres_frames(cfg, timestamp)
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


def read_ply(path: str) -> np.ndarray:
    pts = []
    with open(path, encoding="ascii") as f:
        in_data = False
        for line in f:
            if line.strip() == "end_header":
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(pts, dtype=np.float64)
