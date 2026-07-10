"""Unified depth/camera loader for VGGT-Omega and DA3 reconstructions."""
from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np

ReconBackend = Literal["vggt", "da3"]
SUPPORTED_BACKENDS = ("vggt", "da3")


def resolve_backend(cfg: dict, override: str | None = None) -> ReconBackend:
    backend = (override or cfg.get("recon_backend", "vggt")).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"recon_backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}")
    return backend  # type: ignore[return-value]


def vggt_npz_path(vggt_dir: str, timestamp: str) -> str:
    return os.path.join(vggt_dir, timestamp, "predictions.npz")


def da3_npz_path(da3_dir: str, timestamp: str) -> str:
    return os.path.join(da3_dir, timestamp, "exports", "npz", "results.npz")


def recon_npz_path(cfg: dict, timestamp: str, backend: ReconBackend | None = None) -> str:
    backend = resolve_backend(cfg, backend)
    if backend == "vggt":
        return vggt_npz_path(cfg["vggt_dir"], timestamp)
    return da3_npz_path(cfg["da3_dir"], timestamp)


def output_paths(root: str, backend: ReconBackend, timestamp: str) -> dict[str, str]:
    """Per-backend output dirs so vggt/da3 results can coexist."""
    return {
        "masks": os.path.join(root, "outputs", "masks", backend, timestamp),
        "parts_ply": os.path.join(root, "outputs", "parts_ply", backend, timestamp),
    }


def load_vggt_recon(vggt_dir: str, timestamp: str) -> dict[str, Any]:
    path = vggt_npz_path(vggt_dir, timestamp)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path)
    depth = d["depth"]
    if depth.ndim == 4:
        depth = depth[..., 0]
    conf = d["depth_conf"] if "depth_conf" in d.files else d.get("conf")
    K = d["intrinsic"] if "intrinsic" in d.files else d["intrinsics"]
    E = d["extrinsic"] if "extrinsic" in d.files else d["extrinsics"]
    return {
        "path": path,
        "backend": "vggt",
        "depth": depth,
        "conf": conf,
        "intrinsics": K,
        "extrinsics": E,
        "images": None,
        "n_views": depth.shape[0],
        "depth_hw": tuple(depth.shape[1:3]),
    }


def load_da3_recon(da3_dir: str, timestamp: str) -> dict[str, Any]:
    path = da3_npz_path(da3_dir, timestamp)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path)
    depth = d["depth"]
    if depth.ndim == 4:
        depth = depth[..., 0]
    conf = d["conf"]
    K = d["intrinsics"]
    E = d["extrinsics"]
    images = d["image"] if "image" in d.files else None
    return {
        "path": path,
        "backend": "da3",
        "depth": depth,
        "conf": conf,
        "intrinsics": K,
        "extrinsics": E,
        "images": images,
        "n_views": depth.shape[0],
        "depth_hw": tuple(depth.shape[1:3]),
    }


def load_recon(cfg: dict, timestamp: str, backend: str | None = None) -> dict[str, Any]:
    """Load normalized reconstruction for one timestamp.

    cfg keys: recon_backend, vggt_dir, da3_dir
    CLI/scripts can override backend without editing pipeline.json.
    """
    backend = resolve_backend(cfg, backend)
    if backend == "vggt":
        return load_vggt_recon(cfg["vggt_dir"], timestamp)
    return load_da3_recon(cfg["da3_dir"], timestamp)


def load_frame_colors(frames_dir: str, timestamp: str, view_names: list[str],
                      depth_hw: tuple[int, int]) -> np.ndarray:
    """Load per-view RGB uint8 at depth resolution from original frames."""
    import cv2

    h, w = depth_hw
    colors = []
    for vname in view_names:
        path = os.path.join(frames_dir, timestamp, f"{vname}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        colors.append(small)
    return np.stack(colors, axis=0)


def load_recon_colors(recon: dict[str, Any], frames_dir: str, timestamp: str,
                      view_names: list[str]) -> np.ndarray:
    """Colors aligned with depth: DA3 npz image if present, else resize frames."""
    if recon.get("images") is not None:
        return np.asarray(recon["images"], dtype=np.uint8)
    return load_frame_colors(frames_dir, timestamp, view_names, recon["depth_hw"])


def load_fullres_frames(frames_dir: str, timestamp: str, view_names: list[str]) -> np.ndarray:
    """Load per-view RGB uint8 at original frame resolution."""
    import cv2

    frames = []
    for vname in view_names:
        path = os.path.join(frames_dir, timestamp, f"{vname}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        bgr = cv2.imread(path)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def scale_intrinsics(K: np.ndarray, from_hw: tuple[int, int], to_hw: tuple[int, int]) -> np.ndarray:
    """Scale pinhole K from one image resolution to another."""
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
    import cv2

    h, w = hw
    return cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def load_view_bundle(cfg: dict, timestamp: str, view_names: list[str],
                     backend: str | None = None) -> dict[str, Any]:
    """Recon + full-res frames + intrinsics/depth scaled for projection viz."""
    recon = load_recon(cfg, timestamp, backend=backend)
    frames = load_fullres_frames(cfg["frames_dir"], timestamp, view_names)
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
