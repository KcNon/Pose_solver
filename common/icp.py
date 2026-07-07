"""Rigid ICP via kornia-rs (Rust kornia-icp) + geometry helpers.

6DoF pose: T maps source -> target, i.e. apply_transform(source, T) ~= target.
Uses kornia_rs.icp_vanilla under the hood.
"""
from __future__ import annotations

import numpy as np

try:
    import kornia_rs as kr
except ImportError as e:
    raise ImportError("kornia-rs is required: uv pip install kornia-rs") from e


def apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts @ R.T) + t


def centroid_init(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = dst.mean(axis=0) - src.mean(axis=0)
    return T


def _as_cloud(pts: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(pts, dtype=np.float64).reshape(-1, 3))


def _result_to_T(result) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(result.rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(result.translation, dtype=np.float64)
    return T


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-6,
    reject_ratio: float = 0.0,  # kept for API compat; kornia-icp has no reject_ratio
    init: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Align source -> target with kornia-rs icp_vanilla.

    Returns 4x4 T such that apply_transform(source, T) ~= target.
    """
    del reject_ratio  # not exposed by kornia-icp vanilla
    src = _as_cloud(source)
    tgt = _as_cloud(target)
    if len(src) < 10 or len(tgt) < 10:
        raise ValueError(f"need >=10 points, got src={len(src)} tgt={len(tgt)}")

    if init is None:
        init = np.eye(4, dtype=np.float64)
    else:
        init = np.asarray(init, dtype=np.float64)
    init_R = np.ascontiguousarray(init[:3, :3], dtype=np.float64)
    init_t = np.ascontiguousarray(init[:3, 3], dtype=np.float64)

    criteria = kr.k3d.ICPConvergenceCriteria(max_iterations=max_iters, tolerance=tol)
    result = kr.k3d.icp_vanilla(src, tgt, init_R, init_t, criteria)
    T = _result_to_T(result)
    info = {
        "iters": int(result.num_iterations),
        "rmse": float(result.rmse),
        "n_source": len(src),
        "n_target": len(tgt),
        "backend": "kornia-rs",
    }
    return T, info


def downsample(pts: np.ndarray, cols: np.ndarray | None, max_pts: int, seed: int = 0):
    if len(pts) <= max_pts:
        return pts, cols
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), max_pts, replace=False)
    c = cols[idx] if cols is not None else None
    return pts[idx], c


def write_ply(path: str, pts: np.ndarray, cols: np.ndarray | None = None):
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if cols is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if cols is None:
            for p in pts:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            cols = np.asarray(cols, dtype=np.uint8)
            for p, c in zip(pts, cols):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
