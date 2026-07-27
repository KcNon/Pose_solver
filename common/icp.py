"""Rigid point-cloud registration helpers.

6DoF pose: T maps source -> target, i.e. apply_transform(source, T) ~= target.

Backends:
  - icp  : kornia-rs icp_vanilla (point-to-point)
  - gicp : small_gicp GICP (distribution-to-distribution)
"""
from __future__ import annotations

import numpy as np

from common.cloud_io import nearest_neighbor_rmse as nn_rmse, write_ply

try:
    import kornia_rs as kr
except ImportError as e:
    raise ImportError("kornia-rs is required: uv pip install kornia-rs") from e


def apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts @ R.T) + t


def compose_transform(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Return T_b @ T_a so apply_transform(p, T_total) == apply_transform(apply_transform(p, T_a), T_b)."""
    R_a, t_a = T_a[:3, :3], T_a[:3, 3]
    R_b, t_b = T_b[:3, :3], T_b[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_b @ R_a
    T[:3, 3] = t_a @ R_b.T + t_b
    return T


def centroid_init(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = dst.mean(axis=0) - src.mean(axis=0)
    return T


def camera_centers_from_extrinsics(E: np.ndarray) -> np.ndarray:
    """Camera centers in world frame. E: (V,3,4) world->cam, row-vector convention."""
    E = np.asarray(E, dtype=np.float64)
    centers = []
    for i in range(E.shape[0]):
        R = E[i, :3, :3]
        t = E[i, :3, 3]
        centers.append(-t @ R)
    return np.stack(centers, axis=0)


def kabsch_rigid(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigid transform (Kabsch) mapping source -> target with known correspondences."""
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if len(src) != len(tgt):
        raise ValueError(f"kabsch needs equal counts, got {len(src)} vs {len(tgt)}")
    if len(src) < 3:
        return centroid_init(src, tgt)

    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    X = src - mu_s
    Y = tgt - mu_t
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_t - mu_s @ R.T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def kabsch_nn_init(
    source: np.ndarray,
    target: np.ndarray,
    max_src: int = 4000,
    dist_percentile: float = 70.0,
    seed: int = 0,
) -> np.ndarray:
    """Kabsch on nearest-neighbor correspondences after centroid pre-alignment."""
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if len(src) < 10 or len(tgt) < 10:
        return centroid_init(src, tgt)

    rng = np.random.default_rng(seed)
    if len(src) > max_src:
        src = src[rng.choice(len(src), max_src, replace=False)]
    if len(tgt) > max_src:
        tgt_sub = tgt[rng.choice(len(tgt), max_src, replace=False)]
    else:
        tgt_sub = tgt

    # Pre-align centroids for more stable NN matching.
    T0 = centroid_init(src, tgt_sub)
    src0 = apply_transform(src, T0)

    # Brute-force NN (N<=4000 is fine).
    diff = src0[:, None, :] - tgt_sub[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    nn = np.argmin(d2, axis=1)
    dists = np.sqrt(d2[np.arange(len(src0)), nn])
    thr = np.percentile(dists, dist_percentile)
    keep = dists <= max(thr, 1e-6)
    if keep.sum() < 10:
        return T0

    T1 = kabsch_rigid(src0[keep], tgt_sub[nn[keep]])
    return compose_transform(T0, T1)


def camera_world_align_init(E_src: np.ndarray, E_ref: np.ndarray) -> np.ndarray:
    """Align src world frame to ref using corresponding per-view camera centers."""
    C_src = camera_centers_from_extrinsics(E_src)
    C_ref = camera_centers_from_extrinsics(E_ref)
    if len(C_src) != len(C_ref):
        raise ValueError(f"view count mismatch: src={len(C_src)} ref={len(C_ref)}")
    return kabsch_rigid(C_src, C_ref)


def build_icp_init(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    mode: str,
    T_camera: np.ndarray | None = None,
    T_global: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Build 4x4 init transform mapping src -> ref."""
    info: dict = {"mode": mode}

    if mode == "centroid":
        T = centroid_init(src_pts, ref_pts)
        info["components"] = ["centroid"]
        return T, info

    if mode == "kabsch":
        T = kabsch_nn_init(src_pts, ref_pts, seed=seed)
        info["components"] = ["kabsch_nn"]
        return T, info

    if mode == "camera":
        if T_camera is None:
            raise ValueError("camera mode requires T_camera")
        T = T_camera
        info["components"] = ["camera"]
        return T, info

    if mode == "both":
        if T_camera is None:
            raise ValueError("both mode requires T_camera")
        src_cam = apply_transform(src_pts, T_camera)
        T_local = kabsch_nn_init(src_cam, ref_pts, seed=seed)
        T = compose_transform(T_camera, T_local)
        info["components"] = ["camera", "kabsch_nn"]
        return T, info

    if mode == "rigid":
        if T_global is None:
            raise ValueError("rigid mode requires T_global")
        T = T_global
        info["components"] = ["global_rigid"]
        return T, info

    raise ValueError(f"unknown init mode: {mode}")


def _as_cloud(pts: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(pts, dtype=np.float64).reshape(-1, 3))


def _result_to_T(result) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(result.rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(result.translation, dtype=np.float64)
    return T


def _auto_gicp_resolution(pts: np.ndarray) -> tuple[float, float]:
    extent = np.max(pts, axis=0) - np.min(pts, axis=0)
    ds = max(float(np.max(extent)) / 80.0, 0.002)
    return ds, ds * 8.0


def gicp_align(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-6,
    init: np.ndarray | None = None,
    downsampling_resolution: float | None = None,
    max_correspondence_distance: float | None = None,
    num_neighbors: int = 20,
    num_threads: int = 1,
) -> tuple[np.ndarray, dict]:
    """Align source -> target with small_gicp GICP."""
    try:
        import small_gicp
    except ImportError as e:
        raise ImportError("small-gicp is required: uv pip install small-gicp") from e

    src = _as_cloud(source)
    tgt = _as_cloud(target)
    if len(src) < 10 or len(tgt) < 10:
        raise ValueError(f"need >=10 points, got src={len(src)} tgt={len(tgt)}")

    if init is None:
        init = np.eye(4, dtype=np.float64)
    else:
        init = np.asarray(init, dtype=np.float64)

    if downsampling_resolution is None:
        ds, mcd = _auto_gicp_resolution(np.vstack([src, tgt]))
    else:
        ds = float(downsampling_resolution)
        mcd = float(max_correspondence_distance or ds * 8.0)

    tgt_pp, tgt_tree = small_gicp.preprocess_points(
        tgt, downsampling_resolution=ds, num_neighbors=num_neighbors, num_threads=num_threads,
    )
    src_pp, _ = small_gicp.preprocess_points(
        src, downsampling_resolution=ds, num_neighbors=num_neighbors, num_threads=num_threads,
    )
    result = small_gicp.align(
        tgt_pp,
        src_pp,
        tgt_tree,
        init_T_target_source=init,
        registration_type="GICP",
        max_correspondence_distance=mcd,
        max_iterations=max_iters,
        translation_epsilon=tol,
        num_threads=num_threads,
    )
    T = np.asarray(result.T_target_source, dtype=np.float64)
    info = {
        "iters": int(result.iterations),
        "rmse": nn_rmse(src, tgt, T),
        "lib_error": float(result.error),
        "converged": bool(result.converged),
        "n_source": len(src),
        "n_target": len(tgt),
        "backend": "small-gicp",
        "gicp_downsample": ds,
        "gicp_max_dist": mcd,
    }
    return T, info


def register_points(
    source: np.ndarray,
    target: np.ndarray,
    method: str = "icp",
    max_iters: int = 50,
    tol: float = 1e-6,
    reject_ratio: float = 0.0,
    init: np.ndarray | None = None,
    downsampling_resolution: float | None = None,
    max_correspondence_distance: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Dispatch to icp or gicp backend."""
    if method == "icp":
        return icp_point_to_point(
            source, target, max_iters=max_iters, tol=tol,
            reject_ratio=reject_ratio, init=init,
        )
    if method == "gicp":
        return gicp_align(
            source, target, max_iters=max_iters, tol=tol, init=init,
            downsampling_resolution=downsampling_resolution,
            max_correspondence_distance=max_correspondence_distance,
        )
    raise ValueError(f"unknown registration method: {method!r}")


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
    info["nn_rmse"] = nn_rmse(src, tgt, T)
    return T, info


def downsample(pts: np.ndarray, cols: np.ndarray | None, max_pts: int, seed: int = 0):
    if len(pts) <= max_pts:
        return pts, cols
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), max_pts, replace=False)
    c = cols[idx] if cols is not None else None
    return pts[idx], c
