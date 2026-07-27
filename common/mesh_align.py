"""Align a generated (canonical-frame) mesh to an observed point cloud.

The generated glb meshes live in the generator's normalized frame (centered near
origin, arbitrary scale/orientation). We register them to the DA3-world-frame
observed part clouds with a 7DoF *similarity* transform (scale + rotation +
translation), so they can be rendered back into the DA3 cameras.

Registration direction: observed cloud (source) -> mesh surface (target). Every
observed point has a correspondence on the complete mesh, which avoids the bias
of matching a full mesh against a partially-observed cloud. The returned
``T_mesh_to_world`` is the inverse of that similarity.
"""
from __future__ import annotations

import numpy as np
import trimesh
from trimesh import registration as reg

from common.cloud_io import nearest_neighbor_rmse, read_ply_xyz


def _rms_radius(pts: np.ndarray, center: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pts - center) ** 2, axis=1))))


def _pca_axes(pts: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Columns are principal axes (right-handed), sorted by descending variance."""
    cov = np.cov((pts - center).T)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    V = V[:, order]
    if np.linalg.det(V) < 0:
        V[:, -1] *= -1
    return V


# 4 sign flips that keep a proper rotation (det = +1): flip an even number of axes.
_SIGN_FLIPS = [
    np.diag([1.0, 1.0, 1.0]),
    np.diag([1.0, -1.0, -1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([-1.0, -1.0, 1.0]),
]


def _candidate_inits(
    obs: np.ndarray,
    mesh_pts: np.ndarray,
    *,
    estimate_scale: bool = True,
) -> list[np.ndarray]:
    """Build 4x4 similarity inits mapping obs -> mesh via PCA axis matching."""
    c_o = obs.mean(0)
    c_m = mesh_pts.mean(0)
    s = (
        _rms_radius(mesh_pts, c_m) / max(_rms_radius(obs, c_o), 1e-9)
        if estimate_scale
        else 1.0
    )
    V_o = _pca_axes(obs, c_o)
    V_m = _pca_axes(mesh_pts, c_m)

    inits = []
    for F in _SIGN_FLIPS:
        R = V_m @ F @ V_o.T  # rotate obs-axes onto mesh-axes
        if np.linalg.det(R) < 0:
            continue
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3] = c_m - (s * R) @ c_o
        inits.append(T)
    return inits


def align_mesh_to_cloud(
    mesh: trimesh.Trimesh,
    obs: np.ndarray,
    n_mesh_sample: int = 20000,
    n_obs_max: int = 6000,
    coarse_iters: int = 12,
    fine_iters: int = 60,
    seed: int = 0,
    fixed_scale: float | None = None,
) -> dict:
    """Return a raw-mesh-to-world similarity and fit diagnostics.

    When ``fixed_scale`` is supplied the raw mesh is scaled first and ICP only
    estimates a rigid transform.  This is materially different from fitting a
    free similarity and replacing its scale afterwards: the latter leaves the
    translation optimized for the wrong-sized object and can displace anchors
    by many centimetres.
    """
    if fixed_scale is not None and float(fixed_scale) <= 0.0:
        raise ValueError("fixed_scale must be positive")
    rng = np.random.default_rng(seed)
    obs = np.asarray(obs, dtype=np.float64)
    if len(obs) > n_obs_max:
        obs = obs[rng.choice(len(obs), n_obs_max, replace=False)]

    mesh_pts, _ = trimesh.sample.sample_surface(mesh, n_mesh_sample)
    mesh_pts = np.asarray(mesh_pts, dtype=np.float64)
    fit_mesh_pts = (
        mesh_pts
        if fixed_scale is None
        else float(fixed_scale) * mesh_pts
    )

    # Target = sampled mesh points (trimesh icp uses cKDTree; avoids rtree dep).
    # Coarse: try each PCA-init, keep the lowest-cost obs->mesh fit.
    best = None
    for T0 in _candidate_inits(
        obs,
        fit_mesh_pts,
        estimate_scale=fixed_scale is None,
    ):
        matrix, _, cost = reg.icp(
            obs, fit_mesh_pts, initial=T0, max_iterations=coarse_iters,
            scale=fixed_scale is None, reflection=False,
        )
        if best is None or cost < best[1]:
            best = (matrix, cost)

    # Fine refinement from the best coarse result.
    T_obs_to_mesh, _, cost = reg.icp(
        obs, fit_mesh_pts, initial=best[0], max_iterations=fine_iters,
        scale=fixed_scale is None, reflection=False,
    )
    T_fit_mesh_to_world = np.linalg.inv(T_obs_to_mesh)

    # ``fit_mesh_pts`` are already scaled in fixed-scale mode.  Convert the
    # rigid transform back to one that maps the original raw mesh to world.
    T_mesh_to_world = T_fit_mesh_to_world.copy()
    if fixed_scale is not None:
        T_mesh_to_world[:3, :3] *= float(fixed_scale)

    # Decompose the similarity: M = s * R (assumes uniform scale).
    M = T_mesh_to_world[:3, :3]
    scale = float(np.cbrt(max(np.linalg.det(M), 1e-12)))
    R = M / scale
    t = T_mesh_to_world[:3, 3]

    # World-frame fit error: for each observed point, nearest transformed mesh pt.
    mesh_world = mesh_pts @ M.T + t
    fit_rmse = nearest_neighbor_rmse(obs, mesh_world, np.eye(4))

    return {
        "T_mesh_to_world": T_mesh_to_world,
        "scale": scale,
        "R": R,
        "t": t,
        "fit_rmse": float(fit_rmse),
        "icp_cost": float(cost),
        "n_obs": int(len(obs)),
        "n_mesh_sample": int(len(mesh_pts)),
    }
