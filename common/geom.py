"""Geometry helpers: backproject depth maps into world-frame point clouds."""
import numpy as np


def backproject_view(depth, K, E, mask=None, color=None, depth_min=1e-3, depth_max=1e3):
    """Backproject one view's depth map into world coordinates.

    Args:
        depth: (H,W) float depth (z along optical axis, in the recon's metric)
        K:     (3,3) intrinsics for this view (matching depth resolution)
        E:     (3,4) extrinsics, world->cam: X_cam = R @ X_world + t
        mask:  (H,W) bool, optional; keep only these pixels
        color: (H,W,3) uint8, optional; per-point colors
    Returns:
        pts_world (N,3) float32, cols (N,3) uint8 or None
    """
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    valid = np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
    if mask is not None:
        valid &= mask.astype(bool)
    u = xs[valid].astype(np.float64)
    v = ys[valid].astype(np.float64)
    z = depth[valid].astype(np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)  # (N,3)

    R = E[:, :3].astype(np.float64)
    t = E[:, 3].astype(np.float64)
    pts_world = (pts_cam - t[None, :]) @ R  # R^T @ (X_cam - t) == (X_cam - t) @ R
    cols = None
    if color is not None:
        cols = color[valid]
    return pts_world.astype(np.float32), cols


def project_points(pts_world, K, E):
    """Project world-frame points into one camera.

    Args:
        pts_world: (N,3) world coords
        K: (3,3) intrinsics matching target resolution
        E: (3,4) extrinsics, world->cam: X_cam = R @ X_world + t
    Returns:
        uv: (N,2) float pixel coords (x=col, y=row)
        z:  (N,) camera-space depth (>0 means in front of camera)
    """
    pts = np.asarray(pts_world, dtype=np.float64)
    R = E[:, :3].astype(np.float64)
    t = E[:, 3].astype(np.float64)
    pts_cam = pts @ R.T + t[None, :]  # (N,3)
    z = pts_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    eps = 1e-8
    u = fx * pts_cam[:, 0] / (z + eps) + cx
    v = fy * pts_cam[:, 1] / (z + eps) + cy
    uv = np.stack([u, v], axis=1)
    return uv, z


def rasterize_points(uv, z, hw, depth_map=None, depth_tol=0.05, dilate=1):
    """Rasterize projected points into a boolean mask, with optional occlusion test.

    Args:
        uv: (N,2) pixel coords, z: (N,) camera depth
        hw: (H,W) target mask size
        depth_map: (H,W) reference depth for occlusion check (keep points whose
                   projected z matches the map within depth_tol); None disables it
        depth_tol: absolute depth tolerance for the visibility test
        dilate: half-window (pixels) to splat each point (fills sparse gaps)
    Returns:
        mask (H,W) bool
    """
    H, W = hw
    mask = np.zeros((H, W), dtype=bool)
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zz = u[ok], v[ok], z[ok]
    if depth_map is not None and len(u) > 0:
        dm = depth_map[v, u]
        vis = np.isfinite(dm) & (np.abs(zz - dm) <= depth_tol)
        u, v = u[vis], v[vis]
    for du in range(-dilate, dilate + 1):
        for dv in range(-dilate, dilate + 1):
            uu = np.clip(u + du, 0, W - 1)
            vv = np.clip(v + dv, 0, H - 1)
            mask[vv, uu] = True
    return mask
