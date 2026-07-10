"""Inter-frame gauge normalization for independently-reconstructed timestamps.

DA3 reconstructs each timestamp cleanly (its views are mutually metric to ~2mm)
but anchors the global gauge differently per scene: two timestamps of the same
rig differ by a single global similarity (~15cm translation + ~3.6% scale in
folder-2). For a static part that masquerades as motion; for a moving part it
biases the estimated pose. This module estimates that similarity from robust
whole-scene consensus (movers drop out via trimming) and bakes it into a
timestamp's depth + extrinsics so all timestamps share one gauge.

Pure numpy/scipy (no kornia) so it imports under the DA3 environment too.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity mapping src -> dst: dst ~= s*R@src + t.

    Returns (s, R, t) with R (3,3), t (3,), s float.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xc, Yc = src - mu_s, dst - mu_d
    Sigma = (Yc.T @ Xc) / n                      # (3,3)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (Xc ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def _trimmed_icp(src, ref, tree, s, R, t, with_scale, iters, trim):
    """One ICP loop from a given (s,R,t) init. Free scale only if with_scale."""
    prev = None
    keep = None
    j = None
    for it in range(iters):
        p = (s * (src @ R.T)) + t
        d, j = tree.query(p, k=1)
        thr = np.quantile(d, trim)
        keep = d <= max(thr, 1e-9)
        if keep.sum() < 10:
            break
        s, R, t = umeyama(src[keep], ref[j[keep]], with_scale)
        rms = float(np.sqrt(np.mean(d[keep] ** 2)))
        if prev is not None and abs(prev - rms) < 1e-7:
            break
        prev = rms
    return s, R, t, keep, j, prev


def estimate_similarity_robust(
    src: np.ndarray,
    ref: np.ndarray,
    with_scale: bool = True,
    iters: int = 40,
    trim: float = 0.6,
    max_pts: int = 6000,
    seed: int = 0,
    scale_clamp: tuple[float, float] = (0.5, 2.0),
):
    """Robust similarity src -> ref. Moving/outlier regions drop via trimming.

    Scale is decoupled from correspondence search to avoid the free-scale ICP
    collapse (shrinking the cloud to fake low NN distances): converge a trimmed
    RIGID ICP first, then refine scale on the settled inliers. A few outer passes
    re-select correspondences with the current similarity and re-fit.

    Returns (s, R, t, info). Apply to a point p as: s * R @ p + t.
    """
    rng = np.random.default_rng(seed)
    src = np.asarray(src, np.float64)
    ref = np.asarray(ref, np.float64)
    if len(src) > max_pts:
        src = src[rng.choice(len(src), max_pts, replace=False)]
    if len(ref) > max_pts:
        ref = ref[rng.choice(len(ref), max_pts, replace=False)]
    tree = cKDTree(ref)

    # Stage 1: rigid ICP (no collapse risk) to lock correspondences.
    s, R, t, keep, j, rms = _trimmed_icp(src, ref, tree, 1.0, np.eye(3), np.zeros(3),
                                         with_scale=False, iters=iters, trim=trim)
    # Stage 2: introduce scale via a few outer refinements on settled inliers.
    if with_scale:
        for _ in range(4):
            p = (s * (src @ R.T)) + t
            d, j = tree.query(p, k=1)
            keep = d <= max(np.quantile(d, trim), 1e-9)
            if keep.sum() < 10:
                break
            s, R, t = umeyama(src[keep], ref[j[keep]], with_scale=True)
            rms = float(np.sqrt(np.mean(d[keep] ** 2)))
        s = float(np.clip(s, *scale_clamp))
    info = {
        "iters": iters,
        "scale": s,
        "trim_rmse": rms if rms is not None else float("nan"),
        "n_src": len(src),
        "n_ref": len(ref),
        "trim": trim,
        "n_inliers": int(keep.sum()) if keep is not None else 0,
    }
    return s, R, t, info


def apply_similarity_to_frame(depth: np.ndarray, E: np.ndarray, s: float,
                              R: np.ndarray, t: np.ndarray):
    """Bake world-space similarity (X' = s*R@X + t) into a view's (depth, E).

    E is world->cam [R_e | t_e] (3,4), backprojection uses X = R_e^T (x_cam - t_e).
    New view satisfies backproject(depth', E') == s*R@backproject(depth, E) + t.
    Returns (depth', E').  K is unchanged.
    """
    E = np.asarray(E, np.float64)
    Re, te = E[:3, :3], E[:3, 3]
    Re2 = Re @ R.T
    te2 = s * te - Re2 @ t
    E2 = E.copy()
    E2[:3, :3] = Re2
    E2[:3, 3] = te2
    depth2 = (np.asarray(depth, np.float64) * s).astype(depth.dtype)
    return depth2, E2
