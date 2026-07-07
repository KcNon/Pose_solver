"""Cross-view mask propagation to fill missing/empty per-view part masks.

Idea:
  For each part, some views have a good SAM3 mask and some are empty (missed
  detection or occlusion). We fuse the 3D point cloud from the *good* views,
  then reproject it into every view (with an occlusion test against that view's
  depth map) to fill in the missing masks.

Runs entirely on saved masks + results.npz; does NOT invoke SAM3.

Input:  outputs/masks_da3/<ts>/<view>_<part>.png   (from seg_backproject_parts.py)
Output: overwrites those masks with propagated versions (backup: *_raw.png),
        rebuilds outputs/parts_ply/<ts>/<part>.ply,
        writes propagated overlays *_overlay_prop.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.geom import backproject_view, project_points, rasterize_points
from common.icp import write_ply

DA3 = "/data_ft_9_10/wentai/projects/vggt-omega/试标数据-6.30/2/output_test/da3_output"
OUT_MASK = os.path.join(ROOT, "outputs/masks_da3")
OUT_PLY = os.path.join(ROOT, "outputs/parts_ply")
COLORS = {"lid": (0, 0, 255), "body": (255, 0, 0), "inner_pot": (0, 255, 0)}


def largest_cc(mask):
    mask = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == keep


def load_parts(ts):
    with open(os.path.join(ROOT, "configs", f"parts_{ts}.json"), encoding="utf-8") as f:
        return json.load(f)["parts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamps", nargs="+", required=True)
    ap.add_argument("--min-px", type=int, default=200, help="views with fewer px are treated as empty/to-fill")
    ap.add_argument("--depth-tol", type=float, default=0.03, help="occlusion depth tolerance (m)")
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-pts", type=int, default=80000)
    args = ap.parse_args()

    for ts in args.timestamps:
        parts = load_parts(ts)
        d = np.load(os.path.join(DA3, ts, "exports/npz/results.npz"))
        img, depth, K, E, conf = d["image"], d["depth"], d["intrinsics"], d["extrinsics"], d["conf"]
        V, H, W = depth.shape
        conf_thr = float(np.median(conf))
        odir = os.path.join(OUT_MASK, ts)
        ply_dir = os.path.join(OUT_PLY, ts)
        os.makedirs(ply_dir, exist_ok=True)

        for pname in parts:
            raw = []
            for v in range(V):
                m = cv2.imread(os.path.join(odir, f"{v}_{pname}.png"), cv2.IMREAD_GRAYSCALE)
                raw.append((m > 127) if m is not None else np.zeros((H, W), bool))

            # fuse cloud from views that have a solid mask
            good = [v for v in range(V) if raw[v].sum() >= args.min_px]
            fused_pts = []
            for v in good:
                m = raw[v] & np.isfinite(depth[v]) & (depth[v] > 1e-3) & (conf[v] > conf_thr)
                pts, _ = backproject_view(depth[v], K[v], E[v], mask=m)
                fused_pts.append(pts)
            fused = np.concatenate(fused_pts, 0) if fused_pts else np.empty((0, 3), np.float32)

            prop = [r.copy() for r in raw]
            filled = []
            if len(fused) > 0:
                for v in range(V):
                    uv, z = project_points(fused, K[v], E[v])
                    proj_mask = rasterize_points(uv, z, (H, W), depth_map=depth[v],
                                                 depth_tol=args.depth_tol, dilate=args.dilate)
                    if proj_mask.sum() > 0:
                        proj_mask = largest_cc(proj_mask)
                    before = int(prop[v].sum())
                    prop[v] = prop[v] | proj_mask
                    after = int(prop[v].sum())
                    if before < args.min_px and after >= args.min_px:
                        filled.append(v)

            # persist: backup raw once, overwrite with propagated
            for v in range(V):
                raw_path = os.path.join(odir, f"{v}_{pname}_raw.png")
                if not os.path.exists(raw_path):
                    cv2.imwrite(raw_path, (raw[v] * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(odir, f"{v}_{pname}.png"), (prop[v] * 255).astype(np.uint8))

            print(f"{ts} {pname}: good_views={good} filled_views={filled} "
                  f"px_per_view={[int(m.sum()) for m in prop]}")

        # rebuild overlays (propagated) + per-part plys
        for v in range(V):
            vis = cv2.cvtColor(img[v], cv2.COLOR_RGB2BGR)
            for pname in parts:
                m = cv2.imread(os.path.join(odir, f"{v}_{pname}.png"), cv2.IMREAD_GRAYSCALE) > 127
                ov = vis.copy()
                ov[m] = COLORS.get(pname, (255, 255, 255))
                vis = cv2.addWeighted(vis, 0.55, ov, 0.45, 0)
            cv2.imwrite(os.path.join(odir, f"{v}_overlay_prop.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

        for pname in parts:
            all_pts, all_cols = [], []
            for v in range(V):
                m = cv2.imread(os.path.join(odir, f"{v}_{pname}.png"), cv2.IMREAD_GRAYSCALE) > 127
                m = m & np.isfinite(depth[v]) & (depth[v] > 1e-3) & (conf[v] > conf_thr)
                sub = np.zeros_like(m)
                sub[::args.stride, ::args.stride] = True
                m &= sub
                if m.sum() == 0:
                    continue
                pts, cols = backproject_view(depth[v], K[v], E[v], mask=m, color=img[v])
                all_pts.append(pts)
                all_cols.append(cols)
            if all_pts:
                pts = np.concatenate(all_pts, 0)
                cols = np.concatenate(all_cols, 0)
                if len(pts) > args.max_pts:
                    idx = np.random.default_rng(0).choice(len(pts), args.max_pts, replace=False)
                    pts, cols = pts[idx], cols[idx]
            else:
                pts, cols = np.empty((0, 3), np.float32), None
            write_ply(os.path.join(ply_dir, f"{pname}.ply"), pts, cols)
            print(f"{ts} {pname}: rebuilt cloud {len(pts)} pts")

    print("done")


if __name__ == "__main__":
    main()
