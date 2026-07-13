"""Verify DA3 results.npz: view order + extrinsic convention via backprojection.

Backprojects all 6 views of one timestamp into world frame; if the extrinsic
convention is correct, the static background/object surfaces from different views
should overlap. Saves a combined colored PLY and per-view result images.
"""
import argparse
import os
import sys

sys.path.insert(0, "/data_ft_9_10/wentai/projects/pose_solver")

import cv2
import numpy as np
import open3d as o3d
from common.geom import backproject_view

DA3 = "/data_ft_9_10/wentai/projects/depth-anything-3/试标数据-6.30/2/output_test/da3_output"
OUT = "/data_ft_9_10/wentai/projects/pose_solver/outputs/verify"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="000010")
    ap.add_argument("--stride", type=int, default=3, help="pixel subsample")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    d = np.load(os.path.join(DA3, args.ts, "exports/npz/results.npz"))
    img, depth, K, E, conf = d["image"], d["depth"], d["intrinsics"], d["extrinsics"], d["conf"]
    print("shapes:", img.shape, depth.shape, K.shape, E.shape)
    print("depth stats: min %.3f max %.3f median %.3f" %
          (np.nanmin(depth), np.nanmax(depth), float(np.nanmedian(depth))))
    print("conf stats: min %.3f max %.3f median %.3f" %
          (conf.min(), conf.max(), float(np.median(conf))))

    # dump per-view images to compare with frames 2-1..2-6
    for v in range(img.shape[0]):
        cv2.imwrite(os.path.join(OUT, f"{args.ts}_da3view_{v}.jpg"),
                    cv2.cvtColor(img[v], cv2.COLOR_RGB2BGR))

    all_pts, all_cols = [], []
    conf_thr = float(np.median(conf))
    for v in range(img.shape[0]):
        m = conf[v] > conf_thr
        m[::args.stride, :] = m[::args.stride, :]  # keep; subsample below
        sub = np.zeros_like(m)
        sub[::args.stride, ::args.stride] = True
        m = m & sub
        pts, cols = backproject_view(depth[v], K[v], E[v], mask=m, color=img[v])
        all_pts.append(pts)
        all_cols.append(cols)
        print(f"view {v}: {len(pts)} pts")
    pts = np.concatenate(all_pts, 0)
    cols = np.concatenate(all_cols, 0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    out_ply = os.path.join(OUT, f"{args.ts}_scene.ply")
    o3d.io.write_point_cloud(out_ply, pc)
    print("wrote", out_ply, "npts", len(pts))


if __name__ == "__main__":
    main()
