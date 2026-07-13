"""Render a colored point cloud (ply) to 2D images from a few viewpoints."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_pts", type=int, default=60000)
    args = ap.parse_args()

    pc = o3d.io.read_point_cloud(args.ply)
    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors)
    if len(pts) > args.max_pts:
        idx = np.random.choice(len(pts), args.max_pts, replace=False)
        pts, cols = pts[idx], cols[idx]

    views = [(20, -60), (90, -90), (0, -90)]  # (elev, azim): oblique, top, front
    fig = plt.figure(figsize=(18, 6))
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev} azim={azim}")
        ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
    plt.tight_layout()
    plt.savefig(args.out, dpi=90)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
