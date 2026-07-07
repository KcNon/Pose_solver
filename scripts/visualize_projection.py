"""Validate ICP poses by projecting point clouds onto images.

T is src->ref: apply_transform(src_pts, T) ~= ref_pts.

Two consistent checks (same inv(T) / T for ALL parts, no special-casing):
  BACKWARD (ref -> src image): ref_pts @ inv(T) projected with src cameras
  FORWARD  (src -> ref image): src_pts @ T      projected with ref cameras

If ICP is correct, FORWARD overlay should match the object on the ref image for
every part. BACKWARD overlay should match the object on the src image.

Output: outputs/proj_vis/<ref>_to_<src>/
  view<v>.jpg          = backward | forward  side-by-side
  montage.jpg
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

from common.geom import project_points
from common.icp import apply_transform

DA3 = "/data_ft_9_10/wentai/projects/vggt-omega/试标数据-6.30/2/output_test/da3_output"
COLORS = {"lid": (0, 0, 255), "body": (255, 0, 0), "inner_pot": (0, 255, 0)}


def read_ply(path):
    pts = []
    with open(path, encoding="ascii") as f:
        hdr = False
        for line in f:
            if line.strip() == "end_header":
                hdr = True
                continue
            if not hdr:
                continue
            p = line.split()
            if len(p) >= 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
    return np.asarray(pts, dtype=np.float64)


def draw(img, uv, z, color, depth_map=None, depth_tol=0.05, radius=1):
    H, W = img.shape[:2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zz = u[ok], v[ok], z[ok]
    if depth_map is not None and len(u):
        dm = depth_map[v, u]
        vis = ~np.isfinite(dm) | (zz <= dm + depth_tol)
        u, v = u[vis], v[vis]
    out = img.copy()
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            out[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = color
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="000019")
    ap.add_argument("--src", default="000018")
    ap.add_argument("--icp-json", default=None)
    args = ap.parse_args()

    icp_json = args.icp_json or os.path.join(ROOT, "outputs/icp", f"pose_{args.src}_to_{args.ref}.json")
    with open(icp_json, encoding="utf-8") as f:
        pose = json.load(f)

    d_src = np.load(os.path.join(DA3, args.src, "exports/npz/results.npz"))
    d_ref = np.load(os.path.join(DA3, args.ref, "exports/npz/results.npz"))
    img_s, depth_s, K_s, E_s = d_src["image"], d_src["depth"], d_src["intrinsics"], d_src["extrinsics"]
    img_r, depth_r, K_r, E_r = d_ref["image"], d_ref["depth"], d_ref["intrinsics"], d_ref["extrinsics"]

    ref_clouds, src_clouds, T_map, inv_T = {}, {}, {}, {}
    for pname, pinfo in pose["parts"].items():
        rp = os.path.join(ROOT, "outputs/parts_ply", args.ref, f"{pname}.ply")
        sp = os.path.join(ROOT, "outputs/parts_ply", args.src, f"{pname}.ply")
        if os.path.exists(rp):
            ref_clouds[pname] = read_ply(rp)
        if os.path.exists(sp):
            src_clouds[pname] = read_ply(sp)
        T = np.array(pinfo["transform_src_to_ref"], dtype=np.float64)
        T_map[pname] = T
        inv_T[pname] = np.linalg.inv(T)

    out_dir = os.path.join(ROOT, "outputs/proj_vis", f"{args.ref}_to_{args.src}")
    os.makedirs(out_dir, exist_ok=True)
    panels = []

    for v in range(img_s.shape[0]):
        backward = cv2.cvtColor(img_s[v], cv2.COLOR_RGB2BGR)
        forward = cv2.cvtColor(img_r[v], cv2.COLOR_RGB2BGR)

        for pname in pose["parts"]:
            color = COLORS.get(pname, (255, 255, 255))
            if pname in ref_clouds:
                pts_b = apply_transform(ref_clouds[pname], inv_T[pname])
                uv, z = project_points(pts_b, K_s[v], E_s[v])
                backward = draw(backward, uv, z, color, depth_map=depth_s[v])
            if pname in src_clouds:
                pts_f = apply_transform(src_clouds[pname], T_map[pname])
                uv, z = project_points(pts_f, K_r[v], E_r[v])
                forward = draw(forward, uv, z, color, depth_map=depth_r[v])

        cv2.putText(backward, f"backward: ref@inv(T) on {args.src}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(forward, f"forward: src@T on {args.ref}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        pair = np.hstack([backward, np.full((backward.shape[0], 4, 3), 255, np.uint8), forward])
        cv2.imwrite(os.path.join(out_dir, f"view{v}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 92])
        panels.append(pair)

    montage = np.vstack([cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2)) for p in panels])
    cv2.imwrite(os.path.join(out_dir, "montage.jpg"), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
