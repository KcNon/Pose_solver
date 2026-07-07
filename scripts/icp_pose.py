"""ICP: align source timestamp parts to reference timestamp parts.

Every common part gets its own ICP (kornia-rs). If a part is truly static in world
frame, ICP should converge to T ~= I on its own — never hard-code identity.

Output: pose_<src>_to_<ref>.json  with per-part 4x4 T (src -> ref).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.icp import apply_transform, centroid_init, downsample, icp_point_to_point, write_ply


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path, encoding="ascii") as f:
        header = []
        while True:
            line = f.readline()
            header.append(line)
            if line.strip() == "end_header":
                break
        has_color = any("property uchar red" in h for h in header)
        pts, cols = [], [] if has_color else None
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if has_color:
                cols.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.asarray(pts, dtype=np.float32), cols


def load_parts(ts: str) -> list[str]:
    with open(os.path.join(ROOT, "configs", f"parts_{ts}.json"), encoding="utf-8") as f:
        return json.load(f)["parts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="000019")
    ap.add_argument("--src", default="000018")
    ap.add_argument("--max-pts", type=int, default=30000)
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/icp"))
    args = ap.parse_args()

    ref_parts = set(load_parts(args.ref))
    src_parts = set(load_parts(args.src))
    common = sorted(ref_parts & src_parts, key=lambda p: (p != "lid", p))
    if not common:
        raise SystemExit(f"no common parts between {args.ref} and {args.src}")

    ply_root = os.path.join(ROOT, "outputs/parts_ply")
    os.makedirs(args.out, exist_ok=True)
    summary = {
        "ref": args.ref,
        "src": args.src,
        "common_parts": common,
        "ref_only_parts": sorted(ref_parts - src_parts),
        "src_only_parts": sorted(src_parts - ref_parts),
        "parts": {},
    }

    for pname in common:
        ref_pts, ref_cols = read_ply(os.path.join(ply_root, args.ref, f"{pname}.ply"))
        src_pts, src_cols = read_ply(os.path.join(ply_root, args.src, f"{pname}.ply"))
        ref_pts, _ = downsample(ref_pts, None, args.max_pts, seed=1)
        src_pts, _ = downsample(src_pts, None, args.max_pts, seed=2)
        print(f"[{pname}] ref={len(ref_pts)} src={len(src_pts)}")

        init = centroid_init(src_pts, ref_pts)
        T, info = icp_point_to_point(src_pts, ref_pts, init=init,
                                     max_iters=args.max_iters, tol=args.tol)
        aligned = apply_transform(src_pts, T).astype(np.float32)

        np.save(os.path.join(args.out, f"T_{args.src}_to_{args.ref}_{pname}.npy"), T)
        write_ply(os.path.join(args.out, f"aligned_{args.src}_{pname}.ply"), aligned, src_cols)
        write_ply(os.path.join(args.out, f"ref_{args.ref}_{pname}.ply"), ref_pts, ref_cols)
        summary["parts"][pname] = {
            "transform_src_to_ref": T.tolist(),
            "icp": info,
            "translation": T[:3, 3].tolist(),
            "rotation": T[:3, :3].tolist(),
        }
        print(f"  ICP rmse={info['rmse']:.6f} iters={info['iters']} t={T[:3,3]}")

    out_json = os.path.join(args.out, f"pose_{args.src}_to_{args.ref}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
