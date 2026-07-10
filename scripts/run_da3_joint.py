#!/usr/bin/env python
"""Pose-conditioned DA3 reconstruction of MULTIPLE timestamps in ONE forward pass.

Why: running DA3 independently per timestamp leaves a per-frame gauge drift
(~16cm translation + ~3.6% scale in folder-2) that contaminates every per-part
ICP pose downstream (a static part resolves to |t|~0.2 instead of ~0). Because
the rig is fixed and calibrated, feeding all views of several timestamps
together (T*N images, rig extrinsics tiled T times) makes DA3 solve one
globally-consistent scale/frame. Static geometry across timestamps then
coincides; only genuinely moving parts differ.

The T*N-view Prediction is sliced back into per-timestamp results.npz with the
same layout DA3 normally writes, so the rest of the pose_solver pipeline reads
it unchanged -- just point --recon-backend da3 at a pipeline whose da3_dir is
this script's --out.

Run with DA3's environment (it owns the depth_anything_3 package + torch):
    /data_ft_9_10/wentai/projects/depth-anything-3/.venv/bin/python \
        scripts/run_da3_joint.py --timestamps 000010 000019
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import replace

import numpy as np

MODEL_DIR = "/data_ft_9_10/wentai/projects/depth-anything-3/DA3NESTED-GIANT-LARGE-1.1"
FULL_W, FULL_H = 1920, 1080  # on-disk frame resolution

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_rig(base: str):
    """Fixed-rig calibration written next to the frames (same as run_da3_pose_cond)."""
    cam = np.load(os.path.join(base, "stability/000000/camera_params.npz"))
    ext34 = cam["extrinsic"].astype(np.float32)   # (N,3,4) w2c opencv
    K = cam["intrinsic"].astype(np.float32)       # (N,3,3) in (2cx,2cy) space
    N = ext34.shape[0]
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    extrinsics[:, :3, :4] = ext34
    intrinsics = K.copy()
    for i in range(N):
        intrinsics[i, 0, :] *= FULL_W / (2.0 * K[i, 0, 2])
        intrinsics[i, 1, :] *= FULL_H / (2.0 * K[i, 1, 2])
    return extrinsics, intrinsics, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamps", nargs="+", required=True)
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline.json"))
    ap.add_argument("--out", default=None,
                    help="output da3_dir (default: <base>/da3_joint_<ts...>)")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--use-ray-pose", action="store_true",
                    help="use DA3 ray-pose head (slower, often more consistent depth)")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    frames_dir = cfg["frames_dir"]
    base = os.path.dirname(frames_dir)  # .../output_test

    ext_rig, in_rig, N = load_rig(base)

    all_imgs, blocks = [], []
    for ts in args.timestamps:
        imgs = sorted(glob.glob(os.path.join(frames_dir, ts, "*.png")))
        if len(imgs) != N:
            raise SystemExit(f"{ts}: {len(imgs)} imgs != {N} rig cams")
        blocks.append((ts, len(all_imgs), len(all_imgs) + N))
        all_imgs.extend(imgs)

    T = len(args.timestamps)
    ext_all = np.tile(ext_rig, (T, 1, 1))   # (T*N,4,4)
    in_all = np.tile(in_rig, (T, 1, 1))     # (T*N,3,3)

    out_dir = args.out or os.path.join(base, "da3_joint_" + "_".join(args.timestamps))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[joint] {T} timestamps x {N} views = {len(all_imgs)} imgs -> {out_dir}")

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.export.npz import export_to_npz

    print(f"[model] loading {MODEL_DIR}")
    model = DepthAnything3.from_pretrained(MODEL_DIR).to(device="cuda")

    t0 = time.time()
    pred = model.inference(
        all_imgs,
        extrinsics=ext_all,
        intrinsics=in_all,
        align_to_input_ext_scale=True,
        use_ray_pose=args.use_ray_pose,
        export_dir=None,               # slice + export per-timestamp below
        process_res=args.process_res,
    )
    print(f"[infer] {len(all_imgs)} views in {time.time()-t0:.1f}s; "
          f"depth={pred.depth.shape} scale_factor={getattr(pred, 'scale_factor', None)}")

    def sl(a, s, e):
        return None if a is None else a[s:e]

    for ts, s, e in blocks:
        sub = replace(
            pred,
            depth=pred.depth[s:e],
            conf=sl(pred.conf, s, e),
            sky=sl(pred.sky, s, e),
            extrinsics=sl(pred.extrinsics, s, e),
            intrinsics=sl(pred.intrinsics, s, e),
            processed_images=sl(pred.processed_images, s, e),
        )
        export_to_npz(sub, os.path.join(out_dir, ts))
        print(f"  [{ts}] wrote {os.path.join(out_dir, ts, 'exports/npz/results.npz')}")

    print(f"[done] {time.time()-t0:.1f}s  ->  set da3_dir={out_dir}")


if __name__ == "__main__":
    main()
