#!/usr/bin/env python
"""Run VGGT-Omega reconstruction for one timestamp (6 views).

Example:
    CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/run_vggt_timestamp.py --timestamp 000029
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

VGGT_ROOT = "/data_ft_9_10/wentai/projects/vggt-omega"
sys.path.insert(0, VGGT_ROOT)

from demo_gradio import load_model, unproject_depth_map_to_point_map  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

CKPT = os.path.join(VGGT_ROOT, "checkpoints/vggt_omega_1b_512.pt")
RES = 512


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", required=True)
    ap.add_argument("--pipeline", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "pipeline.json"))
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--resolution", type=int, default=RES)
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    frames_dir = os.path.join(cfg["frames_dir"], args.timestamp)
    out_dir = os.path.join(cfg["vggt_dir"], args.timestamp)
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if len(imgs) != 6:
        raise SystemExit(f"expected 6 png views, got {len(imgs)} in {frames_dir}")

    print(f"[vggt] {args.timestamp}: {len(imgs)} views -> {out_dir}")
    model = load_model(args.ckpt)
    images = load_and_preprocess_images(imgs, image_resolution=args.resolution).to("cuda")
    t0 = time.time()
    with torch.inference_mode():
        pred = model(images)
    extrinsic, intrinsic = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    pred["extrinsic"] = extrinsic
    pred["intrinsic"] = intrinsic

    pn = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().cpu().numpy()
            if v.shape[0] == 1:
                v = v[0]
            pn[k] = v
    pn["world_points_from_depth"] = unproject_depth_map_to_point_map(
        pn["depth"], pn["extrinsic"], pn["intrinsic"])
    pn.pop("camera_and_register_tokens", None)

    out_path = os.path.join(out_dir, "predictions.npz")
    np.savez(out_path, **pn)
    print(f"[done] depth={pn['depth'].shape} saved {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
