#!/usr/bin/env python
"""Run DA3 on a multi-view sequence with one fixed self-estimated rig.

The camera rig is estimated once on ``--camera-frame`` (chosen to contain
well-separated objects and background features), then reused for every requested
timestamp. Outputs keep the prediction.npz format consumed by pose_solver.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

DA3_ROOT = Path("/data_ft_9_10/wentai/projects/depth-anything-3")
sys.path.insert(0, str(DA3_ROOT))

import run_da3_multiview_video as base
from depth_anything_3.api import DepthAnything3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument(
        "--camera-frames-dir",
        help=(
            "optional frame root used only to estimate the fixed camera rig; "
            "defaults to --frames-dir"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--timestamps", nargs="+")
    parser.add_argument("--camera-frame", required=True)
    parser.add_argument(
        "--model-dir",
        default=str(DA3_ROOT / "DA3NESTED-GIANT-LARGE-1.1"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument("--ref-view-strategy", default="saddle_balanced")
    parser.add_argument("--full-w", type=int, default=1920)
    parser.add_argument("--full-h", type=int, default=1080)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    camera_frames_dir = Path(args.camera_frames_dir or args.frames_dir)
    output_dir = Path(args.output_dir)
    available = base.discover_frames(str(frames_dir), args.views)
    camera_available = base.discover_frames(str(camera_frames_dir), args.views)
    if args.camera_frame not in camera_available:
        raise ValueError(
            f"camera frame {args.camera_frame} is unavailable in "
            f"{camera_frames_dir}"
        )
    frame_ids = args.timestamps or available
    missing = sorted(set(frame_ids) - set(available))
    if missing:
        raise ValueError(f"timestamps unavailable in all views: {missing}")

    device = torch.device(args.device)
    print(f"[views] {len(args.views)}: {args.views}", flush=True)
    print(f"[frames] {len(frame_ids)}, camera frame={args.camera_frame}", flush=True)
    print(f"[model] loading {args.model_dir}", flush=True)
    model = DepthAnything3.from_pretrained(args.model_dir).to(device=device)

    camera_paths = base.frame_image_paths(
        str(camera_frames_dir), args.views, args.camera_frame
    )
    ext44, K_native = base.cameras_from_self(
        model, camera_paths, args.process_res, args.process_res_method,
        args.use_ray_pose,
    )
    K_input = K_native.astype(np.float32).copy()
    for index in range(len(args.views)):
        K_input[index, 0, :] *= args.full_w / (2.0 * K_native[index, 0, 2])
        K_input[index, 1, :] *= args.full_h / (2.0 * K_native[index, 1, 2])
    print("[cameras] fixed self-estimated rig ready", flush=True)

    todo = []
    for frame_id in frame_ids:
        destination = output_dir / frame_id / "predictions.npz"
        if destination.exists() and not args.overwrite:
            try:
                with np.load(destination) as existing:
                    required = {
                        "images", "depth", "depth_conf", "extrinsic", "intrinsic",
                        "world_points_from_depth", "view_names",
                    }
                    valid = required.issubset(existing.files)
                    valid = valid and int(existing["depth"].shape[0]) == len(args.views)
                    if valid:
                        # Force reads so truncated ZIP members and CRC errors are
                        # detected, rather than trusting only the central index.
                        for key in required:
                            _ = existing[key].shape
                if valid:
                    continue
                print(f"[repair] incomplete output {destination}", flush=True)
            except Exception as exc:
                print(f"[repair] unreadable output {destination}: {exc}", flush=True)
        todo.append(frame_id)
    print(f"[run] {len(todo)}/{len(frame_ids)} -> {output_dir}", flush=True)

    started, done = time.time(), 0
    for index in range(0, len(todo), args.batch_size):
        chunk = todo[index:index + args.batch_size]
        paths = [
            base.frame_image_paths(str(frames_dir), args.views, frame_id)
            for frame_id in chunk
        ]
        tick = time.time()
        predictions = base.run_batch(
            model, paths, ext44, K_input,
            args.process_res, args.process_res_method,
            args.use_ray_pose, args.ref_view_strategy, device,
        )
        for frame_id, prediction in zip(chunk, predictions):
            destination = output_dir / frame_id / "predictions.npz"
            base.save_prediction(
                prediction, frame_id, args.views, "self_fixed",
                str(destination),
            )
        done += len(chunk)
        elapsed = time.time() - tick
        print(
            f"[{done}/{len(todo)}] {chunk[0]}..{chunk[-1]} "
            f"({elapsed:.2f}s, {elapsed / len(chunk):.2f}s/frame)",
            flush=True,
        )
    print(f"[done] {done} frames in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
