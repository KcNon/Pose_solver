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
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DA3_ROOT = Path("/data_ft_9_10/wentai/projects/depth-anything-3")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DA3_ROOT))

import run_da3_multiview_video as base
from depth_anything_3.api import DepthAnything3

from common.depth_artifact import (
    prediction_compatibility,
    prediction_metadata,
)


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
    camera_group = parser.add_mutually_exclusive_group(required=True)
    camera_group.add_argument("--camera-frame")
    camera_group.add_argument(
        "--camera-npz",
        type=Path,
        help="reuse a previous fixed rig while changing DA3 process resolution",
    )
    camera_group.add_argument(
        "--camera-frames",
        nargs="+",
        help=(
            "estimate the fixed rig on several static timestamps and robustly "
            "average the relative cameras"
        ),
    )
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
    parser.add_argument(
        "--allow-legacy-shape-resume",
        action="store_true",
        help=(
            "reuse pre-metadata NPZ files only when their actual depth shape "
            "matches the requested upper-bound resolution"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a noisy 3x3 matrix onto SO(3)."""

    left, _, right = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def robust_average_relative_rigs(
    rigs: list[np.ndarray],
) -> tuple[np.ndarray, dict]:
    """Average w2c rigs after expressing every estimate in camera-0 space."""

    if not rigs:
        raise ValueError("at least one camera rig is required")
    normalized = []
    baselines = []
    for rig in rigs:
        value = np.asarray(rig, dtype=np.float64)
        relative = value @ np.linalg.inv(value[0])[None]
        normalized.append(relative)
        lengths = np.linalg.norm(relative[1:, :3, 3], axis=1)
        positive = lengths[np.isfinite(lengths) & (lengths > 1e-8)]
        baselines.append(float(np.median(positive)) if len(positive) else 1.0)
    target_baseline = float(np.median(baselines))
    for relative, baseline in zip(normalized, baselines):
        if baseline > 1e-8:
            relative[:, :3, 3] *= target_baseline / baseline

    stack = np.stack(normalized)
    averaged = np.tile(np.eye(4, dtype=np.float64), (stack.shape[1], 1, 1))
    rotation_spread = []
    translation_spread = []
    for view in range(stack.shape[1]):
        averaged[view, :3, :3] = _project_rotation(
            np.sum(stack[:, view, :3, :3], axis=0)
        )
        averaged[view, :3, 3] = np.median(stack[:, view, :3, 3], axis=0)
        angles = []
        distances = []
        for sample in stack[:, view]:
            delta = sample[:3, :3] @ averaged[view, :3, :3].T
            cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cosine))))
            distances.append(float(np.linalg.norm(
                sample[:3, 3] - averaged[view, :3, 3]
            )))
        rotation_spread.append(float(np.median(angles)))
        translation_spread.append(float(np.median(distances)))
    return averaged.astype(np.float32), {
        "samples": len(rigs),
        "baseline_median": target_baseline,
        "input_baselines": baselines,
        "rotation_spread_median_deg_by_view": rotation_spread,
        "translation_spread_median_m_by_view": translation_spread,
    }


def _source_image_hw(paths: list[str]) -> tuple[int, int]:
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append((int(image.height), int(image.width)))
    unique = sorted(set(sizes))
    if len(unique) != 1:
        raise ValueError(f"source views have inconsistent image sizes: {unique}")
    return unique[0]


def load_camera_npz(
    path: Path,
    views: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load and reorder a fixed w2c rig from a prior DA3 prediction."""

    with np.load(path) as artifact:
        stored_views = [str(value) for value in artifact["view_names"].tolist()]
        index_by_view = {view: index for index, view in enumerate(stored_views)}
        missing = [view for view in views if view not in index_by_view]
        if missing:
            raise ValueError(f"camera NPZ is missing configured views: {missing}")
        selected = [index_by_view[view] for view in views]
        ext34 = np.asarray(artifact["extrinsic"], np.float32)[selected]
        intrinsic = np.asarray(artifact["intrinsic"], np.float32)[selected]
    ext44 = np.tile(np.eye(4, dtype=np.float32), (len(views), 1, 1))
    ext44[:, :3, :4] = ext34
    return ext44, intrinsic


def save_prediction(
    prediction,
    frame_id: str,
    view_names: list[str],
    output_path: Path,
    *,
    process_res: int,
    process_res_method: str,
    source_image_hw: tuple[int, int],
    camera_frames: list[str],
    model_dir: str,
    use_ray_pose: bool,
    ref_view_strategy: str,
) -> None:
    """Write the DA3 tensors and the exact settings needed for safe resume."""

    depth = np.asarray(prediction.depth, np.float32)
    confidence = (
        np.asarray(prediction.conf, np.float32)
        if prediction.conf is not None
        else None
    )
    extrinsic = np.asarray(prediction.extrinsics, np.float32)
    intrinsic = np.asarray(prediction.intrinsics, np.float32)
    images = prediction.processed_images.astype(np.float32) / 255.0
    n_views, height, width = depth.shape
    ext44 = np.tile(np.eye(4, dtype=np.float32), (n_views, 1, 1))
    ext44[:, :3, :4] = extrinsic
    camera_to_world = np.linalg.inv(ext44)
    world_points = base.unproject_depth(
        torch.from_numpy(depth[None, ..., None]).float(),
        torch.from_numpy(intrinsic[None]).float(),
        torch.from_numpy(camera_to_world[None]).float(),
    )[0].numpy().astype(np.float32)
    payload = {
        "images": np.transpose(images, (0, 3, 1, 2)),
        "depth": depth[..., None],
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "world_points_from_depth": world_points,
        "frame_id": np.asarray(frame_id),
        "view_names": np.asarray(view_names),
        "camera_mode": np.asarray(
            "self_fixed_multiframe" if len(camera_frames) > 1 else "self_fixed"
        ),
        **prediction_metadata(
            process_res=process_res,
            process_res_method=process_res_method,
            source_image_hw=source_image_hw,
            processed_image_hw=(height, width),
            camera_frames=camera_frames,
            model_dir=model_dir,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
        ),
    }
    if confidence is not None:
        payload["depth_conf"] = confidence
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output_path)


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    camera_frames_dir = Path(args.camera_frames_dir or args.frames_dir)
    output_dir = Path(args.output_dir)
    available = base.discover_frames(str(frames_dir), args.views)
    camera_frames = (
        [f"npz:{args.camera_npz.resolve()}"]
        if args.camera_npz is not None
        else list(args.camera_frames or [args.camera_frame])
    )
    if args.camera_npz is None:
        camera_available = base.discover_frames(
            str(camera_frames_dir), args.views
        )
        unavailable_camera_frames = sorted(
            set(camera_frames).difference(camera_available)
        )
        if unavailable_camera_frames:
            raise ValueError(
                f"camera frames {unavailable_camera_frames} are unavailable in "
                f"{camera_frames_dir}"
            )
    frame_ids = args.timestamps or available
    missing = sorted(set(frame_ids) - set(available))
    if missing:
        raise ValueError(f"timestamps unavailable in all views: {missing}")

    first_paths = base.frame_image_paths(
        str(frames_dir), args.views, frame_ids[0]
    )
    source_image_hw = _source_image_hw(first_paths)
    requested_hw = (int(args.full_h), int(args.full_w))
    if source_image_hw != requested_hw:
        raise ValueError(
            f"configured full image size {requested_hw} does not match "
            f"source images {source_image_hw}"
        )

    device = torch.device(args.device)
    print(f"[views] {len(args.views)}: {args.views}", flush=True)
    print(
        f"[frames] {len(frame_ids)}, camera frames={camera_frames}",
        flush=True,
    )
    print(f"[model] loading {args.model_dir}", flush=True)
    model = DepthAnything3.from_pretrained(args.model_dir).to(device=device)

    if args.camera_npz is not None:
        ext44, K_native = load_camera_npz(args.camera_npz, args.views)
        rig_report = {"source": str(args.camera_npz.resolve())}
    else:
        rig_samples = []
        intrinsic_samples = []
        for camera_frame in camera_frames:
            camera_paths = base.frame_image_paths(
                str(camera_frames_dir), args.views, camera_frame
            )
            ext_sample, intrinsic_sample = base.cameras_from_self(
                model,
                camera_paths,
                args.process_res,
                args.process_res_method,
                args.use_ray_pose,
            )
            rig_samples.append(ext_sample)
            intrinsic_samples.append(intrinsic_sample)
        if len(rig_samples) == 1:
            ext44 = rig_samples[0]
            K_native = intrinsic_samples[0]
            rig_report = {"samples": 1}
        else:
            ext44, rig_report = robust_average_relative_rigs(rig_samples)
            K_native = np.median(np.stack(intrinsic_samples), axis=0).astype(
                np.float32
            )
    K_input = K_native.astype(np.float32).copy()
    for index in range(len(args.views)):
        K_input[index, 0, :] *= args.full_w / (2.0 * K_native[index, 0, 2])
        K_input[index, 1, :] *= args.full_h / (2.0 * K_native[index, 1, 2])
    print(f"[cameras] fixed self-estimated rig ready {rig_report}", flush=True)

    todo = []
    for frame_id in frame_ids:
        destination = output_dir / frame_id / "predictions.npz"
        if destination.exists() and not args.overwrite:
            try:
                with np.load(destination) as existing:
                    valid, reason = prediction_compatibility(
                        existing,
                        views=args.views,
                        process_res=args.process_res,
                        process_res_method=args.process_res_method,
                        source_image_hw=source_image_hw,
                        camera_frames=camera_frames,
                        model_dir=args.model_dir,
                        use_ray_pose=args.use_ray_pose,
                        ref_view_strategy=args.ref_view_strategy,
                        allow_legacy_shape_resume=(
                            args.allow_legacy_shape_resume
                        ),
                    )
                if valid:
                    continue
                print(
                    f"[repair] incompatible output {destination}: {reason}",
                    flush=True,
                )
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
            save_prediction(
                prediction,
                frame_id,
                args.views,
                destination,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                source_image_hw=source_image_hw,
                camera_frames=camera_frames,
                model_dir=args.model_dir,
                use_ray_pose=args.use_ray_pose,
                ref_view_strategy=args.ref_view_strategy,
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
