#!/usr/bin/env python
"""Run the complete Qwen-first-frame + SAM3 temporal mask pipeline.

The runner keeps model-specific commands in their own virtual environments,
uses resumable intermediate directories, and writes the validated six-view
palette masks plus review videos into ``output_root``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json


DEFAULT_VIEWS = [f"2-{index}" for index in range(1, 7)]


def frame_ids(frames_dir: Path, view: str) -> list[str]:
    return sorted(
        path.stem
        for path in (frames_dir / view).iterdir()
        if path.stem.isdigit() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def branch_complete(mask_dir: Path, frames: list[str], views: list[str]) -> bool:
    return bool(frames) and all(
        (mask_dir / timestamp / f"{view}.png").exists()
        for timestamp in frames
        for view in views
    )


def run(command: list[str], *, gpu: int | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "mask_pipeline_normalized.json")
    )
    parser.add_argument("--seed-timestamp", default="000000")
    parser.add_argument("--views", nargs="+", default=DEFAULT_VIEWS)
    parser.add_argument("--qwen-gpu", type=int, default=5)
    parser.add_argument("--video-gpu", type=int, default=6)
    parser.add_argument("--body-gpus", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument("--force", action="store_true", help="rerun completed branches")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument("--skip-body", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--skip-review", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    frames_dir = Path(config["frames_dir"])
    output_root = Path(config["output_root"])
    work_root = Path(config["work_root"])
    temporal_dir = work_root / "temporal" / "masks"
    body_dir = work_root / "body" / "masks"
    runtime_dir = work_root / "runtime_configs"
    frames = frame_ids(frames_dir, args.views[0])
    if not frames:
        raise RuntimeError(f"no frames found under {frames_dir / args.views[0]}")
    for view in args.views[1:]:
        if frame_ids(frames_dir, view) != frames:
            raise RuntimeError(f"timestamp sequence differs for {view}")
    if args.seed_timestamp not in frames:
        raise ValueError(f"seed timestamp {args.seed_timestamp} not found")

    base = {
        "frames_layout": "normalized",
        "frames_dir": str(frames_dir),
        "qwen_model": config["qwen_model"],
        "sam_ckpt": config["sam_ckpt"],
        "parts": ["lid", "body", "inner_pot"],
        "temporal_parts": ["lid", "inner_pot"],
        "prompts": config.get("prompts", {}),
    }
    temporal_config = {**base, "masks_dir": str(temporal_dir)}
    body_config = {
        **base,
        "masks_dir": str(body_dir),
        "bbox_json": str(temporal_dir / "bbox.json"),
    }
    temporal_config_path = runtime_dir / "temporal.json"
    body_config_path = runtime_dir / "body.json"
    write_json(temporal_config_path, temporal_config)
    write_json(body_config_path, body_config)

    qwen_python = config["qwen_python"]
    sam_python = config["sam_python"]
    bbox_path = temporal_dir / "bbox.json"
    if not args.skip_qwen and (args.force or not bbox_path.exists()):
        run(
            [
                qwen_python, "-u", "scripts/detect_bbox_batch.py",
                "--pipeline", str(temporal_config_path),
                "--timestamp", args.seed_timestamp,
                "--vis",
            ],
            gpu=args.qwen_gpu,
        )
    elif not bbox_path.exists():
        raise FileNotFoundError(f"Qwen bbox is required: {bbox_path}")
    else:
        print(f"Qwen: reuse {bbox_path}", flush=True)

    if not args.skip_temporal and (
        args.force or not branch_complete(temporal_dir, frames, args.views)
    ):
        run(
            [
                sam_python, "-u", "scripts/seg_masks_temporal.py",
                "--pipeline", str(temporal_config_path),
                "--all", "--init-timestamp", args.seed_timestamp,
                "--views", *args.views,
                "--parts", "lid", "inner_pot",
                "--gpu", str(args.video_gpu),
            ],
            gpu=args.video_gpu,
        )
    elif not branch_complete(temporal_dir, frames, args.views):
        raise RuntimeError("temporal branch is incomplete but --skip-temporal was requested")
    else:
        print("temporal SAM3: reuse complete branch", flush=True)

    if not args.skip_body and (args.force or not branch_complete(body_dir, frames, args.views)):
        run(
            [
                sys.executable, "-u", "scripts/run_body_multigpu.py",
                "--pipeline", str(body_config_path),
                "--views", *args.views,
                "--gpus", *(str(gpu) for gpu in args.body_gpus),
                "--python", sam_python,
                "--log-dir", str(work_root / "body" / "logs"),
            ]
        )
    elif not branch_complete(body_dir, frames, args.views):
        raise RuntimeError("body branch is incomplete but --skip-body was requested")
    else:
        print("body SAM3: reuse complete branch", flush=True)

    if not args.skip_fusion:
        command = [
            sys.executable, "-u", "scripts/fuse_all_views_temporal_masks.py",
            "--temporal-dir", str(temporal_dir),
            "--body-dir", str(body_dir),
            "--frames-dir", str(frames_dir),
            "--output-dir", str(output_root),
            "--views", *args.views,
        ]
        if args.skip_review:
            command.append("--skip-review")
        run(command)

        final_config = {
            **base,
            "data_root": str(frames_dir.parent),
            "output_root": str(output_root),
            "masks_dir": str(output_root / "masks"),
            "bbox_json": str(output_root / "masks" / "bbox.json"),
            "views": args.views,
            "mask_method": "qwen_first_frame_bbox_plus_sam3_temporal_hybrid",
        }
        write_json(output_root / "pipeline.json", final_config)
    print(f"\ndone -> {output_root}", flush=True)


if __name__ == "__main__":
    main()
