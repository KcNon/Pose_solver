#!/usr/bin/env python
"""Orchestrate reusable Qwen -> SAM -> compose mask extraction.

Qwen and SAM keep separate Python environments.  This runner owns only task
planning, resumability, optional depth-aware multi-view priors, and final mask
composition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.masking.compose import compose_track_tree
from common.masking.io import (
    frame_path,
    load_binary_mask,
    save_binary_mask,
    track_path,
    validate_synchronized_frames,
    write_json,
)
from common.masking.multiview import multiview_geometric_prior
from common.masking.schema import MaskPipelineConfig, load_mask_pipeline_config


def _run(command: list[str], gpu: int | None = None) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _flatten_frames(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        result = []
        for nested in value.values():
            result.extend(_flatten_frames(nested))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for nested in value:
            result.extend(_flatten_frames(nested))
        return result
    text = str(value)
    return [f"{int(text):06d}" if text.isdigit() else text]


def _part_config(config: MaskPipelineConfig, part: str) -> dict[str, Any]:
    parts = config.raw.get("parts", {})
    return dict(parts.get(part, {})) if isinstance(parts, dict) else {}


def _tracking_config(config: MaskPipelineConfig, part: str) -> dict[str, Any]:
    part_values = _part_config(config, part)
    configured = part_values.get("tracking", part_values.get("tracker", {}))
    if not configured and isinstance(config.raw.get("parts"), list):
        # Compatibility with the original two-branch rice-cooker config.
        return {"mode": "fixed-image" if part == "body" else "video"}
    if isinstance(configured, str):
        return {"mode": configured}
    return dict(configured)


def _seed_value(config: MaskPipelineConfig, part: str) -> Any:
    tracking = _tracking_config(config, part)
    value = tracking.get("seed_frames", tracking.get("seed_frame"))
    if value is None:
        value = config.raw.get("seed_frames", {}).get(part)
    if value is None:
        value = config.raw.get("temporal_seed_timestamps", {}).get(part)
    if part == "body":
        body_seeds = config.raw.get("body_seed_timestamps", {})
        value = body_seeds or value
    return config.part_map[part].start_frame if value is None else value


def _seed_frames(
    config: MaskPipelineConfig,
    selected_parts: Iterable[str] | None = None,
) -> list[str]:
    frames = []
    for part in selected_parts or config.part_names:
        tracking = _tracking_config(config, part)
        mode = tracking.get("mode", "video")
        if mode == "image":
            explicit = tracking.get("frames", config.raw.get("qwen_timestamps"))
            frames.extend(_flatten_frames(explicit))
        else:
            frames.extend(_flatten_frames(_seed_value(config, part)))
        for segment in tracking.get("segments", []):
            frames.extend(_flatten_frames(segment.get("seed_frame")))
    return sorted(set(frames), key=int)


def _sam_jobs(
    config: MaskPipelineConfig,
    timestamps: list[str],
    selected_parts: Iterable[str],
    selected_views: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    jobs = []
    requested_views = list(selected_views or config.views)
    for part in selected_parts:
        spec = config.part_map[part]
        tracking = _tracking_config(config, part)
        mode = tracking.get("mode", "video")
        base = {
            "part": part,
            "mode": mode,
            "views": requested_views,
            "range": [spec.start_frame, int(timestamps[-1])],
            "seed_frame": None,
            "hold_previous": bool(tracking.get("hold_previous", mode == "fixed-image")),
            "repair": False,
        }
        scalar_seed = _flatten_frames(_seed_value(config, part))
        if len(set(scalar_seed)) == 1:
            base["seed_frame"] = scalar_seed[0]
        jobs.append(base)
        for segment in tracking.get("segments", []):
            start, end = segment["range"]
            seed_text = str(segment["seed_frame"])
            segment_views = set(segment.get("views", config.views))
            views = [view for view in requested_views if view in segment_views]
            if not views:
                continue
            jobs.append({
                "part": part,
                "mode": segment.get("mode", mode),
                "views": views,
                "range": [int(start), int(end)],
                "seed_frame": (
                    f"{int(seed_text):06d}" if seed_text.isdigit() else seed_text
                ),
                "hold_previous": bool(segment.get("hold_previous", False)),
                "repair": True,
            })
    return jobs


def _run_qwen(
    config: MaskPipelineConfig,
    seed_frames: list[str],
    *,
    views: list[str],
    parts: list[str],
    gpu: int,
    force: bool,
) -> None:
    if not seed_frames:
        raise ValueError("no Qwen seed frames were configured")
    command = [
        config.raw["qwen_python"],
        "-u",
        "tools/stages/masking/detect_mask_seeds.py",
        "--config",
        str(config.source_path),
        "--timestamps",
        *seed_frames,
        "--views",
        *views,
        "--parts",
        *parts,
        "--vis",
    ]
    if force:
        command.append("--force")
    _run(command, gpu)


def _run_sam_jobs(
    config: MaskPipelineConfig,
    jobs: list[dict[str, Any]],
    *,
    gpu: int,
    timestamps: list[str],
    force: bool,
) -> None:
    for job in jobs:
        marker_payload = {"config": str(config.source_path), **job}
        marker_id = hashlib.sha256(
            json.dumps(marker_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        marker_path = (
            config.work_root / "manifests" / "jobs"
            / f"{job['part']}_{marker_id}.json"
        )
        expected = [
            track_path(config.tracks_root, job["part"], timestamp, view)
            for timestamp in timestamps
            if job["range"][0] <= int(timestamp) <= job["range"][1]
            for view in job["views"]
        ]
        if (
            not force
            and marker_path.exists()
            and expected
            and all(path.exists() for path in expected)
        ):
            print(
                f"reuse SAM job: {job['part']} {job['range']} {job['views']}",
                flush=True,
            )
            continue
        command = [
            config.raw["sam_python"],
            "-u",
            "tools/stages/masking/track_part_masks.py",
            "--config",
            str(config.source_path),
            "--mode",
            job["mode"],
            "--part",
            job["part"],
            "--all",
            "--views",
            *job["views"],
            "--range-start",
            f"{job['range'][0]:06d}",
            "--range-end",
            f"{job['range'][1]:06d}",
            "--gpu",
            str(gpu),
        ]
        if job["seed_frame"] is not None:
            command.extend(["--seed-frame", str(job["seed_frame"])])
        if job["hold_previous"]:
            command.append("--hold-previous")
        _run(command, gpu)
        write_json(marker_path, marker_payload)


def _multiview_priors(
    config: MaskPipelineConfig,
    timestamps: list[str],
    selected_parts: Iterable[str],
    selected_views: Iterable[str] | None = None,
) -> dict:
    import cv2

    settings = config.raw.get("multiview_completion", {})
    if not settings.get("enabled", False):
        return {"enabled": False}
    from common.normalized_recon import load_recon

    prior_root = config.work_root / "multiview_priors"
    minimum_pixels = int(settings.get("minimum_source_pixels", 100))
    target_threshold = int(settings.get("target_failure_pixels", 100))
    minimum_views = int(settings.get("minimum_source_views", 1))
    depth_tolerance = float(settings.get("depth_tolerance", 0.03))
    apply_mode = str(settings.get("apply_mode", "prior_only"))
    if apply_mode not in {"prior_only", "replace_failed"}:
        raise ValueError("multiview_completion.apply_mode must be prior_only or replace_failed")
    reports: dict[str, Any] = {}
    views = list(selected_views or config.views)
    view_indices = [config.views.index(view) for view in views]
    for timestamp in timestamps:
        recon = load_recon(config.raw, timestamp)
        depth = recon["depth"][view_indices]
        intrinsics = recon["intrinsics"][view_indices]
        extrinsics = recon["extrinsics"][view_indices]
        height, width = recon["depth_hw"]
        timestamp_report = {}
        for part in selected_parts:
            if int(timestamp) < config.part_map[part].start_frame:
                continue
            masks = []
            reliable = []
            full_shapes = []
            for view in views:
                source_path = track_path(config.tracks_root, part, timestamp, view)
                if source_path.exists():
                    mask = load_binary_mask(source_path)
                else:
                    image = cv2.imread(
                        str(frame_path(config.frames_dir, view, timestamp)),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    if image is None:
                        raise RuntimeError(f"failed to read frame {timestamp}/{view}")
                    mask = np.zeros(image.shape, dtype=bool)
                full_shapes.append(mask.shape)
                small = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                masks.append(small)
                reliable.append(int(small.sum()) >= minimum_pixels)
            part_report = {}
            for target_index, view in enumerate(views):
                if int(masks[target_index].sum()) >= target_threshold:
                    continue
                prior, report = multiview_geometric_prior(
                    masks,
                    reliable,
                    depth,
                    intrinsics,
                    extrinsics,
                    target_index,
                    minimum_source_views=minimum_views,
                    depth_tolerance=depth_tolerance,
                    minimum_pixels=minimum_pixels,
                )
                full = cv2.resize(
                    prior.astype(np.uint8),
                    (full_shapes[target_index][1], full_shapes[target_index][0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                destination = track_path(prior_root, part, timestamp, view)
                save_binary_mask(destination, full)
                report["full_resolution_pixels"] = int(full.sum())
                part_report[view] = report
                if apply_mode == "replace_failed" and int(full.sum()) >= target_threshold:
                    save_binary_mask(
                        track_path(config.tracks_root, part, timestamp, view),
                        full,
                    )
            if part_report:
                timestamp_report[part] = part_report
        if timestamp_report:
            reports[timestamp] = timestamp_report
    summary = {
        "enabled": True,
        "apply_mode": apply_mode,
        "prior_root": str(prior_root),
        "frames_with_candidates": len(reports),
        "reports": reports,
    }
    write_json(config.work_root / "multiview_completion.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "qwen", "sam", "geometry", "compose"),
        default="all",
    )
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--qwen-gpu", type=int)
    parser.add_argument("--sam-gpu", type=int)
    parser.add_argument("--force-qwen", action="store_true")
    parser.add_argument("--force-sam", action="store_true")
    args = parser.parse_args()

    config = load_mask_pipeline_config(args.config)
    views = args.views or list(config.views)
    unknown_views = set(views).difference(config.views)
    if unknown_views:
        raise ValueError(f"unknown views: {sorted(unknown_views)}")
    selected_parts = args.parts or config.part_names
    unknown_parts = set(selected_parts).difference(config.part_names)
    if unknown_parts:
        raise ValueError(f"unknown parts: {sorted(unknown_parts)}")
    timestamps = validate_synchronized_frames(config.frames_dir, views)
    qwen_gpu = args.qwen_gpu
    if qwen_gpu is None:
        qwen_gpu = int(config.raw.get("runtime", {}).get("qwen_gpu", 0))
    sam_gpu = args.sam_gpu
    if sam_gpu is None:
        sam_gpu = int(config.raw.get("runtime", {}).get("sam_gpu", 0))

    if args.stage in {"all", "qwen"}:
        _run_qwen(
            config,
            _seed_frames(config, selected_parts),
            views=views,
            parts=selected_parts,
            gpu=qwen_gpu,
            force=args.force_qwen,
        )
    if args.stage in {"all", "sam"}:
        _run_sam_jobs(
            config,
            _sam_jobs(config, timestamps, selected_parts, views),
            gpu=sam_gpu,
            timestamps=timestamps,
            force=args.force_sam,
        )
    if args.stage in {"all", "geometry"}:
        _multiview_priors(config, timestamps, selected_parts, views)
    if args.stage in {"all", "compose"}:
        summary = compose_track_tree(
            config,
            timestamps,
            views=views,
        )
        print(f"final masks -> {summary['masks_root']}", flush=True)


if __name__ == "__main__":
    main()
