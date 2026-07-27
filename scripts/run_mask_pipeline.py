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
import shutil
import subprocess
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.masking.compose import compose_track_tree
from common.masking.io import (
    frame_path,
    load_bbox_json,
    load_binary_mask,
    save_binary_mask,
    track_path,
    validate_synchronized_frames,
    write_json,
)
from common.masking.multiview import multiview_geometric_prior
from common.masking.planning import (
    discovery_timestamps,
    needs_discovery,
    repair_jobs_from_quality,
    resolve_mask_config,
)
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
        mode = str(config.raw.get("default_tracking_mode", "video"))
        if part in config.raw.get("fixed_image_parts", []):
            mode = "fixed-image"
        return {"mode": mode}
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


def _coalesce_sam_jobs(
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for job in jobs:
        key = (
            job["part"],
            job["mode"],
            tuple(job["range"]),
            job.get("seed_frame"),
            bool(job.get("hold_previous", False)),
            bool(job.get("repair", False)),
        )
        if key not in grouped:
            grouped[key] = {**job, "views": list(job["views"])}
            continue
        for view in job["views"]:
            if view not in grouped[key]["views"]:
                grouped[key]["views"].append(view)
    return list(grouped.values())


def _job_bbox_fingerprint(
    config: MaskPipelineConfig,
    job: dict[str, Any],
    timestamps: list[str],
    bbox_data: dict[str, Any],
) -> str:
    def selected_record(timestamp: str, view: str) -> list[dict[str, Any]]:
        record = (
            bbox_data.get("frames", {})
            .get(timestamp, {})
            .get(view, {})
        )
        return [
            row for row in record.get("parts", [])
            if row.get("label") == job["part"]
        ]

    if job["mode"] == "image":
        seed_by_view = {
            view: [
                timestamp for timestamp in timestamps
                if job["range"][0] <= int(timestamp) <= job["range"][1]
            ]
            for view in job["views"]
        }
    else:
        configured = (
            job.get("seed_frame")
            if job.get("seed_frame") is not None
            else _seed_value(config, job["part"])
        )
        seed_by_view = {}
        for view in job["views"]:
            value = configured
            if isinstance(value, dict):
                value = value.get(view, value.get("default"))
            values = _flatten_frames(value)
            seed_by_view[view] = values[:1]
    evidence = {
        view: {
            timestamp: selected_record(timestamp, view)
            for timestamp in seed_by_view[view]
        }
        for view in job["views"]
    }
    return hashlib.sha256(json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _run_sam_jobs(
    config: MaskPipelineConfig,
    jobs: list[dict[str, Any]],
    *,
    gpu: int,
    timestamps: list[str],
    force: bool,
) -> None:
    jobs = _coalesce_sam_jobs(jobs)
    config_sha = hashlib.sha256(
        config.source_path.read_bytes()
    ).hexdigest()
    bbox_data = (
        load_bbox_json(config.bbox_path)
        if config.bbox_path.exists()
        else {}
    )
    for job in jobs:
        marker_payload = {
            "config": str(config.source_path),
            "config_sha256": config_sha,
            "bbox_evidence_sha256": _job_bbox_fingerprint(
                config, job, timestamps, bbox_data
            ),
            **job,
        }
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


def _anomalies_in(
    quality: dict[str, Any],
    job: dict[str, Any],
) -> int:
    view = job["views"][0]
    start, end = job["range"]
    frames = (
        quality.get(view, {})
        .get(job["part"], {})
        .get("anomaly_runs", [])
    )
    return sum(
        max(0, min(end, int(run_end)) - max(start, int(run_start)) + 1)
        for run_start, run_end in frames
    )


def _write_repair_report(
    config: MaskPipelineConfig,
    report: dict[str, Any],
) -> dict[str, Any]:
    path = config.work_root / "automatic_repair_report.json"
    history: list[dict[str, Any]] = []
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        history.extend(previous.pop("history", []))
        history.append(previous)
    if history:
        report["history"] = history
    write_json(path, report)
    return report


def _automatic_repairs(
    config: MaskPipelineConfig,
    timestamps: list[str],
    *,
    qwen_gpu: int,
    gpu: int,
    force_qwen: bool,
    selected_parts: Iterable[str] | None = None,
    selected_views: Iterable[str] | None = None,
) -> dict[str, Any]:
    quality_path = config.output_root / "quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    jobs = repair_jobs_from_quality(quality, config, timestamps)
    part_filter = set(selected_parts or config.part_names)
    view_filter = set(selected_views or config.views)
    jobs = [
        job for job in jobs
        if job["part"] in part_filter
        and job["views"][0] in view_filter
    ]
    if not jobs:
        return _write_repair_report(config, {
            "enabled": True,
            "jobs": [],
            "accepted": 0,
            "rejected": 0,
        })
    settings = config.raw.get("automation", {}).get("repair", {})
    if not settings.get("apply", False):
        report = {
            "enabled": True,
            "apply": False,
            "jobs": jobs,
            "accepted": 0,
            "rejected": 0,
        }
        return _write_repair_report(config, report)

    _run_qwen(
        config,
        sorted({job["seed_frame"] for job in jobs}, key=int),
        views=sorted({job["views"][0] for job in jobs}),
        parts=sorted({job["part"] for job in jobs}),
        gpu=qwen_gpu,
        force=force_qwen,
    )
    bbox = load_bbox_json(config.bbox_path)
    runnable = []
    for job in jobs:
        rows = (
            bbox.get("frames", {})
            .get(job["seed_frame"], {})
            .get(job["views"][0], {})
            .get("parts", [])
        )
        if any(row.get("label") == job["part"] for row in rows):
            runnable.append(job)

    backup_root = config.work_root / "repair_backups"
    backups: dict[tuple[str, str, str], Path] = {}
    for job in runnable:
        for frame in range(job["range"][0], job["range"][1] + 1):
            timestamp = f"{frame:06d}"
            source = track_path(
                config.tracks_root,
                job["part"],
                timestamp,
                job["views"][0],
            )
            if source.exists():
                destination = (
                    backup_root / job["part"] / timestamp
                    / f"{job['views'][0]}.png"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                backups[(job["part"], timestamp, job["views"][0])] = destination

    _run_sam_jobs(
        config,
        runnable,
        gpu=gpu,
        timestamps=timestamps,
        force=True,
    )
    compose_track_tree(config, timestamps)
    repaired_quality = json.loads(quality_path.read_text(encoding="utf-8"))
    accepted, rejected = [], []
    for job in runnable:
        if _anomalies_in(repaired_quality, job) < _anomalies_in(quality, job):
            accepted.append(job)
            continue
        rejected.append(job)
        for frame in range(job["range"][0], job["range"][1] + 1):
            timestamp = f"{frame:06d}"
            backup = backups.get(
                (job["part"], timestamp, job["views"][0])
            )
            if backup is not None:
                shutil.copy2(
                    backup,
                    track_path(
                        config.tracks_root,
                        job["part"],
                        timestamp,
                        job["views"][0],
                    ),
                )
    if rejected:
        compose_track_tree(config, timestamps)
    report = {
        "enabled": True,
        "apply": True,
        "jobs": jobs,
        "runnable_jobs": len(runnable),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "skipped_without_qwen_box": len(jobs) - len(runnable),
        "accepted_jobs": accepted,
        "rejected_jobs": rejected,
    }
    return _write_repair_report(config, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "all", "discover", "qwen", "sam", "geometry", "compose", "repair"
        ),
        default="all",
    )
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--qwen-gpu", type=int)
    parser.add_argument("--sam-gpu", type=int)
    parser.add_argument("--force-qwen", action="store_true")
    parser.add_argument("--force-sam", action="store_true")
    args = parser.parse_args()

    source_config = load_mask_pipeline_config(args.config)
    resolved_path = (
        source_config.work_root / "manifests" / "resolved_mask_config.json"
    )
    config = source_config
    if (
        args.stage not in {"all", "discover"}
        and needs_discovery(source_config)
    ):
        if not resolved_path.exists():
            raise RuntimeError(
                "mask config contains automatic starts/seeds; run "
                "--stage discover first"
            )
        config = load_mask_pipeline_config(resolved_path)
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

    if args.stage in {"all", "discover"} and needs_discovery(source_config):
        discovery = source_config.raw.get(
            "automation", {}
        ).get("discovery", {})
        scan_frames = discovery_timestamps(
            timestamps,
            stride=int(discovery.get("stride", 10)),
        )
        _run_qwen(
            source_config,
            scan_frames,
            views=views,
            parts=selected_parts,
            gpu=qwen_gpu,
            force=args.force_qwen,
        )
        _coarse_raw, coarse_report = resolve_mask_config(
            source_config,
            load_bbox_json(source_config.bbox_path),
            scan_frames,
            output_path=resolved_path,
        )
        refinement_frames: set[str] = set()
        for part in source_config.parts:
            if not part.start_frame_auto:
                continue
            evidence = coarse_report["parts"][part.name]["start_evidence"]
            previous = evidence.get("previous_negative_scan_frame")
            selected = evidence.get("selected_scan_frame")
            if previous is None or selected is None:
                continue
            refinement_frames.update(
                f"{frame:06d}"
                for frame in range(int(previous) + 1, int(selected) + 1)
            )
        if refinement_frames:
            ordered_refinement = sorted(refinement_frames, key=int)
            _run_qwen(
                source_config,
                ordered_refinement,
                views=views,
                parts=selected_parts,
                gpu=qwen_gpu,
                force=args.force_qwen,
            )
            resolve_mask_config(
                source_config,
                load_bbox_json(source_config.bbox_path),
                sorted(set(scan_frames) | refinement_frames, key=int),
                output_path=resolved_path,
            )
        config = load_mask_pipeline_config(resolved_path)
        print(f"resolved mask config -> {resolved_path}", flush=True)
    elif args.stage == "discover":
        print(
            "mask config has no automatic start or seed fields; "
            "explicit overrides are already resolved",
            flush=True,
        )
    if args.stage == "discover":
        return

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
    if args.stage in {"all", "compose", "repair"}:
        compose_views = (
            list(config.views)
            if args.stage == "repair"
            or (
                args.stage == "all"
                and config.raw.get("automation", {})
                .get("repair", {})
                .get("enabled", False)
            )
            else views
        )
        summary = compose_track_tree(
            config,
            timestamps,
            views=compose_views,
        )
        print(f"final masks -> {summary['masks_root']}", flush=True)
    repair = config.raw.get("automation", {}).get("repair", {})
    if args.stage == "repair" or (
        args.stage == "all" and repair.get("enabled", False)
    ):
        report = _automatic_repairs(
            config,
            timestamps,
            qwen_gpu=qwen_gpu,
            gpu=sam_gpu,
            force_qwen=args.force_qwen,
            selected_parts=selected_parts,
            selected_views=views,
        )
        print(
            "automatic mask repair: "
            f"accepted={report['accepted']} rejected={report['rejected']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
