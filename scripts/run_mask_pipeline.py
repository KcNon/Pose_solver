#!/usr/bin/env python
"""Internal Qwen -> SAM -> compose adapter used by :mod:`pose_solver`.

Qwen and SAM keep separate Python environments.  This runner owns only task
planning, resumability, seed validation, and final mask composition.
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
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.masking.compose import compose_track_tree
from common.masking.io import (
    load_bbox_json,
    track_path,
    validate_synchronized_frames,
    write_json,
)
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
    requested_parts = list(selected_parts or config.part_names)
    event_parts = (
        list(config.part_names)
        if config.raw.get("qwen_reanchor_on_part_events", False)
        else requested_parts
    )
    seed_window = config.raw.get("qwen_seed_window", {})
    window_length = int(seed_window.get("length", 0))
    window_stride = max(1, int(seed_window.get("stride", 1)))
    for part in event_parts:
        tracking = _tracking_config(config, part)
        mode = tracking.get("mode", "video")
        if mode == "image":
            explicit = tracking.get("frames", config.raw.get("qwen_timestamps"))
            frames.extend(_flatten_frames(explicit))
        else:
            seeds = _flatten_frames(_seed_value(config, part))
            frames.extend(seeds)
            if window_length > 0:
                for seed in seeds:
                    if not seed.isdigit():
                        continue
                    start = int(seed)
                    frames.extend(
                        f"{frame:06d}"
                        for frame in range(
                            start + window_stride,
                            start + window_length + 1,
                            window_stride,
                        )
                    )
        if part in requested_parts:
            for segment in tracking.get("segments", []):
                frames.extend(_flatten_frames(segment.get("seed_frame")))
    periodic_stride = int(config.raw.get("qwen_periodic_stride", 0))
    if periodic_stride > 0 and requested_parts:
        frame_ids = validate_synchronized_frames(config.frames_dir, config.views)
        first = min(config.part_map[part].start_frame for part in requested_parts)
        last = int(frame_ids[-1])
        frames.extend(
            f"{frame:06d}"
            for frame in range(first, last + 1, periodic_stride)
        )
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
    reference_guided: bool = False,
    initialization: bool = False,
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
    if reference_guided:
        command.append("--mesh-references")
    if config.raw.get("qwen_separate_parts", False):
        command.append("--separate-parts")
    if initialization:
        seed_window = config.raw.get("qwen_seed_window", {})
        if config.raw.get("qwen_reanchor_on_part_events", False):
            command.append("--reanchor-active-parts")
        if int(seed_window.get("length", 0)) > 0:
            command.extend([
                "--start-window",
                str(int(seed_window["length"])),
            ])
        elif config.raw.get("qwen_only_starting_parts", False):
            command.append("--only-starting-parts")
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
        mesh_dir = Path(
            config.raw.get("mesh_dir", config.frames_dir.parent / "meshes")
        )
        require_mesh_assignment = bool(
            config.raw.get(
                "require_mesh_assignment",
                (mesh_dir / f"{job['part']}.glb").exists(),
            )
        )
        for view in job["views"]:
            value = configured
            if isinstance(value, dict):
                value = value.get(view, value.get("default"))
            values = _flatten_frames(value)
            seed_by_view[view] = values[:1]
            # Video mode can use every trusted Qwen re-anchor in the job
            # range. Include all corresponding evidence so changing an
            # adaptive or periodic anchor invalidates the resumability marker.
            seed_by_view[view].extend(
                timestamp
                for timestamp, records in bbox_data.get("frames", {}).items()
                if job["range"][0] <= int(timestamp) <= job["range"][1]
                if any(
                    row.get("label") == job["part"]
                    for row in records.get(view, {}).get("parts", [])
                )
                and (
                    not require_mesh_assignment
                    or records.get(view, {}).get("mesh_assignment", {}).get(
                        "status"
                    ) == "ok"
                )
            )
            seed_by_view[view] = sorted(set(seed_by_view[view]), key=int)
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


def _run_multiview_seed_validation(
    config: MaskPipelineConfig,
    seed_frames: list[str],
    *,
    views: list[str],
    parts: list[str],
    gpu: int,
) -> None:
    if not config.raw.get("multiview_seed_validation", {}).get(
        "enabled", False
    ):
        return
    command = [
        config.raw["sam_python"],
        "-u",
        "tools/stages/masking/validate_multiview_seeds.py",
        "--config",
        str(config.source_path),
        "--timestamps",
        *seed_frames,
        "--views",
        *views,
        "--parts",
        *parts,
        "--gpu",
        str(gpu),
    ]
    _run(command, gpu)


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
        # Repair must preserve the same object-agnostic contract as initial
        # discovery.  Mesh/reconstruction labels are optional and can be
        # semantically wrong for a new object category.
        reference_guided=bool(config.raw.get("qwen_reference_guided", True)),
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
                # Repair backups only need the mask bytes.  ``copy2`` also
                # attempts to preserve timestamps, which is rejected by some
                # read/write data mounts even though creating the backup is
                # allowed.  Avoid making pipeline success depend on metadata
                # operations that are irrelevant to rollback.
                shutil.copyfile(source, destination)
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
                shutil.copyfile(
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "all", "discover", "qwen", "sam", "compose", "repair"
        ),
        default="all",
    )
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--qwen-gpu", type=int)
    parser.add_argument("--sam-gpu", type=int)
    parser.add_argument(
        "--range-start",
        type=int,
        help="only process synchronized frames at or after this frame index",
    )
    parser.add_argument(
        "--range-end",
        type=int,
        help="only process synchronized frames at or before this frame index",
    )
    parser.add_argument("--force-qwen", action="store_true")
    parser.add_argument("--force-sam", action="store_true")
    args = parser.parse_args(argv)

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
    if (
        args.range_start is not None
        and args.range_end is not None
        and args.range_start > args.range_end
    ):
        raise ValueError("--range-start must not exceed --range-end")
    if args.range_start is not None:
        timestamps = [
            timestamp for timestamp in timestamps
            if int(timestamp) >= args.range_start
        ]
    if args.range_end is not None:
        timestamps = [
            timestamp for timestamp in timestamps
            if int(timestamp) <= args.range_end
        ]
    if not timestamps:
        raise ValueError("requested frame range contains no synchronized frames")
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
        planned_scan_frames = discovery_timestamps(
            timestamps,
            stride=int(discovery.get("stride", 10)),
            maximum_frame=(
                int(discovery["maximum_frame"])
                if discovery.get("maximum_frame") is not None
                else None
            ),
        )
        stop_when_resolved = bool(discovery.get("stop_when_resolved", False))
        batch_size = int(discovery.get("batch_size", len(planned_scan_frames)))
        if batch_size <= 0:
            raise ValueError("discovery batch_size must be positive")
        scan_frames: list[str] = []
        coarse_report = None
        for offset in range(0, len(planned_scan_frames), batch_size):
            batch = planned_scan_frames[offset:offset + batch_size]
            _run_qwen(
                source_config,
                batch,
                views=views,
                parts=selected_parts,
                gpu=qwen_gpu,
                force=args.force_qwen,
                reference_guided=True,
            )
            scan_frames.extend(batch)
            if not stop_when_resolved:
                continue
            try:
                _coarse_raw, coarse_report = resolve_mask_config(
                    source_config,
                    load_bbox_json(source_config.bbox_path),
                    scan_frames,
                    output_path=resolved_path,
                )
            except RuntimeError:
                if offset + batch_size >= len(planned_scan_frames):
                    raise
                continue
            print(
                "discovery stopped after all parts had consecutive "
                "multi-view support and per-view seeds: "
                f"{scan_frames[-1]}",
                flush=True,
            )
            break
        if coarse_report is None:
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
            refinement_batch_size = int(
                discovery.get("refinement_batch_size", len(ordered_refinement))
            )
            if refinement_batch_size <= 0:
                raise ValueError("discovery refinement_batch_size must be positive")
            stop_when_refined = bool(
                discovery.get("stop_when_refined_resolved", False)
            )
            processed_refinement: list[str] = []
            for offset in range(0, len(ordered_refinement), refinement_batch_size):
                batch = ordered_refinement[offset:offset + refinement_batch_size]
                _run_qwen(
                    source_config,
                    batch,
                    views=views,
                    parts=selected_parts,
                    gpu=qwen_gpu,
                    force=args.force_qwen,
                    reference_guided=True,
                )
                processed_refinement.extend(batch)
                _refined_raw, refined_report = resolve_mask_config(
                    source_config,
                    load_bbox_json(source_config.bbox_path),
                    sorted(set(scan_frames) | set(processed_refinement), key=int),
                    output_path=resolved_path,
                )
                if not stop_when_refined:
                    continue
                consecutive = int(discovery.get("consecutive_scans", 2))
                latest_resolvable_start = int(processed_refinement[-1]) - consecutive + 1
                auto_parts = [
                    part.name for part in source_config.parts
                    if part.start_frame_auto and part.name in selected_parts
                ]
                all_refined = all(
                    int(
                        refined_report["parts"][part]["start_evidence"]
                        ["selected_scan_frame"]
                    ) <= latest_resolvable_start
                    for part in auto_parts
                )
                if all_refined:
                    print(
                        "discovery refinement stopped after every automatic "
                        "part had an earliest consecutive run: "
                        f"{processed_refinement[-1]}",
                        flush=True,
                    )
                    break
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
            reference_guided=bool(
                config.raw.get("qwen_reference_guided", True)
            ),
            initialization=True,
        )
    if args.stage in {"all", "sam"}:
        _run_multiview_seed_validation(
            config,
            _seed_frames(config, selected_parts),
            views=views,
            parts=selected_parts,
            gpu=sam_gpu,
        )
        _run_sam_jobs(
            config,
            _sam_jobs(config, timestamps, selected_parts, views),
            gpu=sam_gpu,
            timestamps=timestamps,
            force=args.force_sam,
        )
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
