"""Data-driven planning for Qwen seed discovery and SAM tracking.

This module has no model dependency.  Qwen writes normalized boxes; these
helpers turn that evidence into a resolved, auditable mask configuration.
Explicit integer starts and seed frames remain authoritative overrides.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import write_json
from .schema import MaskPipelineConfig


AUTO = "auto"


def is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO


def discovery_timestamps(
    timestamps: Iterable[str],
    *,
    stride: int,
) -> list[str]:
    values = list(timestamps)
    if not values:
        raise ValueError("cannot plan mask discovery without frames")
    if stride <= 0:
        raise ValueError("discovery stride must be positive")
    selected = values[::stride]
    if values[-1] not in selected:
        selected.append(values[-1])
    return selected


def _boxes_for(
    bbox_data: dict[str, Any],
    timestamp: str,
    view: str,
    part: str,
) -> list[list[float]]:
    rows = (
        bbox_data.get("frames", {})
        .get(timestamp, {})
        .get(view, {})
        .get("parts", [])
    )
    return [
        [float(value) for value in row["bbox_2d"]]
        for row in rows
        if row.get("label") == part and len(row.get("bbox_2d", [])) == 4
    ]


def _box_area(box: list[float]) -> float:
    return (
        max(0.0, box[2] - box[0])
        * max(0.0, box[3] - box[1])
        / 1_000_000.0
    )


def detection_evidence(
    bbox_data: dict[str, Any],
    timestamps: Iterable[str],
    views: Iterable[str],
    part: str,
) -> list[dict[str, Any]]:
    result = []
    ordered_views = list(views)
    for timestamp in timestamps:
        per_view = {
            view: _boxes_for(bbox_data, timestamp, view, part)
            for view in ordered_views
        }
        areas = [
            max(_box_area(box) for box in boxes)
            for boxes in per_view.values()
            if boxes
        ]
        result.append({
            "timestamp": str(timestamp),
            "detecting_views": sum(bool(value) for value in per_view.values()),
            "median_box_area_ratio": (
                float(np.median(areas)) if areas else 0.0
            ),
            "views": {
                view: {
                    "detected": bool(boxes),
                    "largest_box_area_ratio": (
                        max((_box_area(box) for box in boxes), default=0.0)
                    ),
                }
                for view, boxes in per_view.items()
            },
        })
    return result


def infer_presence_start(
    evidence: list[dict[str, Any]],
    *,
    minimum_views: int,
    consecutive: int = 1,
) -> tuple[int, dict[str, Any]]:
    if minimum_views <= 0:
        raise ValueError("minimum_views must be positive")
    if consecutive <= 0:
        raise ValueError("consecutive must be positive")
    present = [
        int(row["detecting_views"]) >= minimum_views
        for row in evidence
    ]
    selected = None
    for index in range(0, len(present) - consecutive + 1):
        if all(present[index:index + consecutive]):
            selected = index
            break
    if selected is None:
        raise RuntimeError(
            "Qwen did not detect the part with sufficient multi-view support"
        )
    frame = int(evidence[selected]["timestamp"])
    previous_negative = next(
        (
            int(evidence[index]["timestamp"])
            for index in range(selected - 1, -1, -1)
            if not present[index]
        ),
        None,
    )
    return frame, {
        "selected_scan_frame": frame,
        "previous_negative_scan_frame": previous_negative,
        "minimum_views": int(minimum_views),
        "consecutive_scans": int(consecutive),
    }


def choose_seed_frames(
    evidence: list[dict[str, Any]],
    views: Iterable[str],
    *,
    start_frame: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Choose a visible, large, multi-view-supported seed for every camera."""

    ordered_views = list(views)
    eligible = [
        row for row in evidence if int(row["timestamp"]) >= int(start_frame)
    ]
    selected: dict[str, int] = {}
    details: dict[str, Any] = {}
    for view in ordered_views:
        candidates = [
            row for row in eligible
            if row["views"][view]["detected"]
        ]
        if not candidates:
            raise RuntimeError(
                f"Qwen found no usable seed for view {view!r}"
            )
        best = max(
            candidates,
            key=lambda row: (
                int(row["detecting_views"]),
                float(row["views"][view]["largest_box_area_ratio"]),
                float(row["median_box_area_ratio"]),
                -int(row["timestamp"]),
            ),
        )
        selected[view] = int(best["timestamp"])
        details[view] = {
            "frame": int(best["timestamp"]),
            "detecting_views": int(best["detecting_views"]),
            "box_area_ratio": float(
                best["views"][view]["largest_box_area_ratio"]
            ),
        }
    return selected, details


def needs_discovery(config: MaskPipelineConfig) -> bool:
    for part in config.parts:
        if part.start_frame_auto:
            return True
        configured = config.raw.get("parts", {}).get(part.name, {})
        tracking = configured.get("tracking", {})
        seeds = tracking.get("seed_frames", tracking.get("seed_frame"))
        if is_auto(seeds):
            return True
        if isinstance(seeds, dict) and any(
            is_auto(value) for value in seeds.values()
        ):
            return True
    return False


def resolve_mask_config(
    config: MaskPipelineConfig,
    bbox_data: dict[str, Any],
    scan_timestamps: Iterable[str],
    *,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = config.raw.get("automation", {}).get("discovery", {})
    minimum_views = int(settings.get(
        "minimum_views",
        max(1, min(2, len(config.views))),
    ))
    consecutive = int(settings.get("consecutive_scans", 1))
    raw = deepcopy(config.raw)
    raw_parts = raw.setdefault("parts", {})
    if not isinstance(raw_parts, dict):
        raise ValueError("automatic mask planning requires dictionary parts")
    report: dict[str, Any] = {
        "source_config": str(config.source_path),
        "resolved_config": str(output_path),
        "scan_timestamps": list(scan_timestamps),
        "parts": {},
    }
    for part in config.parts:
        evidence = detection_evidence(
            bbox_data,
            scan_timestamps,
            config.views,
            part.name,
        )
        values = raw_parts[part.name]
        if part.start_frame_auto:
            start, start_report = infer_presence_start(
                evidence,
                minimum_views=minimum_views,
                consecutive=consecutive,
            )
            values["start_frame"] = start
            start_source = "qwen_multiview_discovery"
        else:
            start = int(part.start_frame)
            start_report = {}
            start_source = "explicit_override"

        tracking = values.setdefault("tracking", {})
        configured_seeds = tracking.get(
            "seed_frames", tracking.get("seed_frame")
        )
        auto_seeds = is_auto(configured_seeds) or (
            isinstance(configured_seeds, dict)
            and any(is_auto(value) for value in configured_seeds.values())
        )
        seed_report: dict[str, Any] = {"source": "explicit_override"}
        if auto_seeds:
            seeds, seed_details = choose_seed_frames(
                evidence,
                config.views,
                start_frame=start,
            )
            tracking.pop("seed_frame", None)
            tracking["seed_frames"] = seeds
            seed_report = {
                "source": "qwen_visibility_and_box_score",
                "per_view": seed_details,
            }
        report["parts"][part.name] = {
            "start_frame": start,
            "start_source": start_source,
            "start_evidence": start_report,
            "seed_selection": seed_report,
            "detections": evidence,
        }

    write_json(output_path, raw)
    write_json(output_path.with_name("mask_discovery_report.json"), report)
    return raw, report


def repair_jobs_from_quality(
    quality: dict[str, Any],
    config: MaskPipelineConfig,
    timestamps: Iterable[str],
) -> list[dict[str, Any]]:
    """Turn QA anomaly runs into bounded, object-agnostic re-anchor jobs."""

    settings = config.raw.get("automation", {}).get("repair", {})
    padding = int(settings.get("padding_frames", 2))
    maximum = int(settings.get("maximum_jobs_per_part", 3))
    first, last = min(map(int, timestamps)), max(map(int, timestamps))
    jobs = []
    for view in config.views:
        for part in config.parts:
            part_values = config.raw.get("parts", {}).get(part.name, {})
            tracking = (
                part_values.get("tracking", part_values.get("tracker", {}))
                if isinstance(part_values, dict)
                else {}
            )
            configured_mode = (
                tracking
                if isinstance(tracking, str)
                else tracking.get("mode", "video")
            )
            repair_mode = (
                configured_mode
                if configured_mode in {"video", "fixed-image"}
                else "fixed-image"
            )
            runs = (
                quality.get(view, {})
                .get(part.name, {})
                .get("anomaly_runs", [])
            )
            selected = runs[:maximum]
            for start, end in selected:
                start = max(first, part.start_frame, int(start) - padding)
                end = min(last, int(end) + padding)
                jobs.append({
                    "part": part.name,
                    "mode": repair_mode,
                    "views": [view],
                    "range": [start, end],
                    "seed_frame": f"{(start + end) // 2:06d}",
                    "hold_previous": False,
                    "repair": True,
                })
    merged: list[dict[str, Any]] = []
    for job in sorted(
        jobs,
        key=lambda row: (
            row["part"], row["views"][0], row["range"][0], row["range"][1]
        ),
    ):
        if (
            merged
            and merged[-1]["part"] == job["part"]
            and merged[-1]["views"] == job["views"]
            and job["range"][0] <= merged[-1]["range"][1] + 1
        ):
            merged[-1]["range"][1] = max(
                merged[-1]["range"][1], job["range"][1]
            )
            start, end = merged[-1]["range"]
            merged[-1]["seed_frame"] = f"{(start + end) // 2:06d}"
        else:
            merged.append(job)
    return merged
