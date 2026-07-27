"""Dependency-light mask quality checks and re-anchor suggestions."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    import cv2

    source = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, 8)
    if count <= 1:
        return source.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == keep


def clean_mask(mask: np.ndarray, close_kernel: int = 5, fill_holes: bool = True) -> np.ndarray:
    import cv2

    source = largest_component(mask)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        source = cv2.morphologyEx(
            source.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    if not fill_holes:
        return source
    padded = np.pad(source.astype(np.uint8), 1)
    flooded = padded.copy()
    workspace = np.zeros((flooded.shape[0] + 2, flooded.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, workspace, (0, 0), 1)
    return (padded.astype(bool) | (flooded == 0))[1:-1, 1:-1]


def mask_metrics(mask: np.ndarray) -> dict:
    selected = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(selected)
    if not len(xs):
        return {
            "pixels": 0,
            "area_ratio": 0.0,
            "centroid_xy": None,
            "bbox_xyxy": None,
        }
    height, width = selected.shape
    return {
        "pixels": int(len(xs)),
        "area_ratio": float(len(xs) / (height * width)),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
    }


def _runs(values: Sequence[int]) -> list[list[int]]:
    if not values:
        return []
    groups = [[int(values[0])]]
    for value in values[1:]:
        if int(value) == groups[-1][-1] + 1:
            groups[-1].append(int(value))
        else:
            groups.append([int(value)])
    return groups


def summarize_area_series(
    timestamps: Sequence[str],
    areas: Sequence[int],
    *,
    start_frame: int,
    min_area_ratio: float = 0.2,
    max_area_ratio: float = 4.0,
) -> dict:
    if len(timestamps) != len(areas):
        raise ValueError("timestamps and areas must have the same length")
    active = [
        (int(timestamp), int(area))
        for timestamp, area in zip(timestamps, areas)
        if int(timestamp) >= start_frame
    ]
    positive = np.asarray([area for _, area in active if area > 0], dtype=np.float64)
    median = float(np.median(positive)) if len(positive) else 0.0
    empty = [frame for frame, area in active if area == 0]
    anomalous = []
    if median > 0:
        anomalous = [
            frame for frame, area in active
            if area > 0 and (area < median * min_area_ratio or area > median * max_area_ratio)
        ]
    suggestions = sorted(set(empty + anomalous))
    return {
        "start_frame": int(start_frame),
        "active_frames": len(active),
        "nonempty_frames": int(sum(area > 0 for _, area in active)),
        "median_positive_pixels": int(round(median)),
        "empty_runs": [[run[0], run[-1]] for run in _runs(empty)],
        "area_anomaly_frames": anomalous,
        "suggested_reanchor_frames": [run[len(run) // 2] for run in _runs(suggestions)],
    }


def mask_iou(first: np.ndarray | None, second: np.ndarray) -> float | None:
    if first is None:
        return None
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def summarize_track_series(
    timestamps: Sequence[str],
    areas: Sequence[int],
    centroids: Sequence[Sequence[float] | None],
    temporal_ious: Sequence[float | None],
    *,
    start_frame: int,
    image_diagonal: float,
    min_area_ratio: float = 0.2,
    max_area_ratio: float = 4.0,
    max_centroid_step_ratio: float = 0.08,
    min_temporal_iou: float = 0.05,
) -> dict:
    """Summarize independent-track failures without assuming object class.

    Area catches empty/exploded tracks.  A low temporal IoU is only considered
    suspicious together with a large centroid jump, so legitimate deformation
    or rapid in-place rotation is not automatically treated as a failure.
    """

    report = summarize_area_series(
        timestamps,
        areas,
        start_frame=start_frame,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    jumps: list[int] = []
    low_iou: list[int] = []
    previous_centroid: np.ndarray | None = None
    for timestamp, area, centroid, iou in zip(
        timestamps, areas, centroids, temporal_ious
    ):
        frame = int(timestamp)
        if frame < start_frame:
            continue
        current = (
            None
            if centroid is None
            else np.asarray(centroid, dtype=np.float64)
        )
        jump = (
            0.0
            if current is None or previous_centroid is None
            else float(np.linalg.norm(current - previous_centroid))
            / max(float(image_diagonal), 1.0)
        )
        if (
            int(area) > 0
            and iou is not None
            and float(iou) < min_temporal_iou
            and jump > max_centroid_step_ratio
        ):
            low_iou.append(frame)
        if jump > max_centroid_step_ratio:
            jumps.append(frame)
        if current is not None:
            previous_centroid = current
    empty_frames = [
        frame
        for start, end in report["empty_runs"]
        for frame in range(int(start), int(end) + 1)
    ]
    anomaly_frames = sorted(set(
        empty_frames + report["area_anomaly_frames"] + jumps + low_iou
    ))
    anomaly_runs = [
        [run[0], run[-1]]
        for run in _runs(anomaly_frames)
    ]
    report.update({
        "centroid_jump_frames": jumps,
        "low_temporal_iou_frames": low_iou,
        "anomaly_runs": anomaly_runs,
        "suggested_reanchor_frames": [
            run[len(run) // 2] for run in _runs(anomaly_frames)
        ],
    })
    return report
