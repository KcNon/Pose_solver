"""Frozen-pose scale diagnostics with visual and geometric objectives.

This module deliberately never updates a trajectory.  A candidate uniformly
scales canonical part geometry about the existing part-frame origin while the
observed SE(3) pose remains fixed.  That separation is important: scale
diagnosis must not silently turn a physics projection into pose supervision.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def pareto_indices(
    rows: list[dict[str, Any]],
    objectives: tuple[str, ...],
    *,
    epsilon: float = 1e-12,
) -> list[int]:
    """Return non-dominated row indices for objectives that are minimized."""
    values = np.asarray(
        [[float(row[key]) for key in objectives] for row in rows],
        dtype=np.float64,
    )
    if values.ndim != 2 or not len(values):
        return []
    finite = np.all(np.isfinite(values), axis=1)
    result: list[int] = []
    for index, value in enumerate(values):
        if not finite[index]:
            continue
        dominated = False
        for other_index, other in enumerate(values):
            if index == other_index or not finite[other_index]:
                continue
            no_worse = np.all(other <= value + float(epsilon))
            strictly_better = np.any(other < value - float(epsilon))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(index)
    return result


def select_visual_gated_candidate(
    rows: list[dict[str, Any]],
    *,
    baseline_factor: float = 1.0,
    maximum_visual_loss_degradation: float,
    penetration_key: str = "max_penetration_m",
    visual_key: str = "visual_loss",
) -> tuple[int, dict[str, Any]]:
    """Choose least penetration within a visual-loss gate.

    Ties are resolved by visual loss and then distance to the original scale.
    The returned report makes the acceptance boundary auditable.
    """
    if not rows:
        raise ValueError("cannot select from an empty candidate list")
    baseline_index = min(
        range(len(rows)),
        key=lambda index: abs(
            float(rows[index]["scale_factor"]) - float(baseline_factor)
        ),
    )
    baseline_loss = float(rows[baseline_index][visual_key])
    limit = baseline_loss + float(maximum_visual_loss_degradation)
    eligible = [
        index
        for index, row in enumerate(rows)
        if np.isfinite(float(row[visual_key]))
        and float(row[visual_key]) <= limit + 1e-12
    ]
    if not eligible:
        eligible = [baseline_index]
    selected = min(
        eligible,
        key=lambda index: (
            float(rows[index][penetration_key]),
            float(rows[index][visual_key]),
            abs(float(rows[index]["scale_factor"]) - float(baseline_factor)),
        ),
    )
    return selected, {
        "baseline_index": baseline_index,
        "baseline_scale_factor": float(rows[baseline_index]["scale_factor"]),
        "baseline_visual_loss": baseline_loss,
        "maximum_visual_loss_degradation": float(
            maximum_visual_loss_degradation
        ),
        "visual_loss_limit": limit,
        "eligible_indices": eligible,
        "selected_index": selected,
        "selected_scale_factor": float(rows[selected]["scale_factor"]),
        "trajectory_mutated": False,
    }


def scaled_surface(surface: Any, factor: float) -> Any:
    """Return a scaled ``SampledSurface`` without importing it at module load."""
    from common.trajectory_constraints import SampledSurface

    value = float(factor)
    if value <= 0.0:
        raise ValueError("scale factor must be positive")
    return SampledSurface(
        np.asarray(surface.points, dtype=np.float64) * value,
        np.asarray(surface.normals, dtype=np.float64),
    )


def aggregate_visual_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame render metrics without hiding missing observations."""
    valid = [row for row in rows if np.isfinite(float(row.get("loss", np.inf)))]
    if not valid:
        return {
            "visual_loss": float("inf"),
            "mean_iou": 0.0,
            "mean_contour_chamfer_px": None,
            "evaluated_frames": 0,
            "evaluated_views": 0,
        }
    return {
        "visual_loss": float(np.mean([row["loss"] for row in valid])),
        "mean_iou": float(np.mean([row["mean_iou"] for row in valid])),
        "mean_contour_chamfer_px": float(
            np.mean([row["mean_contour_chamfer_px"] for row in valid])
        ),
        "evaluated_frames": len(valid),
        "evaluated_views": int(
            sum(len(row.get("views", [])) for row in valid)
        ),
    }
