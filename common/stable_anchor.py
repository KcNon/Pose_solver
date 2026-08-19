"""Reusable scoring and selection for stable multi-view pose anchors.

The state detector identifies intervals, not the best observation inside an
interval.  These helpers rank every usable frame from state, mask, and cloud
evidence, then retain time-separated candidates for the expensive mesh fit.
They deliberately do not inspect a mesh: render/depth losses are a second
stage and must not be mixed into the cheap shortlist score.
"""
from __future__ import annotations

import math
from typing import Any


def _finite(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def stable_frame_evidence(
    state: dict[str, Any],
    cloud: dict[str, Any] | None,
    *,
    maximum_views: int,
) -> dict[str, Any]:
    """Return normalized, auditable evidence for one potential anchor."""

    cloud = dict(cloud or {})
    gate = dict(cloud.get("quality_gate", {}))
    cross = dict(cloud.get("cross_view", {}))
    reprojection = dict(cloud.get("reprojection_depth", {}))
    mask = dict(cloud.get("mask_quality", {}))
    view_rows = list(mask.get("views", {}).values())
    cloud_view_rows = list(cloud.get("views", []))
    area_ratios = [
        _finite(row.get("area_to_median"), 0.0)
        for row in view_rows
        if _finite(row.get("area_to_median"), 0.0) > 0.0
    ]
    area_log_spread = (
        sum(abs(math.log(value)) for value in area_ratios) / len(area_ratios)
        if area_ratios
        else float("inf")
    )
    state_name = str(state.get("state", "unobserved"))
    observing_views = int(state.get("observing_views", 0) or 0)
    valid_views = int(mask.get("valid_view_count", observing_views) or 0)
    motion_px = _finite(state.get("motion_px"), float("inf"))
    surface_shift_mm = _finite(
        state.get("surface_shift_mm"), float("inf")
    )
    # Schema-v1 quality summaries predate the explicit ``quality_gate``
    # block.  Their ``status == ok`` rows were already filtered before the
    # fused PLY was written, so retain them as auditable legacy evidence
    # instead of making every existing dataset ineligible for anchor
    # selection.  New summaries must continue to pass their explicit gate.
    explicit_gate = "passed" in gate
    cloud_ok = bool(
        cloud
        and cloud.get("status") == "ok"
        and (bool(gate.get("passed")) if explicit_gate else True)
    )
    static = state_name in {"static", "assembled"}

    visibility_score = min(observing_views, valid_views) / max(
        int(maximum_views), 1
    )
    motion_score = math.exp(-max(motion_px, 0.0) / 1.5)
    surface_score = math.exp(-max(surface_shift_mm, 0.0) / 15.0)
    candidate_points = sum(
        max(_finite(row.get("candidate_points"), 0.0), 0.0)
        for row in cloud_view_rows
    )
    supported_points = sum(
        max(_finite(row.get("supported_points"), 0.0), 0.0)
        for row in cloud_view_rows
    )
    legacy_support_fraction = (
        supported_points / candidate_points if candidate_points > 0.0 else 0.0
    )
    support_score = min(
        max(
            _finite(
                gate.get("support_fraction"), legacy_support_fraction
            ),
            0.0,
        ),
        1.0,
    )
    overlap_score = min(max(_finite(cross.get("overlap_ratio"), 0.0), 0.0), 1.0)
    reprojection_inlier_score = min(
        max(_finite(reprojection.get("inlier_ratio"), 0.0), 0.0), 1.0
    )
    cross_residual_score = math.exp(
        -max(_finite(cross.get("median_m"), 1.0), 0.0) / 0.03
    )
    reprojection_residual_score = math.exp(
        -max(_finite(reprojection.get("median_m"), 1.0), 0.0) / 0.03
    )
    mask_balance_score = (
        math.exp(-area_log_spread) if math.isfinite(area_log_spread) else 0.0
    )

    score = (
        0.20 * visibility_score
        + 0.14 * motion_score
        + 0.14 * surface_score
        + 0.10 * support_score
        + 0.10 * overlap_score
        + 0.08 * reprojection_inlier_score
        + 0.08 * cross_residual_score
        + 0.08 * reprojection_residual_score
        + 0.08 * mask_balance_score
    )
    usable = bool(static and cloud_ok and min(observing_views, valid_views) >= 3)
    return {
        "usable": usable,
        "state": state_name,
        "observing_views": observing_views,
        "valid_mask_views": valid_views,
        "motion_px": None if not math.isfinite(motion_px) else motion_px,
        "surface_shift_mm": (
            None if not math.isfinite(surface_shift_mm) else surface_shift_mm
        ),
        "cloud_quality_passed": cloud_ok,
        "cloud_quality_source": (
            "explicit_gate" if explicit_gate else "legacy_status"
        ),
        "support_fraction": support_score,
        "cross_view_overlap_ratio": overlap_score,
        "cross_view_median_m": _finite(cross.get("median_m"), float("nan")),
        "reprojection_inlier_ratio": reprojection_inlier_score,
        "reprojection_median_m": _finite(
            reprojection.get("median_m"), float("nan")
        ),
        "mask_area_log_spread": (
            area_log_spread if math.isfinite(area_log_spread) else None
        ),
        "shortlist_score": float(score),
    }


def rank_stable_frames(
    states: dict[int, dict[str, Any]],
    clouds: dict[int, dict[str, Any]],
    *,
    start: int,
    end: int,
    maximum_views: int,
    temporal_radius: int = 2,
    settling_frames: int = 5,
) -> list[dict[str, Any]]:
    """Rank frames while requiring a stable neighbourhood around each one."""

    result = []
    for frame in range(int(start), int(end) + 1):
        evidence = stable_frame_evidence(
            states.get(frame, {}),
            clouds.get(frame),
            maximum_views=maximum_views,
        )
        neighbours = [
            states.get(value, {})
            for value in range(
                max(int(start), frame - int(temporal_radius)),
                min(int(end), frame + int(temporal_radius)) + 1,
            )
        ]
        stable_fraction = sum(
            row.get("state") in {"static", "assembled"} for row in neighbours
        ) / max(len(neighbours), 1)
        settled = frame - int(start) >= max(0, int(settling_frames))
        settling_score = min(
            max((frame - int(start)) / max(float(settling_frames), 1.0), 0.0),
            1.0,
        )
        evidence["stable_neighbour_fraction"] = float(stable_fraction)
        evidence["settled_after_interval_start"] = bool(settled)
        evidence["frames_after_interval_start"] = frame - int(start)
        evidence["shortlist_score"] = float(
            0.82 * evidence["shortlist_score"]
            + 0.10 * stable_fraction
            + 0.08 * settling_score
        )
        evidence["usable"] = bool(
            evidence["usable"] and stable_fraction >= 0.8 and settled
        )
        evidence["frame"] = frame
        result.append(evidence)
    return sorted(
        result,
        key=lambda row: (
            bool(row["usable"]), float(row["shortlist_score"]), -row["frame"]
        ),
        reverse=True,
    )


def select_separated_candidates(
    ranking: list[dict[str, Any]],
    *,
    count: int,
    minimum_separation: int,
) -> list[int]:
    """Greedily retain the strongest usable time-separated frames."""

    selected: list[int] = []
    for row in ranking:
        if not row.get("usable", False):
            continue
        frame = int(row["frame"])
        if any(abs(frame - previous) < int(minimum_separation) for previous in selected):
            continue
        selected.append(frame)
        if len(selected) >= int(count):
            break
    if not selected:
        raise RuntimeError("stable interval has no usable multi-view anchor")
    return sorted(selected)


def centered_stable_window(
    states: dict[int, dict[str, Any]],
    anchor: int,
    *,
    start: int,
    end: int,
    size: int,
    minimum_views: int = 3,
) -> list[int]:
    """Build a symmetric fusion window without crossing unstable frames."""

    candidates = sorted(
        range(max(start, anchor - size), min(end, anchor + size) + 1),
        key=lambda frame: (abs(frame - anchor), frame),
    )
    selected = []
    for frame in candidates:
        row = states.get(frame, {})
        if row.get("state") not in {"static", "assembled"}:
            continue
        if int(row.get("observing_views", 0) or 0) < minimum_views:
            continue
        selected.append(frame)
        if len(selected) >= size:
            break
    if anchor not in selected:
        selected.append(anchor)
    return sorted(set(selected))


def stable_runs(
    states: dict[int, dict[str, Any]],
    *,
    start: int,
    end: int,
    minimum_views: int,
) -> list[tuple[int, int]]:
    """Return contiguous, sufficiently observed static/assembled intervals."""

    result: list[tuple[int, int]] = []
    for frame in range(int(start), int(end) + 1):
        row = states.get(frame, {})
        usable = bool(
            row.get("state") in {"static", "assembled"}
            and int(row.get("observing_views", 0) or 0) >= int(minimum_views)
        )
        if not usable:
            continue
        if result and result[-1][1] == frame - 1:
            result[-1] = (result[-1][0], frame)
        else:
            result.append((frame, frame))
    return result


def first_settled_window(
    states: dict[int, dict[str, Any]],
    clouds: dict[int, dict[str, Any]],
    *,
    start: int,
    end: int,
    maximum_views: int,
    minimum_views: int,
    size: int,
    settling_frames: int = 5,
    require_cloud_quality: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    """Select the first trusted window after a part has visibly settled.

    The temporal event is more important than a globally best late frame: the
    first released part defines the reusable world reference.  Within the first
    qualifying run we still choose the strongest consecutive window, so a
    boundary frame or a briefly occluded view cannot become the representative.
    """

    size = max(2, int(size))
    settling_frames = max(0, int(settling_frames))
    has_cloud_evidence = any(bool(value) for value in clouds.values())
    candidates: list[tuple[float, int, list[int], list[dict[str, Any]]]] = []
    rejected_runs: list[dict[str, Any]] = []
    for run_start, run_end in stable_runs(
        states, start=start, end=end, minimum_views=minimum_views
    ):
        # Preserve a useful window in very short clips/tests while using the
        # full release delay whenever the observed stable run permits it.
        available_delay = max(0, run_end - run_start + 1 - size)
        effective_settling = min(settling_frames, available_delay)
        settled_start = run_start + effective_settling
        if settled_start + size - 1 > run_end:
            rejected_runs.append({
                "range": [run_start, run_end],
                "reason": "shorter_than_settling_plus_window",
            })
            continue
        per_run = []
        for first in range(settled_start, run_end - size + 2):
            window = list(range(first, first + size))
            evidence = [
                stable_frame_evidence(
                    states.get(frame, {}),
                    clouds.get(frame),
                    maximum_views=maximum_views,
                )
                for frame in window
            ]
            usable = all(
                row["usable"]
                if require_cloud_quality and has_cloud_evidence
                else (
                    row["state"] in {"static", "assembled"}
                    and row["observing_views"] >= minimum_views
                )
                for row in evidence
            )
            if not usable:
                continue
            score = float(sum(row["shortlist_score"] for row in evidence) / size)
            per_run.append((score, first, window, evidence))
        if per_run:
            # Stop at the first physical stable run. Quality only chooses within
            # that event; it cannot silently move the reference to the sequence end.
            earliest = min(value[1] for value in per_run)
            candidates = [
                value for value in per_run
                if value[1] <= earliest + size - 1
            ]
            break
        rejected_runs.append({
            "range": [run_start, run_end],
            "reason": "no_window_passed_multiview_cloud_quality",
        })
    if not candidates:
        raise RuntimeError(
            "no trusted settled multi-view window; "
            f"rejected_runs={rejected_runs}"
        )
    score, first, window, evidence = max(
        candidates, key=lambda value: (value[0], -value[1])
    )
    representative = max(
        zip(window, evidence),
        key=lambda value: (value[1]["shortlist_score"], -value[0]),
    )[0]
    return window, {
        "policy": "first_settled_static_run_multiview_consensus",
        "window": window,
        "representative_frame": int(representative),
        "score": score,
        "settling_frames": settling_frames,
        "require_cloud_quality": bool(require_cloud_quality and has_cloud_evidence),
        "evidence": {
            str(frame): row for frame, row in zip(window, evidence)
        },
        "rejected_earlier_runs": rejected_runs,
    }
