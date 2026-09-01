"""Resolve per-sequence pose priors from mask/cloud state diagnostics.

The resolver is deliberately separate from the solver.  It produces an
ordinary validated pose JSON plus a provenance report; the solver never
silently changes state ranges or anchors while running.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable

from common.stable_anchor import first_settled_window


def _merge_ranges(values: Iterable[Iterable[int]]) -> list[list[int]]:
    ordered = sorted(
        ([int(pair[0]), int(pair[1])] for pair in values),
        key=lambda pair: (pair[0], pair[1]),
    )
    result: list[list[int]] = []
    for start, end in ordered:
        if result and start <= result[-1][1] + 1:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _merge_nearby_ranges(
    values: Iterable[Iterable[int]], maximum_gap: int
) -> list[list[int]]:
    result = _merge_ranges(values)
    if maximum_gap <= 0:
        return result
    merged: list[list[int]] = []
    for start, end in result:
        if merged and start - merged[-1][1] - 1 <= maximum_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def _complement(
    ranges: list[list[int]],
    start: int,
    end: int,
) -> list[list[int]]:
    result = []
    cursor = start
    for range_start, range_end in _merge_ranges(ranges):
        if cursor < range_start:
            result.append([cursor, range_start - 1])
        cursor = max(cursor, range_end + 1)
    if cursor <= end:
        result.append([cursor, end])
    return result


def _state_rows(report: dict[str, Any], part: str) -> dict[int, dict[str, Any]]:
    return {
        int(timestamp): dict(values)
        for timestamp, values in (
            report.get("parts", {})
            .get(part, {})
            .get("states", {})
            .items()
        )
    }


def _frame_score(row: dict[str, Any]) -> float:
    visible = float(row.get("observing_views", 0))
    shift = row.get("surface_shift_mm")
    motion = row.get("motion_px")
    score = (
        visible
        - 0.05 * (0.0 if shift is None else float(shift))
        - 0.01 * (0.0 if motion is None else float(motion))
    )
    quality = row.get("cloud_quality")
    if quality is not None:
        if quality.get("status") != "ok":
            return score - 1000.0
        gate = quality.get("quality_gate", {})
        cross = quality.get("cross_view", {})
        reprojection = quality.get("reprojection_depth", {})
        score += 4.0
        score += 2.0 * float(gate.get("support_fraction", 0.0))
        score += 2.0 * float(cross.get("overlap_ratio") or 0.0)
        score += 2.0 * float(reprojection.get("inlier_ratio") or 0.0)
        score -= 20.0 * float(cross.get("median_m") or 0.0)
        score -= 20.0 * float(reprojection.get("median_m") or 0.0)
    return score


def _appearance_evidence_frames(
    window: Iterable[int],
    rows: dict[int, dict[str, Any]],
    *,
    count: int,
    minimum_separation: int,
    require_cloud_quality: bool = True,
) -> list[int]:
    """Choose several high-quality, temporally distinct stable observations."""

    candidates = [
        int(frame)
        for frame in window
        if int(frame) in rows
        and _anchor_frame_usable(
            rows[int(frame)],
            require_cloud_quality=require_cloud_quality,
        )
    ]
    ranked = sorted(candidates, key=lambda frame: _frame_score(rows[frame]), reverse=True)
    chosen: list[int] = []
    for frame in ranked:
        if all(abs(frame - other) >= minimum_separation for other in chosen):
            chosen.append(frame)
            if len(chosen) >= count:
                break
    if len(chosen) < min(count, len(ranked)):
        for frame in ranked:
            if frame not in chosen:
                chosen.append(frame)
                if len(chosen) >= count:
                    break
    return sorted(chosen)


def _cloud_quality_rows(
    config: dict[str, Any], part: str
) -> dict[int, dict[str, Any]]:
    root = config.get("point_cloud_root")
    if not root:
        return {}
    path = Path(root) / "quality_cloud_summary.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for timestamp, frame in report.get("frames", {}).items():
        if part in frame:
            result[int(timestamp)] = dict(frame[part])
    return result


def _anchor_frame_usable(
    row: dict[str, Any],
    *,
    require_cloud_quality: bool = True,
) -> bool:
    quality = row.get("cloud_quality")
    return (
        not require_cloud_quality
        or quality is None
        or quality.get("status") == "ok"
    )


def choose_reference_part(
    config: dict[str, Any],
    state_report: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    start_frames = config.get("part_start_frames", {})
    # A reference frame must be stationary.  Raw observation count alone can
    # otherwise select a large moving component simply because it appears
    # earlier or occupies more views, after which the resolver incorrectly
    # forces that motion to be static.  Weight motion by the number of views
    # that actually observed it and make it dominate modest coverage gains.
    motion_penalty = float(
        config.get("automation", {}).get(
            "reference_motion_view_penalty", 4.0
        )
    )
    scores = {}
    for part in config["parts"]:
        rows = _state_rows(state_report, part)
        visible = sum(int(row.get("observing_views", 0)) for row in rows.values())
        moving_evidence = sum(
            int(row.get("observing_views", 0))
            for row in rows.values()
            if row.get("state") == "moving"
        )
        start = int(start_frames.get(part, config["frames"]["start"]))
        scores[part] = float(
            visible - motion_penalty * moving_evidence - 0.01 * start
        )
    if not scores:
        raise ValueError("cannot choose a reference part without parts")
    return max(scores, key=scores.get), scores


def choose_reference_part_from_settled_windows(
    config: dict[str, Any],
    state_report: dict[str, Any],
    *,
    minimum_views: int,
    window_size: int,
    settling_frames: int,
    require_cloud_quality: bool,
) -> tuple[str, dict[str, Any]]:
    """Choose the physical part whose first trusted stable window occurs first."""

    start = int(config["frames"]["start"])
    end = int(config["frames"]["end"])
    starts = config.get("part_start_frames", {})
    candidates: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for part in config["parts"]:
        rows = _state_rows(state_report, part)
        clouds = _cloud_quality_rows(config, part)
        try:
            window, report = first_settled_window(
                rows,
                clouds,
                start=max(start, int(starts.get(part, start))),
                end=end,
                maximum_views=len(config["views"]),
                minimum_views=minimum_views,
                size=window_size,
                settling_frames=settling_frames,
                require_cloud_quality=require_cloud_quality,
            )
        except RuntimeError as error:
            failures[part] = str(error)
            continue
        candidates[part] = {"window": window, **report}
    if not candidates:
        raise RuntimeError(
            "no part has a trusted stable release window; "
            f"failures={failures}"
        )
    reference = min(
        candidates,
        key=lambda part: (
            candidates[part]["window"][0],
            int(starts.get(part, start)),
            -float(candidates[part]["score"]),
            config["parts"].index(part),
        ),
    )
    return reference, {
        "policy": "first_part_with_trusted_settled_window",
        "candidates": candidates,
        "failures": failures,
        "selected": reference,
    }


def _stable_frames(
    rows: dict[int, dict[str, Any]],
    *,
    start: int,
    end: int,
    minimum_views: int,
    require_cloud_quality: bool = True,
    allow_occluded: bool = False,
) -> list[int]:
    stable_states = {"static", "assembled"}
    if allow_occluded:
        # The state detector assigns ``occluded`` only after its kinematic
        # hysteresis classified the frame as static.  Such a frame is valid as
        # a render-validated fallback, never as an ordinary complete anchor.
        stable_states.add("occluded")
    return [
        frame
        for frame in range(start, end + 1)
        if frame in rows
        and rows[frame].get("state") in stable_states
        and int(rows[frame].get("observing_views", 0)) >= minimum_views
        and _anchor_frame_usable(
            rows[frame],
            require_cloud_quality=require_cloud_quality,
        )
    ]


def choose_calibration_window(
    rows: dict[int, dict[str, Any]],
    *,
    start: int,
    end: int,
    minimum_views: int,
    window_size: int,
    require_cloud_quality: bool = True,
    allow_occluded: bool = False,
) -> list[int]:
    stable = set(_stable_frames(
        rows,
        start=start,
        end=end,
        minimum_views=minimum_views,
        require_cloud_quality=require_cloud_quality,
        allow_occluded=allow_occluded,
    ))
    candidates: list[tuple[float, list[int]]] = []
    for first in range(start, end + 1):
        window = list(range(first, min(end + 1, first + window_size)))
        if len(window) < min(2, window_size) or not all(
            frame in stable for frame in window
        ):
            continue
        candidates.append((
            sum(_frame_score(rows[frame]) for frame in window) / len(window),
            window,
        ))
    if candidates:
        best_score = max(value[0] for value in candidates)
        # Surface-shift noise should not force calibration to an arbitrary end
        # of a long static sequence.  Among statistically similar windows,
        # prefer the one nearest the temporal centre, which maximizes context
        # on both sides and avoids sequence-boundary transients.
        comparable = [
            value for value in candidates
            if value[0] >= best_score - 0.25
        ]
        sequence_middle = 0.5 * (start + end)
        return min(
            comparable,
            key=lambda value: abs(
                0.5 * (value[1][0] + value[1][-1]) - sequence_middle
            ),
        )[1]
    available = sorted(stable, key=lambda frame: _frame_score(rows[frame]), reverse=True)
    if available:
        return sorted(available[:window_size])
    raise RuntimeError("no stable, sufficiently visible calibration frame")


def choose_calibration_windows(
    rows: dict[int, dict[str, Any]],
    *,
    start: int,
    end: int,
    minimum_views: int,
    window_size: int,
    maximum_windows: int,
    minimum_separation: int,
    require_cloud_quality: bool = True,
    allow_occluded: bool = False,
) -> list[list[int]]:
    """Return several high-quality, time-separated calibration windows."""

    remaining = dict(rows)
    result: list[list[int]] = []
    for _ in range(max(1, int(maximum_windows))):
        try:
            window = choose_calibration_window(
                remaining,
                start=start,
                end=end,
                minimum_views=minimum_views,
                window_size=window_size,
                require_cloud_quality=require_cloud_quality,
                allow_occluded=allow_occluded,
            )
        except RuntimeError:
            break
        result.append(window)
        anchor = max(window, key=lambda frame: _frame_score(rows[frame]))
        radius = max(int(minimum_separation), window_size)
        remaining = {
            frame: row
            for frame, row in remaining.items()
            if abs(frame - anchor) >= radius
        }
    return result


def _anchor_window(
    rows: dict[int, dict[str, Any]],
    anchor: int,
    direction: int,
    *,
    sequence_start: int,
    sequence_end: int,
    part_start: int,
    minimum_views: int,
    window_size: int,
) -> list[int]:
    selected = [anchor]
    for offset in range(1, window_size):
        frame = anchor + direction * offset
        if frame < max(sequence_start, part_start) or frame > sequence_end:
            break
        row = rows.get(frame)
        if (
            row is None
            or row.get("state") not in {"static", "assembled"}
            or int(row.get("observing_views", 0)) < minimum_views
        ):
            break
        selected.append(frame)
    return sorted(selected)


def infer_anchors(
    rows: dict[int, dict[str, Any]],
    dynamic_ranges: list[list[int]],
    *,
    sequence_start: int,
    sequence_end: int,
    part_start: int,
    minimum_views: int,
    window_size: int,
    search_frames: int | None = None,
    candidates_per_interval: int = 1,
    candidate_minimum_separation: int = 8,
    require_cloud_quality: bool = True,
    allow_occluded: bool = False,
) -> tuple[list[int], dict[str, list[int]]]:
    """Choose registration anchors from the static intervals around motion.

    A motion boundary is frequently the worst possible calibration image: a
    part is entering the scene, still held by a hand, or motion blurred.  The
    old resolver nevertheless used the boundary itself as an anchor.  Search
    the adjacent static interval instead and use its best stable multi-view
    window.  The tracker can propagate backwards from a post-motion anchor,
    so the first visible moving interval does not need a weak start anchor.
    """
    anchors: list[int] = []
    windows: dict[str, list[int]] = {}
    merged_dynamic = _merge_ranges(dynamic_ranges)
    stable_states = {"static", "assembled"}
    if allow_occluded:
        stable_states.add("occluded")

    def add_static_interval(
        interval_start: int,
        interval_end: int,
        *,
        prefer: str,
    ) -> None:
        interval_start = max(sequence_start, part_start, interval_start)
        interval_end = min(sequence_end, interval_end)
        if interval_start > interval_end:
            return
        if search_frames is not None and search_frames > 0:
            if prefer == "start":
                interval_end = min(interval_end, interval_start + search_frames - 1)
            elif prefer == "end":
                interval_start = max(interval_start, interval_end - search_frames + 1)
            else:
                raise ValueError(f"unknown anchor interval preference: {prefer}")
        candidate_windows = choose_calibration_windows(
            rows,
            start=interval_start,
            end=interval_end,
            minimum_views=minimum_views,
            window_size=window_size,
            maximum_windows=candidates_per_interval,
            minimum_separation=candidate_minimum_separation,
            require_cloud_quality=require_cloud_quality,
            allow_occluded=allow_occluded,
        )
        for window in candidate_windows:
            anchor = max(window, key=lambda frame: _frame_score(rows[frame]))
            if anchor not in anchors:
                anchors.append(anchor)
                windows[str(anchor)] = window

    for index, (dynamic_start, dynamic_end) in enumerate(merged_dynamic):
        previous_end = (
            part_start - 1
            if index == 0
            else merged_dynamic[index - 1][1]
        )
        next_start = (
            sequence_end + 1
            if index + 1 == len(merged_dynamic)
            else merged_dynamic[index + 1][0]
        )
        add_static_interval(
            previous_end + 1, dynamic_start - 1, prefer="end"
        )
        add_static_interval(
            dynamic_end + 1, next_start - 1, prefer="start"
        )

    # Never turn a moving boundary or a distant visible fragment into an
    # absolute anchor. Every motion must end in a real stable window. A missing
    # window is an observability failure and must remain visible to the caller.
    for index, (dynamic_start, dynamic_end) in enumerate(merged_dynamic):
        next_motion_start = (
            sequence_end + 1
            if index + 1 == len(merged_dynamic)
            else merged_dynamic[index + 1][0]
        )
        after = [
            anchor for anchor in anchors
            if dynamic_end < anchor < next_motion_start
        ]
        if not after:
            raise RuntimeError(
                "motion range has no trusted stable anchor after it: "
                f"{dynamic_start}..{dynamic_end}"
            )
        stable_before = [
            frame
            for frame in range(part_start, dynamic_start)
            if frame in rows
            and rows[frame].get("state") in stable_states
            and int(rows[frame].get("observing_views", 0)) >= minimum_views
            and _anchor_frame_usable(
                rows[frame],
                require_cloud_quality=require_cloud_quality,
            )
        ]
        if stable_before and not any(anchor < dynamic_start for anchor in anchors):
            raise RuntimeError(
                "motion range has observable static evidence but no trusted "
                f"anchor before it: {dynamic_start}..{dynamic_end}"
            )
    if not anchors:
        visible = [
            frame for frame, row in rows.items()
            if frame >= part_start
            and row.get("state") in stable_states
            and int(row.get("observing_views", 0)) >= minimum_views
            and _anchor_frame_usable(
                row,
                require_cloud_quality=require_cloud_quality,
            )
        ]
        if not visible:
            raise RuntimeError("no sufficiently visible anchor frame")
        anchor = max(visible, key=lambda frame: _frame_score(rows[frame]))
        anchors = [anchor]
        windows[str(anchor)] = [anchor]
    return sorted(anchors), windows


def resolve_pose_config(
    config: dict[str, Any],
    state_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a normal solver config and an audit report.

    Automatic values are enabled by ``automation``.  Explicit fields remain
    fallback evidence and can be preserved by disabling the corresponding
    flag.
    """

    resolved = deepcopy(config)
    settings = dict(config.get("automation", {}))
    # A reference part defines the world frame at a trusted stable window; it
    # is not a promise that the same physical part never moves again.
    settings.setdefault("allow_moving_reference", True)
    settings.setdefault("reference_policy", "first_settled_window")
    settings.setdefault("stable_settling_frames", 5)
    settings.setdefault("require_stable_cloud_quality", True)
    # Small or distant parts can have excellent multi-view masks while a
    # strict fused-cloud gate rejects every frame in a short stable interval.
    # Keep cloud-qualified anchors as the first choice, then permit an
    # auditable mask-visible fallback that downstream render objectives must
    # verify instead of making the whole sequence unsolvable.
    settings.setdefault("allow_mask_only_anchor_fallback", True)
    settings.setdefault("scale_consensus_first_static_interval", True)
    settings.setdefault("validate_table_support", True)
    settings.setdefault("require_table_support", True)
    settings.setdefault("align_reference_to_table", True)
    settings.setdefault("maximum_table_alignment_m", 0.03)
    scale_settings = settings.setdefault("silhouette_scale_calibration", {})
    scale_settings.setdefault("enabled", True)
    scale_settings.setdefault(
        "scale_factors",
        [
            0.70, 0.80, 0.90, 0.92, 0.94, 0.96, 0.98,
            1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.20, 1.30,
        ],
    )
    scale_settings.setdefault("maximum_anchor_frames", 3)
    # A single rigid part has one physical scale.  Re-estimating it from later
    # static intervals (often cropped, occluded, or already assembled) lets a
    # pose/visibility error masquerade as a metric size correction.
    scale_settings.setdefault("first_static_interval_only", True)
    scale_settings.setdefault("exact_mesh_render", True)
    scale_settings.setdefault("resolution", [320, 180])
    scale_settings.setdefault("minimum_improvement", 0.005)
    scale_settings.setdefault("maximum_holdout_degradation", 0.02)
    scale_settings.setdefault("maximum_area_log_degradation", 0.0)
    scale_settings.setdefault(
        "weights",
        {
            "iou": 1.0,
            "contour": 0.05,
            "target_coverage": 0.0,
            "depth": 0.02,
        },
    )
    scale_settings.setdefault("minimum_selected_iou", 0.05)
    scale_settings.setdefault("minimum_selected_holdout_iou", 0.02)
    scale_settings.setdefault("require_quality_gate", True)
    # Scale and SE(3) are coupled under partial visibility.  Re-evaluate scale
    # after the winning static anchor has been selected and render-refined so
    # a rejected 90/180-degree anchor cannot contaminate the final metric size.
    scale_settings.setdefault("post_anchor_render_pass", True)
    # If the scale change triggers one last pose refinement, validate scale
    # once more at that final pose.  Nothing may move SE(3) after this pass.
    scale_settings.setdefault("final_pose_scale_pass", True)
    scale_settings.setdefault("visual_loss_tie_tolerance", 0.01)
    scale_settings.setdefault(
        "minimum_iou_to_trust_configured_scale_prior", 0.62
    )
    scale_settings.setdefault(
        "maximum_configured_prior_frame_iou_degradation", 0.01
    )
    scale_settings.setdefault(
        "minimum_configured_prior_improved_frame_fraction", 0.5
    )
    resolved["automation"] = settings
    anchor_refinement = settings.setdefault("anchor_render_refinement", {})
    anchor_refinement.setdefault("enabled", True)
    anchor_refinement.setdefault(
        "anchor_translation_steps_m", [0.020, 0.010, 0.005, 0.002]
    )
    anchor_refinement.setdefault(
        "anchor_rotation_steps_deg", [15.0, 10.0, 5.0, 2.0, 1.0]
    )
    anchor_refinement.setdefault("anchor_maximum_translation_delta_m", 0.08)
    anchor_refinement.setdefault("anchor_maximum_rotation_delta_deg", 45.0)
    anchor_refinement.setdefault("anchor_minimum_improvement", 0.003)
    anchor_refinement.setdefault("anchor_maximum_holdout_degradation", 0.02)
    anchor_refinement.setdefault("anchor_prior_weight", 0.01)
    anchor_refinement.setdefault(
        "anchor_reference_maximum_translation_delta_m", 0.05
    )
    anchor_refinement.setdefault(
        "anchor_reference_maximum_rotation_delta_deg", 15.0
    )
    view_quality = resolved.setdefault("view_quality", {})
    view_quality.setdefault("minimum_full_mask_pixels", 800)
    view_quality.setdefault("maximum_mask_area_ratio", 4.0)
    render_refinement = resolved.get("render_loss_refinement", {})
    if "frames_dir" in resolved:
        render_refinement = resolved.setdefault("render_loss_refinement", {})
        render_refinement.setdefault(
            "enabled", bool(settings.get("enable_render_loss_refinement", True))
        )
        render_refinement.setdefault("resolution", [160, 90])
        render_refinement.setdefault("occlusion_aware", True)
        render_refinement.setdefault("full_view_correction_fps", 5.0)
        render_refinement.setdefault("minimum_optimize_views", 3)
        render_refinement.setdefault("minimum_per_view_iou", 0.005)
        render_refinement.setdefault("maximum_worst_view_loss", 2.0)
        if len(resolved["views"]) >= 4:
            render_refinement.setdefault(
                "holdout_views", [resolved["views"][-1]]
            )
            render_refinement.setdefault(
                "optimize_views", resolved["views"][:-1]
            )
            render_refinement.setdefault("minimum_holdout_iou", 0.005)
        refinement_parts = render_refinement.setdefault("parts", {})
        for part in resolved["parts"]:
            refinement_parts.setdefault(part, {"enabled": True})
    if render_refinement.get("enabled", False):
        render_refinement.setdefault("minimum_full_mask_pixels", 800)
        render_refinement.setdefault("maximum_mask_area_ratio", 4.0)
        # A synchronized camera that sees only a small occlusion fragment must
        # not pull a correct pose toward that fragment.  Empty masks are
        # already rejected absolutely; this cross-view ratio also rejects
        # severely truncated modal masks during hand/object occlusion.
        render_refinement.setdefault("minimum_mask_area_ratio", 0.35)
        render_refinement.setdefault("trim_worst_views", 1)
        quality_summary = (
            Path(resolved["point_cloud_root"]) / "quality_cloud_summary.json"
            if resolved.get("point_cloud_root")
            else None
        )
        if quality_summary is not None and quality_summary.exists():
            render_refinement.setdefault("use_cloud_supported_view_gate", True)
            render_refinement.setdefault(
                "quality_cloud_summary", str(quality_summary)
            )
            render_refinement.setdefault("minimum_supported_points_per_view", 30)
            render_refinement.setdefault("minimum_view_support_fraction", 0.25)
        render_refinement.setdefault("minimum_refined_iou", 0.02)
        render_refinement.setdefault(
            "minimum_refined_target_coverage", 0.02
        )
        render_refinement.setdefault("minimum_holdout_iou", 0.0)
    start, end = (
        int(config["frames"]["start"]),
        int(config["frames"]["end"]),
    )
    # State detection is intentionally conservative: a lagged motion vote,
    # median filtering, and enter dwell suppress hand-occlusion false
    # positives.  The detected boundary therefore describes when motion was
    # confirmed, not necessarily its first physical frame.  Backdate the
    # unlocked range by that known detector latency so static consensus does
    # not visibly pin a part during pickup.  Explicit automation settings
    # remain authoritative, and legacy reports without thresholds retain the
    # previous zero-padding behavior.
    detector_thresholds = state_report.get("thresholds", {})
    if "motion_padding_before" not in settings and detector_thresholds:
        detector_latency = (
            max(1, int(detector_thresholds.get("motion_lag", 1)))
            + max(1, int(detector_thresholds.get("dwell_on", 1)))
            + 2  # one median-filter radius plus the first threshold frame
        )
        settings["motion_padding_before"] = min(12, detector_latency)
        settings["motion_padding_before_source"] = (
            "detector_latency_compensation"
        )
    minimum_views = int(settings.get(
        "minimum_observing_views",
        max(1, min(2, len(config["views"]))),
    ))
    minimum_dynamic = int(settings.get("minimum_dynamic_frames", 2))
    pad_before = int(settings.get("motion_padding_before", 0))
    pad_after = int(settings.get("motion_padding_after", 0))
    merge_dynamic_gap = int(settings.get("merge_dynamic_gap_frames", 0))
    anchor_window = int(settings.get("anchor_window_frames", 8))
    anchor_search = settings.get("anchor_search_frames")
    anchor_search = None if anchor_search is None else int(anchor_search)
    candidates_per_interval = int(
        settings.get("anchor_candidates_per_interval", 3)
    )
    candidate_minimum_separation = int(
        settings.get("anchor_candidate_minimum_separation_frames", 12)
    )
    calibration_window = int(settings.get("calibration_window_frames", 12))
    allow_moving_reference = bool(settings["allow_moving_reference"])
    settling_frames = int(settings["stable_settling_frames"])
    require_stable_cloud_quality = bool(
        settings["require_stable_cloud_quality"]
    )

    if settings.get("infer_reference_part", False):
        if settings["reference_policy"] == "first_settled_window":
            reference, reference_selection = (
                choose_reference_part_from_settled_windows(
                    resolved,
                    state_report,
                    minimum_views=minimum_views,
                    window_size=calibration_window,
                    settling_frames=settling_frames,
                    require_cloud_quality=require_stable_cloud_quality,
                )
            )
            reference_scores = {
                part: -float(value["window"][0])
                for part, value in reference_selection["candidates"].items()
            }
        elif settings["reference_policy"] == "legacy_visibility_motion_score":
            reference, reference_scores = choose_reference_part(
                config, state_report
            )
            reference_selection = {
                "policy": "legacy_visibility_motion_score",
                "selected": reference,
            }
        else:
            raise ValueError(
                f"unknown automation reference_policy: "
                f"{settings['reference_policy']!r}"
            )
        resolved["reference_part"] = reference
    else:
        reference = str(config["reference_part"])
        reference_scores = {reference: 1.0}
        reference_selection = {
            "policy": "configured_part_stable_window",
            "selected": reference,
        }

    audit: dict[str, Any] = {
        "reference_part": reference,
        "reference_scores": reference_scores,
        "reference_selection": reference_selection,
        "settings": settings,
        "parts": {},
    }
    multiframe = resolved.get("multiframe_optimization", {})
    multiframe_enabled = bool(multiframe.get("enabled", False))
    multiframe_auto = dict(multiframe.get("auto", {}))
    multiframe_windows = multiframe.setdefault("windows", [])
    multiframe_window_names = {
        str(window.get("name", "")) for window in multiframe_windows
    }
    for part in resolved["parts"]:
        rows = _state_rows(state_report, part)
        if not rows:
            raise RuntimeError(f"{part}: state diagnostics are missing")
        quality_rows = _cloud_quality_rows(resolved, part)
        for frame, quality in quality_rows.items():
            if frame in rows:
                rows[frame]["cloud_quality"] = quality
        state = resolved["states"][part]
        if settings.get("infer_symmetry", False):
            from common.mesh_observability import infer_mesh_observability

            observability = infer_mesh_observability(
                Path(resolved["mesh_dir"]) / f"{part}.glb"
            )
            inferred_symmetry = dict(observability["symmetry"])
            inferred_is_textured = bool(observability["has_texture"])
            should_infer = settings.get(
                "force_inferred_symmetry", False
            ) or not state.get("symmetry")
            if should_infer:
                # Geometry can be axially symmetric while texture or asymmetric
                # sub-components make that rotation observable.  Keep the
                # inferred symmetry as a proposal generator, not as a pose
                # equivalence used to waive temporal rotation errors.
                state["symmetry"] = (
                    {"equivalence": "none"}
                    if inferred_is_textured
                    and inferred_symmetry.get("equivalence") != "none"
                    else inferred_symmetry
                )
            if (
                settings.get("infer_appearance", True)
                and observability["has_texture"]
            ):
                appearance = state.setdefault("appearance", {})
                declared_symmetry = dict(
                    state.get("symmetry", {"equivalence": "none"})
                )
                candidate_geometry_symmetry = (
                    declared_symmetry
                    if declared_symmetry.get("equivalence", "none") != "none"
                    else inferred_symmetry
                )
                if candidate_geometry_symmetry.get("equivalence") != "none":
                    # Geometry proposes equivalent-looking orientations, but a
                    # textured mesh makes those rotations physically
                    # observable. Generate them as candidates while enforcing
                    # full-rotation temporal continuity between anchors.
                    candidate_geometry_symmetry = dict(
                        candidate_geometry_symmetry
                    )
                    candidate_geometry_symmetry.setdefault(
                        "candidate_step_deg",
                        float(settings.get(
                            "appearance_candidate_step_deg", 30.0
                        )),
                    )
                    appearance.setdefault(
                        "candidate_symmetry", candidate_geometry_symmetry
                    )
                    state["symmetry"] = {
                        "equivalence": "none",
                        "axis_raw": candidate_geometry_symmetry.get(
                            "axis_raw", [0.0, 1.0, 0.0]
                        ),
                    }
                appearance.setdefault("enabled", True)
                appearance.setdefault(
                    "resolution",
                    settings.get("appearance_resolution", [240, 135]),
                )
                appearance.setdefault("texture_erosion_pixels", 3)
                appearance.setdefault("min_mask_pixels", 20)
                appearance.setdefault("minimum_full_mask_pixels", 800)
                appearance.setdefault("maximum_mask_area_ratio", 4.0)
                appearance.setdefault("trim_worst_views", 1)
                appearance.setdefault("minimum_observations", 3)
                appearance.setdefault("minimum_mean_iou", 0.02)
                appearance.setdefault("minimum_worst_view_iou", 0.005)
                appearance.setdefault("silhouette_weight", 0.5)
                appearance.setdefault("texture_weight", 1.0)
                appearance.setdefault("photometric_weight", 0.75)
                appearance.setdefault("photometric_erosion_pixels", 2)
                appearance.setdefault("photometric_blur_sigma", 1.5)
                appearance.setdefault("photometric_minimum_pixels", 60)
                # For the world-reference body, PCA fixes the upright basin
                # but does not densely test rotation around gravity.  Search
                # that remaining observable degree of freedom on the first
                # clean tabletop interval.  Thin cables and other articulated
                # appendages are removed from this auxiliary yaw score only;
                # the original masks remain authoritative for pose/scale QA.
                appearance.setdefault(
                    "table_yaw_search",
                    part == resolved.get("reference_part"),
                )
                appearance.setdefault("table_yaw_step_deg", 30.0)
                appearance.setdefault(
                    "table_yaw_first_static_interval_only", True
                )
                appearance.setdefault("rigid_core_opening_pixels", 2)
                appearance.setdefault("rigid_core_keep_largest", True)
                appearance.setdefault(
                    "rigid_core_silhouette_weight",
                    0.75 if part == resolved.get("reference_part") else 0.0,
                )
                appearance.setdefault("table_support_weight", 0.3)
                appearance.setdefault(
                    "table_support_maximum_observed_bottom_gap_m", 0.05
                )
                appearance.setdefault("table_support_contact_quantile", 0.01)
                appearance.setdefault("table_support_gap_cap_m", 0.02)
                appearance.setdefault(
                    "table_support_penetration_tolerance_m", 0.005
                )
                # Compare orientation basins only after removing their
                # table-normal translation nuisance.  Otherwise a wrong
                # top/bottom basin can win solely because its registration
                # happened to place one face closer to the support plane.
                appearance.setdefault(
                    "normalize_table_height_before_scoring", True
                )
                appearance.setdefault(
                    "table_height_maximum_shift_m", 0.05
                )
                # Texture may resolve a near-symmetric rotation, but it must
                # not override a materially worse multi-view silhouette.
                appearance.setdefault("maximum_candidate_iou_drop", 0.04)
                appearance.setdefault("relative_iou_gate_penalty", 1.0)
                appearance.setdefault("worst_view_weight", 0.25)
                appearance.setdefault("transition_weight", 0.5)
                # Candidate-chain continuity and final trajectory validation
                # must use the same physical rate envelope.  A stricter hidden
                # appearance default can make every orientation candidate
                # unreachable even though that motion is valid downstream.
                appearance.setdefault(
                    "max_rotation_deg_per_frame",
                    float(
                        state.get("validation", {}).get(
                            "max_rotation_step_deg", 25.0
                        )
                    ),
                )
                appearance.setdefault("hard_rotation_rate", True)
                appearance.setdefault("max_translation_m_per_frame", 0.05)
        else:
            observability = None
        part_start = int(
            resolved.get("part_start_frames", {}).get(part, start)
        )
        detected = [
            [int(pair[0]), int(pair[1])]
            for pair in (
                state_report["parts"][part]
                .get("detected_moving_ranges", [])
            )
            if int(pair[1]) - int(pair[0]) + 1 >= minimum_dynamic
        ]
        detected = _merge_nearby_ranges([
            [
                max(part_start, range_start - pad_before),
                min(end, range_end + pad_after),
            ]
            for range_start, range_end in detected
        ], merge_dynamic_gap)

        if part == reference and not allow_moving_reference:
            dynamic: list[list[int]] = []
            static = [[start, end]]
        elif settings.get("use_detected_states", False):
            dynamic = detected
            static = _complement(dynamic, start, end)
            state["dynamic_ranges"] = dynamic
            state["static_ranges"] = static
        else:
            dynamic = [
                [int(pair[0]), int(pair[1])]
                for pair in state.get("dynamic_ranges", [])
            ]
            static = [
                [int(pair[0]), int(pair[1])]
                for pair in state.get("static_ranges", [])
            ]

        if (
            settings.get("use_detected_states", False)
            and settings.get("lock_detected_static_pose", True)
        ):
            consensus = resolved.setdefault("static_pose_consensus", {})
            consensus.setdefault("maximum_consensus_translation_m", 0.04)
            consensus.setdefault("maximum_consensus_rotation_deg", 25.0)
            consensus.setdefault("parts", {})[part] = [
                [int(pair[0]), int(pair[1])] for pair in static
            ]

        generated_multiframe_windows = []
        if multiframe_enabled and multiframe.get(
            "auto_static_windows", False
        ):
            minimum_static_frames = int(
                multiframe_auto.get("minimum_static_frames", 5)
            )
            for range_start, range_end in static:
                if range_end - range_start + 1 < minimum_static_frames:
                    continue
                name = f"auto_static_{part}_{range_start}_{range_end}"
                if name in multiframe_window_names:
                    continue
                window = {
                    "name": name,
                    "mode": "static_window",
                    "part": part,
                    "frame_range": [int(range_start), int(range_end)],
                    "maximum_evidence_frames": int(
                        multiframe_auto.get("maximum_evidence_frames", 5)
                    ),
                    "maximum_candidate_frames": int(
                        multiframe_auto.get("maximum_candidate_frames", 9)
                    ),
                    "minimum_observing_views": int(
                        multiframe_auto.get("minimum_observing_views", 2)
                    ),
                    "maximum_visual_loss_degradation": float(
                        multiframe_auto.get(
                            "static_maximum_visual_loss_degradation", 0.08
                        )
                    ),
                }
                follows_dynamic = any(
                    int(dynamic_end) + 1 == int(range_start)
                    for _, dynamic_end in dynamic
                )
                is_initial_static = int(range_start) == int(part_start)
                initial_parts = multiframe_auto.get("initial_refinement_parts")
                refine_initial = bool(
                    multiframe_auto.get("refine_initial_static_pose", False)
                ) and (
                    initial_parts is None or part in set(map(str, initial_parts))
                )
                refine_post_dynamic = bool(
                    multiframe_auto.get(
                        "refine_post_dynamic_static_pose", False
                    )
                )
                if (
                    (is_initial_static and refine_initial)
                    or (
                        follows_dynamic
                        and part != reference
                        and refine_post_dynamic
                    )
                ):
                    refinement = {
                        "refine_constant_pose": True,
                        "constant_translation_steps_m": list(
                            multiframe_auto.get(
                                "constant_translation_steps_m",
                                [0.015, 0.008, 0.004, 0.002],
                            )
                        ),
                        "constant_rotation_steps_deg": list(
                            multiframe_auto.get(
                                "constant_rotation_steps_deg", [2.0, 1.0]
                            )
                        ),
                        "constant_maximum_translation_delta_m": float(
                            multiframe_auto.get(
                                "constant_maximum_translation_delta_m", 0.03
                            )
                        ),
                        "constant_maximum_rotation_delta_deg": float(
                            multiframe_auto.get(
                                "constant_maximum_rotation_delta_deg", 5.0
                            )
                        ),
                        "constant_minimum_improvement": float(
                            multiframe_auto.get(
                                "constant_minimum_improvement", 0.005
                            )
                        ),
                        "constant_maximum_holdout_degradation": float(
                            multiframe_auto.get(
                                "constant_maximum_holdout_degradation", 0.02
                            )
                        ),
                        "constant_optimize_rotation": bool(
                            multiframe_auto.get(
                                (
                                    "initial_constant_optimize_rotation"
                                    if is_initial_static
                                    else "constant_optimize_rotation"
                                ),
                                is_initial_static,
                            )
                        ),
                    }
                    if part != reference:
                        refinement["reference_part"] = reference
                    window.update(refinement)
                multiframe_windows.append(window)
                multiframe_window_names.add(name)
                generated_multiframe_windows.append(name)
        if multiframe_enabled and multiframe.get(
            "auto_dynamic_windows", False
        ):
            minimum_dynamic_frames = int(
                multiframe_auto.get("minimum_dynamic_frames", 3)
            )
            for range_start, range_end in dynamic:
                if range_end - range_start + 1 < minimum_dynamic_frames:
                    continue
                name = f"auto_dynamic_{part}_{range_start}_{range_end}"
                if name in multiframe_window_names:
                    continue
                window = {
                    "name": name,
                    "mode": "dynamic_window",
                    "part": part,
                    "frame_range": [int(range_start), int(range_end)],
                    "candidate_radius_frames": int(
                        multiframe_auto.get("candidate_radius_frames", 2)
                    ),
                    "maximum_translation_step_m": float(
                        multiframe_auto.get(
                            "maximum_translation_step_m", 0.04
                        )
                    ),
                    "maximum_rotation_step_deg": float(
                        multiframe_auto.get(
                            "maximum_rotation_step_deg", 20.0
                        )
                    ),
                    "maximum_internal_translation_degradation_m": float(
                        multiframe_auto.get(
                            "maximum_internal_translation_degradation_m", 0.0
                        )
                    ),
                    "maximum_internal_rotation_degradation_deg": float(
                        multiframe_auto.get(
                            "maximum_internal_rotation_degradation_deg", 0.0
                        )
                    ),
                    "temporal_weight": float(
                        multiframe_auto.get("temporal_weight", 0.20)
                    ),
                    "baseline_weight": float(
                        multiframe_auto.get("baseline_weight", 0.03)
                    ),
                    "maximum_visual_loss_degradation": float(
                        multiframe_auto.get(
                            "maximum_visual_loss_degradation", 0.03
                        )
                    ),
                }
                multiframe_windows.append(window)
                multiframe_window_names.add(name)
                generated_multiframe_windows.append(name)

        part_audit: dict[str, Any] = {
            "detected_dynamic_ranges": detected,
            "resolved_dynamic_ranges": dynamic,
            "resolved_static_ranges": static,
            "static_pose_lock_ranges": (
                static
                if settings.get("use_detected_states", False)
                and settings.get("lock_detected_static_pose", True)
                else []
            ),
            "generated_multiframe_windows": generated_multiframe_windows,
        }
        assembly_parent = state.get("assembly_parent")
        assembled_from = state_report["parts"][part].get(
            "detected_assembled_from"
        )
        assembled_confirmed = state_report["parts"][part].get(
            "detected_assembled_confirmed_at"
        )
        if assembly_parent is not None and assembled_from is not None:
            assembled_from = int(assembled_from)
            assembled_confirmed = int(
                assembled_confirmed
                if assembled_confirmed is not None
                else assembled_from
            )
            candidate_frames = [
                frame
                for frame in range(assembled_from, assembled_confirmed + 1)
                if frame in rows
                and rows[frame].get("state") == "assembled"
                and int(rows[frame].get("observing_views", 0)) > 0
            ]
            if candidate_frames:
                maximum = max(1, int(
                    state.get("assembly_latch", {}).get(
                        "maximum_relative_anchor_frames", 5
                    )
                ))
                if len(candidate_frames) > maximum:
                    indices = [
                        round(index * (len(candidate_frames) - 1) / (maximum - 1))
                        for index in range(maximum)
                    ] if maximum > 1 else [len(candidate_frames) - 1]
                    candidate_frames = [candidate_frames[index] for index in indices]
                consensus = resolved.setdefault("static_pose_consensus", {})
                rigid_follow = consensus.setdefault("rigid_follow", {})
                rigid_follow.setdefault(part, []).append({
                    "reference_part": str(assembly_parent),
                    "frame_range": [assembled_from, end],
                    "relative_anchor_frames": candidate_frames,
                    "maximum_relative_translation_residual_m": float(
                        state.get("assembly_latch", {}).get(
                            "maximum_relative_translation_residual_m", 0.03
                        )
                    ),
                    "maximum_relative_rotation_residual_deg": float(
                        state.get("assembly_latch", {}).get(
                            "maximum_relative_rotation_residual_deg", 20.0
                        )
                    ),
                })
                part_audit["assembly_rigid_follow"] = {
                    "reference_part": str(assembly_parent),
                    "assembled_from": assembled_from,
                    "confirmed_at": assembled_confirmed,
                    "relative_anchor_frames": candidate_frames,
                }
        if part == reference and allow_moving_reference and dynamic:
            validation = state.setdefault("validation", {})
            old_limits = {
                key: validation.get(key)
                for key in (
                    "max_translation_step_m",
                    "max_rotation_step_deg",
                )
            }
            validation["max_translation_step_m"] = max(
                float(validation.get("max_translation_step_m", 0.0)),
                float(settings.get(
                    "moving_reference_min_translation_step_m", 0.03
                )),
            )
            validation["max_rotation_step_deg"] = max(
                float(validation.get("max_rotation_step_deg", 0.0)),
                float(settings.get(
                    "moving_reference_min_rotation_step_deg", 5.0
                )),
            )
            part_audit["moving_reference_validation"] = {
                "previous": old_limits,
                "resolved": {
                    "max_translation_step_m": validation[
                        "max_translation_step_m"
                    ],
                    "max_rotation_step_deg": validation[
                        "max_rotation_step_deg"
                    ],
                },
                "reason": "reference_part_is_tracked_after_stable_initialization",
            }
        if observability is not None:
            part_audit["mesh_observability"] = observability
        if part == reference and settings.get(
            "infer_calibration_frames", False
        ):
            calibration, stable_report = first_settled_window(
                rows,
                quality_rows,
                start=max(start, part_start),
                end=end,
                maximum_views=len(resolved["views"]),
                minimum_views=minimum_views,
                size=calibration_window,
                settling_frames=settling_frames,
                require_cloud_quality=require_stable_cloud_quality,
            )
            state["calibration_frames"] = calibration
            part_audit["calibration_frames"] = calibration
            part_audit["stable_reference_window"] = stable_report
        if settings.get("infer_anchors", False) and (
            part != reference or allow_moving_reference
        ):
            anchor_requires_cloud_quality = True
            try:
                anchors, windows = infer_anchors(
                    rows,
                    dynamic,
                    sequence_start=start,
                    sequence_end=end,
                    part_start=part_start,
                    minimum_views=minimum_views,
                    window_size=anchor_window,
                    search_frames=anchor_search,
                    candidates_per_interval=candidates_per_interval,
                    candidate_minimum_separation=(
                        candidate_minimum_separation
                    ),
                    require_cloud_quality=True,
                    allow_occluded=False,
                )
            except RuntimeError as error:
                if not settings["allow_mask_only_anchor_fallback"]:
                    raise
                anchors, windows = infer_anchors(
                    rows,
                    dynamic,
                    sequence_start=start,
                    sequence_end=end,
                    part_start=part_start,
                    minimum_views=minimum_views,
                    window_size=anchor_window,
                    search_frames=anchor_search,
                    candidates_per_interval=candidates_per_interval,
                    candidate_minimum_separation=(
                        candidate_minimum_separation
                    ),
                    require_cloud_quality=False,
                    allow_occluded=True,
                )
                anchor_requires_cloud_quality = False
                part_audit["anchor_quality_fallback"] = {
                    "mode": "static_multiview_mask_visible",
                    "strict_failure": str(error),
                    "requires_downstream_render_validation": True,
                }
            state["anchor_frames"] = anchors
            state["anchor_windows"] = windows
            if (
                part == reference
                and "calibration_frames" in state
            ):
                calibration = [
                    int(value) for value in state["calibration_frames"]
                ]
                representative = int(
                    part_audit.get("stable_reference_window", {}).get(
                        "representative_frame", calibration[len(calibration) // 2]
                    )
                )
                state["anchor_frames"] = sorted(set(
                    [int(value) for value in anchors] + [representative]
                ))
                state["anchor_windows"] = dict(windows)
                state["anchor_windows"][str(representative)] = calibration
                anchors = state["anchor_frames"]
                windows = state["anchor_windows"]
            part_audit["anchor_frames"] = anchors
            part_audit["anchor_windows"] = windows
            appearance = state.get("appearance", {})
            if appearance.get("enabled", False) and settings.get(
                "infer_appearance_evidence", True
            ):
                evidence_count = int(
                    settings.get("appearance_evidence_frames_per_anchor", 3)
                )
                evidence_separation = int(
                    settings.get("appearance_evidence_minimum_separation", 3)
                )
                appearance["anchor_evidence_frames"] = {
                    str(anchor): _appearance_evidence_frames(
                        windows[str(anchor)],
                        rows,
                        count=evidence_count,
                        minimum_separation=evidence_separation,
                        require_cloud_quality=(
                            anchor_requires_cloud_quality
                        ),
                    )
                    for anchor in anchors
                }
                part_audit["appearance_evidence_frames"] = appearance[
                    "anchor_evidence_frames"
                ]
        if state.get("method") == "auto":
            state["method"] = "cloud_registration"
            part_audit["method"] = {
                "value": "cloud_registration",
                "source": "generic_default",
            }
        refinement_part = (
            resolved.get("render_loss_refinement", {})
            .get("parts", {})
            .get(part)
        )
        if (
            settings.get("use_detected_states", False)
            and isinstance(refinement_part, dict)
            and refinement_part.get("enabled", False)
        ):
            refinement_part["ranges"] = dynamic
            part_audit["render_loss_ranges"] = dynamic
        audit["parts"][part] = part_audit
    return resolved, audit
