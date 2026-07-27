"""Resolve per-sequence pose priors from mask/cloud state diagnostics.

The resolver is deliberately separate from the solver.  It produces an
ordinary validated pose JSON plus a provenance report; the solver never
silently changes state ranges or anchors while running.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


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
    return (
        visible
        - 0.05 * (0.0 if shift is None else float(shift))
        - 0.01 * (0.0 if motion is None else float(motion))
    )


def choose_reference_part(
    config: dict[str, Any],
    state_report: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    start_frames = config.get("part_start_frames", {})
    scores = {}
    for part in config["parts"]:
        rows = _state_rows(state_report, part)
        visible = sum(int(row.get("observing_views", 0)) for row in rows.values())
        moving = sum(row.get("state") == "moving" for row in rows.values())
        start = int(start_frames.get(part, config["frames"]["start"]))
        scores[part] = float(visible - 2 * moving - 0.01 * start)
    if not scores:
        raise ValueError("cannot choose a reference part without parts")
    return max(scores, key=scores.get), scores


def _stable_frames(
    rows: dict[int, dict[str, Any]],
    *,
    start: int,
    end: int,
    minimum_views: int,
) -> list[int]:
    return [
        frame
        for frame in range(start, end + 1)
        if frame in rows
        and rows[frame].get("state") in {"static", "assembled"}
        and int(rows[frame].get("observing_views", 0)) >= minimum_views
    ]


def choose_calibration_window(
    rows: dict[int, dict[str, Any]],
    *,
    start: int,
    end: int,
    minimum_views: int,
    window_size: int,
) -> list[int]:
    stable = set(_stable_frames(
        rows,
        start=start,
        end=end,
        minimum_views=minimum_views,
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
) -> tuple[list[int], dict[str, list[int]]]:
    anchors: list[int] = []
    windows: dict[str, list[int]] = {}
    for dynamic_start, dynamic_end in dynamic_ranges:
        for anchor, direction in ((dynamic_start, -1), (dynamic_end, 1)):
            if anchor not in anchors:
                anchors.append(anchor)
            windows[str(anchor)] = _anchor_window(
                rows,
                anchor,
                direction,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                part_start=part_start,
                minimum_views=minimum_views,
                window_size=window_size,
            )
    if not anchors:
        visible = [
            frame for frame, row in rows.items()
            if frame >= part_start
            and int(row.get("observing_views", 0)) >= minimum_views
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
    start, end = (
        int(config["frames"]["start"]),
        int(config["frames"]["end"]),
    )
    minimum_views = int(settings.get(
        "minimum_observing_views",
        max(1, min(2, len(config["views"]))),
    ))
    minimum_dynamic = int(settings.get("minimum_dynamic_frames", 2))
    pad_before = int(settings.get("motion_padding_before", 0))
    pad_after = int(settings.get("motion_padding_after", 0))
    anchor_window = int(settings.get("anchor_window_frames", 8))
    calibration_window = int(settings.get("calibration_window_frames", 12))

    if settings.get("infer_reference_part", False):
        reference, reference_scores = choose_reference_part(
            config, state_report
        )
        resolved["reference_part"] = reference
    else:
        reference = str(config["reference_part"])
        reference_scores = {reference: 1.0}

    audit: dict[str, Any] = {
        "reference_part": reference,
        "reference_scores": reference_scores,
        "settings": settings,
        "parts": {},
    }
    for part in resolved["parts"]:
        rows = _state_rows(state_report, part)
        if not rows:
            raise RuntimeError(f"{part}: state diagnostics are missing")
        state = resolved["states"][part]
        if settings.get("infer_symmetry", False):
            from common.mesh_observability import infer_mesh_observability

            observability = infer_mesh_observability(
                Path(resolved["mesh_dir"]) / f"{part}.glb"
            )
            if settings.get("force_inferred_symmetry", False) or not state.get(
                "symmetry"
            ):
                state["symmetry"] = observability["symmetry"]
            if (
                settings.get("infer_appearance", True)
                and observability["has_texture"]
            ):
                appearance = state.setdefault("appearance", {})
                appearance.setdefault("enabled", True)
                appearance.setdefault("resolution", [240, 135])
                appearance.setdefault("texture_erosion_pixels", 3)
                appearance.setdefault("min_mask_pixels", 80)
                appearance.setdefault("silhouette_weight", 0.5)
                appearance.setdefault("texture_weight", 1.0)
                appearance.setdefault("transition_weight", 0.5)
                appearance.setdefault("max_rotation_deg_per_frame", 10.0)
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
        detected = _merge_ranges([
            [
                max(part_start, range_start - pad_before),
                min(end, range_end + pad_after),
            ]
            for range_start, range_end in detected
        ])

        if part == reference:
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

        part_audit: dict[str, Any] = {
            "detected_dynamic_ranges": detected,
            "resolved_dynamic_ranges": dynamic,
            "resolved_static_ranges": static,
        }
        if observability is not None:
            part_audit["mesh_observability"] = observability
        if part == reference and settings.get(
            "infer_calibration_frames", False
        ):
            calibration = choose_calibration_window(
                rows,
                start=max(start, part_start),
                end=end,
                minimum_views=minimum_views,
                window_size=calibration_window,
            )
            state["calibration_frames"] = calibration
            part_audit["calibration_frames"] = calibration
        elif part != reference and settings.get("infer_anchors", False):
            anchors, windows = infer_anchors(
                rows,
                dynamic,
                sequence_start=start,
                sequence_end=end,
                part_start=part_start,
                minimum_views=minimum_views,
                window_size=anchor_window,
            )
            state["anchor_frames"] = anchors
            state["anchor_windows"] = windows
            part_audit["anchor_frames"] = anchors
            part_audit["anchor_windows"] = windows
            appearance = state.get("appearance", {})
            if appearance.get("enabled", False) and settings.get(
                "infer_appearance_evidence", True
            ):
                appearance["anchor_evidence_frames"] = {
                    str(anchor): [
                        max(
                            windows[str(anchor)],
                            key=lambda frame: _frame_score(rows[frame]),
                        )
                    ]
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
