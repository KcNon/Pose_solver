#!/usr/bin/env python
"""Render a fused trajectory as overlay, mesh-only, and mesh+XYZ-axis videos."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.depth_gauge import apply_depth_gauge, load_depth_gauge
from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon, load_recon_camera
from common.pose_visualization import camera_from_recon, part_color, solid_mesh


AXIS_COLORS_RGB = {
    "x": (255, 40, 40),
    "y": (40, 235, 70),
    "z": (50, 100, 255),
}


def resolved_render_settings(cfg: dict) -> dict:
    """Return renderer defaults for the minimal unified pose contract."""

    settings = dict(cfg.get("render", {}))
    settings.setdefault("primary_view", cfg["views"][0])
    settings.setdefault(
        "fps", float(cfg.get("frames", {}).get("fps", 5.0) or 5.0)
    )
    settings.setdefault("axis_length_m", 0.09)
    settings.setdefault("axis_radius_m", 0.004)
    return settings


def colored_primitive(mesh: trimesh.Trimesh, rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    rgba = np.asarray([*rgb, 255], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=np.tile(rgba, (len(mesh.faces), 1)))
    return mesh


def make_axis_meshes(length: float, radius: float) -> list[trimesh.Trimesh]:
    values = []
    directions = {
        "x": np.asarray([1.0, 0.0, 0.0]),
        "y": np.asarray([0.0, 1.0, 0.0]),
        "z": np.asarray([0.0, 0.0, 1.0]),
    }
    shaft_length = 0.78 * length
    head_length = 0.22 * length
    for name, direction in directions.items():
        align = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
        shaft_tf = np.asarray(align, float)
        shaft_tf[:3, 3] = direction * (shaft_length / 2.0)
        shaft = trimesh.creation.cylinder(
            radius=radius, height=shaft_length, sections=16, transform=shaft_tf)
        head_tf = np.asarray(align, float)
        head_tf[:3, 3] = direction * (shaft_length + head_length / 2.0)
        head = trimesh.creation.cone(
            radius=2.1 * radius, height=head_length, sections=20, transform=head_tf)
        values.extend([
            colored_primitive(shaft, AXIS_COLORS_RGB[name]),
            colored_primitive(head, AXIS_COLORS_RGB[name]),
        ])
    values.append(colored_primitive(
        trimesh.creation.icosphere(subdivisions=2, radius=1.7 * radius), (245, 245, 245)))
    return values


def project_origin(T: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple[int, int] | None:
    point = E[:3, :3] @ T[:3, 3] + E[:3, 3]
    if point[2] <= 0.01:
        return None
    uvw = K @ point
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def record_visible_in_view(record: dict, view: str) -> bool:
    """Return per-camera render visibility with legacy compatibility."""

    if record.get("pose_valid") is False:
        return False
    visible_views = record.get("visible_views")
    if visible_views is not None:
        return str(view) in {str(value) for value in visible_views}
    return int(record.get("observing_views", 0)) > 0


def depth_visible_foreground(
    mesh_depth: np.ndarray,
    observed_depth: np.ndarray,
    *,
    margin_m: float,
    dilation_pixels: int = 1,
) -> np.ndarray:
    """Remove rendered pixels hidden behind measured scene geometry."""

    rendered = np.asarray(mesh_depth, dtype=np.float32) > 0.0
    observed = np.asarray(observed_depth, dtype=np.float32)
    occluded = (
        rendered
        & np.isfinite(observed)
        & (observed > 1e-4)
        & (observed + float(margin_m) < mesh_depth)
    )
    if dilation_pixels > 0 and occluded.any():
        size = 2 * int(dilation_pixels) + 1
        occluded = cv2.dilate(
            occluded.astype(np.uint8),
            np.ones((size, size), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    return rendered & ~occluded


def mask_visible_foreground(
    foreground: np.ndarray,
    labels: np.ndarray,
    *,
    occluder_labels: list[int] | tuple[int, ...],
    dilation_pixels: int = 0,
) -> np.ndarray:
    """Remove pixels hidden by explicitly labelled modal occluders.

    This is a visualization/compositing operation only. It does not alter the
    trajectory and deliberately requires explicit labels so a rigid part is
    not accidentally treated as a hand.
    """

    visible = np.asarray(foreground, dtype=bool)
    values = np.asarray(labels)
    if visible.shape != values.shape or values.ndim != 2:
        raise ValueError("foreground and labels must be same-size 2-D arrays")
    selected = sorted({int(value) for value in occluder_labels})
    if any(value < 1 or value > 255 for value in selected):
        raise ValueError("mask occluder labels must be in [1, 255]")
    if not selected:
        return visible.copy()
    occluded = np.isin(values, np.asarray(selected, dtype=values.dtype))
    if dilation_pixels > 0 and occluded.any():
        size = 2 * int(dilation_pixels) + 1
        occluded = cv2.dilate(
            occluded.astype(np.uint8),
            np.ones((size, size), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    return visible & ~occluded


def should_report_progress(index: int, total: int, interval: int = 25) -> bool:
    """Keep long renders observable without flooding the output pipe."""

    return index == 1 or index == total or index % interval == 0


def annotate_axes(image: np.ndarray, records: dict, K: np.ndarray, E: np.ndarray) -> None:
    for part, record in records.items():
        T = np.asarray(record["T_world_from_part"], float)
        uv = project_origin(T, K, E)
        if uv is None:
            continue
        x, y = uv
        cv2.putText(image, part, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, part, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (20, 20, 20), 1, cv2.LINE_AA)


def annotate_status(image: np.ndarray, frame: int, records: dict) -> None:
    if records:
        states = " | ".join(
            f"{part}: {record['state']}" for part, record in records.items()
        )
        label = f"frame {frame:03d} | {states}"
    else:
        label = f"frame {frame:03d} | before pose trajectory"
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(image, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--view", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--trajectory", default=None,
                        help="optional trajectory.json instead of <output_root>/pose/trajectory.json")
    parser.add_argument("--output-root", default=None,
                        help="optional result root; useful for derived/refined trajectories")
    parser.add_argument(
        "--timestamps",
        nargs="+",
        help="render only these timestamps; useful for fast multi-stage QA",
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--include-source-prelude",
        action="store_true",
        help=(
            "prepend source frames before the first trajectory timestamp; those "
            "frames contain the original image and no mesh overlay"
        ),
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="write only overlay.mp4 and skip textured/axes/metric passes",
    )
    parser.add_argument(
        "--mesh-operation-only",
        action="store_true",
        help="write only mesh_only.mp4 and skip overlay/textured/axes/metrics",
    )
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="write overlay.mp4 and mesh_only.mp4 with one render pass",
    )
    args = parser.parse_args()
    if sum((args.overlay_only, args.mesh_operation_only, args.basic_only)) > 1:
        raise ValueError(
            "--overlay-only, --mesh-operation-only, and --basic-only are "
            "mutually exclusive"
        )

    cfg = load_json(Path(args.config))
    render_settings = resolved_render_settings(cfg)
    output = Path(args.output_root or cfg["output_root"])
    trajectory_path = Path(args.trajectory) if args.trajectory else output / "pose" / "trajectory.json"
    trajectory = load_json(trajectory_path)
    view = args.view or render_settings["primary_view"]
    view_index = cfg["views"].index(view)
    width, height = args.width, args.height
    fps = float(render_settings["fps"])
    render_dir = output / "render" / view
    render_dir.mkdir(parents=True, exist_ok=True)

    raw_meshes = {
        part: trimesh.load(Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh")
        for part in cfg["parts"]
    }
    color_meshes = {
        part: solid_mesh(raw_meshes[part], part_color(index))
        for index, part in enumerate(cfg["parts"])
    }
    axes = (
        []
        if args.basic_only or args.overlay_only or args.mesh_operation_only
        else make_axis_meshes(
            float(render_settings["axis_length_m"]),
            float(render_settings["axis_radius_m"]),
        )
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if args.basic_only:
        writers = {
            "overlay": cv2.VideoWriter(
                str(render_dir / "overlay.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
            "mesh_only": cv2.VideoWriter(
                str(render_dir / "mesh_only.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
        }
    elif args.mesh_operation_only:
        writers = {
            "mesh_only": cv2.VideoWriter(
                str(render_dir / "mesh_only.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
        }
    else:
        writers = {
            "overlay": cv2.VideoWriter(
                str(render_dir / "overlay.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
        }
    if (
        not args.overlay_only
        and not args.mesh_operation_only
        and not args.basic_only
    ):
        writers.update({
            "mesh_only": cv2.VideoWriter(
                str(render_dir / "mesh_only.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
            "mesh_textured": cv2.VideoWriter(
                str(render_dir / "mesh_textured.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
            "mesh_axes": cv2.VideoWriter(
                str(render_dir / "mesh_axes.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
        })
    if not all(writer.isOpened() for writer in writers.values()):
        raise RuntimeError("could not open output video writers")

    trajectory_items = list(trajectory["frames"].items())
    if args.timestamps and (
        args.start_frame is not None or args.end_frame is not None
    ):
        raise ValueError("--timestamps cannot be combined with frame bounds")
    if args.timestamps:
        if args.include_source_prelude:
            raise ValueError("--timestamps and --include-source-prelude are mutually exclusive")
        wanted = {f"{int(value):06d}" for value in args.timestamps}
        unknown = sorted(wanted - set(trajectory["frames"]))
        if unknown:
            raise ValueError(f"timestamps not in trajectory: {unknown}")
        trajectory_items = [
            item for item in trajectory_items if item[0] in wanted
        ]
    elif args.start_frame is not None or args.end_frame is not None:
        if args.include_source_prelude:
            raise ValueError(
                "frame bounds and --include-source-prelude are mutually exclusive"
            )
        trajectory_items = [
            item
            for item in trajectory_items
            if (args.start_frame is None or int(item[0]) >= args.start_frame)
            and (args.end_frame is None or int(item[0]) <= args.end_frame)
        ]
    elif args.include_source_prelude:
        first_trajectory_frame = min(int(key) for key in trajectory["frames"])
        source_keys = sorted(
            path.stem
            for path in (Path(cfg["frames_dir"]) / view).glob("*.jpg")
            if path.stem.isdigit() and int(path.stem) < first_trajectory_frame
        )
        trajectory_items = [
            (key, None) for key in source_keys
        ] + trajectory_items
    available_frames = [int(key) for key, _ in trajectory_items]
    if args.timestamps:
        sample_frames = set(available_frames)
    else:
        available_set = set(available_frames)
        sample_frames = {
            int(value)
            for value in cfg.get("review_keyframes", [])
            if int(value) in available_set
        }
        if sample_frames and available_frames:
            sample_frames.update((available_frames[0], available_frames[-1]))
        elif available_frames:
            sample_indices = np.linspace(
                0,
                len(available_frames) - 1,
                min(10, len(available_frames)),
                dtype=int,
            )
            sample_frames = {
                available_frames[int(index)] for index in sample_indices
            }
    samples = []
    metrics = {"view": view, "frames": {}}
    mask_root = Path(cfg["masks_dir"])
    use_depth_occlusion = bool(
        render_settings.get("occlusion_aware", False)
    )
    mask_occluder_labels = [
        int(value)
        for value in render_settings.get("mask_occluder_labels", [])
    ]
    mask_occlusion_dilation = int(
        render_settings.get("mask_occlusion_dilation_pixels", 0)
    )
    depth_gauge = (
        load_depth_gauge(str(cfg["depth_gauge_path"]))
        if use_depth_occlusion and cfg.get("depth_gauge_path")
        else None
    )
    try:
        with SceneRenderer(
            width, height, cache_mesh_resources=True
        ) as renderer:
            for item_index, (key, frame_record) in enumerate(trajectory_items, start=1):
                frame = int(key)
                image_path = Path(cfg["frames_dir"]) / view / f"{key}.jpg"
                source = cv2.imread(str(image_path))
                if source is None:
                    raise FileNotFoundError(image_path)
                source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
                labels = np.asarray(
                    Image.open(mask_root / key / f"{view}.png")
                )
                labels = cv2.resize(
                    labels.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                if frame_record is None:
                    blank = np.zeros_like(source)
                    overlay = source.copy()
                    annotate_status(overlay, frame, {})
                    for name, writer in writers.items():
                        writer.write(overlay if name == "overlay" else blank)
                    if frame in sample_frames:
                        samples.append(np.hstack([
                            cv2.resize(overlay, (480, 270), interpolation=cv2.INTER_AREA),
                            cv2.resize(blank, (480, 270), interpolation=cv2.INTER_AREA),
                        ]))
                    if should_report_progress(item_index, len(trajectory_items)):
                        print(
                            f"rendered {frame:03d} ({item_index}/{len(trajectory_items)}; source only)",
                            flush=True,
                        )
                    continue

                records = frame_record["parts"]
                recon_loader = load_recon if use_depth_occlusion else load_recon_camera
                recon = recon_loader(
                    cfg, f"{frame:06d}", backend=cfg["recon_backend"]
                )
                K, E = camera_from_recon(
                    recon, view_index, (height, width)
                )

                transforms = {
                    part: np.asarray(records[part]["S_world_from_raw_mesh"], float)
                    for part in cfg["parts"]
                }
                rigid = {
                    part: np.asarray(records[part]["T_world_from_part"], float)
                    for part in cfg["parts"]
                }
                visible_parts = [
                    part for part in cfg["parts"]
                    if record_visible_in_view(records[part], view)
                ]
                mesh_parts = [
                    (color_meshes[part], transforms[part]) for part in visible_parts
                ]
                mesh_rgb, mesh_depth = renderer.render(mesh_parts, K, E)
                mesh_bgr = cv2.cvtColor(mesh_rgb, cv2.COLOR_RGB2BGR)
                foreground = mesh_depth > 0
                if use_depth_occlusion:
                    observed_depths = recon["depth"]
                    if depth_gauge is not None:
                        observed_depths = apply_depth_gauge(
                            observed_depths, depth_gauge, key
                        )
                    observed_depth = cv2.resize(
                        observed_depths[view_index].astype(np.float32),
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    foreground = depth_visible_foreground(
                        mesh_depth,
                        observed_depth,
                        margin_m=float(
                            render_settings.get(
                                "occlusion_depth_margin_m", 0.02
                            )
                        ),
                        dilation_pixels=int(
                            render_settings.get(
                                "occlusion_dilation_pixels", 1
                            )
                        ),
                    )
                if mask_occluder_labels:
                    foreground = mask_visible_foreground(
                        foreground,
                        labels,
                        occluder_labels=mask_occluder_labels,
                        dilation_pixels=mask_occlusion_dilation,
                    )
                mesh_bgr[~foreground] = 0

                if args.mesh_operation_only:
                    annotate_status(mesh_bgr, frame, records)
                    writers["mesh_only"].write(mesh_bgr)
                    if frame in sample_frames:
                        samples.append(np.hstack([
                            cv2.resize(
                                mesh_bgr,
                                (480, 270),
                                interpolation=cv2.INTER_AREA,
                            ),
                            cv2.resize(
                                source,
                                (480, 270),
                                interpolation=cv2.INTER_AREA,
                            ),
                        ]))
                    if should_report_progress(item_index, len(trajectory_items)):
                        print(
                            f"rendered {frame:03d} "
                            f"({item_index}/{len(trajectory_items)})",
                            flush=True,
                        )
                    continue

                overlay = source.copy().astype(np.float32)
                overlay[foreground] = (
                    0.42 * overlay[foreground]
                    + 0.58 * mesh_bgr[foreground]
                )
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                annotate_status(overlay, frame, records)

                if args.overlay_only:
                    writers["overlay"].write(overlay)
                    if frame in sample_frames:
                        tile = np.hstack([
                            cv2.resize(
                                overlay,
                                (480, 270),
                                interpolation=cv2.INTER_AREA,
                            ),
                            cv2.resize(
                                source,
                                (480, 270),
                                interpolation=cv2.INTER_AREA,
                            ),
                        ])
                        samples.append(tile)
                    if should_report_progress(item_index, len(trajectory_items)):
                        print(
                            f"rendered {frame:03d} "
                            f"({item_index}/{len(trajectory_items)})",
                            flush=True,
                        )
                    continue

                if args.basic_only:
                    writers["overlay"].write(overlay)
                    writers["mesh_only"].write(mesh_bgr)
                    if frame in sample_frames:
                        samples.append(np.hstack([
                            cv2.resize(
                                overlay,
                                (480, 270),
                                interpolation=cv2.INTER_AREA,
                            ),
                            cv2.resize(
                                mesh_bgr,
                                (480, 270),
                                interpolation=cv2.INTER_AREA,
                            ),
                        ]))
                    if should_report_progress(item_index, len(trajectory_items)):
                        print(
                            f"rendered {frame:03d} "
                            f"({item_index}/{len(trajectory_items)})",
                            flush=True,
                        )
                    continue

                textured_rgb, textured_depth = renderer.render(
                    [(raw_meshes[part], transforms[part]) for part in visible_parts], K, E)
                textured_bgr = cv2.cvtColor(textured_rgb, cv2.COLOR_RGB2BGR)
                textured_foreground = textured_depth > 0
                if mask_occluder_labels:
                    textured_foreground = mask_visible_foreground(
                        textured_foreground,
                        labels,
                        occluder_labels=mask_occluder_labels,
                        dilation_pixels=mask_occlusion_dilation,
                    )
                textured_bgr[~textured_foreground] = 0
                annotate_status(textured_bgr, frame, records)

                axis_parts = list(mesh_parts)
                for part in visible_parts:
                    axis_parts.extend((axis, rigid[part]) for axis in axes)
                axes_rgb, axes_depth = renderer.render(axis_parts, K, E)
                axes_bgr = cv2.cvtColor(axes_rgb, cv2.COLOR_RGB2BGR)
                axes_foreground = axes_depth > 0
                if mask_occluder_labels:
                    axes_foreground = mask_visible_foreground(
                        axes_foreground,
                        labels,
                        occluder_labels=mask_occluder_labels,
                        dilation_pixels=mask_occlusion_dilation,
                    )
                axes_bgr[~axes_foreground] = 0
                annotate_axes(
                    axes_bgr,
                    {part: records[part] for part in visible_parts},
                    K,
                    E,
                )
                annotate_status(axes_bgr, frame, records)

                seg = renderer.render_seg(
                    [
                        (part, raw_meshes[part], transforms[part])
                        for part in visible_parts
                    ],
                    K,
                    E,
                )
                metrics["frames"][key] = {
                    part: {
                        "silhouette_iou": iou(
                            seg.get(part, np.zeros((height, width), dtype=bool)),
                            labels == int(cfg["part_ids"][part]),
                        ),
                        "rendered_pixels": int(
                            seg.get(part, np.zeros((height, width), dtype=bool)).sum()
                        ),
                        "mask_pixels": int((labels == int(cfg["part_ids"][part])).sum()),
                        "state": records[part]["state"],
                    }
                    for part in cfg["parts"]
                }

                writers["overlay"].write(overlay)
                writers["mesh_only"].write(mesh_bgr)
                writers["mesh_textured"].write(textured_bgr)
                writers["mesh_axes"].write(axes_bgr)
                if frame in sample_frames:
                    tile = np.hstack([
                        cv2.resize(overlay, (480, 270), interpolation=cv2.INTER_AREA),
                        cv2.resize(axes_bgr, (480, 270), interpolation=cv2.INTER_AREA),
                    ])
                    samples.append(tile)
                if should_report_progress(item_index, len(trajectory_items)):
                    print(
                        f"rendered {frame:03d} ({item_index}/{len(trajectory_items)})",
                        flush=True,
                    )
    finally:
        for writer in writers.values():
            writer.release()

    metrics["summary"] = {}
    metrics["basic_only"] = bool(args.basic_only)
    metrics["rendered_frame_count"] = len(trajectory_items)
    metrics["pose_frame_count"] = len(metrics["frames"])
    for part in cfg["parts"]:
        visible = [
            value[part]["silhouette_iou"] for value in metrics["frames"].values()
            if value[part]["mask_pixels"] > 0 and value[part]["rendered_pixels"] > 0
        ]
        moving = [
            value[part]["silhouette_iou"] for value in metrics["frames"].values()
            if value[part]["state"] == "moving"
            and value[part]["mask_pixels"] > 0 and value[part]["rendered_pixels"] > 0
        ]
        metrics["summary"][part] = {
            "visible_frames": len(visible),
            "visible_mean_iou": float(np.mean(visible)) if visible else None,
            "visible_median_iou": float(np.median(visible)) if visible else None,
            "moving_visible_frames": len(moving),
            "moving_visible_mean_iou": float(np.mean(moving)) if moving else None,
        }
    write_json(render_dir / "metrics.json", metrics)
    if samples:
        contact = np.vstack(samples)
        cv2.imwrite(str(render_dir / "contact_sheet.jpg"), contact,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote videos to {render_dir}", flush=True)


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/stages/pose/render_multiview_pose.py")
    main()
