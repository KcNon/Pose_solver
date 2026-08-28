#!/usr/bin/env python
"""Render connector axes, origins, and local metrics in every camera view."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.connector_geometry import (
    connector_frame_metrics,
    connector_origin_part_m,
    evaluate_connectors,
)
from common.io_utils import load_json, write_json
from common.normalized_recon import load_recon
from common.pose_visualization import camera_from_recon, tile_image_panels


COLORS = {
    "reference": (255, 255, 0),
    "moving": (255, 0, 255),
    "offset": (0, 80, 255),
}


def _unit(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _project(point_world: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple[int, int] | None:
    camera = E[:3, :3] @ point_world + E[:3, 3]
    if camera[2] <= 0.01:
        return None
    projected = K @ camera
    return (
        int(round(projected[0] / projected[2])),
        int(round(projected[1] / projected[2])),
    )


def _world_connector(
    connector: dict[str, Any],
    trajectory: dict[str, Any],
    records: dict[str, Any],
    role: str,
) -> tuple[np.ndarray, np.ndarray]:
    part = str(connector[f"{role}_part"])
    transform = np.asarray(records[part]["T_world_from_part"], dtype=np.float64)
    origin = connector_origin_part_m(connector, role, part, trajectory)
    point = transform[:3, :3] @ origin + transform[:3, 3]
    axis = transform[:3, :3] @ _unit(connector[f"{role}_axis_part"])
    return point, axis


def _draw_axis(
    image: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    *,
    length_m: float,
    color: tuple[int, int, int],
    label: str,
) -> tuple[int, int] | None:
    start = _project(origin, K, E)
    end = _project(origin + length_m * axis, K, E)
    if start is None or end is None:
        return start
    cv2.circle(image, start, 7, color, -1, cv2.LINE_AA)
    cv2.arrowedLine(image, start, end, color, 4, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(
        image,
        label,
        (start[0] + 8, start[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return start


def _draw_mask_contours(
    image: np.ndarray,
    mask_path: Path,
    part_ids: dict[str, int],
) -> None:
    if not mask_path.exists():
        return
    with Image.open(mask_path) as source:
        labels = np.asarray(source)
    if labels.ndim == 3:
        labels = labels[..., 0]
    labels = cv2.resize(
        labels,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    for index, (part, part_id) in enumerate(part_ids.items()):
        contours, _ = cv2.findContours(
            (labels == int(part_id)).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        color = COLORS["moving"] if index == 0 else COLORS["reference"]
        cv2.drawContours(image, contours, -1, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frames", required=True, nargs="+", type=int)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--axis-length-m", type=float, default=0.10)
    args = parser.parse_args()

    cfg = load_json(args.config.expanduser().resolve())
    trajectory = load_json(args.trajectory.expanduser().resolve())
    connectors = {
        name: value
        for name, value in dict(cfg.get("connectors", {})).items()
        if value.get("enabled", True)
    }
    if not connectors:
        raise ValueError("pose config has no enabled connectors")
    output_root = args.output_root.expanduser().resolve()
    keyframe_root = output_root / "keyframes"
    keyframe_root.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(cfg["frames_dir"])
    masks_dir = Path(cfg["masks_dir"])
    views = [str(value) for value in cfg["views"]]

    for frame in sorted(set(args.frames)):
        timestamp = f"{frame:06d}"
        if timestamp not in trajectory["frames"]:
            raise ValueError(f"trajectory lacks frame {timestamp}")
        records = trajectory["frames"][timestamp]["parts"]
        recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
        panels = []
        connector_rows = {
            name: connector_frame_metrics(value, trajectory, frame)
            for name, value in connectors.items()
        }
        for view_index, view in enumerate(views):
            source_path = frames_dir / view / f"{timestamp}.jpg"
            image = cv2.imread(str(source_path))
            if image is None:
                raise FileNotFoundError(source_path)
            image = cv2.resize(
                image,
                (args.width, args.height),
                interpolation=cv2.INTER_AREA,
            )
            _draw_mask_contours(
                image,
                masks_dir / timestamp / f"{view}.png",
                cfg["part_ids"],
            )
            K, E = camera_from_recon(
                recon, view_index, (args.height, args.width)
            )
            for name, connector in connectors.items():
                reference_origin, reference_axis = _world_connector(
                    connector, trajectory, records, "reference"
                )
                moving_origin, moving_axis = _world_connector(
                    connector, trajectory, records, "moving"
                )
                reference_uv = _draw_axis(
                    image,
                    reference_origin,
                    reference_axis,
                    K,
                    E,
                    length_m=args.axis_length_m,
                    color=COLORS["reference"],
                    label=f"{name}:body",
                )
                moving_uv = _draw_axis(
                    image,
                    moving_origin,
                    moving_axis,
                    K,
                    E,
                    length_m=args.axis_length_m,
                    color=COLORS["moving"],
                    label=f"{name}:nozzle",
                )
                if reference_uv is not None and moving_uv is not None:
                    cv2.line(
                        image,
                        reference_uv,
                        moving_uv,
                        COLORS["offset"],
                        2,
                        cv2.LINE_AA,
                    )
                row = connector_rows[name]
                text = (
                    f"angle {row['axis_angle_deg']:.2f} deg | "
                    f"radial {1000.0 * row['radial_offset_m']:.1f} mm | "
                    f"axial {1000.0 * row['axial_offset_m']:.1f} mm"
                )
                cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    f"{timestamp} {view} | {text}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            panels.append(image)
        mosaic = tile_image_panels(panels, columns=4)
        cv2.imwrite(str(keyframe_root / f"{timestamp}.jpg"), mosaic)

    report = evaluate_connectors(connectors, trajectory)
    report.update({
        "config": str(args.config.expanduser().resolve()),
        "trajectory": str(args.trajectory.expanduser().resolve()),
        "review_frames": sorted(set(args.frames)),
        "keyframe_root": str(keyframe_root),
    })
    write_json(output_root / "connector_metrics.json", report)
    print(f"connector review -> {output_root}")


if __name__ == "__main__":
    main()
