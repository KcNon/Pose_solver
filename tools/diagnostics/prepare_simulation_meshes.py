#!/usr/bin/env python3
"""Create bounded simulation visual meshes without materializing components."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.resource_safety import require_memory_guard
from common.simulation_assets import load_flat_mesh, sha256_file


MAX_FACES = 1_000_000


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def main() -> None:
    require_memory_guard("tools/diagnostics/prepare_simulation_meshes.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    output_root = _resolve(config_path.parent, config["output_dir"])
    report_path = output_root / "simulation_mesh_preparation.json"
    if report_path.is_file() and not args.force:
        print(f"[resume] {report_path}")
        return
    output_root.mkdir(parents=True, exist_ok=True)

    rows = {}
    for part, spec in config["parts"].items():
        source = _resolve(config_path.parent, spec["source"])
        destination = output_root / f"{part}.glb"
        crop = spec.get("raw_aabb")
        if crop is None:
            shutil.copy2(source, destination)
            rows[part] = {
                "method": "copy",
                "source": str(source),
                "output": str(destination),
                "output_sha256": sha256_file(destination),
            }
            continue

        mesh = load_flat_mesh(source)
        if len(mesh.faces) > int(config.get("maximum_faces", MAX_FACES)):
            raise RuntimeError(f"{part}: refusing dense crop of {len(mesh.faces)} faces")
        lower = np.asarray(crop["minimum"], dtype=np.float64)
        upper = np.asarray(crop["maximum"], dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError(f"{part}: invalid raw_aabb")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        face_centers = vertices[np.asarray(mesh.faces, dtype=np.int64)].mean(axis=1)
        keep = np.all((face_centers >= lower) & (face_centers <= upper), axis=1)
        if not np.any(keep):
            raise ValueError(f"{part}: crop rejected every face")
        source_faces = int(len(mesh.faces))
        source_bounds = np.asarray(mesh.bounds, dtype=float).tolist()
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        mesh.export(destination)
        rows[part] = {
            "method": "raw_aabb_face_centroid_crop",
            "source": str(source),
            "output": str(destination),
            "raw_aabb": crop,
            "source_faces": source_faces,
            "output_faces": int(len(mesh.faces)),
            "retained_face_fraction": float(len(mesh.faces) / source_faces),
            "source_bounds": source_bounds,
            "output_bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
            "output_sha256": sha256_file(destination),
        }

    write_json(report_path, {
        "schema_version": 1,
        "method": "bounded_visual_mesh_preparation",
        "config": str(config_path),
        "parts": rows,
    })
    print(f"simulation meshes -> {output_root}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
