#!/usr/bin/env python3
"""Create a geometry-preserving proxy mesh suitable for Particulate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_flat_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh scene has no geometry: {path}")
        mesh = loaded.to_geometry()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Expected a triangle mesh, got {type(loaded).__name__}")
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Mesh is empty: {path}")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mesh", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=15000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    source_path = args.input_mesh.expanduser().resolve()
    if args.target_faces < 1000:
        raise ValueError("--target-faces must be at least 1000")
    source = load_flat_mesh(source_path)

    output_path = args.output_mesh.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".source.ply")
    source.export(temporary_path)

    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(temporary_path))
    mesh_set.meshing_decimation_quadric_edge_collapse(
        targetfacenum=args.target_faces,
        preservenormal=True,
        preservetopology=False,
        # ReconViaGen meshes contain sparse outlier surfaces.  Allowing the
        # quadric optimum to create new vertex positions can expand the bounds
        # noticeably; endpoint placement keeps the proxy in the source frame.
        optimalplacement=False,
        autoclean=True,
    )
    mesh_set.save_current_mesh(str(output_path))
    temporary_path.unlink(missing_ok=True)

    proxy = load_flat_mesh(output_path)
    if len(proxy.faces) > args.target_faces * 1.05:
        raise RuntimeError(
            f"Decimation missed target: {len(proxy.faces)} > {args.target_faces}"
        )
    source_bounds = np.asarray(source.bounds, dtype=np.float64)
    proxy_bounds = np.asarray(proxy.bounds, dtype=np.float64)
    diagonal = float(np.linalg.norm(source.extents))
    bounds_error = float(np.max(np.abs(proxy_bounds - source_bounds)))
    relative_bounds_error = bounds_error / max(diagonal, 1e-12)
    if relative_bounds_error > 0.01:
        raise RuntimeError(
            "Proxy bounds drifted too far from the source: "
            f"{relative_bounds_error:.4%} of the source diagonal"
        )
    report = {
        "schema_version": 1,
        "source_mesh": str(source_path),
        "source_sha256": sha256_file(source_path),
        "proxy_mesh": str(output_path),
        "proxy_sha256": sha256_file(output_path),
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "proxy_vertices": int(len(proxy.vertices)),
        "proxy_faces": int(len(proxy.faces)),
        "target_faces": int(args.target_faces),
        "source_bounds": source_bounds.tolist(),
        "proxy_bounds": proxy_bounds.tolist(),
        "bounds_max_error": bounds_error,
        "bounds_error_over_diagonal": relative_bounds_error,
        "coordinate_frame": "source_mesh",
    }
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output_path.with_suffix(".json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
