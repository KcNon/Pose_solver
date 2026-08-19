#!/usr/bin/env python3
"""Transfer proxy Particulate labels to a source mesh and export named parts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh


PALETTE = np.asarray(
    [[255, 59, 48, 255], [52, 199, 89, 255], [0, 122, 255, 255], [255, 204, 0, 255]],
    dtype=np.uint8,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_flat_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Expected a non-empty triangle mesh: {path}")
    return mesh


def parse_part_spec(value: str) -> tuple[str, set[int]]:
    try:
        name, raw_ids = value.split("=", 1)
        ids = {int(item) for item in raw_ids.split(",")}
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Expected NAME=ID[,ID...]") from error
    if not name or not ids:
        raise argparse.ArgumentTypeError("Expected NAME=ID[,ID...]")
    return name, ids


def transfer_face_labels(
    source: trimesh.Trimesh,
    proxy: trimesh.Trimesh,
    proxy_labels: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    proxy_centers = np.asarray(proxy.triangles_center, dtype=np.float64)
    proxy_normals = np.asarray(proxy.face_normals, dtype=np.float64)
    source_centers = np.asarray(source.triangles_center, dtype=np.float64)
    source_normals = np.asarray(source.face_normals, dtype=np.float64)
    tree = cKDTree(proxy_centers)
    distances, indices = tree.query(source_centers, k=neighbors, workers=-1)
    if neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    candidate_normals = proxy_normals[indices]
    normal_agreement = np.abs(np.einsum("nij,nj->ni", candidate_normals, source_normals))
    diagonal = max(float(np.linalg.norm(source.extents)), 1e-12)
    scores = distances + 0.01 * diagonal * (1.0 - normal_agreement)
    selected_column = np.argmin(scores, axis=1)
    rows = np.arange(len(source.faces))
    selected_indices = indices[rows, selected_column]
    selected_distances = distances[rows, selected_column]
    return proxy_labels[selected_indices], selected_distances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part", action="append", type=parse_part_spec, required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    if args.neighbors < 1:
        raise ValueError("--neighbors must be positive")

    source_path = args.source_mesh.expanduser().resolve()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_flat_mesh(source_path)
    proxy_path = prediction_dir / "eval" / "pred.obj"
    proxy = load_flat_mesh(proxy_path)
    prediction = np.load(prediction_dir / "eval" / "pred.npz")
    proxy_labels = np.asarray(prediction["face_part_ids"], dtype=np.int64)
    if len(proxy_labels) != len(proxy.faces):
        raise ValueError("Proxy prediction labels do not match pred.obj faces")

    source_labels, transfer_distances = transfer_face_labels(
        source, proxy, proxy_labels, args.neighbors
    )
    requested_ids = set().union(*(part_ids for _, part_ids in args.part))
    unknown_ids = requested_ids - set(np.unique(proxy_labels).tolist())
    if unknown_ids:
        raise ValueError(f"Requested Particulate labels are absent: {sorted(unknown_ids)}")

    assignment = np.full(len(source.faces), -1, dtype=np.int16)
    part_reports = {}
    total_area = float(source.area)
    for output_id, (name, part_ids) in enumerate(args.part, start=1):
        mask = np.isin(source_labels, list(part_ids))
        if not np.any(mask):
            raise RuntimeError(f"No source faces mapped to part {name!r}")
        overlap = assignment[mask] >= 0
        if np.any(overlap):
            raise ValueError(f"Part {name!r} overlaps a previous part specification")
        assignment[mask] = output_id
        part_mesh = source.submesh([mask], append=True, repair=False)
        part_dir = output_dir / name
        part_dir.mkdir(parents=True, exist_ok=True)
        output_path = part_dir / "model.glb"
        part_mesh.export(output_path)
        part_reports[name] = {
            "id": output_id,
            "particulate_labels": sorted(part_ids),
            "faces": int(mask.sum()),
            "face_fraction": float(mask.mean()),
            "surface_area_fraction": float(source.area_faces[mask].sum() / total_area),
            "bounds": np.asarray(part_mesh.bounds, dtype=np.float64).tolist(),
            "mesh": str(output_path),
            "mesh_sha256": sha256_file(output_path),
        }
    if np.any(assignment < 0):
        missing = sorted(set(np.unique(source_labels[assignment < 0]).tolist()))
        raise ValueError(f"Some labels were not assigned to an output part: {missing}")

    np.savez_compressed(
        output_dir / "source_face_labels.npz",
        source_part_ids=assignment,
        source_particulate_labels=source_labels,
        transfer_distances=transfer_distances,
    )
    review = source.copy()
    review.visual.face_colors = PALETTE[(assignment - 1) % len(PALETTE)]
    review.export(output_dir / "parts_colored.glb")

    diagonal = max(float(np.linalg.norm(source.extents)), 1e-12)
    report = {
        "schema_version": 1,
        "source_mesh": str(source_path),
        "source_sha256": sha256_file(source_path),
        "prediction_dir": str(prediction_dir),
        "proxy_mesh": str(proxy_path),
        "source_faces": int(len(source.faces)),
        "proxy_faces": int(len(proxy.faces)),
        "neighbors": int(args.neighbors),
        "transfer_distance_quantiles": {
            str(q): float(np.quantile(transfer_distances, q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "transfer_distance_p95_over_diagonal": float(
            np.quantile(transfer_distances, 0.95) / diagonal
        ),
        "parts": part_reports,
    }
    (output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
