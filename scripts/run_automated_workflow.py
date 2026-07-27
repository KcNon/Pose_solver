#!/usr/bin/env python
"""Run one auditable dataset workflow from videos through refined 6D pose.

The source stage configs remain reusable and immutable.  This runner validates
their contracts, wires actual upstream artifact paths into generated runtime
configs, and then invokes the existing stage CLIs as subprocesses.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.masking.schema import load_mask_pipeline_config


def _path(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _run(command: list[str]) -> None:
    print("[workflow] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _same(label: str, values: dict[str, list[str]]) -> list[str]:
    first_name, first = next(iter(values.items()))
    mismatched = {
        name: value for name, value in values.items() if value != first
    }
    if mismatched:
        raise ValueError(
            f"{label} differs across stage configs; "
            f"reference {first_name}={first}, mismatched={mismatched}"
        )
    return first


def _mask_parts(raw: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    configured = raw["parts"]
    if not isinstance(configured, dict):
        raise ValueError("workflow requires dictionary mask parts")
    names = list(configured)
    ids = {
        name: int(values.get("id", raw.get("part_ids", {}).get(name)))
        for name, values in configured.items()
    }
    return names, ids


def _resolved_mask_raw(mask_config_path: Path, mask_raw: dict[str, Any]) -> dict:
    work_root = Path(mask_raw["work_root"])
    resolved = work_root / "manifests" / "resolved_mask_config.json"
    return load_json(resolved) if resolved.exists() else mask_raw


def _part_starts(mask_raw: dict[str, Any]) -> dict[str, int]:
    result = {}
    for name, values in mask_raw["parts"].items():
        value = values.get("start_frame", 0)
        if isinstance(value, str) and value.lower() == "auto":
            raise RuntimeError(
                f"{name}: mask start is still automatic; run mask discovery "
                "before depth/pose"
            )
        result[name] = int(value)
    return result


def _runtime_configs(
    *,
    workflow: dict[str, Any],
    source_paths: dict[str, Path],
    sources: dict[str, dict[str, Any]],
    runtime_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    preprocess = sources["preprocess"]
    mask = _resolved_mask_raw(source_paths["mask"], sources["mask"])
    depth = deepcopy(sources["depth"])
    pose = deepcopy(sources["pose"])
    recon = sources["reconviagen"]

    views = _same("views", {
        "preprocess": list(preprocess["views"]),
        "mask": list(mask["views"]),
        "depth": list(depth["views"]),
        "pose": list(pose["views"]),
    })
    mask_parts, mask_ids = _mask_parts(mask)
    parts = _same("parts", {
        "mask": mask_parts,
        "reconviagen": list(recon["parts"]),
        "depth": list(depth["parts"]),
        "pose": list(pose["parts"]),
    })
    if mask_ids != {name: int(depth["part_ids"][name]) for name in parts}:
        raise ValueError("mask and depth part IDs differ")
    if mask_ids != {name: int(pose["part_ids"][name]) for name in parts}:
        raise ValueError("mask and pose part IDs differ")

    frames_dir = str(Path(preprocess["frames_dir"]).resolve())
    masks_dir = str((Path(mask["output_root"]) / "masks").resolve())
    mesh_dir = str(Path(recon["mesh_root"]).resolve())
    starts = _part_starts(mask)

    depth.update({
        "frames_dir": frames_dir,
        "masks_dir": masks_dir,
        "mesh_dir": mesh_dir,
        "views": views,
        "parts": parts,
        "part_ids": mask_ids,
        "part_start_frames": starts,
    })
    depth["reference_part"] = str(
        depth.get("reference_part")
        or pose.get("reference_part")
        or parts[0]
    )
    backend = str(depth["recon_backend"])
    tag = depth.get("point_cloud_tag")
    variant = backend if not tag else f"{backend}_{tag}"
    artifact_root = Path(
        depth.get("point_cloud_output_root", depth["output_root"])
    ).resolve()
    cloud_root = artifact_root / "parts_ply" / variant
    pose.update({
        "frames_dir": frames_dir,
        "masks_dir": masks_dir,
        "mesh_dir": mesh_dir,
        "views": views,
        "parts": parts,
        "part_ids": mask_ids,
        "part_start_frames": starts,
        "depth_gauge_path": depth["depth_gauge_path"],
        "point_cloud_variant": variant,
        "point_cloud_root": str(cloud_root),
    })

    runtime_dir = runtime_root / "configs"
    depth_path = runtime_dir / "depth.resolved.json"
    pose_path = runtime_dir / "pose.resolved.json"
    write_json(depth_path, depth)
    write_json(pose_path, pose)
    contract = {
        "views": views,
        "parts": parts,
        "part_ids": mask_ids,
        "part_start_frames": starts,
        "frames_dir": frames_dir,
        "masks_dir": masks_dir,
        "mesh_dir": mesh_dir,
        "depth_gauge_path": depth["depth_gauge_path"],
        "point_cloud_root": str(cloud_root),
        "pose_output_root": pose["output_root"],
        "source_configs": {
            name: str(path) for name, path in source_paths.items()
        },
        "runtime_configs": {
            "depth": str(depth_path),
            "pose": str(pose_path),
        },
    }
    write_json(runtime_root / "workflow_contract.json", contract)
    return depth_path, pose_path, contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "preflight",
            "frames",
            "mesh",
            "mask",
            "depth",
            "depth-postprocess",
            "pose",
        ),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    workflow_path = Path(args.config).resolve()
    workflow = load_json(workflow_path)
    base = workflow_path.parent
    names = ("preprocess", "reconviagen", "mask", "depth", "pose")
    source_paths = {
        name: _path(workflow["configs"][name], base)
        for name in names
    }
    sources = {name: load_json(path) for name, path in source_paths.items()}
    runtime_root = _path(
        workflow.get("runtime_root", "workflow_runtime"),
        base,
    )

    # Validate the static contract before launching any expensive model.
    mask_config = load_mask_pipeline_config(source_paths["mask"])
    if Path(sources["preprocess"]["frames_dir"]).resolve() != (
        mask_config.frames_dir.resolve()
    ):
        raise ValueError("preprocess frames_dir and mask frames_dir differ")
    for video in sources["preprocess"]["videos"].values():
        if not Path(video).is_file():
            raise FileNotFoundError(video)
    for video in sources["reconviagen"]["videos"].values():
        if not Path(video).is_file():
            raise FileNotFoundError(video)
    mask_parts, mask_ids = _mask_parts(sources["mask"])
    contract_views = _same("views", {
        "preprocess": list(sources["preprocess"]["views"]),
        "mask": list(sources["mask"]["views"]),
        "depth": list(sources["depth"]["views"]),
        "pose": list(sources["pose"]["views"]),
    })
    contract_parts = _same("parts", {
        "mask": mask_parts,
        "reconviagen": list(sources["reconviagen"]["parts"]),
        "depth": list(sources["depth"]["parts"]),
        "pose": list(sources["pose"]["parts"]),
    })
    for stage in ("depth", "pose"):
        ids = {
            part: int(sources[stage]["part_ids"][part])
            for part in contract_parts
        }
        if ids != mask_ids:
            raise ValueError(f"mask and {stage} part IDs differ")
    print(
        f"preflight: {len(contract_views)} views, "
        f"{len(contract_parts)} parts",
        flush=True,
    )
    if args.stage == "preflight":
        return

    python = sys.executable
    if args.stage in {"all", "frames"}:
        frame_manifest = (
            Path(sources["preprocess"]["frames_dir"]).resolve().parent
            / "frame_extraction.json"
        )
        if frame_manifest.exists() and not args.force:
            print(f"[resume] {frame_manifest}", flush=True)
        else:
            command = [
                python,
                "tools/stages/preprocess/extract_synchronized_video_frames.py",
                "--config",
                str(source_paths["preprocess"]),
            ]
            if args.force:
                command.append("--force")
            _run(command)
    if args.stage in {"all", "mesh"}:
        command = [
            python,
            "scripts/run_reconviagen_pipeline.py",
            "--config",
            str(source_paths["reconviagen"]),
            "--stage",
            "all",
        ]
        if args.force:
            command.append("--force")
        _run(command)
    if args.stage in {"all", "mask"}:
        command = [
            python,
            "scripts/run_mask_pipeline.py",
            "--config",
            str(source_paths["mask"]),
            "--stage",
            "all",
        ]
        if args.force:
            command.extend(["--force-qwen", "--force-sam"])
        _run(command)

    depth_path, pose_path, contract = _runtime_configs(
        workflow=workflow,
        source_paths=source_paths,
        sources=sources,
        runtime_root=runtime_root,
    )
    if args.stage in {"all", "depth", "depth-postprocess"}:
        command = [
            python,
            "scripts/run_depth_pipeline.py",
            "--config",
            str(depth_path),
            "--stage",
            "postprocess" if args.stage == "depth-postprocess" else "all",
        ]
        if args.force:
            command.append("--force")
        _run(command)
    if args.stage in {"all", "pose"}:
        command = [
            python,
            "scripts/run_pose_pipeline.py",
            "--config",
            str(pose_path),
            "--stage",
            "all",
        ]
        if args.force:
            command.append("--force")
        _run(command)

    write_json(runtime_root / "workflow.complete.json", {
        "config": str(workflow_path),
        "completed_stage": args.stage,
        "contract": contract,
    })
    print(
        f"workflow complete -> {runtime_root / 'workflow.complete.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
