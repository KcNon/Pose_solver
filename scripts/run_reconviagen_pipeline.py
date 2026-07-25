#!/usr/bin/env python
"""Run object-video extraction, Qwen/SAM masks, RGBA prep, and ReconViaGen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "reconviagen_objects.json"
STAGES = ROOT / "tools" / "stages" / "reconviagen"


def _run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _ready(paths: list[Path], force: bool) -> bool:
    if force or not paths or not all(path.exists() for path in paths):
        return False
    print(f"[resume] {len(paths)} expected outputs already exist", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--stage",
        choices=("all", "frames", "masks", "rgba", "mesh"),
        default="all",
    )
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--qwen-gpu", type=int, default=0)
    parser.add_argument("--sam-gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parts = args.parts or list(config["parts"])
    unknown = set(parts).difference(config["parts"])
    if unknown:
        raise ValueError(f"unknown parts: {sorted(unknown)}")
    output_root = Path(config["output_root"]).resolve()

    if args.stage in {"all", "frames"}:
        command = [
            sys.executable,
            str(STAGES / "extract_object_video_frames.py"),
            "--config",
            str(config_path),
        ]
        if args.force:
            command.append("--force")
        frame_outputs = [
            output_root / part / "frames" / part / "000000.jpg"
            for part in config["videos"]
        ]
        if not _ready(frame_outputs, args.force):
            _run(command)

    if args.stage in {"all", "masks"}:
        command = [
            sys.executable,
            str(STAGES / "run_object_masks.py"),
            "--config",
            str(config_path),
            "--parts",
            *parts,
            "--qwen-gpu",
            str(args.qwen_gpu),
            "--sam-gpu",
            str(args.sam_gpu),
        ]
        if args.force:
            command.extend(["--force-qwen", "--force-sam"])
        mask_outputs = [
            output_root / part / "masks" / timestamp / f"{part}.png"
            for part in parts
            for timestamp in config["parts"][part]["mask_timestamps"]
        ]
        if not _ready(mask_outputs, args.force):
            _run(command)

    if args.stage in {"all", "rgba"}:
        rgba_root = Path(config["rgba_root"]).resolve()
        rgba_outputs = [
            rgba_root / part / f"{timestamp}.png"
            for part in parts
            for timestamp in config["parts"][part]["rgba_timestamps"]
        ]
        if not _ready(rgba_outputs, args.force):
            _run([
                sys.executable,
                str(STAGES / "prepare_rgba.py"),
                "--config",
                str(config_path),
                "--parts",
                *parts,
            ])

    if args.stage in {"all", "mesh"}:
        mesh_root = Path(config["mesh_root"]).resolve()
        mesh_outputs = [mesh_root / f"{part}.glb" for part in parts]
        if not _ready(mesh_outputs, args.force):
            _run([
                str(config["recon_python"]),
                str(STAGES / "reconstruct_meshes.py"),
                "--config",
                str(config_path),
                "--parts",
                *parts,
            ])


if __name__ == "__main__":
    main()
