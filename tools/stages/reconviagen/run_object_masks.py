#!/usr/bin/env python
"""Run Qwen seed detection and SAM segmentation for ReconViaGen videos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "reconviagen_objects.json"


def _runtime_config(config: dict[str, Any], part: str) -> dict[str, Any]:
    root = Path(config["output_root"]).resolve() / part
    spec = config["parts"][part]
    return {
        "frames_dir": str(root / "frames"),
        "work_root": str(root / "mask_work"),
        "output_root": str(root),
        "masks_dir": str(root / "masks"),
        "bbox_json": str(root / "mask_work" / "bboxes" / "bbox.json"),
        "views": [part],
        "parts": {
            part: {
                "id": int(spec["id"]),
                "color": spec.get("color"),
                "start_frame": 0,
                "prompts": spec["prompts"],
            }
        },
        "occlusion_order": [part],
        "qwen_model": config["qwen_model"],
        "sam_ckpt": config["sam_ckpt"],
    }


def _run(command: list[str], gpu: int) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--qwen-gpu", type=int, default=0)
    parser.add_argument("--sam-gpu", type=int, default=0)
    parser.add_argument("--force-qwen", action="store_true")
    parser.add_argument("--force-sam", action="store_true")
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parts = args.parts or list(config["parts"])
    unknown = set(parts).difference(config["parts"])
    if unknown:
        raise ValueError(f"unknown parts: {sorted(unknown)}")

    runtime_root = Path(config["output_root"]).resolve() / "runtime_configs"
    runtime_root.mkdir(parents=True, exist_ok=True)
    qwen_script = (
        ROOT / "tools" / "stages" / "masking" / "detect_mask_seeds.py"
    )
    sam_script = (
        ROOT / "tools" / "stages" / "masking" / "track_part_masks.py"
    )
    for part in parts:
        timestamps = [str(value) for value in config["parts"][part]["mask_timestamps"]]
        runtime_path = runtime_root / f"{part}.json"
        runtime_path.write_text(
            json.dumps(_runtime_config(config, part), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        qwen_command = [
            str(config["qwen_python"]),
            "-u",
            str(qwen_script),
            "--config",
            str(runtime_path),
            "--timestamps",
            *timestamps,
            "--views",
            part,
            "--parts",
            part,
        ]
        if not args.no_vis:
            qwen_command.append("--vis")
        if args.force_qwen:
            qwen_command.append("--force")
        _run(qwen_command, args.qwen_gpu)

        sam_command = [
            str(config["sam_python"]),
            "-u",
            str(sam_script),
            "--config",
            str(runtime_path),
            "--mode",
            "image",
            "--part",
            part,
            "--timestamps",
            *timestamps,
            "--views",
            part,
            "--gpu",
            str(args.sam_gpu),
            "--legacy-palette-output",
        ]
        if not args.force_sam:
            sam_command.append("--skip-existing")
        _run(sam_command, args.sam_gpu)
    print(f"object masks -> {Path(config['output_root']).resolve()}", flush=True)


if __name__ == "__main__":
    main()
