#!/usr/bin/env python
"""Schedule fixed-camera body segmentation for independent views across GPUs."""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--python", default="/data_ft_9_10/wentai/projects/sam3/.venv/bin/python")
    parser.add_argument("--log-dir", default="outputs/normalized_body_all6_000000/logs")
    parser.add_argument("--init-timestamp", default="000000")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(args.views)
    free_gpus = list(args.gpus)
    running: dict[int, tuple[str, int, subprocess.Popen, object]] = {}
    failures: list[tuple[str, int]] = []

    while pending or running:
        while pending and free_gpus:
            view = pending.pop(0)
            gpu = free_gpus.pop(0)
            log_file = open(log_dir / f"{view}.log", "w", encoding="utf-8")
            command = [
                args.python, "-u", "scripts/seg_masks_body_reanchor.py",
                "--pipeline", args.pipeline,
                "--view", view,
                "--init-timestamp", args.init_timestamp,
                "--gpu", str(gpu),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(command, env=env, stdout=log_file,
                                       stderr=subprocess.STDOUT, text=True)
            running[process.pid] = (view, gpu, process, log_file)
            print(f"launched {view} on GPU {gpu}, pid={process.pid}", flush=True)

        finished = []
        for pid, (view, gpu, process, log_file) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_file.close()
            finished.append(pid)
            free_gpus.append(gpu)
            print(f"finished {view} on GPU {gpu}, exit={return_code}", flush=True)
            if return_code != 0:
                failures.append((view, return_code))
        for pid in finished:
            del running[pid]
        if pending or running:
            time.sleep(2)

    if failures:
        raise SystemExit(f"body segmentation failures: {failures}")
    print("all body views completed", flush=True)


if __name__ == "__main__":
    main()
