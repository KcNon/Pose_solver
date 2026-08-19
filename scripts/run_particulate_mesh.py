#!/usr/bin/env python3
"""Run Particulate inference with explicit, offline model checkpoints.

The upstream CLI always contacts Hugging Face and writes the PartField
checkpoint below the Particulate checkout.  This wrapper keeps model assets and
all predictions in the pose-solver experiment tree instead, while calling the
upstream inference and export implementation unchanged.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--particulate-root",
        type=Path,
        default=Path("/data_ft_9_10/wentai/projects/particulate"),
    )
    parser.add_argument("--model-ckpt", type=Path, required=True)
    parser.add_argument("--partfield-ckpt", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--up-dir",
        default="-Z",
        choices=("X", "Y", "Z", "-X", "-Y", "-Z"),
    )
    parser.add_argument(
        "--up-dirs",
        nargs="+",
        choices=("X", "Y", "Z", "-X", "-Y", "-Z"),
        help="Run a candidate sweep and create one subdirectory per direction/confidence.",
    )
    parser.add_argument("--num-points", type=int, default=102400)
    parser.add_argument("--min-part-confidence", type=float, default=0.0)
    parser.add_argument("--min-part-confidences", type=float, nargs="+")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--animation-frames", type=int, default=50)
    parser.add_argument("--export-urdf", action="store_true")
    parser.add_argument("--export-mjcf", action="store_true")
    parser.add_argument("--eval", action="store_true", default=True)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    particulate_root = args.particulate_root.expanduser().resolve()
    if not (particulate_root / "infer.py").is_file():
        raise FileNotFoundError(f"Not a Particulate checkout: {particulate_root}")

    input_mesh = require_file(args.input_mesh, "input mesh")
    model_ckpt = require_file(args.model_ckpt, "Particulate checkpoint")
    partfield_ckpt = require_file(args.partfield_ckpt, "PartField checkpoint")
    model_config = (
        args.model_config.expanduser().resolve()
        if args.model_config is not None
        else particulate_root / "configs" / "particulate-B.yaml"
    )
    require_file(model_config, "model config")

    # The upstream imports use checkout-relative top-level modules.
    sys.path.insert(0, str(particulate_root))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import torch
    from omegaconf import OmegaConf

    import infer as particulate_infer
    import partfield_utils

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Particulate inference")

    cfg = OmegaConf.load(model_config)
    model_size = cfg.get("model_size", "B")
    cfg.pop("model_size", None)
    model_class = getattr(particulate_infer, f"PAT_{model_size}")
    model = model_class(**cfg)
    model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
    model.eval().to("cuda")

    partfield_config = particulate_root / "PartField" / "configs" / "final" / "demo.yaml"
    partfield_cfg = partfield_utils.setup(
        argparse.Namespace(config_file=str(partfield_config), opts=[]),
        freeze=False,
    )
    partfield_model = partfield_utils.Model.load_from_checkpoint(
        str(partfield_ckpt),
        cfg=partfield_cfg,
    )
    partfield_model.eval().to(device="cuda")

    # prepare_inputs resolves this imported global for every inference. Reusing
    # the already loaded model avoids loading the 1.2 GB checkpoint repeatedly.
    particulate_infer.get_partfield_model = lambda device="cuda": partfield_model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    up_dirs = args.up_dirs or [args.up_dir]
    confidence_values = args.min_part_confidences or [args.min_part_confidence]
    sweep = args.up_dirs is not None or args.min_part_confidences is not None
    for up_dir in up_dirs:
        for confidence in confidence_values:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Part confidence must lie in [0, 1], got {confidence}")
            if sweep:
                direction_tag = up_dir.replace("-", "neg_").lower()
                confidence_tag = f"{confidence:.3f}".replace(".", "p")
                output_dir = args.output_dir / f"up_{direction_tag}_confidence_{confidence_tag}"
            else:
                output_dir = args.output_dir
            print(
                f"Running candidate up_dir={up_dir} "
                f"min_part_confidence={confidence:g} -> {output_dir}"
            )
            upstream_args = SimpleNamespace(
                up_dir=up_dir,
                num_points=args.num_points,
                min_part_confidence=confidence,
                no_strict=not args.strict,
                animation_frames=args.animation_frames,
                export_urdf=args.export_urdf,
                export_mjcf=args.export_mjcf,
                eval=args.eval,
            )
            particulate_infer.infer_single_mesh(
                input_mesh,
                output_dir.resolve(),
                model,
                upstream_args,
            )


if __name__ == "__main__":
    main()
