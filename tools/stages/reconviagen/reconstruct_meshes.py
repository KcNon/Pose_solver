#!/usr/bin/env python
"""Batch ReconViaGen/TRELLIS.2 reconstruction from prepared RGBA views."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RECON_ROOT = Path("/data_ft_9_10/wentai/projects/ReconViaGen")
TRELLIS2_ROOT = RECON_ROOT / "wheels" / "TRELLIS.2"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRELLIS2_ROOT))
sys.path.insert(0, str(RECON_ROOT))

import o_voxel
import torch
from PIL import Image
from trellis.pipelines import TrellisVGGTTo3DPipeline
from trellis.pipelines.trellis_hybrid_pipeline import TrellisHybridPipeline
from trellis2.pipelines import Trellis2ImageTo3DPipeline


SS_PARAMS = {
    "steps": 12,
    "cfg_strength": 7.5,
    "cfg_interval": [0.6, 1.0],
    "guidance_rescale": 0.7,
    "rescale_t": 5.0,
}
SLAT_PARAMS = {
    "steps": 12,
    "cfg_strength": 7.5,
    "cfg_interval": [0.6, 1.0],
    "guidance_rescale": 0.5,
    "rescale_t": 3.0,
}
SHAPE_PARAMS = {
    "steps": 12,
    "guidance_strength": 7.5,
    "guidance_rescale": 0.5,
    "rescale_t": 3.0,
}
TEXTURE_PARAMS = {
    "steps": 12,
    "guidance_strength": 1.0,
    "guidance_rescale": 0.0,
    "rescale_t": 3.0,
}


def load_pipeline() -> TrellisHybridPipeline:
    print("[1/2] loading ReconViaGen sparse-structure pipeline", flush=True)
    vggt = TrellisVGGTTo3DPipeline.from_pretrained("Stable-X/trellis-vggt-v0-2")
    vggt.cuda()
    vggt.VGGT_model.cuda()
    vggt.birefnet_model.cuda()
    if "slat_decoder_gs" in vggt.models:
        del vggt.models["slat_decoder_gs"]
    vggt.VGGT_model.cpu()
    for model in vggt.models.values():
        model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    print("[2/2] loading TRELLIS.2 shape/texture pipeline", flush=True)
    trellis2 = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    trellis2.cuda()
    trellis2.low_vram = True
    gc.collect()
    torch.cuda.empty_cache()
    return TrellisHybridPipeline(vggt, trellis2, low_vram=True)


def export_glb(pipeline: TrellisHybridPipeline, mesh, path: Path,
               decimation_target: int, texture_size: int) -> None:
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=int(round(1.0 / float(mesh.voxel_size))),
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(path, extension_webp=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "reconviagen_objects.json"),
    )
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--strategy")
    parser.add_argument("--pipeline-type")
    parser.add_argument("--ss-source")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--decimation-target", type=int)
    parser.add_argument("--texture-size", type=int)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    reconstruction = config.get("reconstruction", {})
    parts = args.parts or list(config["parts"])
    strategy = args.strategy or reconstruction.get(
        "strategy", "adaptive_guidance_weight"
    )
    pipeline_type = args.pipeline_type or reconstruction.get(
        "pipeline_type", "1024_cascade"
    )
    ss_source = args.ss_source or reconstruction.get("ss_source", "mesh")
    seed = args.seed if args.seed is not None else int(reconstruction.get("seed", 0))
    decimation_target = (
        args.decimation_target
        if args.decimation_target is not None
        else int(reconstruction.get("decimation_target", 500_000))
    )
    texture_size = (
        args.texture_size
        if args.texture_size is not None
        else int(reconstruction.get("texture_size", 2048))
    )
    pipeline = load_pipeline()
    input_root = Path(args.input_root or config["rgba_root"]).resolve()
    output_root = Path(args.output_root or config["mesh_root"]).resolve()
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "engine": "ReconViaGen_x_TRELLIS2_hybrid",
            "strategy": strategy,
            "pipeline_type": pipeline_type,
            "ss_source": ss_source,
            "seed": seed,
            "parts": {},
        }
    for part in parts:
        paths = sorted((input_root / part).glob("*.png"))
        if not paths:
            raise RuntimeError(f"no RGBA inputs found for {part}")
        images = [Image.open(path).convert("RGBA") for path in paths]
        print(f"\n{part}: reconstructing from {len(images)} RGBA views", flush=True)
        meshes = pipeline.run_multi_image(
            images,
            strategy=strategy,
            seed=seed,
            ss_sampler_params=SS_PARAMS,
            slat_sampler_params=SLAT_PARAMS,
            shape_slat_sampler_params=SHAPE_PARAMS,
            tex_slat_sampler_params=TEXTURE_PARAMS,
            pipeline_type=pipeline_type,
            preprocess_image=True,
            return_latent=False,
            ss_source=ss_source,
        )
        mesh = meshes[0]
        destination = output_root / f"{part}.glb"
        export_glb(
            pipeline, mesh, destination,
            decimation_target, texture_size,
        )
        manifest["parts"][part] = {
            "input_count": len(paths),
            "inputs": [str(path) for path in paths],
            "mesh": str(destination),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        del images, meshes, mesh
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{part}: exported {destination}", flush=True)


if __name__ == "__main__":
    main()
