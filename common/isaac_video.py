"""Complete multi-view Isaac pose replay video generation.

This module is imported only after ``SimulationApp`` has been constructed.
It reuses the USD cache produced by :mod:`common.isaac_runtime` and keeps
rendering concerns separate from physics validation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from common.io_utils import write_json
from common.isaac_runtime import align_vectors, np_to_gf_matrix


PANE_SPECS = {
    "perspective": (640, 720),
    "top": (640, 360),
    "side": (640, 360),
}


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


FONT_SMALL = _font(18)
FONT_MEDIUM = _font(24)
FONT_LARGE = _font(30)


def _load_usd_cache(runtime_root: Path, parts: list[str]) -> dict[str, Path]:
    cache_path = runtime_root / "usd/import_cache.json"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Isaac USD cache not found: {cache_path}. "
            "Run scripts/run_isaac_insertion.py first."
        )
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    paths = {
        part: Path(cache["usd_paths"][part]).resolve()
        for part in parts
        if part in cache.get("usd_paths", {})
    }
    missing = [part for part in parts if part not in paths or not paths[part].is_file()]
    if missing:
        raise RuntimeError(f"USD cache is missing parts: {missing}")
    return paths


def _bind_floor_material(stage: Usd.Stage, prim: Usd.Prim) -> None:
    material = UsdShade.Material.Define(stage, "/World/Materials/Floor")
    shader = UsdShade.Shader.Define(stage, "/World/Materials/Floor/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.16, 0.18, 0.22)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _create_environment(stage: Usd.Stage) -> None:
    UsdGeom.Scope.Define(stage, "/World/Materials")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr().Set(450.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Distant")
    distant.CreateIntensityAttr().Set(1300.0)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-35.0, 25.0, 20.0)
    )
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr().Set(2.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.10))
    floor.AddScaleOp().Set(Gf.Vec3d(0.8, 0.8, 0.01))
    _bind_floor_material(stage, floor.GetPrim())


def _add_asset_reference(
    stage: Usd.Stage,
    part: str,
    asset_path: Path,
) -> dict[str, Any]:
    root = UsdGeom.Xform.Define(stage, f"/World/ReplayAssets/{part}")
    root.GetPrim().GetReferences().AddReference(str(asset_path))
    transform_op = root.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    return {"root": root.GetPrim(), "transform_op": transform_op}


def _set_visibility(prim: Usd.Prim, visible: bool) -> None:
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def _is_observable(state: str) -> bool:
    return state not in {
        "inferred_unobservable",
        "unobservable",
        "out_of_frame",
        "unknown",
    }


def _part_world_transform(
    frame_record: dict[str, Any],
    part: str,
    reference_part: str,
    world_from_body: np.ndarray,
) -> np.ndarray:
    if part == reference_part:
        return world_from_body.copy()
    record = frame_record["parts"][part]
    if "T_body_from_part" in record:
        relative = np.asarray(record["T_body_from_part"], dtype=np.float64)
    else:
        body_world = np.asarray(
            frame_record["parts"][reference_part]["T_world_from_part"],
            dtype=np.float64,
        )
        part_world = np.asarray(record["T_world_from_part"], dtype=np.float64)
        relative = np.linalg.inv(body_world) @ part_world
    return world_from_body @ relative


def _to_image(data: Any) -> Image.Image:
    array = np.asarray(data)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array, mode="RGB")


class MultiViewCapture:
    def __init__(self, app: Any, rt_subframes: int) -> None:
        self.app = app
        self.rt_subframes = rt_subframes
        rep.orchestrator.set_capture_on_play(False)
        cameras = {
            "perspective": rep.functional.create.camera(
                position=(0.48, -0.48, 0.36),
                look_at=(0.0, 0.0, 0.02),
                clipping_range=(0.01, 10.0),
                parent="/World",
                name="CompletePerspectiveCamera",
            ),
            "top": rep.functional.create.camera(
                position=(0.0, 0.0, 0.72),
                look_at=(0.0, 0.0, 0.0),
                look_at_up_axis=(0.0, 1.0, 0.0),
                clipping_range=(0.01, 10.0),
                parent="/World",
                name="CompleteTopCamera",
            ),
            "side": rep.functional.create.camera(
                position=(0.58, 0.02, 0.20),
                look_at=(0.0, 0.0, 0.02),
                clipping_range=(0.01, 10.0),
                parent="/World",
                name="CompleteSideCamera",
            ),
        }
        self.render_products: dict[str, Any] = {}
        self.annotators: dict[str, Any] = {}
        for name, camera in cameras.items():
            product = rep.create.render_product(
                camera,
                PANE_SPECS[name],
                name=f"Complete{name.title()}RenderProduct",
                force_new=True,
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(product)
            self.render_products[name] = product
            self.annotators[name] = annotator
        for _ in range(60):
            self.app.update()

    def capture(self) -> dict[str, Image.Image]:
        async def capture_async() -> None:
            await rep.orchestrator.step_async(
                rt_subframes=self.rt_subframes,
                delta_time=0.0,
            )

        task = asyncio.ensure_future(capture_async())
        for _ in range(3600):
            if task.done():
                break
            self.app.update()
        if not task.done():
            task.cancel()
            raise TimeoutError("Isaac Replicator capture timed out")
        task.result()
        return {
            name: _to_image(annotator.get_data())
            for name, annotator in self.annotators.items()
        }

    def close(self) -> None:
        for annotator in self.annotators.values():
            annotator.detach()
        for product in self.render_products.values():
            product.destroy()


def _compose_views(
    views: dict[str, Image.Image],
    frame_id: int,
    states: dict[str, str],
    visible_parts: list[str],
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), (22, 22, 24))
    canvas.paste(views["perspective"], (0, 0))
    canvas.paste(views["top"], (640, 0))
    canvas.paste(views["side"], (640, 360))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, 1280, 60), fill=(0, 0, 0, 190))
    draw.text(
        (18, 13),
        f"Complete Isaac pose replay | frame {frame_id:03d}",
        font=FONT_LARGE,
        fill=(255, 255, 255, 255),
    )
    lines = [
        "All timeline frames are rendered in Isaac Sim; collision dynamics are reported separately.",
        "visible: " + (", ".join(visible_parts) if visible_parts else "none"),
        " | ".join(f"{part}: {state}" for part, state in states.items()),
    ]
    y = 64
    for line in lines:
        width = min(1240, 20 + int(draw.textlength(line, font=FONT_SMALL)))
        draw.rectangle((8, y, 8 + width, y + 27), fill=(0, 0, 0, 150))
        draw.text((16, y + 3), line, font=FONT_SMALL, fill=(245, 245, 245, 255))
        y += 29
    draw.rectangle((2, 2, 1277, 717), outline=(45, 145, 225, 255), width=5)
    draw.text((654, 328), "TOP", font=FONT_MEDIUM, fill=(255, 255, 255, 230))
    draw.text((654, 688), "SIDE", font=FONT_MEDIUM, fill=(255, 255, 255, 230))
    return canvas


def _encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:g}",
            "-i",
            str(frame_dir / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def render_complete_pose_video(
    args: argparse.Namespace,
    app: Any,
    asset_root: Path,
    runtime_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / "frames/complete_pose"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    parts = list(trajectory["parts"])
    reference_part = str(manifest["reference_part"])
    usd_paths = _load_usd_cache(runtime_root, parts)
    simulation = manifest["simulation"]
    world_from_body = np.eye(4, dtype=np.float64)
    world_from_body[:3, :3] = align_vectors(
        np.asarray(simulation["up_axis_body"], dtype=np.float64),
        np.array([0.0, 0.0, 1.0]),
    )

    stage = omni.usd.get_context().new_stage()
    if stage is None:
        stage = omni.usd.get_context().get_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/ReplayAssets")
    _create_environment(stage)
    assets = {
        part: _add_asset_reference(stage, part, usd_paths[part])
        for part in parts
    }
    for _ in range(20):
        app.update()

    capture = MultiViewCapture(app, int(args.rt_subframes))
    trajectory_ids = sorted(int(frame_id) for frame_id in trajectory["frames"])
    start_frame = (
        int(args.start_frame)
        if args.start_frame is not None
        else 0
    )
    end_frame = (
        int(args.end_frame)
        if args.end_frame is not None
        else trajectory_ids[-1]
    )
    frame_ids = list(range(start_frame, end_frame + 1))
    rendered: list[dict[str, Any]] = []
    try:
        for output_index, frame_id in enumerate(frame_ids):
            key = f"{frame_id:06d}"
            frame_record = trajectory["frames"].get(key)
            states: dict[str, str] = {}
            visible_parts: list[str] = []
            for part in parts:
                if frame_record is None:
                    state = "not_started"
                    visible = False
                else:
                    state = str(
                        frame_record["parts"][part].get("state", "unknown")
                    )
                    visible = (
                        part == reference_part
                        or _is_observable(state)
                    )
                    transform = _part_world_transform(
                        frame_record,
                        part,
                        reference_part,
                        world_from_body,
                    )
                    assets[part]["transform_op"].Set(np_to_gf_matrix(transform))
                states[part] = state
                _set_visibility(assets[part]["root"], visible)
                if visible:
                    visible_parts.append(part)

            views = capture.capture()
            image = _compose_views(views, frame_id, states, visible_parts)
            image.save(frame_dir / f"{output_index:06d}.jpg", quality=94)
            rendered.append(
                {
                    "frame_id": frame_id,
                    "trajectory_frame": frame_record is not None,
                    "states": states,
                    "visible_parts": visible_parts,
                }
            )
            if output_index % 10 == 0 or output_index + 1 == len(frame_ids):
                print(
                    f"Isaac video frame {output_index + 1}/{len(frame_ids)} "
                    f"(trajectory frame {frame_id:03d})",
                    flush=True,
                )
    finally:
        capture.close()

    video_path = output_root / "complete_isaac_pose_replay.mp4"
    _encode_video(frame_dir, video_path, float(args.fps))
    scene_path = output_root / "complete_isaac_pose_replay.usda"
    stage.GetRootLayer().Export(str(scene_path))
    report = {
        "schema_version": 1,
        "status": "complete",
        "mode": "complete kinematic pose replay rendered in Isaac Sim",
        "asset_root": str(asset_root),
        "runtime_root": str(runtime_root),
        "trajectory": str(
            (asset_root.parents[2] / manifest["inputs"]["trajectory"]).resolve()
        ),
        "output_video": str(video_path),
        "scene_usd": str(scene_path),
        "fps": float(args.fps),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": len(frame_ids),
        "duration_s": len(frame_ids) / float(args.fps),
        "rendered_frames": rendered,
        "physics_validation_report": str(
            runtime_root / manifest["outputs"]["isaac_report"]
        ),
        "interpretation": (
            "This video is the complete pose-solver trajectory replayed in "
            "Isaac Sim. Dynamic insertion acceptance remains documented by "
            "the separate Isaac insertion report."
        ),
    }
    write_json(output_root / "complete_isaac_video_report.json", report)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    return report
