#!/usr/bin/env python3
"""Generate Isaac Sim pose-replay, collision-audit, and physics videos.

Run with Isaac Sim's ``python.sh``. The script reuses the URDF-import cache
created by ``run_isaac_insertion.py`` and produces four MP4 visualizations:

1. complete three-part solved-pose replay;
2. the same replay annotated with PhysX contact/penetration diagnostics;
3. several dynamic inner-pot insertion trials;
4. a full inner-pot + lid assembly release.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "experiments/rice_cooker_simulation_assets"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "experiments/rice_cooker_isaac_runtime"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments/rice_cooker_isaac_visualizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--physics-seconds", type=float, default=2.5)
    parser.add_argument("--pose-stride", type=int, default=1)
    parser.add_argument("--max-pose-frames", type=int, default=None)
    parser.add_argument("--trial-limit", type=int, default=3)
    parser.add_argument("--rt-subframes", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--trajectory-fps", type=float, default=15.0)
    parser.add_argument("--controller-force-scale", type=float, default=1.0)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
ASSET_ROOT = ARGS.asset_root.resolve()
RUNTIME_ROOT = ARGS.runtime_root.resolve()
OUTPUT_ROOT = ARGS.output_dir.resolve()
MANIFEST = json.loads((ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))
TRAJECTORY_PATH = PROJECT_ROOT / MANIFEST["inputs"]["trajectory"]
TRAJECTORY = json.loads(TRAJECTORY_PATH.read_text(encoding="utf-8"))

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "width": 1280,
        "height": 720,
        "renderer": "RayTracedLighting",
        "sync_loads": True,
        "limit_cpu_threads": 16,
        "multi_gpu": False,
    }
)

# Omniverse imports must follow SimulationApp construction.
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


PART_COLORS = {
    "body": (0.67, 0.37, 0.18),
    "inner_pot": (0.72, 0.76, 0.82),
    "lid": (0.18, 0.35, 0.78),
}
STATE_COLORS = {
    "clear": (48, 190, 86),
    "contact": (245, 190, 35),
    "penetrating": (232, 65, 55),
    "unavailable": (150, 150, 150),
}
PANE_SPECS = {
    "perspective": (640, 720),
    "top": (640, 360),
    "side": (640, 360),
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def np_to_gf_matrix(matrix: np.ndarray) -> Gf.Matrix4d:
    values = np.asarray(matrix, dtype=np.float64)
    return Gf.Matrix4d(*values.T.reshape(-1).tolist())


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    c, s, one = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=np.float64,
    )


def align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, trial)
        return axis_angle_matrix(axis, math.pi)
    return axis_angle_matrix(cross, math.acos(dot))


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def load_usd_cache() -> dict[str, str]:
    cache_path = RUNTIME_ROOT / "usd/import_cache.json"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Isaac USD cache not found: {cache_path}. Run scripts/run_isaac_insertion.py first."
        )
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    paths = cache.get("usd_paths", {})
    required = {"body", "inner_pot", "lid"}
    if not required.issubset(paths) or not all(Path(paths[name]).is_file() for name in required):
        raise RuntimeError(f"Incomplete Isaac USD cache: {cache_path}")
    return paths


def add_reference(stage: Usd.Stage, prim_path: str, asset_path: str, transform: np.ndarray | None = None) -> Usd.Prim:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.GetPrim().GetReferences().AddReference(asset_path)
    if transform is not None:
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(np_to_gf_matrix(transform))
    return xform.GetPrim()


def deinstance(root: Usd.Prim, max_passes: int = 8) -> None:
    for _ in range(max_passes):
        instances = [prim for prim in Usd.PrimRange(root) if prim.IsValid() and prim.IsInstance()]
        if not instances:
            return
        for prim in instances:
            prim.SetInstanceable(False)
        simulation_app.update()


def choose_rigid(root: Usd.Prim, name: str) -> Usd.Prim:
    candidates = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not candidates:
        raise RuntimeError(f"No rigid body found below {root.GetPath()}")
    preferred = [prim for prim in candidates if name.lower() in prim.GetName().lower()]
    return preferred[0] if preferred else candidates[0]


def configure_collision(root: Usd.Prim, approximation: str) -> list[str]:
    paths: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(approximation)
        paths.append(str(prim.GetPath()))
    if not paths:
        raise RuntimeError(f"No colliders found below {root.GetPath()}")
    return paths


def remove_articulation(root: Usd.Prim) -> None:
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def make_static(rigid: Usd.Prim) -> None:
    if rigid.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        rigid.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
    if rigid.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid.RemoveAPI(UsdPhysics.RigidBodyAPI)


def hide_imported_visuals(root: Usd.Prim) -> None:
    for prim in Usd.PrimRange(root):
        if "visual" in prim.GetName().lower():
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()


def read_obj(path: Path) -> tuple[list[Gf.Vec3f], list[int], list[int], list[Gf.Vec2f]]:
    vertices: list[Gf.Vec3f] = []
    texcoords: list[Gf.Vec2f] = []
    counts: list[int] = []
    indices: list[int] = []
    face_uvs: list[Gf.Vec2f] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                x, y, z = (float(value) for value in line.split()[1:4])
                vertices.append(Gf.Vec3f(x, y, z))
            elif line.startswith("vt "):
                u, v = (float(value) for value in line.split()[1:3])
                texcoords.append(Gf.Vec2f(u, v))
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    continue
                counts.append(len(tokens))
                for token in tokens:
                    fields = token.split("/")
                    vertex_index = int(fields[0])
                    if vertex_index < 0:
                        vertex_index = len(vertices) + vertex_index
                    else:
                        vertex_index -= 1
                    indices.append(vertex_index)
                    if len(fields) > 1 and fields[1]:
                        uv_index = int(fields[1])
                        if uv_index < 0:
                            uv_index = len(texcoords) + uv_index
                        else:
                            uv_index -= 1
                        face_uvs.append(texcoords[uv_index])
                    else:
                        face_uvs.append(Gf.Vec2f(0.0, 0.0))
    if not vertices or not counts:
        raise RuntimeError(f"OBJ contains no renderable geometry: {path}")
    return vertices, counts, indices, face_uvs


def bind_texture_material(stage: Usd.Stage, mesh: Usd.Prim, path: str, texture_path: Path) -> None:
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    texture = UsdShade.Shader.Define(stage, f"{path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, f"{path}/Primvar")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)


def bind_solid_material(
    stage: Usd.Stage,
    prim: Usd.Prim,
    path: str,
    color: tuple[float, float, float],
    metallic: float = 0.0,
    opacity: float = 1.0,
) -> UsdShade.Shader:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return shader


def set_visibility(prim: Usd.Prim, visible: bool) -> None:
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def author_axis(
    stage: Usd.Stage,
    parent_path: str,
    name: str,
    direction: np.ndarray,
    color: tuple[float, float, float],
    length: float = 0.09,
    radius: float = 0.003,
) -> None:
    direction = np.asarray(direction, dtype=np.float64)
    align = align_vectors(np.array([0.0, 0.0, 1.0]), direction)
    shaft = UsdGeom.Cylinder.Define(stage, f"{parent_path}/{name}Shaft")
    shaft.CreateRadiusAttr(radius)
    shaft.CreateHeightAttr(length * 0.76)
    shaft_tf = np.eye(4)
    shaft_tf[:3, :3] = align
    shaft_tf[:3, 3] = direction * length * 0.38
    shaft.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(np_to_gf_matrix(shaft_tf))
    bind_solid_material(stage, shaft.GetPrim(), f"/World/Materials/Axis{name}", color, metallic=0.05)

    head = UsdGeom.Cone.Define(stage, f"{parent_path}/{name}Head")
    head.CreateRadiusAttr(radius * 2.2)
    head.CreateHeightAttr(length * 0.24)
    head_tf = np.eye(4)
    head_tf[:3, :3] = align
    head_tf[:3, 3] = direction * length * 0.88
    head.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(np_to_gf_matrix(head_tf))
    bind_solid_material(stage, head.GetPrim(), f"/World/Materials/Axis{name}", color, metallic=0.05)


def author_part_visual(
    stage: Usd.Stage,
    part: str,
    *,
    scope: str = "/World/ReplayVisuals",
    material_suffix: str = "",
    debug_opacity: float | None = None,
) -> dict[str, Any]:
    root = UsdGeom.Xform.Define(stage, f"{scope}/{part}")
    transform_op = root.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    obj_path = ASSET_ROOT / f"meshes/visual/{part}/{part}.obj"
    texture_path = ASSET_ROOT / f"meshes/visual/{part}/material_0.png"
    vertices, counts, indices, face_uvs = read_obj(obj_path)

    textured = UsdGeom.Mesh.Define(stage, f"{scope}/{part}/Textured")
    textured.CreatePointsAttr(vertices)
    textured.CreateFaceVertexCountsAttr(counts)
    textured.CreateFaceVertexIndicesAttr(indices)
    textured.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    textured.CreateDoubleSidedAttr().Set(True)
    primvar = UsdGeom.PrimvarsAPI(textured.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set(face_uvs)
    bind_texture_material(
        stage,
        textured.GetPrim(),
        f"/World/Materials/{part}Texture{material_suffix}",
        texture_path,
    )

    debug = UsdGeom.Mesh.Define(stage, f"{scope}/{part}/Debug")
    debug.CreatePointsAttr(vertices)
    debug.CreateFaceVertexCountsAttr(counts)
    debug.CreateFaceVertexIndicesAttr(indices)
    debug.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    debug.CreateDoubleSidedAttr().Set(True)
    shader = bind_solid_material(
        stage,
        debug.GetPrim(),
        f"/World/Materials/{part}Debug{material_suffix}",
        PART_COLORS[part],
        metallic=0.55 if part != "body" else 0.05,
        opacity=debug_opacity if debug_opacity is not None else (0.72 if part == "body" else 1.0),
    )
    set_visibility(debug.GetPrim(), False)

    axes = UsdGeom.Xform.Define(stage, f"{scope}/{part}/Axes")
    author_axis(stage, str(axes.GetPath()), "X", np.array([1.0, 0.0, 0.0]), (0.9, 0.05, 0.05))
    author_axis(stage, str(axes.GetPath()), "Y", np.array([0.0, 1.0, 0.0]), (0.05, 0.85, 0.1))
    author_axis(stage, str(axes.GetPath()), "Z", np.array([0.0, 0.0, 1.0]), (0.05, 0.2, 0.95))
    return {
        "root": root.GetPrim(),
        "transform_op": transform_op,
        "textured": textured.GetPrim(),
        "debug": debug.GetPrim(),
        "debug_shader": shader,
        "axes": axes.GetPrim(),
    }


def configure_physics_scene(stage: Usd.Stage) -> float:
    physics_hz = int(MANIFEST["simulation"]["physics_hz"])
    dt = 1.0 / physics_hz
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateTimeStepsPerSecondAttr().Set(physics_hz)
    physx.CreateEnableCCDAttr().Set(True)
    physx.CreateEnableStabilizationAttr().Set(True)
    physx.CreateSolverTypeAttr().Set("TGS")
    return dt


def set_pose(view: RigidPrim, transform: np.ndarray, zero_velocity: bool = True) -> None:
    view.set_world_poses(
        positions=[transform[:3, 3].tolist()],
        orientations=[matrix_to_quaternion_wxyz(transform[:3, :3]).tolist()],
    )
    if zero_velocity:
        view.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]])


def set_kinematic(prim: Usd.Prim, enabled: bool) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(enabled)


def rotation_error_vector_world(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.asarray(target, dtype=np.float64) @ np.asarray(actual, dtype=np.float64).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-7:
        return np.zeros(3, dtype=np.float64)
    skew = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    sine = math.sin(angle)
    if abs(sine) > 1e-5:
        axis = skew / (2.0 * sine)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(relative)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis /= max(np.linalg.norm(axis), 1e-12)
    return axis * angle


def clipped_vector(value: np.ndarray, maximum_norm: float) -> tuple[np.ndarray, bool]:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= maximum_norm or norm < 1e-12:
        return value, False
    return value * (maximum_norm / norm), True


def create_force_controller(prim: Usd.Prim) -> dict[str, Any]:
    api = PhysxSchema.PhysxForceAPI.Apply(prim)
    api.CreateForceEnabledAttr().Set(True)
    api.CreateWorldFrameEnabledAttr().Set(True)
    api.CreateModeAttr().Set("force")
    force_attr = api.CreateForceAttr()
    torque_attr = api.CreateTorqueAttr()
    force_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    torque_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    return {"api": api, "force_attr": force_attr, "torque_attr": torque_attr}


def apply_pose_controller(
    controller: dict[str, Any],
    *,
    actual: np.ndarray,
    target: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    mass: float,
    inertia_scale: float,
    force_limit: float,
    torque_limit: float,
) -> dict[str, Any]:
    position_error = target[:3, 3] - actual[:3, 3]
    rotation_error = rotation_error_vector_world(actual[:3, :3], target[:3, :3])
    natural_frequency = 16.0
    damping_ratio = 1.0
    kp_position = mass * natural_frequency**2
    kd_position = 2.0 * damping_ratio * mass * natural_frequency
    kp_rotation = inertia_scale * natural_frequency**2
    kd_rotation = 2.0 * damping_ratio * inertia_scale * natural_frequency
    force = (
        kp_position * position_error
        - kd_position * np.asarray(linear_velocity, dtype=np.float64)
        + np.array([0.0, 0.0, mass * 9.81], dtype=np.float64)
    )
    torque = (
        kp_rotation * rotation_error
        - kd_rotation * np.asarray(angular_velocity, dtype=np.float64)
    )
    force, force_saturated = clipped_vector(force, force_limit * ARGS.controller_force_scale)
    torque, torque_saturated = clipped_vector(torque, torque_limit * ARGS.controller_force_scale)
    controller["force_attr"].Set(Gf.Vec3f(*force.astype(float).tolist()))
    controller["torque_attr"].Set(Gf.Vec3f(*torque.astype(float).tolist()))
    return {
        "position_error_m": float(np.linalg.norm(position_error)),
        "rotation_error_deg": float(np.degrees(np.linalg.norm(rotation_error))),
        "force_world_n": force.tolist(),
        "torque_world_nm": torque.tolist(),
        "force_saturated": force_saturated,
        "torque_saturated": torque_saturated,
    }


def contact_snapshot(view: RigidPrim, dt: float) -> dict[str, Any]:
    try:
        forces, _points, _normals, separations, counts, starts, _actor_ids = view.get_raw_contact_data(dt=dt)
        count_values = counts.numpy().reshape(-1)
        count = int(count_values[0]) if len(count_values) else 0
        start_values = starts.numpy().reshape(-1)
        start = int(start_values[0]) if len(start_values) else 0
        if count <= 0:
            return {"available": True, "count": 0, "max_force_n": 0.0, "max_penetration_m": 0.0}
        force_values = forces.numpy().reshape(-1)[start : start + count]
        separation_values = separations.numpy().reshape(-1)[start : start + count]
        return {
            "available": True,
            "count": count,
            "max_force_n": float(np.max(np.abs(force_values))) if len(force_values) else 0.0,
            "max_penetration_m": float(max(0.0, -np.min(separation_values))) if len(separation_values) else 0.0,
        }
    except Exception as error:
        return {
            "available": False,
            "count": None,
            "max_force_n": None,
            "max_penetration_m": None,
            "error": repr(error),
        }


def audit_level(records: dict[str, dict[str, Any]]) -> str:
    available = [value for value in records.values() if value.get("available")]
    if not available:
        return "unavailable"
    penetration = max(float(value.get("max_penetration_m") or 0.0) for value in available)
    contacts = sum(int(value.get("count") or 0) for value in available)
    if penetration > 0.003:
        return "penetrating"
    if contacts > 0:
        return "contact"
    return "clear"


def part_world_transform(
    frame_record: dict[str, Any],
    part: str,
    world_from_body: np.ndarray,
) -> np.ndarray:
    record = frame_record["parts"][part]
    if part == "body":
        return world_from_body.copy()
    if "T_body_from_part" in record:
        relative = np.asarray(record["T_body_from_part"], dtype=np.float64)
    else:
        body_world = np.asarray(frame_record["parts"]["body"]["T_world_from_part"], dtype=np.float64)
        part_world = np.asarray(record["T_world_from_part"], dtype=np.float64)
        relative = np.linalg.inv(body_world) @ part_world
    return world_from_body @ relative


def set_visual_pose(visual: dict[str, Any], transform: np.ndarray) -> None:
    visual["transform_op"].Set(np_to_gf_matrix(transform))


def set_debug_mode(visuals: dict[str, dict[str, Any]], enabled: bool) -> None:
    for visual in visuals.values():
        set_visibility(visual["textured"], not enabled)
        set_visibility(visual["debug"], enabled)


def set_debug_color(visual: dict[str, Any], rgb255: tuple[int, int, int]) -> None:
    color = Gf.Vec3f(*(float(value) / 255.0 for value in rgb255))
    visual["debug_shader"].GetInput("diffuseColor").Set(color)


def get_font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_SMALL = get_font(18)
FONT_MEDIUM = get_font(24)
FONT_LARGE = get_font(30)


def to_image(data: Any) -> Image.Image:
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
    def __init__(self) -> None:
        rep.orchestrator.set_capture_on_play(False)
        cameras = {
            "perspective": rep.functional.create.camera(
                position=(0.48, -0.48, 0.36),
                look_at=(0.0, 0.0, 0.02),
                clipping_range=(0.01, 10.0),
                parent="/World",
                name="PerspectiveCamera",
            ),
            "top": rep.functional.create.camera(
                position=(0.0, 0.0, 0.72),
                look_at=(0.0, 0.0, 0.0),
                look_at_up_axis=(0.0, 1.0, 0.0),
                clipping_range=(0.01, 10.0),
                parent="/World",
                name="TopCamera",
            ),
            "side": rep.functional.create.camera(
                position=(0.58, 0.02, 0.20),
                look_at=(0.0, 0.0, 0.02),
                clipping_range=(0.01, 10.0),
                parent="/World",
                name="SideCamera",
            ),
        }
        self.render_products = {}
        self.annotators = {}
        for name, camera in cameras.items():
            render_product = rep.create.render_product(
                camera,
                PANE_SPECS[name],
                name=f"{name.title()}RenderProduct",
                force_new=True,
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(render_product)
            self.render_products[name] = render_product
            self.annotators[name] = annotator
        for _ in range(60):
            simulation_app.update()

    def capture(self) -> dict[str, Image.Image]:
        async def capture_async() -> None:
            await rep.orchestrator.step_async(rt_subframes=ARGS.rt_subframes, delta_time=0.0)

        task = asyncio.ensure_future(capture_async())
        for _ in range(3600):
            if task.done():
                break
            simulation_app.update()
        if not task.done():
            task.cancel()
            raise TimeoutError("Replicator capture timed out")
        task.result()
        return {name: to_image(annotator.get_data()) for name, annotator in self.annotators.items()}

    def close(self) -> None:
        for annotator in self.annotators.values():
            annotator.detach()
        for render_product in self.render_products.values():
            render_product.destroy()


def compose_views(
    views: dict[str, Image.Image],
    title: str,
    lines: list[str],
    *,
    border_color: tuple[int, int, int] = (30, 30, 30),
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), (22, 22, 24))
    canvas.paste(views["perspective"], (0, 0))
    canvas.paste(views["top"], (640, 0))
    canvas.paste(views["side"], (640, 360))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, 1280, 60), fill=(0, 0, 0, 190))
    draw.text((18, 13), title, font=FONT_LARGE, fill=(255, 255, 255, 255))
    y = 64
    for line in lines:
        width = min(1240, 20 + int(draw.textlength(line, font=FONT_SMALL)))
        draw.rectangle((8, y, 8 + width, y + 27), fill=(0, 0, 0, 150))
        draw.text((16, y + 3), line, font=FONT_SMALL, fill=(245, 245, 245, 255))
        y += 29
    draw.rectangle((2, 2, 1277, 717), outline=(*border_color, 255), width=5)
    draw.text((654, 328), "TOP", font=FONT_MEDIUM, fill=(255, 255, 255, 230))
    draw.text((654, 688), "SIDE", font=FONT_MEDIUM, fill=(255, 255, 255, 230))
    return canvas


def save_frame(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=94)


def create_contact_sheet(frame_root: Path, output_path: Path) -> None:
    sources = [
        frame_root / "pose" / "000110.jpg",
        frame_root / "collision" / "000032.jpg",
        frame_root / "inner_physics" / "000022.jpg",
        frame_root / "assembly_release" / "000022.jpg",
    ]
    fallback_dirs = [
        frame_root / "pose",
        frame_root / "collision",
        frame_root / "inner_physics",
        frame_root / "assembly_release",
    ]
    images: list[Image.Image] = []
    for source, directory in zip(sources, fallback_dirs, strict=True):
        if not source.is_file():
            candidates = sorted(directory.glob("*.jpg"))
            if not candidates:
                raise RuntimeError(f"No preview frame available in {directory}")
            source = candidates[-1]
        images.append(Image.open(source).convert("RGB").resize((640, 360), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (1280, 720))
    for image, origin in zip(images, ((0, 0), (640, 0), (0, 360), (640, 360)), strict=True):
        sheet.paste(image, origin)
    sheet.save(output_path, quality=92)


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
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
    ]
    subprocess.run(command, check=True)


def trial_transform(target: np.ndarray, trial: dict[str, Any], drop_height: float) -> np.ndarray:
    tilt_x, tilt_y = np.radians(np.asarray(trial.get("tilt_deg", [0.0, 0.0]), dtype=np.float64))
    yaw = math.radians(float(trial.get("yaw_deg", 0.0)))
    perturbation = (
        axis_angle_matrix(np.array([0.0, 0.0, 1.0]), yaw)
        @ axis_angle_matrix(np.array([0.0, 1.0, 0.0]), tilt_y)
        @ axis_angle_matrix(np.array([1.0, 0.0, 0.0]), tilt_x)
    )
    result = target.copy()
    result[:3, :3] = perturbation @ target[:3, :3]
    dx, dy = trial.get("xy_offset_m", [0.0, 0.0])
    result[:3, 3] += np.array([float(dx), float(dy), float(drop_height)])
    return result


def view_state(view: RigidPrim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, orientations = view.get_world_poses()
    linear, angular = view.get_velocities()
    transform = np.eye(4)
    transform[:3, :3] = quaternion_wxyz_to_matrix(orientations.numpy()[0])
    transform[:3, 3] = positions.numpy()[0]
    return transform, linear.numpy()[0], angular.numpy()[0]


def selected_pose_frames() -> list[str]:
    frame_ids = sorted(TRAJECTORY["frames"], key=int)[:: max(1, ARGS.pose_stride)]
    if ARGS.max_pose_frames is not None:
        frame_ids = frame_ids[: ARGS.max_pose_frames]
    return frame_ids


def create_lights(stage: Usd.Stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr().Set(450.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Distant")
    distant.CreateIntensityAttr().Set(1300.0)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 20.0))


def create_floor(stage: Usd.Stage) -> Usd.Prim:
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr().Set(2.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.10))
    floor.AddScaleOp().Set(Gf.Vec3d(0.8, 0.8, 0.01))
    collision = UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    collision.CreateCollisionEnabledAttr().Set(False)
    bind_solid_material(
        stage,
        floor.GetPrim(),
        "/World/Materials/Floor",
        (0.16, 0.18, 0.22),
        metallic=0.05,
    )
    return floor.GetPrim()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame_root = OUTPUT_ROOT / "frames"
    if frame_root.exists():
        shutil.rmtree(frame_root)
    for name in ("pose", "collision", "inner_physics", "assembly_release", "physics_driven"):
        (frame_root / name).mkdir(parents=True, exist_ok=True)

    usd_paths = load_usd_cache()
    simulation = MANIFEST["simulation"]
    world_from_body = np.eye(4)
    world_from_body[:3, :3] = align_vectors(np.asarray(simulation["up_axis_body"]), np.array([0.0, 0.0, 1.0]))

    stage_utils.create_new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Materials")
    UsdGeom.Scope.Define(stage, "/World/ReplayVisuals")
    UsdGeom.Scope.Define(stage, "/World/TargetVisuals")

    roots = {
        "body": add_reference(stage, "/World/PhysicsAssets/Body", usd_paths["body"], world_from_body),
        "inner_pot": add_reference(stage, "/World/PhysicsAssets/InnerPot", usd_paths["inner_pot"]),
        "lid": add_reference(stage, "/World/PhysicsAssets/Lid", usd_paths["lid"]),
    }
    for _ in range(12):
        simulation_app.update()
    for root in roots.values():
        deinstance(root)
        remove_articulation(root)
        hide_imported_visuals(root)

    rigid_prims = {part: choose_rigid(roots[part], part) for part in ("body", "inner_pot", "lid")}
    colliders = {
        "body": configure_collision(roots["body"], "none"),
        "inner_pot": configure_collision(roots["inner_pot"], "convexDecomposition"),
        "lid": configure_collision(roots["lid"], "convexDecomposition"),
    }
    make_static(rigid_prims["body"])
    for part in ("inner_pot", "lid"):
        physx = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prims[part])
        physx.CreateSolverPositionIterationCountAttr().Set(32)
        physx.CreateSolverVelocityIterationCountAttr().Set(4)
        physx.CreateEnableCCDAttr().Set(False)
        physx.CreateDisableGravityAttr().Set(True)
        physx.CreateLinearDampingAttr().Set(0.08)
        physx.CreateAngularDampingAttr().Set(0.08)
        physx.CreateMaxDepenetrationVelocityAttr().Set(0.35)
        physx.CreateMaxLinearVelocityAttr().Set(0.8)
        physx.CreateMaxAngularVelocityAttr().Set(8.0)
        set_kinematic(rigid_prims[part], False)

    visuals = {part: author_part_visual(stage, part) for part in TRAJECTORY["parts"]}
    target_visuals = {
        part: author_part_visual(
            stage,
            part,
            scope="/World/TargetVisuals",
            material_suffix="Target",
            debug_opacity=0.28,
        )
        for part in ("inner_pot", "lid")
    }
    set_debug_mode(target_visuals, True)
    for visual in target_visuals.values():
        set_visibility(visual["axes"], False)
        set_visibility(visual["root"], False)
    dt = configure_physics_scene(stage)
    create_lights(stage)
    floor_prim = create_floor(stage)
    SimulationManager.switch_physics_engine("physx")
    SimulationManager.setup_simulation(dt=dt, device=ARGS.device)
    views = {
        part: RigidPrim(
            str(rigid_prims[part].GetPath()),
            reset_xform_op_properties=True,
            max_contact_count=4096,
        )
        for part in ("inner_pot", "lid")
    }
    for view in views.values():
        view.set_enabled_contact_tracking([True], threshold=0.0)

    first_frame = TRAJECTORY["frames"][sorted(TRAJECTORY["frames"], key=int)[0]]
    initial_transforms = {
        part: part_world_transform(first_frame, part, world_from_body)
        for part in TRAJECTORY["parts"]
    }
    for part, transform in initial_transforms.items():
        set_visual_pose(visuals[part], transform)

    app_utils.play(commit=True)
    for _ in range(4):
        simulation_app.update()
    for part in ("inner_pot", "lid"):
        set_pose(views[part], initial_transforms[part])
    SimulationManager.step(steps=2)
    capture = MultiViewCapture()

    pose_report: list[dict[str, Any]] = []
    pose_ids = selected_pose_frames()
    for output_index, frame_id in enumerate(pose_ids):
        frame_record = TRAJECTORY["frames"][frame_id]
        transforms = {
            part: part_world_transform(frame_record, part, world_from_body)
            for part in TRAJECTORY["parts"]
        }
        for part, transform in transforms.items():
            set_visual_pose(visuals[part], transform)
        for part in ("inner_pot", "lid"):
            set_pose(views[part], transforms[part])
        SimulationManager.step(steps=1)
        contacts = {part: contact_snapshot(views[part], dt) for part in ("inner_pot", "lid")}
        level = audit_level(contacts)
        images = capture.capture()
        states = {
            part: frame_record["parts"][part].get("state", "unknown")
            for part in TRAJECTORY["parts"]
        }
        pose_image = compose_views(
            images,
            f"Isaac pose replay | frame {int(frame_id):03d}",
            [
                "Pose-driven replay: the three meshes follow pose_solver; this is not free dynamics.",
                " | ".join(f"{part}: {states[part]}" for part in TRAJECTORY["parts"]),
                "Axes: X red, Y green, Z blue.",
            ],
        )
        save_frame(frame_root / "pose" / f"{output_index:06d}.jpg", pose_image)

        border = STATE_COLORS[level]
        contact_lines = []
        for part in ("inner_pot", "lid"):
            value = contacts[part]
            if value.get("available"):
                contact_lines.append(
                    f"{part}: contacts={value['count']}  max penetration={1000.0 * value['max_penetration_m']:.2f} mm"
                )
            else:
                contact_lines.append(f"{part}: contact query unavailable")
        collision_image = compose_views(
            images,
            f"PhysX collision audit | frame {int(frame_id):03d} | {level.upper()}",
            [
                "Gravity-disabled dynamic proxies are reset to the solved pose and stepped once for contact detection.",
                *contact_lines,
            ],
            border_color=border,
        )
        save_frame(frame_root / "collision" / f"{output_index:06d}.jpg", collision_image)
        pose_report.append(
            {
                "frame_id": frame_id,
                "states": states,
                "audit_level": level,
                "contacts": contacts,
            }
        )
        print(f"pose frame {frame_id}: {level}", flush=True)

    app_utils.stop()
    for _ in range(4):
        simulation_app.update()
    SimulationManager.invalidate_physics()
    UsdPhysics.CollisionAPI.Apply(floor_prim).CreateCollisionEnabledAttr().Set(True)

    target_transforms = {
        "body": world_from_body,
        "inner_pot": world_from_body @ np.asarray(MANIFEST["assembled_T_body_from_part"]["inner_pot"], dtype=np.float64),
        "lid": world_from_body @ np.asarray(MANIFEST["assembled_T_body_from_part"]["lid"], dtype=np.float64),
    }
    set_kinematic(rigid_prims["inner_pot"], False)
    set_kinematic(rigid_prims["lid"], True)
    inner_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prims["inner_pot"])
    lid_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prims["lid"])
    inner_physx.CreateDisableGravityAttr().Set(False)
    lid_physx.CreateDisableGravityAttr().Set(False)
    inner_physx.CreateEnableCCDAttr().Set(True)
    app_utils.play(commit=True)
    for _ in range(4):
        simulation_app.update()

    physics_frame = 0
    physics_trials: list[dict[str, Any]] = []
    steps_per_capture = max(1, int(round(1.0 / (dt * ARGS.fps))))
    captures_per_trial = max(2, int(math.ceil(ARGS.physics_seconds * ARGS.fps)))
    configured_trials = simulation["trials"]
    preferred_names = ("aligned", "x_plus_2mm", "yaw_plus_5deg")
    trials = [trial for name in preferred_names for trial in configured_trials if trial["name"] == name]
    trials = trials[: max(1, ARGS.trial_limit)]
    set_visual_pose(visuals["lid"], target_transforms["lid"])
    set_visibility(visuals["lid"]["root"], False)
    lid_away = target_transforms["lid"].copy()
    lid_away[2, 3] += 1.0
    set_pose(views["lid"], lid_away, zero_velocity=False)
    for trial in trials:
        initial = trial_transform(target_transforms["inner_pot"], trial, float(simulation["drop_height_m"]))
        set_pose(views["inner_pot"], initial)
        SimulationManager.step(steps=1)
        trial_samples = []
        for sample_index in range(captures_per_trial):
            SimulationManager.step(steps=steps_per_capture)
            transform, linear, angular = view_state(views["inner_pot"])
            set_visual_pose(visuals["inner_pot"], transform)
            images = capture.capture()
            translation_error = float(np.linalg.norm(transform[:3, 3] - target_transforms["inner_pot"][:3, 3]))
            rotation_error = rotation_error_deg(transform[:3, :3], target_transforms["inner_pot"][:3, :3])
            speed = float(np.linalg.norm(linear))
            angular_speed = float(np.linalg.norm(angular))
            image = compose_views(
                images,
                f"Inner-pot free physics | {trial['name']} | t={sample_index / ARGS.fps:.2f}s",
                [
                    "The inner_pot is dynamic; pose_solver is used only for the initial/target pose.",
                    f"translation error={1000.0 * translation_error:.1f} mm | rotation error={rotation_error:.1f} deg",
                    f"linear speed={speed:.3f} m/s | angular speed={angular_speed:.3f} rad/s",
                ],
                border_color=(58, 125, 220),
            )
            save_frame(frame_root / "inner_physics" / f"{physics_frame:06d}.jpg", image)
            physics_frame += 1
            trial_samples.append(
                {
                    "time_s": sample_index / ARGS.fps,
                    "T_world_from_inner_pot": transform.tolist(),
                    "translation_error_m": translation_error,
                    "rotation_error_deg": rotation_error,
                    "linear_speed_mps": speed,
                    "angular_speed_radps": angular_speed,
                }
            )
        physics_trials.append({"name": trial["name"], "input": trial, "samples": trial_samples})

    app_utils.stop()
    for _ in range(4):
        simulation_app.update()
    SimulationManager.invalidate_physics()

    set_visibility(visuals["lid"]["root"], True)
    set_kinematic(rigid_prims["inner_pot"], False)
    set_kinematic(rigid_prims["lid"], False)
    lid_physx.CreateDisableGravityAttr().Set(False)
    lid_physx.CreateEnableCCDAttr().Set(True)
    app_utils.play(commit=True)
    for _ in range(4):
        simulation_app.update()
    set_pose(views["inner_pot"], target_transforms["inner_pot"])
    set_pose(views["lid"], target_transforms["lid"])
    SimulationManager.step(steps=1)

    assembly_samples: list[dict[str, Any]] = []
    assembly_count = max(2, int(math.ceil(ARGS.physics_seconds * ARGS.fps)))
    for sample_index in range(assembly_count):
        SimulationManager.step(steps=steps_per_capture)
        inner_tf, inner_linear, inner_angular = view_state(views["inner_pot"])
        lid_tf, lid_linear, lid_angular = view_state(views["lid"])
        set_visual_pose(visuals["inner_pot"], inner_tf)
        set_visual_pose(visuals["lid"], lid_tf)
        images = capture.capture()
        inner_error = float(np.linalg.norm(inner_tf[:3, 3] - target_transforms["inner_pot"][:3, 3]))
        lid_error = float(np.linalg.norm(lid_tf[:3, 3] - target_transforms["lid"][:3, 3]))
        image = compose_views(
            images,
            f"Full assembly release | t={sample_index / ARGS.fps:.2f}s",
            [
                "body is static; inner_pot and lid are both dynamic and released from the solved assembly pose.",
                f"inner displacement={1000.0 * inner_error:.1f} mm | lid displacement={1000.0 * lid_error:.1f} mm",
                f"inner speed={np.linalg.norm(inner_linear):.3f} m/s | lid speed={np.linalg.norm(lid_linear):.3f} m/s",
            ],
            border_color=(120, 90, 215),
        )
        save_frame(frame_root / "assembly_release" / f"{sample_index:06d}.jpg", image)
        assembly_samples.append(
            {
                "time_s": sample_index / ARGS.fps,
                "inner_pot": {
                    "T_world_from_part": inner_tf.tolist(),
                    "displacement_m": inner_error,
                    "linear_speed_mps": float(np.linalg.norm(inner_linear)),
                    "angular_speed_radps": float(np.linalg.norm(inner_angular)),
                },
                "lid": {
                    "T_world_from_part": lid_tf.tolist(),
                    "displacement_m": lid_error,
                    "linear_speed_mps": float(np.linalg.norm(lid_linear)),
                    "angular_speed_radps": float(np.linalg.norm(lid_angular)),
                },
            }
        )

    app_utils.stop()
    for _ in range(4):
        simulation_app.update()
    SimulationManager.invalidate_physics()

    set_visibility(visuals["lid"]["root"], True)
    set_visibility(target_visuals["inner_pot"]["root"], True)
    set_visibility(target_visuals["lid"]["root"], True)
    set_debug_color(target_visuals["inner_pot"], (35, 220, 235))
    set_debug_color(target_visuals["lid"], (245, 70, 210))
    for part in ("inner_pot", "lid"):
        set_kinematic(rigid_prims[part], False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prims[part]).CreateDisableGravityAttr().Set(False)
    controllers = {
        part: create_force_controller(rigid_prims[part])
        for part in ("inner_pot", "lid")
    }
    app_utils.play(commit=True)
    for _ in range(4):
        simulation_app.update()

    first_physics_frame = TRAJECTORY["frames"][pose_ids[0]]
    first_physics_targets = {
        part: part_world_transform(first_physics_frame, part, world_from_body)
        for part in TRAJECTORY["parts"]
    }
    for part in ("inner_pot", "lid"):
        set_pose(views[part], first_physics_targets[part])
        set_visual_pose(visuals[part], first_physics_targets[part])
        set_visual_pose(target_visuals[part], first_physics_targets[part])
    SimulationManager.step(steps=2)

    controller_parameters = {
        "inner_pot": {
            "mass": float(MANIFEST["parts"]["inner_pot"]["mass_kg"]),
            "inertia_scale": 0.0032,
            "force_limit": 12.0,
            "torque_limit": 0.35,
        },
        "lid": {
            "mass": float(MANIFEST["parts"]["lid"]["mass_kg"]),
            "inertia_scale": 0.0038,
            "force_limit": 16.0,
            "torque_limit": 0.45,
        },
    }
    controller_steps = max(1, int(round(1.0 / (dt * ARGS.trajectory_fps))))
    physics_driven_samples: list[dict[str, Any]] = []
    first_blocked_frame: str | None = None
    assembly_failed = False
    for output_index, frame_id in enumerate(pose_ids):
        frame_record = TRAJECTORY["frames"][frame_id]
        targets = {
            part: part_world_transform(frame_record, part, world_from_body)
            for part in TRAJECTORY["parts"]
        }
        for part in ("inner_pot", "lid"):
            set_visual_pose(target_visuals[part], targets[part])

        last_control: dict[str, dict[str, Any]] = {}
        for _ in range(controller_steps):
            for part in ("inner_pot", "lid"):
                actual, linear, angular = view_state(views[part])
                last_control[part] = apply_pose_controller(
                    controllers[part],
                    actual=actual,
                    target=targets[part],
                    linear_velocity=linear,
                    angular_velocity=angular,
                    **controller_parameters[part],
                )
            SimulationManager.step(steps=1)

        actual_states: dict[str, dict[str, Any]] = {}
        contacts = {}
        for part in ("inner_pot", "lid"):
            actual, linear, angular = view_state(views[part])
            set_visual_pose(visuals[part], actual)
            contacts[part] = contact_snapshot(views[part], dt)
            actual_states[part] = {
                "T_world_from_part": actual.tolist(),
                "linear_velocity_mps": linear.tolist(),
                "angular_velocity_radps": angular.tolist(),
                **last_control[part],
            }
        max_position_error = max(
            actual_states[part]["position_error_m"] for part in ("inner_pot", "lid")
        )
        max_rotation_error = max(
            actual_states[part]["rotation_error_deg"] for part in ("inner_pot", "lid")
        )
        contact_count = sum(int(value.get("count") or 0) for value in contacts.values())
        part_states = {
            part: frame_record["parts"][part].get("state", "unknown")
            for part in ("inner_pot", "lid")
        }
        active_parts = [
            part for part, state in part_states.items()
            if state == "moving"
        ] or ["inner_pot", "lid"]
        blocked_parts = [
            part
            for part in active_parts
            if int(contacts[part].get("count") or 0) > 0
            and actual_states[part]["position_error_m"] > 0.02
        ]
        blocked = bool(blocked_parts)
        if blocked and first_blocked_frame is None:
            first_blocked_frame = frame_id
        assembly_failed = assembly_failed or blocked
        status_label = "BLOCKED" if blocked else ("ASSEMBLY FAILED" if assembly_failed else "TRACKING")
        images = capture.capture()
        image = compose_views(
            images,
            f"Physics-driven trajectory | frame {int(frame_id):03d} | {status_label}",
            [
                "Textured meshes/axes are actual PhysX poses; cyan/magenta ghosts are pose_solver targets.",
                f"max translation error={1000.0 * max_position_error:.1f} mm | max rotation error={max_rotation_error:.1f} deg",
                f"contacts={contact_count} | force limits: inner=12 N, lid=16 N | no pose overwrite after frame 0",
            ],
            border_color=(225, 60, 55) if assembly_failed else (45, 180, 100),
        )
        save_frame(frame_root / "physics_driven" / f"{output_index:06d}.jpg", image)
        physics_driven_samples.append(
            {
                "frame_id": frame_id,
                "blocked": blocked,
                "assembly_failed": assembly_failed,
                "blocked_parts": blocked_parts,
                "states": part_states,
                "targets": {
                    part: targets[part].tolist()
                    for part in ("inner_pot", "lid")
                },
                "actual": actual_states,
                "contacts": contacts,
            }
        )
        print(
            f"physics-driven frame {frame_id}: error={1000.0 * max_position_error:.1f} mm "
            f"contacts={contact_count} blocked={blocked}",
            flush=True,
        )

    for controller in controllers.values():
        controller["force_attr"].Set(Gf.Vec3f(0.0, 0.0, 0.0))
        controller["torque_attr"].Set(Gf.Vec3f(0.0, 0.0, 0.0))
    app_utils.stop()
    capture.close()
    stage_path = OUTPUT_ROOT / "isaac_visualization_scene.usda"
    stage.GetRootLayer().Export(str(stage_path))

    videos = {
        "pose_replay": OUTPUT_ROOT / "01_pose_replay_textured.mp4",
        "collision_audit": OUTPUT_ROOT / "02_pose_replay_collision_debug.mp4",
        "inner_pot_physics": OUTPUT_ROOT / "03_inner_pot_physics_trials.mp4",
        "full_assembly_release": OUTPUT_ROOT / "04_full_assembly_release.mp4",
        "physics_driven_trajectory": OUTPUT_ROOT / "05_physics_driven_trajectory.mp4",
    }
    encode_video(frame_root / "pose", videos["pose_replay"], ARGS.fps)
    encode_video(frame_root / "collision", videos["collision_audit"], ARGS.fps)
    encode_video(frame_root / "inner_physics", videos["inner_pot_physics"], ARGS.fps)
    encode_video(frame_root / "assembly_release", videos["full_assembly_release"], ARGS.fps)
    encode_video(frame_root / "physics_driven", videos["physics_driven_trajectory"], ARGS.fps)
    preview_path = OUTPUT_ROOT / "preview_contact_sheet.jpg"
    create_contact_sheet(frame_root, preview_path)
    physics_preview_path = OUTPUT_ROOT / "physics_driven_final.jpg"
    physics_preview_frames = sorted((frame_root / "physics_driven").glob("*.jpg"))
    if not physics_preview_frames:
        raise RuntimeError("Physics-driven trajectory produced no preview frames")
    shutil.copy2(physics_preview_frames[-1], physics_preview_path)

    level_counts = {
        level: sum(record["audit_level"] == level for record in pose_report)
        for level in STATE_COLORS
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "asset_root": str(ASSET_ROOT),
        "trajectory": str(TRAJECTORY_PATH),
        "isaac_runtime_root": str(RUNTIME_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "scene_usd": str(stage_path),
        "preview_contact_sheet": str(preview_path),
        "physics_driven_preview": str(physics_preview_path),
        "fps": ARGS.fps,
        "physics_seconds": ARGS.physics_seconds,
        "pose_frames_rendered": len(pose_ids),
        "colliders": colliders,
        "videos": {name: str(path) for name, path in videos.items()},
        "video_sizes_bytes": {name: path.stat().st_size for name, path in videos.items()},
        "collision_audit_summary": level_counts,
        "pose_replay": pose_report,
        "inner_pot_trials": physics_trials,
        "full_assembly_release": assembly_samples,
        "physics_driven_trajectory": {
            "trajectory_fps": ARGS.trajectory_fps,
            "physics_steps_per_target": controller_steps,
            "controller_parameters": controller_parameters,
            "first_blocked_frame": first_blocked_frame,
            "samples": physics_driven_samples,
        },
        "interpretation": {
            "pose_replay": "Kinematic visualization of pose_solver outputs; not a physics prediction.",
            "collision_audit": "Gravity-disabled dynamic proxies are reset to each solved pose and stepped once; the floor collider is disabled during this audit.",
            "physics_trials": "Dynamic free-release experiments; pose_solver supplies only initial and target poses.",
            "physics_driven_trajectory": "Pose solver frames are targets for bounded world-frame force/torque control. Actual rigid-body poses are initialized once and never overwritten afterwards.",
        },
    }
    write_json(OUTPUT_ROOT / "visualization_report.json", report)
    write_json(
        OUTPUT_ROOT / "complete.json",
        {"status": "complete", "report": str(OUTPUT_ROOT / "visualization_report.json"), "videos": report["videos"]},
    )
    if not ARGS.keep_frames:
        shutil.rmtree(frame_root)
    print(f"Visualization report: {OUTPUT_ROOT / 'visualization_report.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
