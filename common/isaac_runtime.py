"""Isaac Sim USD import, trajectory replay, and insertion validation."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


from common.io_utils import write_json
from common.assembly_validation import (
    assembly_validation_settings,
    summarize_validation_trials,
)
from common.physics_control import (
    assembly_target_translation,
    dynamic_collision_approximation,
    elastic_tube_wrench,
    place_release_settings,
    physics_pose_refinement_settings,
    score_physics_pose_candidate,
    sustained_contact_summary,
)
from common.physics_pose_projection import (
    project_pose_without_axial_yaw,
    write_physics_refined_trajectory,
)

# This module is imported only after the CLI has constructed SimulationApp.
# Omniverse requires that ordering. These values are initialized once by
# ``run_insertion`` and are process-local to the Isaac invocation.
ARGS: argparse.Namespace
ASSET_ROOT: Path
RUNTIME_ROOT: Path
MANIFEST: dict[str, Any]
simulation_app: Any

# Omniverse imports must happen after SimulationApp construction.
import carb
import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from isaacsim.core.experimental.prims import RigidPrim
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enable_import_extensions() -> None:
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in ("omni.scene.optimizer.core", "isaacsim.robot.schema"):
        manager.set_extension_enabled_immediate(extension, True)


def import_urdf_assets() -> dict[str, str]:
    cache_path = RUNTIME_ROOT / "usd/import_cache.json"
    urdfs = {
        part: ASSET_ROOT / relative
        for part, relative in MANIFEST["outputs"]["independent_urdfs"].items()
    }
    display_asset_key = "assembly_display"
    urdfs[display_asset_key] = ASSET_ROOT / MANIFEST["outputs"]["display_urdf"]
    urdf_hashes = {name: sha256_file(path) for name, path in urdfs.items()}
    mesh_hashes = {}
    for part, info in MANIFEST["parts"].items():
        paths = {
            "visual_mesh": [info["visual_mesh"]],
            "collision_mesh": [info["collision_mesh"]],
            "collision_components": info.get("collision_meshes", []),
        }
        for kind, relative_paths in paths.items():
            for index, relative_path in enumerate(relative_paths):
                path = ASSET_ROOT / relative_path
                mesh_hashes[f"{part}:{kind}:{index}"] = sha256_file(path)
    asset_hashes = {"urdf": urdf_hashes, "mesh": mesh_hashes}
    if cache_path.is_file() and not ARGS.force_import:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_paths = cache.get("usd_paths", {})
        if cache.get("asset_sha256") == asset_hashes and all(Path(path).is_file() for path in cached_paths.values()):
            print(f"Using cached USD assets from {cache_path}")
            return cached_paths

    enable_import_extensions()
    usd_paths: dict[str, str] = {}
    for name, urdf_path in urdfs.items():
        output_dir = RUNTIME_ROOT / "usd/imported" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        config = URDFImporterConfig(
            urdf_path=str(urdf_path),
            usd_path=str(output_dir),
            merge_fixed_joints=False,
            merge_mesh=False,
            collision_from_visuals=False,
            allow_self_collision=False,
            fix_base=name == display_asset_key,
            robot_type="Default",
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
        output_usd = URDFImporter(config).import_urdf()
        if not output_usd or not Path(output_usd).is_file():
            raise RuntimeError(f"URDF import failed for {urdf_path}; output={output_usd!r}")
        usd_paths[name] = str(Path(output_usd).resolve())
        print(f"Imported {name}: {output_usd}")
    write_json(
        cache_path,
        {
            "asset_sha256": asset_hashes,
            "urdf_sha256": urdf_hashes,
            "usd_paths": usd_paths,
        },
    )
    return usd_paths


def np_to_gf_matrix(matrix: np.ndarray) -> Gf.Matrix4d:
    values = np.asarray(matrix, dtype=np.float64)
    # NumPy poses in this project use column vectors (translation in the last
    # column), while Gf.Matrix4d/XformOp uses row-vector convention.
    return Gf.Matrix4d(*values.T.reshape(-1).tolist())


def add_reference(stage: Usd.Stage, prim_path: str, asset_path: str, transform: np.ndarray | None = None) -> Usd.Prim:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.GetPrim().GetReferences().AddReference(asset_path)
    if transform is not None:
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(np_to_gf_matrix(transform))
    return xform.GetPrim()


def descendants_with_api(root: Usd.Prim, api: Any) -> list[Usd.Prim]:
    return [prim for prim in Usd.PrimRange(root) if prim.HasAPI(api)]


def deinstance_composed_tree(root: Usd.Prim, max_passes: int = 8) -> list[str]:
    """Expand Asset Transformer instance roots so nested physics APIs are editable."""
    expanded: list[str] = []
    for _ in range(max_passes):
        instance_roots = [
            prim
            for prim in Usd.PrimRange(root)
            if prim.IsValid() and prim.IsInstance()
        ]
        if not instance_roots:
            break
        for prim in instance_roots:
            prim.SetInstanceable(False)
            expanded.append(str(prim.GetPath()))
        simulation_app.update()
    return sorted(set(expanded))


def choose_rigid_prim(root: Usd.Prim, preferred_name: str) -> Usd.Prim:
    candidates = descendants_with_api(root, UsdPhysics.RigidBodyAPI)
    if not candidates:
        raise RuntimeError(f"No rigid body found below {root.GetPath()}")
    preferred = [prim for prim in candidates if preferred_name.lower() in prim.GetName().lower()]
    return preferred[0] if preferred else candidates[0]


def configure_collision(
    root: Usd.Prim,
    approximation: str,
    *,
    sdf_resolution: int = 192,
) -> list[str]:
    configured: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            if approximation == "sdf":
                if sdf_resolution <= 0:
                    raise ValueError("SDF collision resolution must be positive")
                mesh_collision.CreateApproximationAttr().Set(
                    PhysxSchema.Tokens.sdf
                )
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(
                    prim
                ).CreateSdfResolutionAttr().Set(int(sdf_resolution))
            else:
                mesh_collision.CreateApproximationAttr().Set(approximation)
        configured.append(str(prim.GetPath()))
    if not configured:
        raise RuntimeError(f"No colliders found below {root.GetPath()}")
    return configured


def configure_contact_offsets(
    stage: Usd.Stage,
    collider_paths: list[str],
    simulation: dict[str, Any],
) -> dict[str, float]:
    """Author configured PhysX contact margins before scene initialization."""
    contact_offset = float(simulation.get("contact_offset_m", 0.001))
    rest_offset = float(simulation.get("rest_offset_m", 0.0))
    if contact_offset <= 0.0:
        raise ValueError("simulation.contact_offset_m must be positive")
    if rest_offset > contact_offset:
        raise ValueError(
            "simulation.rest_offset_m cannot exceed contact_offset_m"
        )
    for path in collider_paths:
        prim = stage.GetPrimAtPath(path)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr().Set(contact_offset)
        collision.CreateRestOffsetAttr().Set(rest_offset)
    return {
        "contact_offset_m": contact_offset,
        "rest_offset_m": rest_offset,
    }


def remove_articulation_apis(root: Usd.Prim) -> list[str]:
    removed: list[str] = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed.append(str(prim.GetPath()))
    return removed


def make_body_static(body_rigid: Usd.Prim) -> None:
    for prim in Usd.PrimRange(body_rigid.GetParent()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if body_rigid.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        body_rigid.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
    if body_rigid.HasAPI(UsdPhysics.RigidBodyAPI):
        body_rigid.RemoveAPI(UsdPhysics.RigidBodyAPI)


def bind_physics_material(stage: Usd.Stage, collider_paths: list[str], simulation: dict[str, Any]) -> str:
    material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/ContactMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(float(simulation["static_friction"]))
    physics_material.CreateDynamicFrictionAttr().Set(float(simulation["dynamic_friction"]))
    physics_material.CreateRestitutionAttr().Set(float(simulation["restitution"]))
    for path in collider_paths:
        prim = stage.GetPrimAtPath(path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    return str(material.GetPath())


def read_obj_mesh(path: Path) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    points: list[Gf.Vec3f] = []
    counts: list[int] = []
    indices: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                x, y, z = (float(value) for value in line.split()[1:4])
                points.append(Gf.Vec3f(x, y, z))
            elif line.startswith("f "):
                face = [int(value.split("/", 1)[0]) - 1 for value in line.split()[1:]]
                if len(face) >= 3:
                    counts.append(len(face))
                    indices.extend(face)
    if not points or not counts:
        raise RuntimeError(f"OBJ contains no renderable mesh: {path}")
    return points, counts, indices


def bind_display_material(
    stage: Usd.Stage,
    prim: Usd.Prim,
    path: str,
    color: tuple[float, float, float],
    *,
    metallic: float,
    roughness: float,
    opacity: float,
) -> str:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return str(material.GetPath())


def author_direct_visual(
    stage: Usd.Stage,
    name: str,
    obj_path: Path,
    transform: np.ndarray,
    color: tuple[float, float, float],
    *,
    metallic: float,
    roughness: float,
    opacity: float,
) -> dict[str, Any]:
    points, counts, indices = read_obj_mesh(obj_path)
    root = UsdGeom.Xform.Define(stage, f"/World/DemoVisuals/{name}")
    root.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(np_to_gf_matrix(transform))
    mesh = UsdGeom.Mesh.Define(stage, f"/World/DemoVisuals/{name}/Mesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    material_path = bind_display_material(
        stage,
        mesh.GetPrim(),
        f"/World/DemoMaterials/{name}",
        color,
        metallic=metallic,
        roughness=roughness,
        opacity=opacity,
    )
    return {
        "prim": str(mesh.GetPath()),
        "source_obj": str(obj_path),
        "vertices": len(points),
        "faces": len(counts),
        "material": material_path,
    }


def author_assembly_demo_visuals(
    stage: Usd.Stage,
    world_from_body: np.ndarray,
    target_world: np.ndarray,
    container: str,
    inserted: str,
) -> dict[str, Any]:
    UsdGeom.Scope.Define(stage, "/World/DemoVisuals")
    UsdGeom.Scope.Define(stage, "/World/DemoMaterials")
    visuals = {
        container: author_direct_visual(
            stage,
            "Body",
            ASSET_ROOT / f"meshes/visual/{container}/{container}.obj",
            world_from_body,
            (0.48, 0.20, 0.08),
            metallic=0.05,
            roughness=0.4,
            opacity=0.68,
        ),
        inserted: author_direct_visual(
            stage,
            "InnerPot",
            ASSET_ROOT / f"meshes/visual/{inserted}/{inserted}.obj",
            target_world,
            (0.72, 0.76, 0.82),
            metallic=0.75,
            roughness=0.22,
            opacity=1.0,
        ),
    }
    return visuals


def hide_imported_visuals(root: Usd.Prim) -> list[str]:
    hidden: list[str] = []
    for prim in Usd.PrimRange(root):
        if "visual" not in prim.GetName().lower():
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()
            hidden.append(str(prim.GetPath()))
    return hidden


def align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, trial)
        axis /= np.linalg.norm(axis)
        return axis_angle_matrix(axis, math.pi)
    axis = cross / np.linalg.norm(cross)
    return axis_angle_matrix(axis, math.acos(dot))


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


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return quat / np.linalg.norm(quat)


def quaternion_error_deg(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(a_wxyz, b_wxyz)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized Isaac wxyz quaternion to a rotation matrix."""
    values = np.asarray(quaternion, dtype=np.float64)
    values /= np.linalg.norm(values)
    w, x, y, z = values
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def create_lights(stage: Usd.Stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr().Set(300.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Distant")
    distant.CreateIntensityAttr().Set(1200.0)
    distant_xform = UsdGeom.Xformable(distant.GetPrim())
    distant_xform.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 20.0))


def configure_physics_scene(stage: Usd.Stage, simulation: dict[str, Any]) -> float:
    physics_hz = int(simulation["physics_hz"])
    dt = 1.0 / physics_hz
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr().Set(physics_hz)
    physx_scene.CreateEnableCCDAttr().Set(True)
    physx_scene.CreateEnableStabilizationAttr().Set(True)
    physx_scene.CreateSolverTypeAttr().Set("TGS")
    return dt


def set_kinematic(prim: Usd.Prim, enabled: bool) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(enabled)


def set_pose(view: RigidPrim, transform: np.ndarray, *, zero_velocity: bool = True) -> None:
    view.set_world_poses(
        positions=[transform[:3, 3].tolist()],
        orientations=[matrix_to_quaternion_wxyz(transform[:3, :3]).tolist()],
    )
    if zero_velocity:
        view.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]])


def replay_observed_path(
    view: RigidPrim,
    world_from_body: np.ndarray,
) -> dict[str, Any]:
    replay_path = ASSET_ROOT / MANIFEST["outputs"]["observed_replay"]
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    samples = []
    for frame in replay["frames"]:
        world_from_part = world_from_body @ np.asarray(frame["T_body_from_part"], dtype=np.float64)
        set_pose(view, world_from_part, zero_velocity=False)
        SimulationManager.step(steps=1)
        samples.append({"frame_id": frame["frame_id"], "state": frame["state"]})
    return {
        "frames_tested": len(samples),
        "mode": "kinematic pose replay",
        "samples": samples,
    }


def trial_initial_transform(target_world: np.ndarray, trial: dict[str, Any], drop_height: float) -> np.ndarray:
    tilt_x, tilt_y = np.radians(np.asarray(trial.get("tilt_deg", [0.0, 0.0]), dtype=np.float64))
    yaw = math.radians(float(trial.get("yaw_deg", 0.0)))
    perturbation = (
        axis_angle_matrix(np.array([0.0, 0.0, 1.0]), yaw)
        @ axis_angle_matrix(np.array([0.0, 1.0, 0.0]), tilt_y)
        @ axis_angle_matrix(np.array([1.0, 0.0, 0.0]), tilt_x)
    )
    result = target_world.copy()
    result[:3, :3] = perturbation @ target_world[:3, :3]
    dx, dy = trial.get("xy_offset_m", [0.0, 0.0])
    result[:3, 3] += np.array([float(dx), float(dy), float(drop_height)])
    return result


def run_drop_trial(
    view: RigidPrim,
    target_world: np.ndarray,
    trial: dict[str, Any],
    simulation: dict[str, Any],
    dt: float,
) -> dict[str, Any]:
    initial = trial_initial_transform(target_world, trial, float(simulation["drop_height_m"]))
    set_pose(view, initial)
    SimulationManager.step(steps=1)
    steps = int(math.ceil(float(simulation["settle_seconds"]) / dt))
    max_speed = max_angular_speed = 0.0
    min_world_z = float("inf")
    for _ in range(steps):
        SimulationManager.step(steps=1)
        positions, orientations = view.get_world_poses()
        linear_velocities, angular_velocities = view.get_velocities()
        position = positions.numpy()[0]
        linear = linear_velocities.numpy()[0]
        angular = angular_velocities.numpy()[0]
        min_world_z = min(min_world_z, float(position[2]))
        max_speed = max(max_speed, float(np.linalg.norm(linear)))
        max_angular_speed = max(max_angular_speed, float(np.linalg.norm(angular)))

    positions, orientations = view.get_world_poses()
    linear_velocities, angular_velocities = view.get_velocities()
    final_position = positions.numpy()[0]
    final_orientation = orientations.numpy()[0]
    final_linear = linear_velocities.numpy()[0]
    final_angular = angular_velocities.numpy()[0]
    target_orientation = matrix_to_quaternion_wxyz(target_world[:3, :3])
    translation_error = float(np.linalg.norm(final_position - target_world[:3, 3]))
    rotation_error = quaternion_error_deg(final_orientation, target_orientation)
    final_linear_speed = float(np.linalg.norm(final_linear))
    final_angular_speed = float(np.linalg.norm(final_angular))
    success = bool(
        translation_error <= float(simulation["success_translation_m"])
        and rotation_error <= float(simulation["success_rotation_deg"])
        and final_linear_speed <= float(simulation["success_linear_speed_mps"])
        and final_angular_speed <= float(simulation["success_angular_speed_radps"])
    )
    return {
        "name": trial["name"],
        "input": trial,
        "initial_T_world_from_part": initial.tolist(),
        "final_position_world_m": final_position.tolist(),
        "final_quaternion_world_wxyz": final_orientation.tolist(),
        "translation_error_m": translation_error,
        "rotation_error_deg": rotation_error,
        "final_linear_speed_mps": final_linear_speed,
        "final_angular_speed_radps": final_angular_speed,
        "max_linear_speed_mps": max_speed,
        "max_angular_speed_radps": max_angular_speed,
        "min_world_z_m": min_world_z,
        "contact_query": "disabled; pose settling is evaluated without the Isaac tensor contact sensor",
        "success": success,
    }


def contact_snapshot(view: RigidPrim, dt: float) -> dict[str, Any]:
    """Return bounded raw-contact evidence for one rigid body."""
    try:
        forces, _points, _normals, separations, counts, starts, actor_ids = (
            view.get_raw_contact_data(dt=dt)
        )
        count = int(np.asarray(counts.numpy()).sum())
        start = int(np.asarray(starts.numpy()).reshape(-1)[0]) if count else 0
        contact_slice = slice(start, start + count)
        force_values = forces.numpy()[contact_slice]
        separation_values = separations.numpy()[contact_slice]
        actor_paths = (
            sorted(
                {
                    path
                    for path in view.get_actor_paths_from_ids(
                        actor_ids[contact_slice]
                    )
                    if path
                }
            )
            if count
            else []
        )
        return {
            "available": True,
            "count": count,
            "max_force_n": (
                float(np.linalg.norm(force_values, axis=-1).max())
                if force_values.size
                else 0.0
            ),
            "max_penetration_m": (
                max(0.0, -float(separation_values.min()))
                if separation_values.size
                else 0.0
            ),
            "other_actor_paths": actor_paths,
        }
    except Exception as error:
        return {
            "available": False,
            "count": 0,
            "error": repr(error),
            "other_actor_paths": [],
        }


def place_release_pose_metrics(
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    target_world: np.ndarray,
    inserted_axis_part: np.ndarray,
) -> dict[str, float]:
    """Measure passive rest error without penalizing screw-axis yaw as tilt."""
    delta = np.asarray(position, dtype=np.float64) - target_world[:3, 3]
    axial_error_signed = float(delta[2])
    lateral_error = float(np.linalg.norm(delta[:2]))
    rotation = quaternion_wxyz_to_matrix(orientation_wxyz)
    target_axis = target_world[:3, :3] @ inserted_axis_part
    actual_axis = rotation @ inserted_axis_part
    target_axis /= np.linalg.norm(target_axis)
    actual_axis /= np.linalg.norm(actual_axis)
    tilt_error = math.degrees(math.acos(float(np.clip(
        np.dot(target_axis, actual_axis), -1.0, 1.0
    ))))
    return {
        "translation_error_m": float(np.linalg.norm(delta)),
        "lateral_error_m": lateral_error,
        "axial_error_signed_m": axial_error_signed,
        "axial_error_m": abs(axial_error_signed),
        "rotation_error_deg": quaternion_error_deg(
            np.asarray(orientation_wxyz, dtype=np.float64),
            matrix_to_quaternion_wxyz(target_world[:3, :3]),
        ),
        "tilt_error_deg": tilt_error,
    }


def run_place_release_trial(
    view: RigidPrim,
    target_world: np.ndarray,
    inserted_axis_part: np.ndarray,
    trial: dict[str, Any],
    settings: dict[str, Any],
    dt: float,
) -> dict[str, Any]:
    """Place just above the contact seat, release, and observe passive rest."""
    initial = trial_initial_transform(
        target_world,
        trial,
        float(settings["initial_height_m"]),
    )
    set_pose(view, initial)
    SimulationManager.step(steps=1)
    steps = int(math.ceil(float(settings["settle_seconds"]) / dt))
    sample_interval = max(1, int(round(0.1 / dt)))
    contact_history: list[bool] = []
    contact_paths: set[str] = set()
    contact_available = True
    maximum_penetration = 0.0
    maximum_linear_speed = 0.0
    maximum_angular_speed = 0.0
    maximum_lateral_error = 0.0
    maximum_tilt_error = 0.0
    samples: list[dict[str, Any]] = []
    for step in range(steps):
        SimulationManager.step(steps=1)
        positions, orientations = view.get_world_poses()
        linear_velocities, angular_velocities = view.get_velocities()
        position = positions.numpy()[0]
        orientation = orientations.numpy()[0]
        linear = linear_velocities.numpy()[0]
        angular = angular_velocities.numpy()[0]
        metrics = place_release_pose_metrics(
            position,
            orientation,
            target_world,
            inserted_axis_part,
        )
        contact = contact_snapshot(view, dt)
        contact_available &= bool(contact["available"])
        has_contact = bool(contact["count"] > 0)
        contact_history.append(has_contact)
        contact_paths.update(contact.get("other_actor_paths", []))
        maximum_penetration = max(
            maximum_penetration,
            float(contact.get("max_penetration_m", 0.0)),
        )
        linear_speed = float(np.linalg.norm(linear))
        angular_speed = float(np.linalg.norm(angular))
        maximum_linear_speed = max(maximum_linear_speed, linear_speed)
        maximum_angular_speed = max(maximum_angular_speed, angular_speed)
        maximum_lateral_error = max(
            maximum_lateral_error,
            metrics["lateral_error_m"],
        )
        maximum_tilt_error = max(
            maximum_tilt_error,
            metrics["tilt_error_deg"],
        )
        if step % sample_interval == sample_interval - 1 or step == steps - 1:
            samples.append(
                {
                    "time_s": float(step + 1) * dt,
                    **metrics,
                    "linear_speed_mps": linear_speed,
                    "angular_speed_radps": angular_speed,
                    "contact_count": int(contact["count"]),
                }
            )

    positions, orientations = view.get_world_poses()
    linear_velocities, angular_velocities = view.get_velocities()
    final_position = positions.numpy()[0]
    final_orientation = orientations.numpy()[0]
    final_linear_speed = float(np.linalg.norm(linear_velocities.numpy()[0]))
    final_angular_speed = float(np.linalg.norm(angular_velocities.numpy()[0]))
    final_metrics = place_release_pose_metrics(
        final_position,
        final_orientation,
        target_world,
        inserted_axis_part,
    )
    contact_gate = sustained_contact_summary(
        contact_history,
        physics_dt=dt,
        settings=settings,
    )
    criteria = {
        "lateral": final_metrics["lateral_error_m"]
        <= float(settings["maximum_lateral_error_m"]),
        "axial": final_metrics["axial_error_m"]
        <= float(settings["maximum_axial_error_m"]),
        "tilt": final_metrics["tilt_error_deg"]
        <= float(settings["maximum_tilt_error_deg"]),
        "linear_speed": final_linear_speed
        <= float(settings["maximum_final_linear_speed_mps"]),
        "angular_speed": final_angular_speed
        <= float(settings["maximum_final_angular_speed_radps"]),
        "sustained_contact": bool(contact_gate["sustained"]),
        "contact_query": contact_available,
    }
    return {
        "name": str(trial["name"]),
        "mode": "passive_place_and_release",
        "input": trial,
        "initial_T_world_from_part": initial.tolist(),
        "target_T_world_from_part": target_world.tolist(),
        "final_position_world_m": final_position.tolist(),
        "final_quaternion_world_wxyz": final_orientation.tolist(),
        **final_metrics,
        "final_linear_speed_mps": final_linear_speed,
        "final_angular_speed_radps": final_angular_speed,
        "maximum_linear_speed_mps": maximum_linear_speed,
        "maximum_angular_speed_radps": maximum_angular_speed,
        "maximum_lateral_error_m": maximum_lateral_error,
        "maximum_tilt_error_deg": maximum_tilt_error,
        "maximum_contact_penetration_m": maximum_penetration,
        "contact_actor_paths": sorted(contact_paths),
        "contact_gate": contact_gate,
        "criteria_passed": criteria,
        "success": all(criteria.values()),
        "samples": samples,
    }


def read_rigid_state(
    view: RigidPrim,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions, orientations = view.get_world_poses()
    linear_velocities, angular_velocities = view.get_velocities()
    position = positions.numpy()[0].astype(np.float64)
    orientation = orientations.numpy()[0].astype(np.float64)
    return (
        position,
        quaternion_wxyz_to_matrix(orientation),
        linear_velocities.numpy()[0].astype(np.float64),
        angular_velocities.numpy()[0].astype(np.float64),
    )


def tube_curve_points(
    *,
    position_world: np.ndarray,
    rotation_world_from_part: np.ndarray,
    body_axis_origin_world: np.ndarray,
    body_axis_world: np.ndarray,
    tube: dict[str, Any],
    count: int = 16,
) -> np.ndarray:
    """Create a bounded visual centerline for the latent flexible tube."""
    body_axis = np.asarray(body_axis_world, dtype=np.float64)
    body_axis /= np.linalg.norm(body_axis)
    direction_part = np.asarray(tube["direction_part"], dtype=np.float64)
    direction_part /= np.linalg.norm(direction_part)
    direction_world = rotation_world_from_part @ direction_part
    mount = (
        np.asarray(position_world, dtype=np.float64)
        + rotation_world_from_part
        @ np.asarray(tube["mount_point_part_m"], dtype=np.float64)
    )
    origin = np.asarray(body_axis_origin_world, dtype=np.float64)
    axis_start = origin + body_axis * float(np.dot(mount - origin, body_axis))
    length = float(tube["length_m"])
    control0 = mount
    control1 = mount + direction_world * min(0.04, 0.25 * length)
    control2 = axis_start - body_axis * (0.65 * length)
    lateral = mount - axis_start
    lateral -= body_axis * float(np.dot(lateral, body_axis))
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1e-9:
        lateral = np.cross(body_axis, np.asarray([1.0, 0.0, 0.0]))
        if np.linalg.norm(lateral) <= 1e-9:
            lateral = np.cross(body_axis, np.asarray([0.0, 1.0, 0.0]))
        lateral /= np.linalg.norm(lateral)
    else:
        lateral /= lateral_norm
    control3 = axis_start - body_axis * (0.92 * length) + lateral * (0.04 * length)
    points = []
    for value in np.linspace(0.0, 1.0, max(4, int(count))):
        inverse = 1.0 - value
        points.append(
            inverse**3 * control0
            + 3.0 * inverse**2 * value * control1
            + 3.0 * inverse * value**2 * control2
            + value**3 * control3
        )
    return np.asarray(points, dtype=np.float64)


def author_tube_visual(
    stage: Usd.Stage,
    points_world: np.ndarray,
    tube: dict[str, Any],
) -> dict[str, Any]:
    curve = UsdGeom.BasisCurves.Define(stage, "/World/PhysicsPoseRefinement/DipTube")
    curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr().Set([int(len(points_world))])
    points_attr = curve.CreatePointsAttr()
    points_attr.Set([Gf.Vec3f(*point.tolist()) for point in points_world])
    curve.CreateWidthsAttr().Set(
        [2.0 * float(tube["radius_m"])] * int(len(points_world))
    )
    curve.CreateDisplayColorAttr().Set([Gf.Vec3f(0.18, 0.82, 0.28)])
    material_path = bind_display_material(
        stage,
        curve.GetPrim(),
        "/World/DemoMaterials/PhysicsDipTube",
        (0.08, 0.85, 0.18),
        metallic=0.0,
        roughness=0.35,
        opacity=1.0,
    )
    return {
        "prim": str(curve.GetPath()),
        "points_attr": points_attr,
        "point_count": int(len(points_world)),
        "material": material_path,
    }


def update_tube_visual(visual: dict[str, Any], points_world: np.ndarray) -> None:
    visual["points_attr"].Set(
        [Gf.Vec3f(*point.tolist()) for point in points_world]
    )


def run_physics_pose_refinement_trial(
    view: RigidPrim,
    visual_target_world: np.ndarray,
    candidate: dict[str, Any],
    settings: dict[str, Any],
    body_axis_origin_world: np.ndarray,
    body_axis_world: np.ndarray,
    dt: float,
    *,
    tube_visual: dict[str, Any] | None = None,
    capture: Any | None = None,
    frame_dir: Path | None = None,
    capture_fps: float = 5.0,
) -> dict[str, Any]:
    """Run one passive candidate with only contact, gravity, and tube elasticity."""
    initial = trial_initial_transform(
        visual_target_world,
        candidate,
        float(settings["initial_height_m"]),
    )
    set_pose(view, initial)
    SimulationManager.step(steps=1)
    tube = settings["tube"]
    steps = int(math.ceil(float(settings["settle_seconds"]) / dt))
    sample_interval = max(1, int(round(float(settings["sample_seconds"]) / dt)))
    capture_interval = max(1, int(round(1.0 / (dt * float(capture_fps)))))
    contact_observed = False
    contact_paths: set[str] = set()
    maximum_penetration = 0.0
    maximum_force = 0.0
    maximum_torque = 0.0
    final_wrench: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []
    captured_frames = 0
    for step in range(steps):
        position, rotation, linear, angular = read_rigid_state(view)
        wrench = elastic_tube_wrench(
            position_world=position,
            rotation_world_from_part=rotation,
            linear_velocity_world=linear,
            angular_velocity_world=angular,
            body_axis_origin_world=body_axis_origin_world,
            body_axis_world=body_axis_world,
            tube=tube,
        )
        if tube["enabled"]:
            view.apply_forces_and_torques_at_pos(
                forces=[wrench["force_world_n"].tolist()],
                torques=[wrench["torque_world_nm"].tolist()],
                local_frame=False,
            )
        SimulationManager.step(steps=1)
        position, rotation, linear, angular = read_rigid_state(view)
        orientation = matrix_to_quaternion_wxyz(rotation)
        metrics = place_release_pose_metrics(
            position,
            orientation,
            visual_target_world,
            np.asarray(tube["direction_part"], dtype=np.float64),
        )
        contact = contact_snapshot(view, dt)
        contact_observed |= bool(contact.get("count", 0) > 0)
        contact_paths.update(contact.get("other_actor_paths", []))
        maximum_penetration = max(
            maximum_penetration,
            float(contact.get("max_penetration_m", 0.0)),
        )
        maximum_force = max(
            maximum_force, float(np.linalg.norm(wrench["force_world_n"]))
        )
        maximum_torque = max(
            maximum_torque, float(np.linalg.norm(wrench["torque_world_nm"]))
        )
        final_wrench = wrench
        if tube_visual is not None:
            update_tube_visual(
                tube_visual,
                tube_curve_points(
                    position_world=position,
                    rotation_world_from_part=rotation,
                    body_axis_origin_world=body_axis_origin_world,
                    body_axis_world=body_axis_world,
                    tube=tube,
                ),
            )
        if step % sample_interval == sample_interval - 1 or step == steps - 1:
            samples.append(
                {
                    "time_s": float(step + 1) * dt,
                    **metrics,
                    "linear_speed_mps": float(np.linalg.norm(linear)),
                    "angular_speed_radps": float(np.linalg.norm(angular)),
                    "contact_count": int(contact.get("count", 0)),
                    "tube_radial_deflection_m": float(wrench["radial_deflection_m"]),
                    "tube_bend_angle_deg": float(wrench["bend_angle_deg"]),
                    "tube_elastic_energy_j": float(wrench["elastic_energy_j"]),
                }
            )
        if (
            capture is not None
            and frame_dir is not None
            and (step % capture_interval == capture_interval - 1 or step == steps - 1)
        ):
            from common.isaac_physics_video import _compose

            views = capture.capture()
            snapshot = {
                "position_error_m": metrics["translation_error_m"],
                "rotation_error_deg": metrics["tilt_error_deg"],
            }
            composed = _compose(
                views,
                captured_frames,
                {MANIFEST["simulation"]["inserted_part"]: "physics_pose_refinement"},
                {MANIFEST["simulation"]["inserted_part"]: snapshot},
                {MANIFEST["simulation"]["inserted_part"]: contact},
                [MANIFEST["simulation"]["inserted_part"]],
                False,
            )
            from PIL import ImageDraw

            draw = ImageDraw.Draw(composed, "RGBA")
            label = (
                "Green curve = flexible dip-tube proxy | cyan = visual pose | "
                "no pose controller / no FixedJoint"
            )
            draw.rectangle((8, 121, 760, 146), fill=(0, 0, 0, 165))
            draw.text((16, 126), label, fill=(225, 255, 225, 255))
            composed.save(
                frame_dir / f"{captured_frames:06d}.jpg", quality=94
            )
            captured_frames += 1

    position, rotation, linear, angular = read_rigid_state(view)
    final_pose = np.eye(4, dtype=np.float64)
    final_pose[:3, :3] = rotation
    final_pose[:3, 3] = position
    projected_pose = project_pose_without_axial_yaw(
        visual_target_world,
        final_pose,
        np.asarray(tube["direction_part"], dtype=np.float64),
    )
    final_metrics = place_release_pose_metrics(
        projected_pose[:3, 3],
        matrix_to_quaternion_wxyz(projected_pose[:3, :3]),
        visual_target_world,
        np.asarray(tube["direction_part"], dtype=np.float64),
    )
    score_input = {
        **final_metrics,
        "final_linear_speed_mps": float(np.linalg.norm(linear)),
        "final_angular_speed_radps": float(np.linalg.norm(angular)),
        "maximum_contact_penetration_m": maximum_penetration,
        "contact_observed": contact_observed,
        "final_tube_energy_j": float(final_wrench["elastic_energy_j"] if final_wrench else 0.0),
    }
    scoring = score_physics_pose_candidate(score_input, settings)
    return {
        "name": str(candidate["name"]),
        "mode": "isaac_tube_constrained_pose_projection",
        "input": candidate,
        "initial_T_world_from_part": initial.tolist(),
        "visual_target_T_world_from_part": visual_target_world.tolist(),
        "physical_final_T_world_from_part": final_pose.tolist(),
        "projected_T_world_from_part": projected_pose.tolist(),
        **score_input,
        "maximum_tube_force_n": maximum_force,
        "maximum_tube_torque_nm": maximum_torque,
        "contact_actor_paths": sorted(contact_paths),
        "score": scoring,
        "samples": samples,
        "captured_frames": captured_frames,
    }


def encode_refinement_video(frame_dir: Path, output_path: Path, fps: float) -> None:
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


def capture_replicator(path: Path) -> dict[str, Any]:
    try:
        import omni.replicator.core as rep

        rep.orchestrator.set_capture_on_play(False)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer_dir = path.parent / "_replicator_capture"
        if writer_dir.exists():
            shutil.rmtree(writer_dir)
        writer_dir.mkdir(parents=True)

        render_camera = rep.functional.create.camera(
            position=(0.48, -0.48, 0.36),
            look_at=(0.0, 0.0, 0.02),
            clipping_range=(0.01, 10.0),
            parent="/World",
            name="ReplicatorCamera",
        )
        render_product = rep.create.render_product(
            render_camera,
            (ARGS.width, ARGS.height),
            name="RiceCookerFinal",
            force_new=True,
        )
        for _ in range(60):
            simulation_app.update()
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(writer_dir))
        writer = rep.writers.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        writer.attach(render_product)

        async def capture_once() -> None:
            await rep.orchestrator.step_async(rt_subframes=16, delta_time=0.0)
            await rep.orchestrator.wait_until_complete_async()

        task = asyncio.ensure_future(capture_once())
        for _ in range(3600):
            if task.done():
                break
            simulation_app.update()
        if not task.done():
            task.cancel()
            raise TimeoutError("Replicator capture did not finish within 3600 application updates")
        task.result()
        writer.detach()
        render_product.destroy()

        for _ in range(120):
            candidates = sorted(writer_dir.rglob("rgb*.png"))
            if candidates:
                shutil.copy2(candidates[-1], path)
                break
            simulation_app.update()
        if not path.is_file():
            return {"success": False, "error": "Replicator completed without an RGB output file"}
        result = {"success": True, "path": str(path), "size_bytes": path.stat().st_size}
        try:
            from PIL import Image

            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            result["mean_rgb"] = float(pixels.mean())
            result["pixel_std"] = float(pixels.std())
            result["validator_passed"] = bool(path.stat().st_size >= 150_000 and pixels.mean() > 30.0)
        except Exception as error:
            result["validation_error"] = repr(error)
        return result
    except Exception as error:
        return {"success": False, "error": repr(error)}


def run_insertion(
    args: argparse.Namespace,
    app: Any,
    asset_root: Path,
    runtime_root: Path,
    manifest: dict[str, Any],
) -> None:
    global ARGS, ASSET_ROOT, RUNTIME_ROOT, MANIFEST, simulation_app
    ARGS = args
    ASSET_ROOT = asset_root
    RUNTIME_ROOT = runtime_root
    MANIFEST = manifest
    simulation_app = app
    place_release_only = bool(getattr(args, "place_release_only", False))
    validation_only = bool(getattr(args, "validate_only", False))
    physics_refine_only = bool(getattr(args, "physics_refine_only", False))
    release_validation_only = place_release_only or validation_only
    if validation_only:
        report_value = "qa/assembly_validation_report.json"
    elif place_release_only:
        report_value = "qa/isaac_place_release_report.json"
    elif physics_refine_only:
        report_value = "qa/isaac_physics_pose_refinement_report.json"
    else:
        report_value = MANIFEST["outputs"]["isaac_report"]
    report_path = RUNTIME_ROOT / report_value
    try:
        usd_paths = import_urdf_assets()
        simulation = MANIFEST["simulation"]
        if validation_only and MANIFEST.get("assembly_interface") is None:
            raise ValueError(
                "--validate-only requires a declared assembly_interface in the "
                "export configuration"
            )
        container = simulation["container_part"]
        inserted = simulation["inserted_part"]

        stage_utils.create_new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        rotation_world_from_body = align_vectors(np.asarray(simulation["up_axis_body"]), np.array([0.0, 0.0, 1.0]))
        world_from_body = np.eye(4)
        world_from_body[:3, :3] = rotation_world_from_body
        body_root = add_reference(stage, "/World/BodyAsset", usd_paths[container], world_from_body)
        inserted_root = add_reference(stage, "/World/InsertedAsset", usd_paths[inserted])
        for _ in range(10):
            simulation_app.update()

        deinstanced = {
            container: deinstance_composed_tree(body_root),
            inserted: deinstance_composed_tree(inserted_root),
        }
        for _ in range(4):
            simulation_app.update()

        body_rigid = choose_rigid_prim(body_root, container)
        inserted_rigid = choose_rigid_prim(inserted_root, inserted)
        body_colliders = configure_collision(body_root, "none")
        inserted_colliders = configure_collision(
            inserted_root,
            dynamic_collision_approximation(simulation),
            sdf_resolution=int(simulation.get("sdf_resolution", 192)),
        )
        removed_articulations = {
            container: remove_articulation_apis(body_root),
            inserted: remove_articulation_apis(inserted_root),
        }
        make_body_static(body_rigid)
        material_path = bind_physics_material(stage, body_colliders + inserted_colliders, simulation)

        inserted_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(inserted_rigid)
        inserted_physx.CreateSolverPositionIterationCountAttr().Set(32)
        inserted_physx.CreateSolverVelocityIterationCountAttr().Set(4)
        inserted_physx.CreateEnableCCDAttr().Set(False)
        # PhysX contact sensors require sleeping to be disabled; otherwise a
        # stable resting manifold disappears from tensor contact data even
        # though the body remains supported.
        inserted_physx.CreateSleepThresholdAttr().Set(0.0)
        contact_offsets = configure_contact_offsets(
            stage,
            body_colliders + inserted_colliders,
            simulation,
        )
        # This API must exist before setup_simulation builds the PhysX scene.
        # Applying it afterwards updates USD but does not reliably create the
        # tensor contact sensor for the current run.
        PhysxSchema.PhysxContactReportAPI.Apply(
            inserted_rigid
        ).CreateThresholdAttr().Set(0.0)
        dt = configure_physics_scene(stage, simulation)
        create_lights(stage)
        carb.settings.get_settings().set("/rtx/post/tonemap/op", 4)
        carb.settings.get_settings().set("/rtx/post/tonemap/filmIso", 200.0)

        if validation_only:
            scene_value = "usd/assembly_validation_scene.usda"
        elif place_release_only:
            scene_value = "usd/place_release_scene.usda"
        elif physics_refine_only:
            scene_value = "usd/physics_pose_refinement_scene.usda"
        else:
            scene_value = "usd/insertion_scene.usda"
        scene_path = RUNTIME_ROOT / scene_value
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(scene_path))

        SimulationManager.switch_physics_engine("physx")
        SimulationManager.setup_simulation(dt=dt, device=ARGS.device)
        inserted_view = RigidPrim(
            str(inserted_rigid.GetPath()),
            reset_xform_op_properties=True,
            max_contact_count=4096,
        )

        target_body = np.asarray(
            MANIFEST["assembled_T_body_from_part"][inserted],
            dtype=np.float64,
        ).copy()
        visual_target_body = target_body.copy()
        inserted_axis_part = np.asarray(
            simulation.get("inserted_axis_part", [0.0, 1.0, 0.0]),
            dtype=np.float64,
        )
        inserted_axis_norm = float(np.linalg.norm(inserted_axis_part))
        if inserted_axis_norm <= 1e-12:
            raise ValueError("simulation.inserted_axis_part must be non-zero")
        inserted_axis_part /= inserted_axis_norm
        inserted_axis_body_before = target_body[:3, :3] @ inserted_axis_part
        container_up_body = np.asarray(
            simulation["up_axis_body"], dtype=np.float64
        )
        container_up_body /= np.linalg.norm(container_up_body)
        axis_alignment_deg_before = math.degrees(math.acos(float(np.clip(
            np.dot(inserted_axis_body_before, container_up_body),
            -1.0,
            1.0,
        ))))
        requested_axis_alignment = bool(
            ARGS.align_insert_up_axis or simulation.get("align_insert_up_axis", False)
        )
        align_insert_up_axis = requested_axis_alignment and not validation_only
        if align_insert_up_axis:
            correction = align_vectors(
                inserted_axis_body_before, container_up_body
            )
            target_body[:3, :3] = correction @ target_body[:3, :3]
        inserted_axis_body_after = target_body[:3, :3] @ inserted_axis_part
        axis_alignment_deg_after = math.degrees(math.acos(float(np.clip(
            np.dot(inserted_axis_body_after, container_up_body),
            -1.0,
            1.0,
        ))))
        target_world = world_from_body @ target_body
        visual_target_world = world_from_body @ visual_target_body
        place_release_target_world = target_world.copy()
        place_release_correction = assembly_target_translation(
            simulation,
            part=inserted,
            frame_id=10**9,
            reference_rotation=world_from_body[:3, :3],
        )
        place_release_target_world[:3, 3] += place_release_correction[
            "translation_world_m"
        ]
        replay_result = None
        if not ARGS.skip_replay and not release_validation_only and not physics_refine_only:
            set_kinematic(inserted_rigid, True)
            app_utils.play(commit=True)
            simulation_app.update()
            replay_result = replay_observed_path(inserted_view, world_from_body)
            app_utils.stop()
            for _ in range(4):
                simulation_app.update()
            SimulationManager.invalidate_physics()

        trial_results = []
        place_release_results = []
        physics_refinement_results: list[dict[str, Any]] = []
        physics_refinement_best: dict[str, Any] | None = None
        refined_trajectory: dict[str, Any] | None = None
        refinement_video: dict[str, Any] = {"success": False, "skipped": True}
        tube_visual_report: dict[str, Any] = {}
        physics_hidden_body_visuals: list[str] = []
        resolved_place_release = place_release_settings(simulation)
        resolved_assembly_validation = assembly_validation_settings(simulation)
        resolved_release_settings = (
            resolved_assembly_validation if validation_only else resolved_place_release
        )
        resolved_physics_refinement = physics_pose_refinement_settings(simulation)
        if physics_refine_only:
            if not resolved_physics_refinement["enabled"]:
                raise ValueError(
                    "simulation.physics_pose_refinement.enabled must be true"
                )
            set_kinematic(inserted_rigid, False)
            inserted_physx.CreateEnableCCDAttr().Set(True)
            tube_settings = resolved_physics_refinement["tube"]
            body_axis_world = world_from_body[:3, :3] @ container_up_body
            body_axis_world /= np.linalg.norm(body_axis_world)
            body_axis_origin_body = np.asarray(
                tube_settings["body_axis_origin_body_m"], dtype=np.float64
            )
            body_axis_origin_world = (
                world_from_body[:3, :3] @ body_axis_origin_body
                + world_from_body[:3, 3]
            )
            UsdGeom.Scope.Define(stage, "/World/PhysicsPoseRefinement")
            initial_tube_points = tube_curve_points(
                position_world=target_world[:3, 3],
                rotation_world_from_part=target_world[:3, :3],
                body_axis_origin_world=body_axis_origin_world,
                body_axis_world=body_axis_world,
                tube=tube_settings,
            )
            tube_visual = author_tube_visual(
                stage, initial_tube_points, tube_settings
            )
            tube_visual_report = {
                key: value
                for key, value in tube_visual.items()
                if key != "points_attr"
            }
            target_ghost = author_direct_visual(
                stage,
                "PhysicsPoseTarget",
                ASSET_ROOT / f"meshes/visual/{inserted}/{inserted}.obj",
                target_world,
                (0.05, 0.92, 0.96),
                metallic=0.05,
                roughness=0.32,
                opacity=0.22,
            )
            physics_hidden_body_visuals = hide_imported_visuals(body_root)
            transparent_body = author_direct_visual(
                stage,
                "PhysicsBodyShell",
                ASSET_ROOT / f"meshes/visual/{container}/{container}.obj",
                world_from_body,
                (0.48, 0.10, 0.10),
                metallic=0.02,
                roughness=0.45,
                opacity=0.30,
            )
            tube_visual_report["transparent_body_shell"] = transparent_body
            for _ in range(8):
                simulation_app.update()
            app_utils.play(commit=True)
            simulation_app.update()
            candidates = resolved_physics_refinement["candidates"]
            if ARGS.trial_limit is not None:
                candidates = candidates[: max(0, ARGS.trial_limit)]
            for candidate in candidates:
                result = run_physics_pose_refinement_trial(
                    inserted_view,
                    target_world,
                    candidate,
                    resolved_physics_refinement,
                    body_axis_origin_world,
                    body_axis_world,
                    dt,
                    tube_visual=tube_visual,
                )
                physics_refinement_results.append(result)
                print(
                    f"Physics refinement {result['name']}: "
                    f"accepted={result['score']['accepted']} "
                    f"score={result['score']['score']:.3f} "
                    f"delta={result['translation_error_m'] * 1000.0:.2f}mm "
                    f"tilt={result['tilt_error_deg']:.2f}deg "
                    f"contact={result['contact_observed']}",
                    flush=True,
                )
            if not physics_refinement_results:
                raise RuntimeError("physics pose refinement ran no candidates")
            accepted = [
                result
                for result in physics_refinement_results
                if result["score"]["accepted"]
            ]
            physics_refinement_best = min(
                accepted or physics_refinement_results,
                key=lambda result: float(result["score"]["score"]),
            )
            selected_candidate = physics_refinement_best["input"]
            if ARGS.capture:
                from common.isaac_video import MultiViewCapture

                frame_dir = RUNTIME_ROOT / "video/physics_pose_refinement_frames"
                if frame_dir.exists():
                    shutil.rmtree(frame_dir)
                frame_dir.mkdir(parents=True)
                capture_device = MultiViewCapture(
                    simulation_app, int(getattr(ARGS, "rt_subframes", 1))
                )
                try:
                    captured = run_physics_pose_refinement_trial(
                        inserted_view,
                        target_world,
                        selected_candidate,
                        resolved_physics_refinement,
                        body_axis_origin_world,
                        body_axis_world,
                        dt,
                        tube_visual=tube_visual,
                        capture=capture_device,
                        frame_dir=frame_dir,
                        capture_fps=float(ARGS.refinement_video_fps),
                    )
                finally:
                    capture_device.close()
                captured["score"] = score_physics_pose_candidate(
                    captured, resolved_physics_refinement
                )
                physics_refinement_best = captured
                video_path = (
                    RUNTIME_ROOT / "video/physics_pose_refinement.mp4"
                )
                encode_refinement_video(
                    frame_dir,
                    video_path,
                    float(ARGS.refinement_video_fps),
                )
                refinement_video = {
                    "success": True,
                    "path": str(video_path),
                    "fps": float(ARGS.refinement_video_fps),
                    "frame_count": int(captured["captured_frames"]),
                    "duration_s": float(captured["captured_frames"])
                    / float(ARGS.refinement_video_fps),
                    "target_ghost": target_ghost,
                }
                shutil.rmtree(frame_dir)

            if physics_refinement_best["score"]["accepted"]:
                trajectory_value = Path(MANIFEST["inputs"]["trajectory"])
                trajectory_path = (
                    trajectory_value
                    if trajectory_value.is_absolute()
                    else Path(__file__).resolve().parents[1] / trajectory_value
                )
                trajectory = json.loads(
                    trajectory_path.read_text(encoding="utf-8")
                )
                projected_world = np.asarray(
                    physics_refinement_best["projected_T_world_from_part"],
                    dtype=np.float64,
                )
                refined_body = np.linalg.inv(world_from_body) @ projected_world
                refined_trajectory = write_physics_refined_trajectory(
                    trajectory,
                    moving_part=inserted,
                    reference_part=container,
                    visual_body_pose=target_body,
                    refined_body_pose=refined_body,
                    apply_frame_range=resolved_physics_refinement[
                        "apply_frame_range"
                    ],
                    report_path=report_path,
                    output_path=(
                        RUNTIME_ROOT
                        / "pose/trajectory_physics_refined.json"
                    ),
                )
            app_utils.stop()
            for _ in range(4):
                simulation_app.update()
        elif release_validation_only:
            set_kinematic(inserted_rigid, False)
            inserted_physx.CreateEnableCCDAttr().Set(True)
            for _ in range(4):
                simulation_app.update()
            app_utils.play(commit=True)
            simulation_app.update()
            active_release_target = (
                visual_target_world if validation_only else place_release_target_world
            )
            trials = resolved_release_settings["trials"]
            if ARGS.trial_limit is not None:
                trials = trials[:max(0, ARGS.trial_limit)]
            for trial in trials:
                result = run_place_release_trial(
                    inserted_view,
                    active_release_target,
                    inserted_axis_part,
                    trial,
                    resolved_release_settings,
                    dt,
                )
                place_release_results.append(result)
                print(
                    f"Place-release {result['name']}: "
                    f"success={result['success']} "
                    f"lateral={result['lateral_error_m'] * 1000.0:.2f}mm "
                    f"axial={result['axial_error_signed_m'] * 1000.0:.2f}mm "
                    f"tilt={result['tilt_error_deg']:.2f}deg "
                    f"contact={result['contact_gate']['sustained']}",
                    flush=True,
                )
            app_utils.stop()
            for _ in range(4):
                simulation_app.update()
        elif not ARGS.skip_drop:
            set_kinematic(inserted_rigid, False)
            inserted_physx.CreateEnableCCDAttr().Set(True)
            for _ in range(4):
                simulation_app.update()
            app_utils.play(commit=True)
            simulation_app.update()
            trials = simulation["trials"]
            if ARGS.trial_limit is not None:
                trials = trials[:max(0, ARGS.trial_limit)]
            for trial in trials:
                result = run_drop_trial(inserted_view, target_world, trial, simulation, dt)
                trial_results.append(result)
                print(
                    f"Trial {result['name']}: success={result['success']} "
                    f"translation={result['translation_error_m']:.4f}m "
                    f"rotation={result['rotation_error_deg']:.2f}deg"
                )
            app_utils.stop()
            for _ in range(4):
                simulation_app.update()

        if release_validation_only or physics_refine_only:
            hidden_imported_visuals = {
                container: physics_hidden_body_visuals,
                inserted: [],
            }
            demo_visuals = {}
        else:
            set_kinematic(inserted_rigid, True)
            set_pose(inserted_view, target_world, zero_velocity=False)
            for _ in range(20):
                simulation_app.update()
            hidden_imported_visuals = {
                container: hide_imported_visuals(body_root),
                inserted: hide_imported_visuals(inserted_root),
            }
            demo_visuals = author_assembly_demo_visuals(
                stage,
                world_from_body,
                target_world,
                container,
                inserted,
            )
            for _ in range(20):
                simulation_app.update()
        stage.GetRootLayer().Export(str(scene_path))
        capture_name = (
            "qa/assembly_validation_final.png"
            if validation_only
            else "qa/isaac_place_release_final.png"
            if place_release_only
            else "qa/isaac_final.png"
        )
        capture = (
            refinement_video
            if physics_refine_only
            else (
                capture_replicator(RUNTIME_ROOT / capture_name)
                if ARGS.capture
                else {"success": False, "skipped": True}
            )
        )
        successes = sum(bool(result["success"]) for result in trial_results)
        place_release_successes = sum(
            bool(result["success"]) for result in place_release_results
        )
        validation_summary = (
            summarize_validation_trials(
                place_release_results, resolved_assembly_validation
            )
            if validation_only
            else None
        )
        if physics_refine_only:
            active_results = physics_refinement_results
            active_successes = sum(
                bool(result["score"]["accepted"])
                for result in physics_refinement_results
            )
        else:
            active_results = (
                place_release_results if release_validation_only else trial_results
            )
            active_successes = (
                place_release_successes if release_validation_only else successes
            )
        report = {
            "schema_version": 1,
            "status": "complete",
            "isaac_version_target": "6.0+",
            "physics_engine": "physx",
            "device": ARGS.device,
            "asset_root": str(ASSET_ROOT),
            "runtime_output_root": str(RUNTIME_ROOT),
            "usd_assets": usd_paths,
            "deinstanced_prims": deinstanced,
            "removed_articulation_apis": removed_articulations,
            "scene_usd": str(scene_path),
            "body_rigid_removed_for_static_triangle_mesh": str(body_rigid.GetPath()),
            "inserted_rigid_body": str(inserted_rigid.GetPath()),
            "body_colliders": body_colliders,
            "inserted_colliders": inserted_colliders,
            "hidden_imported_visuals": hidden_imported_visuals,
            "demo_visuals": demo_visuals,
            "physics_material": material_path,
            "contact_offsets": contact_offsets,
            "world_from_body": world_from_body.tolist(),
            "target_T_world_from_inserted": target_world.tolist(),
            "frozen_visual_target_T_world_from_inserted": visual_target_world.tolist(),
            "place_release_target_T_world_from_inserted": (
                place_release_target_world.tolist()
            ),
            "place_release_target_correction": {
                "enabled": bool(place_release_correction["enabled"]),
                "fraction": float(place_release_correction["fraction"]),
                "translation_reference_m": np.asarray(
                    place_release_correction["translation_reference_m"],
                    dtype=float,
                ).tolist(),
                "translation_world_m": np.asarray(
                    place_release_correction["translation_world_m"],
                    dtype=float,
                ).tolist(),
                "source": place_release_correction["source"],
            },
            "insert_axis_alignment": {
                "diagnostic_override_enabled": bool(
                    ARGS.align_insert_up_axis
                ),
                "configured_alignment_enabled": bool(
                    simulation.get("align_insert_up_axis", False)
                ),
                "requested_but_ignored_by_validation": bool(
                    validation_only and requested_axis_alignment
                ),
                "angle_before_deg": axis_alignment_deg_before,
                "angle_after_deg": axis_alignment_deg_after,
                "inserted_axis_part": inserted_axis_part.tolist(),
                "center_body_m": target_body[:3, 3].tolist(),
            },
            "observed_replay": replay_result,
            "drop_trials": trial_results,
            "place_release": {
                "enabled": place_release_only,
                "controller_forces_enabled": False,
                "preload_enabled": False,
                "fixed_joint_enabled": False,
                "settings": resolved_place_release,
                "trials": place_release_results if place_release_only else [],
            },
            "assembly_validation": {
                "enabled": validation_only,
                "protocol": resolved_assembly_validation,
                "assembly_interface": MANIFEST.get("assembly_interface"),
                "target_pose_correction_applied": False,
                "axis_alignment_applied": False,
                "trajectory_rewritten": False,
                "controller_forces_enabled": False,
                "preload_enabled": False,
                "fixed_joint_enabled": False,
                "trials": place_release_results if validation_only else [],
                "summary": validation_summary,
            },
            "physics_pose_refinement": {
                "enabled": physics_refine_only,
                "perception_policy": {
                    "icp_uses_dip_tube": False,
                    "dip_tube_pose_source_before_assembly": "nozzle_mount_transform",
                    "dip_tube_pose_source_during_assembly": "bottle_axis_and_elastic_state",
                },
                "controller_forces_enabled": False,
                "tube_elastic_force_enabled": bool(
                    resolved_physics_refinement["tube"]["enabled"]
                ),
                "preload_enabled": False,
                "fixed_joint_enabled": False,
                "collision_geometry": {
                    container: MANIFEST["parts"][container]["collision_proxy"],
                    inserted: MANIFEST["parts"][inserted]["collision_proxy"],
                },
                "tube_visual": tube_visual_report,
                "settings": resolved_physics_refinement,
                "candidates": physics_refinement_results,
                "selected": physics_refinement_best,
                "accepted": bool(
                    physics_refinement_best
                    and physics_refinement_best["score"]["accepted"]
                ),
                "refined_trajectory": refined_trajectory,
            },
            "summary": {
                "mode": (
                    "isaac_tube_constrained_pose_projection"
                    if physics_refine_only
                    else (
                        "frozen_visual_pose_assembly_validation"
                        if validation_only
                        else "passive_place_and_release"
                        if place_release_only
                        else "legacy_drop"
                    )
                ),
                "trial_count": len(active_results),
                "success_count": active_successes,
                "success_rate": (
                    float(active_successes / len(active_results))
                    if active_results
                    else None
                ),
                "criteria": (
                    {
                        key: resolved_physics_refinement[key]
                        for key in (
                            "maximum_visual_translation_m",
                            "maximum_visual_tilt_deg",
                            "maximum_final_linear_speed_mps",
                            "maximum_final_angular_speed_radps",
                            "maximum_penetration_m",
                            "require_contact",
                        )
                    }
                    if physics_refine_only
                    else (
                    {
                        key: resolved_release_settings[key]
                        for key in (
                            "maximum_lateral_error_m",
                            "maximum_axial_error_m",
                            "maximum_tilt_error_deg",
                            "maximum_final_linear_speed_mps",
                            "maximum_final_angular_speed_radps",
                            "contact_window_seconds",
                            "minimum_contact_fraction",
                            "maximum_contact_gap_seconds",
                        )
                    }
                    if release_validation_only
                    else {
                        key: simulation[key]
                        for key in (
                            "success_translation_m",
                            "success_rotation_deg",
                            "success_linear_speed_mps",
                            "success_angular_speed_radps",
                        )
                    })
                ),
            },
            "capture": capture,
            "limitations": [
                "Assembly validation establishes feasibility only under the declared collision geometry and physical assumptions; it is not ground-truth pose evidence.",
                "The physical projection is active only during the configured assembly phase; the dip tube never enters ICP or render-loss observations.",
                "Mass, inertia, and friction are configured assumptions and should be replaced by measurements when dynamics, rather than fit, becomes the target.",
                "The dip tube is a stage-aware elastic rod proxy. Its force constrains lateral translation and tilt, while axial yaw remains vision-derived.",
                "Low-poly connector proxies, not raw reconstruction fragments, define the bottle seat and nozzle contact surface.",
            ],
        }
        write_json(report_path, report)
        if validation_only:
            stage_record = "stages/assembly_validation.complete.json"
            failure_record = "stages/assembly_validation.failed.json"
        elif place_release_only:
            stage_record = "stages/isaac_place_release.complete.json"
            failure_record = "stages/isaac_place_release.failed.json"
        elif physics_refine_only:
            stage_record = "stages/isaac_physics_pose_refinement.complete.json"
            failure_record = "stages/isaac_physics_pose_refinement.failed.json"
        else:
            stage_record = "stages/isaac.complete.json"
            failure_record = "stages/isaac.failed.json"
        write_json(
            RUNTIME_ROOT / stage_record,
            {"stage": "isaac", "status": "complete", "report": str(report_path), "success_rate": report["summary"]["success_rate"]},
        )
        (RUNTIME_ROOT / failure_record).unlink(missing_ok=True)
        print(f"Isaac report: {report_path}")
    except Exception as error:
        failure = {"schema_version": 1, "status": "failed", "error": repr(error)}
        write_json(report_path, failure)
        if validation_only:
            failure_record = "stages/assembly_validation.failed.json"
        elif place_release_only:
            failure_record = "stages/isaac_place_release.failed.json"
        elif physics_refine_only:
            failure_record = "stages/isaac_physics_pose_refinement.failed.json"
        else:
            failure_record = "stages/isaac.failed.json"
        write_json(RUNTIME_ROOT / failure_record, failure)
        raise
