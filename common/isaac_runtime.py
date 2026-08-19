"""Isaac Sim USD import, trajectory replay, and insertion validation."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


from common.io_utils import write_json
from common.physics_control import dynamic_collision_approximation

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
    urdfs["rice_cooker_display"] = ASSET_ROOT / MANIFEST["outputs"]["display_urdf"]
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
            fix_base=name == "rice_cooker_display",
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
    report_path = RUNTIME_ROOT / MANIFEST["outputs"]["isaac_report"]
    try:
        usd_paths = import_urdf_assets()
        simulation = MANIFEST["simulation"]
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
        dt = configure_physics_scene(stage, simulation)
        create_lights(stage)
        carb.settings.get_settings().set("/rtx/post/tonemap/op", 4)
        carb.settings.get_settings().set("/rtx/post/tonemap/filmIso", 200.0)

        scene_path = RUNTIME_ROOT / "usd/insertion_scene.usda"
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(scene_path))

        SimulationManager.switch_physics_engine("physx")
        SimulationManager.setup_simulation(dt=dt, device=ARGS.device)
        inserted_view = RigidPrim(str(inserted_rigid.GetPath()), reset_xform_op_properties=True)

        target_body = np.asarray(
            MANIFEST["assembled_T_body_from_part"][inserted],
            dtype=np.float64,
        ).copy()
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
        align_insert_up_axis = bool(
            ARGS.align_insert_up_axis
            or simulation.get("align_insert_up_axis", False)
        )
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
        replay_result = None
        if not ARGS.skip_replay:
            set_kinematic(inserted_rigid, True)
            app_utils.play(commit=True)
            simulation_app.update()
            replay_result = replay_observed_path(inserted_view, world_from_body)
            app_utils.stop()
            for _ in range(4):
                simulation_app.update()
            SimulationManager.invalidate_physics()

        trial_results = []
        if not ARGS.skip_drop:
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
        capture = capture_replicator(RUNTIME_ROOT / "qa/isaac_final.png") if ARGS.capture else {"success": False, "skipped": True}
        successes = sum(bool(result["success"]) for result in trial_results)
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
            "world_from_body": world_from_body.tolist(),
            "target_T_world_from_inserted": target_world.tolist(),
            "insert_axis_alignment": {
                "diagnostic_override_enabled": bool(
                    ARGS.align_insert_up_axis
                ),
                "configured_alignment_enabled": bool(
                    simulation.get("align_insert_up_axis", False)
                ),
                "angle_before_deg": axis_alignment_deg_before,
                "angle_after_deg": axis_alignment_deg_after,
                "inserted_axis_part": inserted_axis_part.tolist(),
                "center_body_m": target_body[:3, 3].tolist(),
            },
            "observed_replay": replay_result,
            "drop_trials": trial_results,
            "summary": {
                "trial_count": len(trial_results),
                "success_count": successes,
                "success_rate": float(successes / len(trial_results)) if trial_results else None,
                "criteria": {
                    key: simulation[key]
                    for key in (
                        "success_translation_m",
                        "success_rotation_deg",
                        "success_linear_speed_mps",
                        "success_angular_speed_radps",
                    )
                },
            },
            "capture": capture,
            "limitations": [
                "Collision results validate the reconstructed meshes, scales, and solved relative poses; they do not prove real-world manufacturing clearance.",
                "Mass, inertia, and friction are configured assumptions and should be replaced by measurements when dynamics, rather than fit, becomes the target.",
                "The body uses a static triangle mesh to preserve its cavity; the dynamic inner part uses PhysX convex decomposition.",
            ],
        }
        write_json(report_path, report)
        write_json(
            RUNTIME_ROOT / "stages/isaac.complete.json",
            {"stage": "isaac", "status": "complete", "report": str(report_path), "success_rate": report["summary"]["success_rate"]},
        )
        (RUNTIME_ROOT / "stages/isaac.failed.json").unlink(missing_ok=True)
        print(f"Isaac report: {report_path}")
    except Exception as error:
        failure = {"schema_version": 1, "status": "failed", "error": repr(error)}
        write_json(report_path, failure)
        write_json(RUNTIME_ROOT / "stages/isaac.failed.json", failure)
        raise
