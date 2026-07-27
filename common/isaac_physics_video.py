"""Render a complete, force-controlled PhysX trajectory in Isaac Sim.

The reconstructed textured meshes are the simulated rigid bodies.  The
transparent meshes show pose-solver targets only; target poses are never
written directly to an active rigid body after its first observable frame.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import carb
import numpy as np
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from common.io_utils import write_json
from common.physics_control import (
    select_control_profile,
    settled_contact_settings,
)
import common.isaac_runtime as runtime
from common.isaac_runtime import (
    add_reference,
    align_vectors,
    bind_physics_material,
    choose_rigid_prim,
    configure_collision,
    configure_physics_scene,
    deinstance_composed_tree,
    make_body_static,
    matrix_to_quaternion_wxyz,
    np_to_gf_matrix,
    read_obj_mesh,
    remove_articulation_apis,
    set_kinematic,
    set_pose,
)
CONTROLLERS = {
    "inner_pot": {
        "inertia_scale": 0.0032,
        "force_limit_n": 12.0,
        "torque_limit_nm": 0.35,
    },
    "lid": {
        "inertia_scale": 0.0038,
        "force_limit_n": 16.0,
        "torque_limit_nm": 0.45,
    },
}
TARGET_COLORS = {
    "inner_pot": (0.05, 0.92, 0.96),
    "lid": (1.0, 0.12, 0.72),
}
UNOBSERVABLE_STATES = {"inferred_unobservable", "unobservable", "unknown"}


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.is_file()
        else ImageFont.load_default()
    )


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
    missing = [
        part
        for part in parts
        if part not in paths or not paths[part].is_file()
    ]
    if missing:
        raise RuntimeError(f"USD cache is missing parts: {missing}")
    return paths


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


def _set_visibility(prim: Usd.Prim, visible: bool) -> None:
    imageable = UsdGeom.Imageable(prim)
    imageable.MakeVisible() if visible else imageable.MakeInvisible()


def _set_collisions(stage: Usd.Stage, paths: list[str], enabled: bool) -> None:
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(enabled)


def _world_from_part(
    frame_record: dict[str, Any],
    part: str,
    reference_part: str,
    world_from_body: np.ndarray,
) -> np.ndarray:
    if part == reference_part:
        return world_from_body.copy()
    record = frame_record["parts"][part]
    if "T_body_from_part" in record:
        body_from_part = np.asarray(record["T_body_from_part"], dtype=np.float64)
    else:
        body_world = np.asarray(
            frame_record["parts"][reference_part]["T_world_from_part"],
            dtype=np.float64,
        )
        part_world = np.asarray(record["T_world_from_part"], dtype=np.float64)
        body_from_part = np.linalg.inv(body_world) @ part_world
    return world_from_body @ body_from_part


def _quat_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_error_vector(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = target @ actual.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if math.pi - angle < 1e-5:
        quaternion = matrix_to_quaternion_wxyz(relative)
        if quaternion[0] < 0.0:
            quaternion = -quaternion
        vector_norm = float(np.linalg.norm(quaternion[1:]))
        return (
            quaternion[1:] / max(vector_norm, 1e-12)
        ) * (2.0 * math.atan2(vector_norm, quaternion[0]))
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def _clip_vector(vector: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= limit:
        return vector, False
    return vector * (limit / max(magnitude, 1e-12)), True


def _read_state(view: RigidPrim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, orientations = view.get_world_poses()
    linear, angular = view.get_velocities()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = positions.numpy()[0]
    pose[:3, :3] = _quat_to_matrix(orientations.numpy()[0])
    return pose, linear.numpy()[0].astype(np.float64), angular.numpy()[0].astype(np.float64)


def _control(
    view: RigidPrim,
    target: np.ndarray,
    parameters: dict[str, float],
    natural_frequency: float,
    damping_ratio: float,
    mode: str,
) -> dict[str, Any]:
    actual, linear_velocity, angular_velocity = _read_state(view)
    mass = float(parameters["mass_kg"])
    inertia = float(parameters["inertia_scale"])
    position_error = target[:3, 3] - actual[:3, 3]
    rotation_error = _rotation_error_vector(actual[:3, :3], target[:3, :3])
    force = (
        mass * natural_frequency**2 * position_error
        - 2.0 * damping_ratio * mass * natural_frequency * linear_velocity
        + np.array([0.0, 0.0, mass * 9.81])
    )
    torque = (
        inertia * natural_frequency**2 * rotation_error
        - 2.0
        * damping_ratio
        * inertia
        * natural_frequency
        * angular_velocity
    )
    force, force_saturated = _clip_vector(force, float(parameters["force_limit_n"]))
    torque, torque_saturated = _clip_vector(
        torque, float(parameters["torque_limit_nm"])
    )
    view.apply_forces_and_torques_at_pos(
        forces=[force.tolist()],
        torques=[torque.tolist()],
        local_frame=False,
    )
    return {
        "force_world_n": force.tolist(),
        "torque_world_nm": torque.tolist(),
        "force_saturated": force_saturated,
        "torque_saturated": torque_saturated,
        "controller_mode": mode,
        "controller_frequency_radps": natural_frequency,
        "controller_damping_ratio": damping_ratio,
    }


def _snapshot(
    view: RigidPrim,
    target: np.ndarray,
    control: dict[str, Any],
) -> dict[str, Any]:
    actual, linear_velocity, angular_velocity = _read_state(view)
    rotation_error = _rotation_error_vector(actual[:3, :3], target[:3, :3])
    return {
        "T_world_from_part": actual.tolist(),
        "linear_velocity_mps": linear_velocity.tolist(),
        "angular_velocity_radps": angular_velocity.tolist(),
        "position_error_m": float(np.linalg.norm(target[:3, 3] - actual[:3, 3])),
        "rotation_error_deg": math.degrees(float(np.linalg.norm(rotation_error))),
        **control,
    }


def _contact_snapshot(view: RigidPrim, dt: float) -> dict[str, Any]:
    try:
        forces, _points, _normals, separations, counts, starts, actor_ids = (
            view.get_raw_contact_data(dt=dt)
        )
        count = int(np.asarray(counts.numpy()).sum())
        start = (
            int(np.asarray(starts.numpy()).reshape(-1)[0])
            if count
            else 0
        )
        contact_slice = slice(start, start + count)
        force_values = forces.numpy()[contact_slice]
        separation_values = separations.numpy()[contact_slice]
        other_actor_paths = (
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
        max_force = (
            float(np.linalg.norm(force_values, axis=-1).max())
            if force_values.size
            else 0.0
        )
        penetration = (
            max(0.0, -float(separation_values.min()))
            if separation_values.size
            else 0.0
        )
        return {
            "available": True,
            "count": count,
            "max_force_n": max_force,
            "max_penetration_m": penetration,
            "other_actor_paths": other_actor_paths,
        }
    except Exception as error:
        return {"available": False, "error": repr(error), "count": 0}


def _configure_contact_offsets(
    stage: Usd.Stage,
    collider_paths: list[str],
    simulation: dict[str, Any],
) -> dict[str, float] | None:
    contact_offset = simulation.get("contact_offset_m")
    rest_offset = simulation.get("rest_offset_m")
    if contact_offset is None and rest_offset is None:
        return None
    contact_offset = float(
        0.001 if contact_offset is None else contact_offset
    )
    rest_offset = float(0.0 if rest_offset is None else rest_offset)
    if contact_offset <= rest_offset:
        raise ValueError(
            "simulation.contact_offset_m must be greater than rest_offset_m"
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


def _bind_target_material(
    stage: Usd.Stage,
    prim: Usd.Prim,
    name: str,
    color: tuple[float, float, float],
) -> None:
    path = f"/World/TargetMaterials/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(0.18 * np.asarray(color)))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.24)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.32)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _create_target_ghost(
    stage: Usd.Stage,
    asset_root: Path,
    part: str,
    collision_mesh: str,
) -> dict[str, Any]:
    root = UsdGeom.Xform.Define(stage, f"/World/Targets/{part}")
    transform_op = root.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    points, counts, indices = read_obj_mesh(asset_root / collision_mesh)
    mesh = UsdGeom.Mesh.Define(stage, f"/World/Targets/{part}/Mesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    _bind_target_material(stage, mesh.GetPrim(), part, TARGET_COLORS[part])
    _set_visibility(root.GetPrim(), False)
    return {"root": root.GetPrim(), "transform_op": transform_op}


def _create_environment(stage: Usd.Stage) -> list[str]:
    UsdGeom.Scope.Define(stage, "/World/PhysicsMaterials")
    UsdGeom.Scope.Define(stage, "/World/TargetMaterials")
    UsdGeom.Scope.Define(stage, "/World/Targets")
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
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    return [str(floor.GetPath())]


def _compose(
    views: dict[str, Image.Image],
    frame_id: int,
    states: dict[str, str],
    snapshots: dict[str, dict[str, Any]],
    contacts: dict[str, dict[str, Any]],
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), (22, 22, 24))
    canvas.paste(views["perspective"], (0, 0))
    canvas.paste(views["top"], (640, 0))
    canvas.paste(views["side"], (640, 360))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, 1280, 63), fill=(0, 0, 0, 200))
    draw.text(
        (18, 12),
        f"PhysX force-driven trajectory | frame {frame_id:03d}",
        font=FONT_LARGE,
        fill=(255, 255, 255, 255),
    )
    lines = [
        "Textured = actual PhysX pose | cyan = inner target | magenta = lid target",
    ]
    for part in ("inner_pot", "lid"):
        if part not in snapshots:
            lines.append(f"{part}: {states.get(part, 'not started')} | inactive")
            continue
        sample = snapshots[part]
        contact = contacts[part]
        lines.append(
            f"{part}: err {1000.0 * sample['position_error_m']:.1f} mm / "
            f"{sample['rotation_error_deg']:.1f} deg | "
            f"contacts {contact.get('count', 0)} | "
            f"max force {contact.get('max_force_n', 0.0):.1f} N"
        )
    y = 65
    for line in lines:
        width = min(1240, 20 + int(draw.textlength(line, font=FONT_SMALL)))
        draw.rectangle((8, y, 8 + width, y + 27), fill=(0, 0, 0, 165))
        draw.text((16, y + 3), line, font=FONT_SMALL, fill=(245, 245, 245, 255))
        y += 29
    draw.rectangle((2, 2, 1277, 717), outline=(238, 154, 46, 255), width=5)
    draw.text((654, 328), "TOP", font=FONT_MEDIUM, fill=(255, 255, 255, 230))
    draw.text((654, 688), "SIDE", font=FONT_MEDIUM, fill=(255, 255, 255, 230))
    return canvas


def render_complete_physics_video(
    args: argparse.Namespace,
    app: Any,
    asset_root: Path,
    runtime_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / "frames/physics_driven"
    capture_enabled = not bool(getattr(args, "no_capture", False))
    if capture_enabled:
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True)

    parts = list(trajectory["parts"])
    reference_part = str(manifest["reference_part"])
    dynamic_parts = [part for part in parts if part != reference_part]
    missing_controllers = [part for part in dynamic_parts if part not in CONTROLLERS]
    if missing_controllers:
        raise RuntimeError(f"Physics controller parameters are missing: {missing_controllers}")
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
    UsdGeom.Scope.Define(stage, "/World/PhysicsAssets")
    floor_colliders = _create_environment(stage)

    roots = {
        reference_part: add_reference(
            stage,
            f"/World/PhysicsAssets/{reference_part}",
            str(usd_paths[reference_part]),
            world_from_body,
        )
    }
    roots.update(
        {
            part: add_reference(
                stage,
                f"/World/PhysicsAssets/{part}",
                str(usd_paths[part]),
            )
            for part in dynamic_parts
        }
    )
    runtime.simulation_app = app
    for _ in range(12):
        app.update()
    for root in roots.values():
        deinstance_composed_tree(root)
    for _ in range(4):
        app.update()

    rigid_prims = {
        part: choose_rigid_prim(roots[part], part)
        for part in parts
    }
    collider_paths = {
        reference_part: configure_collision(roots[reference_part], "none"),
        **{
            part: configure_collision(roots[part], "convexDecomposition")
            for part in dynamic_parts
        },
    }
    for root in roots.values():
        remove_articulation_apis(root)
    make_body_static(rigid_prims[reference_part])
    all_colliders = floor_colliders + [
        path for values in collider_paths.values() for path in values
    ]
    bind_physics_material(stage, all_colliders, simulation)
    contact_offsets = _configure_contact_offsets(
        stage,
        all_colliders,
        simulation,
    )

    for part in dynamic_parts:
        physx = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prims[part])
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(
            rigid_prims[part]
        )
        contact_report.CreateThresholdAttr().Set(0.0)
        physx.CreateSolverPositionIterationCountAttr().Set(32)
        physx.CreateSolverVelocityIterationCountAttr().Set(4)
        physx.CreateEnableCCDAttr().Set(False)
        physx.CreateLinearDampingAttr().Set(0.08)
        physx.CreateAngularDampingAttr().Set(0.08)
        physx.CreateMaxDepenetrationVelocityAttr().Set(0.35)
        physx.CreateMaxLinearVelocityAttr().Set(0.8)
        physx.CreateMaxAngularVelocityAttr().Set(8.0)
        set_kinematic(rigid_prims[part], True)
        _set_collisions(stage, collider_paths[part], False)
        _set_visibility(roots[part], False)

    ghosts = {
        part: _create_target_ghost(
            stage,
            asset_root,
            part,
            str(manifest["parts"][part]["collision_mesh"]),
        )
        for part in dynamic_parts
    }
    _set_visibility(roots[reference_part], False)
    dt = configure_physics_scene(stage, simulation)
    if capture_enabled:
        carb.settings.get_settings().set("/rtx/post/tonemap/op", 4)
        carb.settings.get_settings().set("/rtx/post/tonemap/filmIso", 200.0)

    SimulationManager.switch_physics_engine("physx")
    SimulationManager.setup_simulation(dt=dt, device=str(args.device))
    views = {
        part: RigidPrim(
            str(rigid_prims[part].GetPath()),
            reset_xform_op_properties=True,
            max_contact_count=4096,
        )
        for part in dynamic_parts
    }
    controller_parameters = {}
    for part in dynamic_parts:
        controller_parameters[part] = {
            **CONTROLLERS[part],
            "mass_kg": float(manifest["parts"][part]["mass_kg"]),
        }
    settled_settings = settled_contact_settings(simulation)

    trajectory_ids = sorted(int(frame_id) for frame_id in trajectory["frames"])
    output_start = int(args.start_frame) if args.start_frame is not None else 0
    output_end = (
        int(args.end_frame)
        if args.end_frame is not None
        else trajectory_ids[-1]
    )
    if output_start < 0 or output_end < output_start:
        raise ValueError(f"Invalid output range: {output_start}..{output_end}")
    physics_steps = max(1, int(round(1.0 / (dt * float(args.fps)))))
    active = {part: False for part in dynamic_parts}
    contact_latched = {part: False for part in dynamic_parts}
    last_targets: dict[str, np.ndarray] = {}
    samples: list[dict[str, Any]] = []

    app_utils.play(commit=True)
    app.update()
    if capture_enabled:
        from common.isaac_video import MultiViewCapture

        capture = MultiViewCapture(app, int(args.rt_subframes))
    else:
        capture = None
    output_index = 0
    try:
        for frame_id in range(0, output_end + 1):
            key = f"{frame_id:06d}"
            frame_record = trajectory["frames"].get(key)
            states = {part: "not_started" for part in dynamic_parts}
            if frame_record is not None:
                _set_visibility(roots[reference_part], True)
                for part in dynamic_parts:
                    states[part] = str(
                        frame_record["parts"][part].get("state", "unknown")
                    )
                    target = _world_from_part(
                        frame_record,
                        part,
                        reference_part,
                        world_from_body,
                    )
                    last_targets[part] = target
                    ghosts[part]["transform_op"].Set(np_to_gf_matrix(target))
                    observable = states[part] not in UNOBSERVABLE_STATES
                    _set_visibility(ghosts[part]["root"], observable)
                    if observable and not active[part]:
                        set_pose(views[part], target, zero_velocity=False)
                        set_kinematic(rigid_prims[part], False)
                        views[part].set_velocities(
                            linear_velocities=[[0.0, 0.0, 0.0]],
                            angular_velocities=[[0.0, 0.0, 0.0]],
                        )
                        _set_collisions(stage, collider_paths[part], True)
                        PhysxSchema.PhysxRigidBodyAPI.Apply(
                            rigid_prims[part]
                        ).CreateEnableCCDAttr().Set(True)
                        _set_visibility(roots[part], True)
                        active[part] = True

            latest_controls = {
                part: {
                    "force_world_n": [0.0, 0.0, 0.0],
                    "torque_world_nm": [0.0, 0.0, 0.0],
                    "force_saturated": False,
                    "torque_saturated": False,
                    "controller_mode": "inactive",
                    "controller_frequency_radps": 0.0,
                    "controller_damping_ratio": 0.0,
                }
                for part in dynamic_parts
            }
            if frame_record is not None:
                for _ in range(physics_steps):
                    for part in dynamic_parts:
                        if active[part]:
                            current_pose, _linear, _angular = _read_state(
                                views[part]
                            )
                            current_position_error = float(
                                np.linalg.norm(
                                    last_targets[part][:3, 3]
                                    - current_pose[:3, 3]
                                )
                            )
                            if states[part] not in settled_settings["states"]:
                                contact_latched[part] = False
                            elif (
                                settled_settings["enabled"]
                                and not contact_latched[part]
                                and current_position_error
                                <= settled_settings[
                                    "maximum_position_error_m"
                                ]
                            ):
                                contact_latched[part] = (
                                    _contact_snapshot(views[part], dt).get(
                                        "count", 0
                                    )
                                    > 0
                                )
                            profile = select_control_profile(
                                state=states[part],
                                contact_latched=contact_latched[part],
                                position_error_m=current_position_error,
                                tracking_frequency_radps=float(
                                    args.controller_frequency
                                ),
                                settled_settings=settled_settings,
                            )
                            latest_controls[part] = _control(
                                views[part],
                                last_targets[part],
                                controller_parameters[part],
                                float(profile["frequency_radps"]),
                                float(profile["damping_ratio"]),
                                str(profile["mode"]),
                            )
                    SimulationManager.step(steps=1)

            if frame_id < output_start:
                continue
            snapshots = {
                part: _snapshot(
                    views[part],
                    last_targets[part],
                    latest_controls[part],
                )
                for part in dynamic_parts
                if active[part]
            }
            contacts = {
                part: _contact_snapshot(views[part], dt)
                for part in dynamic_parts
                if active[part]
            }
            blocked_parts = [
                part
                for part in dynamic_parts
                if part in snapshots
                and contacts[part].get("count", 0) > 0
                and snapshots[part]["position_error_m"] > float(args.blocked_error_m)
            ]
            sample = {
                "frame_id": frame_id,
                "states": states,
                "active_parts": [part for part in dynamic_parts if active[part]],
                "settled_contact_latched": {
                    part: contact_latched[part]
                    for part in dynamic_parts
                    if active[part]
                },
                "blocked_parts": blocked_parts,
                "blocked": bool(blocked_parts),
                "actual": snapshots,
                "contacts": contacts,
            }
            samples.append(sample)
            if capture is not None:
                rendered_views = capture.capture()
                _compose(
                    rendered_views,
                    frame_id,
                    states,
                    snapshots,
                    contacts,
                ).save(
                    frame_dir / f"{output_index:06d}.jpg",
                    quality=94,
                )
            output_index += 1
            if output_index % 10 == 0 or frame_id == output_end:
                print(
                    f"Physics video frame {output_index}/"
                    f"{output_end - output_start + 1} "
                    f"(trajectory frame {frame_id:03d})",
                    flush=True,
                )
    finally:
        if capture is not None:
            capture.close()
        app_utils.stop()

    video_path = (
        output_root / "complete_physics_driven_trajectory.mp4"
        if capture_enabled
        else None
    )
    if video_path is not None:
        _encode_video(frame_dir, video_path, float(args.fps))
    scene_path = output_root / "complete_physics_driven_scene.usda"
    stage.GetRootLayer().Export(str(scene_path))
    first_blocked = next(
        (sample["frame_id"] for sample in samples if sample["blocked"]),
        None,
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "mode": (
            "complete force-controlled PhysX trajectory"
            if capture_enabled
            else "physics-only force-controlled PhysX trajectory"
        ),
        "output_video": str(video_path) if video_path is not None else None,
        "scene_usd": str(scene_path),
        "asset_root": str(asset_root),
        "runtime_root": str(runtime_root),
        "fps": float(args.fps),
        "physics_hz": int(round(1.0 / dt)),
        "physics_steps_per_target": physics_steps,
        "start_frame": output_start,
        "end_frame": output_end,
        "frame_count": len(samples),
        "duration_s": len(samples) / float(args.fps),
        "controller_frequency_radps": float(args.controller_frequency),
        "controller_parameters": controller_parameters,
        "settled_contact_control": settled_settings,
        "contact_offsets": contact_offsets,
        "first_blocked_frame": first_blocked,
        "samples": samples,
        "interpretation": (
            "Textured meshes are actual PhysX rigid-body poses. Cyan and "
            "magenta meshes are pose-solver targets. Once a part becomes "
            "observable, its pose is controlled only by bounded forces and "
            "torques while collision remains enabled."
        ),
    }
    write_json(output_root / "complete_physics_video_report.json", report)
    if capture_enabled and not args.keep_frames:
        shutil.rmtree(frame_dir)
    return report
