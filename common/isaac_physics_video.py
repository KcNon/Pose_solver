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
    assembly_target_translation,
    dynamic_collision_approximation,
    elastic_tube_wrench,
    physics_pose_refinement_settings,
    rigid_body_controller_parameters,
    select_control_profile,
    settled_contact_settings,
    sustained_contact_summary,
    transformed_bounds_minimum_z,
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
TARGET_COLORS = {
    "inner_pot": (0.05, 0.92, 0.96),
    "lid": (1.0, 0.12, 0.72),
}
DEFAULT_TARGET_COLORS = (
    (0.05, 0.92, 0.96),
    (1.0, 0.12, 0.72),
    (1.0, 0.72, 0.10),
    (0.42, 0.96, 0.28),
    (0.58, 0.35, 1.0),
)
UNOBSERVABLE_STATES = {
    "inferred_unobservable",
    "unobservable",
    "out_of_frame",
    "unknown",
}


def _author_fixed_assembly_lock(
    stage: Usd.Stage,
    *,
    reference_part: str,
    moving_part: str,
    moving_prim: Usd.Prim,
    achieved_world_pose: np.ndarray,
    position_error_m: float,
    rotation_error_deg: float,
    contact_observed: bool,
    contact_sustained: bool,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Persist an achieved seated pose as a fixed assembly constraint.

    The bottle collider is intentionally static, so fixing the inserted body
    to the world at its achieved pose is physically equivalent to a fixed
    bottle-to-nozzle joint while avoiding a joint target without a rigid-body
    API.  The semantic reference part and measured lock error remain explicit
    in the report.
    """

    enabled = bool(settings.get("enabled", False))
    maximum_position = float(settings.get("maximum_position_error_m", 0.02))
    maximum_rotation = float(settings.get("maximum_rotation_error_deg", 5.0))
    require_contact = bool(settings.get("require_contact", True))
    require_sustained_contact = bool(
        settings.get("require_sustained_contact", require_contact)
    )
    qualified = bool(
        enabled
        and position_error_m <= maximum_position
        and rotation_error_deg <= maximum_rotation
        and (contact_observed or not require_contact)
        and (contact_sustained or not require_sustained_contact)
    )
    result = {
        "enabled": enabled,
        "qualified": qualified,
        "reference_part": reference_part,
        "moving_part": moving_part,
        "position_error_m": float(position_error_m),
        "rotation_error_deg": float(rotation_error_deg),
        "contact_observed": bool(contact_observed),
        "contact_sustained": bool(contact_sustained),
        "require_contact": require_contact,
        "require_sustained_contact": require_sustained_contact,
        "maximum_position_error_m": maximum_position,
        "maximum_rotation_error_deg": maximum_rotation,
        "joint_path": None,
    }
    if not qualified:
        return result

    scope = UsdGeom.Scope.Define(stage, "/World/AssemblyJoints")
    path = scope.GetPath().AppendChild(
        f"{reference_part}_to_{moving_part}"
    )
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody1Rel().SetTargets([moving_prim.GetPath()])
    position = np.asarray(achieved_world_pose[:3, 3], dtype=np.float64)
    quaternion = matrix_to_quaternion_wxyz(achieved_world_pose[:3, :3])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position.tolist()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(
        float(quaternion[0]),
        Gf.Vec3f(*np.asarray(quaternion[1:], dtype=float).tolist()),
    ))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    result["joint_path"] = str(path)
    result["achieved_T_world_from_part"] = achieved_world_pose.tolist()
    result["constraint"] = "fixed_to_world_with_static_reference_part"
    return result


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
    external_force_world: np.ndarray | None = None,
    external_torque_world: np.ndarray | None = None,
    angular_natural_frequency: float | None = None,
) -> dict[str, Any]:
    actual, linear_velocity, angular_velocity = _read_state(view)
    mass = float(parameters["mass_kg"])
    inertia = float(parameters["inertia_scale"])
    rotation_frequency = float(
        natural_frequency
        if angular_natural_frequency is None
        else angular_natural_frequency
    )
    if rotation_frequency <= 0.0:
        raise ValueError("angular_natural_frequency must be positive")
    position_error = target[:3, 3] - actual[:3, 3]
    rotation_error = _rotation_error_vector(actual[:3, :3], target[:3, :3])
    feedforward_force = (
        np.zeros(3, dtype=np.float64)
        if external_force_world is None
        else np.asarray(external_force_world, dtype=np.float64)
    )
    if feedforward_force.shape != (3,) or not np.isfinite(
        feedforward_force
    ).all():
        raise ValueError("external_force_world must contain three finite values")
    feedforward_torque = (
        np.zeros(3, dtype=np.float64)
        if external_torque_world is None
        else np.asarray(external_torque_world, dtype=np.float64)
    )
    if feedforward_torque.shape != (3,) or not np.isfinite(
        feedforward_torque
    ).all():
        raise ValueError("external_torque_world must contain three finite values")
    force = (
        mass * natural_frequency**2 * position_error
        - 2.0 * damping_ratio * mass * natural_frequency * linear_velocity
        + np.array([0.0, 0.0, mass * 9.81])
        + feedforward_force
    )
    torque = (
        inertia * rotation_frequency**2 * rotation_error
        - 2.0
        * damping_ratio
        * inertia
        * rotation_frequency
        * angular_velocity
        + feedforward_torque
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
        "external_force_world_n": feedforward_force.tolist(),
        "external_torque_world_nm": feedforward_torque.tolist(),
        "force_saturated": force_saturated,
        "torque_saturated": torque_saturated,
        "controller_mode": mode,
        "controller_frequency_radps": natural_frequency,
        "controller_rotation_frequency_radps": rotation_frequency,
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
        "target_T_world_from_part": target.tolist(),
        "position_error_vector_m": (
            target[:3, 3] - actual[:3, 3]
        ).tolist(),
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
    color: tuple[float, float, float],
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
    _bind_target_material(stage, mesh.GetPrim(), part, color)
    _set_visibility(root.GetPrim(), False)
    return {"root": root.GetPrim(), "transform_op": transform_op}


def _create_environment(stage: Usd.Stage, floor_top_z: float) -> list[str]:
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
    floor.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, float(floor_top_z) - 0.01)
    )
    floor.AddScaleOp().Set(Gf.Vec3d(0.8, 0.8, 0.01))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    return [str(floor.GetPath())]


def _trajectory_floor_top(
    manifest: dict[str, Any],
    trajectory: dict[str, Any],
    reference_part: str,
    world_from_body: np.ndarray,
) -> float:
    simulation = manifest["simulation"]
    clearance = float(simulation.get("floor_clearance_m", 0.03))
    if clearance < 0.0:
        raise ValueError("simulation.floor_clearance_m must be non-negative")
    minimum_z = transformed_bounds_minimum_z(
        manifest["parts"][reference_part]["canonical_bounds_m"],
        world_from_body,
    )
    for frame_record in trajectory["frames"].values():
        for part, record in frame_record["parts"].items():
            if str(record.get("state", "unknown")) in UNOBSERVABLE_STATES:
                continue
            target = _world_from_part(
                frame_record,
                part,
                reference_part,
                world_from_body,
            )
            minimum_z = min(
                minimum_z,
                transformed_bounds_minimum_z(
                    manifest["parts"][part]["canonical_bounds_m"],
                    target,
                ),
            )
    return float(minimum_z - clearance)


def _compose(
    views: dict[str, Image.Image],
    frame_id: int,
    states: dict[str, str],
    snapshots: dict[str, dict[str, Any]],
    contacts: dict[str, dict[str, Any]],
    dynamic_parts: list[str],
    target_is_contact_corrected: bool,
    tube_constrained_assembly: bool = False,
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), (22, 22, 24))
    canvas.paste(views["perspective"], (0, 0))
    canvas.paste(views["top"], (640, 0))
    canvas.paste(views["side"], (640, 360))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, 1280, 63), fill=(0, 0, 0, 200))
    title = (
        "Isaac assembly | hand-constrained path -> passive physics"
        if tube_constrained_assembly
        else "PhysX force-driven trajectory"
    )
    draw.text(
        (18, 12),
        f"{title} | frame {frame_id:03d}",
        font=FONT_LARGE,
        fill=(255, 255, 255, 255),
    )
    target_label = (
        "contact-corrected target"
        if target_is_contact_corrected
        else "trajectory target (no correction)"
    )
    lines = [f"Textured = actual PhysX pose | transparent cyan = {target_label}"]
    if tube_constrained_assembly:
        lines.append(
            "Green curve = flexible dip tube | assembled = contact + gravity + tube only"
        )
    for part in dynamic_parts:
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
    usd_paths = _load_usd_cache(runtime_root, parts)
    simulation = manifest["simulation"]
    tube_constrained_assembly = bool(
        getattr(args, "tube_constrained_assembly", False)
    )
    physics_refinement = (
        physics_pose_refinement_settings(simulation)
        if tube_constrained_assembly
        else None
    )
    if tube_constrained_assembly and not physics_refinement["enabled"]:
        raise ValueError(
            "--tube-constrained-assembly requires "
            "simulation.physics_pose_refinement.enabled"
        )
    target_is_contact_corrected = bool(
        simulation.get("assembly_target_corrections")
    )
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
    floor_top_z = _trajectory_floor_top(
        manifest,
        trajectory,
        reference_part,
        world_from_body,
    )
    floor_colliders = _create_environment(stage, floor_top_z)

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
            part: configure_collision(
                roots[part],
                dynamic_collision_approximation(simulation),
                sdf_resolution=int(simulation.get("sdf_resolution", 192)),
            )
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

    target_colors = {
        part: TARGET_COLORS.get(
            part,
            DEFAULT_TARGET_COLORS[index % len(DEFAULT_TARGET_COLORS)],
        )
        for index, part in enumerate(dynamic_parts)
    }
    ghosts = (
        {
            part: _create_target_ghost(
                stage,
                asset_root,
                part,
                str(
                    manifest["parts"][part].get(
                        "visual_mesh",
                        manifest["parts"][part]["collision_mesh"],
                    )
                ),
                target_colors[part],
            )
            for part in dynamic_parts
        }
        if capture_enabled
        else {}
    )
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
    controller_overrides = simulation.get("controllers", {})
    controller_parameters = {
        part: rigid_body_controller_parameters(
            manifest["parts"][part],
            controller_overrides.get(part),
        )
        for part in dynamic_parts
    }
    settled_settings = settled_contact_settings(simulation)
    trajectory_preload_start = simulation.get(
        "assembly_trajectory_preload_start_frame"
    )
    if tube_constrained_assembly:
        trajectory_preload_start = None
    trajectory_preload_ramp_start = float(
        simulation.get("assembly_trajectory_preload_ramp_start_seconds", 1.5)
    )
    trajectory_preload_ramp_end = float(
        simulation.get("assembly_trajectory_preload_ramp_end_seconds", 3.0)
    )
    if (
        trajectory_preload_ramp_start < 0.0
        or trajectory_preload_ramp_end < trajectory_preload_ramp_start
    ):
        raise ValueError("invalid assembly trajectory preload ramp interval")
    trajectory_preload_initial = dict(
        simulation.get("assembly_hold_preload_forces_reference_n", {})
    )
    trajectory_preload_final = dict(
        simulation.get(
            "assembly_hold_final_preload_forces_reference_n",
            trajectory_preload_initial,
        )
    )
    trajectory_preload_vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for part in dynamic_parts:
        initial_value = np.asarray(
            trajectory_preload_initial.get(part, [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        final_value = np.asarray(
            trajectory_preload_final.get(part, initial_value),
            dtype=np.float64,
        )
        if (
            initial_value.shape != (3,)
            or final_value.shape != (3,)
            or not np.isfinite(initial_value).all()
            or not np.isfinite(final_value).all()
        ):
            raise ValueError(
                "assembly trajectory preload forces must contain three "
                "finite values"
            )
        trajectory_preload_vectors[part] = (initial_value, final_value)

    trajectory_ids = sorted(int(frame_id) for frame_id in trajectory["frames"])
    output_start = int(args.start_frame) if args.start_frame is not None else 0
    output_end = (
        int(args.end_frame)
        if args.end_frame is not None
        else trajectory_ids[-1]
    )
    if output_start < 0 or output_end < output_start:
        raise ValueError(f"Invalid output range: {output_start}..{output_end}")
    tube_start_frame = (
        int(physics_refinement["apply_frame_range"][0])
        if physics_refinement is not None
        else None
    )
    passive_hold_start_frame = (
        int(physics_refinement["apply_frame_range"][1])
        if physics_refinement is not None
        else None
    )
    body_axis_world = world_from_body[:3, :3] @ np.asarray(
        [0.0, 1.0, 0.0], dtype=np.float64
    )
    body_axis_origin_world = None
    tube_visual = None
    tube_visual_report: dict[str, Any] | None = None
    if physics_refinement is not None:
        tube = physics_refinement["tube"]
        body_axis_origin_world = (
            world_from_body
            @ np.append(
                np.asarray(tube["body_axis_origin_body_m"], dtype=np.float64),
                1.0,
            )
        )[:3]
        initial_frame_id = next(
            frame_id
            for frame_id in trajectory_ids
            if frame_id >= output_start
        )
        initial_record = trajectory["frames"][f"{initial_frame_id:06d}"]
        initial_part_pose = _world_from_part(
            initial_record,
            str(simulation["inserted_part"]),
            reference_part,
            world_from_body,
        )
        if capture_enabled:
            UsdGeom.Scope.Define(stage, "/World/PhysicsPoseRefinement")
            UsdGeom.Scope.Define(stage, "/World/DemoVisuals")
            UsdGeom.Scope.Define(stage, "/World/DemoMaterials")
            initial_points = runtime.tube_curve_points(
                position_world=initial_part_pose[:3, 3],
                rotation_world_from_part=initial_part_pose[:3, :3],
                body_axis_origin_world=body_axis_origin_world,
                body_axis_world=body_axis_world,
                tube=tube,
            )
            tube_visual = runtime.author_tube_visual(stage, initial_points, tube)
            hidden_body_visuals = runtime.hide_imported_visuals(
                roots[reference_part]
            )
            transparent_body = runtime.author_direct_visual(
                stage,
                "CompleteAssemblyBodyShell",
                asset_root
                / str(
                    manifest["parts"][reference_part].get(
                        "visual_mesh",
                        manifest["parts"][reference_part]["collision_mesh"],
                    )
                ),
                world_from_body,
                (0.48, 0.10, 0.10),
                metallic=0.02,
                roughness=0.45,
                opacity=0.30,
            )
            tube_visual_report = {
                key: value
                for key, value in tube_visual.items()
                if key != "points_attr"
            }
            tube_visual_report["hidden_body_visuals"] = hidden_body_visuals
            tube_visual_report["transparent_body_shell"] = transparent_body
    physics_steps = max(1, int(round(1.0 / (dt * float(args.fps)))))
    active = {part: False for part in dynamic_parts}
    contact_latched = {part: False for part in dynamic_parts}
    passive_terminal_latched = {part: False for part in dynamic_parts}
    passive_terminal_ready_steps = {part: 0 for part in dynamic_parts}
    passive_terminal_release_frame: dict[str, int] = {}
    passive_terminal_dwell_steps = max(1, int(round(0.25 / dt)))
    last_targets: dict[str, np.ndarray] = {}
    last_target_corrections: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    assembly_locks: dict[str, dict[str, Any]] = {}

    app_utils.play(commit=True)
    app.update()
    if capture_enabled:
        from common.isaac_video import MultiViewCapture

        capture = MultiViewCapture(app, int(args.rt_subframes))
    else:
        capture = None
    output_index = 0
    assembly_hold_report: dict[str, Any] = {
        "enabled": False,
        "seconds": 0.0,
        "samples": 0,
        "contact_observed": {},
    }
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
                    correction = assembly_target_translation(
                        simulation,
                        part=part,
                        frame_id=frame_id,
                        reference_rotation=world_from_body[:3, :3],
                    )
                    target[:3, 3] += correction["translation_world_m"]
                    last_targets[part] = target
                    last_target_corrections[part] = {
                        "enabled": bool(correction["enabled"]),
                        "fraction": float(correction["fraction"]),
                        "translation_reference_m": np.asarray(
                            correction["translation_reference_m"], dtype=float
                        ).tolist(),
                        "translation_world_m": np.asarray(
                            correction["translation_world_m"], dtype=float
                        ).tolist(),
                        "source": correction["source"],
                    }
                    observable = states[part] not in UNOBSERVABLE_STATES
                    if part in ghosts:
                        ghosts[part]["transform_op"].Set(
                            np_to_gf_matrix(target)
                        )
                        _set_visibility(ghosts[part]["root"], observable)
                    if observable and not active[part]:
                        set_pose(views[part], target, zero_velocity=False)
                        activate_as_kinematic = bool(
                            tube_constrained_assembly
                            and frame_id < int(passive_hold_start_frame)
                        )
                        set_kinematic(
                            rigid_prims[part],
                            activate_as_kinematic,
                        )
                        if not activate_as_kinematic:
                            views[part].set_velocities(
                                linear_velocities=[[0.0, 0.0, 0.0]],
                                angular_velocities=[[0.0, 0.0, 0.0]],
                            )
                        _set_collisions(stage, collider_paths[part], True)
                        PhysxSchema.PhysxRigidBodyAPI.Apply(
                            rigid_prims[part]
                        ).CreateEnableCCDAttr().Set(not activate_as_kinematic)
                        _set_visibility(roots[part], True)
                        active[part] = True

                if tube_constrained_assembly:
                    for part in dynamic_parts:
                        if not active[part]:
                            continue
                        if frame_id < int(passive_hold_start_frame):
                            set_kinematic(rigid_prims[part], True)
                            set_pose(
                                views[part],
                                last_targets[part],
                                zero_velocity=False,
                            )
                        elif not passive_terminal_latched[part]:
                            # The preceding phases are explicitly hand-constrained.
                            # Release exactly from the separately validated physical
                            # projection, rather than carrying controller collision
                            # error into the passive terminal validation.
                            set_pose(
                                views[part],
                                last_targets[part],
                                zero_velocity=False,
                            )
                            set_kinematic(rigid_prims[part], False)
                            PhysxSchema.PhysxRigidBodyAPI.Apply(
                                rigid_prims[part]
                            ).CreateEnableCCDAttr().Set(True)
                            views[part].set_velocities(
                                linear_velocities=[[0.0, 0.0, 0.0]],
                                angular_velocities=[[0.0, 0.0, 0.0]],
                            )
                            passive_terminal_latched[part] = True
                            passive_terminal_release_frame[part] = frame_id

            latest_controls = {
                part: {
                    "force_world_n": [0.0, 0.0, 0.0],
                    "torque_world_nm": [0.0, 0.0, 0.0],
                    "force_saturated": False,
                    "torque_saturated": False,
                    "controller_mode": "inactive",
                    "controller_frequency_radps": 0.0,
                    "controller_damping_ratio": 0.0,
                    "external_force_world_n": [0.0, 0.0, 0.0],
                    "external_torque_world_nm": [0.0, 0.0, 0.0],
                }
                for part in dynamic_parts
            }
            if frame_record is not None:
                for physics_step_index in range(physics_steps):
                    for part in dynamic_parts:
                        if active[part]:
                            if (
                                tube_constrained_assembly
                                and frame_id < int(passive_hold_start_frame)
                            ):
                                latest_controls[part] = {
                                    "force_world_n": [0.0, 0.0, 0.0],
                                    "torque_world_nm": [0.0, 0.0, 0.0],
                                    "external_force_world_n": [0.0, 0.0, 0.0],
                                    "external_torque_world_nm": [0.0, 0.0, 0.0],
                                    "force_saturated": False,
                                    "torque_saturated": False,
                                    "controller_mode": "external_kinematic_hand_constraint",
                                    "controller_frequency_radps": 0.0,
                                    "controller_rotation_frequency_radps": 0.0,
                                    "controller_damping_ratio": 0.0,
                                }
                                continue
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
                            trajectory_external_force = None
                            trajectory_external_torque = None
                            trajectory_angular_frequency = None
                            if (
                                trajectory_preload_start is not None
                                and frame_id >= int(trajectory_preload_start)
                                and states[part] in settled_settings["states"]
                            ):
                                elapsed = (
                                    frame_id - int(trajectory_preload_start)
                                    + (physics_step_index + 1) / physics_steps
                                ) / float(args.fps)
                                if elapsed <= trajectory_preload_ramp_start:
                                    preload_fraction = 0.0
                                elif trajectory_preload_ramp_end <= (
                                    trajectory_preload_ramp_start
                                ):
                                    preload_fraction = 1.0
                                else:
                                    preload_fraction = float(np.clip(
                                        (elapsed - trajectory_preload_ramp_start)
                                        / (
                                            trajectory_preload_ramp_end
                                            - trajectory_preload_ramp_start
                                        ),
                                        0.0,
                                        1.0,
                                    ))
                                initial_value, final_value = (
                                    trajectory_preload_vectors[part]
                                )
                                preload_reference_value = (
                                    (1.0 - preload_fraction) * initial_value
                                    + preload_fraction * final_value
                                )
                                trajectory_external_force = (
                                    world_from_body[:3, :3]
                                    @ preload_reference_value
                                )
                                trajectory_angular_frequency = float(
                                    simulation.get(
                                        "assembly_hold_rotation_frequency_radps",
                                        settled_settings["frequency_radps"],
                                    )
                                )
                            tube_wrench = None
                            if (
                                physics_refinement is not None
                                and part == str(simulation["inserted_part"])
                                and frame_id >= int(tube_start_frame)
                            ):
                                tube_wrench = elastic_tube_wrench(
                                    position_world=current_pose[:3, 3],
                                    rotation_world_from_part=current_pose[:3, :3],
                                    linear_velocity_world=_linear,
                                    angular_velocity_world=_angular,
                                    body_axis_origin_world=body_axis_origin_world,
                                    body_axis_world=body_axis_world,
                                    tube=physics_refinement["tube"],
                                )
                                trajectory_external_force = np.asarray(
                                    tube_wrench["force_world_n"], dtype=np.float64
                                )
                                trajectory_external_torque = np.asarray(
                                    tube_wrench["torque_world_nm"], dtype=np.float64
                                )
                            passive_tube_hold = bool(
                                physics_refinement is not None
                                and part == str(simulation["inserted_part"])
                                and frame_id >= int(passive_hold_start_frame)
                                and passive_terminal_latched[part]
                            )
                            if (
                                physics_refinement is not None
                                and part == str(simulation["inserted_part"])
                                and frame_id >= int(passive_hold_start_frame)
                                and not passive_terminal_latched[part]
                            ):
                                rotation_error_deg = math.degrees(float(np.linalg.norm(
                                    _rotation_error_vector(
                                        current_pose[:3, :3],
                                        last_targets[part][:3, :3],
                                    )
                                )))
                                contact_ready = bool(
                                    _contact_snapshot(views[part], dt).get("count", 0) > 0
                                )
                                terminal_ready = bool(
                                    current_position_error <= 0.002
                                    and rotation_error_deg <= 2.0
                                    and float(np.linalg.norm(_linear)) <= 0.01
                                    and float(np.linalg.norm(_angular)) <= 0.1
                                    and contact_ready
                                )
                                passive_terminal_ready_steps[part] = (
                                    passive_terminal_ready_steps[part] + 1
                                    if terminal_ready
                                    else 0
                                )
                                if (
                                    passive_terminal_ready_steps[part]
                                    >= passive_terminal_dwell_steps
                                ):
                                    passive_terminal_latched[part] = True
                                    passive_terminal_release_frame[part] = frame_id
                                    passive_tube_hold = True
                            if passive_tube_hold:
                                views[part].apply_forces_and_torques_at_pos(
                                    forces=[trajectory_external_force.tolist()],
                                    torques=[trajectory_external_torque.tolist()],
                                    local_frame=False,
                                )
                                latest_controls[part] = {
                                    "force_world_n": trajectory_external_force.tolist(),
                                    "torque_world_nm": trajectory_external_torque.tolist(),
                                    "external_force_world_n": trajectory_external_force.tolist(),
                                    "external_torque_world_nm": trajectory_external_torque.tolist(),
                                    "force_saturated": bool(tube_wrench["force_saturated"]),
                                    "torque_saturated": bool(tube_wrench["torque_saturated"]),
                                    "controller_mode": "passive_tube_terminal_hold",
                                    "controller_frequency_radps": 0.0,
                                    "controller_rotation_frequency_radps": 0.0,
                                    "controller_damping_ratio": 0.0,
                                    "tube_radial_deflection_m": float(tube_wrench["radial_deflection_m"]),
                                    "tube_bend_angle_deg": float(tube_wrench["bend_angle_deg"]),
                                    "tube_elastic_energy_j": float(tube_wrench["elastic_energy_j"]),
                                }
                            else:
                                control_frequency = float(profile["frequency_radps"])
                                control_damping = float(profile["damping_ratio"])
                                control_rotation_frequency = trajectory_angular_frequency
                                if (
                                    physics_refinement is not None
                                    and part == str(simulation["inserted_part"])
                                    and frame_id >= int(passive_hold_start_frame)
                                ):
                                    control_frequency = max(
                                        float(args.controller_frequency), 24.0
                                    )
                                    control_damping = max(control_damping, 1.25)
                                    control_rotation_frequency = control_frequency
                                latest_controls[part] = _control(
                                    views[part],
                                    last_targets[part],
                                    controller_parameters[part],
                                    control_frequency,
                                    control_damping,
                                    (
                                        "external_constraint_plus_tube"
                                        if tube_wrench is not None
                                        else str(profile["mode"])
                                    ),
                                    external_force_world=trajectory_external_force,
                                    external_torque_world=trajectory_external_torque,
                                    angular_natural_frequency=(
                                        control_rotation_frequency
                                    ),
                                )
                                if tube_wrench is not None:
                                    latest_controls[part].update({
                                        "tube_radial_deflection_m": float(tube_wrench["radial_deflection_m"]),
                                        "tube_bend_angle_deg": float(tube_wrench["bend_angle_deg"]),
                                        "tube_elastic_energy_j": float(tube_wrench["elastic_energy_j"]),
                                    })
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
            if tube_visual is not None and str(simulation["inserted_part"]) in snapshots:
                actual_pose = np.asarray(
                    snapshots[str(simulation["inserted_part"])]["T_world_from_part"],
                    dtype=np.float64,
                )
                runtime.update_tube_visual(
                    tube_visual,
                    runtime.tube_curve_points(
                        position_world=actual_pose[:3, 3],
                        rotation_world_from_part=actual_pose[:3, :3],
                        body_axis_origin_world=body_axis_origin_world,
                        body_axis_world=body_axis_world,
                        tube=physics_refinement["tube"],
                    ),
                )
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
                "passive_terminal_latched": {
                    part: passive_terminal_latched[part]
                    for part in dynamic_parts
                    if active[part]
                },
                "blocked_parts": blocked_parts,
                "blocked": bool(blocked_parts),
                "actual": snapshots,
                "contacts": contacts,
                "target_corrections": {
                    part: last_target_corrections[part]
                    for part in dynamic_parts
                    if part in last_target_corrections
                },
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
                    dynamic_parts,
                    target_is_contact_corrected,
                    tube_constrained_assembly,
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
        hold_seconds = float(simulation.get("assembly_hold_seconds", 0.0))
        hold_contact_observed = {part: False for part in dynamic_parts}
        hold_contact_history = {part: [] for part in dynamic_parts}
        preload_reference_initial = (
            {}
            if tube_constrained_assembly
            else dict(simulation.get("assembly_hold_preload_forces_reference_n", {}))
        )
        preload_reference_final = (
            {}
            if tube_constrained_assembly
            else dict(
                simulation.get(
                    "assembly_hold_final_preload_forces_reference_n",
                    preload_reference_initial,
                )
            )
        )
        preload_initial: dict[str, np.ndarray] = {}
        preload_final: dict[str, np.ndarray] = {}
        for part in dynamic_parts:
            initial_value = np.asarray(
                preload_reference_initial.get(part, [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            final_value = np.asarray(
                preload_reference_final.get(part, initial_value),
                dtype=np.float64,
            )
            if (
                initial_value.shape != (3,)
                or final_value.shape != (3,)
                or not np.isfinite(initial_value).all()
                or not np.isfinite(final_value).all()
            ):
                raise ValueError(
                    "assembly hold preload force values must "
                    "contain three finite values"
                )
            preload_initial[part] = initial_value
            preload_final[part] = final_value
        preload_ramp_start = float(
            simulation.get("assembly_hold_preload_ramp_start_seconds", 0.0)
        )
        if preload_ramp_start < 0.0 or preload_ramp_start > hold_seconds:
            raise ValueError(
                "assembly_hold_preload_ramp_start_seconds must be within hold"
            )
        current_preload_world = {
            part: world_from_body[:3, :3] @ preload_initial[part]
            for part in dynamic_parts
        }
        if hold_seconds > 0.0 and last_targets:
            hold_steps = max(1, int(math.ceil(hold_seconds / dt)))
            capture_interval = max(
                1, int(round(1.0 / (dt * float(args.fps))))
            )
            hold_frames = 0
            for hold_step in range(hold_steps):
                elapsed = float(hold_step + 1) * dt
                if hold_seconds <= preload_ramp_start:
                    preload_fraction = 0.0
                else:
                    preload_fraction = float(np.clip(
                        (elapsed - preload_ramp_start)
                        / (hold_seconds - preload_ramp_start),
                        0.0,
                        1.0,
                    ))
                current_preload_world = {
                    part: world_from_body[:3, :3]
                    @ (
                        (1.0 - preload_fraction) * preload_initial[part]
                        + preload_fraction * preload_final[part]
                    )
                    for part in dynamic_parts
                }
                latest_controls = {}
                for part in dynamic_parts:
                    if not active[part]:
                        continue
                    if (
                        physics_refinement is not None
                        and part == str(simulation["inserted_part"])
                    ):
                        current_pose, linear, angular = _read_state(views[part])
                        wrench = elastic_tube_wrench(
                            position_world=current_pose[:3, 3],
                            rotation_world_from_part=current_pose[:3, :3],
                            linear_velocity_world=linear,
                            angular_velocity_world=angular,
                            body_axis_origin_world=body_axis_origin_world,
                            body_axis_world=body_axis_world,
                            tube=physics_refinement["tube"],
                        )
                        if not passive_terminal_latched[part]:
                            position_error = float(np.linalg.norm(
                                last_targets[part][:3, 3] - current_pose[:3, 3]
                            ))
                            rotation_error_deg = math.degrees(float(np.linalg.norm(
                                _rotation_error_vector(
                                    current_pose[:3, :3],
                                    last_targets[part][:3, :3],
                                )
                            )))
                            contact_ready = bool(
                                _contact_snapshot(views[part], dt).get("count", 0) > 0
                            )
                            terminal_ready = bool(
                                position_error <= 0.002
                                and rotation_error_deg <= 2.0
                                and float(np.linalg.norm(linear)) <= 0.01
                                and float(np.linalg.norm(angular)) <= 0.1
                                and contact_ready
                            )
                            passive_terminal_ready_steps[part] = (
                                passive_terminal_ready_steps[part] + 1
                                if terminal_ready
                                else 0
                            )
                            if (
                                passive_terminal_ready_steps[part]
                                >= passive_terminal_dwell_steps
                            ):
                                passive_terminal_latched[part] = True
                                passive_terminal_release_frame[part] = (
                                    output_end
                                    + max(1, int(round(elapsed * float(args.fps))))
                                )
                        if passive_terminal_latched[part]:
                            views[part].apply_forces_and_torques_at_pos(
                                forces=[wrench["force_world_n"].tolist()],
                                torques=[wrench["torque_world_nm"].tolist()],
                                local_frame=False,
                            )
                            latest_controls[part] = {
                                "force_world_n": wrench["force_world_n"].tolist(),
                                "torque_world_nm": wrench["torque_world_nm"].tolist(),
                                "external_force_world_n": wrench["force_world_n"].tolist(),
                                "external_torque_world_nm": wrench["torque_world_nm"].tolist(),
                                "force_saturated": bool(wrench["force_saturated"]),
                                "torque_saturated": bool(wrench["torque_saturated"]),
                                "controller_mode": "passive_tube_terminal_hold",
                                "controller_frequency_radps": 0.0,
                                "controller_rotation_frequency_radps": 0.0,
                                "controller_damping_ratio": 0.0,
                                "tube_radial_deflection_m": float(wrench["radial_deflection_m"]),
                                "tube_bend_angle_deg": float(wrench["bend_angle_deg"]),
                                "tube_elastic_energy_j": float(wrench["elastic_energy_j"]),
                            }
                        else:
                            latest_controls[part] = _control(
                                views[part],
                                last_targets[part],
                                controller_parameters[part],
                                max(float(args.controller_frequency), 24.0),
                                1.25,
                                "terminal_pose_acquisition_plus_tube",
                                external_force_world=np.asarray(
                                    wrench["force_world_n"], dtype=np.float64
                                ),
                                external_torque_world=np.asarray(
                                    wrench["torque_world_nm"], dtype=np.float64
                                ),
                                angular_natural_frequency=max(
                                    float(args.controller_frequency), 24.0
                                ),
                            )
                            latest_controls[part].update({
                                "tube_radial_deflection_m": float(wrench["radial_deflection_m"]),
                                "tube_bend_angle_deg": float(wrench["bend_angle_deg"]),
                                "tube_elastic_energy_j": float(wrench["elastic_energy_j"]),
                            })
                    else:
                        latest_controls[part] = _control(
                            views[part],
                            last_targets[part],
                            controller_parameters[part],
                            float(settled_settings["frequency_radps"]),
                            float(settled_settings["damping_ratio"]),
                            "assembly_hold",
                            external_force_world=current_preload_world[part],
                            angular_natural_frequency=float(
                                simulation.get(
                                    "assembly_hold_rotation_frequency_radps",
                                    settled_settings["frequency_radps"],
                                )
                            ),
                        )
                SimulationManager.step(steps=1)
                contacts_now = {
                    part: _contact_snapshot(views[part], dt)
                    for part in dynamic_parts
                    if active[part]
                }
                for part, contact in contacts_now.items():
                    has_contact = bool(contact.get("count", 0) > 0)
                    hold_contact_observed[part] |= has_contact
                    hold_contact_history[part].append(has_contact)
                if (
                    hold_step % capture_interval != capture_interval - 1
                    and hold_step != hold_steps - 1
                ):
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
                if tube_visual is not None and str(simulation["inserted_part"]) in snapshots:
                    actual_pose = np.asarray(
                        snapshots[str(simulation["inserted_part"])]["T_world_from_part"],
                        dtype=np.float64,
                    )
                    runtime.update_tube_visual(
                        tube_visual,
                        runtime.tube_curve_points(
                            position_world=actual_pose[:3, 3],
                            rotation_world_from_part=actual_pose[:3, :3],
                            body_axis_origin_world=body_axis_origin_world,
                            body_axis_world=body_axis_world,
                            tube=physics_refinement["tube"],
                        ),
                    )
                hold_frame_id = output_end + hold_frames + 1
                sample = {
                    "frame_id": hold_frame_id,
                    "states": {
                        part: "assembly_hold" for part in dynamic_parts
                    },
                    "active_parts": [
                        part for part in dynamic_parts if active[part]
                    ],
                    "settled_contact_latched": {
                        part: hold_contact_observed[part]
                        for part in dynamic_parts
                        if active[part]
                    },
                    "blocked_parts": [],
                    "blocked": False,
                    "actual": snapshots,
                    "contacts": contacts_now,
                    "target_corrections": {
                        part: last_target_corrections[part]
                        for part in dynamic_parts
                        if part in last_target_corrections
                    },
                }
                samples.append(sample)
                if capture is not None:
                    rendered_views = capture.capture()
                    _compose(
                        rendered_views,
                        hold_frame_id,
                        sample["states"],
                        snapshots,
                        contacts_now,
                        dynamic_parts,
                        target_is_contact_corrected,
                        tube_constrained_assembly,
                    ).save(
                        frame_dir / f"{output_index:06d}.jpg",
                        quality=94,
                    )
                output_index += 1
                hold_frames += 1
            contact_gate = {
                part: sustained_contact_summary(
                    hold_contact_history[part],
                    physics_dt=dt,
                    settings=dict(simulation.get("assembly_lock", {})),
                )
                for part in dynamic_parts
            }
            assembly_hold_report = {
                "enabled": True,
                "seconds": hold_seconds,
                "physics_steps": hold_steps,
                "samples": hold_frames,
                "contact_observed": hold_contact_observed,
                "contact_gate": contact_gate,
                "initial_preload_forces_reference_n": {
                    part: preload_initial[part].tolist()
                    for part in dynamic_parts
                },
                "final_preload_forces_reference_n": {
                    part: preload_final[part].tolist()
                    for part in dynamic_parts
                },
                "final_preload_forces_world_n": {
                    part: current_preload_world[part].tolist()
                    for part in dynamic_parts
                },
                "preload_ramp_start_seconds": preload_ramp_start,
                "rotation_frequency_radps": float(
                    simulation.get(
                        "assembly_hold_rotation_frequency_radps",
                        settled_settings["frequency_radps"],
                    )
                ),
            }

        lock_settings = dict(simulation.get("assembly_lock", {}))
        if tube_constrained_assembly:
            lock_settings["enabled"] = False
        if samples:
            final_actual = samples[-1]["actual"]
            for part in dynamic_parts:
                if part not in final_actual:
                    continue
                achieved = np.asarray(
                    final_actual[part]["T_world_from_part"],
                    dtype=np.float64,
                )
                assembly_locks[part] = _author_fixed_assembly_lock(
                    stage,
                    reference_part=reference_part,
                    moving_part=part,
                    moving_prim=rigid_prims[part],
                    achieved_world_pose=achieved,
                    position_error_m=float(
                        final_actual[part]["position_error_m"]
                    ),
                    rotation_error_deg=float(
                        final_actual[part]["rotation_error_deg"]
                    ),
                    contact_observed=bool(
                        hold_contact_observed.get(part, False)
                    ),
                    contact_sustained=bool(
                        assembly_hold_report.get("contact_gate", {})
                        .get(part, {})
                        .get("sustained", False)
                    ),
                    settings=lock_settings,
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
    blocked_ids = [
        int(sample["frame_id"]) for sample in samples if sample["blocked"]
    ]
    blocked_ranges: list[list[int]] = []
    for frame_id in blocked_ids:
        if not blocked_ranges or frame_id > blocked_ranges[-1][1] + 1:
            blocked_ranges.append([frame_id, frame_id])
        else:
            blocked_ranges[-1][1] = frame_id
    active_snapshots = [
        snapshot
        for sample in samples
        for snapshot in sample["actual"].values()
    ]
    maximum_position_error = max(
        (float(item["position_error_m"]) for item in active_snapshots),
        default=0.0,
    )
    maximum_rotation_error = max(
        (float(item["rotation_error_deg"]) for item in active_snapshots),
        default=0.0,
    )
    contact_actor_paths = sorted(
        {
            str(path)
            for sample in samples
            for contact in sample["contacts"].values()
            for path in contact.get("other_actor_paths", [])
        }
    )
    lock_required = bool(
        simulation.get("assembly_lock", {}).get("enabled", False)
        and not tube_constrained_assembly
    )
    assembly_lock_passed = bool(
        not lock_required
        or (
            assembly_locks
            and all(
                bool(result.get("qualified", False))
                for result in assembly_locks.values()
            )
        )
    )
    passive_release_passed = bool(
        not tube_constrained_assembly
        or all(passive_terminal_latched.values())
    )
    physics_validation = {
        "passed": (
            not blocked_ids
            and assembly_lock_passed
            and passive_release_passed
        ),
        "blocked_frame_count": len(blocked_ids),
        "blocked_frame_ranges": blocked_ranges,
        "maximum_position_error_m": maximum_position_error,
        "maximum_rotation_error_deg": maximum_rotation_error,
        "contact_actor_paths": contact_actor_paths,
        "blocked_error_threshold_m": float(args.blocked_error_m),
        "assembly_lock_required": lock_required,
        "assembly_lock_passed": assembly_lock_passed,
        "passive_terminal_release_passed": passive_release_passed,
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "mode": (
            "complete hand-constrained trajectory with passive physical terminal hold"
            if tube_constrained_assembly and capture_enabled
            else "physics-only hand-constrained trajectory with passive physical terminal hold"
            if tube_constrained_assembly
            else "complete force-controlled PhysX trajectory"
            if capture_enabled
            else "physics-only force-controlled PhysX trajectory"
        ),
        "output_video": str(video_path) if video_path is not None else None,
        "scene_usd": str(scene_path),
        "asset_root": str(asset_root),
        "runtime_root": str(runtime_root),
        "trajectory": str(Path(args.trajectory).resolve()),
        "fps": float(args.fps),
        "physics_hz": int(round(1.0 / dt)),
        "physics_steps_per_target": physics_steps,
        "start_frame": output_start,
        "end_frame": output_end,
        "frame_count": len(samples),
        "duration_s": len(samples) / float(args.fps),
        "controller_frequency_radps": float(args.controller_frequency),
        "controller_parameters": controller_parameters,
        "trajectory_assembly_control": {
            "enabled": trajectory_preload_start is not None,
            "start_frame": trajectory_preload_start,
            "preload_ramp_start_seconds": trajectory_preload_ramp_start,
            "preload_ramp_end_seconds": trajectory_preload_ramp_end,
        },
        "settled_contact_control": settled_settings,
        "contact_offsets": contact_offsets,
        "floor_top_z": floor_top_z,
        "first_blocked_frame": first_blocked,
        "physics_validation": physics_validation,
        "assembly_locks": assembly_locks,
        "assembly_hold": assembly_hold_report,
        "target_corrections_enabled": target_is_contact_corrected,
        "tube_constrained_assembly": {
            "enabled": tube_constrained_assembly,
            "tube_force_start_frame": tube_start_frame,
            "passive_terminal_hold_start_frame": passive_hold_start_frame,
            "external_kinematic_constraint_before_terminal_hold": tube_constrained_assembly,
            "pose_controller_enabled_before_terminal_hold": False if tube_constrained_assembly else True,
            "pose_controller_enabled_during_terminal_hold": False if tube_constrained_assembly else None,
            "passive_release_requires_state_gate": False if tube_constrained_assembly else None,
            "release_pose_source": "trajectory_physics_refined" if tube_constrained_assembly else None,
            "passive_release_dwell_seconds": 0.0 if tube_constrained_assembly else None,
            "passive_release_frame": passive_terminal_release_frame,
            "axial_preload_enabled": False if tube_constrained_assembly else trajectory_preload_start is not None,
            "fixed_joint_enabled": False if tube_constrained_assembly else lock_required,
            "tube_visual": tube_visual_report,
        },
        "samples": samples,
        "interpretation": (
            "Textured meshes are actual PhysX rigid-body poses. Transparent "
            "cyan meshes use the same visual geometry and show "
            + (
                "explicit contact-corrected targets. "
                if target_is_contact_corrected
                else "the unmodified trajectory targets. "
            )
            + (
                "Before terminal hold, an external kinematic constraint represents "
                "the unmodelled hand and follows the observed trajectory exactly. "
                "At the assembled phase the body is released from the independently "
                "validated physical projection; terminal hold uses only contact, "
                "gravity, and tube elasticity, with no pose controller, FixedJoint, "
                "or axial preload."
                if tube_constrained_assembly
                else "Once a part becomes observable, its pose is controlled only by "
                "bounded forces and torques while collision remains enabled."
            )
        ),
    }
    write_json(output_root / "complete_physics_video_report.json", report)
    if capture_enabled and not args.keep_frames:
        shutil.rmtree(frame_dir)
    return report
