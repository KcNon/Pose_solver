"""Isaac PhysX contact probes for frozen-pose scale candidates.

Each candidate is an isolated, zero-gravity simulation clone initialized from
the frozen observed pose.  PhysX advances only long enough to build the first
collision manifold.  The clone may depenetrate afterwards, but no result pose
is read or written back to the observed trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.io_utils import write_json

# Imported only after SimulationApp has been created by the CLI wrapper.
import carb
import omni.usd
from omni.physics.core import (
    ContactEventType,
    get_physics_simulation_interface,
)
from pxr import (
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdUtils,
)

import common.isaac_runtime as runtime


def _header_value(header: Any, *names: str, default: int = 0) -> int:
    for name in names:
        if hasattr(header, name):
            return int(getattr(header, name))
    return int(default)


def _data_value(data: Any, *names: str, default: float = 0.0) -> float:
    for name in names:
        if hasattr(data, name):
            return float(getattr(data, name))
    return float(default)


def _candidate_from_path(path: str) -> str | None:
    prefix = "/World/Queries/"
    if prefix not in path:
        return None
    tail = path.split(prefix, 1)[1]
    return tail.split("/", 1)[0] if tail else None


def _similarity_pose(pose: np.ndarray, factor: float) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3, :3] *= float(factor)
    return result


def _build_query_pair(
    stage: Usd.Stage,
    *,
    key: str,
    offset: np.ndarray,
    reference_asset: str,
    moving_asset: str,
    relative_pose: np.ndarray,
    moving_scale_factor: float,
    reference_part: str,
    moving_part: str,
    app: Any,
) -> dict[str, Any]:
    root_path = f"/World/Queries/{key}"
    UsdGeom.Xform.Define(stage, root_path)
    reference_world = np.eye(4, dtype=np.float64)
    reference_world[:3, 3] = np.asarray(offset, dtype=np.float64)
    offset_transform = np.eye(4, dtype=np.float64)
    offset_transform[:3, 3] = np.asarray(offset, dtype=np.float64)
    moving_world = offset_transform @ np.asarray(
        relative_pose, dtype=np.float64
    )
    reference_root = runtime.add_reference(
        stage,
        f"{root_path}/Reference",
        reference_asset,
        reference_world,
    )
    moving_root = runtime.add_reference(
        stage,
        f"{root_path}/Moving",
        moving_asset,
        _similarity_pose(moving_world, moving_scale_factor),
    )
    for _ in range(3):
        app.update()
    runtime.deinstance_composed_tree(reference_root)
    runtime.deinstance_composed_tree(moving_root)
    for _ in range(2):
        app.update()
    reference_rigid = runtime.choose_rigid_prim(
        reference_root, reference_part
    )
    moving_rigid = runtime.choose_rigid_prim(moving_root, moving_part)
    reference_colliders = runtime.configure_collision(reference_root, "none")
    moving_colliders = runtime.configure_collision(
        moving_root, "convexDecomposition"
    )
    runtime.remove_articulation_apis(reference_root)
    runtime.remove_articulation_apis(moving_root)
    runtime.make_body_static(reference_rigid)
    UsdPhysics.RigidBodyAPI.Apply(
        moving_rigid
    ).CreateKinematicEnabledAttr().Set(False)
    moving_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(moving_rigid)
    moving_physx.CreateDisableGravityAttr().Set(True)
    moving_physx.CreateEnableCCDAttr().Set(False)
    contact_api = PhysxSchema.PhysxContactReportAPI.Apply(moving_rigid)
    contact_api.CreateThresholdAttr().Set(0.0)
    return {
        "key": key,
        "moving_rigid": str(moving_rigid.GetPath()),
        "reference_colliders": reference_colliders,
        "moving_colliders": moving_colliders,
        "moving_scale_factor": float(moving_scale_factor),
        "authored_T_reference_from_moving": np.asarray(
            relative_pose, dtype=np.float64
        ).tolist(),
    }


def run_frozen_scale_queries(
    args: argparse.Namespace,
    app: Any,
    asset_root: Path,
    runtime_root: Path,
    manifest: dict[str, Any],
    scale_report: dict[str, Any],
) -> dict[str, Any]:
    """Run contact-manifold construction without dynamic settling."""
    runtime.ARGS = args
    runtime.ASSET_ROOT = asset_root
    runtime.RUNTIME_ROOT = runtime_root
    runtime.MANIFEST = manifest
    runtime.simulation_app = app
    usd_paths = runtime.import_urdf_assets()

    trajectory_path = Path(scale_report["inputs"]["trajectory"])
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    runtime.stage_utils.create_new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Queries")
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityMagnitudeAttr().Set(0.0)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(
        physics_scene.GetPrim()
    )
    physx_scene.CreateTimeStepsPerSecondAttr().Set(60)
    physx_scene.CreateSolverTypeAttr().Set("TGS")

    candidates: dict[str, dict[str, Any]] = {}
    candidate_to_relation: dict[str, str] = {}
    item_index = 0
    for relation_name, relation in scale_report["relations"].items():
        reference_part = str(relation["reference_part"])
        moving_part = str(relation["moving_part"])
        query_frame = int(
            relation.get("isaac_query_frame", relation["geometry_frames"][-1])
        )
        records = trajectory["frames"][f"{query_frame:06d}"]["parts"]
        reference_pose = np.asarray(
            records[reference_part]["T_world_from_part"], dtype=np.float64
        )
        moving_pose = np.asarray(
            records[moving_part]["T_world_from_part"], dtype=np.float64
        )
        relative = np.linalg.inv(reference_pose) @ moving_pose
        for candidate_index, row in enumerate(relation["candidates"]):
            key = f"Q{item_index:03d}"
            offset = np.array(
                [
                    1.0 * (item_index % 8),
                    1.0 * (item_index // 8),
                    0.0,
                ],
                dtype=np.float64,
            )
            item = _build_query_pair(
                stage,
                key=key,
                offset=offset,
                reference_asset=usd_paths[reference_part],
                moving_asset=usd_paths[moving_part],
                relative_pose=relative,
                moving_scale_factor=float(row["scale_factor"]),
                reference_part=reference_part,
                moving_part=moving_part,
                app=app,
            )
            item.update(
                {
                    "relation": relation_name,
                    "candidate_index": candidate_index,
                    "query_frame": query_frame,
                    "analytic_max_penetration_m": float(
                        row["max_penetration_m"]
                    ),
                    "analytic_contact_gap_violation_m": float(
                        row.get("max_contact_gap_violation_m", 0.0)
                    ),
                }
            )
            candidates[key] = item
            candidate_to_relation[key] = relation_name
            item_index += 1

    for _ in range(5):
        app.update()

    contact_rows: dict[str, list[dict[str, Any]]] = {
        key: [] for key in candidates
    }

    def on_contact_event(
        contact_headers: Any,
        contact_data: Any,
        friction_anchors: Any,
    ) -> None:
        del friction_anchors
        for header in contact_headers:
            if header.type == ContactEventType.CONTACT_LOST:
                continue
            collider0 = str(
                PhysicsSchemaTools.intToSdfPath(header.collider0)
            )
            collider1 = str(
                PhysicsSchemaTools.intToSdfPath(header.collider1)
            )
            key0 = _candidate_from_path(collider0)
            key1 = _candidate_from_path(collider1)
            key = key0 if key0 == key1 else None
            if key not in contact_rows:
                continue
            start = _header_value(
                header, "contact_data_offset", "contactDataOffset"
            )
            count = _header_value(
                header, "num_contact_data", "numContactData"
            )
            details = []
            for data in contact_data[start : start + count]:
                details.append(
                    {
                        "separation_m": _data_value(
                            data, "separation", default=0.0
                        ),
                        "position": [
                            _data_value(data, "position", default=0.0)
                        ]
                        if not hasattr(data, "position")
                        else list(data.position),
                    }
                )
            contact_rows[key].append(
                {
                    "collider0": collider0,
                    "collider1": collider1,
                    "event_type": int(header.type),
                    "contacts": details,
                }
            )

    stage_cache = UsdUtils.StageCache.Get()
    stage_cache.Insert(stage)
    stage_id = stage_cache.GetId(stage).ToLongInt()
    settings = carb.settings.get_settings()
    old_update_to_usd = settings.get_as_bool(
        "/physics/updateToUsd"
    )
    old_fabric = settings.get_as_bool("/physics/fabricEnabled")
    settings.set("/physics/updateToUsd", False)
    settings.set("/physics/fabricEnabled", False)
    interface = get_physics_simulation_interface()
    initially_attached = interface.get_attached_stage() == stage_id
    if not initially_attached:
        interface.initialize(stage_id)
    subscription = interface.subscribe_physics_contact_report_events(
        on_contact_event
    )
    try:
        for step in range(2):
            interface.simulate(1.0 / 60.0, step / 60.0)
    finally:
        subscription = None
        if not initially_attached:
            interface.close()
        settings.set("/physics/updateToUsd", old_update_to_usd)
        settings.set("/physics/fabricEnabled", old_fabric)

    relations: dict[str, Any] = {}
    for relation_name, source_relation in scale_report["relations"].items():
        relation_candidates = []
        for key, item in candidates.items():
            if candidate_to_relation[key] != relation_name:
                continue
            events = contact_rows[key]
            separations = [
                contact["separation_m"]
                for event in events
                for contact in event["contacts"]
            ]
            relation_candidates.append(
                {
                    **item,
                    "contact_event_count": len(events),
                    "contact_point_count": len(separations),
                    "minimum_separation_m": (
                        float(min(separations)) if separations else None
                    ),
                    "maximum_penetration_m": (
                        float(max(0.0, -min(separations)))
                        if separations
                        else 0.0
                    ),
                    "has_physx_contact": bool(events),
                    "events": events,
                }
            )
        relation_candidates.sort(key=lambda row: row["candidate_index"])
        acceptance = dict(source_relation.get("isaac_acceptance", {}))
        cpu_limit = float(
            acceptance.get("maximum_cpu_physical_violation_m", 0.002)
        )
        isaac_limit = float(
            acceptance.get("maximum_isaac_penetration_m", 0.002)
        )
        separation_limit = float(
            acceptance.get(
                "maximum_isaac_positive_separation_m", 0.004
            )
        )
        require_contact = bool(
            acceptance.get("require_physx_contact", True)
        )
        visual_eligible = set(
            source_relation["selection"]["eligible_indices"]
        )
        accepted_indices = []
        for item in relation_candidates:
            index = int(item["candidate_index"])
            positive_separation = max(
                float(item["minimum_separation_m"] or 0.0), 0.0
            )
            accepted = bool(
                index in visual_eligible
                and float(
                    source_relation["candidates"][index][
                        "physical_violation_m"
                    ]
                )
                <= cpu_limit + 1e-12
                and float(item["maximum_penetration_m"])
                <= isaac_limit + 1e-12
                and positive_separation <= separation_limit + 1e-12
                and (
                    item["has_physx_contact"] or not require_contact
                )
            )
            item["accepted"] = accepted
            item["acceptance_checks"] = {
                "visual_gate": index in visual_eligible,
                "cpu_physical_violation": float(
                    source_relation["candidates"][index][
                        "physical_violation_m"
                    ]
                )
                <= cpu_limit + 1e-12,
                "isaac_penetration": float(
                    item["maximum_penetration_m"]
                )
                <= isaac_limit + 1e-12,
                "isaac_positive_separation": (
                    positive_separation <= separation_limit + 1e-12
                ),
                "physx_contact": (
                    item["has_physx_contact"] or not require_contact
                ),
            }
            if accepted:
                accepted_indices.append(index)
        preferred_index = (
            min(
                accepted_indices,
                key=lambda index: (
                    abs(
                        float(
                            relation_candidates[index][
                                "moving_scale_factor"
                            ]
                        )
                        - 1.0
                    ),
                    float(
                        source_relation["candidates"][index][
                            "visual_loss"
                        ]
                    ),
                ),
            )
            if accepted_indices
            else None
        )
        accepted_factors = [
            float(relation_candidates[index]["moving_scale_factor"])
            for index in accepted_indices
        ]
        relations[relation_name] = {
            "reference_part": source_relation["reference_part"],
            "moving_part": source_relation["moving_part"],
            "selected_scale_factor_cpu": source_relation["selection"][
                "selected_scale_factor"
            ],
            "joint_acceptance": {
                **acceptance,
                "policy": (
                    "visual gate AND CPU contact/nonpenetration gate AND "
                    "initial PhysX manifold gate; preferred candidate is "
                    "the accepted scale closest to the original calibration"
                ),
                "accepted_candidate_indices": accepted_indices,
                "accepted_scale_interval": (
                    [min(accepted_factors), max(accepted_factors)]
                    if accepted_factors
                    else None
                ),
                "preferred_candidate_index": preferred_index,
                "preferred_scale_factor": (
                    float(
                        relation_candidates[preferred_index][
                            "moving_scale_factor"
                        ]
                    )
                    if preferred_index is not None
                    else None
                ),
            },
            "candidates": relation_candidates,
        }

    report = {
        "schema_version": 1,
        "method": "isaac_physx_frozen_pose_initial_contact_probe",
        "trajectory_mutated": False,
        "dynamic_settling": False,
        "physics_projection_pose_produced": False,
        "note": (
            "Each zero-gravity dynamic clone starts at the immutable observed "
            "pose. PhysX was stepped only to construct its initial collision "
            "manifold; no simulated transform was read or accepted as pose "
            "supervision."
        ),
        "inputs": {
            "asset_root": str(asset_root),
            "scale_report": str(args.scale_report.resolve()),
            "trajectory": str(trajectory_path),
        },
        "relations": relations,
    }
    report_path = (
        args.report.resolve()
        if args.report
        else runtime_root / "qa/isaac_frozen_scale_query.json"
    )
    write_json(report_path, report)
    print(f"Isaac frozen-scale query report: {report_path}", flush=True)
    return report
