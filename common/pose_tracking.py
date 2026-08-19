"""Reusable pose tracking strategies for synchronized multi-view sequences."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from common.cloud_io import read_ply_xyz
from common.gicp import multiscale_gicp, subsample, transform_angle, voxel_unique
from common.normalized_recon import load_recon, recon_npz_path, scale_intrinsics
from common.multiview_quality import mask_area_quality
from common.pose_transforms import transform_points
from common.symmetry import SymmetrySpec, resolve_symmetric_pose


def load_part_cloud(
    root: Path, frame: int, part: str, minimum_points: int = 30
) -> np.ndarray | None:
    path = root / f"{frame:06d}" / f"{part}.ply"
    if not path.exists():
        return None
    points = read_ply_xyz(path)
    return points if len(points) >= minimum_points else None


def fuse_part_clouds(
    root: Path,
    frames: list[int],
    part: str,
    max_points: int,
    seed: int,
) -> np.ndarray:
    clouds = [load_part_cloud(root, frame, part) for frame in frames]
    clouds = [cloud for cloud in clouds if cloud is not None]
    if not clouds:
        raise RuntimeError(f"no cloud for {part} in frames {frames}")
    points = voxel_unique(np.concatenate(clouds), 0.002)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points


def align_symmetric_pose(
    pose: np.ndarray,
    reference: np.ndarray,
    axis_raw: np.ndarray,
    step_degrees: float = 5.0,
) -> np.ndarray:
    """Compatibility wrapper for continuous axial symmetry.

    New code should construct a :class:`common.symmetry.SymmetrySpec` and call
    :func:`common.symmetry.resolve_symmetric_pose` directly.
    """
    symmetry = SymmetrySpec(
        axis_raw=tuple(np.asarray(axis_raw, dtype=np.float64)),
        equivalence="continuous_axial",
        candidate_step_deg=float(step_degrees),
    )
    return resolve_symmetric_pose(
        pose,
        reference,
        symmetry,
        include_observation_ambiguities=False,
    ).pose


def interpolate_transform(delta: np.ndarray, fraction: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(
        Rotation.from_matrix(delta[:3, :3]).as_rotvec() * fraction
    ).as_matrix()
    result[:3, 3] = delta[:3, 3] * fraction
    return result


def interpolate_rigid_pose(
    first: np.ndarray, second: np.ndarray, fraction: float
) -> np.ndarray:
    """Interpolate SE(3) without rotating translation around the world origin."""

    amount = float(fraction)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (
        (1.0 - amount) * first[:3, 3] + amount * second[:3, 3]
    )
    rotations = Rotation.from_matrix(
        np.stack([first[:3, :3], second[:3, :3]])
    )
    result[:3, :3] = Slerp([0.0, 1.0], rotations)(
        [amount]
    ).as_matrix()[0]
    return result


def smooth_pose_sequence(
    poses: dict[int, np.ndarray],
    start: int,
    end: int,
    passes: int = 2,
) -> None:
    for _ in range(passes):
        old = {frame: poses[frame].copy() for frame in range(start, end + 1)}
        for frame in range(start + 1, end):
            weights = np.asarray([1.0, 4.0, 1.0])
            translations = np.stack([
                old[frame - 1][:3, 3],
                old[frame][:3, 3],
                old[frame + 1][:3, 3],
            ])
            rotations = Rotation.from_matrix(np.stack([
                old[frame - 1][:3, :3],
                old[frame][:3, :3],
                old[frame + 1][:3, :3],
            ]))
            current = old[frame].copy()
            current[:3, 3] = np.average(
                translations, axis=0, weights=weights
            )
            current[:3, :3] = rotations.mean(weights=weights).as_matrix()
            poses[frame] = current


def bridge_pose_ranges(
    poses: dict[int, np.ndarray],
    ranges: list[list[int]] | list[tuple[int, int]],
) -> list[dict]:
    """Interpolate only a short unreliable transition between two estimates.

    This is intended for the seam where forward-from-initial and
    backward-from-final anchor branches meet inside a known occlusion.  The
    endpoints are independently estimated neighbouring frames, not the two
    distant anchors, so observed motion outside the seam is preserved.
    """

    reports = []
    for raw_start, raw_end in ranges:
        start, end = int(raw_start), int(raw_end)
        left, right = start - 1, end + 1
        if start > end or left not in poses or right not in poses:
            raise ValueError(
                f"pose bridge {start}..{end} requires endpoints "
                f"{left} and {right}"
            )
        for frame in range(start, end + 1):
            fraction = (frame - left) / (right - left)
            poses[frame] = interpolate_rigid_pose(
                poses[left], poses[right], fraction
            )
        reports.append({
            "range": [start, end],
            "endpoint_frames": [left, right],
            "method": "local_se3_bridge_between_independent_branches",
        })
    return reports


def select_temporal_anchor(
    frame: int,
    anchor_frames: list[int],
    *,
    static_ranges: list[list[int]] | list[tuple[int, int]] = (),
    dynamic_ranges: list[list[int]] | list[tuple[int, int]] = (),
) -> int:
    """Select an anchor without switching inside a known static interval.

    Nearest-frame selection can switch to a post-motion anchor before the
    detected motion starts whenever anchors are unevenly spaced.  Static
    intervals instead prefer an anchor from their own interval; a dynamic
    interval switches between its bracketing anchors near the motion midpoint.
    """

    anchors = sorted(int(value) for value in anchor_frames)
    if not anchors:
        raise ValueError("temporal anchor selection requires an anchor")
    frame = int(frame)
    for raw_start, raw_end in static_ranges:
        start, end = int(raw_start), int(raw_end)
        if not start <= frame <= end:
            continue
        local = [value for value in anchors if start <= value <= end]
        if local:
            return min(local, key=lambda value: abs(value - frame))
        break
    for raw_start, raw_end in dynamic_ranges:
        start, end = int(raw_start), int(raw_end)
        if not start <= frame <= end:
            continue
        before = [value for value in anchors if value <= start]
        after = [value for value in anchors if value >= end]
        if before and after:
            return max(before) if frame <= (start + end) / 2.0 else min(after)
        if before:
            return max(before)
        if after:
            return min(after)
        break
    return min(anchors, key=lambda value: abs(value - frame))


def _best_pair_registration(
    source: np.ndarray,
    target: np.ndarray,
    previous: np.ndarray | None,
    config: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    maximum = int(config["max_points"])
    source = subsample(source, maximum, seed)
    target = subsample(target, maximum, seed + 1)
    centroid = np.eye(4)
    centroid[:3, 3] = target.mean(0) - source.mean(0)
    candidates = [("identity", np.eye(4)), ("centroid", centroid)]
    if previous is not None:
        candidates.insert(0, ("previous", previous))
    tree = cKDTree(target)
    probe = subsample(source, min(3000, len(source)), seed + 2)
    scored = []
    for name, initial in candidates:
        distances, _ = tree.query(transform_points(probe, initial), k=1)
        scored.append((float(np.median(distances)), name, initial))
    _, initial_name, initial = min(scored, key=lambda item: item[0])
    transform, quality = multiscale_gicp(source, target, initial, config)
    quality["init"] = initial_name
    quality["translation_m"] = float(np.linalg.norm(transform[:3, 3]))
    quality["rotation_deg"] = transform_angle(transform)
    quality["candidate_median_nn_m"] = {
        name: score for score, name, _initial in scored
    }
    return transform, quality


def track_cloud_registration(
    part: str,
    start: int,
    end: int,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    cloud_root: Path,
    registration_config: dict,
    symmetry: SymmetrySpec | None,
) -> tuple[dict[int, np.ndarray], dict]:
    clouds = {
        frame: load_part_cloud(cloud_root, frame, part)
        for frame in range(start, end + 1)
    }
    observed = [frame for frame, cloud in clouds.items() if cloud is not None]
    if not observed or observed[0] != start or observed[-1] != end:
        bounds = "none" if not observed else f"{observed[0]}..{observed[-1]}"
        raise RuntimeError(
            f"{part}: dynamic endpoints must have clouds, got {bounds}"
        )
    poses: dict[int, np.ndarray] = {start: start_pose.copy()}
    registrations = {}
    previous_pair = None
    previous_frame = start
    for frame in observed[1:]:
        gap = frame - previous_frame
        pair, quality = _best_pair_registration(
            clouds[frame],
            clouds[previous_frame],
            previous_pair,
            registration_config,
            seed=1000 + frame,
        )
        rejected = (
            quality["translation_m"]
            > float(registration_config.get("max_pair_translation_m", 0.12))
            * gap
            or quality["rotation_deg"]
            > float(registration_config.get("max_pair_rotation_deg", 30.0))
            * gap
            or quality["fitness_8mm"]
            < float(registration_config.get("minimum_pair_fitness", 0.15))
        )
        if rejected:
            pair = (
                np.linalg.matrix_power(previous_pair, gap)
                if previous_pair is not None
                else np.eye(4)
            )
            quality["rejected"] = True
            quality["fallback"] = (
                "constant_velocity" if previous_pair is not None else "identity"
            )
        else:
            quality["rejected"] = False
            previous_pair = pair
        current = np.linalg.inv(pair) @ poses[previous_frame]
        if symmetry is not None and registration_config.get(
            "symmetry_lock", True
        ):
            current = resolve_symmetric_pose(
                current,
                poses[previous_frame],
                symmetry,
                include_observation_ambiguities=False,
            ).pose
        poses[frame] = current
        registrations[f"{frame:06d}_to_{previous_frame:06d}"] = {
            "source": frame,
            "target": previous_frame,
            "T_target_from_source": pair.tolist(),
            "quality": quality,
        }
        print(
            f"{part} {frame:03d}->{previous_frame:03d} "
            f"fitness={quality['fitness_8mm']:.3f} "
            f"t={quality['translation_m']:.3f}m "
            f"r={quality['rotation_deg']:.1f}deg "
            f"rejected={quality['rejected']}",
            flush=True,
        )
        previous_frame = frame
    for left, right in zip(observed[:-1], observed[1:]):
        if right == left + 1:
            continue
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(
                np.stack([poses[left][:3, :3], poses[right][:3, :3]])
            ),
        )
        for frame in range(left + 1, right):
            fraction = (frame - left) / (right - left)
            pose = np.eye(4)
            pose[:3, :3] = slerp([fraction]).as_matrix()[0]
            pose[:3, 3] = (
                (1.0 - fraction) * poses[left][:3, 3]
                + fraction * poses[right][:3, 3]
            )
            poses[frame] = pose
    predicted_end = poses[end]
    if symmetry is not None:
        end_pose = resolve_symmetric_pose(
            end_pose,
            predicted_end,
            symmetry,
            include_observation_ambiguities=False,
        ).pose
    delta = end_pose @ np.linalg.inv(predicted_end)
    for frame in range(start, end + 1):
        fraction = (frame - start) / max(end - start, 1)
        poses[frame] = interpolate_transform(delta, fraction) @ poses[frame]
    poses[start] = start_pose.copy()
    poses[end] = end_pose.copy()
    smooth_pose_sequence(poses, start, end, passes=2)
    poses[start] = start_pose.copy()
    poses[end] = end_pose.copy()
    return poses, registrations


def track_cloud_registration_reverse(
    part: str,
    start: int,
    end: int,
    end_pose: np.ndarray,
    cloud_root: Path,
    registration_config: dict,
    symmetry: SymmetrySpec | None,
) -> tuple[dict[int, np.ndarray], dict]:
    """Track backward from a reliable end anchor.

    A part's first visible frame is often only a sliver and is a poor absolute
    mesh-registration anchor.  Pairwise cloud registration still carries
    useful motion, so this mirrors :func:`track_cloud_registration` while
    fixing the reliable pose at the end of the segment.
    """

    clouds = {
        frame: load_part_cloud(cloud_root, frame, part)
        for frame in range(start, end + 1)
    }
    observed = [frame for frame, cloud in clouds.items() if cloud is not None]
    if not observed or observed[0] != start or observed[-1] != end:
        bounds = "none" if not observed else f"{observed[0]}..{observed[-1]}"
        raise RuntimeError(
            f"{part}: dynamic endpoints must have clouds, got {bounds}"
        )
    poses: dict[int, np.ndarray] = {end: end_pose.copy()}
    registrations = {}
    previous_pair = None
    previous_frame = end
    for frame in reversed(observed[:-1]):
        gap = previous_frame - frame
        pair, quality = _best_pair_registration(
            clouds[frame],
            clouds[previous_frame],
            previous_pair,
            registration_config,
            seed=2000 + frame,
        )
        rejected = (
            quality["translation_m"]
            > float(registration_config.get("max_pair_translation_m", 0.12))
            * gap
            or quality["rotation_deg"]
            > float(registration_config.get("max_pair_rotation_deg", 30.0))
            * gap
            or quality["fitness_8mm"]
            < float(registration_config.get("minimum_pair_fitness", 0.15))
        )
        if rejected:
            pair = (
                np.linalg.matrix_power(previous_pair, gap)
                if previous_pair is not None
                else np.eye(4)
            )
            quality["rejected"] = True
            quality["fallback"] = (
                "constant_velocity" if previous_pair is not None else "identity"
            )
        else:
            quality["rejected"] = False
            previous_pair = pair
        current = np.linalg.inv(pair) @ poses[previous_frame]
        if symmetry is not None and registration_config.get(
            "symmetry_lock", True
        ):
            current = resolve_symmetric_pose(
                current,
                poses[previous_frame],
                symmetry,
                include_observation_ambiguities=False,
            ).pose
        poses[frame] = current
        registrations[f"{frame:06d}_to_{previous_frame:06d}"] = {
            "source": frame,
            "target": previous_frame,
            "T_target_from_source": pair.tolist(),
            "quality": quality,
            "tracking_direction": "reverse_from_end_anchor",
        }
        print(
            f"{part} {frame:03d}->{previous_frame:03d} reverse "
            f"fitness={quality['fitness_8mm']:.3f} "
            f"t={quality['translation_m']:.3f}m "
            f"r={quality['rotation_deg']:.1f}deg "
            f"rejected={quality['rejected']}",
            flush=True,
        )
        previous_frame = frame
    for left, right in zip(observed[:-1], observed[1:]):
        if right == left + 1:
            continue
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(
                np.stack([poses[left][:3, :3], poses[right][:3, :3]])
            ),
        )
        for frame in range(left + 1, right):
            fraction = (frame - left) / (right - left)
            pose = np.eye(4)
            pose[:3, :3] = slerp([fraction]).as_matrix()[0]
            pose[:3, 3] = (
                (1.0 - fraction) * poses[left][:3, 3]
                + fraction * poses[right][:3, 3]
            )
            poses[frame] = pose
    smooth_pose_sequence(poses, start, end, passes=2)
    poses[end] = end_pose.copy()
    return poses, registrations


def track_anchor_relative_registration(
    part: str,
    mesh: trimesh.Trimesh,
    scale: float,
    origin_raw: np.ndarray,
    start: int,
    end: int,
    anchor_poses: dict[int, np.ndarray],
    cloud_root: Path,
    registration_config: dict,
    observable_frames: set[int] | None = None,
) -> tuple[dict[int, np.ndarray], dict]:
    """Fit every observed frame directly from a stable absolute anchor.

    Unlike pairwise cloud tracking, no fitted transform is used to initialize
    the next frame.  Each frame starts from the nearest configured anchor's
    orientation and the current cloud centroid, so registration errors cannot
    accumulate along the sequence.
    """

    if not anchor_poses:
        raise ValueError(f"{part}: anchor-relative tracking needs an anchor")
    rng = np.random.default_rng(4103 + start)
    raw, _ = trimesh.sample.sample_surface(
        mesh,
        int(registration_config.get("anchor_model_points", 20000)),
        seed=rng,
    )
    canonical = float(scale) * (
        np.asarray(raw, dtype=np.float64)
        - np.asarray(origin_raw, dtype=np.float64)
    )
    canonical = subsample(
        canonical,
        int(registration_config.get("max_points", 16000)),
        5103 + start,
    )
    canonical_center = np.median(canonical, axis=0)
    anchor_frames = sorted(anchor_poses)
    static_ranges = registration_config.get("state_static_ranges", [])
    dynamic_ranges = registration_config.get("state_dynamic_ranges", [])
    owner_by_frame = {
        frame: select_temporal_anchor(
            frame,
            anchor_frames,
            static_ranges=static_ranges,
            dynamic_ranges=dynamic_ranges,
        )
        for frame in range(start, end + 1)
    }
    tracking_frames = (
        set(range(start, end + 1))
        if observable_frames is None
        else {
            int(frame)
            for frame in observable_frames
            if start <= int(frame) <= end
        }
    )
    # Calibrated anchors remain authoritative even if a later visibility
    # configuration becomes stricter than the calibration gate.
    tracking_frames.update(anchor_frames)
    tracklet_by_frame: dict[int, int] = {}
    tracklet_ranges: list[list[int]] = []
    for frame in sorted(tracking_frames):
        if not tracklet_ranges or frame != tracklet_ranges[-1][1] + 1:
            tracklet_ranges.append([frame, frame])
        else:
            tracklet_ranges[-1][1] = frame
        tracklet_by_frame[frame] = len(tracklet_ranges) - 1
    max_rotation_step = float(
        registration_config.get("maximum_absolute_rotation_step_deg", 35.0)
    )
    max_translation_step = float(
        registration_config.get("maximum_absolute_translation_step_m", 0.08)
    )
    rotation_gap_scale_cap = float(
        registration_config.get(
            "maximum_absolute_rotation_gap_scale", 2.0
        )
    )
    poses: dict[int, np.ndarray] = {}
    cloud_centroids: dict[int, np.ndarray] = {}
    reports: dict[str, dict] = {
        "tracking_summary": {
            "method": (
                "bidirectional_anchor_relative_registration"
                if len(anchor_frames) >= 2
                else "single_anchor_relative_registration"
            ),
            "anchor_frames": [int(value) for value in anchor_frames],
            "no_pairwise_pose_accumulation": True,
            "visibility_tracklets": tracklet_ranges,
            "reacquisition": "independent_absolute_pose_at_tracklet_entry",
        }
    }
    accepted_frames: list[int] = []
    for frame in range(start, end + 1):
        if frame in anchor_poses:
            poses[frame] = anchor_poses[frame].copy()
            accepted_frames.append(frame)
            reports[f"{frame:06d}"] = {
                "status": "anchor",
                "anchor_frame": frame,
                "initialization": "stable_absolute_anchor",
                "tracklet_id": int(tracklet_by_frame[frame]),
            }
            continue
        if frame not in tracking_frames:
            reports[f"{frame:06d}"] = {
                "status": "visibility_rejected",
                "pose_valid": False,
            }
            continue
        cloud = load_part_cloud(cloud_root, frame, part)
        if cloud is None:
            reports[f"{frame:06d}"] = {
                "status": "missing_quality_cloud",
            }
            continue
        anchor_frame = owner_by_frame[frame]
        initial_pose = anchor_poses[anchor_frame].copy()
        rotation = initial_pose[:3, :3]
        cloud_centroid = np.median(cloud, axis=0)
        cloud_centroids[frame] = cloud_centroid
        initial_pose[:3, 3] = cloud_centroid - canonical_center @ rotation.T
        cloud_sample = subsample(
            cloud,
            int(registration_config.get("max_points", 16000)),
            6103 + frame,
        )
        initializations = [("absolute_anchor", initial_pose)]
        # Once an anchor has been reached, add a local orientation proposal
        # from the preceding fitted frame.  Both proposals still register the
        # canonical mesh directly to the current cloud; the previous pose is
        # only an optimizer seed, never a composed pairwise transform.  This
        # lets genuine rotation advance gradually while preventing GICP from
        # repeatedly re-entering a distant 90/180-degree symmetric basin.
        previous_frames = [
            previous
            for previous in poses
            if previous < frame
            and owner_by_frame.get(previous) == anchor_frame
            and previous >= anchor_frame
            and tracklet_by_frame.get(previous) == tracklet_by_frame.get(frame)
        ]
        continuity_frame = max(previous_frames) if previous_frames else anchor_frame
        continuity_pose = poses.get(continuity_frame, anchor_poses[anchor_frame])
        if previous_frames:
            local_pose = continuity_pose.copy()
            local_rotation = local_pose[:3, :3]
            local_pose[:3, 3] = (
                cloud_centroid - canonical_center @ local_rotation.T
            )
            initializations.append(("previous_orientation", local_pose))

        candidate_rows = []
        for candidate_index, (initialization_name, candidate_initial) in enumerate(
            initializations
        ):
            canonical_from_world, candidate_quality = multiscale_gicp(
                cloud_sample,
                canonical,
                np.linalg.inv(candidate_initial),
                registration_config,
            )
            candidate_pose = np.linalg.inv(canonical_from_world)
            gap = max(1, frame - continuity_frame)
            rotation_step = float(np.degrees(Rotation.from_matrix(
                candidate_pose[:3, :3] @ continuity_pose[:3, :3].T
            ).magnitude()))
            translation_step = float(np.linalg.norm(
                candidate_pose[:3, 3] - continuity_pose[:3, 3]
            ))
            quality_passed = bool(
                candidate_quality["fitness_8mm"]
                >= float(
                    registration_config.get("minimum_absolute_fitness", 0.08)
                )
                and candidate_quality["median_nn_m"]
                <= float(
                    registration_config.get(
                        "maximum_absolute_median_nn_m", 0.04
                    )
                )
            )
            continuity_passed = bool(
                not previous_frames
                or (
                    rotation_step
                    <= max_rotation_step
                    * min(float(gap), rotation_gap_scale_cap)
                    and translation_step <= max_translation_step * gap
                )
            )
            candidate_rows.append({
                "index": candidate_index,
                "initialization": initialization_name,
                "pose": candidate_pose,
                "quality": candidate_quality,
                "quality_passed": quality_passed,
                "continuity_passed": continuity_passed,
                "rotation_from_previous_deg": rotation_step,
                "translation_from_previous_m": translation_step,
            })
        selected_candidate = min(
            candidate_rows,
            key=lambda row: (
                not row["quality_passed"],
                not row["continuity_passed"],
                float(row["quality"]["median_nn_m"]),
                -float(row["quality"]["fitness_8mm"]),
            ),
        )
        pose = selected_candidate["pose"]
        quality = selected_candidate["quality"]
        accepted = bool(
            selected_candidate["quality_passed"]
        )
        is_reentry = bool(
            not previous_frames
            and frame > start
            and frame - 1 not in tracking_frames
        )
        if is_reentry:
            accepted = bool(
                accepted
                and quality["fitness_8mm"]
                >= float(
                    registration_config.get(
                        "reentry_minimum_absolute_fitness",
                        registration_config.get(
                            "minimum_absolute_fitness", 0.08
                        ),
                    )
                )
                and quality["median_nn_m"]
                <= float(
                    registration_config.get(
                        "reentry_maximum_absolute_median_nn_m",
                        registration_config.get(
                            "maximum_absolute_median_nn_m", 0.04
                        ),
                    )
                )
            )
        reports[f"{frame:06d}"] = {
            "status": "accepted" if accepted else "rejected",
            "anchor_frame": int(anchor_frame),
            "tracking_direction": (
                "forward_from_earlier_anchor"
                if anchor_frame < frame
                else "backward_from_later_anchor"
            ),
            "initialization": selected_candidate["initialization"],
            "initialization_candidates": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "pose"
                }
                for row in candidate_rows
            ],
            "continuity_reference_frame": int(continuity_frame),
            "tracklet_id": int(tracklet_by_frame[frame]),
            "tracklet_entry": is_reentry,
            "continuity_enforced": bool(previous_frames),
            "quality": quality,
        }
        if is_reentry:
            reports[f"{frame:06d}"]["reacquisition"] = (
                "current_cloud_absolute_pose"
            )
        if accepted:
            poses[frame] = pose
            accepted_frames.append(frame)

    if not accepted_frames:
        raise RuntimeError(f"{part}: no frame passed anchor-relative registration")
    # Missing quality clouds should not turn a short occlusion into permission
    # for an arbitrary 90/180-degree ICP basin switch.  Allow a small amount
    # of gap scaling for genuine motion, but cap it independently from the
    # number of missing frames.
    def gate_from_anchor(anchor_frame: int, frames: list[int]) -> None:
        previous_frame = anchor_frame
        previous_pose = poses[anchor_frame]
        previous_tracklet = tracklet_by_frame.get(anchor_frame)
        for frame in frames:
            if frame not in poses:
                continue
            current_tracklet = tracklet_by_frame.get(frame)
            if current_tracklet != previous_tracklet:
                row = reports[f"{frame:06d}"]
                row["continuity_reset"] = "visibility_tracklet_reacquisition"
                previous_frame = frame
                previous_pose = poses[frame]
                previous_tracklet = current_tracklet
                continue
            gap = max(1, abs(frame - previous_frame))
            rotation_gap_scale = min(float(gap), rotation_gap_scale_cap)
            pose = poses[frame]
            rotation_step = float(np.degrees(Rotation.from_matrix(
                pose[:3, :3] @ previous_pose[:3, :3].T
            ).magnitude()))
            translation_step = float(np.linalg.norm(
                pose[:3, 3] - previous_pose[:3, 3]
            ))
            row = reports[f"{frame:06d}"]
            row["continuity_from_frame"] = int(previous_frame)
            row["absolute_rotation_step_deg"] = rotation_step
            row["absolute_translation_step_m"] = translation_step
            if translation_step > max_translation_step * gap:
                del poses[frame]
                row["status"] = "rejected_translation_discontinuity"
                continue
            if rotation_step > max_rotation_step * rotation_gap_scale:
                pose[:3, :3] = previous_pose[:3, :3]
                # The GICP translation was optimized jointly with the rejected
                # rotation basin.  Keeping it after replacing only the rotation
                # creates an internally inconsistent pose and visible position
                # jumps even for a static object.  Reuse the anchor-rotation
                # centroid initialization, which was computed from this frame's
                # observed cloud, whenever orientation falls back.
                cloud_centroid = cloud_centroids.get(frame)
                if cloud_centroid is not None:
                    pose[:3, 3] = (
                        cloud_centroid
                        - canonical_center @ previous_pose[:3, :3].T
                    )
                    row["translation_fallback"] = (
                        "accepted_rotation_current_cloud_centroid"
                    )
                row["orientation_fallback"] = "previous_accepted_orientation"
            previous_frame = frame
            previous_pose = pose
            previous_tracklet = current_tracklet

    for anchor_frame in anchor_frames:
        gate_from_anchor(
            anchor_frame,
            sorted(
                (
                    frame
                    for frame, owner in owner_by_frame.items()
                    if owner == anchor_frame and frame < anchor_frame
                ),
                reverse=True,
            ),
        )
        gate_from_anchor(
            anchor_frame,
            sorted(
                frame
                for frame, owner in owner_by_frame.items()
                if owner == anchor_frame and frame > anchor_frame
            ),
        )
    observed = sorted(poses)
    for frame in range(start, end + 1):
        if frame in poses:
            continue
        left = max((value for value in observed if value < frame), default=None)
        right = min((value for value in observed if value > frame), default=None)
        if left is None:
            poses[frame] = poses[right].copy()
        elif right is None:
            poses[frame] = poses[left].copy()
        else:
            fraction = (frame - left) / (right - left)
            poses[frame] = interpolate_rigid_pose(
                poses[left], poses[right], fraction
            )
        reports[f"{frame:06d}"]["fallback"] = "absolute_pose_interpolation"
    bridge_reports = bridge_pose_ranges(
        poses,
        registration_config.get("anchor_transition_ranges", []),
    )
    for bridge in bridge_reports:
        for frame in range(bridge["range"][0], bridge["range"][1] + 1):
            reports[f"{frame:06d}"]["fallback"] = (
                "bidirectional_local_transition_bridge"
            )
    reports["tracking_summary"]["transition_bridges"] = bridge_reports
    smoothing_passes = int(
        registration_config.get("anchor_smoothing_passes", 1)
    )
    for tracklet_start, tracklet_end in tracklet_ranges:
        if tracklet_end - tracklet_start >= 2:
            smooth_pose_sequence(
                poses,
                tracklet_start,
                tracklet_end,
                passes=smoothing_passes,
            )
    for frame, pose in anchor_poses.items():
        if start <= frame <= end:
            poses[frame] = pose.copy()
    return poses, reports


def track_model_translation(
    part: str,
    mesh: trimesh.Trimesh,
    scale: float,
    origin_raw: np.ndarray,
    start: int,
    end: int,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    cloud_root: Path,
    seed: int = 0,
) -> tuple[dict[int, np.ndarray], dict]:
    """Track a near-axisymmetric object using translation-only ICP."""
    rng = np.random.default_rng(seed)
    raw, _ = trimesh.sample.sample_surface(mesh, 30000, seed=rng)
    canonical = scale * (np.asarray(raw, float) - origin_raw)
    if len(canonical) > 18000:
        canonical = canonical[
            rng.choice(len(canonical), 18000, replace=False)
        ]
    rotation = start_pose[:3, :3].copy()
    final = end_pose.copy()
    final[:3, :3] = rotation
    poses = {start: start_pose.copy()}
    records = {}
    velocity = np.zeros(3)
    for frame in range(start + 1, end + 1):
        cloud = load_part_cloud(cloud_root, frame, part)
        predicted = poses[frame - 1].copy()
        predicted[:3, :3] = rotation
        predicted[:3, 3] += np.clip(velocity, -0.035, 0.035)
        if cloud is None:
            poses[frame] = predicted
            records[f"{frame:06d}"] = {"status": "inferred", "n_gated": 0}
            continue
        cloud = subsample(cloud, 18000, seed + frame)
        model_world = canonical @ rotation.T + predicted[:3, 3]
        tree = cKDTree(model_world)
        correction_accumulator = np.zeros(3)
        n_gated = 0
        median_distance = None
        for _ in range(12):
            moved = cloud + correction_accumulator
            distances, indices = tree.query(moved, k=1)
            gate = distances < 0.065
            if gate.sum() < 100:
                break
            threshold = np.quantile(distances[gate], 0.60)
            keep = gate & (distances <= threshold)
            residual = model_world[indices[keep]] - moved[keep]
            increment = np.median(residual, axis=0)
            correction_accumulator += 0.75 * increment
            n_gated = int(keep.sum())
            median_distance = float(np.median(distances[keep]))
            if np.linalg.norm(increment) < 2e-4:
                break
        correction = -correction_accumulator
        if np.linalg.norm(correction) > 0.065 or n_gated < 100:
            correction[:] = 0.0
            accepted = False
        else:
            accepted = True
        current = predicted.copy()
        current[:3, 3] += correction
        new_velocity = current[:3, 3] - poses[frame - 1][:3, 3]
        velocity = 0.65 * velocity + 0.35 * new_velocity
        poses[frame] = current
        records[f"{frame:06d}"] = {
            "status": "measured" if accepted else "motion_prior",
            "n_gated": n_gated,
            "median_distance_m": median_distance,
            "correction_m": correction.tolist(),
        }
    drift = final[:3, 3] - poses[end][:3, 3]
    for frame in range(start, end + 1):
        fraction = (frame - start) / max(end - start, 1)
        poses[frame][:3, 3] += fraction * drift
    smooth_pose_sequence(poses, start, end, passes=3)
    poses[start] = start_pose.copy()
    poses[end] = final.copy()
    return poses, records


def _project_bbox(
    points_part: np.ndarray,
    center_world: np.ndarray,
    rotation_world_part: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    world = points_part @ rotation_world_part.T + center_world
    camera = world @ extrinsic[:, :3].T + extrinsic[:, 3]
    uvw = camera @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return np.concatenate([uv.min(axis=0), uv.max(axis=0)])


def _load_mask_bboxes(
    mask_root: Path,
    frame: int,
    part_id: int,
    views: list[str],
    *,
    minimum_pixels: int = 1000,
    maximum_area_ratio: float = 4.0,
) -> list[tuple[np.ndarray, tuple[int, int]] | None]:
    from PIL import Image

    labels_by_view = {
        view: np.asarray(
            Image.open(mask_root / f"{frame:06d}" / f"{view}.png")
        )
        for view in views
        if (mask_root / f"{frame:06d}" / f"{view}.png").exists()
    }
    quality = mask_area_quality(
        labels_by_view,
        part_id,
        minimum_pixels=minimum_pixels,
        maximum_area_ratio=maximum_area_ratio,
    )
    result = []
    for view in views:
        if not quality["views"].get(view, {}).get("valid", False):
            result.append(None)
            continue
        labels = labels_by_view[view]
        rows, columns = np.where(labels == part_id)
        bbox = np.asarray([
            columns.min(), rows.min(), columns.max(), rows.max()
        ], dtype=float)
        result.append((bbox, labels.shape))
    return result


def track_mask_bbox_translation(
    part: str,
    mesh: trimesh.Trimesh,
    scale: float,
    origin_raw: np.ndarray,
    start: int,
    end: int,
    start_pose: np.ndarray | None,
    end_pose: np.ndarray,
    mask_root: Path,
    part_id: int,
    views: list[str],
    config: dict,
) -> tuple[dict[int, np.ndarray], dict]:
    """Track translation from multi-view envelopes and anchor orientations."""
    envelope = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
    if len(envelope) > 12000:
        envelope = envelope[
            np.random.default_rng(1701 + start).choice(
                len(envelope), 12000, replace=False
            )
        ]
    canonical = scale * (envelope - origin_raw)
    final = end_pose.copy()
    has_start_anchor = start_pose is not None
    start_template = final.copy() if start_pose is None else start_pose.copy()
    rotation_slerp = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(
            np.stack([start_template[:3, :3], final[:3, :3]])
        ),
    )

    def rotation_at(fraction: float) -> np.ndarray:
        return rotation_slerp(
            [float(np.clip(fraction, 0.0, 1.0))]
        ).as_matrix()[0]

    backend = config["recon_backend"]
    initial_recon = load_recon(config, f"{start:06d}", backend=backend)
    depth_hw = initial_recon["depth_hw"]
    camera_cache = {}
    bbox_cache = {}

    def observations(frame: int):
        if frame not in bbox_cache:
            view_quality = config.get("view_quality", {})
            boxes = _load_mask_bboxes(
                mask_root,
                frame,
                part_id,
                views,
                minimum_pixels=int(
                    view_quality.get("minimum_full_mask_pixels", 800)
                ),
                maximum_area_ratio=float(
                    view_quality.get("maximum_mask_area_ratio", 4.0)
                ),
            )
            with np.load(
                recon_npz_path(config, f"{frame:06d}", backend)
            ) as data:
                intrinsics = np.asarray(
                    data["intrinsic"]
                    if "intrinsic" in data.files
                    else data["intrinsics"],
                    dtype=np.float64,
                )
                extrinsics = np.asarray(
                    data["extrinsic"]
                    if "extrinsic" in data.files
                    else data["extrinsics"],
                    dtype=np.float64,
                )
            cameras = []
            for index, box in enumerate(boxes):
                shape = box[1] if box is not None else (1080, 1920)
                cameras.append((
                    scale_intrinsics(intrinsics[index], depth_hw, shape),
                    extrinsics[index],
                ))
            bbox_cache[frame] = boxes
            camera_cache[frame] = cameras
        return bbox_cache[frame], camera_cache[frame]

    def anchor_bias(anchor_pose: np.ndarray, frames: list[int]) -> np.ndarray:
        per_view = [[] for _ in views]
        for frame in frames:
            boxes, cameras = observations(frame)
            for index, item in enumerate(boxes):
                if item is None:
                    continue
                observed, _shape = item
                predicted = _project_bbox(
                    canonical,
                    anchor_pose[:3, 3],
                    anchor_pose[:3, :3],
                    *cameras[index],
                )
                per_view[index].append(predicted - observed)
        return np.stack([
            np.median(values, axis=0) if values else np.zeros(4)
            for values in per_view
        ])

    tracker_config = config["states"][part].get("mask_bbox_tracking", {})
    tracker_config = tracker_config.get("ranges", {}).get(
        f"{start}-{end}", tracker_config
    )
    start_bias_frames = [
        int(value)
        for value in tracker_config.get("start_bias_frames", [start])
    ]
    end_bias_frames = [
        int(value)
        for value in tracker_config.get("end_bias_frames", [end])
    ]
    bias_end = anchor_bias(final, end_bias_frames)
    bias_start = (
        anchor_bias(start_template, start_bias_frames)
        if has_start_anchor
        else bias_end.copy()
    )
    poses = {start: start_template.copy()} if has_start_anchor else {}
    records = {}
    velocity = np.zeros(3)
    rotations = (
        {start: start_template[:3, :3].copy()}
        if has_start_anchor
        else {}
    )
    velocity_cap = float(
        tracker_config.get("prediction_velocity_cap_m", 0.15)
    )
    motion_weight = float(tracker_config.get("motion_prior_weight", 0.5))
    motion_sigma = float(
        tracker_config.get("motion_prior_sigma_m", 0.08)
    )
    linear_weight = float(tracker_config.get("linear_prior_weight", 0.1))
    linear_sigma = float(
        tracker_config.get("linear_prior_sigma_m", 0.1)
    )
    maximum_step = float(
        tracker_config.get("max_translation_step_m", 0.35)
    )
    first_frame = start + 1 if has_start_anchor else start
    for frame in range(first_frame, end + 1):
        boxes, cameras = observations(frame)
        fraction = (frame - start) / max(end - start, 1)
        current_rotation = rotation_at(fraction)
        rotations[frame] = current_rotation
        bias = (1.0 - fraction) * bias_start + fraction * bias_end
        previous = (
            poses[frame - 1][:3, 3]
            if frame - 1 in poses
            else final[:3, 3]
        )
        predicted_motion = previous + np.clip(
            velocity, -velocity_cap, velocity_cap
        )
        linear_prior = (
            (1.0 - fraction) * start_template[:3, 3]
            + fraction * final[:3, 3]
        )
        edge_count = 0

        def residual(center: np.ndarray) -> np.ndarray:
            nonlocal edge_count
            values = []
            current_edge_count = 0
            for index, item in enumerate(boxes):
                if item is None:
                    continue
                observed, (height, width) = item
                predicted = _project_bbox(
                    canonical, center, current_rotation, *cameras[index]
                )
                valid = np.asarray([
                    observed[0] > 1,
                    observed[1] > 1,
                    observed[2] < width - 2,
                    observed[3] < height - 2,
                ])
                values.extend(
                    (
                        predicted[valid]
                        - observed[valid]
                        - bias[index, valid]
                    ).tolist()
                )
                current_edge_count += int(valid.sum())
            edge_count = current_edge_count
            values.extend(
                (
                    motion_weight
                    * (center - predicted_motion)
                    / motion_sigma
                ).tolist()
            )
            values.extend(
                (
                    linear_weight
                    * (center - linear_prior)
                    / linear_sigma
                ).tolist()
            )
            return np.asarray(values)

        result = least_squares(
            residual,
            predicted_motion,
            loss="soft_l1",
            f_scale=10.0,
            max_nfev=80,
            xtol=1e-7,
            ftol=1e-7,
        )
        center = result.x
        step = center - previous
        if np.linalg.norm(step) > maximum_step:
            center = previous + step * (
                maximum_step / np.linalg.norm(step)
            )
        velocity = 0.65 * velocity + 0.35 * (center - previous)
        pose = start_template.copy()
        pose[:3, :3] = current_rotation
        pose[:3, 3] = center
        poses[frame] = pose
        edge_residual = residual(center)[:edge_count]
        records[f"{frame:06d}"] = {
            "status": "measured" if edge_count >= 4 else "motion_prior",
            "valid_silhouette_edges": edge_count,
            "median_edge_error_px": (
                float(np.median(np.abs(edge_residual)))
                if len(edge_residual)
                else None
            ),
            "translation_step_m": float(np.linalg.norm(center - previous)),
        }
    smooth_pose_sequence(poses, start, end, passes=2)
    for frame, pose in poses.items():
        pose[:3, :3] = rotations[frame]
    if has_start_anchor:
        poses[start] = start_template.copy()
    poses[end] = final.copy()
    records["calibration"] = {
        "start_bias_frames": start_bias_frames,
        "end_bias_frames": end_bias_frames,
        "start_edge_bias_px": bias_start.tolist(),
        "end_edge_bias_px": bias_end.tolist(),
        "has_start_anchor": has_start_anchor,
    }
    return poses, records
