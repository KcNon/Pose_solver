"""Canonical orchestration for the reusable mask/depth/pose pipeline."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

from common.io_utils import load_json
from pose_solver.artifacts import ArtifactLayout, source_digest, update_manifest
from pose_solver.config import PipelineConfig
from pose_solver.resolved import (
    depth_config,
    frame_config,
    mask_config,
    masks_dir,
    point_cloud_root,
    pose_config,
    resolved_part_starts,
    write_resolved,
)


ROOT = Path(__file__).resolve().parents[1]
VALID_STAGES = (
    "all", "preflight", "frames", "mask", "depth", "pose", "render"
)


class PipelineRunner:
    """Validate, materialize, and execute one source configuration."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        force: bool = False,
        dry_run: bool = False,
        skip_review: bool = False,
        skip_render: bool = False,
    ) -> None:
        self.config = config
        self.layout = ArtifactLayout(config.output_root)
        self.force = bool(force)
        self.dry_run = bool(dry_run)
        self.skip_review = bool(skip_review)
        self.skip_render = bool(skip_render)

    def _environment(self) -> dict[str, str]:
        return {
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(device) for device in self.config.devices
            ),
            # pyrender's EGL backend ignores CUDA_VISIBLE_DEVICES. On shared
            # hosts its enumeration may differ from CUDA indices, so configs
            # can provide the mapping audited by list_egl_devices.py.
            "EGL_DEVICE_ID": str(
                self.config.raw.get("runtime", {}).get(
                    "egl_device", self.config.devices[0]
                )
            ),
        }

    def _run_adapter(
        self,
        command: Iterable[str],
        adapter: Callable[[list[str]], Any],
        arguments: list[str],
    ) -> Any:
        normalized = [str(value) for value in command]
        print("[pipeline] " + " ".join(normalized), flush=True)
        if self.dry_run:
            return None
        stage_environment = self._environment()
        previous = {key: os.environ.get(key) for key in stage_environment}
        os.environ.update(stage_environment)
        try:
            return adapter(arguments)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def preflight(self, *, selected_stage: str = "all") -> dict[str, Any]:
        config = self.config
        missing_views = (
            [
                view for view in config.views
                if not (config.frames_dir / view).is_dir()
            ]
            if config.frames_dir.is_dir()
            else list(config.views)
        )
        can_extract = bool(config.videos) and selected_stage in {
            "all", "preflight", "frames"
        }
        if missing_views and not can_extract:
            raise FileNotFoundError(
                f"input.frames_dir is missing views {missing_views}; provide "
                "input.videos and run --stage frames first"
            )
        needs_video_files = bool(missing_views) or (
            selected_stage in {"all", "frames"} and self.force
        )
        if config.videos and needs_video_files:
            missing_videos = [
                str(path) for path in config.videos.values() if not path.is_file()
            ]
            if missing_videos:
                raise FileNotFoundError(f"missing input videos: {missing_videos}")
        missing_meshes = [
            str(part.mesh)
            for part in config.parts
            if not part.mesh.is_file()
        ]
        if missing_meshes:
            raise FileNotFoundError(f"missing part meshes: {missing_meshes}")
        if config.mask.mode == "reuse" and not config.mask.artifact.is_dir():
            raise FileNotFoundError(f"mask.artifact: {config.mask.artifact}")
        if config.depth.mode == "reuse" and not config.depth.artifact.is_dir():
            raise FileNotFoundError(f"depth.artifact: {config.depth.artifact}")
        if config.pose.mode == "reuse" and not config.pose.artifact.is_file():
            raise FileNotFoundError(f"pose.artifact: {config.pose.artifact}")
        needs_masks = (
            selected_stage == "depth" and config.depth.mode == "run"
        ) or (
            selected_stage == "pose" and config.pose.mode == "run"
        )
        if needs_masks and config.mask.mode == "run":
            existing_masks = self.layout.mask_output / "masks"
            if not existing_masks.is_dir():
                raise FileNotFoundError(
                    f"{selected_stage} requires existing masks at "
                    f"{existing_masks}; run --stage mask first"
                )
        if (
            selected_stage == "pose"
            and config.pose.mode == "run"
            and config.depth.mode == "run"
        ):
            existing_clouds = point_cloud_root(config, self.layout)
            if not existing_clouds.is_dir():
                raise FileNotFoundError(
                    f"pose requires existing point clouds at {existing_clouds}; "
                    "run --stage depth first"
                )
        required_models: list[tuple[str, bool]] = []
        if selected_stage in {"all", "mask"} and config.mask.mode == "run":
            required_models.extend((
                ("qwen_python", True),
                ("qwen_model", False),
                ("sam_python", True),
                ("sam_checkpoint", True),
            ))
        if (
            selected_stage in {"all", "depth"}
            and config.depth.mode == "run"
            and not config.input_options.get("depth_dir")
        ):
            required_models.append(("da3_python", True))
        for name, require_file in required_models:
            value = config.models.get(name)
            if not value:
                raise ValueError(f"models.{name} is required")
            path = Path(value)
            if require_file and not path.is_file():
                raise FileNotFoundError(f"models.{name}: {path}")
            if not require_file and not path.exists():
                raise FileNotFoundError(f"models.{name}: {path}")
        for stage_name, stage in (
            ("mask", config.mask),
            ("depth", config.depth),
            ("pose", config.pose),
        ):
            if (
                stage.compatibility_config is not None
                and not stage.compatibility_config.is_file()
            ):
                raise FileNotFoundError(
                    f"{stage_name}.compatibility_config: "
                    f"{stage.compatibility_config}"
                )
        report = {
            "dataset": config.dataset,
            "frames": [config.frame_start, config.frame_end],
            "views": list(config.views),
            "parts": config.part_names,
            "devices": list(config.devices),
            "selected_stage": selected_stage,
            "modes": {
                "frames": "run" if config.videos else "reuse",
                "mask": config.mask.mode,
                "depth": config.depth.mode,
                "pose": config.pose.mode,
            },
        }
        print(
            f"preflight: {len(config.views)} views, {len(config.parts)} parts, "
            f"frames {config.frame_start}..{config.frame_end}, "
            f"GPUs {list(config.devices)}",
            flush=True,
        )
        return report

    def _frames(self) -> Path:
        config = self.config
        existing = all(
            (config.frames_dir / view).is_dir() for view in config.views
        )
        if existing and not self.force:
            print(f"[reuse] frames {config.frames_dir}", flush=True)
            return config.frames_dir
        if not config.videos:
            if existing:
                return config.frames_dir
            raise FileNotFoundError(f"input.frames_dir: {config.frames_dir}")
        resolved_path = write_resolved(
            self.layout.resolved_config("frames"),
            frame_config(config),
        )
        arguments = ["--config", str(resolved_path)]
        if self.force:
            arguments.append("--force")
        command = [
            sys.executable,
            "-u",
            str(
                ROOT
                / "tools"
                / "stages"
                / "preprocess"
                / "extract_synchronized_video_frames.py"
            ),
            *arguments,
        ]
        from tools.stages.preprocess.extract_synchronized_video_frames import (
            main as extract_frames,
        )

        self._run_adapter(command, extract_frames, arguments)
        return config.frames_dir

    def _mask(self) -> Path:
        config = self.config
        if config.mask.mode == "reuse":
            assert config.mask.artifact is not None
            source_layout = str(
                config.mask.overrides.get("source_layout", "frame_first")
            )
            if source_layout == "view_first":
                from common.reuse_inputs import normalize_view_first_masks

                destination = masks_dir(config, self.layout)
                report = normalize_view_first_masks(
                    config.mask.artifact,
                    destination,
                    views=config.views,
                    frame_start=config.frame_start,
                    frame_end=config.frame_end,
                    force=self.force,
                    dry_run=self.dry_run,
                )
                print(
                    f"[reuse] normalized {report['links']} view-first masks "
                    f"-> {destination}",
                    flush=True,
                )
                return destination
            print(f"[reuse] masks {config.mask.artifact}", flush=True)
            return config.mask.artifact
        resolved_path = write_resolved(
            self.layout.resolved_config("mask"),
            mask_config(config, self.layout),
        )
        from common.masking.schema import load_mask_pipeline_config

        load_mask_pipeline_config(resolved_path)
        arguments = [
            "--config",
            str(resolved_path),
            "--stage",
            "all",
            "--qwen-gpu",
            str(config.devices[0]),
            "--sam-gpu",
            str(config.devices[-1]),
        ]
        if self.force:
            arguments.extend(("--force-qwen", "--force-sam"))
        command = [
            "internal:mask",
            *arguments,
        ]
        from scripts.run_mask_pipeline import main as run_mask

        self._run_adapter(command, run_mask, arguments)
        return masks_dir(config, self.layout)

    def _depth(self, starts: dict[str, int]) -> Path:
        config = self.config
        if config.depth.mode == "reuse":
            assert config.depth.artifact is not None
            print(f"[reuse] point clouds {config.depth.artifact}", flush=True)
            return config.depth.artifact
        resolved_path = write_resolved(
            self.layout.resolved_config("depth"),
            depth_config(config, self.layout, starts),
        )
        arguments = [
            "--config",
            str(resolved_path),
            "--stage",
            "all",
        ]
        if self.force:
            arguments.append("--force")
        command = [
            "internal:depth",
            *arguments,
        ]
        from scripts.run_depth_pipeline import main as run_depth

        self._run_adapter(command, run_depth, arguments)
        return point_cloud_root(config, self.layout)

    def _pose(self, starts: dict[str, int]) -> Path:
        config = self.config
        if config.pose.mode == "reuse":
            assert config.pose.artifact is not None
            print(f"[reuse] pose {config.pose.artifact}", flush=True)
            return config.pose.artifact
        resolved_path = self._resolved_pose_config(starts)
        arguments = [
            "--config",
            str(resolved_path),
            "--stage",
            "all",
        ]
        if self.skip_review:
            arguments.append("--skip-review")
        if self.skip_render:
            arguments.append("--skip-render")
        if self.force:
            arguments.append("--force")
        command = [
            "internal:pose",
            *arguments,
        ]
        from scripts.run_pose_pipeline import main as run_pose

        active = self._run_adapter(command, run_pose, arguments)
        if active is not None:
            return Path(active)
        return self.layout.pose_output / "pose" / "trajectory_final.json"

    def _render(self, starts: dict[str, int]) -> Path:
        """Render an existing trajectory through the guarded pose adapter."""

        trajectory = self._trajectory_path()
        if trajectory is None or (not self.dry_run and not trajectory.is_file()):
            raise FileNotFoundError(
                "render requires pose.mode='reuse' with a trajectory artifact, "
                "or an existing pose/pose/trajectory_final.json"
            )
        resolved_path = self._resolved_pose_config(starts)
        arguments = [
            "--config",
            str(resolved_path),
            "--stage",
            "render",
            "--trajectory",
            str(trajectory),
        ]
        from scripts.run_pose_pipeline import main as run_pose

        self._run_adapter(["internal:pose", *arguments], run_pose, arguments)
        return trajectory

    def _resolved_pose_config(self, starts: dict[str, int]) -> Path:
        """Materialize and validate the internal pose-stage configuration."""

        resolved_path = write_resolved(
            self.layout.resolved_config("pose"),
            pose_config(self.config, self.layout, starts),
        )
        from common.pose_config import validate_pose_config

        resolved = load_json(resolved_path)
        validate_pose_config(
            resolved,
            check_paths=not self.dry_run,
            allow_auto=bool(
                resolved.get("automation", {}).get("enabled", False)
            ),
        )
        return resolved_path

    def _trajectory_path(self) -> Path | None:
        if self.config.pose.mode == "reuse":
            return self.config.pose.artifact
        return self.layout.pose_output / "pose" / "trajectory_final.json"

    def _resolved_starts(self) -> dict[str, int]:
        try:
            return resolved_part_starts(self.config, self.layout)
        except RuntimeError:
            if not self.dry_run:
                raise
            starts = {
                part.name: (
                    self.config.frame_start
                    if part.appearance_hint == "auto"
                    else int(part.appearance_hint)
                )
                for part in self.config.parts
            }
            print(
                "[dry-run] automatic appearance hints remain unresolved; "
                "using frame-range start only to materialize the plan",
                flush=True,
            )
            return starts

    def _run_selected_stage(self, stage: str) -> dict[str, str]:
        """Execute one public stage and return its resolved artifact paths."""

        artifacts = {
            "frames": str(
                self._frames()
                if stage in {"all", "frames"}
                else self.config.frames_dir
            )
        }
        if stage == "frames":
            return artifacts

        normalize_reused_masks = bool(
            self.config.mask.mode == "reuse"
            and self.config.mask.overrides.get(
                "source_layout", "frame_first"
            ) == "view_first"
        )
        artifacts["masks"] = str(
            self._mask()
            if stage in {"all", "mask"} or normalize_reused_masks
            else masks_dir(self.config, self.layout)
        )

        starts = self._resolved_starts()
        artifacts["point_clouds"] = str(
            self._depth(starts)
            if stage in {"all", "depth"}
            else point_cloud_root(self.config, self.layout)
        )
        if stage in {"all", "pose"}:
            artifacts["trajectory"] = str(self._pose(starts))
        elif stage == "render":
            artifacts["trajectory"] = str(self._render(starts))
        return artifacts

    def run(self, stage: str = "all") -> Path:
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}")
        self.preflight(selected_stage=stage)
        update_manifest(
            self.layout,
            source=self.config.source_path,
            dataset=self.config.dataset,
            devices=self.config.devices,
            stage=stage,
            status="running",
        )
        artifacts: dict[str, str] = {}
        try:
            if stage == "preflight":
                result = self.layout.manifest
            else:
                artifacts = self._run_selected_stage(stage)
                result = self.layout.manifest
        except Exception as error:
            update_manifest(
                self.layout,
                source=self.config.source_path,
                dataset=self.config.dataset,
                devices=self.config.devices,
                stage=stage,
                status="failed",
                artifacts=artifacts,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        update_manifest(
            self.layout,
            source=self.config.source_path,
            dataset=self.config.dataset,
            devices=self.config.devices,
            stage=stage,
            status="planned" if self.dry_run else "complete",
            artifacts=artifacts,
        )
        outcome = "planned" if self.dry_run else "complete"
        print(f"pipeline {stage} {outcome} -> {result}", flush=True)
        return result


def inspect_result(config: PipelineConfig) -> dict[str, Any]:
    """Return a compact quality/status summary without changing artifacts."""

    layout = ArtifactLayout(config.output_root)
    result: dict[str, Any] = {
        "dataset": config.dataset,
        "output_root": str(config.output_root),
        "manifest": load_json(layout.manifest) if layout.manifest.exists() else None,
    }
    trajectory = (
        config.pose.artifact
        if config.pose.mode == "reuse"
        else layout.pose_output / "pose" / "trajectory_final.json"
    )
    if trajectory is not None and trajectory.is_file():
        payload = load_json(trajectory)
        result["trajectory"] = {
            "path": str(trajectory),
            "parts": payload.get("parts", []),
            "frames": len(payload.get("frames", {})),
            "reference_part": payload.get("reference_part"),
        }
    metrics_path = layout.pose_output / "diagnostics" / "multiview_metrics.json"
    if metrics_path.is_file():
        metrics = load_json(metrics_path)
        result["pose_quality"] = {
            part: {
                key: values.get("all_views", {}).get(key)
                for key in (
                    "visible_observations",
                    "mean_iou",
                    "mean_contour_chamfer_px",
                )
            }
            for part, values in metrics.get("summary", {}).items()
        }
    connector_path = (
        layout.pose_output / "diagnostics" / "connector_readiness.json"
    )
    if connector_path.is_file():
        connectors = load_json(connector_path)
        result["connector_readiness"] = {
            "path": str(connector_path),
            "simulation_ready": connectors.get("simulation_ready", False),
            "failures": connectors.get("failures", {}),
        }
    assembly_path = (
        layout.pose_output / "diagnostics" / "assembly_task_readiness.json"
    )
    if assembly_path.is_file():
        assembly = load_json(assembly_path)
        result["assembly_task_readiness"] = {
            "path": str(assembly_path),
            "pose_product_ready": assembly.get("pose_product_ready", False),
            "physics_replay_ready": assembly.get(
                "physics_replay_ready", False
            ),
            "pose_failures": assembly.get("pose_failures", []),
            "physics_blockers": assembly.get("physics_blockers", []),
        }
    regression = config.raw.get("regression", {})
    baseline_value = regression.get("trajectory")
    if baseline_value:
        baseline = Path(baseline_value)
        if not baseline.is_absolute():
            baseline = (config.source_path.parent / baseline).resolve()
        if not baseline.is_file():
            result["regression"] = {"path": str(baseline), "status": "missing"}
        else:
            payload = load_json(baseline)
            actual = {
                "parts": payload.get("parts", []),
                "frames": len(payload.get("frames", {})),
                "reference_part": payload.get("reference_part"),
                "sha256": source_digest(baseline),
            }
            expected = regression.get("expected", {})
            mismatches = {
                key: {"expected": value, "actual": actual.get(key)}
                for key, value in expected.items()
                if actual.get(key) != value
            }
            result["regression"] = {
                "path": str(baseline),
                "status": "pass" if not mismatches else "fail",
                **actual,
                "mismatches": mismatches,
            }
    return result
