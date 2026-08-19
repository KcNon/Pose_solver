"""Stable output layout and audit manifest for the unified pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from common.io_utils import load_json, write_json


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def mask_work(self) -> Path:
        return self.root / "mask_work"

    @property
    def mask_output(self) -> Path:
        return self.root / "mask"

    @property
    def depth_output(self) -> Path:
        return self.root / "depth"

    @property
    def pose_output(self) -> Path:
        return self.root / "pose"

    @property
    def manifest(self) -> Path:
        return self.runtime / "pipeline_manifest.json"

    def resolved_config(self, stage: str) -> Path:
        return self.runtime / f"{stage}.resolved.json"


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def update_manifest(
    layout: ArtifactLayout,
    *,
    source: Path,
    dataset: str,
    devices: tuple[int, ...],
    stage: str,
    status: str,
    artifacts: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    current = load_json(layout.manifest) if layout.manifest.exists() else {}
    stages = dict(current.get("stages", {}))
    stages[stage] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **({"error": error} if error else {}),
    }
    value = {
        "schema_version": 1,
        "dataset": dataset,
        "source_config": str(source),
        "source_sha256": source_digest(source),
        "devices": list(devices),
        "stages": stages,
        "artifacts": {**current.get("artifacts", {}), **(artifacts or {})},
    }
    write_json(layout.manifest, value)
