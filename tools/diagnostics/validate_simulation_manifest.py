#!/usr/bin/env python3
"""Preflight an exported assembly case before launching a physics backend."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json


def build_preflight_report(
    manifest: dict[str, Any],
    geometry: dict[str, Any],
    *,
    manifest_path: Path,
    connector_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = dict(geometry.get("validation_readiness", {}))
    failures = list(map(str, readiness.get("failures", [])))
    interface = manifest.get("assembly_interface")
    if interface is None and "assembly_interface_missing" not in failures:
        failures.append("assembly_interface_missing")
    connector_summary = None
    if connector_evidence is not None:
        connectors = list(connector_evidence.get("connectors", {}).values())
        connector_summary = {
            "connector_count": len(connectors),
            "all_simulation_ready": bool(
                connectors and all(row.get("simulation_ready", False) for row in connectors)
            ),
            "failures": sorted(
                {
                    str(reason)
                    for row in connectors
                    for reason in row.get("failures", [])
                }
            ),
        }
        if connectors and not connector_summary["all_simulation_ready"]:
            failures.append("external_connector_evidence_not_simulation_ready")
    required_outputs = [
        manifest_path.parent / manifest["outputs"][key]
        for key in ("geometry_report", "observed_replay")
    ]
    for part, relative in manifest["outputs"]["independent_urdfs"].items():
        path = manifest_path.parent / relative
        required_outputs.append(path)
        if not path.is_file():
            failures.append(f"missing_urdf:{part}")
    for path in required_outputs:
        if not path.is_file():
            failures.append(f"missing_output:{path.name}")
    return {
        "schema_version": 1,
        "status": "passed" if not failures else "blocked",
        "physics_launch_allowed": not failures,
        "metric_physical_accuracy_claim_allowed": bool(
            readiness.get("metric_physical_accuracy_claim_allowed", False)
        ),
        "manifest": str(manifest_path.resolve()),
        "assembly_interface": interface,
        "readiness": readiness,
        "external_connector_evidence": connector_summary,
        "failures": sorted(set(failures)),
        "claim_scope": (
            "Passing preflight permits a frozen-pose physics feasibility test; "
            "it does not establish pose ground truth."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--connector-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    geometry_path = manifest_path.parent / manifest["outputs"]["geometry_report"]
    report = build_preflight_report(
        manifest,
        load_json(geometry_path),
        manifest_path=manifest_path,
        connector_evidence=(
            load_json(args.connector_evidence) if args.connector_evidence else None
        ),
    )
    write_json(args.output, report)
    print(
        f"assembly validation preflight: {report['status']} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
