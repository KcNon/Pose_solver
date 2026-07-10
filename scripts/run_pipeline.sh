#!/usr/bin/env bash
# Normalized data end-to-end pipeline (2-normalized inputs -> pose_solver/outputs).
#
# Usage:
#   ./scripts/run_pipeline.sh masks [--all]           # Qwen bbox + SAM masks
#   ./scripts/run_pipeline.sh backproject [--sample]  # mask + depth -> parts_ply
#   ./scripts/run_pipeline.sh viz-raw [--sample]        # raw point cloud projection
#   ./scripts/run_pipeline.sh icp [--sample]          # chain ICP + ICP projection viz
#   ./scripts/run_pipeline.sh all [--sample]            # backproject + icp (masks must exist)
#
# Env: PIPELINE, BACKEND, QGPU=5, SGPU=4, VIS=1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
QWEN_PY="/data_ft_9_10/wentai/projects/qwen3-vl/.venv/bin/python"
SAM_PY="/data_ft_9_10/wentai/projects/sam3/.venv/bin/python"
PIPELINE="${PIPELINE:-$ROOT/configs/pipeline_normalized.json}"
BACKEND="${BACKEND:-da3_self_cond}"

SAMPLE_FLAG=()
[[ "${2:-}" == "--sample" || "${3:-}" == "--sample" ]] && SAMPLE_FLAG=(--sample)

cmd="${1:-help}"

run_masks() {
  if [[ "${2:-}" == "--all" ]]; then
    "$ROOT/scripts/run_timestamp.sh" --all
  else
    TS="${2:?timestamp required, e.g. 000000}"
    "$ROOT/scripts/run_timestamp.sh" "$TS"
  fi
}

run_backproject() {
  "$PY" "$ROOT/scripts/backproject_normalized.py" \
    --pipeline "$PIPELINE" --backend "$BACKEND" "${SAMPLE_FLAG[@]}"
}

run_viz_raw() {
  "$PY" "$ROOT/scripts/visualize_raw_ply.py" \
    --pipeline "$PIPELINE" --backend "$BACKEND" "${SAMPLE_FLAG[@]}"
}

run_icp() {
  "$PY" "$ROOT/scripts/icp_chain.py" --pipeline "$PIPELINE" --backend "$BACKEND"
  "$PY" "$ROOT/scripts/visualize_icp_chain.py" \
    --pipeline "$PIPELINE" --backend "$BACKEND" --from-summary
}

case "$cmd" in
  masks)       run_masks "${2:-}" ;;
  backproject) run_backproject ;;
  viz-raw)     run_viz_raw ;;
  icp)         run_icp ;;
  all)
    run_backproject
    run_viz_raw
    run_icp
    ;;
  help|*)
    sed -n '3,12p' "$0"
    exit 0
    ;;
esac

OUT="$("$PY" -c "import json; print(json.load(open('$PIPELINE'))['output_root'])")"
echo "done. outputs under $OUT/"
