#!/usr/bin/env bash
# Normalized data pipeline: Qwen bbox -> SAM palette masks (no depth / ICP).
#
# Single timestamp:
#   ./scripts/run_timestamp.sh 000000
# All timestamps (111 frames):
#   ./scripts/run_timestamp.sh --all
# Reset bbox.json and re-run everything:
#   FRESH=1 ./scripts/run_timestamp.sh --all
#
# Note: --all runs ALL qwen first, then ALL sam. Mask PNG folders appear after
# step 1 completes (~1.5h), not during qwen. Use single-frame mode to preview.
#
# Env: QGPU (default 5), SGPU (default 4), VIS=1 (bbox overlay pngs)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
QWEN_PY="/data_ft_9_10/wentai/projects/qwen3-vl/.venv/bin/python"
SAM_PY="/data_ft_9_10/wentai/projects/sam3/.venv/bin/python"
PIPELINE="${PIPELINE:-$ROOT/configs/pipeline_normalized.json}"
QGPU="${QGPU:-5}"
SGPU="${SGPU:-4}"
VIS="${VIS:-0}"
FRESH="${FRESH:-0}"

QWEN_EXTRA=()
[[ "$VIS" == "1" ]] && QWEN_EXTRA+=(--vis)
[[ "$FRESH" == "1" ]] && QWEN_EXTRA+=(--fresh)

MASKS_DIR="$("$PY" -c "import json; print(json.load(open('$PIPELINE'))['masks_dir'])")"

run_one() {
  local TS="$1"
  echo "==> [1/2] Qwen bbox: $TS (GPU $QGPU)"
  CUDA_VISIBLE_DEVICES=$QGPU "$QWEN_PY" "$ROOT/scripts/detect_bbox_batch.py" \
    --pipeline "$PIPELINE" --timestamp "$TS" "${QWEN_EXTRA[@]}"

  echo "==> [2/2] SAM segment: $TS (GPU $SGPU)"
  CUDA_VISIBLE_DEVICES=$SGPU "$SAM_PY" "$ROOT/scripts/seg_masks_only.py" \
    --pipeline "$PIPELINE" --timestamp "$TS" --gpu "$SGPU"
}

if [[ "${1:-}" == "--all" ]]; then
  echo "==> [1/2] Qwen bbox: ALL (GPU $QGPU)"
  CUDA_VISIBLE_DEVICES=$QGPU "$QWEN_PY" "$ROOT/scripts/detect_bbox_batch.py" \
    --pipeline "$PIPELINE" --all "${QWEN_EXTRA[@]}"

  echo "==> [2/2] SAM segment: ALL (GPU $SGPU)"
  CUDA_VISIBLE_DEVICES=$SGPU "$SAM_PY" "$ROOT/scripts/seg_masks_only.py" \
    --pipeline "$PIPELINE" --all --gpu "$SGPU"
else
  TS="${1:?timestamp required, e.g. 000000  (or pass --all for every frame)}"
  run_one "$TS"
fi

echo "done. outputs:"
echo "  bbox.json  $MASKS_DIR/bbox.json"
echo "  masks      $MASKS_DIR/<timestamp>/{2-1..2-6}.png  (palette indexed)"
