#!/usr/bin/env bash
# Segment + backproject + ICP for one pair. Respects RECON_BACKEND / pipeline.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
GPU="${GPU:-4}"

REF="${1:-000019}"
SRC="${2:-000018}"

if [[ -z "${RECON_BACKEND:-}" ]]; then
  RECON_BACKEND="$("$PY" -c "import json; print(json.load(open('$ROOT/configs/pipeline.json'))['recon_backend'])")"
fi
BACKEND_ARGS=(--recon-backend "$RECON_BACKEND")

echo "==> segment + backproject [$RECON_BACKEND]: $REF $SRC"
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/seg_backproject_parts.py" \
  --timestamps "$REF" "$SRC" --gpu "$GPU" "${BACKEND_ARGS[@]}"

echo "==> ICP [$RECON_BACKEND]: $SRC -> $REF"
"$PY" "$ROOT/scripts/icp_pose.py" --ref "$REF" --src "$SRC" "${BACKEND_ARGS[@]}"

echo "done. outputs/icp/$RECON_BACKEND/"
