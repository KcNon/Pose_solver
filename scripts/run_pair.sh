#!/usr/bin/env bash
# End-to-end: segment + backproject + ICP for one pair of timestamps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
GPU="${GPU:-4}"

REF="${1:-000019}"
SRC="${2:-000018}"

echo "==> segment + backproject: $REF $SRC"
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/seg_backproject_parts.py" \
  --timestamps "$REF" "$SRC" --gpu "$GPU"

echo "==> ICP: $SRC -> $REF"
"$PY" "$ROOT/scripts/icp_pose.py" --ref "$REF" --src "$SRC"

echo "done. outputs in $ROOT/outputs/icp/"
