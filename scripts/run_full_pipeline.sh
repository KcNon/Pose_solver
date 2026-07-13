#!/usr/bin/env bash
# Full normalized pipeline: backproject -> chain ICP (000000..110) -> proj viz -> MP4.
#
#   ./scripts/run_full_pipeline.sh
#   BACKEND=da3_self_cond ./scripts/run_full_pipeline.sh
#   SKIP_BACKPROJECT=1 ./scripts/run_full_pipeline.sh   # resume from ICP
#
# Log: /tmp/pose_solver_full_pipeline.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
CFG="${CFG:-$ROOT/configs/pipeline_normalized.json}"
BACKEND="${BACKEND:-da3_self_cond}"
LOG="${LOG:-/tmp/pose_solver_full_pipeline.log}"

export PYTHONUNBUFFERED=1

echo "==> [1/4] backproject ALL -> parts_ply/$BACKEND"
if [[ "${SKIP_BACKPROJECT:-0}" != "1" ]]; then
  "$PY" "$ROOT/scripts/backproject_normalized.py" \
    --pipeline "$CFG" --backend "$BACKEND" --all --skip-existing
else
  echo "    skipped (SKIP_BACKPROJECT=1)"
fi

echo "==> [2/4] chain ICP ALL (000000..110)"
"$PY" "$ROOT/scripts/icp_chain.py" \
  --pipeline "$CFG" --backend "$BACKEND" --all --skip-existing

echo "==> [3/4] ICP projection visualization"
"$PY" "$ROOT/scripts/visualize_icp_chain.py" \
  --pipeline "$CFG" --backend "$BACKEND" --from-summary --skip-existing

echo "==> [4/4] pack montages -> MP4"
"$PY" "$ROOT/scripts/make_icp_video.py" \
  --pipeline "$CFG" --backend "$BACKEND" --use-ffmpeg

echo ""
echo "done."
echo "  parts_ply  $ROOT/outputs/normalized/parts_ply/$BACKEND/"
echo "  icp        $ROOT/outputs/normalized/icp/$BACKEND/"
echo "  proj_vis   $ROOT/outputs/normalized/proj_vis/$BACKEND/"
echo "  video      $ROOT/outputs/normalized/videos/${BACKEND}_icp_chain.mp4"
