#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-.venv/bin/python}"
BASE="${WCM_CHECKPOINT_ROOT:-/media/Data/user/WCM_outputs}"
EVAL_ROOT="${WCM_T120_EVAL_ROOT:-outputs/eval_spacetime_t120_pair}"
VIDEO_ROOT="${WCM_T120_VIDEO_ROOT:-$EVAL_ROOT/episode_value_videos}"
DATASET_5CUT="${WCM_5CUT_ROOT:-/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return}"
DATASET_OOD="${WCM_OOD_ROOT:-/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return}"
SPEED="${WCM_VIDEO_SPEED:-2.0}"
MAX_EPISODES="${WCM_VIDEO_MAX_EPISODES:-}"
EPISODE_ID="${WCM_VIDEO_EPISODE_ID:-}"
SEEDS=(3072 42)

render_one() {
  local dataset_name=$1 dataset_root=$2 seed=$3
  local curves="$EVAL_ROOT/$dataset_name/t120_s$seed/episode_curves/episode_curves.json"
  local checkpoint="$BASE/wcm_spacetime_vit_t8_direct_full_t120_s$seed/deploy.pt"
  local output="$VIDEO_ROOT/t120_${dataset_name}_s$seed"
  [[ -f "$curves" ]] || { echo "[abort] missing curves: $curves"; exit 1; }
  [[ -f "$checkpoint" ]] || { echo "[abort] missing checkpoint: $checkpoint"; exit 1; }
  local optional=()
  [[ -n "$MAX_EPISODES" ]] && optional+=(--max-episodes "$MAX_EPISODES")
  [[ -n "$EPISODE_ID" ]] && optional+=(--episode-id "$EPISODE_ID")
  "$PYTHON" -m episode_value_video render \
    --curves "$curves" \
    --checkpoint "$checkpoint" \
    --dataset-root "$dataset_root" \
    --output-dir "$output" \
    --speed "$SPEED" \
    --overwrite \
    "${optional[@]}"
}

mkdir -p "$VIDEO_ROOT"
for SEED in "${SEEDS[@]}"; do
  render_one ood "$DATASET_OOD" "$SEED"
  render_one 5cut "$DATASET_5CUT" "$SEED"
done

echo "H120 videos written under: $VIDEO_ROOT"
