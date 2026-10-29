#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

BASE="${WCM_CHECKPOINT_ROOT:-/media/Data/user/WCM_outputs}"
EVAL_ROOT="${WCM_EIGHT_MOSAIC_EVAL_ROOT:-outputs/eval_eight_frame_mosaic_pair}"
DATASET_5CUT="${WCM_5CUT_ROOT:-/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return}"
DATASET_OOD="${WCM_OOD_ROOT:-/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return}"
TRAIN_BATCH="${WCM_PER_DEVICE_BATCH_SIZE:-4}"
EVAL_BATCH="${WCM_EVAL_BATCH_SIZE:-4}"
EVAL_WORKERS="${WCM_EVAL_NUM_WORKERS:-4}"
BOOTSTRAP_SAMPLES="${WCM_BOOTSTRAP_SAMPLES:-20000}"
SEEDS=(3072 42)
H60_OFFSETS="0,9,17,26,34,43,51,60"

mkdir -p outputs/batch_logs "$EVAL_ROOT/comparisons"

train_run() {
  local seed=$1 output=$2 init_from=$3
  if [[ -f "$output/deploy.pt" && -f "$output/metrics.jsonl" ]]; then
    echo "[skip] completed: $output"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "[abort] partial output exists: $output"
    exit 1
  fi
  env WCM_SEED="$seed" WCM_OUTPUT_DIR="$output" WCM_INIT_FROM="$init_from" \
    WCM_HISTORY_OFFSETS="$H60_OFFSETS" WCM_PER_DEVICE_BATCH_SIZE="$TRAIN_BATCH" \
    WCM_EVAL_BATCH_SIZE="$EVAL_BATCH" \
    "$PYTHON" -u -m world_critic.train \
    --config "configs/wcm_sparse4_image_b4_exp_s${seed}.yaml" \
    2>&1 | tee "outputs/batch_logs/$(basename "$output").log"
}

echo "== 0/4 input and config tests =="
"$PYTHON" -m pytest tests/test_temporal_input.py tests/test_register_config.py -q

echo "== 1/4 train 2x4 H60 mosaic controls =="
for SEED in "${SEEDS[@]}"; do
  train_run "$SEED" "$BASE/wcm_mosaic8_image_h60_s$SEED" \
    "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
done

run_eval() {
  local dataset_name=$1 dataset_root=$2 role=$3 checkpoint=$4
  local output="$EVAL_ROOT/$dataset_name/$role"
  [[ -f "$checkpoint" ]] || { echo "[abort] missing checkpoint: $checkpoint"; exit 1; }
  if [[ -f "$output/summary.json" && -f "$output/episode_curves/episode_curves.csv" ]]; then
    echo "[skip] evaluated: $dataset_name/$role"
    return
  fi
  "$PYTHON" -u -m world_critic.evaluate --checkpoint "$checkpoint" \
    --dataset-root "$dataset_root" --split all --output-dir "$output" \
    --batch-size "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
    --no-episode-curves --episode-metrics --log-every-batches 50 \
    > "outputs/batch_logs/eight_mosaic_${dataset_name}_${role}.log" 2>&1
}

echo "== 2/4 evaluate 4-frame mosaic, 8-frame mosaic, and SpaceTime-T8 =="
for DATASET_NAME in 5cut ood; do
  if [[ "$DATASET_NAME" == 5cut ]]; then DATASET_ROOT="$DATASET_5CUT"; else DATASET_ROOT="$DATASET_OOD"; fi
  for SEED in "${SEEDS[@]}"; do
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "mosaic4_s$SEED" \
      "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "mosaic8_s$SEED" \
      "$BASE/wcm_mosaic8_image_h60_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "spacetime_t8_s$SEED" \
      "$BASE/wcm_spacetime_vit_t8_direct_full_s$SEED/deploy.pt"
  done
done

echo "== 3/4 paired bootstrap =="
for DATASET_NAME in 5cut ood; do
  for SEED in "${SEEDS[@]}"; do
    # compare_experiments takes BASELINE first and CANDIDATE second.
    for PAIR in "mosaic8_s$SEED spacetime_t8_s$SEED" "mosaic8_s$SEED mosaic4_s$SEED"; do
      set -- $PAIR
      "$PYTHON" -m scripts.compare_experiments \
        "$EVAL_ROOT/$DATASET_NAME/$1" "$EVAL_ROOT/$DATASET_NAME/$2" \
        --align common --bootstrap-samples "$BOOTSTRAP_SAMPLES" --seed "$SEED" \
        --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_${2}_vs_${1}.json"
    done
  done
done

echo "== 4/4 complete =="
echo "Inspect OOD comparisons under $EVAL_ROOT/comparisons"
