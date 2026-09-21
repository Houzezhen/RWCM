#!/usr/bin/env bash
# SpaceTime temporal-span ablation: 8 frames over 120 steps vs the existing 60-step model.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

BASE="${WCM_CHECKPOINT_ROOT:-/media/Data/user/WCM_outputs}"
EVAL_ROOT="${WCM_T120_EVAL_ROOT:-outputs/eval_spacetime_t120_pair}"
DATASET_5CUT="${WCM_5CUT_ROOT:-/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return}"
DATASET_OOD="${WCM_OOD_ROOT:-/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return}"
TRAIN_BATCH="${WCM_PER_DEVICE_BATCH_SIZE:-2}"
EVAL_BATCH="${WCM_EVAL_BATCH_SIZE:-4}"
EVAL_WORKERS="${WCM_EVAL_NUM_WORKERS:-4}"
BOOTSTRAP_SAMPLES="${WCM_BOOTSTRAP_SAMPLES:-20000}"
T120_OFFSETS="0,17,34,51,69,86,103,120"
T120_OFFSET_ARGS=(0 17 34 51 69 86 103 120)
SEEDS=(3072 42)

mkdir -p outputs/batch_logs "$EVAL_ROOT/comparisons"

echo "== 0/4 tests and CUDA gradient smoke =="
"$PYTHON" -m pytest tests/test_register_config.py tests/test_temporal_input.py \
  tests/test_spacetime_training.py tests/test_compare_experiments.py -q
env WCM_HISTORY_OFFSETS="$T120_OFFSETS" \
  "$PYTHON" -u -m scripts.smoke_test_spacetime_perceiver \
  --config configs/wcm_spacetime_direct.yaml --device cuda

train_run() {
  local config=$1 seed=$2 output=$3 init_from=$4
  if [[ -f "$output/deploy.pt" && -f "$output/metrics.jsonl" ]]; then
    echo "[skip] completed: $output"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "[abort] partial or incompatible output exists: $output"
    exit 1
  fi
  [[ -f "$init_from" ]] || { echo "[abort] missing checkpoint: $init_from"; exit 1; }
  env WCM_SEED="$seed" WCM_OUTPUT_DIR="$output" WCM_INIT_FROM="$init_from" \
    WCM_HISTORY_OFFSETS="$T120_OFFSETS" \
    WCM_PER_DEVICE_BATCH_SIZE="$TRAIN_BATCH" WCM_EVAL_BATCH_SIZE="$EVAL_BATCH" \
    "$PYTHON" -u -m world_critic.train --config "$config" \
    2>&1 | tee "outputs/batch_logs/$(basename "$output").log"
}

echo "== 1/4 train teacher-free SpaceTime T120 pair =="
for SEED in "${SEEDS[@]}"; do
  train_run configs/wcm_spacetime_direct.yaml "$SEED" \
    "$BASE/wcm_spacetime_vit_t8_direct_t120_s$SEED" \
    "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
  train_run configs/wcm_spacetime_direct_full.yaml "$SEED" \
    "$BASE/wcm_spacetime_vit_t8_direct_full_t120_s$SEED" \
    "$BASE/wcm_spacetime_vit_t8_direct_t120_s$SEED/deploy.pt"
done
"$PYTHON" -m scripts.check_spacetime_gates --checkpoint \
  "$BASE/wcm_spacetime_vit_t8_direct_full_t120_s3072/deploy.pt" \
  "$BASE/wcm_spacetime_vit_t8_direct_full_t120_s42/deploy.pt" \
  --history-offsets "${T120_OFFSET_ARGS[@]}"

run_eval() {
  local dataset_name=$1 dataset_root=$2 model_name=$3 checkpoint=$4
  local output="$EVAL_ROOT/$dataset_name/$model_name"
  local log="outputs/batch_logs/spacetime_t120_${dataset_name}_${model_name}.log"
  [[ -f "$checkpoint" ]] || { echo "[abort] missing checkpoint: $checkpoint"; exit 1; }
  if [[ -f "$output/summary.json" && -f "$output/episode_curves/episode_curves.csv" ]] \
    && grep -q '"evaluation_schema_version": 2' "$output/summary.json"; then
    echo "[skip] evaluated: $dataset_name/$model_name"
    return
  fi
  "$PYTHON" -u -m world_critic.evaluate \
    --checkpoint "$checkpoint" --dataset-root "$dataset_root" --split all \
    --output-dir "$output" --batch-size "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
    --no-episode-curves --episode-metrics --log-every-batches 50 \
    > "$log" 2>&1
}

echo "== 2/4 evaluate T60, T120, and mosaic controls =="
for DATASET_NAME in 5cut ood; do
  if [[ "$DATASET_NAME" == 5cut ]]; then
    DATASET_ROOT="$DATASET_5CUT"
  else
    DATASET_ROOT="$DATASET_OOD"
  fi
  for SEED in "${SEEDS[@]}"; do
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "mosaic_s$SEED" \
      "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "t60_s$SEED" \
      "$BASE/wcm_spacetime_vit_t8_direct_full_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "t120_s$SEED" \
      "$BASE/wcm_spacetime_vit_t8_direct_full_t120_s$SEED/deploy.pt"
  done
done

echo "== 3/4 paired endpoint bootstrap =="
for DATASET_NAME in 5cut ood; do
  for SEED in "${SEEDS[@]}"; do
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/t60_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/t120_s$SEED" \
      --align common --bootstrap-samples "$BOOTSTRAP_SAMPLES" --seed "$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_t120_vs_t60_s${SEED}.json"
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/mosaic_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/t120_s$SEED" \
      --align common --bootstrap-samples "$BOOTSTRAP_SAMPLES" --seed "$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_t120_vs_mosaic_s${SEED}.json"
  done
done

echo "== 4/4 complete =="
echo "Primary result: $EVAL_ROOT/comparisons/ood_t120_vs_t60_s{3072,42}.json"
echo "Control result: $EVAL_ROOT/comparisons/ood_t120_vs_mosaic_s{3072,42}.json"
