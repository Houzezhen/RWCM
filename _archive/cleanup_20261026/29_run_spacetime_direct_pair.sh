#!/usr/bin/env bash
# Decisive teacher-free t8 ablation: Image-B4 -> direct joint -> full.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1
BASE="/media/Data/user/WCM_outputs"
TRAIN_BATCH="${WCM_PER_DEVICE_BATCH_SIZE:-4}"
EVAL_BATCH="${WCM_EVAL_BATCH_SIZE:-4}"
EVAL_ROOT="outputs/eval_spacetime_vit_t8_direct_pair"
mkdir -p outputs/batch_logs "$EVAL_ROOT/comparisons"

echo "== 0/4 tests and CUDA gradient smoke =="
"$PYTHON" -m pytest tests/test_temporal_input.py tests/test_register_config.py \
  tests/test_spacetime_training.py tests/test_ranking_loss.py \
  tests/test_compare_experiments.py tests/test_spacetime_gates.py -q
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
    WCM_PER_DEVICE_BATCH_SIZE="$TRAIN_BATCH" WCM_EVAL_BATCH_SIZE="$EVAL_BATCH" \
    "$PYTHON" -u -m world_critic.train --config "$config" \
    2>&1 | tee "outputs/batch_logs/$(basename "$output").log"
}

echo "== 1/4 train teacher-free direct SpaceTime pair =="
for SEED in 3072 42; do
  train_run configs/wcm_spacetime_direct.yaml "$SEED" \
    "$BASE/wcm_spacetime_vit_t8_direct_s$SEED" \
    "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
  train_run configs/wcm_spacetime_direct_full.yaml "$SEED" \
    "$BASE/wcm_spacetime_vit_t8_direct_full_s$SEED" \
    "$BASE/wcm_spacetime_vit_t8_direct_s$SEED/deploy.pt"
done
"$PYTHON" -m scripts.check_spacetime_gates --checkpoint \
  "$BASE/wcm_spacetime_vit_t8_direct_full_s3072/deploy.pt" \
  "$BASE/wcm_spacetime_vit_t8_direct_full_s42/deploy.pt"

run_eval() {
  local dataset_name=$1 dataset_root=$2 model_name=$3 checkpoint=$4
  local output="$EVAL_ROOT/$dataset_name/$model_name"
  [[ -f "$checkpoint" ]] || { echo "[abort] missing checkpoint: $checkpoint"; exit 1; }
  if [[ -f "$output/summary.json" && -f "$output/episode_curves/episode_curves.csv" ]]; then
    echo "[skip] evaluated: $dataset_name/$model_name"
    return
  fi
  "$PYTHON" -u -m world_critic.evaluate \
    --checkpoint "$checkpoint" --dataset-root "$dataset_root" --split all \
    --output-dir "$output" --batch-size "$EVAL_BATCH" --num-workers 4 \
    --no-episode-curves --episode-metrics --log-every-batches 50 \
    > "outputs/batch_logs/spacetime_direct_${dataset_name}_${model_name}.log" 2>&1
}

echo "== 2/4 evaluate mosaic, staged, and direct SpaceTime models =="
for DATASET_NAME in 5cut ood; do
  if [[ "$DATASET_NAME" == 5cut ]]; then
    DATASET_ROOT="/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return"
  else
    DATASET_ROOT="/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return"
  fi
  for SEED in 3072 42; do
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "mosaic_s$SEED" \
      "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "staged_s$SEED" \
      "$BASE/wcm_spacetime_vit_t8_full_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "direct_s$SEED" \
      "$BASE/wcm_spacetime_vit_t8_direct_full_s$SEED/deploy.pt"
  done
done

echo "== 3/4 paired bootstrap and final OOD gates =="
for DATASET_NAME in 5cut ood; do
  for SEED in 3072 42; do
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/mosaic_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/direct_s$SEED" --seed "$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_direct_vs_mosaic_s${SEED}.json"
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/staged_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/direct_s$SEED" --seed "$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_direct_vs_staged_s${SEED}.json"
  done
done
"$PYTHON" -m scripts.check_spacetime_gates --ood \
  "$EVAL_ROOT/comparisons/ood_direct_vs_mosaic_s3072.json" \
  "$EVAL_ROOT/comparisons/ood_direct_vs_mosaic_s42.json"

echo "DONE: teacher-free direct SpaceTime t8 passed all configured gates."
