#!/usr/bin/env bash
# Warm-start an Image-B4 continuation control and zero-gated mosaic residual,
# then compare both fixed-epoch models on common episode endpoints.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

BASE="/media/Data/user/WCM_outputs"
EVAL_ROOT="outputs/eval_mosaic_residual_pair"
mkdir -p outputs/batch_logs "$EVAL_ROOT/comparisons"

echo "== 0/4 unit tests and CUDA gradient/memory gate =="
"$PYTHON" -m pytest \
  tests/test_temporal_input.py tests/test_register_config.py \
  tests/test_compare_experiments.py -q
"$PYTHON" -u -m scripts.smoke_test_mosaic_residual \
  --config configs/wcm_mosaic_residual_s3072.yaml --device cuda --batch-size 4

train_run() {
  local config=$1 output=$2 log=$3
  if [[ -f "$output/deploy.pt" && -f "$output/metrics.jsonl" ]] \
    && [[ $(wc -l < "$output/metrics.jsonl") -eq 5 ]]; then
    echo "[skip] completed: $output"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "[abort] partial or incompatible output exists: $output"
    exit 1
  fi
  "$PYTHON" -u -m world_critic.train --config "$config" 2>&1 | tee "$log"
}

echo "== 1/4 train matched five-epoch warm-start pairs =="
for SEED in 3072 42; do
  train_run \
    "configs/wcm_mosaic_continue_s${SEED}.yaml" \
    "$BASE/wcm_mosaic_continue_s${SEED}" \
    "outputs/batch_logs/wcm_mosaic_continue_s${SEED}.log"
  train_run \
    "configs/wcm_mosaic_residual_s${SEED}.yaml" \
    "$BASE/wcm_mosaic_residual_s${SEED}" \
    "outputs/batch_logs/wcm_mosaic_residual_s${SEED}.log"
done

run_eval() {
  local dataset_name=$1 dataset_root=$2 model_name=$3 checkpoint=$4
  local output="$EVAL_ROOT/$dataset_name/$model_name"
  local log="outputs/batch_logs/mosaic_residual_${dataset_name}_${model_name}.log"
  [[ -f "$checkpoint" ]] || { echo "[abort] missing checkpoint: $checkpoint"; exit 1; }
  if [[ -f "$output/summary.json" && -f "$output/episode_curves/episode_curves.csv" ]]; then
    echo "[skip] evaluated: $dataset_name/$model_name"
    return
  fi
  "$PYTHON" -u -m world_critic.evaluate \
    --checkpoint "$checkpoint" --dataset-root "$dataset_root" --split all \
    --output-dir "$output" --batch-size 8 --num-workers 4 \
    --no-episode-curves --episode-metrics --log-every-batches 50 \
    > "$log" 2>&1
}

echo "== 2/4 evaluate original, continuation, and residual models =="
for DATASET_NAME in 5cut ood; do
  if [[ "$DATASET_NAME" == 5cut ]]; then
    DATASET_ROOT="/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return"
  else
    DATASET_ROOT="/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return"
  fi
  for SEED in 3072 42; do
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "image_b4_s$SEED" \
      "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "continue_s$SEED" \
      "$BASE/wcm_mosaic_continue_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "residual_s$SEED" \
      "$BASE/wcm_mosaic_residual_s$SEED/deploy.pt"
  done
done

echo "== 3/4 paired episode bootstrap =="
for DATASET_NAME in 5cut ood; do
  for SEED in 3072 42; do
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/continue_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/residual_s$SEED" \
      --seed "$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_residual_vs_continue_s${SEED}.json"
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/image_b4_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/residual_s$SEED" \
      --seed "$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_residual_vs_image_b4_s${SEED}.json"
  done
done

echo "DONE: inspect $EVAL_ROOT/comparisons/*.json"
echo "Gate: require OOD improvement vs continuation in both seeds without Pearson regression."
