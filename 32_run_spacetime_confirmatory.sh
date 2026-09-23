#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

BASE="${WCM_CHECKPOINT_ROOT:-/media/Data/user/WCM_outputs}"
EVAL_ROOT="${WCM_CONFIRMATORY_EVAL_ROOT:-outputs/eval_spacetime_confirmatory_s1337}"
DATASET_5CUT="${WCM_5CUT_ROOT:-/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return}"
DATASET_OOD="${WCM_OOD_ROOT:-/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return}"
SEED=1337
TRAIN_BATCH="${WCM_PER_DEVICE_BATCH_SIZE:-4}"
EVAL_BATCH="${WCM_EVAL_BATCH_SIZE:-4}"
EVAL_WORKERS="${WCM_EVAL_NUM_WORKERS:-4}"
BOOTSTRAP_SAMPLES="${WCM_BOOTSTRAP_SAMPLES:-20000}"
H60_OFFSETS="0,9,17,26,34,43,51,60"
H120_OFFSETS="0,17,34,51,69,86,103,120"

mkdir -p outputs/batch_logs "$EVAL_ROOT/comparisons"

train_run() {
  local config=$1 output=$2 init_from=${3:-} offsets=${4:-}
  if [[ -f "$output/deploy.pt" && -f "$output/metrics.jsonl" ]]; then
    echo "[skip] completed: $output"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "[abort] partial output exists: $output"
    exit 1
  fi
  local init_args=()
  [[ -n "$init_from" ]] && init_args+=(WCM_INIT_FROM="$init_from")
  local offset_args=()
  [[ -n "$offsets" ]] && offset_args+=(WCM_HISTORY_OFFSETS="$offsets")
  env WCM_SEED="$SEED" WCM_OUTPUT_DIR="$output" \
    WCM_PER_DEVICE_BATCH_SIZE="$TRAIN_BATCH" WCM_EVAL_BATCH_SIZE="$EVAL_BATCH" \
    "${init_args[@]}" "${offset_args[@]}" \
    "$PYTHON" -u -m world_critic.train --config "$config" \
    2>&1 | tee "outputs/batch_logs/$(basename "$output").log"
}

echo "== 0/5 tests =="
"$PYTHON" -m pytest tests/test_temporal_input.py tests/test_register_config.py \
  tests/test_spacetime_training.py tests/test_compare_experiments.py \
  tests/test_spacetime_gates.py -q

echo "== 1/5 train seed1337 original WCM and Image-B4 =="
train_run configs/wcm_baseline_exp_s3072.yaml "$BASE/wcm_baseline_exp_s1337"
train_run configs/wcm_sparse4_image_b4_exp_s3072.yaml "$BASE/wcm_sparse4_image_b4_exp_s1337"

echo "== 2/5 train seed1337 T8-H60 and T8-H120 =="
train_run configs/wcm_spacetime_direct.yaml \
  "$BASE/wcm_spacetime_vit_t8_direct_s1337" \
  "$BASE/wcm_sparse4_image_b4_exp_s1337/deploy.pt" "$H60_OFFSETS"
train_run configs/wcm_spacetime_direct_full.yaml \
  "$BASE/wcm_spacetime_vit_t8_direct_full_s1337" \
  "$BASE/wcm_spacetime_vit_t8_direct_s1337/deploy.pt" "$H60_OFFSETS"
train_run configs/wcm_spacetime_direct.yaml \
  "$BASE/wcm_spacetime_vit_t8_direct_t120_s1337" \
  "$BASE/wcm_sparse4_image_b4_exp_s1337/deploy.pt" "$H120_OFFSETS"
train_run configs/wcm_spacetime_direct_full.yaml \
  "$BASE/wcm_spacetime_vit_t8_direct_full_t120_s1337" \
  "$BASE/wcm_spacetime_vit_t8_direct_t120_s1337/deploy.pt" "$H120_OFFSETS"

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
    > "outputs/batch_logs/confirmatory_${dataset_name}_${role}.log" 2>&1
}

echo "== 3/5 evaluate seed1337 =="
for DATASET_NAME in 5cut ood; do
  if [[ "$DATASET_NAME" == 5cut ]]; then DATASET_ROOT="$DATASET_5CUT"; else DATASET_ROOT="$DATASET_OOD"; fi
  run_eval "$DATASET_NAME" "$DATASET_ROOT" baseline "$BASE/wcm_baseline_exp_s1337/deploy.pt"
  run_eval "$DATASET_NAME" "$DATASET_ROOT" mosaic "$BASE/wcm_sparse4_image_b4_exp_s1337/deploy.pt"
  run_eval "$DATASET_NAME" "$DATASET_ROOT" h60 "$BASE/wcm_spacetime_vit_t8_direct_full_s1337/deploy.pt"
  run_eval "$DATASET_NAME" "$DATASET_ROOT" h120 "$BASE/wcm_spacetime_vit_t8_direct_full_t120_s1337/deploy.pt"
done

echo "== 4/5 paired bootstrap =="
for DATASET_NAME in 5cut ood; do
  # compare_experiments takes BASELINE first and CANDIDATE second.
  for PAIR in "baseline h60" "baseline h120" "h60 h120" "mosaic h120"; do
    set -- $PAIR
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/$1" "$EVAL_ROOT/$DATASET_NAME/$2" \
      --align common --bootstrap-samples "$BOOTSTRAP_SAMPLES" --seed "$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_${2}_vs_${1}_s${SEED}.json"
  done
done

echo "== 5/5 complete =="
echo "Inspect: $EVAL_ROOT/comparisons/ood_*.json"
