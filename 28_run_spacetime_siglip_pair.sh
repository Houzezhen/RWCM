#!/usr/bin/env bash
# Reuse the trained Image-B4 teacher and train SpaceTime SigLIP Perceiver-64.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1
BASE="/media/Data/user/WCM_outputs"
EVAL_ROOT="outputs/eval_spacetime_siglip_pair"
mkdir -p outputs/batch_logs "$EVAL_ROOT/comparisons"

echo "== 0/5 tests and teacher-free CUDA smoke =="
"$PYTHON" -m pytest tests/test_temporal_input.py tests/test_register_config.py \
  tests/test_spacetime_training.py tests/test_ranking_loss.py \
  tests/test_compare_experiments.py -q
"$PYTHON" -u -m scripts.smoke_test_spacetime_perceiver \
  --config configs/wcm_spacetime_joint.yaml --device cuda

train_run() {
  local config=$1 seed=$2 output=$3 init_from=${4:-} teacher=${5:-}
  if [[ -f "$output/deploy.pt" && -f "$output/metrics.jsonl" ]]; then
    echo "[skip] completed: $output"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "[abort] partial or incompatible output exists: $output"
    exit 1
  fi
  if [[ -n "$teacher" ]]; then
    [[ -f "$teacher" ]] || { echo "[abort] missing teacher checkpoint: $teacher"; exit 1; }
  fi
  if [[ -n "$init_from" ]]; then
    [[ -f "$init_from" ]] || { echo "[abort] missing init checkpoint: $init_from"; exit 1; }
    env WCM_SEED="$seed" WCM_OUTPUT_DIR="$output" WCM_INIT_FROM="$init_from" \
      WCM_TEACHER_CHECKPOINT="$teacher" \
      "$PYTHON" -u -m world_critic.train --config "$config" \
      2>&1 | tee "outputs/batch_logs/$(basename "$output").log"
  else
    env WCM_SEED="$seed" WCM_OUTPUT_DIR="$output" \
      "$PYTHON" -u -m world_critic.train --config "$config" \
      2>&1 | tee "outputs/batch_logs/$(basename "$output").log"
  fi
}

echo "== 1/5 train single-frame SigLIP control; reuse existing Image-B4 mosaic =="
for SEED in 3072 42; do
  train_run configs/wcm_siglip_single.yaml "$SEED" "$BASE/wcm_siglip_single_s$SEED"
done

echo "== 2/5 run four-stage SpaceTime Perceiver training =="
for SEED in 3072 42; do
  train_run configs/wcm_spacetime_align.yaml "$SEED" "$BASE/wcm_spacetime_align_s$SEED" \
    "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt" \
    "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
  "$PYTHON" -m scripts.check_spacetime_gates --alignment \
    "$BASE/wcm_spacetime_align_s$SEED/metrics.jsonl"
  train_run configs/wcm_spacetime_gate.yaml "$SEED" "$BASE/wcm_spacetime_gate_s$SEED" \
    "$BASE/wcm_spacetime_align_s$SEED/deploy.pt" \
    "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
  train_run configs/wcm_spacetime_joint.yaml "$SEED" "$BASE/wcm_spacetime_joint_s$SEED" \
    "$BASE/wcm_spacetime_gate_s$SEED/deploy.pt"
  train_run configs/wcm_spacetime_full.yaml "$SEED" "$BASE/wcm_spacetime_full_s$SEED" \
    "$BASE/wcm_spacetime_joint_s$SEED/deploy.pt"
done
"$PYTHON" -m scripts.check_spacetime_gates --checkpoint \
  "$BASE/wcm_spacetime_full_s3072/deploy.pt" \
  "$BASE/wcm_spacetime_full_s42/deploy.pt"

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
    --output-dir "$output" --batch-size 4 --num-workers 4 \
    --no-episode-curves --episode-metrics --log-every-batches 50 \
    > "outputs/batch_logs/spacetime_${dataset_name}_${model_name}.log" 2>&1
}

echo "== 3/5 evaluate single, mosaic, and SpaceTime models =="
for DATASET_NAME in 5cut ood; do
  if [[ "$DATASET_NAME" == 5cut ]]; then
    DATASET_ROOT="/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return"
  else
    DATASET_ROOT="/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return"
  fi
  for SEED in 3072 42; do
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "single_s$SEED" \
      "$BASE/wcm_siglip_single_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "mosaic_s$SEED" \
      "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "spacetime_s$SEED" \
      "$BASE/wcm_spacetime_full_s$SEED/deploy.pt"
  done
done

echo "== 4/5 paired bootstrap and final OOD gates =="
for DATASET_NAME in 5cut ood; do
  for SEED in 3072 42; do
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/mosaic_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/spacetime_s$SEED" --seed "$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_spacetime_vs_mosaic_s${SEED}.json"
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/single_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/spacetime_s$SEED" --seed "$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_spacetime_vs_single_s${SEED}.json"
  done
done
"$PYTHON" -m scripts.check_spacetime_gates --ood \
  "$EVAL_ROOT/comparisons/ood_spacetime_vs_mosaic_s3072.json" \
  "$EVAL_ROOT/comparisons/ood_spacetime_vs_mosaic_s42.json"

echo "DONE: SpaceTimeSigLIP + Perceiver-64 passed all configured gates."
