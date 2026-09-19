#!/usr/bin/env bash
# Unified preregistered system-level evaluation: WCM vs Image-B4 vs SpaceTime-T8.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

BASE="${WCM_CHECKPOINT_ROOT:-/media/Data/user/WCM_outputs}"
EVAL_ROOT="${WCM_THREEWAY_EVAL_ROOT:-outputs/eval_spacetime_threeway}"
DATASET_5CUT="${WCM_5CUT_ROOT:-/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return}"
DATASET_OOD="${WCM_OOD_ROOT:-/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return}"
EVAL_BATCH="${WCM_EVAL_BATCH_SIZE:-4}"
EVAL_WORKERS="${WCM_EVAL_NUM_WORKERS:-4}"
BOOTSTRAP_SAMPLES="${WCM_BOOTSTRAP_SAMPLES:-20000}"

mkdir -p outputs/batch_logs "$EVAL_ROOT/comparisons" "$EVAL_ROOT/endpoint_manifests"

echo "== 0/4 verify analysis code and checkpoints =="
"$PYTHON" -m pytest tests/test_compare_experiments.py tests/test_spacetime_threeway.py \
  tests/test_spacetime_gates.py -q

"$PYTHON" -m scripts.check_spacetime_threeway_inputs \
  --checkpoint-root "$BASE" --seeds 3072 42 \
  --output "$EVAL_ROOT/checkpoint_inventory.json"

run_eval() {
  local dataset_name=$1 dataset_root=$2 role=$3 seed=$4 checkpoint=$5
  local output="$EVAL_ROOT/$dataset_name/${role}_s${seed}"
  local log="outputs/batch_logs/spacetime_threeway_${dataset_name}_${role}_s${seed}.log"
  if [[ -f "$output/summary.json" && -f "$output/episode_curves/episode_curves.csv" ]] \
    && grep -q '"evaluation_schema_version": 2' "$output/summary.json"; then
    echo "[skip] evaluated: $dataset_name/${role}_s${seed}"
    return
  fi
  "$PYTHON" -u -m world_critic.evaluate \
    --checkpoint "$checkpoint" --dataset-root "$dataset_root" --split all \
    --output-dir "$output" --batch-size "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
    --no-episode-curves --episode-metrics --log-every-batches 50 \
    > "$log" 2>&1
}

echo "== 1/4 evaluate 5cut on all four roles =="
for SEED in 3072 42; do
  run_eval 5cut "$DATASET_5CUT" baseline "$SEED" \
    "$BASE/wcm_baseline_exp_s$SEED/deploy.pt"
  run_eval 5cut "$DATASET_5CUT" mosaic "$SEED" \
    "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
  run_eval 5cut "$DATASET_5CUT" spacetime "$SEED" \
    "$BASE/wcm_spacetime_vit_t8_direct_full_s$SEED/deploy.pt"
  run_eval 5cut "$DATASET_5CUT" staged "$SEED" \
    "$BASE/wcm_spacetime_vit_t8_full_s$SEED/deploy.pt"
done

echo "== 2/4 evaluate OOD on all four roles =="
for SEED in 3072 42; do
  run_eval ood "$DATASET_OOD" baseline "$SEED" \
    "$BASE/wcm_baseline_exp_s$SEED/deploy.pt"
  run_eval ood "$DATASET_OOD" mosaic "$SEED" \
    "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
  run_eval ood "$DATASET_OOD" spacetime "$SEED" \
    "$BASE/wcm_spacetime_vit_t8_direct_full_s$SEED/deploy.pt"
  run_eval ood "$DATASET_OOD" staged "$SEED" \
    "$BASE/wcm_spacetime_vit_t8_full_s$SEED/deploy.pt"
done

echo "== 3/4 build common endpoints, paired bootstrap, gates, and report =="
"$PYTHON" -m scripts.summarize_spacetime_threeway \
  --eval-root "$EVAL_ROOT" --datasets 5cut ood --seeds 3072 42 \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"

echo "== 4/4 complete =="
echo "Results: $EVAL_ROOT/threeway_results.md"
