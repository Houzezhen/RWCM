#!/usr/bin/env bash
# Fair batch-4 comparison: S4-Image mosaic vs Sparse-VGGT patch-token fusion.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

BASE="/media/Data/user/WCM_outputs"
EVAL_ROOT="outputs/eval_sparse_vggt_pair"
mkdir -p outputs/batch_logs "$EVAL_ROOT/comparisons"

echo "== 0/4 unit and real CUDA memory gate =="
"$PYTHON" -m pytest tests/test_register_config.py tests/test_temporal_input.py -q
"$PYTHON" -u -m scripts.smoke_test_sparse_vggt --device cuda --batch-size 4

echo "== 1/4 train fair batch-4 pairs =="
for SEED in 3072 42; do
  for VARIANT in sparse4_image_b4 sparse_vggt; do
    CFG="configs/wcm_${VARIANT}_exp_s${SEED}.yaml"
    OUT="$BASE/wcm_${VARIANT}_exp_s${SEED}"
    LOG="outputs/batch_logs/wcm_${VARIANT}_exp_s${SEED}.log"
    if [[ -f "$OUT/deploy.pt" && -f "$OUT/metrics.jsonl" && $(wc -l < "$OUT/metrics.jsonl") -ge 10 ]]; then
      echo "[skip] completed: $OUT"
      continue
    fi
    echo "[train] $CFG"
    "$PYTHON" -u -m world_critic.train --config "$CFG" 2>&1 | tee "$LOG"
  done
done

run_eval() {
  local dataset_name=$1 dataset_root=$2 model_name=$3 checkpoint=$4
  local output="$EVAL_ROOT/$dataset_name/$model_name"
  local log="outputs/batch_logs/vggt_${dataset_name}_${model_name}.log"
  [[ -f "$checkpoint" ]] || { echo "[abort] missing checkpoint: $checkpoint"; exit 1; }
  if [[ -f "$output/summary.json" && -f "$output/episode_curves/episode_curves.csv" ]]; then
    echo "[skip] evaluated: $dataset_name/$model_name"
    return
  fi
  "$PYTHON" -u -m world_critic.evaluate \
    --checkpoint "$checkpoint" \
    --dataset-root "$dataset_root" \
    --split all \
    --output-dir "$output" \
    --batch-size 4 \
    --num-workers 4 \
    --no-episode-curves \
    --episode-metrics \
    --log-every-batches 50 > "$log" 2>&1
}

echo "== 2/4 evaluate fixed-epoch deploy.pt =="
for DATASET_NAME in 5cut ood; do
  if [[ "$DATASET_NAME" == 5cut ]]; then
    DATASET_ROOT="/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return"
  else
    DATASET_ROOT="/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return"
  fi
  for SEED in 3072 42; do
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "baseline_s$SEED" "$BASE/wcm_baseline_exp_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "image_b4_s$SEED" "$BASE/wcm_sparse4_image_b4_exp_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "vggt_s$SEED" "$BASE/wcm_sparse_vggt_exp_s$SEED/deploy.pt"
  done
done

echo "== 3/4 primary paired comparison: Image-B4 vs Sparse-VGGT =="
for DATASET_NAME in 5cut ood; do
  for SEED in 3072 42; do
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/image_b4_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/vggt_s$SEED" \
      --align common \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_vggt_vs_image_b4_s${SEED}.json"
  done
done

echo "== 4/4 secondary paired comparison: original baseline vs Sparse-VGGT =="
for DATASET_NAME in 5cut ood; do
  for SEED in 3072 42; do
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/baseline_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/vggt_s$SEED" \
      --align common \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_vggt_vs_baseline_s${SEED}.json"
  done
done

echo "DONE: inspect $EVAL_ROOT/comparisons/*.json"
echo "Decision: Sparse-VGGT must beat Image-B4 on OOD centered MSE or Pearson without opposite-seed regression."
