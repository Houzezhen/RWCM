#!/usr/bin/env bash
# Loss-only ranking experiment: train two seeds, evaluate fixed-epoch deploy.pt,
# then bootstrap paired episodes for both MSE and Pearson.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

BASE="/media/Data/user/WCM_outputs"
EVAL_ROOT="outputs/eval_rank_pair"
mkdir -p outputs/batch_logs "$EVAL_ROOT/comparisons"

echo "== 0/3 config and unit smoke =="
"$PYTHON" -m pytest tests/test_ranking_loss.py tests/test_compare_experiments.py -q

echo "== 1/3 train MSE+ranking candidates =="
for SEED in 3072 42; do
  CFG="configs/wcm_rank_exp_s${SEED}.yaml"
  OUT="$BASE/wcm_rank_exp_s${SEED}"
  LOG="outputs/batch_logs/wcm_rank_exp_s${SEED}.log"
  if [[ -f "$OUT/deploy.pt" && -f "$OUT/metrics.jsonl" && $(wc -l < "$OUT/metrics.jsonl") -ge 10 ]]; then
    echo "[skip] completed: $OUT"
    continue
  fi
  echo "[train] $CFG"
  "$PYTHON" -u -m world_critic.train --config "$CFG" 2>&1 | tee "$LOG"
done

run_eval() {
  local dataset_name=$1
  local dataset_root=$2
  local model_name=$3
  local checkpoint=$4
  local output="$EVAL_ROOT/$dataset_name/$model_name"
  local log="outputs/batch_logs/rank_${dataset_name}_${model_name}.log"

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
    --batch-size 16 \
    --num-workers 4 \
    --no-episode-curves \
    --episode-metrics \
    --log-every-batches 50 > "$log" 2>&1
}

echo "== 2/3 fixed-epoch paired evaluation =="
for DATASET_NAME in 5cut ood; do
  if [[ "$DATASET_NAME" == 5cut ]]; then
    DATASET_ROOT="/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return"
  else
    DATASET_ROOT="/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return"
  fi
  for SEED in 3072 42; do
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "baseline_s$SEED" "$BASE/wcm_baseline_exp_s$SEED/deploy.pt"
    run_eval "$DATASET_NAME" "$DATASET_ROOT" "rank_s$SEED" "$BASE/wcm_rank_exp_s$SEED/deploy.pt"
  done
done

echo "== 3/3 paired episode bootstrap =="
for DATASET_NAME in 5cut ood; do
  for SEED in 3072 42; do
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/baseline_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/rank_s$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_s${SEED}.json"
  done
done

echo "DONE: inspect $EVAL_ROOT/comparisons/*.json"
echo "Decision: require OOD Pearson improvement in both seeds; reject on opposite directions."
