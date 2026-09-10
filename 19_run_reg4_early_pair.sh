#!/usr/bin/env bash
# 标准 early-register 两 seed 预检：smoke -> train -> paired episode evaluation。
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

mkdir -p outputs/batch_logs outputs/eval_reg4_early_pair/comparisons

echo "== 0/3 ViT register smoke gate =="
"$PYTHON" -m scripts.smoke_test_register_encoder --device cuda || {
  echo "[abort] smoke gate failed; no training was started"
  exit 1
}

CONFIGS=(
  configs/wcm_reg4_early_exp_s3072.yaml
  configs/wcm_reg4_early_exp_s42.yaml
)

echo "== 1/3 train early-register seeds =="
for CFG in "${CONFIGS[@]}"; do
  OUT=$("$PYTHON" - "$CFG" <<'PY'
import sys
import yaml

print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["output_dir"])
PY
  )
  LOG="outputs/batch_logs/$(basename "$OUT").log"
  if [[ -f "$OUT/deploy.pt" && -f "$OUT/metrics.jsonl" && $(wc -l < "$OUT/metrics.jsonl") -ge 10 ]]; then
    echo "[skip] completed: $OUT"
    continue
  fi
  echo "[train] $CFG -> $OUT"
  "$PYTHON" -m world_critic.train --config "$CFG" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  [[ $RC -eq 0 ]] || { echo "[abort] training failed: $CFG"; exit "$RC"; }
done

FIVECUT="/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return"
OOD="/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return"
BASE="/media/Data/user/WCM_outputs"
EVAL_ROOT="outputs/eval_reg4_early_pair"

run_eval() {
  local DATASET_NAME=$1
  local DATASET_ROOT=$2
  local MODEL_NAME=$3
  local CHECKPOINT=$4
  local OUTPUT="$EVAL_ROOT/$DATASET_NAME/$MODEL_NAME"
  local EPISODES="$OUTPUT/episode_curves/episode_metrics.json"
  local LOG="outputs/batch_logs/pair_${DATASET_NAME}_${MODEL_NAME}.log"

  [[ -f "$CHECKPOINT" ]] || { echo "[abort] missing checkpoint: $CHECKPOINT"; exit 1; }
  if [[ -f "$OUTPUT/summary.json" && -f "$EPISODES" ]]; then
    echo "[skip] evaluated: $DATASET_NAME/$MODEL_NAME"
    return
  fi
  echo "[eval] $DATASET_NAME/$MODEL_NAME"
  "$PYTHON" -u -m world_critic.evaluate \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --split all \
    --output-dir "$OUTPUT" \
    --batch-size 16 \
    --num-workers 4 \
    --no-episode-curves \
    --episode-metrics \
    --log-every-batches 50 > "$LOG" 2>&1 || {
      tail -n 40 "$LOG"
      exit 1
    }
}

echo "== 2/3 fixed-epoch paired evaluation =="
for DATASET_NAME in 5cut ood; do
  if [[ "$DATASET_NAME" == 5cut ]]; then DATASET_ROOT=$FIVECUT; else DATASET_ROOT=$OOD; fi
  run_eval "$DATASET_NAME" "$DATASET_ROOT" baseline_s3072 "$BASE/wcm_baseline_exp_s3072/deploy.pt"
  run_eval "$DATASET_NAME" "$DATASET_ROOT" early_s3072 "$BASE/wcm_reg4_early_exp_s3072/deploy.pt"
  run_eval "$DATASET_NAME" "$DATASET_ROOT" baseline_s42 "$BASE/wcm_baseline_exp_s42/deploy.pt"
  run_eval "$DATASET_NAME" "$DATASET_ROOT" early_s42 "$BASE/wcm_reg4_early_exp_s42/deploy.pt"
done

echo "== 3/3 episode-paired bootstrap =="
for DATASET_NAME in 5cut ood; do
  for SEED in 3072 42; do
    "$PYTHON" -m scripts.compare_experiments \
      "$EVAL_ROOT/$DATASET_NAME/baseline_s$SEED" \
      "$EVAL_ROOT/$DATASET_NAME/early_s$SEED" \
      --output "$EVAL_ROOT/comparisons/${DATASET_NAME}_s${SEED}.json"
  done
done

echo "DONE: inspect $EVAL_ROOT/comparisons/*.json"
echo "Decision: opposite seed directions => stop; both improve => add three paired seeds."
