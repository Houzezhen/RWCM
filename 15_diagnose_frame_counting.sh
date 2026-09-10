#!/usr/bin/env bash
# 帧号计数捷径诊断（§7.5）：纯评估，输出 outputs/frame_count_diag/<模型名>/summary.json + buckets.png
# 用法：bash 15_diagnose_frame_counting.sh [模型名...]；缺省跑 baseline reg4_d2 blockreg8_bs1
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

DATASET_ROOT="/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return"
OUTBASE="outputs/frame_count_diag"

declare -A MODELS=(
  [baseline]="outputs/wcm_baseline_r2/checkpoints/best.pt"
  [reg4_d2]="outputs/wcm_reg4_r2/checkpoints/best.pt"
  [bs1]="outputs/wcm_blockreg8_bs1/checkpoints/best.pt"
  [bs1_s42]="outputs/wcm_blockreg8_bs1_s42/checkpoints/best.pt"
  [bs1_s1337]="outputs/wcm_blockreg8_bs1_s1337/checkpoints/best.pt"
  [reg4_s42]="outputs/wcm_reg4_r2_s42/checkpoints/best.pt"
  [reg4_s1337]="outputs/wcm_reg4_r2_s1337/checkpoints/best.pt"
  [noleak]="outputs/wcm_blockreg8_noleak/checkpoints/best.pt"
  [treg4]="outputs/wcm_temporal_reg4_r1/checkpoints/best.pt"
)

if [[ $# -gt 0 ]]; then
  NAMES=("$@")
else
  NAMES=(baseline reg4_d2 bs1)
fi

mkdir -p outputs/batch_logs
for NAME in "${NAMES[@]}"; do
  CKPT="${MODELS[$NAME]:-}"
  OUT="$OUTBASE/$NAME"
  LOG="outputs/batch_logs/frame_diag_$NAME.log"
  if [[ -z "$CKPT" ]]; then echo "[bad ] unknown model: $NAME"; continue; fi
  if [[ -f "$OUT/summary.json" ]]; then echo "[skip] $NAME"; continue; fi
  if [[ ! -f "$CKPT" ]]; then echo "[wait] $NAME: no checkpoint yet ($CKPT)"; continue; fi
  echo "[run ] $NAME"
  "$PYTHON" -u -m scripts.diagnose_frame_counting \
    --checkpoint "$CKPT" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUT" \
    --split all \
    --batch-size 16 \
    --num-workers 2 \
    --probe-subset "${PROBE_SUBSET:-2000}" > "$LOG" 2>&1
  RC=$?
  echo "[done] $NAME rc=$RC"
  [[ $RC -ne 0 ]] && { echo "[fail] tail of $LOG:"; tail -n 30 "$LOG"; exit $RC; }
done
echo "FRAME DIAG DONE"
