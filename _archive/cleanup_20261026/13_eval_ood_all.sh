#!/usr/bin/env bash
# OOD 评估（0722 加板子环境，21832 窗口）：全部已完成模型，可反复重跑自动补齐
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

DATASET_ROOT="/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return"
OUTBASE="outputs/eval_ood_all"

declare -A MODELS=(
  [baseline]="outputs/wcm_baseline_r2/checkpoints/best.pt"
  [reg4_d2]="outputs/wcm_reg4_r2/checkpoints/best.pt"
  [treg4]="outputs/wcm_temporal_reg4_r1/checkpoints/best.pt"
  [s4t4]="outputs/wcm_s4t4_r1/checkpoints/best.pt"
  [blockreg8_leak5]="outputs/wcm_blockreg8_r1/checkpoints/best.pt"
  [blockreg8_noleak]="outputs/wcm_blockreg8_noleak/checkpoints/best.pt"
  [blockreg8_bs1]="outputs/wcm_blockreg8_bs1/checkpoints/best.pt"
  [bs1_s42]="outputs/wcm_blockreg8_bs1_s42/checkpoints/best.pt"
  [bs1_s1337]="outputs/wcm_blockreg8_bs1_s1337/checkpoints/best.pt"
  [noleak_s42]="outputs/wcm_blockreg8_noleak_s42/checkpoints/best.pt"
  [noleak_s1337]="outputs/wcm_blockreg8_noleak_s1337/checkpoints/best.pt"
  [reg4_s42]="outputs/wcm_reg4_r2_s42/checkpoints/best.pt"
  [reg4_s1337]="outputs/wcm_reg4_r2_s1337/checkpoints/best.pt"
  [baseline_exp_s3072]="/media/Data/user/WCM_outputs/wcm_baseline_exp_s3072/checkpoints/best.pt"
  [reg4_exp_s3072]="/media/Data/user/WCM_outputs/wcm_reg4_exp_s3072/checkpoints/best.pt"
  [bs1_exp_s3072]="/media/Data/user/WCM_outputs/wcm_bs1_exp_s3072/checkpoints/best.pt"
  [baseline_exp_s42]="/media/Data/user/WCM_outputs/wcm_baseline_exp_s42/checkpoints/best.pt"
  [reg4_exp_s42]="/media/Data/user/WCM_outputs/wcm_reg4_exp_s42/checkpoints/best.pt"
  [bs1_exp_s42]="/media/Data/user/WCM_outputs/wcm_bs1_exp_s42/checkpoints/best.pt"
  [reg4_early_exp_s3072]="/media/Data/user/WCM_outputs/wcm_reg4_early_exp_s3072/checkpoints/best.pt"
)

mkdir -p outputs/batch_logs "$OUTBASE"
# 扩训组中途 checkpoint 评估出的旧成绩必须删掉（final 模型重评），否则会被 skip
rm -rf "$OUTBASE/baseline_exp_s3072" "$OUTBASE/reg4_exp_s3072"
ORDER=(baseline reg4_d2 reg4_s42 reg4_s1337 treg4 s4t4 blockreg8_leak5 blockreg8_noleak noleak_s42 noleak_s1337 blockreg8_bs1 bs1_s42 bs1_s1337 baseline_exp_s3072 reg4_exp_s3072 bs1_exp_s3072 baseline_exp_s42 reg4_exp_s42 bs1_exp_s42 reg4_early_exp_s3072)

for NAME in "${ORDER[@]}"; do
  CKPT="${MODELS[$NAME]}"
  OUT="$OUTBASE/$NAME"
  LOG="outputs/batch_logs/evalood_$NAME.log"
  if [[ -f "$OUT/summary.json" ]]; then echo "[skip] $NAME"; continue; fi
  if [[ ! -f "$CKPT" ]]; then echo "[wait] $NAME: no checkpoint yet ($CKPT)"; continue; fi
  echo "[run ] $NAME"
  $PYTHON -u -m world_critic.evaluate --checkpoint "$CKPT" \
      --dataset-root "$DATASET_ROOT" --split all \
      --output-dir "$OUT" --batch-size 16 \
      --num-workers 12 --no-episode-curves \
      --log-every-batches 50 > "$LOG" 2>&1 || { echo "[fail] $NAME（tail 30 行如下）"; tail -30 "$LOG"; }
done
echo "OOD EVAL DONE（缺失的模型稍后重跑本脚本即可自动补齐）"
