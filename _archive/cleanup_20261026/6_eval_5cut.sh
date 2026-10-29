#!/usr/bin/env bash
# 独立同分布测试集评估：0814_5cut_test_with_return（145 集，与训练集零交集）
# 评估所有已完成模型；GPU 占用与训练可共存（batch 16，显存 ~4GB），与 5_run_paper_ablations.sh 并行时可能互相拖慢
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

DATASET_ROOT="/media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return"
OUTBASE="outputs/eval_5cut"

# 名称: checkpoint 路径（跑完的训练会自动补进 outputs/）
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
  [leak1]="outputs/wcm_blockreg8_leak1/checkpoints/best.pt"
  [leak10]="outputs/wcm_blockreg8_leak10/checkpoints/best.pt"
  [baseline_exp_s3072]="/media/Data/user/WCM_outputs/wcm_baseline_exp_s3072/checkpoints/best.pt"
  [reg4_exp_s3072]="/media/Data/user/WCM_outputs/wcm_reg4_exp_s3072/checkpoints/best.pt"
  [bs1_exp_s3072]="/media/Data/user/WCM_outputs/wcm_bs1_exp_s3072/checkpoints/best.pt"
  [baseline_exp_s42]="/media/Data/user/WCM_outputs/wcm_baseline_exp_s42/checkpoints/best.pt"
  [reg4_exp_s42]="/media/Data/user/WCM_outputs/wcm_reg4_exp_s42/checkpoints/best.pt"
  [bs1_exp_s42]="/media/Data/user/WCM_outputs/wcm_bs1_exp_s42/checkpoints/best.pt"
  [reg4_early_exp_s3072]="/media/Data/user/WCM_outputs/wcm_reg4_early_exp_s3072/checkpoints/best.pt"
)

mkdir -p outputs/batch_logs
for NAME in $(printf '%s\n' "${!MODELS[@]}" | sort); do
  CKPT="${MODELS[$NAME]}"
  OUT="$OUTBASE/$NAME"
  LOG="outputs/batch_logs/eval5cut_$NAME.log"
  if [[ -f "$OUT/summary.json" ]]; then
    echo "[skip] $NAME"
    continue
  fi
  if [[ ! -f "$CKPT" ]]; then
    echo "[wait] $NAME: no checkpoint yet ($CKPT)"
    continue
  fi
  echo "[run ] $NAME"
  "$PYTHON" -u -m world_critic.evaluate \
    --checkpoint "$CKPT" \
    --output-dir "$OUT" \
    --dataset-root "$DATASET_ROOT" \
    --split all \
    --batch-size 16 \
    --num-workers 2 \
    --no-episode-curves \
    --log-every-batches 50 > "$LOG" 2>&1
  RC=$?
  echo "[done] $NAME rc=$RC"
  [[ $RC -ne 0 ]] && { echo "[fail] tail of $LOG:"; tail -n 30 "$LOG"; exit $RC; }
done
echo "EVAL DONE (缺失的模型稍后重跑本脚本即可自动补齐)"
