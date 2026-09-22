#!/usr/bin/env bash
# 论文支撑实验：多 seed 复现（B1/B8-noleak/D2 × seed42/1337）+ 泄漏梯度（1%/10%）
# 全部 10ep、协议对齐既有消融；同分布 val 指标随训练写入各自 metrics.jsonl
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

CONFIGS=(
  configs/wcm_blockreg8_bs1_s42.yaml
  configs/wcm_blockreg8_bs1_s1337.yaml
  configs/wcm_blockreg8_noleak_s42.yaml
  configs/wcm_blockreg8_noleak_s1337.yaml
  configs/wcm_reg4_r2_s42.yaml
  configs/wcm_reg4_r2_s1337.yaml
  configs/wcm_blockreg8_leak1.yaml
  configs/wcm_blockreg8_leak10.yaml
)

mkdir -p outputs/batch_logs
for CFG in "${CONFIGS[@]}"; do
  OUT=$("$PYTHON" - "$CFG" <<'EOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output_dir"])
EOF
)
  LOG="outputs/batch_logs/$(basename "$OUT").log"
  if [[ -f "$OUT/metrics.jsonl" && $(wc -l < "$OUT/metrics.jsonl") -ge 10 ]]; then
    echo "[skip] $OUT already has 10 epochs"
    continue
  fi
  echo "[run ] $CFG -> $OUT (log: $LOG)"
  "$PYTHON" -m world_critic.train --config "$CFG" > "$LOG" 2>&1
  RC=$?
  echo "[done] $OUT rc=$RC"
  if [[ $RC -ne 0 ]]; then
    echo "[fail] see $LOG (tail below)"
    tail -n 30 "$LOG"
    exit $RC
  fi
done
echo "ALL DONE"
