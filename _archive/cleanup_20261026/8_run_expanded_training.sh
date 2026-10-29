#!/usr/bin/env bash
# 扩训组训练：baseline / D2 / 单块版 × seed3072（+seed42 备选），177 集扩训数据
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

CONFIGS=(
  configs/wcm_baseline_exp_s3072.yaml
  configs/wcm_reg4_exp_s3072.yaml
  configs/wcm_bs1_exp_s3072.yaml
  configs/wcm_baseline_exp_s42.yaml
  configs/wcm_reg4_exp_s42.yaml
  configs/wcm_bs1_exp_s42.yaml
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
    echo "[skip] $OUT"
    continue
  fi
  echo "[run ] $CFG -> $OUT"
  "$PYTHON" -m world_critic.train --config "$CFG" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  echo "[done] $OUT rc=$RC"
  [[ $RC -ne 0 ]] && { tail -n 30 "$LOG"; exit $RC; }
done
echo "EXPANDED TRAINING DONE"
