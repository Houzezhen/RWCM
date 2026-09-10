#!/usr/bin/env bash
# D2 修复复查：reg4 early 插入重跑（训练 → 5cut → OOD 一条龙）
# 前置：代码已支持 vision.register_insert: early（model.py，默认 late 保持旧行为）
# 判读：vs baseline_exp_s3072（0.04061/0.141）与 reg4_exp_s3072-late（0.04068/0.103），见报告 §8.2/§8.3
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

CFG="configs/wcm_reg4_early_exp_s3072.yaml"
OUT=$("$PYTHON" - "$CFG" <<'EOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output_dir"])
EOF
)
mkdir -p outputs/batch_logs

if [[ -f "$OUT/metrics.jsonl" && $(wc -l < "$OUT/metrics.jsonl") -ge 10 ]]; then
  echo "[skip train] $OUT"
else
  echo "[train] $CFG -> $OUT"
  "$PYTHON" -m world_critic.train --config "$CFG" 2>&1 | tee "outputs/batch_logs/$(basename "$OUT").log"
  RC=${PIPESTATUS[0]}
  echo "[train done] rc=$RC"
  [[ $RC -ne 0 ]] && { tail -n 30 "outputs/batch_logs/$(basename "$OUT").log"; exit $RC; }
fi

echo "== 5cut 评估（自动补齐新模型） =="
bash 6_eval_5cut.sh
echo "== OOD 评估（自动补齐新模型） =="
bash 13_eval_ood_all.sh
echo "REG4-EARLY DONE：结果在 outputs/eval_5cut/reg4_early_exp_s3072 与 outputs/eval_ood_all/reg4_early_exp_s3072"
