#!/usr/bin/env bash
# 断点续训 bs1_exp_s42（昨晚 22:18 终端断连中断于 epoch4/10，resume 自 last.pt epoch3）
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1
export WCM_RESUME="/media/Data/user/WCM_outputs/wcm_bs1_exp_s42/checkpoints/last.pt"

mkdir -p outputs/batch_logs
"$PYTHON" -u -m world_critic.train --config configs/wcm_bs1_exp_s42.yaml 2>&1 | tee outputs/batch_logs/wcm_bs1_exp_s42_resume.log
RC=${PIPESTATUS[0]}
echo "[resume done] rc=$RC"
# 校验：metrics.jsonl 应有 10 行
echo -n "metrics.jsonl lines: "; wc -l < /media/Data/user/WCM_outputs/wcm_bs1_exp_s42/metrics.jsonl
exit $RC
