#!/usr/bin/env bash
# 一键下班脚本：补丁评估清单 + 后台启动 overnight 流水线（nohup 防掉线）
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

bash patch_eval_models.sh
mkdir -p outputs/batch_logs
nohup bash run_overnight.sh > outputs/batch_logs/overnight.log 2>&1 &
echo "overnight pipeline started, PID $!，日志: outputs/batch_logs/overnight.log"
echo "查看进度: tail -f outputs/batch_logs/overnight.log"
