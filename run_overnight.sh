#!/usr/bin/env bash
# 下班全自动流水线：等当前队列 → 扩训 6 个 → 磁盘迁移 → 全量评估（5cut）
# 用法：bash run_overnight.sh > outputs/batch_logs/overnight.log 2>&1 &
#      然后 tmux/nohup 均可，机器不能关机/休眠
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

echo "=== [$(date '+%F %T']) overnight pipeline start ==="

# ① 等待当前训练队列（5_run_paper_ablations.sh）结束：等到没有 world_critic.train 进程
while pgrep -f "world_critic.train" > /dev/null; do
  sleep 60
done
echo "=== [$(date '+%F %T']) paper ablations finished ==="

# ② 磁盘迁移（释放本地盘，给扩训腾空间）
bash 9_move_outputs_to_bigdisk.sh || true

# ③ 扩训 6 个（10ep × 6，数据×4.4）
bash 8_run_expanded_training.sh
RC=$?
echo "=== [$(date '+%F %T')] expanded training done rc=$RC ==="

# ④ 再迁移一次
bash 9_move_outputs_to_bigdisk.sh || true

# ⑤ 全量评估 5cut（含扩训组 + 补齐 leak 组和新 seed）
bash 6_eval_5cut.sh
echo "=== [$(date '+%F %T')] eval done rc=$? ==="

echo "=== [$(date '+%F %T']) ALL OVERNIGHT TASKS DONE ==="
