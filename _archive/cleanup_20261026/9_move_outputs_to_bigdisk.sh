#!/usr/bin/env bash
# 将本地 outputs/ 下已完成的训练目录迁移到 /media/Data 并软链回（磁盘保护）
# 已是软链或目标不存在的自动跳过；训练进行中（无 last.pt）的跳过
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
DEST=/media/Data/user/WCM_outputs

for D in outputs/wcm_* outputs/wcm2_*; do
  [[ -d "$D" ]] || continue
  [[ -L "$D" ]] && continue
  NAME=$(basename "$D")
  # 完成判定：metrics.jsonl 存在且 last.pt 存在（10ep 跑完才有 last）
  if [[ ! -f "$D/metrics.jsonl" || ! -f "$D/checkpoints/last.pt" ]]; then
    echo "[skip-running?] $NAME"
    continue
  fi
  if [[ -e "$DEST/$NAME" ]]; then
    echo "[skip-exists ] $DEST/$NAME 已存在，手动处理"
    continue
  fi
  echo "[move] $NAME"
  mv "$D" "$DEST/$NAME"
  ln -s "$DEST/$NAME" "$D"
done
df -h / | tail -1
