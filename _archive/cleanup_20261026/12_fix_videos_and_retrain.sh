#!/usr/bin/env bash
# 补 0703_f_v21 缺失的视频硬链接，然后重建扩训集 + 重跑扩训训练
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
SRC=/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0703_f
DST=/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0703_f_v21
CACHE=/home/user/.cache/huggingface/lerobot/test

echo "① 补视频硬链接"
for key in observation.images.primary observation.images.wrist; do
  mkdir -p "$DST/videos/chunk-000/$key"
  n=0
  for f in "$SRC/videos/chunk-000/$key"/*.mp4; do
    b=$(basename "$f"); t="$DST/videos/chunk-000/$key/$b"
    [[ -e $t ]] || { ln "$f" "$t"; n=$((n+1)); }
  done
  echo "  $key: +$n linked ($(ls "$DST/videos/chunk-000/$key" | wc -l) total)"
done

echo "② 重建扩训集"
rm -rf "$CACHE/data_toiletButton_0814_expanded" "$CACHE/data_toiletButton_0814_expanded_with_return" \
       /media/Data/user/huggingface/lerobot/test/data_toiletButton_0814_expanded_with_return 2>/dev/null || true
bash 7_build_expanded_dataset.sh

echo "③ 验证视频完整性（177 集 × 2 相机）"
n_pri=$(ls "$CACHE/data_toiletButton_0814_expanded_with_return/videos/chunk-000/observation.images.primary" | wc -l)
n_wri=$(ls "$CACHE/data_toiletButton_0814_expanded_with_return/videos/chunk-000/observation.images.wrist" | wc -l)
echo "  primary=$n_pri wrist=$n_wri（应各 177）"
[[ $n_pri -eq 177 && $n_wri -eq 177 ]] || { echo "视频数不对，停止"; exit 1; }

echo "④ 启动扩训 6 个"
bash 8_run_expanded_training.sh
