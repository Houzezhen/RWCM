#!/usr/bin/env bash
# 备份 WCM 项目 + 记录 + 仿真 v10 十点权重 到 U盘 F101 (FAT32)。
# FAT32 注意: 单文件<4G(已核), 无符号链接(用 rsync -L 展开), 时间戳 2s 粒度。
set -eu
SRC=/home/user/code_1/WCM
OUT=/media/Data/user/WCM_outputs
USB=/media/F101/WCM_backup_20261123
LOG=/media/F101/backup_f101.log
MARK=/media/F101/.backup_done

mkdir -p "$USB"/{code,weights_mw_v10,results}
rm -f "$MARK"

echo "[$(date)] 1/3 复制代码+报告(排除 .venv 和本地旧 outputs)..." | tee -a "$LOG"
# 根目录有个文件名末尾带双引号的误名文件(FAT32 禁 "), 排除后以规范名单独保存内容
rsync -rtL --modify-window=2 --info=progress2 \
  --exclude='.venv/' --exclude='outputs/' --exclude='__pycache__/' --exclude='*.egg-info/' \
  --exclude='aic8 results and T120 inference videos"' \
  "$SRC"/ "$USB/code/"
cp '/home/user/code_1/WCM/aic8 results and T120 inference videos"' "$USB/code/aic8_results_note.txt"

echo "[$(date)] 2/3 压缩并复制仿真 v10 十点 deploy.pt(U盘实速~5MB/s, zstd 压缩省一半时间)..." | tee -a "$LOG"
STAGE=/media/Data/user/WCM_outputs/backup_stage_weights
ZSTD=/home/user/miniforge3/bin/zstd
mkdir -p "$STAGE"
for m in baseline spacetime_scratch levjepa levjepa_st lejepa lejepa_st \
         vjepa2 vjepa2_st vjepa21 vjepa21_st; do
  src="$OUT/mw_button_${m}_s3072/deploy.pt"
  dst="$USB/weights_mw_v10/mw_${m}_s3072_deploy.pt.zst"
  if [ -f "$dst" ] && [ "$dst" -nt "$src" ]; then echo "跳过已有 $dst" | tee -a "$LOG"; continue; fi
  echo "压缩 $m ..." | tee -a "$LOG"
  "$ZSTD" -3 -T0 -f "$src" -o "$STAGE/mw_${m}_s3072_deploy.pt.zst" >/dev/null
  cp "$STAGE/mw_${m}_s3072_deploy.pt.zst" "$dst"
  rm -f "$STAGE/mw_${m}_s3072_deploy.pt.zst"
done

echo "[$(date)] 3/3 复制评估结果与训练记录..." | tee -a "$LOG"
for m in baseline spacetime_scratch levjepa levjepa_st lejepa lejepa_st \
         vjepa2 vjepa2_st vjepa21 vjepa21_st; do
  d="$OUT/mw_button_${m}_s3072"
  mkdir -p "$USB/results/train_${m}"
  cp "$d/metrics.jsonl" "$d/resolved_config.json" "$d/episode_split.json" "$USB/results/train_${m}/" 2>/dev/null || true
  for split in test ood; do
    e="$OUT/mw_eval_${m}_s3072_${split}"
    [ -d "$e" ] && cp "$e/summary.json" "$USB/results/train_${m}/eval_${split}_summary.json"
  done
done
cp "$OUT"/mw_compare_*.json "$USB/results/" 2>/dev/null || true

echo "[$(date)] 校验..." | tee -a "$LOG"
n_pt=$(find "$USB/weights_mw_v10" -name '*.zst' | wc -l)
du -sh "$USB" | tee -a "$LOG"
echo "deploy.pt.zst 数量: $n_pt (应为 10)" | tee -a "$LOG"
date > "$MARK"
echo "[$(date)] BACKUP DONE" | tee -a "$LOG"
