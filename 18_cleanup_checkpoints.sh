#!/usr/bin/env bash
# 清理冗余/作废模型产物（2026-09-08，清理依据见 寄存器变体实验报告.md §9.1 清理记录）
# 1) 保留的训练目录删 epoch-*.pt 中间快照（best.pt + last.pt 保留，续训/评估不受影响）
# 2) 整删三类整目录：
#    a. leak1/leak10（模型权重与 noleak 576/576 张量逐位一致，物理冗余）
#    b. D2-late 系列全部（结构错误：ViT 跑两遍 = 双倍深度混淆，报告 §8.3；数值已存档 §8.2/§8.1，
#       修复版 reg4_early 重跑中，其对照锚点数值在报告里，无需权重本身）
#    c. 其余已作废/无引用变体（treg4/s4t4/mix_c3/mix_d3/bf16/0814 首版）
# 3) 清理本地死软链
# 保留：baseline 全线（0814/r2/exp×2 seed）、bs1 全线（旧 3 seed + exp×2 seed）、noleak 3 seed（bs1 对照）
set -uo pipefail
BIG="/media/Data/user/WCM_outputs"
BEFORE=$(du -s "$BIG" | cut -f1)

# 保留清单（best.pt+last.pt）：baseline×4、bs1×5、noleak×3、扩训 reg4_early（在跑）
KEEP=" wcm_baseline_0814 wcm_baseline_r2 wcm_baseline_exp_s3072 wcm_baseline_exp_s42 \
 wcm_blockreg8_bs1 wcm_blockreg8_bs1_s42 wcm_blockreg8_bs1_s1337 wcm_bs1_exp_s3072 wcm_bs1_exp_s42 \
 wcm_blockreg8_noleak wcm_blockreg8_noleak_s42 wcm_blockreg8_noleak_s1337 \
 wcm_reg4_early_exp_s3072 "
is_kept() { [[ "$KEEP" == *" $1 "* ]]; }

echo "== 1/3 保留目录删 epoch-*.pt 快照 =="
N=0
for d in "$BIG"/*/; do
  name=$(basename "$d")
  is_kept "$name" || continue
  for f in "$d"checkpoints/epoch-*.pt; do
    [[ -f "$f" ]] || continue
    rm "$f" && N=$((N+1))
  done
done
echo "已删 $N 个快照（仅保留目录内）"

echo "== 2/3 整删作废目录 =="
DEAD_DIRS="wcm_blockreg8_leak1 wcm_blockreg8_leak10 \
 wcm_reg4_r2 wcm_reg4_r2_s42 wcm_reg4_r2_s1337 wcm_reg4_0814_bf16 \
 wcm_reg4_exp_s3072 wcm_reg4_exp_s42 wcm_blockreg8_r1 \
 wcm_temporal_reg4_r1 wcm_s4t4_r1 wcm_mix_c3 wcm_mix_d3"
for d in $DEAD_DIRS; do
  if [[ -d "$BIG/$d" ]]; then rm -rf "$BIG/$d" && echo "已删 $BIG/$d"; fi
done

echo "== 3/3 清理本地死软链 =="
for l in outputs/wcm_*; do
  [[ -L "$l" ]] || continue
  target=$(readlink "$l")
  [[ -e "$l" ]] || { rm "$l" && echo "已删软链 $l（→ $target 已不存在）"; }
done

AFTER=$(du -s "$BIG" | cut -f1)
echo
echo "清理完成：大盘 $(( BEFORE/1024/1024 ))G → $(( AFTER/1024/1024 ))G（释放 $(( (BEFORE-AFTER)/1024/1024 ))G）"
echo "保留：baseline 全线×4、bs1 全线×5、noleak×3（对照）、reg4_early（D2 修复复查在跑）、batch_logs、全部评估 summary"
