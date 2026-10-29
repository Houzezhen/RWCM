#!/usr/bin/env bash
# 构建扩训数据集：
#   data_toiletButton_0814_expanded_with_return
#   = 50 集（25S+25F，data_toiletButton_merged_0814_with_return 去掉 return 列前的原始 0814）
#     + 77 集（全成功，toiletButton_0814_merge ep145-221 尾段）
#     + 50 集（全失败，data_toiletButton_0703_f 无板子）
#   return 标尺锁 task_max=229（与原训练集一致，--task-max-override）
# 不占 GPU，纯数据准备
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PY=".venv/bin/python"
CACHE=/home/user/.cache/huggingface/lerobot/test
MEDIA=/media/Data/user/huggingface/lerobot/test
OUTNAME=data_toiletButton_0814_expanded

# ---------- ① 抽 0814_merge ep145-221 尾段（硬链接） ----------
if [[ ! -d "$CACHE/toiletButton_0814_tail77" ]]; then
$PY - <<'EOF'
import os, shutil, json
import pyarrow.parquet as pq
src='/media/Data/user/huggingface/lerobot/test/toiletButton_0814_merge'
dst='/home/user/.cache/huggingface/lerobot/test/toiletButton_0814_tail77'
eps=list(range(145,222))
os.makedirs(f'{dst}/data/chunk-000'); os.makedirs(f'{dst}/meta')
for e in eps:
    os.link(f'{src}/data/chunk-000/episode_{e:06d}.parquet', f'{dst}/data/chunk-000/episode_{e:06d}.parquet')
for key in ['observation.images.primary','observation.images.wrist']:
    os.makedirs(f'{dst}/videos/chunk-000/{key}')
    for e in eps:
        os.link(f'{src}/videos/chunk-000/{key}/episode_{e:06d}.mp4', f'{dst}/videos/chunk-000/{key}/episode_{e:06d}.mp4')
info=json.load(open(f'{src}/meta/info.json'))
lens={e: pq.read_metadata(f'{dst}/data/chunk-000/episode_{e:06d}.parquet').num_rows for e in eps}
info['total_episodes']=len(eps); info['total_frames']=sum(lens.values())
info['total_videos']=len(eps)*2; info['splits']={'train':f'0:{len(eps)}'}
json.dump(info, open(f'{dst}/meta/info.json','w'), indent=4)
with open(f'{dst}/meta/episodes.jsonl','w') as fo:
    for e in eps:
        fo.write(json.dumps({'episode_index':e,'tasks':['press the toilet button'],'length':lens[e],'task_index':0})+'\n')
for f in ['episodes_stats.jsonl','tasks.jsonl','stats.json']:
    shutil.copy2(f'{src}/meta/{f}', f'{dst}/meta/{f}')
json.dump({str(e):1 for e in eps}, open(f'{dst}/success_labels.json','w'))
print('tail77 done:',len(eps),'eps')
EOF
fi

# ---------- ② 三合一（50 原始 + 77 尾段成功 + 50 失败） ----------
if [[ ! -d "$CACHE/$OUTNAME" ]]; then
$PY scripts/merge_v21_datasets.py \
  --datasets /home/user/code_1/WCM/data_toiletButton_merged_0814 \
             $CACHE/toiletButton_0814_tail77 \
             $CACHE/data_toiletButton_0703_f_v21 \
  --output $CACHE/$OUTNAME
fi
# 合并脚本默认全标 failure，需重写真实标签：
#   原 50 集：data_toiletButton_merged_0814/success_labels.json（25S+25F）
#   尾段 77 集：全成功；0703_f 50 集：全失败
$PY - <<'EOF'
import json
cache='/home/user/.cache/huggingface/lerobot/test'
out=f'{cache}/data_toiletButton_0814_expanded'
base=json.load(open('/home/user/code_1/WCM/data_toiletButton_merged_0814/success_labels.json'))  # 0..49
labels={}
for i in range(50):  labels[str(i)]=int(str(i) in base and base[str(i)]==1)
for i in range(50,127): labels[str(i)]=1   # tail77
for i in range(127,177): labels[str(i)]=0  # 0703_f
json.dump(labels, open(f'{out}/success_labels.json','w'))
n1=sum(labels.values()); print(f'success={n1} failure={len(labels)-n1} total={len(labels)}')
EOF

# ---------- ③ 加 return，task_max 锁 229 ----------
if [[ ! -d "$CACHE/${OUTNAME}_with_return" ]]; then
$PY scripts/add_returns_to_lerobot_v21.py \
  --root $CACHE/$OUTNAME \
  --output-dir $CACHE/${OUTNAME}_with_return \
  --failure-penalty 100 \
  --normalization task_max \
  --task-max-override 229 \
  --success-labels $CACHE/$OUTNAME/success_labels.json
fi
echo "EXPANDED DATASET READY: $CACHE/${OUTNAME}_with_return"
