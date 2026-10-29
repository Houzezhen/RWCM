#!/usr/bin/env bash
# 修复扩训数据集 schema（0703_f v1→v2.1）并重建 —— 在用户终端运行（绕过沙箱限制）
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PY=.venv/bin/python
CACHE=/home/user/.cache/huggingface/lerobot/test

# ① 清掉半成品
rm -rf "$CACHE/data_toiletButton_0814_expanded" "$CACHE/data_toiletButton_0814_expanded_with_return" \
       /media/Data/user/huggingface/lerobot/test/data_toiletButton_0814_expanded_with_return 2>/dev/null || true

# ② 把转换好的 v2.1 版 0703_f 移到 cache（避免项目目录被当作数据集根）
if [[ ! -d "$CACHE/data_toiletButton_0703_f_v21" ]]; then
  mv /home/user/code_1/WCM/_tmp_0703f_v21 "$CACHE/data_toiletButton_0703_f_v21"
fi

# ③ 重建扩训集（50 原始 + 77 尾段 + 50 失败 v2.1 + return, task_max=229）
bash 7_build_expanded_dataset.sh

# ④ 验证：schema 一致性 + 加载冒烟测试
$PY - <<'EOF'
import pyarrow.parquet as pq, glob, json
root='/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0814_expanded_with_return'
files=sorted(glob.glob(f'{root}/data/chunk-000/*.parquet'))
schemas={str(pq.read_schema(f).field('action').type) for f in files}
print('episodes:', len(files), '| distinct action schemas:', schemas)
info=json.load(open(f'{root}/meta/info.json'))
print('total_episodes:', info['total_episodes'], '| total_frames:', info['total_frames'])
import pyarrow.parquet as _pq
t=_pq.read_table(files[126]); print('ep126 (0703_f 首集) return 列:', 'return' in t.schema.names, '| rows', t.num_rows)
EOF
echo "FIX DONE"
