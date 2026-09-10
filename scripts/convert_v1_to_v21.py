#!/usr/bin/env python
"""把 v1 (fixed_size_list) parquet cast 成 v2.1 (list<float>, timestamp double) 格式，输出到新目录（数据文件重写，视频硬链接）"""
import os, json, shutil, sys
import pyarrow as pa, pyarrow.parquet as pq

src = '/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0703_f'
dst = '/home/user/code_1/WCM/_tmp_0703f_v21'
if os.path.exists(dst):
    print('already exists'); sys.exit(0)

eps = sorted(int(f[8:-8]) for f in os.listdir(f'{src}/data/chunk-000') if f.endswith('.parquet'))
os.makedirs(f'{dst}/data/chunk-000'); os.makedirs(f'{dst}/meta')

float_list = pa.list_(pa.float32())
new_fields = {}
for e in eps:
    t = pq.read_table(f'{src}/data/chunk-000/episode_{e:06d}.parquet')
    schema = t.schema
    casts = {}
    new_schema = schema
    for col in ['action', 'observation.state']:
        if pa.types.is_fixed_size_list(schema.field(col).type):
            casts[col] = pa.compute.cast(schema.field(col).type, float_list) if False else None
    # 逐列转换
    cols = {}
    for name in schema.names:
        f = schema.field(name)
        if pa.types.is_fixed_size_list(f.type):
            cols[name] = t.column(name).cast(float_list, safe=False)
        elif name == 'timestamp' and pa.types.is_float32(f.type):
            cols[name] = t.column(name).cast(pa.float64(), safe=False)
        else:
            cols[name] = t.column(name)
    pq.write_table(pa.table(cols, schema=None if False else pa.schema([
        pa.field(n, cols[n].type if not pa.types.is_fixed_size_list(schema.field(n).type) else float_list, schema.field(n).nullable)
        for n in schema.names
    ])), f'{dst}/data/chunk-000/episode_{e:06d}.parquet')

# 视频：硬链接
for key in ['observation.images.primary', 'observation.images.wrist']:
    os.makedirs(f'{dst}/videos/chunk-000/{key}')
    for e in eps:
        s = f'{src}/videos/chunk-000/{key}/episode_{e:06d}.mp4'
        if os.path.exists(s):
            os.link(s, f'{dst}/videos/chunk-000/{key}/episode_{e:06d}.mp4')

# meta
info = json.load(open(f'{src}/meta/info.json'))
info['features'] = dict(info.get('features', {}))
# 更新 features 里的类型描述（如有）
shutil.copy2(f'{src}/meta/info.json', f'{dst}/meta/info.json')
for f in os.listdir(f'{src}/meta'):
    if f != 'info.json':
        shutil.copy2(f'{src}/meta/{f}', f'{dst}/meta/{f}')
if os.path.exists(f'{src}/success_labels.json'):
    shutil.copy2(f'{src}/success_labels.json', f'{dst}/success_labels.json')
print('converted', len(eps), 'episodes ->', dst)
