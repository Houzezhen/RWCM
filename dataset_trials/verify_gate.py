"""v2 数据集三重终检: 夹爪对称 + 末帧泄漏 LDA + 画面首末对称。任一不过 exit 1。

用法: python dataset_trials/verify_gate.py <split 默认 train>
"""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
BASE = Path(f"/home/user/code_1/WCM/dataset_trials/metaworld_button_v2_{SPLIT}")
LEROT = Path(f"/home/user/code_1/WCM/dataset_trials/lerobot/metaworld_button_v2_{SPLIT}")

files = sorted(BASE.glob("episode_*.npz"))
assert files, f"no npz under {BASE}"

# 检1: 夹爪 集首 vs 集末 (0=闭 1=全开)
starts, ends = [], []
for f in files:
    z = np.load(f)
    starts.append(z["obs"][0, 3])
    ends.append(z["obs"][-1, 3])
grip_start, grip_end = np.mean(starts), np.mean(ends)
print(f"[检1 夹爪] 集首 {grip_start:.3f}±{np.std(starts):.3f}  集末 {grip_end:.3f}±{np.std(ends):.3f}", flush=True)

# 检2: 转换后(含噪声)末帧单帧 LDA
pq_files = sorted(LEROT.glob("data/**/*.parquet"))
assert pq_files, f"no parquet under {LEROT}"
frames = [pd.read_parquet(f, columns=["observation.state", "episode_success", "episode_index"]) for f in pq_files]
df = pd.concat(frames)
last = df.groupby("episode_index").tail(1)
S = np.stack([np.asarray(x).reshape(-1) for x in last[last.episode_success == 1]["observation.state"]])
F = np.stack([np.asarray(x).reshape(-1) for x in last[last.episode_success == 0]["observation.state"]])
X = np.concatenate([S, F])
y = np.concatenate([np.ones(len(S)), np.zeros(len(F))]).astype(bool)
proj = X @ (S.mean(0) - F.mean(0))
idx = np.arange(len(X))
np.random.default_rng(0).shuffle(idx)
accs = []
for f in range(5):
    te = np.zeros(len(X), bool)
    te[idx[f::5]] = True
    thr = 0.5 * (proj[te & y].mean() + proj[te & ~y].mean())
    accs.append(((proj[te] > thr) == y[te]).mean())
acc = float(np.mean(accs))
print(f"[检2 泄漏] 末帧LDA = {acc:.1%} (成{len(S)}/败{len(F)}, 门限80%)", flush=True)

# 检3: 画面首末对称 (mp4 首帧 vs 末帧平均像素差, 分组; 两组接近且不悬殊为对称)
# ffmpeg reader 的随机定位不可靠, 用迭代取末帧
stats = json.load(open(BASE / "stats.json"))
diffs_s, diffs_f = [], []
for rec in stats[:20]:
    mp4 = BASE / f"episode_{rec['episode']:03d}.mp4"
    r = imageio.get_reader(mp4)
    first = lastfr = None
    for fr in r:
        arr = np.asarray(fr, dtype=np.float32)
        if first is None:
            first = arr
        lastfr = arr
    (diffs_s if rec["success"] else diffs_f).append(np.abs(first - lastfr).mean())
img_s, img_f = float(np.mean(diffs_s)), float(np.mean(diffs_f))
print(f"[检3 画面] 首末平均像素差: 成功组 {img_s:.1f}  失败组 {img_f:.1f} (两组接近为对称)", flush=True)

grip_ok = abs(grip_start - grip_end) < 0.05
leak_ok = acc <= 0.80
img_ok = abs(img_s - img_f) < 5.0
print(f"GATE {'PASS' if (grip_ok and leak_ok and img_ok) else 'FAIL'} "
      f"(夹爪{'✓' if grip_ok else '✗'} 泄漏{'✓' if leak_ok else '✗'} 画面{'✓' if img_ok else '✗'})", flush=True)
sys.exit(0 if (grip_ok and leak_ok and img_ok) else 1)
