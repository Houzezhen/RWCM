"""早停探针: 25集时测图像AUC与右位成功率, 输出 EARLY_OK / EARLY_FAIL。"""
import numpy as np
import imageio.v2 as imageio
from pathlib import Path

raw = Path("/home/user/code_1/WCM/dataset_trials/metaworld_button_v2_train")
files = sorted(raw.glob("episode_*.npz"))
X, y, pos, succ = [], [], [], []
for f in files:
    z = np.load(f)
    r = imageio.get_reader(f.with_suffix(".mp4"))
    last = None
    for fr in r:
        last = np.asarray(fr, dtype=np.float32)
    feats = []
    for half in (last[:, :480], last[:, 480:]):
        h, w = half.shape[:2]
        small = half[: h // 24 * 24, : w // 24 * 24].reshape(24, h // 24, 24, w // 24, 3).mean(axis=(1, 3))
        feats.append(((small - small.mean()) / (small.std() + 1e-6)).reshape(-1))
    X.append(np.concatenate(feats))
    y.append(1.0 if bool(z["success"]) else -1.0)
    pos.append(float(z["obs"][0, 4]))
    succ.append(bool(z["success"]))
X, y = np.array(X), np.array(y)


def rank_tied(s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        r[order[i : j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    return r


def auc_clean(s, yt):
    n1, n0 = int(yt.sum()), int((~yt).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rank_tied(s)
    return float((r[yt].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


n = len(y)
idx = np.arange(n)
np.random.default_rng(0).shuffle(idx)
aucs = []
for f in range(5):
    te = np.zeros(n, bool)
    te[idx[f::5]] = True
    tr = ~te
    A = X[tr].T @ X[tr] + 10.0 * np.eye(X.shape[1])
    w = np.linalg.solve(A, X[tr].T @ y[tr])
    aucs.append(auc_clean(X[te] @ w, y[te] > 0))
a = float(np.mean(aucs))
pos_a = np.array(pos)
succ_a = np.array(succ)
hi_mask = pos_a > 0.05  # 右位 = obs[4] > 0.05 (绝对阈值, 基于dx=+0.08)
hi = succ_a[hi_mask]
right_rate = float(hi.mean()) if len(hi) else 1.0  # 无右位集时不因此项失败
print(f"n={n} AUC={a:.3f} 右位(n={int(hi_mask.sum())})成功率={right_rate:.0%}")
ok = a < 0.70 and (len(hi) < 3 or right_rate > 0)  # 右位<3集不判定
print("EARLY_OK" if ok else "EARLY_FAIL")
