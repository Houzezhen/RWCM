"""v10 裁判: iJEPA 在对称数据上的综合泄漏评估(五指标版)。

用法: python dataset_trials/judge_v10.py [split, 默认 ood]
输入: mw_eval_baseline_s3072_<split>/episode_curves/episode_curves.csv
      + 原始 npz(位置/成败)
输出: 五组指标 + 总结论
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ONLY = "--data-only" in sys.argv
SPLIT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "ood"
CURVES = Path(f"/media/Data/user/WCM_outputs/mw_eval_baseline_s3072_{SPLIT}/episode_curves/episode_curves.csv")
RAW = Path(f"/home/user/code_1/WCM/dataset_trials/metaworld_button_v2_{SPLIT}")

npz = sorted(RAW.glob("episode_*.npz"))
ep_pos, ep_succ, ep_tlen = {}, {}, {}
for i, f in enumerate(npz):
    z = np.load(f)
    ep_pos[i] = float(z["obs"][0, 4])
    ep_succ[i] = bool(z["success"])
    ep_tlen[i] = len(z["rew"])

_PQ = np.quantile(sorted(ep_pos.values()), [1/3, 2/3])  # 数据自适应三段分桶(基准位置随进程而变)

def pos_name(bx):
    return "左" if bx < _PQ[0] else ("中" if bx < _PQ[1] else "右")

# --- 1. 位置-成败独立性(TASK_LEN 修复验证) ---
print("=" * 60)
print("[1] 位置分层: 采集成功率(右位不应再是 0)")
buckets = {"左": [], "中": [], "右": []}
for i, bx in ep_pos.items():
    buckets[pos_name(bx)].append(ep_succ[i])
for n, v in buckets.items():
    print(f"    {n}位: n={len(v)}, 成功率={np.mean(v):.0%}")


def pred_succ(v, thr=-0.5):
    return v > thr

def auc(score, label):
    """二分类 ROC-AUC(rank 法)"""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

if DATA_ONLY:
    # 训前门: 数据级指标不达标即终止, 不浪费训练
    pos_auc = auc(np.array([ep_pos[i] for i in ep_pos]), np.array([ep_succ[i] for i in ep_pos]))
    import pyarrow.parquet as pq
    lrd = pq.read_table(f"/home/user/code_1/WCM/dataset_trials/lerobot/metaworld_button_v2_{SPLIT}/data/chunk-000/file-000.parquet",
                        columns=["observation.state", "episode_index"]).to_pandas()
    last = lrd.groupby("episode_index").tail(1)
    Xs = np.stack([np.asarray(x, dtype=float) for x in last["observation.state"]])
    ys = np.array([ep_succ[i] for i in sorted(ep_succ)])
    w = Xs[ys].mean(0) - Xs[~ys].mean(0)
    st_auc = auc(Xs @ w, ys)
    # 图像探针: 末帧降采样特征 → 岭回归 → CV AUC
    import imageio.v2 as imageio
    Ximg, yimg = [], []
    for i, f in enumerate(npz):
        r = imageio.get_reader(f.with_suffix(".mp4"))
        last = None
        for fr in r:
            last = np.asarray(fr, dtype=np.float32)
        feats = []
        for half in (last[:, :480], last[:, 480:]):
            h, w = half.shape[:2]
            small = half[:h//24*24, :w//24*24].reshape(24, h//24, 24, w//24, 3).mean(axis=(1, 3))
            feats.append(((small - small.mean()) / (small.std() + 1e-6)).reshape(-1))
        Ximg.append(np.concatenate(feats)); yimg.append(1.0 if ep_succ[i] else -1.0)
    Ximg, yimg = np.array(Ximg), np.array(yimg)
    def _rank(s):
        order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(len(s)) + 1
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[order[j+1]] == s[order[i]]: j += 1
            ranks[order[i:j+1]] = (i + j + 2) / 2
            i = j + 1
        return ranks
    def _ridge_auc(Xm, ym, k=5, lam=1.0):
        n = len(ym); idx2 = np.arange(n); np.random.default_rng(0).shuffle(idx2)
        out = []
        for f in range(k):
            te = np.zeros(n, bool); te[idx2[f::k]] = True; tr = ~te
            A = Xm[tr].T @ Xm[tr] + lam * np.eye(Xm.shape[1])
            w2 = np.linalg.solve(A, Xm[tr].T @ ym[tr])
            s = Xm[te] @ w2
            yt = ym[te] > 0
            out.append(((_rank(s)[yt].sum() - yt.sum()*(yt.sum()+1)/2) / (yt.sum()*(~yt).sum())) if not (yt.all() or (~yt).all()) else 0.5)
        return float(np.mean(out))
    img_auc = _ridge_auc(Ximg, yimg)
    ok = pos_auc < 0.65 and st_auc < 0.65 and img_auc < 0.65
    print(f"[训前门] 位置→成败 AUC={pos_auc:.3f}, state→成败 AUC={st_auc:.3f}, 图像→成败 AUC={img_auc:.3f} (均应<0.65)")
    print("DATA_GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

df = pd.read_csv(CURVES)
df["pos"] = df["episode_id"].map(ep_pos)
df["pos_name"] = df["pos"].map(pos_name)
df["true_succ"] = df["episode_id"].map(ep_succ)
df["task_len"] = df["episode_id"].map(ep_tlen)
df["err"] = (df["value"] - df["return"]) ** 2
reset = df[df.frame_index >= df.task_len + 1].copy()   # 复位段(按各自 task_len)
task = df[df.frame_index < df.task_len].copy()

# --- 2. 复位段混淆矩阵 + 按位置分层 ---
print("=" * 60)
print("[2] 复位段模型预测(混淆矩阵 + 按位置)")
rs, rf = reset[reset.true_succ], reset[~reset.true_succ]
tp = pred_succ(rs.value).mean(); fn = 1 - tp
fp = pred_succ(rf.value).mean(); tn = 1 - fp
print(f"    混淆: 成功→判成功 {tp:.0%}, 成功→判失败 {fn:.0%}")
print(f"          失败→判成功 {fp:.0%}, 失败→判失败 {tn:.0%}")
for n in ["左", "中", "右"]:
    g = reset[reset.pos_name == n]
    print(f"    {n}位复位段判成率: {(pred_succ(g.value) == g.true_succ).mean():.0%}  (n集={g.episode_id.nunique()})")

# --- 3. 平衡准确率 + AUC ---
print("=" * 60)
print("[3] 平衡准确率 / ROC-AUC")
bal = 0.5 * (tp + tn)
a = auc(-reset.value, reset.true_succ)  # value 越高越像成功
print(f"    复位段: 平衡准确率 = {bal:.1%}, AUC = {a:.3f}")

# --- 4. 基线参照 ---
print("=" * 60)
print("[4] 基线")
print(f"    随机猜测 = 50%, 类先验(全猜多数类) = {max(reset.true_succ.mean(), 1-reset.true_succ.mean()):.0%}")
pos_rate = {n: np.mean(buckets[n]) for n in buckets}
pos_pred_acc = np.mean([max(pos_rate[pos_name(ep_pos[i])], 1 - pos_rate[pos_name(ep_pos[i])]) for i in ep_pos])
print(f"    按位置猜(用位置-成败率) = {pos_pred_acc:.0%}  ← 判成率若接近此值=还在走位置捷径")

# --- 5. 线性探针 ---
print("=" * 60)
print("[5] 线性探针(复位末帧 state)")
states, outs, poss = [], [], []
import pyarrow.parquet as pq
lrd = pq.read_table(f"/home/user/code_1/WCM/dataset_trials/lerobot/metaworld_button_v2_{SPLIT}/data/chunk-000/file-000.parquet",
                    columns=["observation.state", "episode_index", "frame_index"]).to_pandas()
for i in ep_pos:
    g = lrd[(lrd.episode_index == i)].iloc[-1]
    states.append(np.asarray(g["observation.state"], dtype=float))
    outs.append(ep_succ[i]); poss.append(ep_pos[i])
X = np.array(states); y = np.array(outs); p = np.array(poss)

def probe_auc(Xm, ym):
    w = Xm[ym].mean(0) - Xm[~ym].mean(0)
    return auc(Xm @ w, ym)

print(f"    state→成败 AUC = {probe_auc(X, y):.3f}  (≈0.5=state 通道已堵)")
print(f"    位置→成败 AUC = {auc(p, y):.3f}  (≈0.5=位置与成败独立, TASK_LEN 修复生效)")
p3 = (p < 0.02).astype(int) * 0 + (p >= 0.09).astype(int) * 2 + ((p >= 0.02) & (p < 0.09)).astype(int) * 1
print(f"    state→位置(三分类, one-vs-rest AUC): ", end="")
for k, nm in [(0, "左"), (1, "中"), (2, "右")]:
    yk = (p3 == k)
    print(f"{nm}={probe_auc(X, yk):.2f} ", end="")
print("  (高=位置本就可见, 合法; 关键是它不再预测成败)")

# --- 任务段健康度(判成率降了但任务段别崩) ---
print("=" * 60)
print("[6] 任务段健康度(不应因噪声显著恶化)")
t_mse = task.err.mean()
print(f"    任务段 MSE = {t_mse:.4f} (v9 参照 0.0235)")
r_mse = reset.err.mean()
print(f"    复位段 MSE = {r_mse:.4f} (v9 参照 0.0001, 完全瞎猜应≈0.25)")
print("=" * 60)
verdict = "PASS" if (bal < 0.80 and abs(a - 0.5) < 0.3) else "STILL_LEAKY"
print(f"总结论: {verdict}  (平衡准确率 {bal:.1%}, AUC {a:.3f})")
