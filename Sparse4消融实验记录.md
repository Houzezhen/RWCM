# Sparse4 跨帧输入消融记录

更新：2026-09-13

## 1. 已完成的 Full Sparse4 结果

`WCM-Sparse4` 同时使用：

- 图像历史 `[t, t-20, t-40, t-60]`；
- 固定 2×2 temporal mosaic；
- 四个 7D proprioception 拼接为 28D；
- episode 边界处复制最早可用帧；
- 与 baseline 使用相同扩训数据、split seed、10 epoch、固定 `deploy.pt` 评估。

paired 比较使用共同 endpoint。原始 baseline 每集从 frame 2 开始，Sparse4 额外产生的前两个 endpoint 被排除：5cut 丢弃 `145×2=290` 个 candidate endpoint，OOD 丢弃 `125×2=250` 个 candidate endpoint。

| 数据集 | seed | baseline MSE | Sparse4 MSE | 相对变化 | baseline Pearson | Sparse4 Pearson | Pearson 差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5cut | 3072 | 0.040607 | 0.014012 | -65.5% | 0.334859 | 0.820804 | +0.485945 |
| 5cut | 42 | 0.036748 | 0.012750 | -65.3% | 0.406670 | 0.829384 | +0.422714 |
| OOD | 3072 | 0.150822 | 0.068516 | -54.6% | 0.140906 | 0.847692 | +0.706786 |
| OOD | 42 | 0.109848 | 0.066336 | -39.6% | 0.226144 | 0.844623 | +0.618479 |

### 1.1 OOD paired CI

| 数据集 | seed | MSE 95% CI | Pearson 95% CI | candidate 更低 MSE概率 |
|---|---:|---|---|---:|
| OOD | 3072 | `[-0.09049, -0.07407]` | `[+0.66244, +0.75601]` | 1.0 |
| OOD | 42 | `[-0.05065, -0.03656]` | `[+0.56115, +0.68110]` | 1.0 |

### 1.2 bias 检查

OOD 的平均 bias 也改善，但无法解释全部收益。去除 `bias²` 后的 centered MSE：

| 数据集 | seed | baseline centered MSE | Sparse4 centered MSE |
|---|---:|---:|---:|
| OOD | 3072 | 0.069005 | 0.023766 |
| OOD | 42 | 0.058184 | 0.023163 |

因此 Full Sparse4 的收益不是单纯输出均值校准，值得继续做输入来源消融。

## 2. 本次新增的两个消融

### S4-Image

- 图像：四帧 temporal mosaic；
- state：不进入 value context；
- 目的：测收益是否主要来自视觉历史。

配置：`configs/wcm_sparse4_image_exp_s{3072,42}.yaml`

### S4-State

- 图像：只输入当前帧；
- state：`[state_t,state_t-20,state_t-40,state_t-60]`，28D；
- 目的：测收益是否主要来自 proprioception 历史。

配置：`configs/wcm_sparse4_state_exp_s{3072,42}.yaml`

两组均使用 `history_size=1`，因此与原始 baseline 的 endpoint 集不同；统一使用 `compare_experiments.py --align common`，结果中会记录被排除的 endpoint 数量。

## 3. 运行方式

```bash
git pull
bash 21_run_sparse4_pair.sh
```

结果：

```text
outputs/eval_sparse4_pair/comparisons/
  5cut_sparse4_s3072.json
  5cut_sparse4_s42.json
  ood_sparse4_s3072.json
  ood_sparse4_s42.json
  5cut_image_s3072.json
  5cut_image_s42.json
  ood_image_s3072.json
  ood_image_s42.json
  5cut_state_s3072.json
  5cut_state_s42.json
  ood_state_s3072.json
  ood_state_s42.json
```

## 4. 判读规则

| 结果 | 论文解释 | 后续 |
|---|---|---|
| S4-State 接近 Full，S4-Image 明显较弱 | 主要是 proprioception temporal conditioning | 先做 state-centric offline critic/Q |
| S4-Image 接近 Full，S4-State 明显较弱 | 视觉运动历史是主要来源 | 进入 patch-level cross-frame fusion |
| 两者都改善，Full 最好 | 视觉和 proprioception 互补 | 实现 VGGT-inspired causal token fusion |
| 两者都明显弱于 Full | 需要联合跨模态输入 | 保留 Full，做 token fusion 前先查交互项 |
| 任一消融只改善 MSE、Pearson 不改善 | 主要是校准变化 | 不称为跨帧表示收益 |

VGGT 结构只有在 Full Sparse4 及至少一个单模态消融通过 OOD 多 seed 门禁后才实现。主门禁为：多数 seed 同向、OOD episode bootstrap CI 不跨 0、centered MSE 或 Pearson 同时改善。

## 5. 当前结论

Full Sparse4 已在两个 seed、5cut 和 OOD 上大幅改善 MSE 与 Pearson，说明稀疏历史输入确实解决了原始 WCM 的局部运动歧义。尚未确定收益来自图像 mosaic、proprioception 历史，还是二者交互；本次新增消融用于回答这个归因问题。
