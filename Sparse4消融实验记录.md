# Sparse4 跨帧输入消融记录

更新：2026-09-14

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

## 2. 输入来源消融

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

### 2.1 完整结果

| 数据集 | 模型 | seed | MSE | 相对 baseline | Pearson | Pearson 差 | bias |
|---|---|---:|---:|---:|---:|---:|---:|
| 5cut | S4-Image | 3072 | 0.011905 | -70.7% | 0.851944 | +0.517085 | -0.032926 |
| 5cut | S4-Image | 42 | 0.010734 | -70.8% | 0.848871 | +0.442201 | -0.013678 |
| 5cut | S4-State | 3072 | 0.014550 | -64.2% | 0.812827 | +0.477968 | -0.011129 |
| 5cut | S4-State | 42 | 0.013492 | -63.3% | 0.819381 | +0.412711 | +0.003865 |
| 5cut | Full Sparse4 | 3072 | 0.014012 | -65.5% | 0.820804 | +0.485945 | -0.024856 |
| 5cut | Full Sparse4 | 42 | 0.012750 | -65.3% | 0.829384 | +0.422714 | -0.007692 |
| OOD | S4-Image | 3072 | 0.038883 | -74.2% | 0.809193 | +0.668287 | -0.142031 |
| OOD | S4-Image | 42 | 0.035875 | -67.3% | 0.853740 | +0.627596 | -0.140483 |
| OOD | S4-State | 3072 | 0.082876 | -45.1% | 0.818306 | +0.677399 | -0.247151 |
| OOD | S4-State | 42 | 0.087807 | -20.1% | 0.838980 | +0.612836 | -0.268496 |
| OOD | Full Sparse4 | 3072 | 0.068516 | -54.6% | 0.847692 | +0.706786 | -0.211541 |
| OOD | Full Sparse4 | 42 | 0.066336 | -39.6% | 0.844623 | +0.618479 | -0.207781 |

所有消融相对 baseline 的 MSE 与 Pearson paired CI 均排除 0。S4-Image 的 OOD MSE CI：

| seed | MSE 差 95% CI | Pearson 差 95% CI |
|---:|---|---|
| 3072 | `[-0.11957, -0.10441]` | `[+0.62684, +0.71326]` |
| 42 | `[-0.08058, -0.06768]` | `[+0.57310, +0.68707]` |

### 2.2 两 seed 均值

| 模型 | 5cut MSE | 5cut Pearson | OOD MSE | OOD Pearson |
|---|---:|---:|---:|---:|
| Baseline | 0.038677 | 0.370765 | 0.130335 | 0.183525 |
| S4-State | 0.014021 | 0.816104 | 0.085342 | 0.828643 |
| Full Sparse4 | 0.013381 | 0.825094 | 0.067426 | **0.846158** |
| **S4-Image** | **0.011320** | **0.850408** | **0.037379** | 0.831466 |

### 2.3 centered MSE 与 bias

本文定义：

```text
bias = mean(prediction - target)
MSE = variance(error) + bias²
centered MSE = MSE - bias²
```

`bias < 0` 表示模型整体预测得比真实 return 更低（更悲观）；`bias > 0` 表示整体更乐观。bias 只描述平均偏移，不表示样本排序能力。一个模型即使没有更好地区分样本，也可能仅靠调整输出均值降低 MSE，因此必须同时检查 centered MSE 和 Pearson。

OOD centered MSE：

| 模型 | seed3072 | seed42 | 两 seed 均值 |
|---|---:|---:|---:|
| Baseline | 0.069005 | 0.058184 | 0.063595 |
| S4-State | 0.021792 | **0.015717** | 0.018755 |
| Full Sparse4 | 0.023766 | 0.023163 | 0.023465 |
| S4-Image | **0.018711** | 0.016140 | **0.017426** |

S4-Image 不仅 raw MSE 最低，centered MSE 均值也最低。因此其优势不是单纯校准，而是误差离散程度和排序结构都显著改善。S4-State 的 Pearson 很高，但 OOD bias 达到 `-0.247/-0.268`，说明它能判断相对阶段，却把绝对 value 整体预测得过低。

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

## 5. 消融结论

1. 稀疏视觉历史是主要收益来源。S4-Image 在 5cut 和 OOD 的 MSE 均为最佳，并在 5cut Pearson 上最佳。
2. proprioception 历史独立有效。S4-State 的 OOD Pearson 达到 `0.818/0.839`，证明机械臂位置和运动历史含有强阶段信息。
3. 当前简单加法融合没有产生稳定互补。Full Sparse4 的 OOD Pearson 均值略高，但 OOD MSE 和 centered MSE 均明显差于 S4-Image；两个 seed 上 Pearson 优势方向也不一致。
4. 下一阶段以 S4-Image 作为强 baseline，先不加入 28D state，单独检验 patch-level cross-frame fusion。

## 6. VGGT-inspired 的简明思路

这里不直接复制 VGGT，也不使用其相机/深度/点云几何头，只借鉴它最核心的表示思路：**不要先把每帧压缩成单个向量，而是让多帧 patch tokens 在压缩前联合交互。**

```text
[t, t-20, t-40, t-60] 四帧
            |
      共享 ViT 编码器
            |
  四组 patch tokens + 时间槽 embedding
            |
 causal cross-frame attention
  当前帧 token 可读取所有历史帧 token
            |
     当前状态表征 -> WCM value head
```

与 mosaic 的区别：mosaic 先把每帧缩小到画面的四分之一，再由 ViT 当作一张图处理；token 方案让每帧保持完整分辨率和独立 patch 网格，再学习末端执行器和按钮在跨帧中的对应、位移与运动方向。

主实验保持因果，只允许当前帧读取当前及过去帧。VGGT-inspired 候选与 S4-Image 使用完全相同的四个时间点、数据划分、训练步数和 value loss，唯一主要变量是 `mosaic` 与 `patch-level cross-frame attention`。

第一版不融合 state；若视觉 token 版本稳定优于 S4-Image，再使用 gated fusion 加入 proprioception，避免当前直接相加造成的校准干扰。

## 7. 当前结论

Full Sparse4 及两个单模态消融都在两个 seed 上稳定改善，输入来源已经完成初步归因：稀疏视觉历史贡献最大，state 历史提供强辅助阶段信号，但现有融合方式不是最佳。下一阶段进入 VGGT-inspired patch-level 跨帧视觉融合，并以 S4-Image 而不是 Full Sparse4 作为主要强 baseline。
