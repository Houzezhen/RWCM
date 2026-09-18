# Sparse4 跨帧输入消融记录

更新：2026-09-16

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

## 8. 虚拟机运行与权重清理

切换到当前实验分支并确认环境：

```bash
cd ~/code_1/WCM
git fetch origin
git switch cross-frame-wcm
git pull --ff-only
.venv/bin/python -m pytest tests/test_register_config.py tests/test_temporal_input.py -q
```

复现 Sparse4 及 Image/State 消融：

```bash
CUDA_VISIBLE_DEVICES=0 bash 21_run_sparse4_pair.sh
```

脚本检测到完整的 `deploy.pt` 和 10 轮 `metrics.jsonl` 时会跳过已完成训练，因此可以安全重跑以补齐缺失评估。

清理训练权重前先预览：

```bash
bash 22_cleanup_cross_frame_weights.sh
```

核对输出中的精确目录后执行：

```bash
bash 22_cleanup_cross_frame_weights.sh --apply
```

清理策略：baseline 与 S4-Image 保留完整训练状态；Full Sparse4 与 S4-State 只保留 `deploy.pt`；RankNet 和已否决 register 线整目录删除。仓库内 comparison JSON、评估 summary、训练日志和本文档不删除。

## 9. Sparse-VGGT 实验

实现采用四个完整分辨率稀疏帧的共享 ViT patch tokens。每个 camera 内，当前帧 CLS query 通过两层 cross-attention 读取四帧 `CLS+patch` context；加入四个时间槽 embedding，输出再进入原 WCM view pooling、language fusion、context trunk 和 value head。第一版不融合 proprioception。

为避免 batch size 混淆，主实验同时重训 batch 4 的 S4-Image 对照。两组共享相同 seed、数据 split、epoch、优化器和 loss；跨帧模块在所有公共模块之后初始化，因此不会改变公共参数的初始随机数流。

虚拟机运行：

```bash
cd ~/code_1/WCM
git switch cross-frame-wcm
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 bash 23_run_sparse_vggt_pair.sh
```

脚本首先执行 batch 4 的真实 CUDA forward/backward/AdamW smoke 并报告峰值显存；门禁通过后训练 Image-B4 和 Sparse-VGGT 两 seed，最后分别生成相对 Image-B4（主比较）和原始 WCM（次比较）的 episode-paired JSON。

### 9.1 主比较结果（vs Image-B4）

`outputs/eval_sparse_vggt_pair/comparisons/*_vggt_vs_image_b4_*.json`，共同 endpoint、5cut 145 集 / OOD 125 集，`dropped_*_endpoints` 均为 0：

| 数据集 | seed | Image-B4 MSE | VGGT MSE | 相对变化 | MSE 差 95% CI | Image-B4 Pearson | VGGT Pearson | Pearson 差 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 5cut | 3072 | 0.011045 | 0.015136 | +37.0% | `[+0.00237, +0.00596]` | 0.858099 | 0.785703 | `[-0.09520, -0.05058]` |
| 5cut | 42 | 0.011294 | 0.014012 | +24.1% | `[+0.00159, +0.00388]` | 0.858204 | 0.818451 | `[-0.05440, -0.02552]` |
| OOD | 3072 | 0.045589 | 0.045767 | +0.4% | `[-0.00260, +0.00296]` | 0.850513 | 0.783083 | `[-0.08267, -0.05263]` |
| OOD | 42 | 0.053365 | 0.098087 | +83.8% | `[+0.04138, +0.04813]` | 0.836089 | 0.773999 | `[-0.07760, -0.04644]` |

8 个"数据集 × seed × 指标"格子中，7 个 paired CI 明确劣于 Image-B4；唯一例外是 OOD seed 3072 的 MSE（+0.4%，CI 跨 0，只能算打平）。Pearson 在全部四个组合上均显著更低。

### 9.2 centered MSE

用各组合的 `mean_bias` 按 §2.3 定义 `centered MSE = MSE - bias²` 计算：

| 数据集 | seed | Image-B4 centered MSE | VGGT centered MSE |
|---|---:|---:|---:|
| 5cut | 3072 | 0.010282 | 0.015074 |
| 5cut | 42 | 0.010234 | 0.013454 |
| OOD | 3072 | 0.018376 | 0.024129 |
| OOD | 42 | 0.019841 | 0.042159 |

VGGT 的 centered MSE 同样全线更高，说明这次回归不是校准偏移造成的，误差离散程度本身就变差了。

### 9.3 次比较结果（vs 原始 WCM baseline）

| 数据集 | seed | baseline MSE | VGGT MSE | 相对变化 | MSE 差 95% CI | baseline Pearson | VGGT Pearson | Pearson 差 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 5cut | 3072 | 0.040543 | 0.014961 | -63.1% | `[-0.03288, -0.01989]` | 0.334801 | 0.783993 | `[+0.36751, +0.54487]` |
| 5cut | 42 | 0.036749 | 0.013854 | -62.3% | `[-0.03005, -0.01742]` | 0.407065 | 0.817099 | `[+0.33813, +0.49645]` |
| OOD | 3072 | 0.151672 | 0.045567 | -70.0% | `[-0.11366, -0.09874]` | 0.139923 | 0.780653 | `[+0.59597, +0.69069]` |
| OOD | 42 | 0.110355 | 0.097266 | -11.9% | `[-0.02016, -0.00626]` | 0.225576 | 0.770595 | `[+0.49083, +0.60398]` |

跨帧视觉历史相对原始单帧 baseline 依然是大幅改善的，说明方向成立，失败的是相对 mosaic 的具体实现。

### 9.4 判定：主门禁未通过，本版 patch-level cross-frame fusion 否决

门禁要求 Sparse-VGGT 在 OOD centered MSE 或 Pearson 上胜过 Image-B4 且无反向 seed 回归。实际两个指标在两个 seed 上均未超过对照，且不是方向翻转而是系统性劣化，故判为未通过，停止这版 token fusion 实现，不调层数、注意力头数、温度或时间槽 embedding 追单点最优。

Batch-4 重训的 Image-B4 明显弱于 §2 的 batch-8 S4-Image（OOD MSE 0.0456 vs 0.038883、0.0534 vs 0.035875），说明该线上 batch size 影响不可忽略。主比较本身是同为 batch 4 的公平对照，但 S4-Image 仍是当前 OOD 指标最好的输入方案，后续若继续跨帧方向应回到 mosaic 表示上寻找增量，而不是维持全分辨率 patch tokens。

## 10. 交替式 causal frame/global transformer（下一步）

### 10.1 设计修正

上一版 Sparse-VGGT 的 cross-frame 模块放在每帧 ViT 完整编码之后，并且由一个当前 CLS query 读取所有帧的 CLS+patch。该路径对跨帧空间对应的交互过晚，且会把大量 patch 信息压缩到单个 query，不能充分利用末端执行器和按钮的位移证据。

下一版改为真正的 **frame-wise / global attention 交替视觉编码器**，不是在最终 latent/trunk 后叠加 transformer，也不是只在 ViT 中段交互一次：

```text
四帧共享 patch embedding
        |
2 个 Frame-wise ViT block（帧内 attention，共享权重）
        |
1 个 block-causal Global attention（四帧全部 patch）
        |
2 个 Frame-wise ViT block
        |
1 个 block-causal Global attention
        |
... 共 6 个 stage，四帧始终保留 ...
        |
只读取当前帧 CLS
        |
原 WCM view pooling / language fusion / value head
```

Frame-wise block 是原始 ViT block：每帧内部的 `CLS+patch` 做 self-attention，四帧使用同一套 block 权重，但该步不跨帧。Global block 将四帧全部 `CLS+patch` 展开做 self-attention，允许跨帧、跨空间位置匹配运动目标；更新后的四帧 token 全部保留并进入下一轮 frame-wise block，不在中途丢弃历史帧。

历史输入按 `[t,t-20,t-40,t-60]` 提供，内部改排为 `[t-60,t-40,t-20,t]`。Global attention 使用帧级块因果 mask：帧 `i` 的任意 patch 只能读取本帧和更早帧的全部 patch，同一帧内双向可见；较早帧绝不能读取更晚 observation。该 mask 应通过“修改当前帧不影响最老帧输出”的单元测试。

模块由 `model.use_temporal_transformer=true` 开启。ViT-base 的 12 个 frame-wise block 默认划成 6 个 stage，即每 2 个 frame-wise block 后插入 1 个 global block。为保持轻量，global attention 在 `192D` adapter 空间运行（4 heads、MLP ratio 2），再投影回 ViT 的 `768D`。每个 global block 有独立的零初始化残差门控，使 warm-start 初始行为等价于原始逐帧 ViT；新增 `temporal_adapter.*` 使用 10× learning rate，已训练的共享 ViT/WCM 参数保持 base learning rate。

这一路的 dense attention 计算复杂度约为 `O((K·P)²)`，但 QKV/MLP 通道缩至 192D；服务器 smoke 必须报告真实峰值显存。若 batch 4 不可行，应先减小 batch 并保持与 Image 对照相同的 effective batch/optimizer steps，不能改变模型语义来迁就显存。

### 10.2 Warm-start 方式

配置：

- `configs/wcm_sparse_temporal_exp_s3072.yaml`
- `configs/wcm_sparse_temporal_exp_s42.yaml`
- `24_run_sparse_temporal_pair.sh`

默认从对应的 Sparse-VGGT `deploy.pt` 加载共享的 ViT、语言、trunk、value 和 dynamics 权重；旧的末端 `cross_frame_encoder.*` 参数被明确忽略，6 个新 global block 的 `temporal_adapter.*` 参数初始化。训练日志会打印 missing/ignored keys，禁止静默丢权重。这是结构迁移微调，不恢复旧 optimizer/scheduler，也不等价于从旧 VGGT 输出继续训练。

如果只有 S4-Image 权重，也可以作为迁移初始化加载共享参数，但由于 mosaic 与四帧独立输入不同，这不应称为严格续训，需在报告中单独标记。

### 10.3 判定门禁

先只跑两个 seed，使用 batch-4、10 epoch、共同 endpoint，与 Image-B4 比较。只有满足以下条件才保留交替结构：

1. OOD paired episode bootstrap 的 MSE 或 centered-MSE 95% CI 上界 `< 0`；
2. 两个 seed 均不出现反向回归，最好扩展到 `3/4` seed；
3. Pearson 不系统性下降，且 mean bias 不明显恶化。

若交替结构仍不超过 Image-B4，则冻结 S4-Image 作为跨帧视觉主线，不再在失败的 full patch-token 路径上叠加 register 或继续调 attention 超参。

### 10.4 第 5 轮 OOD 中间预检（s3072）

训练进行中，用第 5 轮 checkpoint（`epoch-0004.pt`，即 `save_every_epochs: 5` 的落盘点）提前做了一次 OOD 预检。评估在 GPU 0 上运行（`outputs/eval_temporal_probe/ood_ep5_s3072/`），与 `eval_sparse_vggt_pair/ood/image_b4_s3072` 做 episode 配对 bootstrap（125 集，22082 共同 endpoint，无丢弃）：

| 指标 | Image-B4 | temporal 第 5 轮 | 差值 | 95% CI |
|---|---:|---:|---:|---|
| MSE | 0.045589 | 0.049184 | +7.9% | `[+0.00123, +0.00602]` 排除 0，更差 |
| Pearson | 0.850513 | 0.806898 | -0.043616 | `[-0.05589, -0.03209]` 排除 0，更差 |
| mean bias | -0.164962 | -0.148906 | 略好 | — |
| centered MSE | 0.018376 | 0.027021 | 更差 | — |

预检解读：

1. temporal 模块相对 warm-start 起点在补回损失——第 5 轮 OOD Pearson 0.807 高于 VGGT 原始的 0.783——但仍显著落后 Image-B4 的 0.851，MSE 与 Pearson 两项 CI 均排除 0。
2. 剩余 5 轮 lr 已衰减至很低，填平 Pearson 0.044 + MSE 8% 差距的可能性不大。本预检倾向否决方向。
3. 处置：s3072 按原协议训满 10 轮，用 `deploy.pt` 做正式比较以与预检互相印证；若正式结果仍输给 Image-B4，则 s42 组不再训练，按 §10.3 门禁冻结该结构。此预检只用于提前决策，不作为最终判定依据。

## 11. Sparse Persistent Memory（下一候选）

### 11.1 动机与对上一版的诊断

§10 的 dense alternating 模型虽然在代码中保留四帧 hidden tokens 直到第 12 个 ViT block，但它没有独立、可显式读出的历史记忆流。每个 dense global block 使用零初始化 residual gate：初始时只有 gate 本身能立即得到有效梯度，gate 打开前 attention/QKV/MLP 主体的梯度被乘到接近 0。最终模型又只读取当前帧 CLS，因此历史信息必须经过多次弱门控 attention 间接传到当前 CLS。

第 5 轮 OOD 预检已经表明该结构在补回旧 VGGT 损失，但仍显著落后 Image-B4。新候选不再增加 dense global patch attention，而是利用 S4-Image 的关键证据：历史从输入开始存在非常重要，但四帧保持完整分辨率并非必要条件。目标是用少量 persistent memory tokens 贯穿全部视觉层。

### 11.2 精确结构

输入顺序在 dataset 中为 `[t,t-20,t-40,t-60]`，进入 memory encoder 后翻转为因果时间顺序 `[t-60,t-40,t-20,t]`。

```text
四帧图像 [t-60, t-40, t-20, t]
        |
共享 ViT patch embedding
        |
每帧 X_i: [197, 768]（CLS + 196 patches）
        |
共享 learned-query attention resampler
Q_mem: [8, 768]
M_i = Resampler(Q_mem, X_i): [8, 768]
        |
Stage 1: Frame-wise ViT blocks 1-2 on [X_i, M_i]
         -> Causal Global Attention on four-frame M only
        |
Stage 2: Frame-wise ViT blocks 3-4 on [X_i, M_i]
         -> Causal Global Attention on four-frame M only
        |
... 6 stages; X_i and M_i both persist ...
        |
current CLS cross-attends current M
        |
current CLS + LayerScale * memory summary
        |
原 projection / view pooling / language fusion / WCM value head
```

压缩只在 patch embedding 后初始化一次，但 memory 不是静态摘要。每个 frame-wise stage 将 `[X_i,M_i]` 拼成 `205` 个 token 送入原始共享 ViT block，因此存在双向更新：patch 持续刷新 memory，memory 也持续把历史摘要注回 patch/CLS。每个 stage 后只把 `M_i` 放入跨帧 global attention；完整 patch 不跨帧做 dense attention。

### 11.3 压缩器

每帧共享 8 个可学习 query：

```text
M_i^0 = Q_mem + CrossAttention(LN(Q_mem), LN(X_i), LN(X_i))
M_i^0 = M_i^0 + FFN(LN(M_i^0)) + time_embedding_i
```

不使用 average pooling。不同 learned queries 可自行分工捕获机械臂/夹爪、按钮、接触区域、相对位置、阶段和背景等信息。第一版不加 slot diversity 或正交正则，避免引入额外实验变量。

### 11.4 Global memory 的严格因果性

每帧只有 `m=8` 个 memory token，global attention 的序列长度为 `K*m=4*8=32`，而 dense alternating 的长度为 `4*197=788`。全局 attention 矩阵由约 `788²` 降为 `32²`。

帧级 block-causal mask：

```text
M_t-60 <- {M_t-60}
M_t-40 <- {M_t-60, M_t-40}
M_t-20 <- {M_t-60, M_t-40, M_t-20}
M_t    <- {M_t-60, M_t-40, M_t-20, M_t}
```

同一帧的 8 个 memory token 双向可见；任何历史帧不能读取更晚 observation。测试必须满足：修改当前帧输入不能改变最老帧 global-memory 输出；当前输出对历史帧输入具有非零梯度。

### 11.5 初始化与优化

- 原始 12 个 frame-wise ViT blocks、projection、语言和 WCM heads 从旧 Sparse-VGGT checkpoint warm-start；
- 旧 `cross_frame_encoder.*` 明确忽略；新 `sparse_memory_encoder.*` 明确列为 missing/initialized，其他 key 不允许静默不匹配；
- global memory block 不使用 zero gate，改用可学习 LayerScale，初值 `1e-3`，保证第一步 resampler、global QKV/MLP 和 readout 都有梯度；
- memory 模块使用 `temporal_lr_scale=10`，共享 ViT/WCM 保持 base lr `1e-5`；
- 这是结构迁移微调，不恢复旧 optimizer、scheduler 或 RNG。

### 11.6 配置、实现和运行

实现入口：

- `world_critic/model.py::SparseTemporalMemoryEncoder`
- `world_critic/model.py::VisionEncoder.encode_sparse_temporal_memory`
- `model.use_sparse_temporal_memory=true`

实验文件：

- `configs/wcm_sparse_memory_exp_s3072.yaml`
- `configs/wcm_sparse_memory_exp_s42.yaml`
- `scripts/smoke_test_sparse_memory.py`
- `25_run_sparse_memory_pair.sh`

服务器运行：

```bash
cd ~/code_1/WCM
git switch cross-frame-wcm
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 bash 25_run_sparse_memory_pair.sh
```

脚本先跑单元测试和 batch-4 真实 CUDA forward/backward/AdamW smoke。smoke 必须报告 `memory_queries`、`resampler_qkv`、`first_global_qkv`、`readout_qkv` 四类非零梯度以及峰值显存；门禁通过后训练两 seed，最后生成相对 Image-B4 的共同 endpoint paired JSON。comparison 显式传递模型 seed，避免旧报告中 s42 JSON 被默认 bootstrap seed 3072 污染元数据。

训练日志每 `log_every` 额外记录 `sparse_memory_layerscale_mean`、`sparse_memory_layerscale_abs_max`、`sparse_memory_readout_scale` 和各参数组学习率。若 LayerScale 长期停留在初值附近、塌到 0 或异常放大，应先判为历史通路未学开/不稳定，不能仅凭最终 value 指标解释 memory 机制。

### 11.7 判定门禁

SparseMemory 的强基线仍是相同 batch/seed 的 Image-B4，而不是原始 WCM。晋级要求：

1. OOD MSE 或 centered MSE 的 episode-paired 95% CI 上界 `<0`；
2. OOD Pearson 不系统性下降；
3. 两 seed 不出现相反方向回归；
4. mean bias 不以牺牲 centered MSE 的方式改善；
5. 若 s3072 第 5 轮预检已在 MSE/Pearson 两项显著劣于 Image-B4，可先训满 s3072 定版，再决定是否启动 s42。

若该结构仍不能超过 Image-B4，应接受当前监督信号只足以稳定利用输入级 mosaic 的结论，停止继续增加时序模块复杂度，转向 return/risk/Q 监督和下游策略收益验证。

### 11.8 从头训练入口

`configs/wcm_sparse_memory_exp_s{3072,42}.yaml` 和 `25_run_sparse_memory_pair.sh` 是 warm-start 版本：它们带有 `init_from`，会从旧 Sparse-VGGT `deploy.pt` 加载共享权重。

若要完全重新开始，使用以下 scratch 配置和脚本：

- `configs/wcm_sparse_memory_scratch_s3072.yaml`
- `configs/wcm_sparse_memory_scratch_s42.yaml`
- `26_run_sparse_memory_scratch_pair.sh`

scratch 配置没有 `init_from`，因此从预训练 ViT/CLIP 和随机初始化的 WCM/memory 模块开始；输出目录使用 `wcm_sparse_memory_scratch_s*`。脚本检测到同名输出目录会直接中止，防止把旧的 `metrics.jsonl` 或 checkpoint 混入新实验。

### 11.9 SparseMemory scratch 结果：未过门禁

`outputs/eval_sparse_memory_scratch_pair/comparisons/*_memory_scratch_vs_image_b4_*.json`（共同 endpoint，5cut 145 集 28017 点 / OOD 125 集 22082 点，无丢弃，bootstrap seed 与模型 seed 一致）：

| 数据集 | seed | Image-B4 MSE | memory-scratch MSE | 相对变化 | MSE 差 95% CI | Image-B4 Pearson | memory Pearson | Pearson 差 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 5cut | 3072 | 0.011045 | 0.011841 | +7.2% | `[-0.00017, +0.00184]` 跨 0 | 0.858099 | 0.842955 | `[-0.03126, +0.00008]` 跨 0 |
| 5cut | 42 | 0.011294 | 0.012038 | +6.6% | `[-0.00074, +0.00238]` 跨 0 | 0.858204 | 0.825637 | `[-0.04995, -0.01634]` 更差 |
| OOD | 3072 | 0.045589 | 0.079970 | +75.4% | `[+0.03116, +0.03761]` 更差 | 0.850513 | 0.654614 | `[-0.22131, -0.17203]` 更差 |
| OOD | 42 | 0.053365 | 0.046920 | -12.1% | `[-0.00984, -0.00302]` 更好 | 0.836089 | 0.802188 | `[-0.05060, -0.01790]` 更差 |

逐条对照 §11.7 门禁：

1. OOD MSE CI 上界 `<0`：只有 s42 满足，s3072 反向大幅劣化（+75.4%）——两 seed 方向相反，违反第 3 条。
2. OOD Pearson 四组全部为负，三组 CI 明确排除 0——系统性下降，违反第 2 条。
3. s42 的 MSE 收益伴随 bias 从 -0.183 收窄到 -0.132，更像校准偏移碰巧变小（违反第 4 条的精神），不构成表示收益。

### 11.10 patch-token 路线三次尝试总结

OOD Pearson（两 seed）：

| 结构 | s3072 | s42 | 判定 |
|---|---:|---:|---|
| Image-B4（mosaic 对照） | 0.851 | 0.836 | 强 baseline |
| Sparse-VGGT cross-attention（§9） | 0.783 | 0.774 | 否决 |
| temporal warm-start（§10，s3072 第 5 轮预检） | 0.807 | — | 倾向否决 |
| SparseMemory scratch（§11.9） | 0.655 | 0.802 | 否决，两 seed 撕裂 |

结论：全分辨率 patch-token 路线三次尝试（VGGT cross-attention → temporal transformer warm-start → SparseMemory scratch）均未超过输入级 mosaic。接受 §11.7 的最终结论：当前监督信号只足以稳定利用输入级 mosaic 跨帧历史，停止增加时序模块复杂度，冻结 S4-Image（mosaic）为视觉输入方案，后续转向 return/risk/Q 监督与下游策略收益验证。

## 12. Mosaic Temporal Residual：锚定强基线的增量实验

### 12.1 动机与结构

本实验不再恢复已否决的 full patch-token/register 路线。它直接复用 Image-B4 的单次 mosaic ViT forward：将最终 `14×14` patch grid 按 mosaic 四象限池化为 `[t-60,t-40,t-20,t]` 四个时间槽，仅在 `384D` WCM latent 空间运行一层 causal self-attention 和 MLP。当前视图表示为：

```text
z_current = z_image_b4 + tanh(alpha) * temporal_residual(quadrant_tokens)
```

`alpha` 严格初始化为 0，因此加载 Image-B4 `deploy.pt` 后，首个 forward 与原模型完全一致；新增分支约 1.2M 参数，使用 `10×` 学习率。历史分支不增加 ViT forward 次数，也不向预训练 ViT 内插入随机 token。

### 12.2 严格 continuation 对照

候选和对照分别从相同 seed 的 Image-B4 `deploy.pt` warm-start，并都继续训练 5 轮：

- continuation control：不增加结构，用于扣除额外训练轮数的收益；
- mosaic residual：唯一变量为零门控 temporal residual。

配置：

- `configs/wcm_mosaic_continue_s{3072,42}.yaml`
- `configs/wcm_mosaic_residual_s{3072,42}.yaml`
- `27_run_mosaic_residual_pair.sh`

主判定必须比较 residual vs continuation，而不是只比较 residual vs 原始 Image-B4。晋级要求：两个 seed 的 OOD MSE 或 centered MSE 同向改善、paired 95% CI 至少一个明确排除 0，且 OOD Pearson 不系统性下降。若只超过原始 Image-B4、但不超过 continuation，则收益归因于额外微调，不归因于结构。

## 13. SpaceTimeViT + Perceiver-64 主实验

### 13.1 定位

Mosaic 只用于证明并教授“历史视觉有效”，不是最终模型输入，也不重新训练。主模型直接复用已经完成的 Image-B4 `deploy.pt` 作为冻结 teacher，并复用其中的 `google/vit-base-patch16-224-in21k` 作为 student 的逐帧空间编码器。原始 `[B,T,V,C,H,W]` 历史帧分别经过共享 ViT，固定空间 patch 位置沿时间做 factorized attention，再由 64 个 Perceiver query 将任意配置长度的 `T×V×P` token 压缩为固定 64 个新视觉 token。主实验使用 `T=8,V=2,P=197`，即 `3152→64`（约 `49.3×`）；`T=4` 仅作为消融，另保留 `T=16` 扩展入口。这 64 个 token 先与语言融合，再以同一环境时间步的 token group 直接进入 causal context trunk；trunk 处理后才做无参数 masked mean readout，避免在 VLM/context 之前提前压成单 token。

### 13.2 四阶段训练

1. `spacetime_align`（最多 5 轮）：完整加载已有 Image-B4 的 ViT backbone、projection 与 WCM 同形权重，并将已训练的 camera embedding复制给 SpaceTime encoder；每个视频 time slot 使用固定的非零初始化以保留帧身份。冻结 Image-B4 visual teacher、student ViT/WCM 和时空层，只训练 Perceiver 与输出 projection。teacher 始终使用与原训练完全相同的 `[0,20,40,60]` 四帧 mosaic；student 的 raw history offsets独立配置，主实验为 8 帧 `[0,9,17,26,34,43,51,60]`。T=4/8/16 都覆盖相同的 60-step 时间范围，只改变采样密度。teacher 同时给出两个只读结果：原 view-attention pooling 状态专供 gate=0 保持 baseline，`A=mean(camera visual tokens)` 专供对齐；student 使用 `B=mean(64 Perceiver tokens)`，因此 A/B 两侧仍是相同的无参数 mean pooling。gate=0，优化 cosine + normalized MSE。validation cosine 达到 `0.9` 时立即停止；否则监控训练 cosine EMA（decay `0.99`），连续 4000 个 optimizer step 未出现至少 `1e-4` 的提升时停止。达到阈值或平台均视为对齐阶段通过；若最大 epoch 内两者均未触发，runner 判定阶段失败。停止后保存 best validation cosine 和实际平台值。
2. `spacetime_gate`（5 轮）：只训练时空层和 Perceiver，gate 按 optimizer step 从 0 线性升到 1，alignment weight 同步从 `0.2` 线性衰减到 0；最后一步 gate=1/weight=0 时不再执行 teacher。该阶段必须导出末步 checkpoint 而非中途 validation best，随后执行传入/不传入 teacher 的逐位一致性门禁，成功后 teacher 永久退出。
3. `spacetime_joint`（5 轮）：关闭 mosaic teacher，gate 固定 1；解冻 ViT 最后 4 层、视觉 projection、language fusion/context trunk 与 value/risk/Q heads。
4. `spacetime_full`（3 轮）：仍为 teacher-free/gate=1，以 `2e-6` 全量微调 ViT 与 WCM（CLIP 文本塔按既有协议继续冻结）。Align、Joint、Full 从验证集 best checkpoint 导出 `deploy.pt`；Gate 阶段例外，导出完成调度后的末步权重。

Risk target 为 `1-episode_success`；Q head 在 value 计算之后读取行为动作，以 return 为监督，保持 `V(s)` 的动作隔离。最终 checkpoint 的配置同时关闭 `history_mosaic` 与 `spacetime_teacher_enabled`，部署图只包含原始历史帧、SpaceTimeViT 和 Perceiver。

### 13.3 对照和门禁

主对照为相同 batch/seed/split 的：

- 单帧 ViT；
- 已有四帧 Image-B4 mosaic（不重训）；
- SpaceTimeViT + Perceiver-64。

门禁：用真实 checkpoint 完整前向验证 gate=0 相对 Image-B4 baseline 的 `context/value/dynamics/target` 最大逐位误差 `<1e-5`；阶段 0 validation cosine 必须达到 `0.9` 或训练 cosine EMA 已达平台；gate=1 在传入/不传入 teacher 时上述输出最大逐位误差 `<1e-5`；最终 OOD Pearson 两个 seed 均不得低于 mosaic；两个 seed 的 MSE 方向一致；若 raw MSE 改善但 centered MSE 未改善，则 Pearson paired CI 下界必须大于 0，排除单纯 bias 收窄。

配置和入口：

- `configs/wcm_vit_single.yaml`
- `configs/wcm_spacetime_{align,gate,joint,full}.yaml`
- `28_run_spacetime_vit_pair.sh`

同一入口通过 `WCM_VARIANT=t4|t8|t16` 选择视频长度；默认 `t8` 是主实验，`t4` 是短历史消融，`t16` 用于检查增加上游计算后 OOD Pearson 是否继续提升。teacher mosaic 在三个变体中都固定为原始四帧输入。

### 13.4 t8 主实验结果（2026-09-18，双 seed，batch 4）

全流程跑通：`wcm_vit_single_s{3072,42}` 对照 + t8 四阶段 + 等价性门禁 + 5cut/OOD 评估 + paired episode bootstrap（20000 次重采样）。比较文件在 `outputs/eval_spacetime_vit_t8_pair/comparisons/`。

**vs 单帧 ViT（下界）——压倒性、两 seed 一致：**

| 指标 | 5cut | OOD |
|---|---|---|
| ΔMSE | -0.028 / -0.031（CI 全负） | -0.100 / -0.146（CI 全负） |
| ΔPearson | +0.453 / +0.463 | +0.654 / +0.719（单帧 0.13~0.16 → spacetime 0.81~0.85） |

**vs mosaic（主对照）——两 seed 方向相反，未过门禁：**

| | s3072 | s42 |
|---|---|---|
| OOD MSE | -0.0011，CI [-0.0036,+0.0015]（含 0，平） | **-0.0271，CI [-0.0301,-0.0242]（显著更优）** |
| OOD Pearson | **-0.039，CI [-0.052,-0.026]（显著更差，0.851→0.812）** | +0.017，CI [0.000,+0.033]（下限贴 0） |

5cut 上 s3072 MSE 略差 / s42 MSE 略优、Pearson 均平——同样无一致方向。

**判定：门禁失败。** OOD Pearson 两 seed 未同时不低于 mosaic（s3072 显著 -0.039），MSE 方向不一致。

**结论与解读：**

1. **架构可行性成立**：共享 ViT + factorized 时序 attention + Perceiver 压缩 3152→64（约 49×），能把 mosaic 的历史编码能力基本搬进 64 token（OOD Pearson 0.81~0.85 vs mosaic 0.84~0.85）。此前三次时序化尝试（§9 cross-attention、§10 alternating、§11 SparseMemory）均未达到此水平。
2. **teacher 锚定嫌疑**：s42 在 MSE 上略有突破而 s3072 Pearson 被 anchor，符合"align/gate 蒸馏把 student 拉进 mosaic 表示空间、后半程不足突破"的预期（与 §13 设计评审时的预判一致）。
3. **方差观察**：mosaic 自身双 seed 波动大（OOD MSE 0.046/0.053），spacetime 波动反而更小（0.045/0.026）；此现象未计入门禁，仅记录。

**下一步（最小消融，决定该线生死）：** 去掉 align/gate 蒸馏阶段，LayerScale 小初值（1e-3）+ 新模块 10× lr 直接端到端训练 spacetime 层 + Perceiver，其余对照与门禁不变。若仍为打平，则接受结论：当前监督下压缩历史无增量，mosaic 为性价比最优，该线终结。
