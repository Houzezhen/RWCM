# Sparse4 跨帧输入消融记录

更新：2026-09-15

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
