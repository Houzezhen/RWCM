# evo pistar06 / WCM / SpaceTime 价值函数对比报告

日期：2026-10-23
评测数据集：OOD `data_toiletButton_0722_with_return`（125 集 / 22207 帧）
判定口径：OOD 共同 endpoint + episode-level paired bootstrap（20000 次，seed = 模型 seed）

---

## 1. 结论摘要

1. **evo pistar06 显著优于原始 WCM**：OOD Pearson 0.511 vs 0.140/0.226，MSE 0.075 vs 0.152/0.110，两套 WCM seed 下配对 CI 均不含 0。
2. **evo 显著落后于带历史视觉的方案**：相对 Image-B4 mosaic，Pearson 低 0.32~0.34、MSE 高 0.022~0.030；相对 SpaceTime-T60 差距更大（Pearson 低 0.33~0.34，MSE 高 0.042~0.051）。
3. **瓶颈不在骨干容量**：evo 使用约 6.7 亿参数的 SigLIP-SO400M + Gemma-3-270M，参数约为 WCM（1.7 亿）的 4 倍，但 OOD 排序能力（每集 Spearman 0.447）仍远低于 1.7 亿参数的 mosaic（0.950）与 SpaceTime-T120（0.995）。
4. **三条差异（输入信息、标签定义、训练数据）同时存在**，因此本报告只在**系统级**比较上成立，不能据此宣称“架构因果”。

---

## 2. 评测设置

### 2.1 参与比较的模型

| 角色 | 模型 | 输入 | 参数量 | 说明 |
|---|---|---|---:|---|
| 原始 WCM | `wcm_baseline_exp_s{3072,42}` | 单帧 + 语言 + 动作 | 1.71 亿 | ViT-base + CLIP |
| 对照基线 | `wcm_sparse4_image_b4_exp_s{3072,42}` | 4 帧 mosaic `[0,20,40,60]` | 1.71 亿 | Image-B4 |
| 候选（主） | `wcm_spacetime_vit_t8_full_s{3072,42}` | 8 帧 × 2 相机 → 64 token | 1.76 亿 | factorized temporal attention + Perceiver-64 |
| 候选（扩展） | `wcm_spacetime_vit_t8_direct_t120_s{3072,42}` | 8 帧 × 2 相机，跨度 120 步 → 64 token | 1.76 亿 | teacher-free direct |
| **外部对照** | `evo_0824_new/checkpoints/last` | **单帧 × 2 相机 + 文本化状态** | ~6.7 亿 | pistar06：SigLIP-SO400M(384) + Gemma-3-270M，201-bin 分布头 |

### 2.2 evo 推理方式

evo 仓库此前只把预测值写回数据集，**没有留下任何评测指标文件**，因此本次重新推理：

- 数据集复制为 v3.0 格式（`/media/Data/user/huggingface/lerobot/lerobotevo/evo_eval_ood/`）；
- 用 `lerobot_value_infer` 全量推理 22207 帧，写入 `complementary_info.value_evo`；
- 预测分布：min -0.314、max -0.0007、mean -0.078、std 0.076；
- 推理耗时约 17 ms/帧（含数据加载，非同一延迟测量框架，仅供参考）。

### 2.3 对齐与统计

- 使用 WCM 三方对照生成的共同 endpoint 清单，evo 侧 **0 个缺失 endpoint**；
- seed 3072：125 集 / 21832 endpoints；seed 42：125 集 / 21832 endpoints；
- target 一致性逐点校验通过（原 v2.1 数据集的 `return`，避免转换精度误差）；
- 配对 bootstrap 20000 次，episode 级重采样。

---

## 3. 结果

### 3.1 OOD 绝对指标（seed 3072）

| 模型 | MSE | Pearson | centered MSE | mean bias | 每集 Spearman（均值） |
|---|---:|---:|---:|---:|---:|
| 原始 WCM | 0.1517 | 0.140 | 0.0691 | -0.287 | 0.141 |
| **evo pistar06** | **0.0753** | **0.511** | **0.0264** | **+0.221** | **0.447** |
| Image-B4 mosaic | 0.0452 | 0.849 | 0.0183 | -0.164 | 0.950 |
| SpaceTime-T60 | 0.0333 | 0.838 | 0.0239 | -0.097 | 0.944 |
| SpaceTime-T120 | 0.0267 | 0.869 | — | -0.095 | 0.995 |

### 3.2 evo 配对增量

| 对照 | seed | ΔMSE [95% CI] | ΔPearson [95% CI] | 结论 |
|---|---:|---|---|---|
| 原始 WCM | 3072 | **-0.0764** [-0.0922, -0.0599] | **+0.3711** [+0.3160, +0.4277] | evo 显著更优 |
| 原始 WCM | 42 | **-0.0351** [-0.0493, -0.0205] | **+0.2855** [+0.2155, +0.3548] | evo 显著更优 |
| Image-B4 mosaic | 3072 | **+0.0300** [+0.0180, +0.0428] | **-0.3381** [-0.3783, -0.2980] | evo 显著更差 |
| Image-B4 mosaic | 42 | **+0.0223** [+0.0097, +0.0356] | **-0.3230** [-0.3662, -0.2805] | evo 显著更差 |
| SpaceTime-T60 | 3072 | **+0.0419** [+0.0309, +0.0536] | **-0.3272** [-0.3688, -0.2854] | evo 显著更差 |
| SpaceTime-T60 | 42 | **+0.0506** [+0.0401, +0.0615] | **-0.3396** [-0.3828, -0.2961] | evo 显著更差 |

注：evo 只有 seed 1000 一个训练实例，表中两行指“与 WCM 不同 seed 的模型比较”，用于确认结论对 WCM 随机性稳健，并不表示 evo 有双 seed。

### 3.3 centered MSE 对比

| 对照 | evo cMSE | 基线 cMSE | ΔcMSE [95% CI] |
|---|---:|---:|---|
| vs 原始 WCM | 0.0264 | 0.0691 | -0.0427（显著更优） |
| vs Image-B4 | 0.0264 | 0.0183 | +0.0081 [+0.0046, +0.0119]（显著更差） |

---

## 4. 分析与讨论

### 4.1 evo 相比原始 WCM 的增益来源

evo 与原始 WCM 的输入信息接近（都是单帧为主），主要差异是：

1. 骨干更大更强（SigLIP-SO400M + Gemma-3-270M vs ViT-base + CLIP）；
2. 状态通过文本化离散输入，图像分辨率 384（WCM 为 224）；
3. 标签带失败惩罚与 mid-bonus。

在输入信息相近的前提下 evo 明显更好，说明**骨干与预训练质量确实带来收益**。这与"骨干不是唯一瓶颈"并不矛盾：骨干有收益，但收益量级远小于引入历史。

### 4.2 evo 相比历史方案的差距来源

差距不能全部归给骨干，因为 evo 骨干更大却更差。主要候选解释：

1. **缺少历史**：mosaic/SpaceTime 能区分"手刚离开按钮"与"手还没靠近按钮"这类视觉同构状态（此前在结尾 value 突降分析中已观察到此混淆区），单帧模型无法区分；
2. **缺少动作条件**：evo 是 V(s)，WCM 系列是 V(s,a)，后者能区分同状态下不同动作的后果；
3. **标签差异**：evo 目标含 `mid_bonus_scale=0.3`，是人为抬高中段的非单调标签，与 return 的单调结构不一致；这会直接压低与 return 的 Pearson，也会让预测产生正 bias（evo bias +0.221，WCM 系列为负）；
4. **训练数据不同**：evo 0824 用旧 `olddata_1/data_toiletButton_0724`，不含 0814 扩展集；这是对 evo 不利的因素。

### 4.3 预测偏差方向

- evo：mean bias **+0.221**（预测过度乐观，均值 -0.078 vs 目标 -0.299）；
- WCM 系列：bias 为负（过度悲观，-0.07 ~ -0.29）。

若下游用 value 做 advantage 二值化（分位数 top 30%），**全局 bias 不影响排序**；但若直接把 value 当 reward 或阈值使用，evo 的标度不可直接迁移。

---

## 5. 有效性边界

本报告**可以**声明：

- 在 OOD `0722_with_return` 上，`evo_0824_new` 的价值预测质量显著优于原始 WCM，但显著低于 Image-B4 mosaic 与 SpaceTime；
- 历史视觉（4 帧 mosaic / 8 帧 SpaceTime）带来的 OOD 提升，大于把骨干从 1.7 亿升级到 6.7 亿带来的提升。

本报告**不能**声明：

- "WCM 架构优于 pistar06 架构"——两者输入信息、标签定义、训练数据都不同；
- "SpaceTime 优于 pistar06"的纯架构结论；
- 用 MSE 绝对值跨模型比较（标签标尺不同），应以 Pearson / Spearman / paired Δ 为主。

---

## 6. 产物与复现

### 6.1 结果文件

```
outputs/evo_eval_0824_ood/
  evo_ood_metrics.json                 # evo 原始指标 + bootstrap CI
  episode_curves/episode_curves.csv    # 逐帧预测 vs return
  comparisons/ood_evo_vs_baseline_s{3072,42}.json
  comparisons/ood_evo_vs_mosaic_s{3072,42}.json
  comparisons/ood_evo_vs_spacetime_s{3072,42}.json
```

### 6.2 evo 推理复现

```bash
# 1) 复制并转 v3.0（源数据为 v2.1，evo 环境只支持 v3）
cp -r /media/Data/user/huggingface/lerobot/test/data_toiletButton_0722_with_return \
      /media/Data/user/huggingface/lerobot/lerobotevo/evo_eval_ood/data_toiletButton_0722_with_return
cd ~/code_1/evo_rl
HF_HUB_OFFLINE=1 ~/miniforge3/envs/evo/bin/python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=data_toiletButton_0722_with_return \
  --root=/media/Data/user/huggingface/lerobot/lerobotevo/evo_eval_ood

# 2) 推理（脚本见 /tmp/run_evo_eval_ood.py，绕过 task 重名写入 bug）
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
~/miniforge3/envs/evo/bin/python /tmp/run_evo_eval_ood.py

# 3) 配对比较
cd ~/code_1/WCM
for base in baseline_s3072 mosaic_s3072 spacetime_s3072; do
  ~/miniforge3/bin/python scripts/compare_experiments.py \
    outputs/eval_spacetime_threeway/ood/$base outputs/evo_eval_0824_ood \
    --align common --seed 3072 \
    --endpoint-manifest outputs/eval_spacetime_threeway/endpoint_manifests/ood_threeway_s3072.json \
    --output outputs/evo_eval_0824_ood/comparisons/ood_evo_vs_${base}.json
done
```

### 6.3 已知实现问题

evo 的 `lerobot_value_infer.py` 在转换后的 v3 数据集上会因 `meta.tasks` 中同名任务多行而在 `_build_episode_info` 抛 `TypeError`，导致"预测完成但未写回"。本次通过外部 monkeypatch（取首行 task_index）绕过，未修改 evo 仓库源码。

---

## 7. 建议

1. **若要公平比较 pistar06 与 WCM**：在 `data_toiletButton_0814_expanded_with_return` 上重训一个 pistar06（同数据、关掉 mid-bonus、同测试口径），否则只能停留在系统级对比；
2. **若目标是论文强调历史视觉**：本报告已足以支撑"历史/动作条件带来的增益大于骨干扩容"，可作为消融或相关工作对照；
3. **下一步可选**：补 5cut 数据集上的同口径对比（需对 evo 再跑一次推理），确认结论不依赖单一测试集。
