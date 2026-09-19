# SpaceTime 三方对照实验计划

状态：执行设计已实现，待训练机完成统一评估
日期：2026-09-19

## 1. 实验目标

对最终选定的 teacher-free SpaceTime-T8 模型做一次统一三方评估：

1. **原始 WCM baseline**：回答 SpaceTime 相对最原始架构是否构成稳定改进；
2. **Image-B4 mosaic**：回答 SpaceTime 是否超过当前最强、最低复杂度的历史视觉方案；
3. **SpaceTime-T8 direct**：共享 ViT + factorized temporal attention + Perceiver-64，部署时不使用 mosaic。

实验结果同时决定性能结论和论文叙事。任何叙事分支均在看结果前固定，不能根据单 seed 或单指标临时改口径。

## 2. 模型定义

| 角色 | 模型与 checkpoint | 用途 |
|---|---|---|
| 原始 baseline | `wcm_baseline_exp_s{3072,42}/deploy.pt` | 历史原始 WCM，单帧视觉、原始 context/dynamics 架构 |
| 强基线 | `wcm_sparse4_image_b4_exp_s{3072,42}/deploy.pt` | 四帧 `[0,20,40,60]` 输入级 mosaic |
| 候选 | `wcm_spacetime_vit_t8_direct_full_s{3072,42}/deploy.pt` | 八帧 `[0,9,17,26,34,43,51,60]`，teacher-free direct SpaceTime |
| 诊断项 | `wcm_spacetime_vit_t8_full_s{3072,42}/deploy.pt` | 原 align/gate staged SpaceTime，仅用于分析 teacher anchoring，不参与最终方法命名 |

这里的“原始 baseline”明确指 `wcm_baseline_exp`，不是 `wcm_vit_single`。后者只是 SpaceTime 主实验中按 batch 4 重训的单帧下界，不承担“最原始 WCM”叙事。

## 3. 需要先冻结的实验口径

### 3.1 数据与 seed

- 训练数据：`data_toiletButton_0814_expanded_with_return`；
- 测试集：5cut 与 0722 OOD；
- 模型 seed：`3072`、`42`；
- bootstrap seed 必须等于对应模型 seed；
- 每项比较使用 20000 次 episode-level paired bootstrap。

### 3.2 endpoint 对齐

原始 WCM、mosaic 和 SpaceTime 的历史窗口不同，三方指标不能直接拼接各自 summary。所有成对比较必须使用：

```text
scripts.compare_experiments --align common
```

每个 JSON 必须记录共同 endpoint 数量、双方丢弃 endpoint 数量和 target 一致性检查。最终三方汇总只使用三模型共同存在的 endpoint；若两两比较的 endpoint 集不同，必须额外生成三方共同 endpoint 清单后重算，不能直接比较两份两两 JSON 的绝对指标。

正式执行由 `scripts.summarize_spacetime_threeway` 先生成三方共同 endpoint manifest，再把同一 manifest 强制应用到全部比较；这是比独立两两 `--align common` 更严格的最终口径。

### 3.3 主次数据集

- **OOD 是主判定集**：决定方法是否晋级以及采用哪条叙事；
- **5cut 是支持性结果**：用于确认 ID/近分布行为，不得覆盖 OOD 失败；
- validation best 只用于 checkpoint 选择，不作为论文性能结论。

## 4. 公平性与可声明范围

现有 checkpoint 并非完全同训练协议：原始 baseline 为 batch 8、10 epoch、原始 loss；Image-B4 为 batch 4、10 epoch；SpaceTime direct 为 joint 10 epoch + full 3 epoch，并包含 ranking/risk/Q loss。此外，当前 direct SpaceTime 从 Image-B4 checkpoint 做 partial init。

因此本实验分两层解释：

1. **系统级比较**：现有三个最终模型在完全相同测试 endpoint 上谁更好。该结论可以直接给出；
2. **架构因果归因**：不能只凭现有 checkpoint 宣称全部收益来自 SpaceTime 模块。若论文需要“仅改变架构”的强结论，后续必须补做 matched-training 三臂实验，或至少增加从对应原始 WCM checkpoint 初始化的 SpaceTime 对照和等训练量 continuation control。

本轮计划先完成系统级三方比较。正文措辞使用“SpaceTime 系统相对原始 WCM 改进”，不使用“纯 SpaceTime 模块带来全部增益”。

## 5. 输出文件规划

统一评估根目录建议为：

```text
outputs/eval_spacetime_threeway/
  5cut/{baseline,mosaic,spacetime,staged}_s{3072,42}/
  ood/{baseline,mosaic,spacetime,staged}_s{3072,42}/
  comparisons/
```

必须生成以下 paired comparison：

```text
{5cut,ood}_spacetime_vs_baseline_s{3072,42}.json
{5cut,ood}_spacetime_vs_mosaic_s{3072,42}.json
{5cut,ood}_spacetime_vs_staged_s{3072,42}.json
{5cut,ood}_mosaic_vs_baseline_s{3072,42}.json
```

前三组分别回答整体提升、强基线超越和 teacher anchoring；最后一组固定历史链条，便于量化 baseline → mosaic → SpaceTime 每一步的增量。

## 6. 指标与判定顺序

每个数据集、每个 seed 报告：

- MSE 与 candidate-minus-baseline paired 95% CI；
- Pearson 与 paired 95% CI；
- mean bias；
- centered MSE，定义为 `MSE - mean_bias²`；
- 共同 episode 数与 endpoint 数；
- 参数量、推理输入帧数、视觉 token 数和推理延迟，作为效率说明，不参与主门禁。

判定顺序固定为：

1. 先判 SpaceTime vs 原始 WCM；
2. 再判 SpaceTime vs mosaic；
3. 最后查看 SpaceTime direct vs staged，只解释 teacher anchoring，不改变前两项输赢。

### 6.1 “优于原始 WCM”门槛

OOD 上同时满足：

- 两个 seed 的 MSE 或 centered MSE 同向改善；
- 至少一个 seed 的对应 paired 95% CI 完全小于 0，另一个不得显著变差；
- 两个 seed 的 Pearson 均不下降，且至少一个 seed 的 paired CI 下界大于 0；
- 改善不能仅由 mean bias 收窄解释。

### 6.2 “优于 mosaic”门槛

使用更严格的强基线门禁：

- 两个 seed 的 OOD MSE 或 centered MSE 同向改善；
- 至少一个 seed 的 paired 95% CI 完全小于 0；
- 两个 seed 的 OOD Pearson 均不低于 mosaic；
- 若 raw MSE 改善但 centered MSE 未改善，则两个 seed 的 Pearson 必须同向提升，且至少一个 paired CI 下界大于 0；
- 任何 seed 出现 Pearson 显著下降，均判为未超过 mosaic。

“单 seed 胜出”“均值更好但 seed 方向相反”或“只改善 bias”一律记为未超过，不进入 mosaic 主叙事。

## 7. 预注册叙事分支

### 分支 A：SpaceTime 同时超过原始 WCM 与 mosaic

核心叙事：

> 原始 WCM 缺少显式长历史视觉建模；mosaic 证明稀疏视觉历史有效，但依赖人工空间拼接和单帧分辨率压缩。SpaceTime 将原始视频历史直接编码为固定 64 token，在不依赖 mosaic 部署输入的情况下进一步超过 mosaic。

此时 mosaic 应加入主叙事，作为关键中间发现和强基线。方法演进写为：

```text
原始 WCM -> mosaic 验证历史有效 -> SpaceTime 学习时空压缩并超过 mosaic
```

允许的主张是“SpaceTime 是对 mosaic 历史编码的学习化替代并取得进一步收益”。若仍沿用 mosaic checkpoint 初始化，必须同时说明训练初始化来源，不能写成完全未使用 mosaic 知识。

### 分支 B：SpaceTime 超过原始 WCM，但未超过 mosaic

核心叙事：

> SpaceTime 是原始 WCM 的直接历史视觉扩展；它显著改善原始单帧架构，并用固定 64 token 建模八帧视频历史。输入级 mosaic 仍是更强或更具性价比的工程基线，但不是最终部署架构的一部分。

此时论文主叙事固定为“原始 WCM → SpaceTime”。mosaic 不进入标题、摘要、方法动机、贡献点、主方法演进图或创新性论证，只在实验部分作为强对照和边界条件完整报告。不得宣称 SpaceTime 优于 mosaic 或优于所有历史输入方案；若差异不显著，应准确写为“相对 mosaic 未检出显著提升”，不能写成“与 mosaic 等效”。

这一规则按 §6 的双 seed OOD 门禁执行，而不是按单个 seed、5cut 结果或点估计决定。只要 SpaceTime 未通过“优于 mosaic”门槛，即使数值接近，也不把 mosaic 纳入方法叙事；但不能从实验表、附录或限制讨论中删除该强基线。

由于当前 direct SpaceTime 从 mosaic checkpoint 初始化，正式写作应采用“部署架构直接改进原始 WCM”的表述。若要使用“训练上也完全独立于 mosaic”的表述，必须先补做原始 WCM 初始化版本。

### 分支 C：SpaceTime 与原始 WCM 打平或更差

不形成架构改进叙事。SpaceTime 只保留为负结果或压缩可行性实验；主方案回退到 mosaic，停止扩大 SpaceTime 线。

### 分支 D：seed 或指标结论冲突

结论写为“不稳定、证据不足”，不选择 A/B 中任何强叙事。优先增加预注册 seed，而不是调整门禁、挑 checkpoint 或只报均值。

## 8. 结果表模板

### 8.1 三方绝对指标

| 数据集 | seed | 原始 WCM MSE / Pearson | mosaic MSE / Pearson | SpaceTime MSE / Pearson |
|---|---:|---:|---:|---:|
| 5cut | 3072 | 待填 | 待填 | 待填 |
| 5cut | 42 | 待填 | 待填 | 待填 |
| OOD | 3072 | 待填 | 待填 | 待填 |
| OOD | 42 | 待填 | 待填 | 待填 |

### 8.2 SpaceTime paired 增量

| 对照 | 数据集 | seed | ΔMSE [95% CI] | Δcentered MSE | ΔPearson [95% CI] | 判定 |
|---|---|---:|---|---:|---|---|
| 原始 WCM | OOD | 3072 | 待填 | 待填 | 待填 | 待填 |
| 原始 WCM | OOD | 42 | 待填 | 待填 | 待填 | 待填 |
| mosaic | OOD | 3072 | 待填 | 待填 | 待填 | 待填 |
| mosaic | OOD | 42 | 待填 | 待填 | 待填 | 待填 |

## 9. 执行前检查清单

- [x] 自动确认两 seed 的三个主 checkpoint 和 staged 诊断 checkpoint 均存在且配置可追溯；
- [x] 自动确认 SpaceTime deploy checkpoint 中 `gate=1`、teacher 关闭、`history_mosaic=false`；
- [x] 由统一入口固定 5cut/OOD 数据路径，并在评估 summary 中记录数据 root、revision 和可用时的 Hugging Face fingerprint；
- [x] 自动生成每个数据集、每个 seed 的三方共同 endpoint 清单；
- [x] 对所有比较显式传递模型 seed；
- [ ] 跑完 paired bootstrap 后才填写结果表；
- [x] 汇总器先按 §6 自动判门禁，再唯一选择 §7 叙事分支；
- [x] 汇总器只使用 OOD 门禁决定分支，5cut 不覆盖 OOD；
- [x] direct vs staged 仅作为诊断输出，不参与主判定。

## 10. 执行入口与产物

统一入口为：

```bash
bash 30_run_spacetime_threeway.sh
```

入口只复用已训练 checkpoint，不启动新训练。它依次完成 checkpoint 审计、5cut/OOD 四角色评估、三方共同 endpoint 对齐、20000 次 episode-level paired bootstrap、自动门禁和 A/B/C/D 分支选择。主要产物为：

```text
outputs/eval_spacetime_threeway/
  checkpoint_inventory.json
  endpoint_manifests/{5cut,ood}_threeway_s{3072,42}.json
  comparisons/{5cut,ood}_{comparison}_s{3072,42}.json
  threeway_results.json
  threeway_results.md
```

默认路径可通过 `WCM_CHECKPOINT_ROOT`、`WCM_5CUT_ROOT`、`WCM_OOD_ROOT` 和 `WCM_THREEWAY_EVAL_ROOT` 覆盖。结果汇总同时记录参数量、输入帧数、视觉 token 数和同一评估协议下的模型 forward 毫秒/样本。

## 11. 本计划之外

matched-training 三臂因果消融仍作为后续增强实验，不与本轮系统级比较混在一起。本轮结果只支持系统级比较，不改变 §4 中对架构因果归因的限制。
