# WCM → SpaceTime 叙事图表说明

生成脚本：`scripts/make_wcm_to_spacetime_figures.py`（数据源与挑集规则见脚本 docstring）
日期：2026-09-22

## 数据来源

| 角色 | 模型 | checkpoint 来源 |
|---|---|---|
| baseline | 原始 WCM（单帧视觉，batch 8 / 10 epoch / 原始 loss） | `outputs/eval_spacetime_threeway/{ood,5cut}/baseline_s{3072,42}` |
| SpaceTime-T60 | 8 帧 [0,9,17,26,34,43,51,60]，teacher-free direct | `outputs/eval_spacetime_t120_pair/.../t60_s{3072,42}` |
| SpaceTime-T120 | 8 帧 [0,17,34,51,69,86,103,120]，其余同上 | `outputs/eval_spacetime_t120_pair/.../t120_s{3072,42}` |

- 数据集：OOD = `data_toiletButton_0722_with_return`（125 集）；所有曲线均为共同 episode（三模型 × 双 seed 交集）上的逐帧预测。
- 每集指标（MSE / Pearson）在 episode 内计算，非全集拼接。
- 排除 mosaic：按叙事决策（分支 B 口径），mosaic 结果仅在论文实验部分作为强对照报告，不进入方法叙事图。

## 图表清单

### fig3_architecture.png — 方法图
8 帧历史 × 2 相机 → 共享 ViT 逐帧编码 → factorized temporal attention → Perceiver（1568 → 64 token）→ 原有 WCM trunk（value / risk / Q）。要点：token 预算固定 64，与历史跨度无关；部署输入不含 mosaic。

### fig1_ood_summary.png — OOD 主结果
三模型 × 双 seed 的每集 MSE（左）与 Pearson（右）箱线图（实心=seed 3072，空心=seed 42，无离群点）。叙事：单帧 WCM OOD 严重退化 → 时空编码大幅恢复 → 跨度 120 进一步改善。

### fig2_episode_curves.png — 定性曲线
自动挑选的 2 条 OOD 代表集（ep 57、119，挑选规则：seed 3072 上 T120 相对 baseline 的每集 Pearson 增益最大）。每图：三条模型预测线 + GT return（黑虚线）。⚠️ 挑集规则为"展示增益最大"，用于定性说明时需注明非随机抽取。

### fig4_paired_delta.png — 配对增量分布
每集 ΔMSE（左）/ ΔPearson（右）相对 baseline 的直方图，T60（蓝）/ T120（红）分色，双 seed 叠加。支撑"改善是分布整体移动而非个别集拉动"。

### fig5_span_t60_vs_t120.png — 跨度消融
T60 vs T120 每集散点（对角虚线=打平）：左 MSE（点在对角线下方=T120 更优）、右 Pearson（上方=更优），双 seed。对应 paired bootstrap：OOD ΔMSE -0.0069/-0.0040（CI 全负），ΔPearson +0.030（显著）/+0.004（不显著）。

## 引用约束（写作时注意）

1. **措辞**：只能写"SpaceTime 系统相对原始 WCM 改进"，不能写"纯 SpaceTime 模块带来全部增益"（direct 版从 Image-B4 checkpoint partial init，见三方对照计划 §4）。
2. **T120 预注册状态**：T120 为 t60 落分支 D 后的事后变体，双 seed 已过 mosaic 门禁，但 seed 1337 确认前，"(ours)"主张建议保留双 seed 措辞（消融记录 §13.7）。
3. fig2 的挑选规则必须在图注或正文披露。
