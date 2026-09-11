# 跨帧 WCM 离线强化学习方案

更新：2026-09-11

## 0. 方案结论

保留原始 WCM 作为严格 baseline，新增一个以输入端历史建模为核心的
`WCM-Sparse4`，再视结果决定是否实现 VGGT 风格的 patch/token 跨帧融合。

核心假设不是“寄存器能修复 ViT 注意力”，而是：

> 机械臂的局部画面存在运动状态歧义；稀疏历史和 proprioception 能提供前后位置、运动方向和接触阶段信息，从而改善离线轨迹评分。

用户在 Pi05 LoRA 任务中已经观察到强先验：

- 当前帧、`-20`、`-40`、`-60` 四个时间点拼接历史；
- 图像按 2×2 拼图输入；
- 7 维 state 拼接为 28 维；
- 开始阶段历史不足时复制同一帧；
- 机械臂前后犹豫/不动显著减少，模型能识别自身前后位置。

这个结果说明，优先研究“如何把跨帧证据提供给 critic”比继续增加 register 更合理。

## 1. 研究问题

主问题：

> 在延迟失败、单帧观测存在状态歧义的机器人任务上，稀疏跨帧视觉和 proprioception 是否能改善 WCM 的离线价值/动作风险评分，并提升下游离线策略？

论文不宣称模型能从完全相同的前期画面预知随机失败。若失败只在末端动作按偏后才产生可见证据，则前期 value 应解释为条件期望，而不是单条轨迹的确定结果。

## 2. 严格 baseline

### 2.1 原始 WCM（O-WCM）

- 当前仓库的原始 WCM；
- `history_size=3`，连续局部历史；
- 每帧独立 ViT 编码，再由 latent-level causal trunk 融合；
- value 分支读取视觉历史和指令，不读取 action；
- dynamics 分支读取 action，只做 next-state 辅助预测；
- 离线训练，使用固定 episode split；
- 主结果使用相同固定 epoch，不混用 `best.pt` 与 `deploy.pt`。

O-WCM 是必要的原始 baseline，但它不能单独代表标准 offline RL critic：要做策略改进，还需要 action-conditioned `Q(s,a)` 或 `A(s,a)`。

## 3. 第一阶段：WCM-Sparse4

这是最小改动、最接近 Pi05 成功经验的候选，不先加入复杂新模块。

### 3.1 稀疏历史采样

以当前 endpoint `t` 为锚点，采样：

```text
[t, t-20, t-40, t-60]
```

所有索引必须在同一 episode 内按 `frame_index` 采样，不能跨 episode 补帧。起始阶段不足的帧，复制 episode 边界处同一可用帧，严格复现 Pi05 的复制规则，并在数据构建脚本中固定记录；绝不使用未来帧补齐当前样本。

### 3.2 图像输入

每个 camera 单独构造一个 2×2 mosaic，固定布局：

```text
左上：t       右上：t-20
左下：t-40    右下：t-60
```

如果有 primary 和 wrist 两个 camera，则得到两个 mosaic，保留原有多 camera pooling。拼图布局是固定的时间槽编码；不额外加入随机排列，避免模型失去时间顺序。

第一阶段不改 ViT 内部 register，不使用 late register，也不引入未来帧。

### 3.3 proprioception 输入

每个采样时刻的 7 维 state 按相同顺序拼接：

```text
[state_t, state_t-20, state_t-40, state_t-60] -> 28 维
```

state 必须只使用当前及过去观测，并进行训练集统计量归一化。建议同时加入差分特征作为消融，而不是默认加入：

```text
delta_20 = state_t - state_t-20
delta_40 = state_t - state_t-40
```

这样可以区分“绝对位置收益”和“运动方向收益”。

### 3.4 模型接口建议

第一阶段使用一个 `TemporalInputAdapter`：

```text
4 个稀疏图像帧 -> mosaic / shared ViT -> visual temporal token
4 个 7D state -> 28D MLP -> proprio temporal token
visual + proprio + instruction -> WCM context/value
```

为了保持与原始 WCM 的可比性：

- value head 仍然预测当前 endpoint 的 value；
- dynamics 先保留原有 next-state 辅助任务；
- 不把 action 偷渡进 value 分支；
- 所有新增输入投影参数使用普通 base learning rate，不使用 10× register LR。

如果直接复用现有 WCM 的完整序列接口代价过高，可以先实现 value-only 的 scorer 版本，再补齐 Q/dynamics 分支；但论文主实验必须明确记录两者差异。

## 4. 第二阶段：VGGT-inspired Cross-Frame WCM

只有当 `WCM-Sparse4` 相对 O-WCM 在多个 seed 和 OOD 上有稳定收益，才进入这一阶段。

### 4.1 借鉴范围

VGGT 最值得借鉴的是跨视角/跨帧 token 的联合交互，不是完整几何重建头。机器人按钮任务没有可靠的 3D 几何标签，因此不直接复制 VGGT 的 camera/geometry heads。

### 4.2 推荐结构

```text
每帧图像 -> 共享 ViT patch tokens X_t
                         |
当前帧 query Q_t 对 [X_t, X_t-20, X_t-40, X_t-60] 做 causal cross-frame attention
                         |
跨帧增强的当前 state token
                         |
proprio / instruction fusion -> WCM trunk -> V/Q/risk
```

与简单 temporal Transformer 的区别是：跨帧交互发生在 patch/token 层，而不是等每帧先压成一个 pooled latent 后才交互。这样模型可以学习：

- 末端执行器相对按钮的位移；
- 接触前后的局部变化；
- 机械臂是接近还是收回；
- 当前动作是否造成预期的视觉变化。

### 4.3 因果性

主实验使用 causal history，只允许看到 `t` 及以前的帧。即使当前目标是离线打分，这个约束能避免论文把未来信息泄漏误称为实时能力。

可以另做一个 bidirectional offline upper bound，但必须单独命名为 retrospective scorer，不能与 causal 结果混合。

## 5. 面向 offline RL 的 value/Q 设计

原始 WCM 的 action-free value 可以作为 baseline scorer，但不能直接完成一般的离线策略改进。

建议最终模型同时输出：

```text
V(s_t)       当前状态的期望回报
Q(s_t, a_t)  执行动作后的期望回报
RISK(s_t,a)  未来 H 步失败概率
```

并计算：

```text
A(s_t,a_t) = Q(s_t,a_t) - V(s_t)
```

动作读取位置必须在 Q/risk 分支，不能改变 baseline 的 action-free V 定义。

### 5.1 任务标签

不要要求模型在完全相同的前期画面上确定性预测最终成功/失败。

- `V`：继续使用 return，解释为行为策略下的条件期望；
- `Q`：读取当前动作或短动作序列；
- `risk`：在按压/接触阶段监督 centered press vs off-center press；
- 发生可观察偏移前的样本可标记为 uncertain，不强行施加成败标签。

如果暂时只有现有 return 数据，先做 V/Sparse4 作为方法验证；Q/risk 需要确认 action、末端位置和成功/失败标签的时间对齐后再训练。

## 6. 实验矩阵

### 阶段 A：输入历史消融

| 组 | 图像 | state | 目的 |
|---|---|---|---|
| O-WCM | 连续 3 帧 | 原始输入 | 原始 baseline |
| S4-Image | 4 帧 2×2 mosaic | 原始 state | 测视觉历史 |
| S4-State | 当前图像 | 4×7=28 维 | 测 proprio 历史 |
| S4-Full | 4 帧 mosaic | 28 维 state | 复现 Pi05 输入经验 |
| S4-Delta | 4 帧 mosaic | state + 差分 | 测运动量信息 |

### 阶段 B：跨帧表示消融

| 组 | 跨帧位置 | 目的 |
|---|---|---|
| S4-Full | 输入拼图/MLP | 简单强 baseline |
| Token-Causal | ViT patch/token | VGGT-inspired 主方法 |
| Token-Bidir | ViT patch/token 双向 | 离线 retrospective upper bound |

### 阶段 C：下游 offline RL

| 组 | critic | 下游 |
|---|---|---|
| O-WCM | 原始 V | 原有离线标签/策略训练 |
| S4-Full-V | 稀疏历史 V | 同一策略训练协议 |
| Token-Causal-Q | 跨帧 Q/V/risk | advantage-weighted offline RL |

所有结构必须使用至少 3 个、最好 5 个训练 seed；同一 seed 的 baseline/candidate 使用同一 episode split、训练步数和 checkpoint 规则。

## 7. 评价指标

不要只用逐窗口 MSE。

### 7.1 离线 scorer

- episode-level MSE / MAE；
- Pearson / Spearman / Kendall 排序；
- calibration bias；
- OOD episode bootstrap CI；
- 接触阶段 failure-risk AUROC/AUPRC；
- 最早可观察分歧后的提前预警帧数。

### 7.2 下游策略

- episode success rate；
- button contact success；
- intervention/retry rate；
- 平均动作偏差；
- OOD 环境成功率；
- 与原始 WCM 生成的 advantage 标签相比的策略改进。

## 8. 数据与有效性门禁

1. 按 episode 划分 train/val/test，禁止相邻窗口跨 split 泄漏。
2. 每个采集域尽量同时包含成功和失败，不能让数据来源直接代理标签。
3. 5cut 全成功集只能评估成功轨迹上的回归，不能作为失败识别主证据。
4. 记录每个样本的实际 frame index 和四个历史索引，抽样可视化 mosaic，确认没有越界或未来帧。
5. 对 S4-Full 做历史打乱、mosaic 槽位打乱和 state 打乱消融，验证收益来自时间关系，而不是额外像素/维度。
6. 对 Token-Causal 做单次 encoder、梯度、shape 和无未来访问 smoke test。
7. 所有结果保留 per-episode JSON；窗口级结果只用于总体误差，置信区间以 episode 为 bootstrap 单位。

## 9. 停止规则

进入 Token-Causal 前必须满足：

- S4-Full 在至少 3 个 seed 中大多数同向改善；
- OOD episode-level 指标改善不能只来自 bias 校准；
- 历史打乱消融会明显损害收益；
- 下游策略至少在一个独立任务或 OOD 域上改善。

若 S4-Full 本身没有稳定收益，就停止 VGGT-inspired 结构线，结论写成“该任务的跨帧信息不足以转化为稳定 value 增益”。

## 10. 实现顺序

1. 新增稀疏历史索引与 2×2 mosaic 数据适配器。
2. 新增 28D state encoder，完成 shape/归一化/边界测试。
3. 实现 O-WCM、S4-Image、S4-State、S4-Full 的统一训练配置。
4. 在固定两 seed 上先做 smoke + 5cut/OOD paired evaluation。
5. 若 S4-Full 通过门禁，再实现 patch-level causal cross-frame attention。
6. 最后增加 action-conditioned Q/risk，并用 advantage-weighted offline RL 做下游验证。

## 11. 预期论文贡献

最稳妥的论文贡献表述是：

1. 发现并刻画延迟接触失败任务中的单帧状态歧义；
2. 提出一种稀疏跨帧视觉/proprioception 输入的离线 WCM scorer；
3. 比较输入拼图、latent 融合和 patch-level cross-frame attention；
4. 证明跨帧表示是否有效取决于是否包含动作/接触证据，而不是增加 register 容量；
5. 在下游 offline RL 中验证 value/Q/risk 标签是否转化为真实策略收益。

如果最终只有单一按钮任务，论文应定位为受控案例研究；要支撑更强的泛化结论，至少需要多个具有延迟接触失败的任务或环境。
