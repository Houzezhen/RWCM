# pistar06 与 WCM 价值函数对比调研

> 调研对象：evo_rl（lerobot fork）中的 `values/pistar06` 价值函数 与 WCM（world critic）价值函数。
> 调研日期：2026-08-17

---

## 1. 背景：pistar06 是什么

**π\*0.6**（Physical Intelligence，2025.11）是 pi0.6 的 RL 版本，核心方法是 **RECAP**（RL with Experience and Corrections via Advantage-conditioned Policies）：

1. 用离线数据训练一个价值函数（预测"距离成功还有多远"）；
2. 给每条轨迹计算 advantage；
3. 将 advantage 二值化为 `Advantage: positive / negative` 文本标签，作为任务文本条件训练策略（advantage-conditioned policy, ACP）。

价值函数是整套流程的地基，价值的好坏直接影响策略改进的效果。

---

## 2. evo_rl 中 pistar06 价值函数实现

代码位置：`/home/user/code_1/evo_rl/src/lerobot/values/pistar06/`

### 2.1 输入 / 输出

| 项 | 说明 |
|---|---|
| 输入 | 单帧图像（N 个摄像头）+ 语言 prompt（Gemma-3 tokenizer） |
| 状态 | **不是张量输入**。`observation.state` 离散化为 256 个 bin 的整数串，拼进文本 prompt（`Task: ..., State: ...\nValue: `） |
| 历史帧 / 动作 | 无。`observation_delta_indices` / `action_delta_indices` / `reward_delta_indices` 均返回 None |
| 输出 | 201 bins 分布头（soft target CE 损失），推理时对 bin 中心求期望 → **单个标量** |

### 2.2 训练标签公式（核心）

`compute_normalized_value_targets()`（`modeling_pistar06.py` L85-147）：

```
target = clip(-剩余步数 / (task_max + c_fail), -1, 0)
失败轨迹额外减去 c_fail，使成功/失败值域不重叠
```

具体实现：

```
remaining_steps = 回合长度 - 当前帧号 - 1
c_fail = task_max * c_fail_coef        # 默认 c_fail_coef = 1.0
g = -remaining_steps
if 失败: g -= c_fail                    # 失败轨迹整体下移
denom = task_max + c_fail
target = clip(g / denom, -1, 0)
```

含义：
- **成功轨迹**：最后一帧 target = 0，第一帧 ≈ -0.5（默认 c_fail_coef=1 时分母为 2·task_max），线性递增；
- **失败轨迹**：整体再减 c_fail，与成功轨迹值域不重叠（约落在 [-1, -0.5]）；
- `task_max` 是每个任务下所有 episode 的最大长度，用于跨 episode 归一化；
- 可选 `mid_bonus`：对成功轨迹叠加以轨迹中段为峰值的高斯 bonus，用于"关键动作在中段"的任务。

**没有 return 列、没有 reward 列**，target 完全由 episode 元数据在线推导。

### 2.3 归一化

- 价值输出：固定 **[-1.0, 0.0]**（`bin_min=-1.0, bin_max=0.0`，配置不可变）；
- 图像：模型内部 /255 + SigLIP mean/std；
- 状态：分位数归一化后离散化进文本。

### 2.4 网络结构

- 视觉骨干：**SigLIP-SO400M**（patch14, 384×384）；
- 语言骨干：**Gemma-3-270M**（文本 backbone，masked mean pooling）；
- 融合：图像 / 语言 projector → 512 维 + GELU + Dropout → LayerNorm(1024) → `value_head`（Linear 1024→512 + Linear 512→201 bins）；
- 多相机特征按 mask 求平均；
- 可冻结视觉 / 语言骨干、gradient checkpointing、bf16；
- 参数量：约 **6.7 亿**（SigLIP ~4 亿 + Gemma-3 ~2.7 亿），冻结骨干后可训练参数约百万级。

### 2.5 使用方式（离线 ACP 流水线）

价值函数**不参与在线 RL 的 critic**（SAC/HILSerl learner 不使用它），是纯离线标注器：

```
lerobot-value-train   # 训练价值模型
        ↓
lerobot-value-infer   # 逐帧推理，写回 complementary_info.value
        ↓
反推稠密 reward: rewards[i] = targets[i] - targets[i+1]（末帧 = targets[i]）
        ↓
n-step advantage（默认 n=50, γ=1）: adv = Σrewards + bootstrap - V(s)
        ↓
按任务分位数二值化（默认 positive_ratio=0.3 → 每任务 top 30% 为正）
        ↓
lerobot-train  # ACP hook 把 "Advantage: positive/negative" 拼进任务文本
```

---

## 3. WCM 价值函数

代码位置：`/home/user/code_1/WCM/world_critic/`

### 3.1 输入 / 输出

| 项 | 说明 |
|---|---|
| 输入 | 最近 3 帧历史图像（`history_size=3`）+ 语言指令（CLIP text encoder）+ **动作**（action-conditioned） |
| 状态 | 不直接使用 `observation.state` |
| 输出 | 线性回归头（MLP，无激活），输出 **1 个标量** = 对 return 的回归预测 |

### 3.2 训练标签：数据集 return 列

需要先运行 `scripts/add_returns_to_lerobot.py` 把 return 列写进数据集。pi\*0.6 式 return 定义：

```
中间步:  r_t = -step_scale（默认 -1）
成功终点: r_T = 0
失败终点: r_T = -C_fail（--failure-penalty，默认 = 每任务最长回合长度）
```

即成功轨迹的 return 从 `-剩余步数` 线性升到 0；失败轨迹终点为负惩罚。

### 3.3 归一化选项（--normalization）

| 选项 | 区间 | 说明 |
|---|---|---|
| `task_max`（默认） | [-1, 0] | 逐任务归一化：`clip(return / 该任务最长长度, -1, 0)`，与 pistar06 同标尺 |
| `global_minmax` | [-1, 1] | 整个数据集全局缩放：`-1 + 2·(raw-min)/(max-min)`，对齐旧版 RLDS→H5 写入器约定 |
| `none` | 原始控制步数 | 不缩放 |

### 3.4 网络结构

- 视觉骨干：**ViT-base**（google/vit-base-patch16-224-in21k）；
- 语言骨干：**CLIP**（openai/clip-vit-base-patch32）；
- 动作编码器 + 融合 + `value_head`（线性 MLP）；
- 参数量：约 1 亿量级，远小于 pistar06。

### 3.5 使用方式

- 离线评估：`3_run_eval.sh` 给数据集每个回合打分，输出价值曲线（summary.json + episode_curves/）；
- 论文定位：作为 RL 的价值函数（critic），但论文 RL 训练代码尚未开源。

---

## 4. 核心对比

| 维度 | evo pistar06 | WCM |
|---|---|---|
| 输入 | 单帧图 + 文本化状态 + 语言 | 3 帧历史图 + 语言 + **动作** |
| 动作条件 | 无，纯状态价值 V(s) | 有，动作条件价值 V(s,a)（更像 Q 函数） |
| 输出头 | 201 bins 分布头（softmax CE） | 线性回归头（MSE） |
| 输出区间 | 固定 [-1, 0] | task_max: [-1, 0]；global_minmax: [-1, 1] |
| 标签来源 | 在线推导，无需 return 列 | 数据集 return 列（需先跑 add_returns） |
| 标签公式 | `-剩余步数/(task_max+c_fail)`，失败再减 c_fail | `-剩余步数`（task_max 归一化后等价） |
| 网络 | SigLIP-SO400M + Gemma-3-270M（~6.7 亿） | ViT-base + CLIP（~1 亿量级） |
| 预处理 | 不需要 | 需要 add_returns 生成 return 列 |
| 价值语义 | "离成功终点还有多远"（状态导向） | 同上，但**给定具体动作**来估计 |

**核心区别两条：**

1. **动作条件**：WCM 输入动作，输出"沿着这个动作走能得多少回报"；pistar06 只看状态，输出"该状态的平均价值"。WCM 能区分同一状态下不同动作的好坏。
2. **标签构造**：公式等价，但 pistar06 在线算、不碰数据集；WCM 依赖预写好的 return 列。

其余为工程差异：区间、模型规模、输出头形式。

---

## 5. 用 WCM 的 task_max value 接入 evo ACP 流水线

### 5.1 可行性结论

**可行。** 用 `--normalization task_max` 训练 WCM 后，其价值标尺与 pistar06 完全同源（都是"负的剩余步数占比"）：

| | 标签公式 | 成功轨迹值域 |
|---|---|---|
| evo pistar06 | `-剩余步数/(task_max + c_fail)` | [-1/2, 0]（c_fail=1×task_max 时） |
| WCM task_max | `-剩余步数/task_max`（clip 到 [-1, 0]） | **[-1, 0]** |

数值差约 2 倍（分母不同），但 **advantage 二值化用任务内分位数（top 30%），对绝对标度不敏感**，排序不变，`acp_indicator` 结果不受影响。

### 5.2 完整链路

1. **数据**：用 `--normalization task_max` 重跑 `add_returns_to_lerobot.py` 生成带 return 列的数据集（task_max 是逐任务归一化，多任务也不混标尺，正是 pistar06 的做法）；
2. **训练**：用该数据集训 WCM；
3. **推理回写**：写适配脚本，用 WCM checkpoint 对数据集逐帧推理（按 3 帧窗口 + 取轨迹动作），写入 `complementary_info.value`；
4. **advantage + acp**：复用 evo `lerobot_value_infer.py` 的 advantage 计算（L240-277）和分位数二值化（L280-319），写回 `complementary_info.advantage` 和 `complementary_info.acp_indicator`；
5. **策略训练**：`lerobot_train` 的 ACP hook 自动读 `acp_indicator`，下游零改动。

### 5.3 注意事项

1. **2 倍标度差异**：不要拿 value 绝对值当 reward 用；只用于分位数二值化则无影响；
2. **动作依赖**：WCM 推理必须提供动作。给已有轨迹标注没问题（action 列在），纯状态在线评估不适用；
3. **适配脚本**：`lerobot_value_infer` 推理部分绑死 pistar06 模型，需替换为 WCM 推理，advantage/二值化部分可直接复用；
4. **数据依赖**：新增轨迹（含失败/干预）需重新跑 add_returns 并增量训练；pistar06 零预处理、随时可训。

### 5.4 建议

- 目标是尽快跑通 evo ACP 策略训练 → 直接用其自带的 pistar06 更省事（数据集零改动）；
- 觉得单帧状态价值不够准、想用动作条件 + 多帧历史做更精细的轨迹打分/筛选 → 切换 WCM，收益是更好的排序质量，代价是 return 预处理 + 训练 + 推理适配脚本。

---

## 参考文件

- evo pistar06：`/home/user/code_1/evo_rl/src/lerobot/values/pistar06/`（configuration / modeling / processor）
- evo 推理与 ACP：`/home/user/code_1/evo_rl/src/lerobot/scripts/lerobot_value_infer.py`、`lerobot_value_train.py`、`/home/user/code_1/evo_rl/src/lerobot/rl/acp_hook.py`、`acp_tags.py`
- WCM：`/home/user/code_1/WCM/world_critic/`、`/home/user/code_1/WCM/scripts/add_returns_to_lerobot.py`
- π\*0.6 论文：arXiv:2511.14759（Physical Intelligence）
