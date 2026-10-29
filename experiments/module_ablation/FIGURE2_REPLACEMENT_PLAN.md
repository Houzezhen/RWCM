# Figure 2 a/b/c 模块替换消融

日期：2026-10-27。以 `paper/stmemory/figure_src/fig02_architecture.tex` 和当前 `world_critic/model.py` 为依据。

## 当前执行版本（覆盖下方早期两阶段描述）

2026-10-27 已改并行：保留 a 正在运行的 Python 进程，停止原串行父调度后，同时启动 b/c。当前监督入口为 `run_scripts/module_replacements_parallel.sh`，tmux 会话 `stmemory-abc-parallel`；三项各自日志仍在原变体目录，监督日志为 `parallel.log`，退出状态为 `parallel_exit_code.txt`。三个真实训练进程已确认运行，5880 Ada 的 GPU 利用率达到99%，显存约17GB（48GB总量）。不重新启动 a，不重复执行 b/c。

按用户最终要求，三个替换版只训练 joint 10 epoch、seed42，不做 full。完整对照直接复用论文最终 `wcm_spacetime_vit_t8_direct_full_t120_s42` 的 OOD CSV，不评估 joint-only 对照、不重训完整模型。新变体取最后一轮，论文完整模型取历史 joint10+full3 验证最优权重，报告明确记录训练量和选模不同。

已实现两个独立配置开关：`spacetime_original_encoding`（b）与 `spacetime_original_readout`（c）；a 使用三帧 `[0,1,2]`。生成器为 `scripts/prepare_module_replacements.py`，入口为 `run_scripts/module_replacements.sh`。46 项现有测试通过，实际 GPU 路径预检在正式训练前逐项执行。

服务器 tmux：`stmemory-abc`。输出根目录 `/media/Data/user/WCM_outputs/module_replacement_20261027/`，`batch.log` 为总日志，变体子目录的 `train.log` 为训练日志；`completed_at.txt` 与 `exit_code.txt` 标明最终完成状态。只保留各模型 `checkpoints/last.pt` 和 `deploy.pt`。运行设备 RTX 5880 Ada。本文档后续两阶段与尚未实现描述为早期方案，执行以本节为准。

## 目标和对照

本轮回答：把 Figure 2 的一个模块换回原始 WCM 对应处理方式后，完整系统的表现如何变化。每次只替换一个模块，其余采用完整 STMemory-120 的结构与训练配置。

沿用用户确定的执行范围：只跑 seed 42；完整 STMemory-120 复用历史权重和预测，不重新训练；双相机固定，不做单相机消融。新任务另建 `module_replacement` 目录，不能覆盖已完成的参数探索。

原始 WCM 的实际视觉路径为：每帧每相机经过 ViT，取 CLS 并加 camera embedding；`pool_views()` 用原有 learned-query view attention 得到每时刻一个向量；`encode_context()` 执行语言融合和因果 context trunk，输出各时刻表示。原始配置使用连续三帧，不是无历史单帧模型。

## 四行主表

| 表格名称 / 任务代号 | a：采样 | b：视觉与时间编码 | c：汇聚与下游接口 |
| --- | --- | --- | --- |
| 完整 STMemory-120 / historical_full | 8 帧，跨度 120 | 每帧完整 ViT token + patch 级时序注意力 | 64 learned queries，送 grouped context |
| 原始采样替换 a / replace_a_sampling | **连续 3 帧 `[0,1,2]`** | 保留 STMemory 的 token 编码和时序层，frame_count=3 | 保留 64 queries 和 grouped context |
| 原始编码替换 b / replace_b_encoding | 保留 8 帧 H120 | **逐帧 ViT 取 CLS，不使用新增 patch 时序层** | 保留 64-query compressor，读入逐帧各相机 CLS |
| 原始汇聚替换 c / replace_c_readout | 保留 8 帧 H120 | 保留完整 patch 级时序编码 | **取编码后的 CLS → 原始 view attention → 原始顺序 context 路径**，无 learned-query compressor |

替换是系统模块干预。a 同时恢复原始帧数和跨度，b 同时恢复原始视觉输出粒度和取消新增时序层，c 同时恢复原始汇聚与 context 输入接口；不能把结果解释成仅某个单独算子的因果贡献。

## 接口定义

统一记 `B` 为 batch，`T` 为采样数，`V=2`，`P` 为 ViT token 数（包括 CLS），`D=384`。历史按时间从旧到新排列。

### a：稀疏长历史换回连续短历史

- 用 `[0,1,2]` 取当前与之前两帧，早于 episode 起点的索引复制第 0 帧，始终输出三个观测。
- 逐帧编码仍输出 `[B,3,V,P,D]`，b、c 的运算不变，64-query 接口不变。
- 数据仍以一个 endpoint 为监督单位；不要设置 `history_size=3` 后同时沿用 STMemory 的 raw-history 分支，从而意外改变 action/target 序列语义。
- 比较的是长跨度稀疏输入相对原始短窗口策略的效果；不能将其称为只改变跨度或只改变帧数。

### b：时空编码换回逐帧 CLS 编码

- a 输出不变，先独立编码每个相机和时间，使用原始 ViT `CLS` 与原始 camera embedding，得到 `[B,8,V,D]`。
- 不做 patch 级跨时间 attention，不传递 patch 网格。保留可标识历史顺序的 time-slot embedding；它属于进入 c 的位置适配，不能隐去不写。
- 以 singleton patch 维 `[B,8,V,1,D]` 交给同规格 64-query compressor；c 因此读取 16 个 CLS 输入，仍输出 `[B,64,D]`。
- 编码与 compressor 的公共边界需拆开，不能调用当前 `SpaceTimePerceiverEncoder.forward()` 后又重复添加 camera/time embedding。
- 替换版使用原始 WCM 的逐帧视觉表征，而时间汇聚由保留的 c 负责。它不同于此前 `no_temporal`（此前仍传递每帧完整 patch 网格）。

### c：查询压缩换回 CLS 与视角池化路径

- b 正常得到经过时序编码的 `[B,8,V,P,D]`。取每时刻每视角的 CLS 槽，得到 `[B,8,V,D]`。
- 用原始 `WorldCriticModel.pool_views()` 将每时刻两视角池化为 `[B,8,D]`。b 已加 camera/time embedding，此处不得再次添加。
- 这八个向量按真实顺序进入原始 `encode_context()`：使用原始语言融合、因果 context trunk，取最后时刻输出作为当前 endpoint context。
- 原始因果 mask 必须保留。不要把八个时刻伪装成同一时间的八个 grouped tokens；那样没有恢复原始 WCM 的时间接口。
- 当前 dynamics 的视觉锚点使用最后时刻的池化 CLS；next-state target 保持原方案的下一帧视觉潜变量和 stop-gradient；V/Q/dynamics head 的网络与 loss 系数保持不变。
- b 已能利用当前及过去的整个历史，故原始 context 的因果 mask 不保证 b 的中间历史 token 完全不含较晚历史槽信息；只评价最后 endpoint，不输出或监督这些中间槽的价值。所有输入均不晚于 endpoint。
- 不设置 `perceiver_queries=1` 来伪装此变体，也不以所有 patch 的均值代替原始 CLS+view attention。

## 训练和选模

1. seed=42；复用同一 train/val episode split、数据、图像尺寸、语言编码器、归一化、batch=4 和 loss 系数。
2. 三个替换版均从同一 seed 的公共上游初始化（原流程的 `wcm_sparse4_image_b4_exp_s42/deploy.pt`），按 joint 10 epoch + full 3 epoch 训练。不能从完整 H120 最终权重关模块后继续训练来替代本实验。
3. 公共 WCM 层按相同键加载；替换模块及 time embedding 按统一 seed 初始化，训练启动日志记录加载层和新增层。c 恢复的 view attention 与 context 权重从相同公共上游加载。
4. 只保留 `checkpoints/last.pt` 和最终 `deploy.pt`，full 成功后清除本次 joint 权重。所有新变体取最后一轮；历史完整模型仍取其已有验证最优权重，报告明确注明差异。
5. 训练前分别验证真实 forward、backward、每个保留模块有梯度、已移除模块未参与计算。重点检查 b 的 CLS token 数、c 的时间排序和仅 endpoint loss；不写兜底分支。

## 评估与结论

- 主评估为独立 OOD，统一四行的共同 `(episode_id, frame_index)` manifest，target 一致；同一指标表不能混用其他 endpoint 集。
- 报告 MSE、Pearson、centered MSE、bias；20,000 次 paired episode bootstrap，差值为替换版减完整模型。
- 完整模型误差更低才支持对应完整模块的收益；替换版更好或不显著时照实报告，不能预先把它写成正向消融。
- 同时记录参数量、视觉输入 token 数、context token 数、同硬件 batch 的推理耗时。a/b/c 各自改变算量，不把更便宜或更昂贵误写成等算力实验。

## 与已完成实验的关系

此前 `no_temporal`、`queries128`、`four_frames` 保留为“时序层开关与规模探索”，不写入新的 a/b/c 替换主表，也不删除已有结果。单主相机取消。均值池化与 16-query 不属于当前模块替换任务，不再补跑。

本文件是重新设计方案；三个替换版尚未实现、生成配置或开始训练。旧 `run_scripts/module_ablation.sh` 不能执行本方案。实现完成并做 GPU 预检后再给一条新的 tmux 批量命令。

历史完整输入单任务约 4.3 小时训练；三项替换含评估先预留约 8–15 小时，a/b 会减少视觉或时序计算，c 会降低 context token 数，实际时长需通过各自首轮确定。这不是已经测得的替换版耗时。
