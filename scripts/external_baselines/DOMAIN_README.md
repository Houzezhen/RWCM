# 领域方法的轻量 baseline（2026-10-22）

服务器运行目录：`/home/user/code_1/WCM`。

本轮运行：`outputs/domain_baselines_20261022/`，tmux 会话：`wcm-domain-baselines`。

```bash
tmux attach -t wcm-domain-baselines
tail -f outputs/domain_baselines_20261022/train.log
```

特征提取进度在 `cache.log`；`completed_at.txt` 表示训练和比较表已全部完成，`exit_code.txt` 记录退出状态。

## 实际训练内容

| 名称 | 保留机制 | 轻量化与适配 |
|---|---|---|
| TD-MPC2-style critic | 官方 SimNorm、NormedLinear、MLP；action-conditioned latent dynamics、reward prediction | latent=128、hidden=256；1-step latent consistency + reward + MC value readout；无规划器、策略、Q ensemble、two-hot 分类 |
| VIP-style critic | 官方 value-implicit representation objective，successful training goal | 冻结 ResNet18，训练 128 维投影；监督 readout 输入 detach，防止 return loss 改写 VIP 表征；不在推理时读取 goal |
| IQL critic | twin-Q TD、expectile value loss、EMA target | hidden=256、latent=128；expectile=0.7；无 actor；gamma=1 与项目 undiscounted return 对齐 |

这三个模型都有**独立训练的编码投影和价值网络**。它们共用冻结的公共 ImageNet ResNet18 与 CLIP 文本特征，**不使用 SpaceTime 训练后的视觉权重**。缓存减少重复视频解码和视觉前向；不是三个各自端到端微调的 ResNet18。

外部方法均输入同一套 8 帧 H120 历史、两路相机、语言以及当前机器人状态。序列不足时复制最早帧，保留早期 endpoint；当前 V 推理不输入 action、未来帧、轨迹长度、success 或终点目标。

所有方法训练 seed=42、1000 updates、batch=128，AdamW lr=3e-4、weight_decay=1e-3。训练 endpoint 间隔 4 帧并额外包含终点；每个 batch 成功/失败各半。每 100 updates 在原有 episode-disjoint validation split 上选择 MSE 最低 checkpoint。验证与 OOD 均逐帧预测。

IQL 的奖励由相邻 return 之差恢复，终点后 bootstrap 为 0，保持项目 return 的归一化尺度（包括 clipping）。这不是重新执行环境获得的原始 reward。TD-MPC2-style 同样预测该归一化奖励。

SpaceTime 使用已有 `wcm_spacetime_vit_t8_direct_full_t120_s42` 的完整 OOD CSV，**不启动任何 SpaceTime 训练**。它的训练预算与外部短训方法不同，故结果属于系统级快速比较，不能归因成纯架构差异。

## 来源与结果

- 官方源仓库：[`nicklashansen/tdmpc2`](https://github.com/nicklashansen/tdmpc2)、[`facebookresearch/vip`](https://github.com/facebookresearch/vip)、[`ikostrikov/implicit_q_learning`](https://github.com/ikostrikov/implicit_q_learning)。
- `vendor_domain/` 保存实际使用或参考的官方源码、LICENSE 与 `sources.json` 中的 SHA256。TD-MPC2 构件由官方源码 AST 原样抽取；VIP 与 IQL 目标在本项目统一数据接口上移植。未将本轮称作完整算法复现。
- `comparison.md` / `comparison.csv`：MSE、Pearson、centered MSE、bias、可训练参数、训练时间、SpaceTime MSE 改善百分比。
- `endpoint_manifest.json`：所有模型共同测试 endpoint。
- `comparisons/*.json`：20000 次 paired episode bootstrap，差值方向为 SpaceTime-minus-baseline，负的 MSE 差值支持 SpaceTime。
- `<method>_s42/best.pt` / `metrics.jsonl` / `episode_curves.csv` / `summary.json`：权重、训练记录、逐帧预测与摘要。

再次完整运行可用 `36_run_domain_baselines.sh`；为新实验指定不同的 `OUT`。当前运行使用随本次上传的 `cache_stage.sh`，直接读取已归档的源码，避免服务器 GitHub 网络下载阻塞。

WCM 原论文参考：[§4.1 Baselines](https://arxiv.org/html/2607.29613#S4.SS1)、[§5.1 / Figure 5](https://arxiv.org/html/2607.29613#S5.SS1)。原论文的 MLP 与 ViT 是 critic 消融；本轮按用户进一步要求改为领域方法的轻量适配，不再跑通用 TCN/Perceiver/ViViT 模块。
