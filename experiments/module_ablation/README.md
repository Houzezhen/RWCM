# STMemory 模块消融准备

> 2026-10-27：用户要求改为 Figure 2 a/b/c 各模块换回原始 WCM 路径。最新方案见 [FIGURE2_REPLACEMENT_PLAN.md](FIGURE2_REPLACEMENT_PLAN.md)。下方旧配置和开关/规模探索不代表新方案；单主相机、均值池化与 16-query 不再属于本次任务。

## 当前执行约定（以此段为准）

只跑 seed 42；完整模型直接复用历史 H120 权重和评估，不重训。生成器默认只生成 `no_temporal`、`queries128`、`four_frames`、`primary_only` 的 joint/full 共 8 份配置。

每阶段启用 `checkpoint_last_only=true`：只覆盖写 `checkpoints/last.pt`，不保存 best 和各 epoch checkpoint；`deploy_from_best=false`，`deploy.pt` 取最后一轮。full 阶段成功后删除本次 joint 的 `last.pt` 和 `deploy.pt`，最终每个变体仅保留 `full/checkpoints/last.pt` 以及轻量评估包 `full/deploy.pt`。日志、配置、数据划分和指标保留，历史模型不改动。

因此本轮所有新消融都使用最后一轮，历史完整模型仍是既有验证最优权重，报告需注明选模方式不同。训练耗时仍约 12–19 小时；不因少存 checkpoint 而预计成分钟级。四项最终权重及评估包约 9–12GB，另留临时保存和训练余量。

下方固定协议和先前三 seed/完整模型重训/10GB每项的建议为前期方案，当前执行已由以上约定替代。

以 `paper/stmemory/stmemory_world_models_en.tex` 中 `tab:modules` 为目标表格。

| 变体 | 输入帧数 | 跨度 | 查询数 | 当前实现状态 |
| --- | --- | --- | --- | --- |
| 完整 STMemory-120 | 8 | 120 | 64 | 已有三 seed 权重与评估，可作历史系统参照 |
| 去时序注意力 | 8 | 120 | 64 | `model.spacetime_temporal_enabled=false` 已支持 |
| 均值池化代替查询压缩 | 8 | 120 | 1 | 需要新增明确的 pooling 分支；不能以推理时关闭查询替代训练消融 |
| 16 queries | 8 | 120 | 16 | 需要放开 config.py 当前仅接受 64/128 的校验并核对 checkpoint 初始化 |
| 128 queries | 8 | 120 | 128 | 配置已支持；查询形状变化，不能完整加载 64-query checkpoint |
| 四帧输入 | 4 | 120 | 64 | 配置可设 `[0,40,80,120]`，同步 `spacetime_frame_count=4` |
| 单主相机 | 8 | 120 | 64 | `data.image_keys=[observation.images.primary]`，两阶段均保持一致 |

## 固定协议

- 模型 seed：42、3072、1337；split_seed=3072，episode split 复用原始清单。
- 所有新变体与**同时重训的完整模型对照**使用相同初始化来源、loss、batch、joint 10 epoch + full 3 epoch、学习率、数据和选模规则。历史完整模型可列系统参照，但不能替代匹配训练的架构对照。
- 为消除结构差异造成的初始化偏差，使用同 seed 的公共上游 checkpoint 初始化公共层；新模块按相同规则初始化。不要从已经学过完整模块的 H120 最终模型继续训练后声称是从零去模块消融。
- 现有原始方案使用同 seed 的 Image-B4 上游初始化。若沿用，应对所有新变体及完整对照一致应用，并在内部记录来源；不要悄悄替换成完整 H120 checkpoint。
- 主测试为 0722 OOD，支持测试为 0814 5cut。按所有变体共同 `(episode_id, frame_index)` 固定 manifest，再用 `scripts.compare_experiments` 做 20000 次 paired episode bootstrap。
- 输出目录使用新的 `outputs/module_ablation/<variant>_s<seed>/`，不要覆盖现有模型或评估。
- 16/128 queries、四帧会改变 checkpoint 张量尺寸，需先做 forward/backward smoke，确认实际初始化层清单后再开始两阶段训练。

## 现有配置与服务器资源

`configs/wcm_spacetime_direct.yaml`、`configs/wcm_spacetime_direct_full.yaml` 是 joint/full 配置参考，默认跨度是 60，不能直接当作模块消融配置。新配置必须显式设置 H120 偏移 `[0,17,34,51,69,86,103,120]` 和本次变体参数。

训练数据：`/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0814_expanded_with_return`。

OOD：`/media/Data/user/huggingface/lerobot/test/data_toiletButton_0722_with_return`。

既有 H120：`/media/Data/user/WCM_outputs/wcm_spacetime_vit_t8_direct_full_t120_s{42,3072,1337}`。

2026-10-26 检查：系统盘约 38GB 可用，数据盘约 122GB 可用。训练产物应放数据盘的新目录；实际开跑前据 checkpoint 保存策略估算空间。不要运行旧清理脚本自动删除权重。

## 服务器运行（2026-10-26 补充）

已生成并通过配置校验的入口：`full`、`no_temporal`、`queries128`、`four_frames`、`primary_only`，每项 42/3072/1337 三 seed，两阶段共 30 份配置。尚未启动训练；完整 GPU 训练仍需首轮观察。

```bash
cd /home/user/code_1/WCM
tmux new -s stmemory-ablation
bash run_scripts/module_ablation.sh no_temporal 42
```

入口依次跑 joint 10 轮、full 3 轮和 OOD 评估，保存在 `/media/Data/user/WCM_outputs/module_ablation_20261026/no_temporal_s42/`。`Ctrl+b` 再按 `d` 脱离 tmux，`tmux attach -t stmemory-ablation` 返回。

先做同一 seed 的匹配对照与去时序层：

```bash
bash run_scripts/module_ablation.sh full 42
bash run_scripts/module_ablation.sh no_temporal 42
```

需要重建配置时用 `.venv/bin/python -m scripts.prepare_module_ablation`。若换 output-root，生成器和运行时 `OUTPUT_ROOT` 必须一致。入口不自动跳过、覆盖重启或恢复半成品；新试验用新的根目录。

历史同卡日志：joint 10 轮约 3.10 小时，full 3 轮约 1.22 小时，合计 4.32 小时，另加加载、保存和评估。完整输入变体按 4.5–5 小时预留；四帧、单相机预计更快，未实测前不作为确定耗时。五项单 seed 约 16–24 小时，三 seed 约 2–3 天，现有环境仅一张 GPU 能直接使用。七项三 seed 全表约 3–5 天，前提是先完成 pooling 和 16-query 实现。

保存 best/last/epoch 和两个 deploy 时，历史单项两阶段约占 10GB。五项三 seed 约需 150GB，七项三 seed 约需 210GB，当前数据盘 122GB 可用，不能直接全量排队；需要先调整 checkpoint 保留方案或增加可用空间。单 seed 的完整对照+去时序层预计约 20GB，可先运行。

pooling/16 queries 的模型实现修改仍未执行，这两项不能用当前入口运行。训练前还应确认结构变化变体的公共层初始化记录。
