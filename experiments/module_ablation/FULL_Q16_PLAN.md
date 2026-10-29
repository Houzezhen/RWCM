# 暂存计划：a/b/c full 微调与16-query

状态：**按用户2026-10-27要求暂停，不启动、不自动等待GPU后开跑。**

排队 tmux `stmemory-full-q16` 已停止。此前仅等待显存，没有启动新训练。

## 工作范围

- a/b/c：分别从本轮 `module_replacement_20261027/<variant>_s42/deploy.pt` 继续 full 3 epoch，不重复joint。
- Q16：保留完整STMemory结构、8帧H120、两相机，将查询数改16。从同seed公共上游初始化，新query随机初始化，训练joint10+full3，不从Q64成品裁剪查询。
- 均为seed42、batch4、last-only，每模型保存最后checkpoint和deploy。
- 对照仍用论文最终H120 Q64预测；新模型last，历史模型validation-best，报告注明选模不同。

## 已完成准备

- `config.py` 支持16、64、128 queries。
- full冻结策略已适配b的camera embedding与c的原始视角池化，使替换路径在full中继续训练。
- `scripts/prepare_replacement_followup.py` 已在服务器生成5份配置，位于 `configs/replacement_followup/`。
- 46项相关测试通过；Q16 CPU真实前向、反向和最老帧梯度检查通过。GPU路径预检待正式运行时执行。
- 待运行入口 `run_scripts/replacement_followup.sh`；结果汇总 `scripts/summarize_replacement_followup.py`。
- 目标输出 `/media/Data/user/WCM_outputs/replacement_followup_20261027/`。当前仅有等待日志，无新模型训练结果。

服务器当前5880显存约47GB/48GB被其他服务使用；本轮不得停用其他服务。显存可用后再决定串行或并行启动。脚本内每任务启动前等待至少24GB可用，但当前没有后台脚本自动等待。

此计划只暂存，恢复执行需用户明确要求。
