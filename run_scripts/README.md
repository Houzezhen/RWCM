# 当前实验入口

服务器项目路径：`/home/user/code_1/WCM`。最新论文仅使用 `paper/stmemory/`。

训练：`bash run_scripts/train.sh /path/to/config.yaml`。

评估：`bash run_scripts/evaluate.sh /path/to/deploy.pt /path/to/dataset /path/to/new_eval_dir`。

两个入口默认使用 RTX 5880 Ada（`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=1`）；现有 torch 2.7 环境不支持服务器另一张 Blackwell 卡。不要改为 0，除非已经更换兼容环境。

所有训练应放在 tmux 中。旧编号 sh 已移到 `_archive/cleanup_20261026/`，仅供查阅；归档脚本的原始相对路径不保证可直接执行。按当前配置调用 `world_critic.train` 和 `world_critic.evaluate`，不要直接运行旧清权重、迁移磁盘或修数据脚本。

模块消融清单见 `experiments/module_ablation/README.md`。本次只整理入口和实验清单，没有启动新训练。

模块消融两阶段训练与 OOD 评估：`bash run_scripts/module_ablation.sh no_temporal 42`。已生成的变体和预计耗时见上述清单；pooling 和 16-query 尚不支持。
