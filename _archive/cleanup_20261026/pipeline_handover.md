# WCM 价值函数流水线工作交接

> 覆盖：`1_add_returns.sh` → `2_run_train.sh` → `3_run_eval.sh` → `4_gen_video.sh` → `wcm_infer_acp_to_dataset.py`
> 更新日期：2026-08-26
> 环境：`/home/user/code_1/WCM/.venv/bin/python`（uv 环境，装有 lerobot 0.5.1）

---

## 0. 流水线总览

```
原始 v2.1 数据集 (success + failure)
        │  merge_v21_datasets.py（合并，生成 success_labels.json）
        ▼
data_toiletButton_merged_0814
        │  ① add_returns：add_returns_to_lerobot_v21.py（v2.1 原地加列）
        ▼
data_toiletButton_merged_0814_with_return   ← 含 return / return_raw / episode_success
        │  ② 训练：world_critic.train（2_run_train.sh）
        ▼
outputs/wcm_toiletButton_merged_0814/       ← checkpoint + resolved_config.json
        ├── ③ 离线评估：world_critic.evaluate（3_run_eval.sh）→ eval/
        ├── ④ 价值视频：episode_value_video pipeline（4_gen_video.sh）→ eval_videos/
        └── ⑤ ACP 标注：wcm_infer_acp_to_dataset.py
                 → 回写 complementary_info.value / advantage / acp_indicator
        │  辅助可视化：make_acp_curves.py、visualize_acp.py
        ▼
⑥ convert_v21_to_pistar.py → data_toiletButton_merged_0814_pistar（已生成）
     adv_ind = WCM acp_indicator（1→"positive"/0→"negative"），pistar 策略训练直接消费
```

核心关系：WCM 价值模型本身不参与在线 RL，是**纯离线标注器**——用训练好的价值函数给数据集的每条轨迹打分，产出 ACP 正负标签供下游策略训练。

---

## 1. 环境与通用约定

- 所有脚本用固定解释器：`/home/user/code_1/WCM/.venv/bin/python`（不要用 base conda 的 piper_lerobot，缺 recompute_stats 等依赖）。
- `HF_ENDPOINT` 默认走镜像 `https://hf-mirror.com`（huggingface.co 直连会被重置）。
- `WANDB_MODE=offline`（训练脚本内默认）。
- 数据集为 **v2.1 格式**：`data/chunk-*/episode_*.parquet`（一集一个文件）；脚本普遍自动兼容 v3（`file-*.parquet`）。
- 只读文件（视频 mp4 等）复制时用硬链接节省磁盘；被改动的 parquet / info.json 才会真复制。

---

## 2. 数据准备

### 2.1 合并数据集（如有多个源数据集）

```bash
python scripts/merge_v21_datasets.py \
    --datasets /path/success_0717 /path/failure_0717_f_old \
    --success /path/success_0717 \
    --output /path/data_toiletButton_merged_0814
```

- 按 v2.1 约定重新编号 `index` / `episode_index`、硬链接视频；
- 成功数据集的 episode 标 1，其余标 0，输出 `success_labels.json` 到输出根目录。

### 2.2 加 return 列

**当前 `1_add_returns.sh` 指向的是 v3 脚本 + 另一个数据集**（`data_toiletButton_0722_2`，输出 `..._with_return_taskmax`，`--failure-penalty 300`）：

```bash
# 1_add_returns.sh（现状，v3 迁移写法）
/home/user/code_1/WCM/.venv/bin/python scripts/add_returns_to_lerobot.py \
  --repo-id data_toiletButton_0722_2 \
  --root /media/Data/user/huggingface/lerobot/lerobotevo \
  --output-dir /media/Data/user/huggingface/lerobot/lerobotevo/data_toiletButton_0722_2_with_return_taskmax \
  --failure-penalty 300 \
  --normalization task_max \
  --success-labels /media/Data/user/huggingface/lerobot/lerobotevo/data_toiletButton_0722_2/success_labels.json
```

> 注意：`add_returns_to_lerobot.py` 会把数据集迁移到 v3.0；**保持 v2.1 用 `add_returns_to_lerobot_v21.py`**（当前 0814 这条线就是 v21 脚本做的，不迁移格式）。

**0814 这条线实际使用的参数**（来自 `data_toiletButton_merged_0814_with_return/meta/return_definition.json`，命令本身未留档，推测如下）：

```bash
/home/user/code_1/WCM/.venv/bin/python scripts/add_returns_to_lerobot_v21.py \
  --root data_toiletButton_merged_0814 \
  --output data_toiletButton_merged_0814_with_return \
  --failure-penalty 100 \
  --normalization task_max \
  --success-labels data_toiletButton_merged_0814/success_labels.json
```

- 返回定义（pi\*0.6 式）：中间步 `-1`，成功终点 `0`，失败终点 `-failure_penalty`；`discount=1`，`step_scale=1`；
- `task_max` 归一化：`clip(return / 该任务最长回合长度, -1, 0)`，与 pistar06 标尺同源；
- 产物：`return` / `return_raw` / `episode_success` 三列 + `meta/return_definition.json`；
- 当前数据：25 成功 + 25 失败 episode，task_max = 229（单任务）。

---

## 3. 训练（`2_run_train.sh` → `world_critic.train`）

```bash
# 改顶部配置区后执行
bash 2_run_train.sh
```

关键配置（顶部变量）：
- `GPUS=1`、`CUDA_VISIBLE_DEVICES=0`、`CONFIG="configs/train_1gpu.yaml"`；
- `DATASET_REPO_ID=data_toiletButton_merged_0814_with_return`、`DATASET_ROOT=/home/user/code_1/WCM/data_toiletButton_merged_0814_with_return`；
- `OUTPUT_DIR=outputs/wcm_toiletButton_merged_0814`、`EPOCHS=30`、`PER_DEVICE_BATCH_SIZE=8`、`PRECISION=bf16`。

实际执行：`python -m world_critic.train --config configs/train_1gpu.yaml`（多卡走 torchrun）。

模型结构（见 `outputs/wcm_toiletButton_merged_0814/resolved_config.json`）：
- 视觉 `google/vit-base-patch16-224-in21k`（可训练），语言 `openai/clip-vit-base-patch32`（冻结），`history_size=3`，`return_key="return"`；
- 损失：`value_weight=1.0` + `next_state_weight=1.0` + `sigreg_weight=0.01`；lr=1e-5，warmup 300。

产物：
- `outputs/wcm_toiletButton_merged_0814/checkpoints/best.pt` / `last.pt` / `epoch-*.pt`
- `outputs/wcm_toiletButton_merged_0814/deploy.pt`（部署用）
- `outputs/wcm_toiletButton_merged_0814/resolved_config.json`（含 data.root，供下游默认取用）

---

## 4. 离线评估（`3_run_eval.sh` → `world_critic.evaluate`）

```bash
# 改顶部配置区后执行
bash 3_run_eval.sh
```

关键配置：`CHECKPOINT` / `OUTPUT_DIR` / `SPLIT`（train|val|all）/ `BATCH_SIZE=16` / `NUM_WORKERS=2` / `PLOT_EPISODE_CURVES=1`。

实际执行（等价于）：
```bash
python -m world_critic.evaluate \
  --checkpoint outputs/wcm_xxx/checkpoints/best.pt \
  --output-dir outputs/wcm_xxx/eval \
  --split val --batch-size 16 --num-workers 2 \
  --expected-world-size 1 --episode-curves --log-every-batches 20
```

产物：`eval/summary.json`、`eval/episode_curves/episode_curves{,_summary,_metrics}.json`（+ PNG/CSV）。

> 注意：`3_run_eval.sh` 当前指向 sink2table_0819 的 checkpoint，评估 0814 模型前需改 `CHECKPOINT` / `OUTPUT_DIR`。

---

## 5. 价值视频（`4_gen_video.sh` → `episode_value_video pipeline`）

```bash
bash 4_gen_video.sh
```

核心命令（当前内容）：
```bash
export WCM_DATASET_ROOT="/home/user/code_1/WCM/data_toiletButton_merged_0814_with_return"
export WCM_DATASET_REPO_ID="data_toiletButton_merged_0814_with_return"
export WCM_VISION_MODEL_NAME="google/vit-base-patch16-224-in21k"
export WCM_LANGUAGE_MODEL_NAME="openai/clip-vit-base-patch32"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python -m episode_value_video pipeline \
  --checkpoint outputs/wcm_toiletButton_merged_0814/checkpoints/best.pt \
  --eval-output-dir outputs/wcm_toiletButton_merged_0814/eval_videos \
  --split all --speed 2.0 --overwrite \
  --output-dir outputs/wcm_toiletButton_merged_0814/eval_videos/episode_value_videos
```

产物：`eval_videos/episode_value_videos/`（每集视频 + `episode_value_videos.json` manifest）、`eval_videos/summary.json`。

---

## 6. ACP 标注（`wcm_infer_acp_to_dataset.py`）

作用：用训练好的 WCM 给数据集逐帧推理价值，**原地回写** 3 列：

| 列 | dtype | 含义 |
|---|---|---|
| `complementary_info.value` | float32 | WCM 预测价值（3 帧历史窗口 + 动作条件） |
| `complementary_info.advantage` | float32 | n-step advantage（默认 n=50, γ=1） |
| `complementary_info.acp_indicator` | int64 | 0/1，按任务内分位数二值化（默认 top 30% 为正） |

命令（0814 数据集对应 checkpoint，`--dataset-root` 可省略，默认取 checkpoint 保存的 data.root）：

```bash
/home/user/code_1/WCM/.venv/bin/python scripts/wcm_infer_acp_to_dataset.py \
  --checkpoint outputs/wcm_toiletButton_merged_0814/deploy.pt \
  --dataset-root data_toiletButton_merged_0814_with_return \
  --batch-size 16 --num-workers 4
```

常用参数：
- `--advantage-mode {n_step,td}`、`--gamma`、`--n-step`（默认 50，与 evo 一致）；
- `--positive-ratio 0.3`（每任务 top 比例为正）；
- `--force-intervention-positive`（干预帧强制为正）；
- `--dry-run`（只推理打印统计，不写回）。

前置条件：
- 数据集必须有 `return` 列（先跑第 2 节 add_returns）；
- checkpoint 的 `schema_version` 必须匹配 `world_critic.checkpoint.CHECKPOINT_SCHEMA_VERSION`。

核心逻辑（移植自 evo `lerobot_value_infer.py`，数值行为一致）：
```
value 推理（窗口 endpoint 价值）→ 真值 return 帧间差分得稠密 reward
→ n-step advantage: adv = Σreward + bootstrap - V(s)
→ 按任务分位数阈值二值化 → acp_indicator
```

---

## 7. 辅助可视化

- `scripts/make_acp_curves.py`：从 parquet 生成 advantage/acp/value 曲线 JSON（`--mode advantage|acp|value`），供 episode_value_video render 叠加 advantage 视频：
  ```bash
  python scripts/make_acp_curves.py --dataset-root data_toiletButton_merged_0814_with_return \
      --mode advantage --output outputs/wcm_toiletButton_merged_0814/eval_videos/advantage_videos/acp_curves.json
  ```
- `scripts/visualize_acp.py`：每集画一张图（predicted value / n-step advantage 曲线 + acp=1 帧高亮）：
  ```bash
  python scripts/visualize_acp.py --dataset-root data_toiletButton_merged_0814_with_return \
      --episodes 0 6 40 45        # 省略画全部
  ```
  输出默认 `<root>/acp_visualization/`（含 `acp_summary.json`）。

---

## 8. （可选）转 pistar 格式（`convert_v21_to_pistar.py`）

把 WCM 标注的 v2.1 数据集转成 pistar RECAP 标准形式（元数据与 pistar 官方 `control_your_robot/scripts/backfill_lerobot_columns.py` 对齐）：
- 重命名列：`observation.images.primary→image`、`observation.images.wrist→wrist_image`、`observation.state→state`、`action→actions`；
- 新增 pistar 列（写入顺序/类型/`names` 均对齐 pistar）：`adv_ind`（string，`names=["adv_ind"]`）、`value_label`（float32）、`reward`（float32）、`reward_label`（float32）、`intervention`（int64，`names=["intervention_flag"]`）；
- **`adv_ind` 优先来自 WCM 的 `complementary_info.acp_indicator`**（1→"positive"，0→"negative"，即用 WCM value 算的 advantage 分位数二值化）；无该列时退化为按成功/失败标记（`--adv-ind-success`/`--adv-ind-fail`，默认 positive/negative）；
- `value_label` 取 WCM return 列 clip 到 [-1,0]（失败末帧 -1）、`reward_label` 末帧 = value_label[-1]（与 pistar `rewrite_rewards.py` 失败感知约定一致）；
- 保留原 WCM 列（return/return_raw/episode_success/complementary_info.*）不动；修复 parquet schema metadata（v2.1 List→Sequence）。

```bash
# 前提：先用 wcm_infer_acp_to_dataset.py 写好 complementary_info.acp_indicator（见第 6 节）
python scripts/convert_v21_to_pistar.py \
  --root data_toiletButton_merged_0814_with_return \
  --output data_toiletButton_merged_0814_pistar \
  [--success-labels data_toiletButton_merged_0814_with_return/success_labels.json]
```

**当前状态：已完成**。2026-08-26 转换 `data_toiletButton_merged_0814_pistar`（50 集，全部基于 WCM `acp_indicator` 生成 adv_ind；全局 positive 2220 / negative 5149，50 集均含 negative 帧）。`adv_ind` 语义与 pistar `label_advantage_from_vlm.py` 输出对齐（逐帧 "positive"/"negative"），可直接被 pistar `train_value.py` / `label_advantage_from_vlm.py` / 策略训练消费（后者只读 adv_ind）。若换 advantage 参数（positive_ratio / TD / n-step），重跑 `wcm_infer_acp_to_dataset.py` 后再转换。

---

## 9. 关键产物路径清单

| 阶段 | 路径 |
|---|---|
| 源数据（v2.1，已合并） | `/home/user/code_1/WCM/data_toiletButton_merged_0814` |
| 加 return 后 | `/home/user/code_1/WCM/data_toiletButton_merged_0814_with_return` |
| 训练输出 | `/home/user/code_1/WCM/outputs/wcm_toiletButton_merged_0814/` |
| 评估输出 | `outputs/wcm_toiletButton_merged_0814/eval_overfit_check/`、`.../eval_videos/` |
| 价值视频 | `outputs/wcm_toiletButton_merged_0814/eval_videos/episode_value_videos/` |
| advantage 视频 | `outputs/wcm_toiletButton_merged_0814/eval_videos/advantage_videos/` |
| ACP 可视化 | `data_toiletButton_merged_0814_with_return/acp_visualization/` |
| 其他实验 | `outputs/wcm/`、`outputs/wcm_sink2table_0819{,_dedup}/`（sink2table 任务） |

---

## 10. 常见注意事项

1. **解释器**：一律用 `.venv/bin/python`；`1_add_returns.sh` 里注释说明了为什么不用 base conda 环境。
2. **HF 网络**：`HF_ENDPOINT=https://hf-mirror.com`；wandb 走 `offline`。
3. **v2.1 vs v3**：0814 线是 v2.1（`episode_*.parquet`）；`add_returns_to_lerobot.py`（v3 版）会迁移格式，不想迁移用 `_v21` 版。
4. **checkpoint schema**：`wcm_infer_acp_to_dataset.py` / 评估脚本对 schema_version 有强校验，混用不同版本 checkpoint 会直接报错。
5. **写回是覆盖式的**：`wcm_infer_acp_to_dataset.py` 原地重写 value/advantage/acp_indicator，重跑前建议备份或先 `--dry-run`。
6. **`acp_indicator`（evo 侧）与 `adv_ind`（pistar 侧）是一体两面**：`wcm_infer_acp_to_dataset.py` 写 `complementary_info.acp_indicator`（int64 0/1，evo ACP hook 直接读）；`convert_v21_to_pistar.py` 把同一份 advantage 标签映射成 pistar 的 `adv_ind`（string "positive"/"negative"）。两者来自同一套 WCM 分位数二值化，改参数（positive_ratio/n_step/γ）时两处都会变。
7. **脚本顶部的配置区要按数据集改**：`2_run_train.sh`/`3_run_eval.sh`/`4_gen_video.sh` 都各自硬编码了 checkpoint / 数据集路径，交接后跑新数据前先核对。
