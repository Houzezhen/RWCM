# Rollout 数据 → WCM 推理 adv → 训练 pi05 操作手册

> 目标：把真实机器人 rollout 数据（关节角）用 **WCM 价值模型**推理出 advantage 标签（`adv_ind`），
> 转成 pistar 四元数格式，再训练 pi05 策略（`pi05_piper_history`）。
> 与 pistar 自家 VLM 标注（`toilet_button_rollout_round1_pistar`）形成对照实验。
> 更新日期：2026-08-26

---

## 0. 数据流总览

```
原始 rollout（关节角 v2.1 标准 key）
  /home/user/lerobot_xiazhi/datasets/rollout
        │  ① 推断 success_labels.json（reward_label 末帧：0=成功 / -1=失败）
        │  ② add_returns_to_lerobot_v21.py → 加 return/return_raw/episode_success
        ▼
rollout_with_return（关节角 + return）
        │  ③ wcm_infer_acp_to_dataset.py（deploy.pt）→ 写 complementary_info.value/advantage/acp_indicator
        ▼
rollout_with_return（关节角 + return + WCM 标注）
        │  ④ convert_rollout_to_quaternion.py → FK 四元数 + 扁平 key + adv_ind 映射
        ▼
rollout_round1_wcm_pistar（pistar 格式，adv_ind = WCM advantage 分位数）
        │  ⑤ pi05 训练（config repo_id 指向新数据集）
        ▼
pi05 策略（ACP 用 WCM adv_ind）
```

## 1. 关键约束（先理解，否则会踩坑）

1. **WCM checkpoint 是在关节角数据上训练的**（`observation.state` names = `main_joint_1..6+gripper`）。
   所以 **WCM 推理必须在关节角空间**（原始 rollout），不能在 pistar 四元数数据上直接跑。
2. **pi05 训练需要末端四元数位姿**（`x,y,z,qx,qy,qz,qw`，与 pistar 官方格式一致），
   所以推理后要用 FK 把关节角转成四元数（`convert_rollout_to_quaternion.py`）。
3. **WCM 推理需要 `return` 列**（用它做帧间差分得稠密 reward）→ 必须先 add_returns。
4. rollout 数据没有显式成功标签 → 用 `reward_label` 末帧推断（末帧 `0`=成功，`-1`=失败，pistar 约定）。

## 2. 前置条件

| 项 | 路径 |
|---|---|
| 原始 rollout（关节角 v2.1） | `/home/user/lerobot_xiazhi/datasets/rollout`（5 集，已含 pistar 列，adv_ind='none'） |
| WCM checkpoint | `/home/user/code_1/WCM/outputs/wcm_toiletButton_merged_0814/deploy.pt` |
| WCM 脚本 | `/home/user/code_1/WCM/scripts/wcm_infer_acp_to_dataset.py` |
| add_returns v21 脚本 | `/home/user/code_1/WCM/scripts/add_returns_to_lerobot_v21.py` |
| FK 转换脚本（已改） | `/home/user/code_1/pistar/scripts/convert_rollout_to_quaternion.py` |
| WCM 环境 | `/home/user/code_1/WCM/.venv/bin/python` |
| pistar 环境 | `/home/user/code_1/pistar/venv/bin/python` |

> 已改：`convert_rollout_to_quaternion.py` 现在会把 `complementary_info.value/advantage/acp_indicator`
> 保留下来，并在存在 `acp_indicator` 时把 `adv_ind` 覆盖为 `"positive"/"negative"`（WCM 分位数标签）。

## 2.5 执行状态（2026-08-26）

| 步骤 | 状态 | 产物/结果 |
|---|---|---|
| ① 推断 success_labels | ✅ 已执行 | `rollout/success_labels.json` = `{0:True, 1:True, 2:True, 3:True, 4:False}`（4 成 1 败） |
| ② add_returns | ✅ 已执行 | `/home/user/code_1/WCM/rollout_with_return/`（5 集 1453 帧，task_max=435） |
| ③ WCM 推理 | ⏳ 待执行 | 命令见 §5 |
| ④ FK 转换 | ⏳ 待执行 | 命令见 §6 |
| ⑤ pi05 训练 | ⏳ 待执行 | 命令见 §7 |

## 3. 步骤 ①：推断 success_labels.json

```bash
cd /home/user/code_1/WCM && .venv/bin/python - <<'EOF'
import json, glob
import pyarrow.parquet as pq
root = "/home/user/lerobot_xiazhi/datasets/rollout"
labels = {}
for f in sorted(glob.glob(root + "/data/**/episode_*.parquet", recursive=True)):
    t = pq.read_table(f, columns=["episode_index", "reward_label"])
    ep = int(t.column("episode_index").to_pylist()[0])
    last = float(t.column("reward_label").to_pylist()[-1])   # 末帧 0=成功, -1=失败
    labels[str(ep)] = (last == 0.0)
json.dump(labels, open(root + "/success_labels.json", "w"), indent=2)
print("success:", labels)
EOF
```

## 4. 步骤 ②：add_returns（加 return 列，保持 v2.1 不迁移）

```bash
cd /home/user/code_1/WCM && \
.venv/bin/python scripts/add_returns_to_lerobot_v21.py \
  --root /home/user/lerobot_xiazhi/datasets/rollout \
  --output-dir /home/user/code_1/WCM/rollout_with_return \
  --success-labels /home/user/lerobot_xiazhi/datasets/rollout/success_labels.json \
  --failure-penalty 100 \
  --normalization task_max
```

- 与 WCM 训练数据同参数（failure_penalty=100, task_max），保证 value 标尺一致。
- 输出 `rollout_with_return/`（新目录，含 `return`/`return_raw`/`episode_success`）。

## 5. 步骤 ③：WCM 推理（写 value / advantage / acp_indicator）

```bash
cd /home/user/code_1/WCM && \
.venv/bin/python scripts/wcm_infer_acp_to_dataset.py \
  --checkpoint outputs/wcm_toiletButton_merged_0814/deploy.pt \
  --dataset-root /home/user/code_1/WCM/rollout_with_return \
  --batch-size 16 --num-workers 4
```

- 默认参数即 0814 数据集所用的：n-step advantage（n=50, γ=1）、每任务 top 30% 为正。
- 跑完验证：`complementary_info.acp_indicator` 在 parquet 中已写、成功集有正有负、失败集以负为主。
- 想调 positive_ratio / n_step / TD 模式，在此步用对应参数，重跑本步 + 步骤④ 即可。

## 6. 步骤 ④：FK 关节角 → 四元数 pistar 格式（adv_ind 自动映射）

```bash
cd /home/user/code_1/pistar && \
./venv/bin/python scripts/convert_rollout_to_quaternion.py \
  --src /home/user/code_1/WCM/rollout_with_return \
  --dst /home/user/.cache/huggingface/lerobot/TCP/rollout_round1_wcm_pistar
```

- FK 用 `piper_no_gripper_description.urdf`（pytorch_kinematics），丢弃 gripper 维度，输出 `[x,y,z,qx,qy,qz,qw]`（cm + 四元数）。
- **警告：`--dst` 若已存在会被脚本直接 `rmtree` 重建，重跑前确认没有重要数据。**
- 扁平 key：`image/wrist_image/state/actions`；保留 `intervention/value_label/reward/reward_label/adv_ind` + WCM 三列。
- `adv_ind`：源含 `complementary_info.acp_indicator` → 覆盖为 `positive/negative`；否则保留源值。
- `value_label/reward/reward_label` 沿用 WCM return 语义（失败集末帧 -1，与 pistar `rewrite_rewards.py` 一致）。
- 若源是角度制输入，加 `--is-degree`。

## 7. 步骤 ⑤：训练 pi05

**7.1 注册配置**：编辑 `/home/user/code_1/pistar/src/openpi/training/config.py`，
把 `pi05_piper_history` 的 `data.repo_id` 改成新数据集（建议复制一份 config 命名 `pi05_piper_history_wcm`，保留原 VLM 版对照）：

```python
data=LeRobotPiperDataConfig(
    repo_id="/home/user/.cache/huggingface/lerobot/TCP/rollout_round1_wcm_pistar",
    history_offsets=piper_policy.PIPER_HISTORY_OFFSETS,
    extra_delta_transform=True,
    use_quantile_norm=False,
    base_config=DataConfig(action_sequence_keys=("actions",), prompt_from_task=True),
),
```

**7.2 算 norm_stats**（新数据集必须，否则 data_loader 报错）：

```bash
cd /home/user/code_1/pistar && \
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python scripts/compute_norm_stats.py --config-name=pi05_piper_history_wcm
```

**7.3 训练**：

```bash
cd /home/user/code_1/pistar && \
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
./venv/bin/python scripts/train.py pi05_piper_history_wcm \
  --exp-name=rollout_round1_wcm \
  --overwrite --batch-size=4 --num-train-steps=100 --save-interval=90 \
  --num-workers=2 --no-wandb-enabled
```

**7.4 推理**：`serve_policy.py` 用 `pi05_piper_history_infer`（或对应新 config）＋ `--default_adv_ind positive`（同 value命令.md）。

## 8. 验证点 / 风险点

1. **图像 shape 声明**：rollout 是 `[3,480,640]`(CHW)，WCM 训练数据是 `[480,640,3]`(HWC) —— 视频解码本身是 HWC，
   若 WCM 推理加载报 shape 不匹配，改 info.json 的 features shape 为 `[480,640,3]` 即可（视频文件不变）。
2. **state 语义一致**：WCM 是关节角、rollout 也是关节角 → 同分布，可直接推理。切勿拿四元数 pistar 数据喂 WCM checkpoint。
3. **schema 校验**：`wcm_infer_acp_to_dataset.py` 对 checkpoint schema_version 强校验，用 0814 的 `deploy.pt` 无问题。
4. **5 集小样本**：WCM 推理的 positive_ratio 是任务内分位数，5 集（4 成 1 败）分布可能偏；确认 adv_ind 正负比例再训。
5. **成功标签推断**：`reward_label` 末帧 = `value_label[-1]`（0 或 -1），推断可靠；若后续数据带显式 success 列则优先用 `--success-key`。
