"""metaworld button v2 npz+并排mp4 → LeRobot v2.1 数据集(含 return 标签)。

v2 结构: 任务段(~61步, 成败同长) + 驻留(5步) + 复位(50步), 无阻尼按钮回弹。
标签(对齐 v1 设计, 归一化改用 TASK_MAX=61):
  - 任务段: pi*0.6 return, task_max 归一化(成功 -(L-1-t)/T→0, 失败 clip -1)
  - 驻留/复位段: absorbing —— 保持终值(成功 0.0 / 失败 -1.0)。
    复位段画面回到初始, 单帧不可分辨成败, 必须依赖历史。
state/action: 任务段真实; 驻留+复位段 state 冻结在末帧, action=0。
输出: dataset_trials/lerobot/metaworld_button_v2_<split>/
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, "/home/user/code_1/WCM")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

BASE = Path("/home/user/code_1/WCM/dataset_trials")
OUTBASE = BASE / "lerobot"
TASK_TEXT = "press the button"
TASK_MAX = 70           # 任务段超时(右位成功P99+8)
FPS = 20

SETS = [
    ("metaworld_button_v2_train", "metaworld_button_v2_train"),
    ("metaworld_button_v2_test", "metaworld_button_v2_test"),
    ("metaworld_button_v2_ood", "metaworld_button_v2_ood"),
]

FEATURES = {
    "observation.images.hand": {"dtype": "video", "shape": (3, 480, 480), "names": ["channels", "height", "width"]},
    "observation.images.ext": {"dtype": "video", "shape": (3, 480, 480), "names": ["channels", "height", "width"]},
    "observation.state": {"dtype": "float32", "shape": (39,), "names": None},
    "action": {"dtype": "float32", "shape": (4,), "names": None},
    "reward": {"dtype": "float32", "shape": (1,), "names": None},
    "return": {"dtype": "float32", "shape": (1,), "names": None},
    "episode_success": {"dtype": "int8", "shape": (1,), "names": None},
}


def episode_return_labels(task_len: int, success: bool, total_frames: int):
    """任务段 pi*0.6 task_max 归一化 + absorbing 驻留/复位段。"""
    t = np.arange(task_len, dtype=np.float32)
    if success:
        raw = -np.arange(task_len - 1, -1, -1, dtype=np.float32)  # G_t 含 r_t, 末端 0
    else:
        raw = -float(TASK_MAX) - np.arange(task_len - 1, -1, -1, dtype=np.float32)
    norm = np.clip(raw / TASK_MAX, -1.0, 0.0)
    absorb = np.full(total_frames - task_len, 0.0 if success else -1.0, dtype=np.float32)
    return np.concatenate([norm, absorb])


def convert(src_name: str, out_name: str):
    src = BASE / src_name
    out = OUTBASE / out_name
    if out.exists():
        print(f"[skip] {out} 已存在")
        return
    npz_files = sorted(src.glob("episode_*.npz"))
    # 传感器噪声注入(与真机对齐): 真机末端有大的无规律方差(运动+噪声), 淹没成败相关
    # 的毫米级残差 —— 仿真无噪声时这些残差可被线性读出(state 泄题)。每维 std = 全局
    # std × NOISE_FRAC, 固定种子可复现; 注入在转换层, 不动原始 npz。
    NOISE_FRAC = 0.02
    NOISE_ABS_CORE = 0.01   # EE/按钮/夹爪维(0:7 及其帧栈副本 18:25) 绝对噪声 1cm, 对齐真机实测
    rng = np.random.default_rng(3072)
    all_obs = np.concatenate([np.load(f)["obs"].reshape(-1, 39) for f in npz_files])
    noise_std = NOISE_FRAC * all_obs.std(axis=0)
    for d in list(range(0, 7)) + list(range(18, 25)):
        noise_std[d] = max(noise_std[d], NOISE_ABS_CORE)
    print(f"[noise] 每维注入 std = {np.round(noise_std[:4], 4)} (核心维绝对 {NOISE_ABS_CORE})")
    repo_id = f"local/{out_name}"
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=FPS, root=str(out), features=FEATURES,
        image_writer_threads=4,
    )
    n_succ = 0
    for f in npz_files:
        d = np.load(f)
        obs = d["obs"] + rng.normal(0.0, 1.0, d["obs"].shape) * noise_std
        act, rew = d["act"], d["rew"].astype(np.float32)
        success = bool(d["success"])
        n_succ += success
        task_len = len(rew)

        reader = imageio.get_reader(str(f.with_suffix(".mp4")))
        frames = [np.asarray(fr) for fr in reader]  # (T, 480, 960, 3) 并排
        total = len(frames)
        rets = episode_return_labels(task_len, success, total)

        for i, fr in enumerate(frames):
            j = min(i, task_len - 1)  # 动作/奖励指针: 驻留/复位段冻结末帧
            # state: 全程用采集录下的真实 obs(含复位段), 与真机数据一致——
            # 手臂物理回撤使 state 自然趋于初始, 天然对称, 无人工替换。
            state = obs[min(i, len(obs) - 1)]
            a = act[j] if i < task_len else np.zeros(4, dtype=np.float32)
            r = rew[j] if i < task_len else np.float32(0.0)
            hand = np.ascontiguousarray(fr[:, :480])       # 左: 腕部
            ext = np.ascontiguousarray(fr[:, 480:])        # 右: 外部(已翻转)
            ds.add_frame({
                "task": TASK_TEXT,
                "observation.images.hand": torch.from_numpy(hand.transpose(2, 0, 1)),
                "observation.images.ext": torch.from_numpy(ext.transpose(2, 0, 1)),
                "observation.state": torch.from_numpy(state.astype(np.float32)),
                "action": torch.from_numpy(a.astype(np.float32)),
                "reward": torch.tensor([r], dtype=torch.float32),
                "return": torch.tensor([rets[i]], dtype=torch.float32),
                "episode_success": torch.tensor([1 if success else 0], dtype=torch.int8),
            })
        ds.save_episode()
        print(f"[{out_name}] {f.stem}: {total}帧 succ={success}", flush=True)
    print(f"[{out_name}] 完成: {len(npz_files)} eps, 成功 {n_succ} / 失败 {len(npz_files)-n_succ}")
    del ds


if __name__ == "__main__":
    wanted = sys.argv[1:] or [name for name, _ in SETS]
    for src_name, out_name in SETS:
        if out_name in wanted or src_name in wanted:
            convert(src_name, out_name)
    print("all done")
