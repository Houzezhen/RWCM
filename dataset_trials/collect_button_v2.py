"""v2 正式采集: metaworld button-press, 成败同结构、无假失败、无阻尼回弹。

结构(每条): 任务段(61步, 成功在~55-61步触发, 失败固定61步) + 驻留(5步) + P控复位(50步)。
- 成功: 脚本策略+小噪声(0~0.05), success 触发即止; 按钮物理修复后跟随按压、手离开3帧弹回。
- 失败: 与成功同策略同噪声跑到 dev_step(默认U(24,30), 接触前), 然后按模式偏移到61步:
    stop     临按钮停住
    side_l/r 按到按钮同高度、左右各0.06m的盒面(两段式: 先横移再前推)
    over     从按钮上方推过, 顶到盒子上沿
    retreat  按空即撤(P控直接回撤)
- 假失败校验门: 失败集按钮全程位移|qpos|<0.02 且 env success=False; 成功集必须 success=True。
  超标重试(逐步提前 dev_step, 最早20), 最多8次。
- OOD: 未见过的偏移配置(更晚 dev_step U(30,36)、侧向偏移 0.09m、更大噪声)。

用法: python collect_button_v2.py <split>   # split ∈ train|test|ood
输出: dataset_trials/metaworld_button_v2_<split>/episode_*.npz + .mp4 + stats.json
"""
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import metaworld
import numpy as np
import mujoco
import scipy.spatial.transform as st
from metaworld.policies.sawyer_button_press_v3_policy import SawyerButtonPressV3Policy

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
OUT = Path(f"/home/user/code_1/WCM/dataset_trials/metaworld_button_v2_{SPLIT}")
OUT.mkdir(parents=True, exist_ok=True)

SPECS = {  # (成功数, 失败数, 成功噪声档, 失败dev范围, 侧向偏移)
    "train": (50, 50, [0.0, 0.02, 0.05], (24, 30), 0.06),
    "test":  (10, 10, [0.0, 0.02, 0.05], (24, 30), 0.06),
    # OOD: 协议与 test 完全一致, 唯一差异是按钮颜色(纯视觉 OOD)
    "ood":   (10, 10, [0.0, 0.02, 0.05], (24, 30), 0.06),
}[SPLIT]

ml1 = metaworld.ML1("button-press-v3")
policy = SawyerButtonPressV3Policy()

ENVS = {}
for key, cam in [("hand", "behindGripper"), ("ext", "corner2")]:
    env = ml1.train_classes["button-press-v3"](render_mode="rgb_array", camera_name=cam)
    env.set_task(ml1.train_tasks[0])
    env._auto_reset = False
    ENVS[key] = env

import numpy.linalg as la

cid = mujoco.mj_name2id(ENVS["hand"].data.model, mujoco.mjtObj.mjOBJ_CAMERA, "behindGripper")


def lookat_quat(f):
    z = -f / la.norm(f)
    x = np.cross([0, 0, 1], z)
    if la.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= la.norm(x)
    y = np.cross(z, x)
    return st.Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()


FWD = np.array([-0.529, 0.848, -0.017])
q = lookat_quat(FWD)
ENVS["hand"].data.model.cam_mode[cid] = 0
ENVS["hand"].data.model.cam_fovy[cid] = 75
ENVS["hand"].data.model.cam_pos[cid] = np.array([-0.06, 0.0, 0.05])
ENVS["hand"].data.model.cam_quat[cid] = np.array([q[3], q[0], q[1], q[2]])

TASK_LEN, HOLD, RETURN_N = 70, 5, 80
Q_HOME = np.array([1.9873, -0.9477, -0.957, 2.146, 0.7572, 0.7621, 2.2734])  # 参考 home 关节角  # 超时=max(P99=59,历史异常观测62)+8余量; 成功即停, 失败超时逐集55~70  # RETURN_N=80 上限, 实际由收敛判据提前停止
FAIL_MODES = ["stop", "side_l", "side_r", "over", "retreat"]
MAX_BTN_MOVE, MAX_TRIES = 0.02, 8

# ---- 按钮物理修复: 无阻尼快回弹(详见 sample 脚本注释) ----
for env in ENVS.values():
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "btnbox_joint")
    adr = env.model.jnt_qposadr[jid]
    env.model.jnt_stiffness[jid] = 100.0
    env.model.dof_damping[env.model.jnt_dofadr[jid]] = 1.0
    env.model.qpos_spring[adr] = 0.0
    env.model.jnt_range[jid] = np.array([-0.09, 0.0])
    btn_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "button")
    box_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "buttonbox")
    for g in range(env.model.ngeom):
        if env.model.geom_bodyid[g] in (btn_bid, box_bid):
            env.model.geom_conaffinity[g] = 0


# OOD 唯一差异: 按钮换成黄色。注意默认红色来自材质 mat_rgba(改 geom_rgba 无效);
# 场景里机械臂连杆是暗红色、按钮内芯半透明蓝, 黄色全场唯一, 便于校验。
if SPLIT == "ood":
    for env in ENVS.values():
        btn_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "button")
        for g in range(env.model.ngeom):
            if env.model.geom_bodyid[g] == btn_bid:
                yellow = np.array([0.95, 0.85, 0.1, 1.0])
                env.model.geom_rgba[g] = yellow
                mat = env.model.geom_matid[g]
                if mat >= 0:
                    env.model.mat_rgba[mat] = yellow
        # 场景变色: 地板/桌面/墙→蓝灰, 控制器盒/底座→深蓝
        for g in range(env.model.ngeom):
            body = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[g])
            if body in ("world", "tablelink", "RetainingWall"):
                env.model.geom_rgba[g] = np.array([0.3, 0.35, 0.5, 1.0])
            elif body in ("controller_box", "pedestal_feet", "torso", "pedestal"):
                env.model.geom_rgba[g] = np.array([0.15, 0.2, 0.35, 1.0])
        # 外部相机偏移(corner2 平移+微转, 打掉空间位置线索)
        cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "corner2")
        env.model.cam_pos[cid] = np.array([1.5, -0.4, 1.3])  # 原始 [1.3,-0.2,1.1]


def btn_qpos():
    env = ENVS["hand"]
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "btnbox_joint")
    return float(env.data.qpos[env.model.jnt_qposadr[jid]])


def render(key):
    img = ENVS[key].render()
    img = np.flipud(img) if key == "ext" else img
    # 相机噪声(注入σ=3, H.264编码后≈1.5, 对齐真机实测): 逐帧独立高斯像素噪声 + 亮度微抖,
    # 封掉"可训练编码器从复位帧毫米级视觉痕迹读成败"的通道(消融已证图像是主泄题通道)
    img = img.astype(np.float32) + rng.normal(0.0, 3.0, img.shape) + rng.uniform(-1.0, 1.0)
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    # 逐帧随机 ±1px 平移(相机抖动): 直接打散毫米级痕迹赖以存在的亚像素位置线索 ——
    # 白噪声是逐像素独立的, 压不住空间相关的结构信号(网络可跨像素平均去噪)
    sh = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    oy, ox = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    return sh[oy:oy + img.shape[0], ox:ox + img.shape[1]]


HOME_TARGET = None  # 每集随机家(真机操作员"差不多到家", 厘米级散布, 与成败无关)


def home_action():
    ee = ENVS["hand"]._get_obs()[:3]
    d = np.clip((HOME_TARGET - ee) * 8.0, -1, 1)
    # 夹爪推到统一全开(-1=张开; metaworld 里 +1 是闭合, 0 是保持): 任务段指令 0 会把
    # 成败集冻结在不同开度(0.93/0.99, d'=11 泄题), 集末也须回到与集首一致的全开(画面对称)
    return np.array([d[0], d[1], d[2], -1.0])


def run_episode(kind, mode, noise, dev_step, side_offset):
    obs = {}
    for k, env in ENVS.items():
        obs[k], _ = env.reset()
    global HOME_TARGET
    HOME_TARGET = np.array([0.0, 0.4, 0.18]) + rng.uniform(-0.02, 0.02, 3)
    # 按钮位置随机化(对齐真机左/中/右三位置 + 安装抖动): 每集从 {-0.12,0,+0.12}+U(-0.01,0.01)
    # 抽一个 x 偏移, 两个相机环境同步移动, 成功判据靶点(hole site)跟随刷新
    dx = float(rng.choice([-0.08, 0.0, 0.08]) + rng.uniform(-0.01, 0.01))
    for env in ENVS.values():
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        env.model.body_pos[bid][0] += dx
        mujoco.mj_forward(env.model, env.data)
        env._target_pos = env._get_site_pos("hole")
    obs = {k: env._get_obs() for k, env in ENVS.items()}
    env0 = ENVS["hand"]
    mujoco.mj_forward(env0.model, env0.data)
    gid = mujoco.mj_name2id(env0.model, mujoco.mjtObj.mjOBJ_GEOM, "btnGeom")
    btn_pos = env0.data.geom_xpos[gid].copy()
    side = -1.0 if mode == "side_l" else 1.0
    side_target = btn_pos + np.array([side * side_offset, 0.0, 0.0])
    side_stage = None
    frames = {k: [render(k)] for k in ENVS}
    obs_seq, act_seq, rew_seq, btn_trace = [], [], [], []
    info, success_at = {}, None

    # 失败段超时逐集抽取(55~67), 不与成功段长度机械同步 —— 真机失败段长度本就散布
    task_timeout = TASK_LEN if kind != "fail" else int(rng.integers(55, TASK_LEN + 1))  # 失败散布55~70
    for i in range(task_timeout):
        a = policy.get_action(obs["hand"]).copy() + rng.normal(0, noise, 4)
        if kind == "fail" and i >= dev_step:
            if mode == "stop":
                a = np.zeros(4)
            elif mode in ("side_l", "side_r"):
                ee = obs["hand"][:3]
                if side_stage is None:
                    side_stage = np.array([side_target[0], ee[1], side_target[2]])
                if i < dev_step + 12:
                    target = side_stage
                else:
                    target = np.array([side_target[0], side_target[1] + 0.02, side_target[2]])
                a = np.clip(np.concatenate([8.0 * (target - ee), [0.0]]), -1, 1)
            elif mode == "over":
                a[2] = 0.6
            elif mode == "retreat":
                a = home_action()
        obs_seq.append(obs["hand"].copy()); act_seq.append(np.clip(a, -1, 1))
        btn_trace.append(btn_qpos())
        new_obs = {}
        for k, env in ENVS.items():
            out = env.step(np.clip(a, -1, 1))
            if len(out) == 5:
                new_obs[k], r, term, tr, info = out
            else:
                new_obs[k], r, done, info = out; term, tr = done, False
        obs = new_obs
        rew_seq.append(float(r))
        for k in ENVS:
            frames[k].append(render(k))
        if info.get("success"):
            success_at = i
            break

    success = success_at is not None
    btn_trace.append(btn_qpos())
    at_home = 0
    for _ in range(HOLD):
        for env in ENVS.values():
            env.step(np.zeros(4))
        btn_trace.append(btn_qpos())
        obs_seq.append(ENVS["hand"]._get_obs().copy())  # 复位段真实 state(与真机一致, 天然对称)
        for k in ENVS:
            frames[k].append(render(k))
    for _ in range(RETURN_N):
        a = home_action()
        for env in ENVS.values():
            env.step(a)
        btn_trace.append(btn_qpos())
        obs_seq.append(ENVS["hand"]._get_obs().copy())
        for k in ENVS:
            frames[k].append(render(k))
        # 收敛判据: EE 连续 12 帧到家(<5mm)提前结束(未触发则走满 RETURN_N, 无害)
        ee = ENVS["hand"]._get_obs()[:3]
        if np.linalg.norm(ee - HOME_TARGET) < 0.005:
            at_home += 1
            if at_home >= 12:
                break
        else:
            at_home = 0

    # 关节空间钉定(无条件执行): 真机回家是关节空间的固定姿态(home例程),
    # EE空间目标的IK是路径依赖的, 不同失败模式留下不同的手臂构型(图像泄题主通道)。
    # 每集以参考home关节角+伺服级噪声(±0.02rad)锁定末10帧, 各模式姿态完全统一。
    q_noise = rng.normal(0.0, 0.02, 7)
    for _ in range(10):
        for env in ENVS.values():
            env.data.qpos[:7] = Q_HOME + q_noise
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
        btn_trace.append(btn_qpos())
        obs_seq.append(ENVS["hand"]._get_obs().copy())
        for k in ENVS:
            frames[k].append(render(k))

    return dict(mode=mode, success=success, success_at=success_at,
                task_len=len(rew_seq), total=len(frames["hand"]),
                obs=obs_seq, act=act_seq, rew=rew_seq, frames=frames, btn_trace=btn_trace)


rng = np.random.default_rng({"train": 2026, "test": 31337, "ood": 90210}[SPLIT])
N_SUCC, N_FAIL, NOISES, DEV_RANGE, SIDE_OFF = SPECS
stats, ep = [], 0
t0 = time.time()


def save(rec, ep, dev):
    np.savez_compressed(OUT / f"episode_{ep:03d}.npz",
                        obs=np.array(rec["obs"]), act=np.array(rec["act"]),
                        rew=np.array(rec["rew"]), success=np.array(rec["success"]),
                        dev_step=np.array(dev if dev is not None else -1),
                        btn_qpos=np.array(rec["btn_trace"]))
    stack = [np.concatenate([rec["frames"]["hand"][i], rec["frames"]["ext"][i]], axis=1)
             for i in range(len(rec["frames"]["hand"]))]
    imageio.mimwrite(str(OUT / f"episode_{ep:03d}.mp4"), stack,
                     fps=20, quality=7, macro_block_size=8)
    stats.append(dict(episode=ep, mode=rec["mode"], success=bool(rec["success"]),
                      task_len=rec["task_len"], total=rec["total"], dev_step=dev,
                      max_btn_move=float(np.abs(rec["btn_trace"]).max())))
    print(f"ep{ep:03d} [{rec['mode']:>7s}] succ={bool(rec['success'])} task={rec['task_len']:2d} "
          f"btn_move={np.abs(rec['btn_trace']).max():.3f} ({time.time()-t0:.0f}s)", flush=True)


n_succ = n_fail = 0
while n_succ < N_SUCC or n_fail < N_FAIL:
    want_succ = n_succ < N_SUCC and (n_fail >= N_FAIL or rng.random() < 0.5)
    noise = float(rng.choice(NOISES))
    if want_succ:
        kind, mode, dev = "succ", "clean", None
    else:
        kind, mode = "fail", FAIL_MODES[n_fail % len(FAIL_MODES)]
        dev = int(rng.integers(DEV_RANGE[0], DEV_RANGE[1] + 1))
    rec = run_episode(kind, mode, noise, dev, SIDE_OFF)
    side_off = SIDE_OFF
    tries = 0
    # 校验门: 想成功必须 success; 想失败必须没成功且按钮没被压(位移<0.02)
    while True:
        ok = rec["success"] if want_succ else (
            (not rec["success"]) and np.abs(rec["btn_trace"]).max() < MAX_BTN_MOVE)
        if ok or tries >= MAX_TRIES:
            break
        tries += 1
        if want_succ:  # 想成功却失败 -> 减噪声
            noise = max(0.0, noise * 0.5)
            rec = run_episode(kind, mode, noise, dev, SIDE_OFF)
        elif mode in ("side_l", "side_r"):
            # 侧压蹭到按钮通常是盒位随机导致偏移量贴边 -> 加大侧向偏移
            side_off = min(side_off + 0.03, 0.15)
            rec = run_episode(kind, mode, noise, dev, side_off)
        else:          # 想失败却按压/成功 -> 逐步提前偏移
            dev = max(dev - 3, 20)
            rec = run_episode(kind, mode, noise, dev, SIDE_OFF)
    if tries >= MAX_TRIES and not ok:
        raise RuntimeError(f"ep{n_succ+n_fail} {mode}: {MAX_TRIES} 次重试未达标 btn_move="
                           f"{np.abs(rec['btn_trace']).max():.3f} succ={rec['success']}")
    save(rec, ep, dev)
    n_succ += bool(rec["success"]); n_fail += not rec["success"]
    ep += 1

(OUT / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
print(f"\n[{SPLIT}] 完成: {ep} 条, 成功 {n_succ} / 失败 {n_fail}, 用时 {(time.time()-t0)/60:.1f} 分钟")
