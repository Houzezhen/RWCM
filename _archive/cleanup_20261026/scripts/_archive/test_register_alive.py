#!/usr/bin/env python
"""决定性测试：寄存 token 是死参数吗？
对 D2 checkpoint 的 vision_encoder.register_tokens 做大幅扰动（×10），
比较前向输出（vision 路径）是否变化。
- 输出不变 → 寄存未接入计算图（死参数），"D2 寄存实验"实为"双程结构实验"
- 输出变化 → 寄存参与前向（此时 std≈0.022 说明梯度极弱但路径存在）
同时对比：扰动 backbone 最后一层参数（应显著改变输出，作为阳性对照）。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_critic.config import LanguageConfig, ModelConfig, VisionConfig  # noqa: E402
from world_critic.model import WorldCriticModel  # noqa: E402

CKPT = "outputs/wcm_reg4_r2/checkpoints/best.pt"

payload = torch.load(CKPT, map_location="cpu", weights_only=False)
mc = payload["config"]["model"]
model_cfg = ModelConfig(**{k: v for k, v in mc.items() if k not in ("vision", "language")})
model_cfg.language = LanguageConfig(**mc["language"])
model_cfg.vision = VisionConfig(**mc["vision"])

model = WorldCriticModel(model_cfg)
sd = payload["model"]
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

B, T, V, C, H, W = 1, 4, 2, 3, 224, 224
images = torch.randn(B, T, V, C, H, W, device=device)
actions = torch.randn(B, T, 7, device=device)
ids = torch.ones(1, 16, dtype=torch.long, device=device)
mask = torch.ones_like(ids)
valid = torch.ones(B, T, device=device)


def forward_vision_only():
    with torch.no_grad():
        out = model.vision_encoder(images)
    return out


# 基准
base = forward_vision_only()
print(f"vision 输出 shape={tuple(base.shape)}  base[0,0,0,:4]={base[0,0,0,:4].tolist()}")

# ① 扰动寄存参数（×10）
rt = model.vision_encoder.register_tokens
assert rt is not None, "K=0 模型没有寄存，用错 checkpoint"
orig_rt = rt.data.clone()
with torch.no_grad():
    rt.data = orig_rt * 10.0
pert_reg = forward_vision_only()
delta_reg = (pert_reg - base).abs().max().item()
print(f"\n① 寄存×10   : max|Δ| = {delta_reg:.3e}")
with torch.no_grad():
    rt.data = orig_rt
# 恢复检查
assert torch.allclose(forward_vision_only(), base), "恢复失败"

# ② 阳性对照：扰动最后一层 attention out_proj（应有显著变化）
last = model.vision_encoder.backbone.layers[-1]
w = last.attention.output.dense if hasattr(last.attention, "output") else None
target = w if w is not None else last.attention.o_proj
orig_w = target.weight.data.clone()
with torch.no_grad():
    target.weight.data = orig_w * 1.01
pert_w = forward_vision_only()
delta_w = (pert_w - base).abs().max().item()
print(f"② out_proj×1.01: max|Δ| = {delta_w:.3e}")
with torch.no_grad():
    target.weight.data = orig_w

print(f"\n结论：寄存扰动/对照扰动 = {delta_reg / (delta_w + 1e-12):.2e}")
print("若比值 ~0 → 寄存是死参数；若 ~O(0.1-1) → 寄存参与前向")
