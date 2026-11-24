"""Frozen pretrained video-model front-ends for the world-critic model.

Two families are supported as drop-in replacements for the trainable ViT
history encoder:

- ``vjepa2``: V-JEPA 2 (Bardes et al., 2025), a non-generative predictive
  video world model trained by masked latent prediction on 1M+ hours of
  video. Open weights, CC-BY-NC license.
- ``videomae``: VideoMAE (Tong et al., 2022), a reconstructive masked video
  autoencoder. Serves as the "video-pretrained but non-predictive" control.

Both are kept frozen; only the projection to the critic latent space and all
downstream WCM modules train. The front-end consumes the same sparse history
frames as the SpaceTime encoder and returns per-tubelet patch tokens
[B, T//2, V, P, latent_dim], so the SpaceTime reducer (and the rest of the
graph) can run unchanged on top.
"""
from __future__ import annotations

import torch
from torch import nn

DEFAULT_FRONTEND_REPOS = {
    "vjepa2": "facebook/vjepa2-vitl-fpc64-256",
    "videomae": "MCG-NJU/videomae-base",
    # V-JEPA 2.1(2026, ViT-L): 自定义建模代码已人工审查
    "vjepa21": "Dev-Jahn/vjepa2.1-vitl-fpc64-384",
    # LeVJEPA(galilai 官方, LeJEPA 家族视频版): 自定义建模代码已人工审查(纯 torch, 无网络/执行调用)
    "levjepa": "galilai-group/LeVJEPA-1.5-Large",
    # LeJEPA(图像版): 官方 galilai-group/lejepa 只开源代码不放权重,
    # 用 OK-AI lite_ssl 复刻(ViT-B/16, IN-1K 300ep, 在线 probe 72.04)
    "lejepa": "OK-AI/lejepa-vitb16-pretrain-in1k",
}


def _load_lejepa(repo_id: str) -> nn.Module:
    """Load the OK-AI LeJEPA reproduction without transformers dynamic modules.

    The repo's modeling code uses absolute imports (``from configuration_vitv2
    import ...``, ``from hf_src... import ...``) which the trust_remote_code
    loader rejects. Instead: snapshot the code+weights, put the snapshot on
    sys.path, and instantiate its PreTrainedModel subclass directly. The code
    has been manually audited (pure torch, no network/exec calls).
    """
    import sys

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id, allow_patterns=["*.py", "config.json", "model.safetensors"]
    )
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)
    from modelling_vitv2 import ViTv2PretrainedModel

    return ViTv2PretrainedModel.from_pretrained(snapshot)


def _remap_videomae_biases(model: nn.Module, repo_id: str) -> None:
    """Restore attention biases dropped by the transformers 5.x rename.

    The original checkpoints use a fused qkv projection without bias plus
    separate q_bias/v_bias tensors. transformers>=5 maps the weights onto
    query/key/value linears but drops q_bias/v_bias, leaving zero-initialized
    biases. Load the raw checkpoint and copy them back.
    """
    from huggingface_hub import hf_hub_download

    try:
        from safetensors.torch import load_file

        raw = load_file(hf_hub_download(repo_id, "model.safetensors"))
    except Exception:
        filename = hf_hub_download(repo_id, "pytorch_model.bin")
        raw = torch.load(filename, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    loaded = 0
    with torch.no_grad():
        for key, value in raw.items():
            if key.endswith("attention.attention.q_bias"):
                bias = "query.bias"
            elif key.endswith("attention.attention.v_bias"):
                bias = "value.bias"
            else:
                continue
            stem = key.replace("videomae.", "").rsplit(".", 1)[0]
            parameter = parameters.get(f"{stem}.{bias}")
            if parameter is not None and parameter.shape == value.shape:
                parameter.copy_(value)
                loaded += 1
    if loaded == 0:
        raise RuntimeError(f"No VideoMAE attention biases were restored from {repo_id}.")


def _slice_videomae_positions(model: nn.Module, frames: int) -> None:
    """Adapt the 16-frame positional embedding to a shorter clip.

    The pretrained positions cover (16/tubelet)=8 temporal blocks of patches
    with no CLS position. For a shorter clip keep the first frames//tubelet
    temporal blocks, matching chronological order.
    """
    embeddings = model.embeddings
    positions = embeddings.position_embeddings
    classes = 0
    blocks = positions.size(1) - classes
    temporal_blocks = embeddings.config.num_frames // embeddings.config.tubelet_size
    spatial = blocks // temporal_blocks
    keep = min(frames // embeddings.config.tubelet_size, temporal_blocks)
    if keep == temporal_blocks:
        return
    with torch.no_grad():
        kept = positions[:, : classes + keep * spatial].clone()
        embeddings.position_embeddings = nn.Parameter(kept)
    embeddings.config.num_frames = frames
    embeddings.num_patches = keep * spatial


class VideoFrontendEncoder(nn.Module):
    """Frozen video front-end producing projected patch tokens per tubelet."""

    def __init__(self, config) -> None:
        super().__init__()
        from transformers import VideoMAEModel, VJEPA2Model

        kind = config.video_frontend
        repo = config.frontend_model_name or DEFAULT_FRONTEND_REPOS[kind]
        self.kind = kind
        if kind == "vjepa2":
            self.backbone = VJEPA2Model.from_pretrained(repo)
        elif kind in ("levjepa", "vjepa21"):
            # 两者的建模代码均来自各自模型卡并经人工审查(纯 torch, 无网络/执行调用)
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(repo, trust_remote_code=True)
        elif kind == "lejepa":
            # OK-AI 复刻: 代码已人工审查(纯 torch, 无网络/执行调用),
            # 但其绝对导入不兼容 trust_remote_code, 走 _load_lejepa
            self.backbone = _load_lejepa(repo)
        elif kind == "videomae":
            self.backbone = VideoMAEModel.from_pretrained(repo)
            _remap_videomae_biases(self.backbone, repo)
            _slice_videomae_positions(self.backbone, config.spacetime_frame_count)
        else:
            raise ValueError(f"Unsupported video_frontend={kind!r}.")
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        cfg_b = self.backbone.config
        self.frontend_image_size = (
            getattr(cfg_b, "image_size", None) or getattr(cfg_b, "img_size", None)
            or getattr(cfg_b, "crop_size", None) or 224
        )
        self.frontend_hidden = (
            getattr(cfg_b, "hidden_size", None) or getattr(cfg_b, "embed_dim", None)
        )
        self.tubelet = getattr(cfg_b, "tubelet_size", None) or (1 if kind == "lejepa" else 2)
        self.patch = getattr(cfg_b, "patch_size", 16)
        self.has_cls_token = kind == "levjepa"  # LeVJEPA hidden 首位是 CLS(pooled 取 hidden[:,0])
        self.latent_dim = config.latent_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(self.frontend_hidden), nn.Linear(self.frontend_hidden, config.latent_dim)
        )

    @torch.no_grad()
    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """images [B,T,V,C,H,W] (dataset order current-to-oldest) ->
        projected tokens [B,T//2,V,P,latent] in the same frame order."""
        if images.ndim != 6:
            raise ValueError(f"video front-end expects [B,T,V,C,H,W], got {images.shape}")
        batch, frames, views = images.shape[:3]
        if frames % self.tubelet != 0:
            raise ValueError(
                f"video front-end needs frames divisible by tubelet={self.tubelet}, got {frames}."
            )
        channels, height, width = images.shape[3:]
        if height != self.frontend_image_size or width != self.frontend_image_size:
            images = torch.nn.functional.interpolate(
                images.reshape(batch * frames * views, channels, height, width).float(),
                size=(self.frontend_image_size, self.frontend_image_size),
                mode="bilinear",
                align_corners=False,
            ).to(images.dtype).reshape(
                batch, frames, views, channels, self.frontend_image_size, self.frontend_image_size
            )
        # Chronological order before tubelet pairing, restored afterwards.
        videos = images.flip(1).permute(0, 2, 1, 3, 4, 5).reshape(
            batch * views, frames, channels, self.frontend_image_size, self.frontend_image_size
        )
        if self.kind in ("levjepa", "vjepa21"):  # 这两者约定 [B, C, T, H, W]
            videos = videos.permute(0, 2, 1, 3, 4)
        if self.kind == "lejepa":
            # 图像前端: 逐帧编码(videos 此刻为时序正序 [B*V, T, C, H, W]),
            # 模型输出 dict 的 patch_latent = 每帧 patch tokens。
            outputs = self.backbone(videos.flatten(0, 1))  # forward(x) 位置参数; pixel_values= 会被 **kwargs 吞掉
            hidden = outputs["patch_latent"] if isinstance(outputs, dict) else outputs.patch_latent
            if hidden is None:
                raise RuntimeError("LeJEPA front-end returned no patch_latent.")
            patches = hidden.size(-2)
            tokens = hidden.view(batch, views, frames, patches, self.frontend_hidden)
            tokens = tokens.permute(0, 2, 1, 3, 4).reshape(
                batch, frames, views, patches, self.frontend_hidden
            ).flip(1)  # 恢复 dataset 约定 current-to-oldest
            return self.projection(tokens)
        hidden = self.backbone(videos).last_hidden_state  # [B*V, 1?+T'P, H]
        if self.has_cls_token:
            hidden = hidden[:, 1:]
        t_prime = frames // self.tubelet
        patches = hidden.size(1) // t_prime
        grid = int(round(patches ** 0.5))
        tokens = hidden.view(batch, views, t_prime, grid, grid, self.frontend_hidden)
        tokens = tokens.permute(0, 2, 1, 3, 4, 5).reshape(
            batch, t_prime, views, patches, self.frontend_hidden
        ).flip(1)
        return self.projection(tokens)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_tokens(images)
