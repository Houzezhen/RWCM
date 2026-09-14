from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


@dataclass
class WorldCriticOutput:
    """Outputs of a single, structurally safe World Critic forward pass."""

    context_latent: torch.Tensor
    value: torch.Tensor
    next_state_pred: torch.Tensor
    target_next_state: torch.Tensor
    valid_mask: torch.Tensor
    next_state_vector_pred: torch.Tensor | None = None


@dataclass
class LatentRolloutOutput:
    latents: torch.Tensor
    values: torch.Tensor


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class VisionEncoder(nn.Module):
    """Hugging Face ViT wrapper returning one latent per frame and camera."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError("VisionEncoder requires transformers. Install the project dependencies.") from exc

        vision_config = config.vision
        if vision_config.pretrained:
            self.backbone = AutoModel.from_pretrained(vision_config.model_name)
        else:
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(vision_config.model_name))
        hidden_dim = int(getattr(self.backbone.config, "hidden_size"))
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, config.latent_dim),
        )
        self.num_register_tokens = int(getattr(config.vision, "num_register_tokens", 0))
        self.register_insert = str(getattr(config.vision, "register_insert", "late")).lower()
        if self.register_insert not in ("early", "late"):
            raise ValueError(
                f"vision.register_insert must be 'early' or 'late', got {self.register_insert!r}."
            )
        if self.num_register_tokens > 0:
            if not vision_config.trainable:
                raise ValueError("register tokens require a trainable vision backbone.")
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_register_tokens, hidden_dim)
            )
            nn.init.normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None
        self.camera_embedding = nn.Parameter(
            torch.zeros(1, 1, config.max_views, config.latent_dim)
        )
        nn.init.normal_(self.camera_embedding, std=0.02)
        self.backbone.requires_grad_(vision_config.trainable)
        self.trainable = vision_config.trainable

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()
        return self

    @staticmethod
    def _last_hidden_state(output) -> torch.Tensor:
        """Normalize Hugging Face encoder return types to one tensor."""
        if torch.is_tensor(output):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise TypeError(f"Unsupported vision encoder output type: {type(output).__name__}")

    def encoder_layers(self) -> nn.ModuleList:
        """Return the ViT layers across supported Transformers layouts."""
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            layers = getattr(encoder, "layers", None)
        if layers is None:
            layers = getattr(self.backbone, "layers", None)
        if layers is None:
            raise TypeError(
                f"{type(self.backbone).__name__} does not expose ViT encoder layers "
                "as encoder.layer, encoder.layers, or layers."
            )
        return layers

    def _run_encoder(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run one encoder pass without re-running image embeddings."""
        encoder = getattr(self.backbone, "encoder", None)
        if encoder is not None:
            return self._last_hidden_state(encoder(tokens))

        hidden = tokens
        for layer in self.encoder_layers():
            hidden = self._last_hidden_state(layer(hidden))
        return hidden

    def _final_layer_norm(self, tokens: torch.Tensor) -> torch.Tensor:
        layer_norm = getattr(self.backbone, "layernorm", None)
        if layer_norm is None:
            layer_norm = getattr(self.backbone, "post_layernorm", None)
        if layer_norm is None:
            raise TypeError(
                f"{type(self.backbone).__name__} does not expose a final vision LayerNorm."
            )
        return layer_norm(tokens)

    def _encode_flat_tokens(self, flat: torch.Tensor) -> torch.Tensor:
        parameters = inspect.signature(self.backbone.forward).parameters
        kwargs = {"pixel_values": flat}
        if "interpolate_pos_encoding" in parameters:
            kwargs["interpolate_pos_encoding"] = True
        if self.register_tokens is None:
            if self.trainable:
                output = self.backbone(**kwargs)
            else:
                with torch.no_grad():
                    output = self.backbone(**kwargs)
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        elif self.register_insert == "early":
            # 标准寄存器语义（Darcet et al. 2023）：K 个寄存器在第 0 层输入
            # 就拼入 CLS+patch 序列，随全部层一遍流动；输出过 backbone 的
            # final LayerNorm 后取前 1+N 个位置（CLS+patch），寄存器丢弃。
            # 深度/算力与无寄存 baseline 严格一致（修复旧 late 路径的
            # 双倍深度混淆，见实验报告 §8.3）。
            embeddings_kwargs = {"pixel_values": flat}
            if "interpolate_pos_encoding" in inspect.signature(
                self.backbone.embeddings.forward
            ).parameters:
                embeddings_kwargs["interpolate_pos_encoding"] = True
            hidden = self.backbone.embeddings(**embeddings_kwargs)
            tokens = torch.cat(
                [hidden, self.register_tokens.expand(hidden.size(0), -1, -1)], dim=1
            )
            hidden = self._final_layer_norm(self._run_encoder(tokens))
        else:
            # late（旧行为，默认）：backbone 完整 forward 后，把寄存器拼到
            # last_hidden_state 末尾再过一遍全部 encoder 层（等效双倍深度，
            # 且缺 final LayerNorm）。保留以兼容旧 checkpoint 与复现旧结果。
            if self.trainable:
                output = self.backbone(**kwargs)
            else:
                with torch.no_grad():
                    output = self.backbone(**kwargs)
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            tokens = torch.cat(
                [hidden, self.register_tokens.expand(hidden.size(0), -1, -1)], dim=1
            )
            hidden = self._run_encoder(tokens)
        return hidden

    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return projected CLS+patch tokens as [B,T,V,N,D]."""

        if images.ndim == 5:
            images = images.unsqueeze(2)
        if images.ndim != 6:
            raise ValueError(f"Expected images [B,T,V,C,H,W], got {images.shape}")
        batch, time, views, channels, height, width = images.shape
        if views > self.camera_embedding.size(2):
            raise ValueError(
                f"Model has {self.camera_embedding.size(2)} camera slots but received {views} views."
            )
        flat = images.reshape(batch * time * views, channels, height, width)
        hidden = self._encode_flat_tokens(flat)
        tokens = self.projection(hidden)
        return tokens.view(batch, time, views, tokens.size(1), tokens.size(2))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B,T,V,C,H,W], normalized for the configured vision backbone.
        Returns:
            Per-camera latents [B,T,V,D].
        """
        tokens = self.encode_tokens(images)
        batch, time, views = tokens.shape[:3]
        frame_latent = tokens[:, :, :, 0]
        return frame_latent + self.camera_embedding[:, :, :views]


class CrossFrameQueryLayer(nn.Module):
    """Update one current-frame query from all sparse frame patch tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.latent_dim)
        self.context_norm = nn.LayerNorm(config.latent_dim)
        self.attention = nn.MultiheadAttention(
            config.latent_dim,
            config.trunk_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(config.latent_dim)
        self.ffn = MLP(
            config.latent_dim,
            int(config.latent_dim * config.trunk_mlp_ratio),
            config.latent_dim,
            config.dropout,
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class SparseCrossFrameEncoder(nn.Module):
    """VGGT-inspired query aggregation over full sparse-frame patch grids."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.frame_count = config.cross_frame_count
        self.time_embedding = nn.Parameter(
            torch.zeros(1, config.cross_frame_count, 1, config.latent_dim)
        )
        self.layers = nn.ModuleList(
            CrossFrameQueryLayer(config) for _ in range(config.cross_frame_layers)
        )
        self.output_norm = nn.LayerNorm(config.latent_dim)
        nn.init.normal_(self.time_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B,K,V,N,D], ordered current to oldest.
        if tokens.ndim != 5:
            raise ValueError(f"Cross-frame tokens must be [B,K,V,N,D], got {tokens.shape}")
        batch, frames, views, token_count, dim = tokens.shape
        if frames != self.frame_count:
            raise ValueError(f"Expected {self.frame_count} sparse frames, got {frames}.")
        by_view = tokens.permute(0, 2, 1, 3, 4).reshape(
            batch * views, frames, token_count, dim
        )
        context = by_view + self.time_embedding
        context = context.reshape(batch * views, frames * token_count, dim)
        query = by_view[:, 0, :1] + self.time_embedding[:, 0]
        for layer in self.layers:
            query = layer(query, context)
        return self.output_norm(query[:, 0]).view(batch, views, dim)


class LanguageEncoder(nn.Module):
    """CLIP text tower plus a learned adapter into the World Critic latent space."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError("LanguageEncoder requires transformers. Install the project dependencies.") from exc

        language_config = config.language
        model_config = AutoConfig.from_pretrained(language_config.model_name)
        if getattr(model_config, "model_type", None) != "clip":
            raise ValueError(
                "LanguageEncoder requires a full CLIP checkpoint so its pretrained "
                f"text_projection is available; got model_type={getattr(model_config, 'model_type', None)!r}."
            )
        if language_config.pretrained:
            model = AutoModel.from_pretrained(language_config.model_name, config=model_config)
        else:
            model = AutoModel.from_config(model_config)
        self.text_model = getattr(model, "text_model", model)
        hidden_dim = int(getattr(self.text_model.config, "hidden_size"))
        text_projection = getattr(model, "text_projection", None)
        if isinstance(text_projection, torch.Tensor):
            if text_projection.ndim != 2:
                raise TypeError(
                    "CLIP text_projection tensor must be rank-2, got "
                    f"shape={tuple(text_projection.shape)}."
                )
            # OpenAI checkpoints conventionally store [projection, hidden],
            # while a few HF-compatible wrappers expose [hidden, projection].
            # Normalize both layouts into nn.Linear's [out, in] weight.
            if text_projection.size(1) == hidden_dim:
                weight = text_projection
            elif text_projection.size(0) == hidden_dim:
                weight = text_projection.T
            else:
                raise ValueError(
                    "CLIP text_projection tensor does not contain the text "
                    f"hidden dimension {hidden_dim}: shape={tuple(text_projection.shape)}."
                )
            projection_layer = nn.Linear(hidden_dim, weight.size(0), bias=False)
            projection_layer.weight.data.copy_(weight)
            text_projection = projection_layer
        if not isinstance(text_projection, nn.Linear):
            raise TypeError(
                "The configured CLIP checkpoint does not expose the expected text_projection linear layer."
            )
        if text_projection.in_features != hidden_dim:
            raise ValueError(
                "CLIP text projection input dimension does not match the text tower: "
                f"{text_projection.in_features} != {hidden_dim}."
            )
        self.clip_projection = text_projection
        projected_dim = int(text_projection.out_features)
        self.adapter = nn.Sequential(
            nn.LayerNorm(projected_dim),
            nn.Linear(projected_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.text_model.requires_grad_(language_config.trainable)
        self.clip_projection.requires_grad_(language_config.trainable)
        self.trainable = language_config.trainable

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.text_model.eval()
            self.clip_projection.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "CLIP text inputs must be [B,L] with matching attention_mask; "
                f"got input_ids={input_ids.shape}, attention_mask={attention_mask.shape}."
            )
        if input_ids.device != attention_mask.device:
            raise ValueError(
                "CLIP text inputs and attention_mask must be on the same device: "
                f"{input_ids.device} != {attention_mask.device}."
            )
        if not attention_mask.bool().any(dim=1).all():
            raise ValueError("Every instruction must contain at least one valid CLIP token.")
        if self.trainable:
            output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # CLIP's text projection is trained for the pooled EOS representation,
        # not for every intermediate token.  Keep the standard CLIP semantics,
        # then expose the projected instruction as one cross-attention token.
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            last_valid = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_valid]
        projected = self.clip_projection(pooled)
        instruction = self.adapter(projected).unsqueeze(1)
        return instruction, torch.ones(
            instruction.shape[:2], dtype=torch.bool, device=instruction.device
        )


class StateLanguageFusion(nn.Module):
    """State queries attend to CLIP tokens before the action/value branch point."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(config.language.fusion_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "state_norm": nn.LayerNorm(config.latent_dim),
                        "text_norm": nn.LayerNorm(config.latent_dim),
                        "cross_attn": nn.MultiheadAttention(
                            config.latent_dim,
                            config.language.fusion_heads,
                            dropout=config.language.fusion_dropout,
                            batch_first=True,
                        ),
                        "ffn_norm": nn.LayerNorm(config.latent_dim),
                        "ffn": MLP(
                            config.latent_dim,
                            int(config.latent_dim * config.trunk_mlp_ratio),
                            config.latent_dim,
                            config.language.fusion_dropout,
                        ),
                    }
                )
            )

    def forward(
        self,
        state_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = ~text_mask
        if not text_mask.any(dim=1).all():
            raise ValueError("Every instruction must contain at least one valid text token.")
        for layer in self.layers:
            attended, _ = layer["cross_attn"](
                layer["state_norm"](state_tokens),
                layer["text_norm"](text_tokens),
                layer["text_norm"](text_tokens),
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            state_tokens = state_tokens + attended
            state_tokens = state_tokens + layer["ffn"](layer["ffn_norm"](state_tokens))
        return state_tokens


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class TemporalRegisterBlock(nn.Module):
    """时间维 register token（因果版）：跨时间步共享的 token 吸收时序不变信息。

    因果约束：实际部署时第 t 帧只能看到 0..t，因此每个时间步 t 只允许
    register 读取前缀 0..t（attn_mask 屏蔽未来 key），读出信息残差注回
    该帧自身。register 参数在任意前缀长度下共享同一组（跨时间步不变），
    输出只保留注回后的 state token。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_registers = config.num_temporal_registers
        self.registers = nn.Parameter(torch.zeros(1, self.num_registers, config.latent_dim))
        nn.init.normal_(self.registers, std=0.02)
        self.reg_norm = nn.LayerNorm(config.latent_dim)
        self.state_norm = nn.LayerNorm(config.latent_dim)
        self.read_attn = nn.MultiheadAttention(
            config.latent_dim, config.trunk_heads, dropout=config.dropout, batch_first=True
        )
        self.write_norm = nn.LayerNorm(config.latent_dim)
        self.write_proj = nn.Linear(config.num_temporal_registers, 1)

    def forward(self, state_tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # state_tokens: [B,T,D]（已加 time embedding），valid_mask: [B,T]
        batch, time, dim = state_tokens.shape
        registers = self.num_registers
        # 每个时间步 t 都用自己的前缀读取：同一组共享 register 复制到每个 t -> [B, T*K, D]
        query = self.registers.expand(batch, time, registers, dim).reshape(batch, time * registers, dim)
        # 因果 prefix mask：[T*K, T]，行 (t,k) 屏蔽未来 key j>t
        # 注意 attn_mask 语义为 True=屏蔽（与 causal_mask 一致）
        row_time = torch.arange(time * registers, device=state_tokens.device) // registers
        col_time = torch.arange(time, device=state_tokens.device)
        prefix_mask = row_time[:, None] < col_time[None, :]
        read, _ = self.read_attn(
            self.reg_norm(query),
            self.state_norm(state_tokens),
            self.state_norm(state_tokens),
            attn_mask=prefix_mask,
            key_padding_mask=~valid_mask.bool(),
            need_weights=False,
        )
        # read: [B, T*K, D] -> 按时间步展开，沿 K 个 register 池化为单向量，残差注回对应帧
        injection = self.write_proj(
            self.write_norm(read).reshape(batch, time, registers, dim).permute(0, 1, 3, 2)
        ).squeeze(-1)
        return state_tokens + injection


class BlockRegisterFusion(nn.Module):
    """块注意力融合时空 register：时间维块内双向、块间因果（可选稀疏未来泄漏）。

    每个时间步的 token 组 = [state ｜ K_s 个空间寄存 ｜ K_t 个时间寄存]
    （寄存 token 参数跨时间步共享）。按时间切块（block_size 帧/块）：
      - 块内：所有 token（含两类寄存）双向全可见——时空寄存在此融合；
      - 块间：只允许看 ≤ 当前块（时间因果，对齐部署可见性）；
      - future_leak_ratio > 0 时：被屏蔽的未来 token 随机放行该比例
        （BigBird 式随机注意力，训练期提供少量前瞻信息，推理期未来
        token 不存在时这些位置由 key_padding 屏蔽，不产生额外开销）。
    输出丢弃全部寄存 token，只保留注回后的 state token。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_spatial = config.num_spatial_registers
        self.num_temporal = config.num_temporal_registers
        self.block_size = config.register_block_size
        self.future_leak_ratio = config.register_future_leak_ratio
        self.tokens_per_step = 1 + self.num_spatial + self.num_temporal
        self.spatial_registers = nn.Parameter(
            torch.zeros(1, self.num_spatial, config.latent_dim)
        )
        self.temporal_registers = nn.Parameter(
            torch.zeros(1, self.num_temporal, config.latent_dim)
        )
        nn.init.normal_(self.spatial_registers, std=0.02)
        nn.init.normal_(self.temporal_registers, std=0.02)
        self.norm = nn.LayerNorm(config.latent_dim)
        self.attn = nn.MultiheadAttention(
            config.latent_dim, config.trunk_heads, dropout=config.dropout, batch_first=True
        )
        self.out_norm = nn.LayerNorm(config.latent_dim)

    def forward(self, state_tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # state_tokens: [B,T,D]，valid_mask: [B,T]
        batch, time, dim = state_tokens.shape
        device = state_tokens.device
        regs = torch.cat([self.spatial_registers, self.temporal_registers], dim=1)  # [1,K,D]
        # 每时间步交错排布 [state | regs] -> [B, T, 1+K, D] -> 展平 [B, T*(1+K), D]
        steps = torch.cat(
            [state_tokens.unsqueeze(2), regs.expand(batch, time, -1, dim)], dim=2
        )
        tokens = steps.reshape(batch, time * self.tokens_per_step, dim)
        # 块因果 mask：行 i 的块 = (i // tokens_per_step) // block_size
        row_pos = torch.arange(time * self.tokens_per_step, device=device) // self.tokens_per_step
        col_pos = torch.arange(time * self.tokens_per_step, device=device) // self.tokens_per_step
        row_block = row_pos // self.block_size
        col_block = col_pos // self.block_size
        # True=屏蔽：行所在块 < 列所在块（列在未来块）
        attn_mask = row_block[:, None] < col_block[None, :]
        # Legacy non-causal diagnostic: this samples all token-token edges, so
        # the configured ratio is not an observation-level leakage probability.
        if self.future_leak_ratio > 0.0:
            blocked = attn_mask
            num_blocked = int(blocked.sum(dim=1).max().item())
            if num_blocked > 0:
                generator = torch.Generator(device="cpu").manual_seed(0)
                keep = (
                    torch.rand(blocked.shape, generator=generator) < self.future_leak_ratio
                ).to(device)
                attn_mask = blocked & ~keep
        # key_padding：无效时间步的所有 token（state+寄存）都屏蔽
        step_valid = valid_mask.bool()[:, :, None].expand(batch, time, self.tokens_per_step)
        key_padding = ~step_valid.reshape(batch, time * self.tokens_per_step)
        out, _ = self.attn(
            self.norm(tokens), self.norm(tokens), self.norm(tokens),
            attn_mask=attn_mask, key_padding_mask=key_padding, need_weights=False,
        )
        out = self.out_norm(out).reshape(batch, time, self.tokens_per_step, dim)
        return state_tokens + out[:, :, 0]

    @property
    def num_registers_total(self) -> int:
        return self.num_spatial + self.num_temporal


class ActionFreeContextTrunk(nn.Module):
    """Causal history model. Its signature deliberately has no action argument."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_history = config.max_history
        self.time_embedding = nn.Parameter(torch.zeros(1, config.max_history, config.latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.trunk_heads,
            dim_feedforward=int(config.latent_dim * config.trunk_mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.trunk_depth, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(config.latent_dim)
        nn.init.normal_(self.time_embedding, std=0.02)

    def forward(self, state_tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        time = state_tokens.size(1)
        if time > self.max_history:
            raise ValueError(f"History {time} exceeds configured max_history={self.max_history}.")
        state_tokens = state_tokens + self.time_embedding[:, :time]
        if not valid_mask.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one valid observation token.")
        output = self.transformer(
            state_tokens,
            mask=causal_mask(time, state_tokens.device),
            src_key_padding_mask=~valid_mask.bool(),
        )
        return self.output_norm(output)


class GatedDynamicsBlock(nn.Module):
    """FiLM-modulated, gated residual update in predictor hidden space."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(latent_dim)
        self.condition = nn.Linear(latent_dim, latent_dim * 3)
        self.update = MLP(latent_dim, hidden_dim, latent_dim, dropout)

        # Begin close to an identity hidden update without severing the action path.
        nn.init.normal_(self.condition.weight, std=0.02)
        nn.init.zeros_(self.condition.bias)
        with torch.no_grad():
            self.condition.bias[latent_dim * 2 :].fill_(-2.0)

    def forward(self, hidden: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.condition(action_latent).chunk(3, dim=-1)
        conditioned = self.context_norm(hidden) * (1.0 + scale) + shift
        return hidden + torch.sigmoid(gate) * self.update(conditioned)


class ActionConditionedDynamics(nn.Module):
    """Predict ``z_(t+1) = z_t + delta(h_t, a_t)`` after the value branch.

    The visual latent is the residual anchor, while the action-free context
    supplies history and instruction information. Actions only modulate this
    module, preserving the structural action isolation of the value estimate.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.action_dim is None:
            raise ValueError("model.action_dim must be set from the dataset schema.")
        if config.dynamics_depth < 1:
            raise ValueError("dynamics_depth must be at least 1 so next-state prediction is action-conditioned.")
        self.action_dim = config.action_dim
        self.action_encoder = MLP(
            config.action_dim,
            config.action_hidden_dim,
            config.latent_dim,
            config.dropout,
        )
        hidden_dim = int(config.latent_dim * config.trunk_mlp_ratio)
        self.blocks = nn.ModuleList(
            GatedDynamicsBlock(config.latent_dim, hidden_dim, config.dropout)
            for _ in range(config.dynamics_depth)
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        nn.init.normal_(self.delta_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        current_state_latent: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the dynamics API explicitly sequence-shaped.  Relying on
        # Linear/LayerNorm broadcasting here would allow a malformed [B, A]
        # action tensor to slip through when its dimensions happen to match,
        # producing a hard-to-diagnose broadcasted update instead of a clear
        # action-conditioned one-step prediction.
        if current_state_latent.ndim != 3 or context.ndim != 3 or actions.ndim != 3:
            raise ValueError(
                "Dynamics expects current_state_latent/context/actions with shapes "
                f"[B,T,D]/[B,T,D]/[B,T,A], got "
                f"{current_state_latent.shape}/{context.shape}/{actions.shape}"
            )
        if (
            current_state_latent.device != context.device
            or context.device != actions.device
        ):
            raise ValueError(
                "Dynamics inputs must be on the same device: "
                f"current={current_state_latent.device}, context={context.device}, "
                f"actions={actions.device}."
            )
        if current_state_latent.shape != context.shape:
            raise ValueError(
                "Current-state/context shapes differ: "
                f"{current_state_latent.shape} vs {context.shape}"
            )
        if actions.shape[:2] != context.shape[:2]:
            raise ValueError(f"Context/action time shapes differ: {context.shape} vs {actions.shape}")
        if actions.size(-1) != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {actions.size(-1)}")
        action_latent = self.action_encoder(actions)
        hidden = context
        for block in self.blocks:
            hidden = block(hidden, action_latent)
        return current_state_latent + self.delta_head(hidden)


class WorldCriticModel(nn.Module):
    """
    Instruction-conditioned V(s) plus action-conditioned latent dynamics.

    `value` has no computational path from `actions`. This is stronger than
    an attention mask: action tensors are first consumed after the value branch.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        self.language_encoder = LanguageEncoder(config)
        self.view_pool_query = nn.Parameter(torch.zeros(1, 1, config.latent_dim))
        self.view_attention = nn.MultiheadAttention(
            config.latent_dim, config.trunk_heads, dropout=config.dropout, batch_first=True
        )
        self.proprioception_encoder = None
        if config.use_proprioception:
            if config.proprioception_dim is None:
                raise ValueError(
                    "use_proprioception=true requires model.proprioception_dim to be inferred before model construction."
                )
            self.proprioception_encoder = MLP(
                config.proprioception_dim,
                config.latent_dim,
                config.latent_dim,
                config.dropout,
            )
        self.language_fusion = StateLanguageFusion(config)
        self.temporal_registers = None
        self.block_register_fusion = None
        if config.use_block_register_fusion:
            self.block_register_fusion = BlockRegisterFusion(config)
        elif config.num_temporal_registers > 0:
            self.temporal_registers = TemporalRegisterBlock(config)
        self.context_trunk = ActionFreeContextTrunk(config)
        self.value_head = MLP(config.latent_dim, config.value_hidden_dim, 1, config.dropout)
        self.dynamics = ActionConditionedDynamics(config)
        self.state_vector_head = None
        if config.predict_state_vector:
            if config.state_dim is None:
                raise ValueError("predict_state_vector=True requires model.state_dim.")
            self.state_vector_head = MLP(
                config.latent_dim,
                config.latent_dim,
                config.state_dim,
                config.dropout,
            )
        nn.init.normal_(self.view_pool_query, std=0.02)
        # Constructed last so enabling the experimental path does not perturb
        # initialization of any parameter shared with the S4-Image baseline.
        self.cross_frame_encoder = (
            SparseCrossFrameEncoder(config) if config.use_cross_frame_tokens else None
        )

    def pool_views(self, view_latents: torch.Tensor) -> torch.Tensor:
        batch, time, views, dim = view_latents.shape
        values = view_latents.reshape(batch * time, views, dim)
        query = self.view_pool_query.expand(batch * time, 1, dim)
        pooled, _ = self.view_attention(query, values, values, need_weights=False)
        return pooled.reshape(batch, time, dim)

    def encode_context(
        self,
        current_state_latent: torch.Tensor,
        instruction_tokens: torch.Tensor,
        instruction_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.language_fusion(current_state_latent, instruction_tokens, instruction_mask)
        if self.block_register_fusion is not None:
            fused = self.block_register_fusion(fused, valid_mask)
        elif self.temporal_registers is not None:
            fused = self.temporal_registers(fused, valid_mask)
        return self.context_trunk(fused, valid_mask)

    def forward(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        instruction_input_ids: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        state_vectors: torch.Tensor | None = None,
        history_images: torch.Tensor | None = None,
    ) -> WorldCriticOutput:
        if images.ndim not in (5, 6):
            raise ValueError(f"Expected images [B,T,C,H,W] or [B,T,V,C,H,W], got {images.shape}")
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,T,A], got {actions.shape}")
        if images.size(0) != actions.size(0):
            raise ValueError(f"Image/action batch sizes differ: {images.shape} vs {actions.shape}")
        if images.device != actions.device:
            raise ValueError(
                "Images and actions must be on the same device: "
                f"images={images.device}, actions={actions.device}."
            )
        if instruction_input_ids.size(0) != images.size(0):
            raise ValueError(
                "Instruction batch size differs from image batch size: "
                f"{instruction_input_ids.shape} vs {images.shape}"
            )
        if instruction_input_ids.device != images.device:
            raise ValueError(
                "Images and instruction_input_ids must be on the same device: "
                f"images={images.device}, instruction_input_ids={instruction_input_ids.device}."
            )
        if images.size(1) != actions.size(1) + 1:
            raise ValueError(
                "Images must have one more timestep than actions: "
                f"images={images.shape}, actions={actions.shape}"
            )
        if valid_mask is None:
            valid_mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        if valid_mask.shape != actions.shape[:2]:
            raise ValueError(f"valid_mask must be [B,T], got {valid_mask.shape}")
        if valid_mask.device != actions.device:
            raise ValueError(
                f"valid_mask and actions must be on the same device: {valid_mask.device} != {actions.device}."
            )

        if self.cross_frame_encoder is not None:
            if history_images is None:
                raise ValueError("This model requires history_images for cross-frame fusion.")
            if history_images.ndim != 6:
                raise ValueError(
                    f"history_images must be [B,K,V,C,H,W], got {history_images.shape}"
                )
            if history_images.size(0) != images.size(0):
                raise ValueError("history_images and images must have the same batch size.")
            if history_images.device != images.device:
                raise ValueError("history_images and images must be on the same device.")
            history_tokens = self.vision_encoder.encode_tokens(history_images)
            current_views = self.cross_frame_encoder(history_tokens)
            views = current_views.size(1)
            current_views = current_views + self.vision_encoder.camera_embedding[:, 0, :views]
            current_state = self.pool_views(current_views.unsqueeze(1))
            next_state = self.pool_views(self.vision_encoder(images[:, 1:]))
            state_latents = torch.cat([current_state, next_state], dim=1)
        else:
            if history_images is not None:
                raise ValueError("history_images were provided to a model without cross-frame fusion.")
            view_latents = self.vision_encoder(images)
            state_latents = self.pool_views(view_latents)
        if self.proprioception_encoder is not None:
            if state_vectors is None:
                raise ValueError("This model requires state_vectors for proprioception fusion.")
            if state_vectors.ndim != 3:
                raise ValueError(
                    f"state_vectors must be [B,T,D], got {tuple(state_vectors.shape)}"
                )
            expected_shape = (images.size(0), state_latents.size(1) - 1)
            if state_vectors.shape[:2] != expected_shape:
                raise ValueError(
                    "state_vectors must provide one vector per current timestep: "
                    f"expected [B,{expected_shape[1]},D], got {tuple(state_vectors.shape)}"
                )
            if state_vectors.size(-1) != self.config.proprioception_dim:
                raise ValueError(
                    f"Expected proprioception dim {self.config.proprioception_dim}, "
                    f"got {state_vectors.size(-1)}"
                )
            if state_vectors.device != state_latents.device:
                raise ValueError(
                    "state_vectors and images must be on the same device: "
                    f"{state_vectors.device} != {state_latents.device}"
                )
            state_latents = state_latents.clone()
            state_latents[:, :-1] = state_latents[:, :-1] + self.proprioception_encoder(state_vectors)
        text_tokens, text_mask = self.language_encoder(
            instruction_input_ids,
            instruction_attention_mask,
        )
        context = self.encode_context(state_latents[:, :-1], text_tokens, text_mask, valid_mask)

        # This line executes before `actions` is consumed anywhere in the graph.
        value = self.value_head(context)
        next_state_pred = self.dynamics(
            current_state_latent=state_latents[:, :-1],
            context=context,
            actions=actions,
        )
        target_next_state = state_latents[:, 1:].detach()
        state_vector_pred = self.state_vector_head(next_state_pred) if self.state_vector_head is not None else None
        return WorldCriticOutput(
            context_latent=context,
            value=value,
            next_state_pred=next_state_pred,
            target_next_state=target_next_state,
            valid_mask=valid_mask,
            next_state_vector_pred=state_vector_pred,
        )

    @torch.inference_mode()
    def rollout_latent(
        self,
        observation_images: torch.Tensor,
        action_sequence: torch.Tensor,
        instruction_input_ids: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
    ) -> LatentRolloutOutput:
        """
        Autoregressively roll out the same trained dynamics branch used by `forward`.

        Args:
            observation_images: [B,H,V,C,H_img,W_img] initial visual history.
            action_sequence: [B,K,A] future actions, including the action after the last observed state.
        Returns:
            Latents [B,H+K,D] containing encoded history followed by K predictions.
        """
        if self.proprioception_encoder is not None:
            raise NotImplementedError(
                "rollout_latent requires an explicit sparse-history state adapter; "
                "use forward() with state_vectors for Sparse4 scoring."
            )
        if self.cross_frame_encoder is not None:
            raise NotImplementedError(
                "rollout_latent requires explicit sparse history images for cross-frame fusion."
            )
        if observation_images.ndim == 5:
            observation_images = observation_images.unsqueeze(2)
        if observation_images.ndim != 6:
            raise ValueError(
                "observation_images must be [B,H,C,H_img,W_img] or [B,H,V,C,H_img,W_img]."
            )
        if action_sequence.ndim != 3:
            raise ValueError(f"action_sequence must be [B,K,A], got {action_sequence.shape}")
        if action_sequence.size(0) != observation_images.size(0):
            raise ValueError(
                "Observation/action batch sizes differ: "
                f"{observation_images.shape} vs {action_sequence.shape}"
            )
        if instruction_input_ids.size(0) != observation_images.size(0):
            raise ValueError(
                "Instruction batch size differs from observation batch size: "
                f"{instruction_input_ids.shape} vs {observation_images.shape}"
            )
        if action_sequence.size(1) < 1:
            raise ValueError("action_sequence must contain at least one future action.")
        history_latents = self.pool_views(self.vision_encoder(observation_images))
        text_tokens, text_mask = self.language_encoder(
            instruction_input_ids,
            instruction_attention_mask,
        )
        latents = history_latents
        predicted_values = []
        for step in range(action_sequence.size(1)):
            history = latents[:, -self.config.max_history :]
            valid = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
            context = self.encode_context(history, text_tokens, text_mask, valid)
            predicted_values.append(self.value_head(context[:, -1:]))
            next_latent = self.dynamics(
                current_state_latent=history[:, -1:],
                context=context[:, -1:],
                actions=action_sequence[:, step : step + 1],
            )
            latents = torch.cat([latents, next_latent], dim=1)
        return LatentRolloutOutput(
            latents=latents,
            values=torch.cat(predicted_values, dim=1),
        )


class SIGReg(nn.Module):
    """Sketch isotropic Gaussian regularizer over an already-global batch."""

    def __init__(self, knots: int = 17, num_projections: int = 1024) -> None:
        super().__init__()
        self.num_projections = num_projections
        points = torch.linspace(0, 3, knots, dtype=torch.float32)
        delta = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * delta, dtype=torch.float32)
        weights[[0, -1]] = delta
        window = torch.exp(-points.square() / 2)
        self.register_buffer("points", points)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, latent: torch.Tensor, projections: torch.Tensor) -> torch.Tensor:
        if projections.shape != (latent.size(-1), self.num_projections):
            raise ValueError(
                f"Expected projection matrix {(latent.size(-1), self.num_projections)}, got {projections.shape}"
            )
        projected = (latent @ projections).unsqueeze(-1) * self.points
        error = (projected.cos().mean(-3) - self.phi).square() + projected.sin().mean(-3).square()
        statistic = (error @ self.weights) * latent.size(-2)
        return statistic.mean()


def normalized_random_projections(
    latent_dim: int,
    num_projections: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    projections = torch.randn(
        latent_dim,
        num_projections,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return F.normalize(projections, dim=0)
