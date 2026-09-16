from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass
class DataConfig:
    repo_id: str
    root: str | None = None
    revision: str | None = None
    image_keys: list[str] = field(default_factory=lambda: ["observation.images.front"])
    action_key: str = "action"
    state_key: str | None = "observation.state"
    return_key: str = "return"
    success_key: str | None = None
    history_size: int = 3
    # Optional sparse observation history for endpoint scoring.  When set,
    # offsets are ordered [0, older, ...] in frame-index units and the regular
    # contiguous history window is reduced to one current endpoint.
    history_offsets: list[int] | None = None
    # Optional four-frame sampling used only to reconstruct an existing mosaic
    # teacher input. This is independent of the student's raw-video history.
    mosaic_history_offsets: list[int] | None = None
    history_mosaic: bool = False
    history_frames: bool = False
    prediction_horizon: int = 1
    val_fraction: float = 0.1
    split_seed: int = 3072
    split_manifest: str | None = None
    # Kept as a compatibility field for old configs.  Value supervision is
    # mandatory in this critic implementation, so enabling it is rejected.
    allow_missing_return: bool = False
    normalize_action: bool = True
    action_mean: list[float] | None = None
    action_std: list[float] | None = None
    normalization_epsilon: float = 1e-6


@dataclass
class VisionConfig:
    model_name: str = "google/vit-base-patch16-224-in21k"
    image_size: int = 224
    trainable: bool = True
    pretrained: bool = True
    # ViT register tokens（Darcet et al. 2023）：追加 K 个可学习全局 token，
    # 前向时拼到序列里参与 attention，输出时丢弃，用于吸收全局信息、
    # 缓解 CLS/patch 注意力中的 artifact，改善空间注意力分布。
    num_register_tokens: int = 0
    # 寄存器插入位置：
    #   early = 第 0 层输入即拼入序列、随全部层一遍流动（Darcet et al. 标准
    #           语义；深度/算力与无寄存 baseline 严格一致）；
    #   late  = backbone 完整 forward 后把寄存器拼到输出、再过一遍全部层
    #           （旧行为，等效双倍深度——历史 D2 结果带该混淆，见实验报告 §8.3；
    #           保留默认值以保证旧 checkpoint 可复现）。
    register_insert: str = "late"


@dataclass
class LanguageConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    max_length: int = 77
    trainable: bool = False
    pretrained: bool = True
    fusion_layers: int = 2
    fusion_heads: int = 8
    fusion_dropout: float = 0.0


@dataclass
class ModelConfig:
    action_dim: int | None = None
    state_dim: int | None = None
    latent_dim: int = 384
    max_views: int = 16
    trunk_depth: int = 6
    trunk_heads: int = 8
    trunk_mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_history: int = 16
    value_hidden_dim: int = 384
    dynamics_depth: int = 3
    action_hidden_dim: int = 384
    predict_state_vector: bool = False
    # Optional proprioception input fused into the value context.  The input
    # dimension is inferred from the dataset when omitted (28 for four 7D
    # sparse states).
    use_proprioception: bool = False
    proprioception_dim: int | None = None
    # VGGT-inspired sparse cross-frame reasoning: retain full patch grids for
    # the configured history offsets and let a current-frame query attend to
    # all historical patch tokens before WCM context/value prediction.
    use_cross_frame_tokens: bool = False
    cross_frame_count: int = 4
    cross_frame_layers: int = 2
    # Alternating frame-wise ViT and block-causal global attention. All sparse
    # frames remain alive through the visual encoder. Global blocks update all
    # frames while preventing any frame from reading a later observation.
    use_temporal_transformer: bool = False
    temporal_transformer_count: int = 4
    temporal_transformer_layers: int = 6
    temporal_transformer_heads: int = 4
    temporal_transformer_dim: int = 192
    temporal_transformer_mlp_ratio: float = 2.0
    # Persistent sparse visual memory. Learned queries compress every frame
    # immediately after patch embedding; memory tokens then travel through all
    # frame-wise ViT stages and communicate through block-causal global layers.
    use_sparse_temporal_memory: bool = False
    sparse_memory_frame_count: int = 4
    sparse_memory_tokens: int = 8
    sparse_memory_global_layers: int = 6
    sparse_memory_heads: int = 8
    sparse_memory_mlp_ratio: float = 2.0
    sparse_memory_layerscale_init: float = 1.0e-3
    # Lightweight residual reasoning over the four temporal quadrants of an
    # S4-Image mosaic. The zero-initialized gate preserves the Image-B4 model
    # exactly at warm-start while the adapter learns an incremental correction.
    use_mosaic_temporal_residual: bool = False
    mosaic_temporal_heads: int = 4
    mosaic_temporal_mlp_ratio: float = 2.0
    # Raw history frames are encoded by the shared image ViT, mixed along time,
    # then compressed to a fixed visual-token budget by a Perceiver reducer.
    use_spacetime_perceiver: bool = False
    # Maximum raw video frames accepted by one checkpoint. Training normally
    # uses exactly len(data.history_offsets); inference may use any shorter T.
    spacetime_frame_count: int = 4
    spacetime_layers: int = 2
    spacetime_heads: int = 8
    perceiver_queries: int = 64
    perceiver_layers: int = 2
    perceiver_mlp_ratio: float = 2.0
    spacetime_teacher_enabled: bool = False
    spacetime_temporal_enabled: bool = True
    predict_risk: bool = False
    predict_q: bool = False
    # temporal register token：跨时间步共享的 K_t 个可学习 token（latent 空间），
    # 通过交叉注意力从历史帧 token 中吸收时序不变信息再注回各时间步。
    # 与 vision.num_register_tokens（空间 register，ViT 内部）相互独立，可分别消融。
    num_temporal_registers: int = 0
    # 块注意力 register 融合：K_s 个空间寄存 + K_t 个时间寄存进入同一注意力层，
    # 时间维块内双向、块间因果（BlockRegisterFusion）。开启后取代独立的
    # TemporalRegisterBlock 与 ViT 内 register（建议 vision.num_register_tokens=0）。
    num_spatial_registers: int = 0
    register_block_size: int = 2
    use_block_register_fusion: bool = False
    # 历史诊断使用的 token-edge 放行率。它在全部 token 对上采样，并不等于
    # “未来 observation 泄漏比例”；新训练应保持为 0。
    register_future_leak_ratio: float = 0.0
    # Historical diagnostics may intentionally violate online causality. This
    # must be explicit so a deployable training run cannot do so by accident.
    allow_noncausal_register_ablation: bool = False
    vision: VisionConfig = field(default_factory=VisionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)


@dataclass
class LossConfig:
    value_weight: float = 1.0
    # RankNet-style ordering objective over the final valid value in each
    # training window.  This complements absolute MSE with a calibration-
    # invariant signal: examples with meaningfully different returns should
    # be ordered the same way by the critic.
    ranking_weight: float = 0.0
    ranking_temperature: float = 0.1
    ranking_min_target_gap: float = 0.1
    next_state_weight: float = 0.1
    next_state_vector_weight: float = 0.0
    sigreg_weight: float = 0.01
    sigreg_knots: int = 17
    sigreg_num_projections: int = 1024
    alignment_weight: float = 0.0
    alignment_mse_weight: float = 0.1
    risk_weight: float = 0.0
    q_weight: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 5e-5
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    # register token 单独学习率缩放（相对 base lr）。>1 让新加入的 register
    # embedding 冷启动更快；仅对名称含 "register_tokens" 的参数生效。
    register_lr_scale: float = 1.0
    # Newly initialized causal global-attention adapters need to open their
    # zero residual gates faster than a warm-started vision backbone.
    temporal_lr_scale: float = 1.0


@dataclass
class TrainConfig:
    output_dir: str = "outputs/wcm"
    seed: int = 3072
    epochs: int = 100
    per_device_batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 8
    # Variable valid-token counts and global SIGReg are normalized per microbatch.
    # Keep this at 1 until accumulation-window global-count normalization is implemented.
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    log_every: int = 20
    eval_every_epochs: int = 1
    save_every_epochs: int = 1
    resume: str | None = None
    # Warm-start：仅加载模型权重（不校验 config、不恢复 optimizer/scheduler/RNG），用于换数据集微调。
    init_from: str | None = None
    teacher_checkpoint: str | None = None
    partial_init: bool = False
    compile: bool = False
    expected_world_size: int | None = None
    ddp_timeout_minutes: int = 30
    deterministic: bool = False
    training_stage: str = "standard"
    gate_start: float = 0.0
    gate_end: float = 0.0
    alignment_weight_start: float | None = None
    alignment_weight_end: float | None = None
    early_stop_metric: str | None = None
    early_stop_threshold: float | None = None
    early_stop_mode: str = "max"
    alignment_plateau_patience_steps: int = 0
    alignment_plateau_min_delta: float = 1.0e-4
    alignment_plateau_ema_decay: float = 0.99
    unfreeze_vision_top_layers: int = 4
    selection_metric: str = "value_mse"
    selection_mode: str = "min"
    deploy_from_best: bool = False
    data: DataConfig | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class EvalConfig:
    checkpoint: str
    output_dir: str = "outputs/wcm_eval"
    batch_size: int = 64
    num_workers: int = 8
    precision: str = "bf16"
    expected_world_size: int | None = None
    max_batches: int | None = None


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    values = dict(values)
    if cls is TrainConfig:
        if "data" not in values:
            raise ValueError("Training config requires a 'data' section.")
        values["data"] = _construct(DataConfig, values["data"])
        values["model"] = _construct(ModelConfig, values.get("model", {}))
        values["loss"] = _construct(LossConfig, values.get("loss", {}))
        values["optim"] = _construct(OptimConfig, values.get("optim", {}))
    elif cls is ModelConfig:
        values["vision"] = _construct(VisionConfig, values.get("vision", {}))
        values["language"] = _construct(LanguageConfig, values.get("language", {}))
    elif cls is LossConfig and "return_weight" in values:
        if "value_weight" in values:
            raise ValueError("Specify only loss.value_weight; return_weight is a deprecated alias.")
        values["value_weight"] = values.pop("return_weight")
    elif cls is OptimConfig and "betas" in values:
        values["betas"] = tuple(values["betas"])
    return cls(**values)


def load_config(path: str | Path, cls: type[T] = TrainConfig) -> T:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML configs require PyYAML. Use JSON or install pyyaml.") from exc
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format: {path}")
    if not isinstance(values, dict):
        raise TypeError(f"Expected an object at the root of {path}")
    return _construct(cls, values)


def apply_runtime_overrides(config: TrainConfig) -> TrainConfig:
    """Apply small launcher-friendly overrides supplied through ``WCM_*`` env vars.

    The checked-in ``run_*.sh`` intentionally keeps all experiment knobs at
    the top of one shell file.  Keeping the override layer here avoids
    generating a temporary YAML file and preserves the exact same config
    validation/checkpoint schema as a direct ``--config`` invocation.  Empty
    variables are ignored, so users can leave a field at its YAML value.
    """

    if config.data is None:
        raise ValueError("Runtime overrides require a training config with data settings.")

    def value(name: str) -> str | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        raw = raw.strip()
        return raw if raw else None

    def integer(name: str) -> int | None:
        raw = value(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc

    def integer_list(name: str) -> list[int] | None:
        raw = value(name)
        if raw is None:
            return None
        try:
            result = [int(item.strip()) for item in raw.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {name} must be comma-separated integers, got {raw!r}."
            ) from exc
        if not result:
            raise ValueError(f"Environment variable {name} cannot be empty.")
        return result

    repo_id = value("WCM_DATASET_REPO_ID")
    root = value("WCM_DATASET_ROOT")
    revision = value("WCM_DATASET_REVISION")
    vision_model_name = value("WCM_VISION_MODEL_NAME")
    language_model_name = value("WCM_LANGUAGE_MODEL_NAME")
    output_dir = value("WCM_OUTPUT_DIR")
    expected_world_size = integer("WCM_EXPECTED_WORLD_SIZE")
    num_workers = integer("WCM_NUM_WORKERS")
    per_device_batch_size = integer("WCM_PER_DEVICE_BATCH_SIZE")
    eval_batch_size = integer("WCM_EVAL_BATCH_SIZE")
    epochs = integer("WCM_EPOCHS")
    seed = integer("WCM_SEED")
    init_from = value("WCM_INIT_FROM")
    teacher_checkpoint = value("WCM_TEACHER_CHECKPOINT")
    resume = value("WCM_RESUME")
    precision = value("WCM_PRECISION")
    history_offsets = integer_list("WCM_HISTORY_OFFSETS")
    spacetime_frame_count = integer("WCM_SPACETIME_FRAME_COUNT")

    if repo_id is not None:
        config.data.repo_id = repo_id
    if root is not None:
        config.data.root = root
    if revision is not None:
        config.data.revision = revision
    if vision_model_name is not None:
        config.model.vision.model_name = vision_model_name
    if language_model_name is not None:
        config.model.language.model_name = language_model_name
    if output_dir is not None:
        config.output_dir = output_dir
    if expected_world_size is not None:
        config.expected_world_size = expected_world_size
    if num_workers is not None:
        config.num_workers = num_workers
    if per_device_batch_size is not None:
        config.per_device_batch_size = per_device_batch_size
    if eval_batch_size is not None:
        config.eval_batch_size = eval_batch_size
    if epochs is not None:
        config.epochs = epochs
    if seed is not None:
        config.seed = seed
    if init_from is not None:
        config.init_from = init_from
    if teacher_checkpoint is not None:
        config.teacher_checkpoint = teacher_checkpoint
    if resume is not None:
        config.resume = resume
    if precision is not None:
        config.precision = precision
    if history_offsets is not None:
        config.data.history_offsets = history_offsets
    if spacetime_frame_count is not None:
        config.model.spacetime_frame_count = spacetime_frame_count
    return config


def save_resolved_config(config: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def config_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a JSON or YAML configuration file.")
    return parser


def validate_train_config(config: TrainConfig) -> None:
    if config.data is None:
        raise ValueError("Training config requires data settings.")
    if config.data.history_size < 1:
        raise ValueError("data.history_size must be positive.")
    if config.data.history_offsets is not None:
        offsets = config.data.history_offsets
        if not offsets or offsets[0] != 0:
            raise ValueError("data.history_offsets must be non-empty and start with 0.")
        if any(offset < 0 for offset in offsets):
            raise ValueError("data.history_offsets cannot contain negative offsets.")
        if offsets != sorted(set(offsets)):
            raise ValueError("data.history_offsets must be sorted and contain no duplicates.")
        if config.data.history_size != 1:
            raise ValueError("data.history_size must be 1 when data.history_offsets is configured.")
        mosaic_offsets = config.data.mosaic_history_offsets or offsets
        if config.data.history_mosaic:
            if len(mosaic_offsets) != 4:
                raise ValueError(
                    "data.history_mosaic requires exactly four mosaic_history_offsets."
                )
            if mosaic_offsets[0] != 0:
                raise ValueError("data.mosaic_history_offsets must start with 0.")
            if any(offset < 0 for offset in mosaic_offsets):
                raise ValueError("data.mosaic_history_offsets cannot contain negative offsets.")
            if mosaic_offsets != sorted(set(mosaic_offsets)):
                raise ValueError(
                    "data.mosaic_history_offsets must be sorted and contain no duplicates."
                )
        if (
            config.data.history_mosaic
            and config.data.history_frames
            and not config.model.use_spacetime_perceiver
        ):
            raise ValueError("history_mosaic and history_frames are mutually exclusive.")
    elif config.data.history_mosaic:
        raise ValueError("data.history_mosaic requires data.history_offsets.")
    elif config.data.history_frames:
        raise ValueError("data.history_frames requires data.history_offsets.")
    if config.data.mosaic_history_offsets is not None and not config.data.history_mosaic:
        raise ValueError("data.mosaic_history_offsets requires history_mosaic=true.")
    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if config.eval_every_epochs < 1:
        raise ValueError("eval_every_epochs must be positive.")
    if config.save_every_epochs < 1:
        raise ValueError("save_every_epochs must be positive.")
    if config.log_every < 1:
        raise ValueError("log_every must be positive.")
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    if not config.data.image_keys:
        raise ValueError("data.image_keys must contain at least one camera feature.")
    if any(not str(key).strip() for key in config.data.image_keys):
        raise ValueError("data.image_keys cannot contain empty feature names.")
    if config.model.max_history < 1:
        raise ValueError("model.max_history must be positive.")
    if config.model.use_proprioception and config.data.state_key is None:
        raise ValueError("use_proprioception=true requires data.state_key.")
    if config.model.proprioception_dim is not None and config.model.proprioception_dim < 1:
        raise ValueError("model.proprioception_dim must be positive when set.")
    temporal_fusion_count = sum(
        (
            config.model.use_cross_frame_tokens,
            config.model.use_temporal_transformer,
            config.model.use_sparse_temporal_memory,
            config.model.use_mosaic_temporal_residual,
            config.model.use_spacetime_perceiver,
        )
    )
    if temporal_fusion_count > 1:
        raise ValueError("temporal fusion implementations are mutually exclusive.")
    if config.model.use_cross_frame_tokens:
        if not config.data.history_frames:
            raise ValueError("use_cross_frame_tokens=true requires data.history_frames=true.")
        if config.data.history_offsets is None:
            raise ValueError("use_cross_frame_tokens=true requires data.history_offsets.")
        if config.model.cross_frame_count != len(config.data.history_offsets):
            raise ValueError("cross_frame_count must match the number of history_offsets.")
    if config.model.use_temporal_transformer:
        if config.model.use_cross_frame_tokens or config.model.use_sparse_temporal_memory:
            raise ValueError(
                "temporal fusion implementations are mutually exclusive."
            )
        if not config.data.history_frames:
            raise ValueError("use_temporal_transformer=true requires data.history_frames=true.")
        if config.data.history_offsets is None:
            raise ValueError("use_temporal_transformer=true requires data.history_offsets.")
        if config.model.temporal_transformer_count != len(config.data.history_offsets):
            raise ValueError(
                "temporal_transformer_count must match the number of history_offsets."
            )
    if config.model.use_sparse_temporal_memory:
        if config.model.use_cross_frame_tokens:
            raise ValueError("temporal fusion implementations are mutually exclusive.")
        if not config.data.history_frames:
            raise ValueError("use_sparse_temporal_memory=true requires data.history_frames=true.")
        if config.data.history_offsets is None:
            raise ValueError("use_sparse_temporal_memory=true requires data.history_offsets.")
        if config.model.sparse_memory_frame_count != len(config.data.history_offsets):
            raise ValueError(
                "sparse_memory_frame_count must match the number of history_offsets."
            )
        if config.model.vision.num_register_tokens > 0:
            raise ValueError(
                "Sparse temporal memory does not support vision register tokens in its first ablation."
            )
    if config.model.use_mosaic_temporal_residual:
        if not config.data.history_mosaic:
            raise ValueError(
                "use_mosaic_temporal_residual=true requires data.history_mosaic=true."
            )
        if config.data.history_offsets is None or len(config.data.history_offsets) != 4:
            raise ValueError(
                "Mosaic temporal residual requires exactly four history_offsets."
            )
        if (
            config.model.use_cross_frame_tokens
            or config.model.use_temporal_transformer
            or config.model.use_sparse_temporal_memory
        ):
            raise ValueError("temporal fusion implementations are mutually exclusive.")
    if config.model.use_spacetime_perceiver:
        if not config.data.history_frames:
            raise ValueError("use_spacetime_perceiver=true requires data.history_frames=true.")
        if config.data.history_offsets is None:
            raise ValueError("use_spacetime_perceiver=true requires data.history_offsets.")
        if config.model.spacetime_frame_count != len(config.data.history_offsets):
            raise ValueError(
                "spacetime_frame_count must match the number of history_offsets."
            )
        if config.model.spacetime_teacher_enabled and not config.data.history_mosaic:
            raise ValueError(
                "spacetime_teacher_enabled=true requires data.history_mosaic=true."
            )
        if config.model.spacetime_teacher_enabled and not config.teacher_checkpoint:
            raise ValueError(
                "spacetime_teacher_enabled=true requires teacher_checkpoint."
            )
        if not config.model.spacetime_teacher_enabled and config.data.history_mosaic:
            raise ValueError(
                "history_mosaic must be false when the SpaceTime teacher is disabled."
            )
        if config.model.predict_risk and config.data.success_key is None:
            raise ValueError("predict_risk=true requires data.success_key.")
        if config.loss.risk_weight > 0 and not config.model.predict_risk:
            raise ValueError("loss.risk_weight>0 requires model.predict_risk=true.")
        if config.loss.q_weight > 0 and not config.model.predict_q:
            raise ValueError("loss.q_weight>0 requires model.predict_q=true.")
    if config.model.cross_frame_count < 1 or config.model.cross_frame_layers < 1:
        raise ValueError("cross_frame_count and cross_frame_layers must be positive.")
    if config.data.history_size > config.model.max_history:
        raise ValueError("data.history_size cannot exceed model.max_history.")
    if config.model.latent_dim < 1:
        raise ValueError("model.latent_dim must be positive.")
    if config.model.max_views < 1:
        raise ValueError("model.max_views must be positive.")
    if config.model.trunk_depth < 1:
        raise ValueError("model.trunk_depth must be positive.")
    if config.model.trunk_heads < 1:
        raise ValueError("model.trunk_heads must be positive.")
    if config.model.language.fusion_layers < 1:
        raise ValueError("model.language.fusion_layers must be positive.")
    if config.model.language.fusion_heads < 1:
        raise ValueError("model.language.fusion_heads must be positive.")
    if config.model.temporal_transformer_count < 1:
        raise ValueError("model.temporal_transformer_count must be positive.")
    if config.model.mosaic_temporal_heads < 1:
        raise ValueError("model.mosaic_temporal_heads must be positive.")
    if config.model.latent_dim % config.model.mosaic_temporal_heads != 0:
        raise ValueError("model.latent_dim must be divisible by mosaic_temporal_heads.")
    if config.model.mosaic_temporal_mlp_ratio <= 0:
        raise ValueError("model.mosaic_temporal_mlp_ratio must be positive.")
    if config.model.spacetime_frame_count < 1 or config.model.spacetime_layers < 1:
        raise ValueError("spacetime_frame_count and spacetime_layers must be positive.")
    if (
        config.model.spacetime_heads < 1
        or config.model.latent_dim % config.model.spacetime_heads != 0
    ):
        raise ValueError("latent_dim must be divisible by positive spacetime_heads.")
    if config.model.perceiver_queries not in (64, 128):
        raise ValueError("perceiver_queries must be 64 or 128.")
    if config.model.perceiver_layers < 1 or config.model.perceiver_mlp_ratio <= 0:
        raise ValueError("perceiver_layers and perceiver_mlp_ratio must be positive.")
    if config.model.temporal_transformer_layers < 1:
        raise ValueError("model.temporal_transformer_layers must be positive.")
    if config.model.temporal_transformer_heads < 1:
        raise ValueError("model.temporal_transformer_heads must be positive.")
    if config.model.temporal_transformer_dim < 1:
        raise ValueError("model.temporal_transformer_dim must be positive.")
    if config.model.temporal_transformer_dim % config.model.temporal_transformer_heads != 0:
        raise ValueError(
            "temporal_transformer_dim must be divisible by temporal_transformer_heads."
        )
    if config.model.temporal_transformer_mlp_ratio <= 0:
        raise ValueError("model.temporal_transformer_mlp_ratio must be positive.")
    if config.model.sparse_memory_frame_count < 1:
        raise ValueError("model.sparse_memory_frame_count must be positive.")
    if config.model.sparse_memory_tokens < 1:
        raise ValueError("model.sparse_memory_tokens must be positive.")
    if config.model.sparse_memory_global_layers < 1:
        raise ValueError("model.sparse_memory_global_layers must be positive.")
    if config.model.sparse_memory_heads < 1:
        raise ValueError("model.sparse_memory_heads must be positive.")
    if config.model.sparse_memory_mlp_ratio <= 0:
        raise ValueError("model.sparse_memory_mlp_ratio must be positive.")
    if config.model.sparse_memory_layerscale_init <= 0:
        raise ValueError("model.sparse_memory_layerscale_init must be positive.")
    if config.model.dynamics_depth < 1:
        raise ValueError("model.dynamics_depth must be positive so dynamics remains action-conditioned.")
    valid_stages = {
        "standard",
        "spacetime_align",
        "spacetime_gate",
        "spacetime_joint",
        "spacetime_full",
    }
    if config.training_stage not in valid_stages:
        raise ValueError(f"training_stage must be one of {sorted(valid_stages)}.")
    if config.training_stage != "standard" and not config.model.use_spacetime_perceiver:
        raise ValueError("SpaceTime training stages require use_spacetime_perceiver=true.")
    if not 0.0 <= config.gate_start <= 1.0 or not 0.0 <= config.gate_end <= 1.0:
        raise ValueError("gate_start and gate_end must be in [0,1].")
    if (config.alignment_weight_start is None) != (config.alignment_weight_end is None):
        raise ValueError(
            "alignment_weight_start and alignment_weight_end must both be set or both be null."
        )
    if config.alignment_weight_start is not None and (
        config.alignment_weight_start < 0 or config.alignment_weight_end < 0
    ):
        raise ValueError("Scheduled alignment weights cannot be negative.")
    if (config.early_stop_metric is None) != (config.early_stop_threshold is None):
        raise ValueError(
            "early_stop_metric and early_stop_threshold must both be set or both be null."
        )
    if config.early_stop_mode not in {"min", "max"}:
        raise ValueError("early_stop_mode must be 'min' or 'max'.")
    if config.alignment_plateau_patience_steps < 0:
        raise ValueError("alignment_plateau_patience_steps cannot be negative.")
    if config.alignment_plateau_min_delta < 0:
        raise ValueError("alignment_plateau_min_delta cannot be negative.")
    if not 0.0 <= config.alignment_plateau_ema_decay < 1.0:
        raise ValueError("alignment_plateau_ema_decay must be in [0,1).")
    if (
        config.alignment_plateau_patience_steps > 0
        and config.training_stage != "spacetime_align"
    ):
        raise ValueError(
            "alignment_plateau_patience_steps is only supported for spacetime_align."
        )
    if config.unfreeze_vision_top_layers < 0:
        raise ValueError("unfreeze_vision_top_layers cannot be negative.")
    if config.selection_mode not in {"min", "max"}:
        raise ValueError("selection_mode must be 'min' or 'max'.")
    if any(
        weight < 0
        for weight in (
            config.loss.alignment_weight,
            config.loss.alignment_mse_weight,
            config.loss.risk_weight,
            config.loss.q_weight,
        )
    ):
        raise ValueError("alignment/risk/q loss weights cannot be negative.")
    if not (0.0 <= config.data.val_fraction < 1.0):
        raise ValueError("data.val_fraction must be in [0, 1).")
    if config.data.normalization_epsilon <= 0:
        raise ValueError("data.normalization_epsilon must be positive.")
    if not config.data.return_key.strip():
        raise ValueError("data.return_key must name the supervised return field.")
    if config.data.allow_missing_return:
        raise ValueError("data.allow_missing_return is unsupported: value training requires a return field.")
    if (config.data.action_mean is None) != (config.data.action_std is None):
        raise ValueError("data.action_mean and data.action_std must either both be set or both be null.")
    if not config.data.normalize_action and config.data.action_mean is not None:
        raise ValueError("Action statistics must be null when normalize_action=false.")
    if config.model.latent_dim % config.model.trunk_heads != 0:
        raise ValueError("model.latent_dim must be divisible by model.trunk_heads.")
    if config.model.latent_dim % config.model.language.fusion_heads != 0:
        raise ValueError("model.latent_dim must be divisible by language.fusion_heads.")
    if config.model.vision.num_register_tokens < 0:
        raise ValueError("model.vision.num_register_tokens cannot be negative.")
    if config.model.vision.register_insert not in {"early", "late"}:
        raise ValueError("model.vision.register_insert must be early or late.")
    if config.model.vision.num_register_tokens > 0 and not config.model.vision.trainable:
        raise ValueError("Vision register tokens require model.vision.trainable=true.")
    if config.model.num_temporal_registers < 0 or config.model.num_spatial_registers < 0:
        raise ValueError("Register token counts cannot be negative.")
    if config.model.register_block_size < 1:
        raise ValueError("model.register_block_size must be positive.")
    if not 0.0 <= config.model.register_future_leak_ratio < 1.0:
        raise ValueError("model.register_future_leak_ratio must be in [0, 1).")
    block_registers = config.model.num_temporal_registers + config.model.num_spatial_registers
    if config.model.use_block_register_fusion:
        if block_registers < 1:
            raise ValueError(
                "use_block_register_fusion=true requires at least one temporal or spatial register."
            )
        if (
            config.model.register_block_size > 1
            and not config.model.allow_noncausal_register_ablation
        ):
            raise ValueError(
                "register_block_size>1 exposes later states inside each block and is not causal. "
                "Use block_size=1 for trainable/evaluable models."
            )
        if (
            config.model.register_future_leak_ratio > 0
            and not config.model.allow_noncausal_register_ablation
        ):
            raise ValueError(
                "register_future_leak_ratio>0 leaks future observations and is disabled for "
                "trainable/evaluable models."
            )
    elif config.model.num_spatial_registers > 0:
        raise ValueError(
            "num_spatial_registers is only used when use_block_register_fusion=true."
        )
    elif config.model.register_future_leak_ratio > 0:
        raise ValueError(
            "register_future_leak_ratio is only used when use_block_register_fusion=true."
        )
    if config.optim.register_lr_scale <= 0:
        raise ValueError("optim.register_lr_scale must be positive.")
    if config.optim.temporal_lr_scale <= 0:
        raise ValueError("optim.temporal_lr_scale must be positive.")
    if config.model.predict_state_vector and config.data.state_key is None:
        raise ValueError("predict_state_vector requires data.state_key.")
    if config.loss.next_state_vector_weight > 0 and not config.model.predict_state_vector:
        raise ValueError("next_state_vector_weight>0 requires model.predict_state_vector=true.")
    if config.loss.sigreg_knots < 2:
        raise ValueError("loss.sigreg_knots must be at least 2.")
    if config.loss.sigreg_num_projections < 1:
        raise ValueError("loss.sigreg_num_projections must be positive.")
    if config.loss.ranking_temperature <= 0:
        raise ValueError("loss.ranking_temperature must be positive.")
    if config.loss.ranking_min_target_gap < 0:
        raise ValueError("loss.ranking_min_target_gap cannot be negative.")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if config.per_device_batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if config.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16; fp16 is disabled without GradScaler support.")
    for name in (
        "value_weight",
        "ranking_weight",
        "next_state_weight",
        "next_state_vector_weight",
        "sigreg_weight",
    ):
        if getattr(config.loss, name) < 0:
            raise ValueError(f"loss.{name} cannot be negative.")
    if config.loss.value_weight <= 0:
        raise ValueError(
            "loss.value_weight must be positive: the requested World Critic always trains a value head."
        )
    if config.loss.next_state_weight <= 0:
        raise ValueError("next_state_weight must be positive to retain the requested world-model auxiliary task.")
    if config.model.predict_state_vector and config.loss.next_state_vector_weight <= 0:
        raise ValueError(
            "predict_state_vector=true requires a positive next_state_vector_weight; "
            "otherwise its parameters would be unused under DDP."
        )
    if config.expected_world_size is not None and config.expected_world_size < 1:
        raise ValueError("expected_world_size must be positive when set.")
    if config.ddp_timeout_minutes < 1:
        raise ValueError("ddp_timeout_minutes must be positive.")
