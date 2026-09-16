from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.nn.parallel import DistributedDataParallel

from .config import LossConfig, TrainConfig, _construct
from .checkpoint import collectively_validate, unwrap_model
from .data import canonicalize_return_target
from .curves import build_episode_curves
from .distributed import DistributedContext, all_reduce_sum, gather_objects
from .metrics import RegressionMetrics, ddp_global_mean_loss, masked_squared_error
from .model import SIGReg, WorldCriticModel, normalized_random_projections


@dataclass
class AlignmentPlateauMonitor:
    patience_steps: int
    min_delta: float
    ema_decay: float
    ema: float | None = None
    best_ema: float = -math.inf
    last_improvement_step: int | None = None
    current_step: int = 0

    def update(self, cosine: float, global_step: int) -> bool:
        if not math.isfinite(cosine):
            raise FloatingPointError(f"Alignment cosine is non-finite: {cosine}")
        self.current_step = global_step
        self.ema = (
            cosine
            if self.ema is None
            else self.ema_decay * self.ema + (1.0 - self.ema_decay) * cosine
        )
        if self.last_improvement_step is None or self.ema >= self.best_ema + self.min_delta:
            self.best_ema = self.ema
            self.last_improvement_step = global_step
        return global_step - self.last_improvement_step >= self.patience_steps

    @property
    def steps_since_improvement(self) -> int:
        if self.last_improvement_step is None:
            return 0
        return self.current_step - self.last_improvement_step


def seed_everything(seed: int, deterministic: bool = False) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _evaluation_log(ctx: DistributedContext, message: str, *, main_only: bool = False) -> None:
    """Print an unbuffered diagnostic message for the evaluation path."""

    if main_only and not ctx.is_main:
        return
    print(f"[eval][rank={ctx.rank}][device={ctx.device}] {message}", flush=True)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def differentiable_global_gather(tensor: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    """Gather a batch while preserving the exact DDP gradient semantics.

    ``torch.distributed.nn.functional.all_gather`` reduces the per-rank
    gradients with a SUM in its backward pass.  Since every rank evaluates
    the same scalar SIGReg objective on the concatenated global batch, DDP's
    subsequent gradient average supplies the compensating ``1/world_size``.
    Do not divide the gathered objective (or its loss) by ``world_size``;
    doing so would under-scale SIGReg gradients by that factor.
    """
    if not ctx.distributed:
        return tensor
    gathered = dist_nn.all_gather(tensor)
    return torch.cat(tuple(gathered), dim=0)


def synchronized_projections(
    latent_dim: int,
    num_projections: int,
    device: torch.device,
    dtype: torch.dtype,
    ctx: DistributedContext,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    projections = normalized_random_projections(
        latent_dim,
        num_projections,
        device,
        dtype,
        generator=generator,
    )
    if ctx.distributed:
        dist.broadcast(projections, src=0)
    return projections


def build_model(config: TrainConfig) -> WorldCriticModel:
    return WorldCriticModel(config.model)


def configure_training_stage(model: torch.nn.Module, config: TrainConfig) -> dict[str, Any]:
    """Apply the explicit SpaceTime freeze policy before optimizer creation."""
    stage = config.training_stage
    raw_model = unwrap_model(model)
    if stage == "standard":
        return {"stage": stage, "unfrozen_vision_layers": []}
    spacetime = getattr(raw_model, "spacetime_encoder", None)
    if spacetime is None:
        raise ValueError(f"Training stage {stage!r} requires spacetime_encoder.")

    for parameter in raw_model.parameters():
        parameter.requires_grad_(False)

    def unfreeze(module: torch.nn.Module | None) -> None:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    unfrozen_layers: list[int] = []
    if stage == "spacetime_align":
        for name, parameter in spacetime.named_parameters():
            if name.startswith(
                (
                    "perceiver_queries",
                    "perceiver_layers.",
                    "output_projection.",
                )
            ):
                parameter.requires_grad_(True)
    elif stage == "spacetime_gate":
        unfreeze(spacetime)
    elif stage == "spacetime_joint":
        unfreeze(spacetime)
        unfreeze(raw_model.vision_encoder.projection)
        layers = raw_model.vision_encoder.encoder_layers()
        count = min(config.unfreeze_vision_top_layers, len(layers))
        for index in range(len(layers) - count, len(layers)):
            unfreeze(layers[index])
            unfrozen_layers.append(index)
        vision_root = getattr(
            raw_model.vision_encoder.backbone,
            "vision_model",
            raw_model.vision_encoder.backbone,
        )
        unfreeze(getattr(vision_root, "post_layernorm", None))
        unfreeze(raw_model.language_fusion)
        unfreeze(raw_model.context_trunk)
        unfreeze(raw_model.value_head)
        unfreeze(raw_model.risk_head)
        unfreeze(raw_model.q_action_encoder)
        unfreeze(raw_model.q_head)
    elif stage == "spacetime_full":
        for parameter in raw_model.parameters():
            parameter.requires_grad_(True)
        if not config.model.language.trainable:
            raw_model.language_encoder.requires_grad_(False)
        # Teacher-free current states come from SpaceTime tokens. These legacy
        # view-pooling parameters only touch the detached next-state target and
        # must stay frozen under DDP find_unused_parameters=False.
        raw_model.vision_encoder.camera_embedding.requires_grad_(False)
        raw_model.view_pool_query.requires_grad_(False)
        raw_model.view_attention.requires_grad_(False)
        unfrozen_layers = list(range(len(raw_model.vision_encoder.encoder_layers())))
    else:
        raise ValueError(f"Unsupported training_stage={stage!r}.")
    return {"stage": stage, "unfrozen_vision_layers": unfrozen_layers}


def update_spacetime_gate(
    model: torch.nn.Module,
    config: TrainConfig,
    global_step: int,
    total_steps: int,
) -> float | None:
    spacetime = getattr(unwrap_model(model), "spacetime_encoder", None)
    if spacetime is None:
        return None
    progress = min(max(global_step / max(total_steps - 1, 1), 0.0), 1.0)
    gate = config.gate_start + (config.gate_end - config.gate_start) * progress
    spacetime.set_gate(float(gate))
    return float(gate)


def scheduled_alignment_weight(
    config: TrainConfig,
    global_step: int,
    total_steps: int,
) -> float:
    if config.alignment_weight_start is None:
        return float(config.loss.alignment_weight)
    progress = min(max(global_step / max(total_steps - 1, 1), 0.0), 1.0)
    return float(
        config.alignment_weight_start
        + (config.alignment_weight_end - config.alignment_weight_start) * progress
    )


def enforce_training_stage_modes(model: torch.nn.Module, config: TrainConfig) -> None:
    """Keep frozen teacher-side WCM modules deterministic during warmup."""
    if config.training_stage not in {"spacetime_align", "spacetime_gate"}:
        return
    raw_model = unwrap_model(model)
    for name in (
        "language_encoder",
        "view_attention",
        "language_fusion",
        "context_trunk",
        "value_head",
        "dynamics",
        "risk_head",
        "q_action_encoder",
        "q_head",
    ):
        module = getattr(raw_model, name, None)
        if module is not None:
            module.eval()


def create_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    register_lr_scale = getattr(config.optim, "register_lr_scale", 1.0)
    temporal_lr_scale = getattr(config.optim, "temporal_lr_scale", 1.0)
    if register_lr_scale == 1.0 and temporal_lr_scale == 1.0:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("Model has no trainable parameters.")
        return torch.optim.AdamW(
            parameters,
            lr=config.optim.lr,
            betas=config.optim.betas,
            weight_decay=config.optim.weight_decay,
        )
    # register token 单独 lr 缩放：按参数名分组，register 用 lr*scale，其余用 base lr。
    # 空间 register（vision.register_tokens）、时间 register（temporal_registers.registers）、
    # 块融合寄存（block_register_fusion 的 *_registers）都命中；块融合的 attn 投影
    # 参数名含 block_register_fusion 但不含 registers，落回主组（避免整块网络被放大）。
    main, register, temporal = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(
            (
                "temporal_adapter.",
                "sparse_memory_encoder.",
                "mosaic_temporal_residual.",
                "spacetime_encoder.",
            )
        ):
            temporal.append(parameter)
        elif (
            "register_tokens" in name
            or "temporal_registers" in name
            or "spatial_registers" in name
        ):
            register.append(parameter)
        else:
            main.append(parameter)
    if not main and not register and not temporal:
        raise ValueError("Model has no trainable parameters.")
    groups = [{"params": main, "lr": config.optim.lr}]
    if register:
        groups.append(
            {"params": register, "lr": config.optim.lr * register_lr_scale}
        )
    if temporal:
        groups.append(
            {"params": temporal, "lr": config.optim.lr * temporal_lr_scale}
        )
    return torch.optim.AdamW(
        groups,
        lr=config.optim.lr,
        betas=config.optim.betas,
        weight_decay=config.optim.weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
):
    total_steps = max(1, math.ceil(steps_per_epoch / config.gradient_accumulation_steps) * config.epochs)
    warmup = min(config.optim.warmup_steps, total_steps - 1)

    def schedule(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return config.optim.min_lr_ratio + (1 - config.optim.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def wrap_ddp(model: torch.nn.Module, ctx: DistributedContext) -> torch.nn.Module:
    if not ctx.distributed:
        return model
    kwargs = {
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "broadcast_buffers": False,
    }
    if ctx.device.type == "cuda":
        kwargs.update(device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    return DistributedDataParallel(model, **kwargs)


def pairwise_ranking_sum(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    min_target_gap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a RankNet loss sum/count over unordered, non-tied pairs.

    Higher supervised return means higher value.  Pair selection is based only
    on detached targets, while gradients flow through both predictions.  The
    caller performs global-count normalization so DDP retains true mean-loss
    semantics.
    """

    prediction = prediction.reshape(-1)
    target = target.reshape(-1).detach()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Ranking prediction/target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if prediction.numel() < 2:
        return prediction.sum() * 0.0, prediction.new_zeros((), dtype=torch.float64)

    row, col = torch.triu_indices(
        prediction.numel(), prediction.numel(), offset=1, device=prediction.device
    )
    target_delta = target[row] - target[col]
    finite = (
        torch.isfinite(prediction[row])
        & torch.isfinite(prediction[col])
        & torch.isfinite(target_delta)
    )
    selected = finite & (target_delta.abs() >= min_target_gap)
    pair_count = selected.sum().to(dtype=torch.float64)
    if not selected.any():
        return prediction.sum() * 0.0, pair_count

    direction = target_delta[selected].sign()
    prediction_delta = prediction[row[selected]] - prediction[col[selected]]
    losses = F.softplus(-direction * prediction_delta / temperature)
    return losses.sum(), pair_count


def compute_losses(
    output,
    batch: dict[str, Any],
    loss_config: LossConfig,
    sigreg: SIGReg | None,
    ctx: DistributedContext,
    global_step: int,
    alignment_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = output.valid_mask.unsqueeze(-1)
    return_target = canonicalize_return_target(batch["return_targets"])
    value_sum, value_count = masked_squared_error(output.value, return_target, valid)
    value_loss, global_value_count = ddp_global_mean_loss(value_sum, value_count, ctx)
    global_value_sum = value_sum.detach().double()
    all_reduce_sum(global_value_sum, ctx)
    value_metric = global_value_sum / global_value_count.clamp_min(1)

    ranking_loss = output.value.new_zeros(())
    ranking_metric = output.value.new_zeros((), dtype=torch.float64)
    global_ranking_count = output.value.new_zeros((), dtype=torch.float64)
    if loss_config.ranking_weight > 0:
        valid_steps = output.valid_mask.bool()
        positions = torch.arange(
            valid_steps.size(1), device=valid_steps.device
        ).expand_as(valid_steps)
        last_index = positions.masked_fill(~valid_steps, -1).max(dim=1).values
        if (last_index < 0).any():
            raise ValueError("Ranking loss requires at least one valid timestep per window.")
        row_index = torch.arange(valid_steps.size(0), device=valid_steps.device)
        ranking_sum, ranking_count = pairwise_ranking_sum(
            output.value[row_index, last_index],
            return_target[row_index, last_index],
            temperature=loss_config.ranking_temperature,
            min_target_gap=loss_config.ranking_min_target_gap,
        )
        ranking_loss, global_ranking_count = ddp_global_mean_loss(
            ranking_sum, ranking_count, ctx
        )
        global_ranking_sum = ranking_sum.detach().double()
        all_reduce_sum(global_ranking_sum, ctx)
        ranking_metric = global_ranking_sum / global_ranking_count.clamp_min(1)

    state_valid = valid.expand_as(output.next_state_pred)
    state_sum, state_count = masked_squared_error(
        output.next_state_pred,
        output.target_next_state,
        state_valid,
    )
    state_loss, global_state_count = ddp_global_mean_loss(state_sum, state_count, ctx)
    global_state_sum = state_sum.detach().double()
    all_reduce_sum(global_state_sum, ctx)
    state_metric = global_state_sum / global_state_count.clamp_min(1)

    vector_loss = output.next_state_pred.new_zeros(())
    vector_metric = output.next_state_pred.new_zeros((), dtype=torch.float64)
    global_vector_count = output.next_state_pred.new_zeros((), dtype=torch.float64)
    if output.next_state_vector_pred is not None and "next_state_vector" in batch:
        vector_valid = valid.expand_as(output.next_state_vector_pred)
        vector_sum, vector_count = masked_squared_error(
            output.next_state_vector_pred,
            batch["next_state_vector"],
            vector_valid,
        )
        vector_loss, global_vector_count = ddp_global_mean_loss(vector_sum, vector_count, ctx)
        global_vector_sum = vector_sum.detach().double()
        all_reduce_sum(global_vector_sum, ctx)
        vector_metric = global_vector_sum / global_vector_count.clamp_min(1)

    sigreg_loss = output.context_latent.new_zeros(())
    if sigreg is not None and loss_config.sigreg_weight > 0:
        global_latent = differentiable_global_gather(output.context_latent, ctx).float()
        projections = synchronized_projections(
            global_latent.size(-1),
            loss_config.sigreg_num_projections,
            global_latent.device,
            torch.float32,
            ctx,
            seed=global_step + 1729,
        )
        sigreg_loss = sigreg(global_latent.transpose(0, 1), projections)

    alignment_loss = output.context_latent.new_zeros(())
    alignment_metric = output.context_latent.new_zeros((), dtype=torch.float64)
    alignment_cosine_metric = output.context_latent.new_zeros((), dtype=torch.float64)
    effective_alignment_weight = (
        loss_config.alignment_weight if alignment_weight is None else alignment_weight
    )
    if effective_alignment_weight > 0:
        if output.alignment_student is None or output.alignment_teacher is None:
            raise ValueError("Alignment loss requires student and teacher visual latents.")
        student = output.alignment_student.float()
        teacher = output.alignment_teacher.detach().float()
        cosine_similarity = F.cosine_similarity(student, teacher, dim=-1)
        cosine_distance = 1.0 - cosine_similarity
        student_norm = F.layer_norm(student, (student.size(-1),))
        teacher_norm = F.layer_norm(teacher, (teacher.size(-1),))
        normalized_mse = (student_norm - teacher_norm).square().mean(dim=-1)
        alignment_values = cosine_distance + loss_config.alignment_mse_weight * normalized_mse
        alignment_sum = alignment_values.sum()
        alignment_count = torch.tensor(
            alignment_values.numel(), device=alignment_values.device, dtype=torch.float64
        )
        alignment_loss, global_alignment_count = ddp_global_mean_loss(
            alignment_sum, alignment_count, ctx
        )
        global_alignment_sum = alignment_sum.detach().double()
        all_reduce_sum(global_alignment_sum, ctx)
        alignment_metric = global_alignment_sum / global_alignment_count.clamp_min(1)
        global_alignment_cosine_sum = cosine_similarity.detach().double().sum()
        all_reduce_sum(global_alignment_cosine_sum, ctx)
        alignment_cosine_metric = (
            global_alignment_cosine_sum / global_alignment_count.clamp_min(1)
        )

    risk_loss = output.context_latent.new_zeros(())
    risk_metric = output.context_latent.new_zeros((), dtype=torch.float64)
    if loss_config.risk_weight > 0:
        if output.risk_logits is None or "success_targets" not in batch:
            raise ValueError("Risk loss requires risk_logits and success_targets.")
        risk_targets = 1.0 - batch["success_targets"].to(output.risk_logits.dtype)
        risk_values = F.binary_cross_entropy_with_logits(
            output.risk_logits, risk_targets, reduction="none"
        )
        risk_sum = risk_values.masked_select(valid).sum()
        risk_count = valid.sum().to(dtype=torch.float64)
        risk_loss, global_risk_count = ddp_global_mean_loss(risk_sum, risk_count, ctx)
        global_risk_sum = risk_sum.detach().double()
        all_reduce_sum(global_risk_sum, ctx)
        risk_metric = global_risk_sum / global_risk_count.clamp_min(1)

    q_loss = output.context_latent.new_zeros(())
    q_metric = output.context_latent.new_zeros((), dtype=torch.float64)
    if loss_config.q_weight > 0:
        if output.q_value is None:
            raise ValueError("Q loss requires q_value output.")
        q_sum, q_count = masked_squared_error(output.q_value, return_target, valid)
        q_loss, global_q_count = ddp_global_mean_loss(q_sum, q_count, ctx)
        global_q_sum = q_sum.detach().double()
        all_reduce_sum(global_q_sum, ctx)
        q_metric = global_q_sum / global_q_count.clamp_min(1)

    total = (
        loss_config.value_weight * value_loss
        + loss_config.ranking_weight * ranking_loss
        + loss_config.next_state_weight * state_loss
        + loss_config.next_state_vector_weight * vector_loss
        + loss_config.sigreg_weight * sigreg_loss
        + effective_alignment_weight * alignment_loss
        + loss_config.risk_weight * risk_loss
        + loss_config.q_weight * q_loss
    )
    total_metric = (
        loss_config.value_weight * value_metric
        + loss_config.ranking_weight * ranking_metric
        + loss_config.next_state_weight * state_metric
        + loss_config.next_state_vector_weight * vector_metric
        + loss_config.sigreg_weight * sigreg_loss.detach().double()
        + effective_alignment_weight * alignment_metric
        + loss_config.risk_weight * risk_metric
        + loss_config.q_weight * q_metric
    )
    return total, {
        "loss": total_metric,
        "value_loss": value_metric,
        "ranking_loss": ranking_metric,
        "next_state_loss": state_metric,
        "next_state_vector_loss": vector_metric,
        "sigreg_loss": sigreg_loss.detach(),
        "alignment_loss": alignment_metric,
        "alignment_cosine": alignment_cosine_metric,
        "alignment_weight": output.context_latent.new_tensor(effective_alignment_weight),
        "risk_loss": risk_metric,
        "q_loss": q_metric,
        "global_value_count": global_value_count.detach(),
        "global_ranking_count": global_ranking_count.detach(),
        "global_state_count": global_state_count.detach(),
        "global_vector_count": global_vector_count.detach(),
    }


@torch.inference_mode()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    config: TrainConfig,
    ctx: DistributedContext,
    max_batches: int | None = None,
    collect_episode_curves: bool = False,
    log_every_batches: int | None = None,
    teacher_model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate online (one endpoint per window) and token-level metrics.

    ``LeRobotWorldCriticDataset`` exposes overlapping history windows.  Scalar
    token metrics intentionally use every valid history position, because that
    is the training objective.  The public ``value_*`` and ``next_state_mse``
    metrics use a different, explicit online convention: each window
    contributes only its last valid timestep.  That timestep has the complete
    configured history and maps to one unique frame.  The corresponding
    ``token_value_*`` and ``token_next_state_mse`` fields remain available for
    diagnosing the dense training objective.  When requested, curve records
    are gathered to rank 0, sorted by ``(episode_id, frame_index)``, and
    duplicate keys are rejected rather than silently averaged/truncated.
    """
    if log_every_batches is not None and log_every_batches < 1:
        raise ValueError("log_every_batches must be positive when provided.")

    model.eval()
    token_metrics = RegressionMetrics()
    endpoint_metrics = RegressionMetrics()
    token_latent_squared_error = 0.0
    token_latent_count = 0.0
    endpoint_latent_squared_error = 0.0
    endpoint_latent_count = 0.0
    alignment_loss_sum = 0.0
    alignment_cosine_sum = 0.0
    alignment_count = 0.0
    curve_records: list[dict[str, Any]] = []
    processed_curve_samples = 0
    local_error: str | None = None
    loop_started = time.monotonic()
    try:
        loader_batches = len(loader)
    except Exception:
        loader_batches = "unknown"
    _evaluation_log(
        ctx,
        "evaluation loop start: "
        f"loader_batches={loader_batches}, max_batches={max_batches}, "
        f"collect_episode_curves={collect_episode_curves}, precision={config.precision}",
    )
    try:
        # Pull batches manually so max_batches does not request one extra
        # batch before stopping.  The old ``for batch in loader`` form made a
        # one-batch smoke test decode a second batch before checking the limit.
        batch_index = 0
        loader_iterator = iter(loader)
        while max_batches is None or batch_index < max_batches:
            should_log = (
                batch_index == 0
                or (log_every_batches is not None and batch_index % log_every_batches == 0)
            )
            if should_log:
                _evaluation_log(ctx, f"requesting batch {batch_index}...", main_only=True)
            fetch_started = time.monotonic()
            try:
                batch = next(loader_iterator)
            except StopIteration:
                break
            fetch_seconds = time.monotonic() - fetch_started
            if should_log:
                image_shape = (
                    tuple(batch["images"].shape)
                    if torch.is_tensor(batch.get("images"))
                    else None
                )
                _evaluation_log(
                    ctx,
                    f"received batch {batch_index}: image_shape={image_shape}, "
                    f"fetch_s={fetch_seconds:.3f}",
                    main_only=True,
                )
            forward_started = time.monotonic()
            batch = move_batch_to_device(batch, ctx.device)
            teacher_current_state = None
            if teacher_model is not None:
                with autocast_context(ctx.device, config.precision):
                    teacher_current_state = teacher_model(batch["images"][:, :1])
            with autocast_context(ctx.device, config.precision):
                output = unwrap_model(model)(
                    images=batch["images"],
                    actions=batch["actions"],
                    instruction_input_ids=batch["instruction_input_ids"],
                    instruction_attention_mask=batch["instruction_attention_mask"],
                    valid_mask=batch["valid_mask"],
                    state_vectors=batch.get("state_vectors"),
                    history_images=batch.get("history_images"),
                    teacher_current_state=teacher_current_state,
                )
            return_target = canonicalize_return_target(batch["return_targets"])
            valid = output.valid_mask.bool()
            token_mask = valid.unsqueeze(-1)
            if not torch.isfinite(output.value[token_mask]).all() or not torch.isfinite(
                return_target[token_mask]
            ).all():
                raise ValueError("Evaluation value tensors contain non-finite valid tokens.")
            token_metrics.update(output.value, return_target, token_mask)

            # A window can contain padding in future dataset variants.  Pick
            # its final valid history position explicitly instead of assuming
            # that ``T - 1`` is valid.  This is the same endpoint used by the
            # episode-curve writer below.
            batch_size, time_steps = valid.shape
            positions = torch.arange(time_steps, device=valid.device).expand(batch_size, -1)
            last_index = positions.masked_fill(~valid, -1).max(dim=1).values
            if (last_index < 0).any():
                raise ValueError("Every evaluation window must contain at least one valid timestep.")
            row_index = torch.arange(batch_size, device=valid.device)
            endpoint_values = output.value[row_index, last_index]
            endpoint_returns = return_target[row_index, last_index]
            if not torch.isfinite(endpoint_values).all() or not torch.isfinite(endpoint_returns).all():
                raise ValueError("Evaluation endpoint value tensors contain non-finite values.")
            endpoint_metrics.update(
                endpoint_values,
                endpoint_returns,
                torch.ones_like(endpoint_values, dtype=torch.bool),
            )
            if output.alignment_student is not None and output.alignment_teacher is not None:
                student = output.alignment_student.float()
                teacher = output.alignment_teacher.float()
                cosine = F.cosine_similarity(student, teacher, dim=-1)
                student_norm = F.layer_norm(student, (student.size(-1),))
                teacher_norm = F.layer_norm(teacher, (teacher.size(-1),))
                normalized_mse = (student_norm - teacher_norm).square().mean(dim=-1)
                alignment_values = (
                    1.0 - cosine + config.loss.alignment_mse_weight * normalized_mse
                )
                alignment_loss_sum += alignment_values.double().sum().item()
                alignment_cosine_sum += cosine.double().sum().item()
                alignment_count += float(cosine.numel())

            mask = token_mask.expand_as(output.next_state_pred)
            if not torch.isfinite(output.next_state_pred[mask]).all() or not torch.isfinite(
                output.target_next_state[mask]
            ).all():
                raise ValueError("Evaluation next-state tensors contain non-finite valid tokens.")
            error = (output.next_state_pred.double() - output.target_next_state.double()).square()
            token_latent_squared_error += error.masked_fill(~mask, 0.0).sum().item()
            token_latent_count += float(mask.sum().item())

            endpoint_state_pred = output.next_state_pred[row_index, last_index]
            endpoint_state_target = output.target_next_state[row_index, last_index]
            if not torch.isfinite(endpoint_state_pred).all() or not torch.isfinite(endpoint_state_target).all():
                raise ValueError("Evaluation endpoint next-state tensors contain non-finite values.")
            endpoint_state_error = (endpoint_state_pred.double() - endpoint_state_target.double()).square()
            endpoint_latent_squared_error += endpoint_state_error.sum().item()
            endpoint_latent_count += float(endpoint_state_error.numel())

            if collect_episode_curves:
                endpoint_episodes = batch["episode_id"][row_index]
                endpoint_frames = batch["frame_indices"][row_index, last_index]
                curve_values = endpoint_values.squeeze(-1)
                curve_returns = endpoint_returns.squeeze(-1)
                if not torch.isfinite(curve_values).all() or not torch.isfinite(curve_returns).all():
                    raise ValueError("Episode curve endpoints contain non-finite value or return targets.")
                processed_curve_samples += batch_size
                curve_records.extend(
                    {
                        "episode_id": int(episode_id),
                        "frame_index": int(frame_index),
                        "value": float(value),
                        "return": float(target),
                    }
                    for episode_id, frame_index, value, target in zip(
                        endpoint_episodes.detach().cpu().tolist(),
                        endpoint_frames.detach().cpu().tolist(),
                        curve_values.detach().cpu().tolist(),
                        curve_returns.detach().cpu().tolist(),
                        strict=True,
                    )
                )
            forward_seconds = time.monotonic() - forward_started
            batch_index += 1
            if should_log:
                _evaluation_log(
                    ctx,
                    f"processed batch {batch_index - 1}: forward_s={forward_seconds:.3f}, "
                    f"elapsed_s={time.monotonic() - loop_started:.1f}",
                    main_only=True,
                )
    except Exception as exc:
        # A malformed sample or a worker/model exception must be surfaced to
        # every rank before metric all-reduces.  Otherwise healthy ranks can
        # block indefinitely while the failing rank unwinds into cleanup.
        local_error = repr(exc)
        _evaluation_log(
            ctx,
            f"evaluation loop raised after {batch_index} processed batches: {local_error}",
        )

    _evaluation_log(
        ctx,
        f"evaluation loop finished: processed_batches={batch_index}, "
        f"elapsed_s={time.monotonic() - loop_started:.1f}",
        main_only=True,
    )

    def validate_evaluation_loop() -> None:
        if local_error is not None:
            raise RuntimeError(local_error)

    collectively_validate(ctx, "Evaluation loop", validate_evaluation_loop)

    if collect_episode_curves:
        gathered = gather_objects(
            {"records": curve_records, "samples": processed_curve_samples},
            ctx,
            dst=0,
        )
        curve_payload: list[dict[str, Any]] | None = None
        curve_error: str | None = None
        if ctx.is_main:
            try:
                assert gathered is not None
                records: list[dict[str, Any]] = []
                sample_count = 0
                for rank_payload in gathered:
                    if not isinstance(rank_payload, dict):
                        raise TypeError("Distributed episode-curve payload is malformed.")
                    records.extend(rank_payload["records"])
                    sample_count += int(rank_payload["samples"])
                if len(records) != sample_count:
                    raise ValueError(
                        "Episode curve endpoint count does not match processed windows: "
                        f"records={len(records)}, windows={sample_count}."
                    )
                curve_payload = build_episode_curves(records)
            except Exception as exc:
                curve_error = repr(exc)

        def validate_curve_assembly() -> None:
            if curve_error is not None:
                raise RuntimeError(curve_error)

        collectively_validate(ctx, "Episode curve assembly", validate_curve_assembly)
        metrics_with_curves: dict[str, Any] = {"episode_curves": curve_payload if ctx.is_main else None}
    else:
        metrics_with_curves = {}

    # ``value_*`` is intentionally the online endpoint metric.  Keep the
    # dense token metric under an explicit prefix so downstream users cannot
    # accidentally interpret overlapping-window statistics as per-frame
    # evaluation.
    reduced = {f"value_{key}": value for key, value in endpoint_metrics.reduce(ctx).compute().items()}
    reduced.update(
        {
            f"token_value_{key}": value
            for key, value in token_metrics.reduce(ctx).compute().items()
        }
    )
    endpoint_state_values = torch.tensor(
        [endpoint_latent_squared_error, endpoint_latent_count],
        dtype=torch.float64,
        device=ctx.device,
    )
    if ctx.distributed:
        dist.all_reduce(endpoint_state_values, op=dist.ReduceOp.SUM)
    reduced["next_state_mse"] = (
        endpoint_state_values[0].item() / endpoint_state_values[1].item()
        if endpoint_state_values[1].item() > 0
        else math.nan
    )
    token_state_values = torch.tensor(
        [token_latent_squared_error, token_latent_count],
        dtype=torch.float64,
        device=ctx.device,
    )
    if ctx.distributed:
        dist.all_reduce(token_state_values, op=dist.ReduceOp.SUM)
    reduced["token_next_state_mse"] = (
        token_state_values[0].item() / token_state_values[1].item()
        if token_state_values[1].item() > 0
        else math.nan
    )
    alignment_values = torch.tensor(
        [alignment_loss_sum, alignment_cosine_sum, alignment_count],
        dtype=torch.float64,
        device=ctx.device,
    )
    if ctx.distributed:
        dist.all_reduce(alignment_values, op=dist.ReduceOp.SUM)
    reduced["alignment_loss"] = (
        alignment_values[0].item() / alignment_values[2].item()
        if alignment_values[2].item() > 0
        else math.nan
    )
    reduced["alignment_cosine"] = (
        alignment_values[1].item() / alignment_values[2].item()
        if alignment_values[2].item() > 0
        else math.nan
    )
    reduced.update(metrics_with_curves)
    return reduced


def config_from_checkpoint_payload(payload: dict[str, Any]) -> TrainConfig:
    values = payload["config"]
    model = values.get("model", {})
    if (
        model.get("use_block_register_fusion", False)
        and (
            int(model.get("register_block_size", 1)) > 1
            or float(model.get("register_future_leak_ratio", 0.0)) > 0
        )
        and "allow_noncausal_register_ablation" not in model
    ):
        # Old checkpoints predate the explicit safety flag. Preserve their
        # evaluation/reproduction path without weakening new config validation.
        values = dict(values)
        values["model"] = dict(model)
        values["model"]["allow_noncausal_register_ablation"] = True
    return _construct(TrainConfig, values)
