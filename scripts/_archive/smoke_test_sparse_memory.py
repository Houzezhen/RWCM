"""One sparse-memory forward/backward step with CUDA memory and gradient checks."""
from __future__ import annotations

import argparse

import torch

from world_critic.config import load_config, validate_train_config
from world_critic.model import WorldCriticModel
from world_critic.training import autocast_context, create_optimizer


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wcm_sparse_memory_exp_s3072.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    config = load_config(args.config)
    validate_train_config(config)
    config.model.action_dim = 7
    model = WorldCriticModel(config.model).to(device).train()
    optimizer = create_optimizer(model, config)
    batch = args.batch_size
    views = len(config.data.image_keys)
    size = config.model.vision.image_size
    frames = config.model.sparse_memory_frame_count
    images = torch.randn(batch, 2, views, 3, size, size, device=device)
    history_images = torch.randn(batch, frames, views, 3, size, size, device=device)
    actions = torch.randn(batch, 1, config.model.action_dim, device=device)
    input_ids = torch.zeros(batch, 8, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    valid_mask = torch.ones(batch, 1, dtype=torch.bool, device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, config.precision):
        output = model(
            images=images,
            actions=actions,
            instruction_input_ids=input_ids,
            instruction_attention_mask=attention_mask,
            valid_mask=valid_mask,
            history_images=history_images,
        )
        loss = output.value.square().mean() + output.next_state_pred.square().mean()
    loss.backward()

    memory = model.sparse_memory_encoder
    gradients = {
        "memory_queries": memory.memory_queries.grad,
        "resampler_qkv": memory.resample_attention.in_proj_weight.grad,
        "first_global_qkv": memory.global_layers[0].attention.in_proj_weight.grad,
        "readout_qkv": memory.readout_attention.in_proj_weight.grad,
    }
    gradient_norms = {
        name: float(gradient.norm()) if gradient is not None else 0.0
        for name, gradient in gradients.items()
    }
    dead = [name for name, norm in gradient_norms.items() if norm <= 0]
    if dead:
        raise RuntimeError(f"Sparse-memory parameters received no gradient: {dead}")
    optimizer.step()

    result = {
        "value_shape": tuple(output.value.shape),
        "next_state_shape": tuple(output.next_state_pred.shape),
        "loss": float(loss.detach()),
        "memory_tokens_per_frame": memory.memory_count,
        "global_sequence_tokens": memory.frame_count * memory.memory_count,
        "gradient_norms": gradient_norms,
        "optimizer_lrs": sorted({group["lr"] for group in optimizer.param_groups}),
    }
    if device.type == "cuda":
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
    print(result, flush=True)


if __name__ == "__main__":
    run()
