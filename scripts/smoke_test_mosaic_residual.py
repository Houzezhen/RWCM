"""One mosaic-residual forward/backward step with CUDA gradient checks."""
from __future__ import annotations

import argparse

import torch

from world_critic.config import load_config, validate_train_config
from world_critic.model import WorldCriticModel
from world_critic.training import autocast_context, create_optimizer


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wcm_mosaic_residual_s3072.yaml")
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
    residual = model.mosaic_temporal_residual
    if residual is None:
        raise RuntimeError("Config did not construct mosaic_temporal_residual.")
    # Unit tests cover exact zero-gate identity. Open it slightly here so every
    # adapter parameter must receive a gradient in the real bf16 CUDA graph.
    residual.residual_gate.data.fill_(1.0e-3)

    batch = args.batch_size
    views = len(config.data.image_keys)
    size = config.model.vision.image_size
    images = torch.randn(batch, 2, views, 3, size, size, device=device)
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
        )
        loss = output.value.square().mean() + output.next_state_pred.square().mean()
    loss.backward()

    gradients = {
        "gate": residual.residual_gate.grad,
        "time_embedding": residual.time_embedding.grad,
        "attention_qkv": residual.attention.in_proj_weight.grad,
        "ffn": residual.ffn.net[0].weight.grad,
    }
    gradient_norms = {
        name: float(gradient.norm()) if gradient is not None else 0.0
        for name, gradient in gradients.items()
    }
    dead = [name for name, norm in gradient_norms.items() if norm <= 0]
    if dead:
        raise RuntimeError(f"Mosaic residual parameters received no gradient: {dead}")
    optimizer.step()

    result = {
        "value_shape": tuple(output.value.shape),
        "next_state_shape": tuple(output.next_state_pred.shape),
        "loss": float(loss.detach()),
        "gradient_norms": gradient_norms,
        "optimizer_lrs": [group["lr"] for group in optimizer.param_groups],
    }
    if device.type == "cuda":
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
    print(result, flush=True)


if __name__ == "__main__":
    run()
