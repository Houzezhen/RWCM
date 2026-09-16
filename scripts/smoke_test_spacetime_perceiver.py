"""CUDA forward/backward gate for the teacher-free SpaceTime Perceiver."""
from __future__ import annotations

import argparse

import torch

from world_critic.config import apply_runtime_overrides, load_config, validate_train_config
from world_critic.model import WorldCriticModel
from world_critic.training import autocast_context, configure_training_stage, create_optimizer


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wcm_spacetime_joint.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    config = apply_runtime_overrides(load_config(args.config))
    validate_train_config(config)
    config.model.action_dim = 7
    model = WorldCriticModel(config.model).to(device).train()
    stage = configure_training_stage(model, config)
    model.spacetime_encoder.set_gate(1.0)
    optimizer = create_optimizer(model, config)

    batch = 1
    views = len(config.data.image_keys)
    frames = config.model.spacetime_frame_count
    size = config.model.vision.image_size
    images = torch.randn(batch, 2, views, 3, size, size, device=device)
    history = torch.randn(
        batch, frames, views, 3, size, size, device=device, requires_grad=True
    )
    actions = torch.randn(batch, 1, 7, device=device)
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
            history_images=history,
        )
        if output.risk_logits is None or output.q_value is None:
            raise RuntimeError("SpaceTime smoke requires risk and Q outputs.")
        loss = (
            output.value.square().mean()
            + output.next_state_pred.square().mean()
            + output.risk_logits.square().mean()
            + output.q_value.square().mean()
        )
    loss.backward()

    gradients = {
        "oldest_history": history.grad[:, -1],
        "temporal_attention": model.spacetime_encoder.temporal_layers[0].self_attn.in_proj_weight.grad,
        "perceiver_queries": model.spacetime_encoder.perceiver_queries.grad,
        "perceiver_attention": model.spacetime_encoder.perceiver_layers[0].cross_attention.in_proj_weight.grad,
        "risk_head": model.risk_head.net[-1].weight.grad,
        "q_head": model.q_head.net[-1].weight.grad,
    }
    norms = {
        name: float(value.norm()) if value is not None else 0.0
        for name, value in gradients.items()
    }
    dead = [name for name, norm in norms.items() if norm <= 0]
    if dead:
        raise RuntimeError(f"SpaceTime smoke found dead gradients: {dead}")
    optimizer.step()

    result = {
        "stage": stage,
        "visual_tokens": config.model.perceiver_queries,
        "value_shape": tuple(output.value.shape),
        "risk_shape": tuple(output.risk_logits.shape),
        "q_shape": tuple(output.q_value.shape),
        "gradient_norms": norms,
    }
    if device.type == "cuda":
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
    print(result, flush=True)


if __name__ == "__main__":
    run()
