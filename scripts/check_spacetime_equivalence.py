"""Numerically verify gate=0 baseline identity and gate=1 teacher independence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from world_critic.model import WorldCriticModel
from world_critic.training import config_from_checkpoint_payload


OUTPUT_FIELDS = (
    "context_latent",
    "value",
    "next_state_pred",
    "target_next_state",
)
DEPLOY_OUTPUT_FIELDS = OUTPUT_FIELDS + (
    "next_state_vector_pred",
    "risk_logits",
    "q_value",
)


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("artifact_type") not in {"deploy", "full_resume"}:
        raise ValueError(f"Unsupported checkpoint artifact: {path}")
    return payload


def load_model(path: Path, device: torch.device) -> tuple[WorldCriticModel, object]:
    payload = load_payload(path)
    config = config_from_checkpoint_payload(payload)
    model = WorldCriticModel(config.model).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return model, config


def max_output_error(left, right, fields=OUTPUT_FIELDS) -> dict[str, float]:
    errors = {}
    for name in fields:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None and right_value is None:
            continue
        if left_value is None or right_value is None:
            raise ValueError(f"Output presence differs for {name}.")
        errors[name] = float((left_value.float() - right_value.float()).abs().max())
    return errors


def require_tolerance(label: str, errors: dict[str, float], tolerance: float) -> None:
    worst = max(errors.values())
    print(json.dumps({label: errors, "worst": worst, "tolerance": tolerance}))
    if not math.isfinite(worst) or worst >= tolerance:
        raise SystemExit(f"{label} failed: max error {worst:.8g} >= {tolerance:.8g}")


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gate-zero", type=Path)
    parser.add_argument("--gate-one", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()

    if args.tolerance <= 0:
        raise ValueError("--tolerance must be positive.")
    if args.gate_zero is None and args.gate_one is None:
        parser.error("provide --gate-zero and/or --gate-one")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    torch.manual_seed(0)

    reference_path = args.gate_zero if args.gate_zero is not None else args.gate_one
    reference_payload = load_payload(reference_path)
    reference_config = config_from_checkpoint_payload(reference_payload)
    batch = 1
    views = len(reference_config.data.image_keys)
    frames = reference_config.model.spacetime_frame_count
    image_size = reference_config.model.vision.image_size
    action_dim = reference_config.model.action_dim
    if action_dim is None:
        raise ValueError("Checkpoint is missing model.action_dim.")

    images = torch.randn(batch, 2, views, 3, image_size, image_size, device=device)
    history = torch.randn(batch, frames, views, 3, image_size, image_size, device=device)
    actions = torch.randn(batch, 1, action_dim, device=device)
    input_ids = torch.zeros(batch, 8, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    valid_mask = torch.ones(batch, 1, dtype=torch.bool, device=device)
    inputs = {
        "images": images,
        "actions": actions,
        "instruction_input_ids": input_ids,
        "instruction_attention_mask": attention_mask,
        "valid_mask": valid_mask,
    }

    with torch.inference_mode():
        baseline, _ = load_model(args.baseline, device)
        baseline_output = baseline(**inputs)
        teacher_views = baseline.vision_encoder(images[:, :1])
        teacher_baseline = baseline.pool_views(teacher_views)
        teacher_alignment = teacher_views.mean(dim=2)
        del baseline
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.gate_zero is not None:
            gate_zero, _ = load_model(args.gate_zero, device)
            actual_gate_zero = float(gate_zero.spacetime_encoder.blend_gate)
            if actual_gate_zero != 0.0:
                raise SystemExit(f"gate-zero checkpoint contains gate={actual_gate_zero}")
            gate_zero_output = gate_zero(
                **inputs,
                history_images=history,
                teacher_current_state=teacher_baseline,
                teacher_alignment_state=teacher_alignment,
            )
            require_tolerance(
                "gate_zero_vs_baseline",
                max_output_error(gate_zero_output, baseline_output),
                args.tolerance,
            )
            del gate_zero, gate_zero_output
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if args.gate_one is not None:
            gate_one, _ = load_model(args.gate_one, device)
            actual_gate_one = float(gate_one.spacetime_encoder.blend_gate)
            if actual_gate_one != 1.0:
                raise SystemExit(f"gate-one checkpoint contains gate={actual_gate_one}")
            with_teacher = gate_one(
                **inputs,
                history_images=history,
                teacher_current_state=teacher_baseline,
                teacher_alignment_state=teacher_alignment,
            )
            without_teacher = gate_one(**inputs, history_images=history)
            require_tolerance(
                "gate_one_teacher_independence",
                max_output_error(with_teacher, without_teacher, DEPLOY_OUTPUT_FIELDS),
                args.tolerance,
            )


if __name__ == "__main__":
    run()
