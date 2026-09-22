"""Smoke-test K=0 and standard early K=4 on the configured Hugging Face ViT."""
from __future__ import annotations

import argparse

import torch

from world_critic.config import ModelConfig, VisionConfig
from world_critic.model import VisionEncoder


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--model-name",
        default="google/vit-base-patch16-224-in21k",
    )
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--pretrained", action="store_true")
    return result


def run_case(args: argparse.Namespace, registers: int) -> dict[str, float | int]:
    config = ModelConfig(
        latent_dim=64,
        max_views=1,
        vision=VisionConfig(
            model_name=args.model_name,
            image_size=args.image_size,
            trainable=True,
            pretrained=args.pretrained,
            num_register_tokens=registers,
            register_insert="early",
        ),
    )
    encoder = VisionEncoder(config).to(args.device).train()
    calls = 0

    def count_layer_call(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handle = encoder.encoder_layers()[-1].register_forward_hook(count_layer_call)
    try:
        images = torch.randn(
            1,
            2,
            1,
            3,
            args.image_size,
            args.image_size,
            device=args.device,
        )
        output = encoder(images)
        if output.shape != (1, 2, 1, config.latent_dim):
            raise AssertionError(f"Unexpected output shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise AssertionError("Vision output contains non-finite values.")
        output.square().mean().backward()
        if calls != 1:
            raise AssertionError(f"Expected one encoder pass, observed last-layer calls={calls}.")
        gradient_norm = 0.0
        if registers:
            gradient = encoder.register_tokens.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise AssertionError("Register gradient is missing or non-finite.")
            gradient_norm = float(gradient.norm())
            if gradient_norm <= 0:
                raise AssertionError("Register gradient is zero.")
        return {
            "registers": registers,
            "encoder_passes": calls,
            "register_gradient_norm": gradient_norm,
        }
    finally:
        handle.remove()


def main() -> None:
    args = parser().parse_args()
    if args.image_size < 1:
        raise ValueError("--image-size must be positive.")
    results = [run_case(args, registers) for registers in (0, 4)]
    print({"model": args.model_name, "device": args.device, "results": results})


if __name__ == "__main__":
    main()
