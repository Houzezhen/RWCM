"""Validate and inventory all checkpoints used by the SpaceTime three-way experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


ROLES = {
    "baseline": "wcm_baseline_exp_s{seed}/deploy.pt",
    "mosaic": "wcm_sparse4_image_b4_exp_s{seed}/deploy.pt",
    "spacetime": "wcm_spacetime_vit_t8_direct_full_s{seed}/deploy.pt",
    "staged": "wcm_spacetime_vit_t8_full_s{seed}/deploy.pt",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint-root", type=Path, required=True)
    result.add_argument("--seeds", type=int, nargs="+", default=[3072, 42])
    result.add_argument("--output", type=Path, required=True)
    return result


def _validate_role(role: str, payload: dict[str, Any], seed: int, path: Path) -> dict[str, Any]:
    if payload.get("artifact_type") != "deploy":
        raise ValueError(f"{role} is not a deploy checkpoint: {path}")
    config = payload.get("config")
    state = payload.get("model")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError(f"Checkpoint lacks config/model dictionaries: {path}")
    if int(config.get("seed", -1)) != seed:
        raise ValueError(f"Checkpoint seed mismatch for {path}: {config.get('seed')} != {seed}")
    data = config.get("data", {})
    model = config.get("model", {})
    if data.get("repo_id") != "data_toiletButton_0814_expanded_with_return":
        raise ValueError(f"Unexpected training dataset in {path}: {data.get('repo_id')!r}")

    is_spacetime = bool(model.get("use_spacetime_perceiver", False))
    if role == "baseline":
        if data.get("history_mosaic", False) or is_spacetime:
            raise ValueError(f"Baseline checkpoint has a temporal visual extension: {path}")
    elif role == "mosaic":
        if not data.get("history_mosaic", False):
            raise ValueError(f"Mosaic checkpoint has history_mosaic=false: {path}")
        if list(data.get("history_offsets") or []) != [0, 20, 40, 60]:
            raise ValueError(f"Mosaic checkpoint has unexpected history offsets: {path}")
    else:
        if not is_spacetime:
            raise ValueError(f"SpaceTime role does not enable the encoder: {path}")
        if list(data.get("history_offsets") or []) != [0, 9, 17, 26, 34, 43, 51, 60]:
            raise ValueError(f"SpaceTime checkpoint has unexpected history offsets: {path}")
        if data.get("history_mosaic", False) or model.get("spacetime_teacher_enabled", False):
            raise ValueError(f"Deploy SpaceTime checkpoint still depends on the teacher: {path}")
        gate = state.get("spacetime_encoder.blend_gate")
        if gate is None or float(gate) != 1.0:
            raise ValueError(f"SpaceTime deploy gate is not 1 for {path}: {gate}")

    state_tensor_elements = sum(
        int(tensor.numel()) for tensor in state.values() if torch.is_tensor(tensor)
    )
    return {
        "path": str(path.resolve()),
        "artifact_type": payload["artifact_type"],
        "schema_version": payload.get("schema_version"),
        "seed": seed,
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "state_tensor_elements": state_tensor_elements,
        "training_dataset": {
            "repo_id": data.get("repo_id"),
            "revision": data.get("revision"),
        },
        "architecture": {
            "history_offsets": data.get("history_offsets"),
            "history_mosaic": bool(data.get("history_mosaic", False)),
            "use_spacetime_perceiver": is_spacetime,
            "spacetime_teacher_enabled": bool(model.get("spacetime_teacher_enabled", False)),
            "spacetime_frame_count": model.get("spacetime_frame_count") if is_spacetime else None,
        },
    }


def inspect(checkpoint_root: Path, seeds: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"checkpoint_root": str(checkpoint_root.resolve()), "models": {}}
    for seed in seeds:
        result["models"][str(seed)] = {}
        for role, pattern in ROLES.items():
            path = checkpoint_root / pattern.format(seed=seed)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {role} checkpoint for seed {seed}: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            result["models"][str(seed)][role] = _validate_role(role, payload, seed, path)
    return result


def run() -> None:
    args = parser().parse_args()
    result = inspect(args.checkpoint_root, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Validated {len(args.seeds) * len(ROLES)} checkpoints: {args.output.resolve()}")


if __name__ == "__main__":
    run()
