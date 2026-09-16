import unittest

import torch
from torch import nn

from world_critic.config import DataConfig, TrainConfig
from world_critic.training import (
    configure_training_stage,
    scheduled_alignment_weight,
    update_spacetime_gate,
)


class DummySpaceTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = nn.Parameter(torch.zeros(1))
        self.temporal_layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
        self.temporal_norm = nn.LayerNorm(2)
        self.perceiver_queries = nn.Parameter(torch.zeros(1, 64, 2))
        self.perceiver_layers = nn.ModuleList([nn.Linear(2, 2)])
        self.output_projection = nn.Linear(2, 2)
        self.register_buffer("blend_gate", torch.tensor(0.0))

    def set_gate(self, value: float):
        self.blend_gate.fill_(value)


class DummyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
        self.backbone = nn.Module()
        self.backbone.vision_model = nn.Module()
        self.backbone.vision_model.post_layernorm = nn.LayerNorm(2)

    def encoder_layers(self):
        return self.layers


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.spacetime_encoder = DummySpaceTime()
        self.vision_encoder = DummyVision()
        self.language_encoder = nn.Linear(2, 2)
        self.language_fusion = nn.Linear(2, 2)
        self.context_trunk = nn.Linear(2, 2)
        self.value_head = nn.Linear(2, 1)
        self.risk_head = nn.Linear(2, 1)
        self.q_action_encoder = nn.Linear(2, 2)
        self.q_head = nn.Linear(4, 1)


class SpaceTimeTrainingTest(unittest.TestCase):
    def config(self, stage: str) -> TrainConfig:
        config = TrainConfig(data=DataConfig(repo_id="dummy"))
        config.training_stage = stage
        config.model.use_spacetime_perceiver = True
        return config

    def test_alignment_only_unfreezes_perceiver_projection(self):
        model = DummyModel()
        configure_training_stage(model, self.config("spacetime_align"))
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        self.assertIn("spacetime_encoder.perceiver_queries", trainable)
        self.assertIn("spacetime_encoder.output_projection.weight", trainable)
        self.assertNotIn("spacetime_encoder.time_embedding", trainable)
        self.assertNotIn("spacetime_encoder.temporal_layers.0.weight", trainable)

    def test_joint_unfreezes_only_top_four_vision_layers(self):
        model = DummyModel()
        info = configure_training_stage(model, self.config("spacetime_joint"))

        self.assertEqual(info["unfrozen_vision_layers"], [8, 9, 10, 11])
        self.assertFalse(model.vision_encoder.layers[7].weight.requires_grad)
        self.assertTrue(model.vision_encoder.layers[8].weight.requires_grad)
        self.assertTrue(
            model.vision_encoder.backbone.vision_model.post_layernorm.weight.requires_grad
        )
        self.assertFalse(model.language_encoder.weight.requires_grad)

    def test_gate_schedule_reaches_exact_endpoints(self):
        model = DummyModel()
        config = self.config("spacetime_gate")
        config.gate_start = 0.0
        config.gate_end = 1.0

        self.assertEqual(update_spacetime_gate(model, config, 0, 11), 0.0)
        self.assertEqual(update_spacetime_gate(model, config, 10, 11), 1.0)

    def test_alignment_weight_decays_to_zero_with_gate(self):
        config = self.config("spacetime_gate")
        config.alignment_weight_start = 0.2
        config.alignment_weight_end = 0.0

        self.assertEqual(scheduled_alignment_weight(config, 0, 11), 0.2)
        self.assertEqual(scheduled_alignment_weight(config, 10, 11), 0.0)

    def test_full_stage_keeps_teacher_only_view_pool_frozen(self):
        model = DummyModel()
        model.vision_encoder.camera_embedding = nn.Parameter(torch.zeros(1, 1, 2, 2))
        model.view_pool_query = nn.Parameter(torch.zeros(1, 1, 2))
        model.view_attention = nn.MultiheadAttention(2, 1, batch_first=True)
        configure_training_stage(model, self.config("spacetime_full"))

        self.assertFalse(model.vision_encoder.camera_embedding.requires_grad)
        self.assertFalse(model.view_pool_query.requires_grad)
        self.assertFalse(model.view_attention.in_proj_weight.requires_grad)
        self.assertTrue(model.vision_encoder.layers[0].weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
