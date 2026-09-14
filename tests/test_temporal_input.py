import unittest

import torch

from world_critic.config import ModelConfig
from world_critic.data import _make_temporal_mosaic
from world_critic.model import SparseCrossFrameEncoder


class TemporalInputTest(unittest.TestCase):
    def test_mosaic_layout_is_current_then_older_frames(self):
        frames = [
            torch.full((2, 3, 1), value, dtype=torch.uint8)
            for value in range(4)
        ]
        mosaic = _make_temporal_mosaic(frames)

        self.assertEqual(tuple(mosaic.shape), (4, 6, 1))
        self.assertEqual(int(mosaic[0, 0, 0]), 0)
        self.assertEqual(int(mosaic[0, 3, 0]), 1)
        self.assertEqual(int(mosaic[2, 0, 0]), 2)
        self.assertEqual(int(mosaic[2, 3, 0]), 3)

    def test_mosaic_requires_four_frames(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            _make_temporal_mosaic([torch.zeros(2, 3, 1)] * 3)

    def test_sparse_cross_frame_encoder_shape_and_gradient(self):
        config = ModelConfig(
            latent_dim=16,
            trunk_heads=4,
            trunk_mlp_ratio=2.0,
            cross_frame_count=4,
            cross_frame_layers=2,
        )
        encoder = SparseCrossFrameEncoder(config)
        tokens = torch.randn(2, 4, 2, 5, 16, requires_grad=True)

        output = encoder(tokens)
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (2, 2, 16))
        self.assertIsNotNone(encoder.time_embedding.grad)
        self.assertGreater(float(encoder.time_embedding.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
