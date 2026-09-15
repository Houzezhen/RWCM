import unittest

import torch

from world_critic.config import ModelConfig
from world_critic.data import _make_temporal_mosaic
from world_critic.model import CausalPatchTemporalAdapter, SparseCrossFrameEncoder


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

    def test_causal_patch_temporal_adapter_warm_start_is_identity(self):
        adapter = CausalPatchTemporalAdapter(
            hidden_dim=16,
            frame_count=4,
            heads=4,
            dropout=0.0,
            mlp_ratio=2.0,
        )
        tokens = torch.randn(2, 2, 4, 5, 16)
        output = adapter(tokens)

        self.assertEqual(tuple(output.shape), tuple(tokens.shape))
        # Every zero-gated global block preserves all frame-wise ViT tokens.
        self.assertTrue(torch.allclose(output, tokens, atol=1e-6))

    def test_causal_patch_temporal_adapter_propagates_gradient(self):
        adapter = CausalPatchTemporalAdapter(
            hidden_dim=16,
            frame_count=4,
            heads=4,
            dropout=0.0,
            mlp_ratio=2.0,
        )
        for layer in adapter.layers:
            layer.residual_gate.data.fill_(1.0)
        tokens = torch.randn(2, 2, 4, 5, 16, requires_grad=True)
        adapter(tokens).square().mean().backward()

        self.assertIsNotNone(adapter.time_embedding.grad)
        self.assertGreater(float(adapter.time_embedding.grad.abs().sum()), 0.0)

    def test_causal_patch_temporal_adapter_reads_historical_frames(self):
        adapter = CausalPatchTemporalAdapter(
            hidden_dim=16,
            frame_count=4,
            heads=4,
            dropout=0.0,
            mlp_ratio=2.0,
        )
        for layer in adapter.layers:
            layer.residual_gate.data.fill_(1.0)
        tokens = torch.randn(2, 2, 4, 5, 16, requires_grad=True)
        output = adapter(tokens)[:, :, 0].square().mean()
        output.backward()

        # The current-frame query attends to all current/older frames.  A
        # nonzero gradient on an older frame guards against accidental
        # current-only or same-index temporal fusion.
        historical_grad = tokens.grad[:, :, 1:].abs().sum()
        self.assertGreater(float(historical_grad), 0.0)

    def test_causal_patch_temporal_adapter_blocks_future_frames(self):
        adapter = CausalPatchTemporalAdapter(
            hidden_dim=16,
            frame_count=4,
            heads=4,
            dropout=0.0,
            mlp_ratio=2.0,
            layers=2,
        )
        for layer in adapter.layers:
            layer.residual_gate.data.fill_(1.0)
        adapter.eval()
        tokens = torch.randn(2, 2, 4, 5, 16)
        changed = tokens.clone()
        changed[:, :, 0] += 100.0  # current frame is future to every history frame

        oldest = adapter(tokens)[:, :, -1]
        changed_oldest = adapter(changed)[:, :, -1]

        self.assertTrue(torch.allclose(oldest, changed_oldest, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
