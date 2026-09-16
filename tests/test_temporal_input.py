import unittest

import torch

from world_critic.config import ModelConfig
from world_critic.data import _make_temporal_mosaic
from world_critic.model import (
    CausalPatchTemporalAdapter,
    MosaicTemporalResidual,
    SpaceTimePerceiverEncoder,
    SparseCrossFrameEncoder,
    SparseTemporalMemoryEncoder,
)


class TemporalInputTest(unittest.TestCase):
    def spacetime_encoder(self, temporal_enabled: bool = True):
        return SpaceTimePerceiverEncoder(
            ModelConfig(
                latent_dim=16,
                max_views=2,
                spacetime_frame_count=4,
                spacetime_layers=2,
                spacetime_heads=4,
                perceiver_queries=64,
                perceiver_layers=1,
                perceiver_mlp_ratio=2.0,
                spacetime_temporal_enabled=temporal_enabled,
            )
        )

    def test_spacetime_perceiver_emits_fixed_visual_tokens(self):
        encoder = self.spacetime_encoder().eval()
        frame_tokens = torch.randn(2, 4, 2, 9, 16)

        visual_tokens, pooled = encoder(frame_tokens)

        self.assertEqual(tuple(visual_tokens.shape), (2, 64, 16))
        self.assertEqual(tuple(pooled.shape), (2, 1, 16))

    def test_spacetime_perceiver_reads_oldest_history(self):
        encoder = self.spacetime_encoder().eval()
        frame_tokens = torch.randn(2, 4, 2, 9, 16, requires_grad=True)

        _, pooled = encoder(frame_tokens)
        pooled.square().mean().backward()

        self.assertGreater(float(frame_tokens.grad[:, -1].abs().sum()), 0.0)

    def test_spacetime_gate_is_bounded(self):
        encoder = self.spacetime_encoder()
        teacher = torch.randn(2, 1, 16)
        student = torch.randn(2, 1, 16)
        self.assertTrue(torch.equal(encoder.blend(teacher, student), teacher))
        encoder.set_gate(1.0)
        self.assertEqual(float(encoder.blend_gate), 1.0)
        self.assertTrue(torch.equal(encoder.blend(teacher, student), student))
        with self.assertRaisesRegex(ValueError, "must be in"):
            encoder.set_gate(1.1)

    def mosaic_residual(self):
        return MosaicTemporalResidual(
            ModelConfig(
                latent_dim=16,
                trunk_heads=4,
                dropout=0.0,
                mosaic_temporal_heads=4,
                mosaic_temporal_mlp_ratio=2.0,
            )
        )

    def test_mosaic_temporal_residual_warm_start_is_identity(self):
        residual = self.mosaic_residual().eval()
        anchor = torch.randn(2, 2, 16)
        tokens = torch.randn(2, 2, 17, 16)

        output = residual(anchor, tokens)

        self.assertTrue(torch.equal(output, anchor))

    def test_mosaic_temporal_residual_reads_historical_quadrants(self):
        residual = self.mosaic_residual().eval()
        residual.residual_gate.data.fill_(1.0)
        anchor = torch.randn(2, 2, 16)
        tokens = torch.randn(2, 2, 17, 16, requires_grad=True)

        residual(anchor, tokens).square().mean().backward()

        # In a 4x4 patch grid, the bottom-right quadrant is the oldest frame.
        patch_grad = tokens.grad[:, :, 1:].view(2, 2, 4, 4, 16)
        self.assertGreater(float(patch_grad[:, :, 2:, 2:].abs().sum()), 0.0)

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

    def sparse_memory_encoder(self, layers: int = 2):
        config = ModelConfig(
            latent_dim=16,
            trunk_heads=4,
            sparse_memory_frame_count=4,
            sparse_memory_tokens=3,
            sparse_memory_global_layers=layers,
            sparse_memory_heads=4,
            sparse_memory_mlp_ratio=2.0,
            sparse_memory_layerscale_init=1.0e-3,
        )
        return SparseTemporalMemoryEncoder(hidden_dim=16, config=config)

    def test_sparse_memory_is_compressed_immediately_and_persists(self):
        encoder = self.sparse_memory_encoder()
        tokens = torch.randn(2, 2, 4, 5, 16)

        chronological, memory = encoder.initialize(tokens)
        memory = encoder.apply_global(memory, 0)

        self.assertEqual(tuple(chronological.shape), (2, 2, 4, 5, 16))
        self.assertEqual(tuple(memory.shape), (2, 2, 4, 3, 16))
        self.assertTrue(torch.allclose(chronological[:, :, -1], tokens[:, :, 0]))

    def test_sparse_memory_blocks_future_observations(self):
        encoder = self.sparse_memory_encoder(layers=1).eval()
        tokens = torch.randn(2, 2, 4, 5, 16)
        changed = tokens.clone()
        changed[:, :, 0] += 100.0

        _, memory = encoder.initialize(tokens)
        _, changed_memory = encoder.initialize(changed)
        oldest = encoder.apply_global(memory, 0)[:, :, 0]
        changed_oldest = encoder.apply_global(changed_memory, 0)[:, :, 0]

        self.assertTrue(torch.allclose(oldest, changed_oldest, atol=1e-5, rtol=1e-5))

    def test_sparse_memory_carries_history_to_current_readout(self):
        encoder = self.sparse_memory_encoder(layers=2).eval()
        tokens = torch.randn(2, 2, 4, 5, 16, requires_grad=True)
        chronological, memory = encoder.initialize(tokens)
        for index in range(2):
            memory = encoder.apply_global(memory, index)
        output = encoder.readout(chronological[:, :, -1, 0], memory[:, :, -1])
        output.square().mean().backward()

        historical_grad = tokens.grad[:, :, 1:].abs().sum()
        self.assertGreater(float(historical_grad), 0.0)
        self.assertIsNotNone(encoder.memory_queries.grad)


if __name__ == "__main__":
    unittest.main()
