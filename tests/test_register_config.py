import unittest

from world_critic.config import DataConfig, TrainConfig, validate_train_config
from world_critic.training import config_from_checkpoint_payload


class RegisterConfigValidationTest(unittest.TestCase):
    def config(self) -> TrainConfig:
        return TrainConfig(data=DataConfig(repo_id="fixture"))

    def test_standard_vision_register_config_is_valid(self):
        config = self.config()
        config.model.vision.num_register_tokens = 4
        config.model.vision.register_insert = "early"
        validate_train_config(config)

    def test_noncausal_block_config_requires_explicit_ablation_flag(self):
        config = self.config()
        config.model.use_block_register_fusion = True
        config.model.num_temporal_registers = 4
        config.model.register_block_size = 2
        with self.assertRaisesRegex(ValueError, "not causal"):
            validate_train_config(config)

        config.model.allow_noncausal_register_ablation = True
        validate_train_config(config)

    def test_old_noncausal_checkpoint_is_migrated_for_evaluation(self):
        config = self.config()
        config.model.use_block_register_fusion = True
        config.model.num_temporal_registers = 4
        config.model.register_block_size = 2
        payload = {"config": {
            "data": config.data.__dict__,
            "model": {
                **config.model.__dict__,
                "vision": config.model.vision.__dict__,
                "language": config.model.language.__dict__,
            },
        }}
        payload["config"]["model"].pop("allow_noncausal_register_ablation")

        restored = config_from_checkpoint_payload(payload)

        self.assertTrue(restored.model.allow_noncausal_register_ablation)
        validate_train_config(restored)

    def test_sparse_mosaic_history_requires_one_endpoint_window(self):
        config = self.config()
        config.data.history_size = 1
        config.data.history_offsets = [0, 20, 40, 60]
        config.data.history_mosaic = True
        config.model.use_proprioception = True
        config.model.proprioception_dim = 28
        validate_train_config(config)

        config.data.history_size = 3
        with self.assertRaisesRegex(ValueError, "history_size must be 1"):
            validate_train_config(config)

    def test_sparse_history_supports_state_only_mode(self):
        config = self.config()
        config.data.history_size = 1
        config.data.history_offsets = [0, 20, 40, 60]
        config.model.use_proprioception = True
        config.model.proprioception_dim = 28
        validate_train_config(config)

    def test_cross_frame_tokens_require_explicit_sparse_frames(self):
        config = self.config()
        config.data.history_size = 1
        config.data.history_offsets = [0, 20, 40, 60]
        config.model.use_cross_frame_tokens = True
        with self.assertRaisesRegex(ValueError, "history_frames=true"):
            validate_train_config(config)

        config.data.history_frames = True
        validate_train_config(config)

    def test_temporal_transformer_requires_sparse_frames_and_is_exclusive(self):
        config = self.config()
        config.data.history_size = 1
        config.data.history_offsets = [0, 20, 40, 60]
        config.model.use_temporal_transformer = True
        with self.assertRaisesRegex(ValueError, "history_frames=true"):
            validate_train_config(config)

        config.data.history_frames = True
        validate_train_config(config)

        config.model.use_cross_frame_tokens = True
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_train_config(config)

    def test_sparse_temporal_memory_requires_frames_and_is_exclusive(self):
        config = self.config()
        config.data.history_size = 1
        config.data.history_offsets = [0, 20, 40, 60]
        config.model.use_sparse_temporal_memory = True
        with self.assertRaisesRegex(ValueError, "history_frames=true"):
            validate_train_config(config)

        config.data.history_frames = True
        validate_train_config(config)

        config.model.use_temporal_transformer = True
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_train_config(config)

    def test_mosaic_temporal_residual_requires_four_frame_mosaic(self):
        config = self.config()
        config.data.history_size = 1
        config.data.history_offsets = [0, 20, 40, 60]
        config.model.use_mosaic_temporal_residual = True
        with self.assertRaisesRegex(ValueError, "history_mosaic=true"):
            validate_train_config(config)

        config.data.history_mosaic = True
        validate_train_config(config)

        config.model.use_cross_frame_tokens = True
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_train_config(config)

    def test_spacetime_teacher_allows_mosaic_and_raw_history(self):
        config = self.config()
        config.data.history_size = 1
        config.data.history_offsets = [0, 20, 40, 60]
        config.data.history_mosaic = True
        config.data.history_frames = True
        config.data.success_key = "episode_success"
        config.model.vision.model_name = "google/siglip-base-patch16-224"
        config.model.use_spacetime_perceiver = True
        config.model.spacetime_teacher_enabled = True
        config.teacher_checkpoint = "teacher.pt"
        config.model.predict_risk = True
        validate_train_config(config)

        config.model.spacetime_teacher_enabled = False
        with self.assertRaisesRegex(ValueError, "history_mosaic must be false"):
            validate_train_config(config)


if __name__ == "__main__":
    unittest.main()
