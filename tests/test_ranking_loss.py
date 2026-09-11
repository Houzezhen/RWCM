import unittest

import torch

from world_critic.config import DataConfig, TrainConfig, validate_train_config
from world_critic.training import pairwise_ranking_sum


class PairwiseRankingLossTest(unittest.TestCase):
    def test_correct_order_has_lower_loss(self):
        target = torch.tensor([-1.0, -0.5, 0.0])
        correct, count = pairwise_ranking_sum(
            target.clone(), target, temperature=0.1, min_target_gap=0.1
        )
        reversed_order, reversed_count = pairwise_ranking_sum(
            -target, target, temperature=0.1, min_target_gap=0.1
        )

        self.assertEqual(count.item(), 3)
        self.assertEqual(reversed_count.item(), 3)
        self.assertLess(correct.item(), reversed_order.item())

    def test_small_target_gaps_and_ties_are_excluded(self):
        prediction = torch.tensor([0.0, 0.2, 0.4], requires_grad=True)
        target = torch.tensor([0.0, 0.05, 0.05])
        loss_sum, count = pairwise_ranking_sum(
            prediction, target, temperature=0.1, min_target_gap=0.1
        )

        self.assertEqual(count.item(), 0)
        loss_sum.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_config_rejects_invalid_ranking_parameters(self):
        config = TrainConfig(data=DataConfig(repo_id="fixture"))
        config.loss.ranking_weight = 0.1
        config.loss.ranking_temperature = 0.0
        with self.assertRaisesRegex(ValueError, "ranking_temperature"):
            validate_train_config(config)

        config.loss.ranking_temperature = 0.1
        config.loss.ranking_min_target_gap = -0.01
        with self.assertRaisesRegex(ValueError, "ranking_min_target_gap"):
            validate_train_config(config)


if __name__ == "__main__":
    unittest.main()
