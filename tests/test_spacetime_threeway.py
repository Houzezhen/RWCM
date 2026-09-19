import unittest

from scripts.summarize_spacetime_threeway import (
    baseline_gate,
    choose_branch,
    mosaic_gate,
)


def comparison(
    *,
    mse=-0.1,
    mse_ci=(-0.2, -0.01),
    centered=-0.08,
    centered_ci=(-0.16, -0.005),
    pearson=0.1,
    pearson_ci=(0.01, 0.2),
):
    return {
        "mse_difference_candidate_minus_baseline": mse,
        "mse_paired_episode_bootstrap_ci95": list(mse_ci),
        "centered_mse_difference_candidate_minus_baseline": centered,
        "centered_mse_paired_episode_bootstrap_ci95": list(centered_ci),
        "pearson_difference_candidate_minus_baseline": pearson,
        "pearson_paired_episode_bootstrap_ci95": list(pearson_ci),
    }


class SpaceTimeThreewayTest(unittest.TestCase):
    def test_both_preregistered_gates_pass_for_two_seed_improvement(self):
        items = [comparison(), comparison(mse_ci=(-0.15, 0.02), pearson_ci=(-0.01, 0.2))]

        self.assertTrue(baseline_gate(items)["passed"])
        self.assertTrue(mosaic_gate(items)["passed"])

    def test_baseline_gate_rejects_bias_only_mse_gain(self):
        items = [
            comparison(centered=0.01, centered_ci=(-0.02, 0.03)),
            comparison(centered=0.02, centered_ci=(-0.01, 0.04)),
        ]

        self.assertFalse(baseline_gate(items)["passed"])

    def test_mosaic_gate_requires_strong_pearson_when_centered_mse_does_not_improve(self):
        items = [
            comparison(centered=0.01, centered_ci=(-0.02, 0.03), pearson_ci=(-0.01, 0.2)),
            comparison(centered=0.02, centered_ci=(-0.01, 0.04), pearson_ci=(-0.02, 0.2)),
        ]

        self.assertFalse(mosaic_gate(items)["passed"])

    def test_conflicting_seed_directions_choose_branch_d(self):
        baseline_items = [comparison(), comparison(mse=0.1)]
        mosaic_items = [comparison(), comparison()]

        decision = choose_branch(
            {"passed": False},
            {"passed": True},
            baseline_items,
            mosaic_items,
        )

        self.assertEqual(decision["branch"], "D")

    def test_conflicting_metrics_choose_branch_d(self):
        baseline_items = [
            comparison(mse=-0.1, centered=0.02),
            comparison(mse=-0.2, centered=0.01),
        ]
        mosaic_items = [comparison(), comparison()]

        decision = choose_branch(
            {"passed": False},
            {"passed": True},
            baseline_items,
            mosaic_items,
        )

        self.assertEqual(decision["branch"], "D")

    def test_branch_b_uses_original_wcm_as_the_main_narrative(self):
        items = [comparison(), comparison()]

        decision = choose_branch(
            {"passed": True},
            {"passed": False},
            items,
            items,
        )

        self.assertEqual(decision["branch"], "B")
        self.assertIn("原始 WCM → SpaceTime", decision["conclusion"])
        self.assertIn("实验强基线", decision["conclusion"])


if __name__ == "__main__":
    unittest.main()
