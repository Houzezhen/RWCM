import csv
import json
import tempfile
import unittest
from pathlib import Path

from world_critic.curves import build_episode_curves, write_episode_curve_artifacts


class EpisodeArtifactTest(unittest.TestCase):
    def test_plot_limit_does_not_truncate_metrics_or_csv(self):
        curves = build_episode_curves(
            [
                {"episode_id": 1, "frame_index": 0, "value": 0.1, "return": 0.0},
                {"episode_id": 2, "frame_index": 0, "value": 0.2, "return": 0.0},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = write_episode_curve_artifacts(
                curves,
                directory,
                render_plots=False,
                max_plot_episodes=1,
            )
            metrics = json.loads(Path(summary["episode_metrics"]).read_text(encoding="utf-8"))
            with Path(summary["csv"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(len(metrics), 2)
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["num_episodes"], 2)


if __name__ == "__main__":
    unittest.main()
