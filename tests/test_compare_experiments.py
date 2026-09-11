import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.compare_experiments import curve_path, episode_statistics, load, metrics, run


class CompareExperimentsTest(unittest.TestCase):
    def test_curve_statistics_recover_mse_pearson_and_bias(self):
        rows = {
            (1, 0): (0.2, 0.0),
            (1, 1): (0.7, 0.5),
            (2, 0): (1.2, 1.0),
        }
        stats = episode_statistics(rows, [1, 2])
        mse, pearson, bias = metrics(stats.sum(axis=0))

        self.assertAlmostEqual(float(mse), 0.04)
        self.assertAlmostEqual(float(pearson), 1.0)
        self.assertAlmostEqual(float(bias), 0.2)

    def test_load_rejects_duplicate_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_curves.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["episode_id", "frame_index", "value", "return"])
                writer.writerow([1, 2, 0.1, 0.0])
                writer.writerow([1, 2, 0.2, 0.0])

            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load(path)

    def test_metric_json_path_resolves_sibling_curve_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            curve = root / "episode_curves.csv"
            curve.touch()
            resolved = curve_path(str(root / "episode_metrics.json"))

            self.assertEqual(resolved, curve)

    def test_end_to_end_comparison_reports_mse_pearson_and_bias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline" / "episode_curves"
            candidate = root / "candidate" / "episode_curves"
            baseline.mkdir(parents=True)
            candidate.mkdir(parents=True)
            header = ["episode_id", "frame_index", "value", "return"]
            baseline_rows = [[1, 0, 0.0, 0.0], [1, 1, 0.5, 0.5], [2, 0, 1.0, 1.0]]
            candidate_rows = [[1, 0, 0.1, 0.0], [1, 1, 0.6, 0.5], [2, 0, 1.1, 1.0]]
            for path, rows in (
                (baseline / "episode_curves.csv", baseline_rows),
                (candidate / "episode_curves.csv", candidate_rows),
            ):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(header)
                    writer.writerows(rows)

            stdout = io.StringIO()
            argv = [
                "compare_experiments.py",
                str(root / "baseline"),
                str(root / "candidate"),
                "--bootstrap-samples",
                "100",
            ]
            with patch("sys.argv", argv), redirect_stdout(stdout):
                run()
            result = json.loads(stdout.getvalue())

            self.assertAlmostEqual(result["candidate_mse"], 0.01)
            self.assertAlmostEqual(result["candidate_pearson"], 1.0)
            self.assertAlmostEqual(result["candidate_mean_bias"], 0.1)


if __name__ == "__main__":
    unittest.main()
