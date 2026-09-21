import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_spacetime_gates import check_alignment, check_history_offsets


class SpaceTimeGateCheckTest(unittest.TestCase):
    def write_summary(self, **updates) -> Path:
        summary = {
            "plateau_reached": True,
            "best_validation_cosine": 0.73,
            "final_cosine_ema": 0.71,
        }
        summary.update(updates)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "alignment_summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def test_alignment_accepts_finite_plateau_without_absolute_threshold(self):
        check_alignment([self.write_summary()])

    def test_alignment_accepts_threshold_without_plateau(self):
        check_alignment(
            [self.write_summary(plateau_reached=False, best_validation_cosine=0.9)]
        )

    def test_alignment_rejects_epoch_limit_below_threshold_without_plateau(self):
        with self.assertRaises(SystemExit):
            check_alignment([self.write_summary(plateau_reached=False)])

    def test_alignment_rejects_non_finite_cosine(self):
        with self.assertRaises(SystemExit):
            check_alignment([self.write_summary(best_validation_cosine=float("nan"))])

    def test_checkpoint_history_offsets_match_t120(self):
        expected = [0, 17, 34, 51, 69, 86, 103, 120]
        check_history_offsets(expected, expected, Path("deploy.pt"))

        with self.assertRaisesRegex(SystemExit, "history offsets mismatch"):
            check_history_offsets(
                [0, 9, 17, 26, 34, 43, 51, 60],
                expected,
                Path("deploy.pt"),
            )


if __name__ == "__main__":
    unittest.main()
