import unittest

import torch

from world_critic.data import _make_temporal_mosaic


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


if __name__ == "__main__":
    unittest.main()
