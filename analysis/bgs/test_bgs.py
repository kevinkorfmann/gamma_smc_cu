"""Boundary and missing-coverage checks for genomic interval annotation."""
import unittest
import numpy as np
from bgs import summarize_interval, merged_intervals


class IntervalTests(unittest.TestCase):
    def setUp(self):
        self.starts = np.array([0, 1000, 3000])
        self.ends = self.starts + 1000
        self.b = np.array([.5, 1., .75])

    def test_half_open_and_length_weighted(self):
        result = summarize_interval(self.starts, self.ends, self.b, 500, 2000)
        self.assertAlmostEqual(result["mean"], 5 / 6)
        self.assertEqual(result["coverage"], 1)
        self.assertAlmostEqual(result["fraction_lt_0.8"], 1 / 3)
        result = summarize_interval(self.starts, self.ends, self.b, 1000, 1001)
        self.assertEqual(result["mean"], 1)

    def test_missing_does_not_become_weak_bgs(self):
        result = summarize_interval(self.starts, self.ends, self.b, 1500, 3500)
        self.assertEqual(result["coverage"], .5)
        self.assertEqual(result["mean"], .875)
        result = summarize_interval(self.starts, self.ends, self.b, 2000, 3000)
        self.assertEqual(result["coverage"], 0)
        self.assertTrue(np.isnan(result["mean"]))

    def test_mask_preserves_gaps_and_threshold_direction(self):
        self.assertEqual(list(merged_intervals(self.starts, self.ends, self.b >= .75)), [(1000, 2000), (3000, 4000)])
        self.assertEqual(list(merged_intervals(self.starts, self.ends, self.b < .75)), [(0, 1000)])
        self.assertEqual(list(merged_intervals(self.starts, self.ends, np.ones(3, bool))), [(0, 2000), (3000, 4000)])


if __name__ == "__main__":
    unittest.main()
