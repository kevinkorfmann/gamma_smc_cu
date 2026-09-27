"""Tests protect physical weighting, no extrapolation and actual tree truth."""
import unittest
import gzip
import tempfile
from pathlib import Path

import msprime
import numpy as np

from common import interval_means, seed_for, snp_cells, truth_on_grid
from infer import asmc_input


class TestCoordinateScoring(unittest.TestCase):
    def test_breakpoint_weighting_and_missing_coverage(self):
        y, cov = interval_means([0, 15], [5, 20], [2, 10], [0, 10, 20])
        np.testing.assert_allclose(y[:, 0], [2, 10])
        np.testing.assert_allclose(cov, [.5, .5])
        y, cov = interval_means([0, 5], [5, 20], [2, 10], [0, 10, 20])
        np.testing.assert_allclose(y[:, 0], [6, 10])
        np.testing.assert_allclose(cov, 1)

    def test_snp_projection_does_not_fill_edges(self):
        left, right = snp_cells([2, 8, 18])
        np.testing.assert_allclose(left, [2, 5, 13])
        np.testing.assert_allclose(right, [5, 13, 18])
        _, coverage = interval_means(left, right, [1, 1, 1], [0, 10, 20])
        np.testing.assert_allclose(coverage, [.8, .8])

    def test_truth_matches_direct_base_enumeration(self):
        ts = msprime.sim_ancestry(3, sequence_length=40, population_size=20,
                                 recombination_rate=.02, random_seed=311)
        pairs = [(0, 1), (2, 5)]
        actual = truth_on_grid(ts, pairs, np.arange(0, 41, 10))
        direct = np.array([[ts.at(x+.5).tmrca(*pair) for pair in pairs] for x in range(40)])
        np.testing.assert_allclose(actual, direct.reshape(4, 10, 2).mean(axis=1))

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            interval_means([0, 4], [5, 10], [1, 2], [0, 10])

    def test_independent_seed_namespaces(self):
        self.assertEqual(seed_for("x", 1), seed_for("x", 1))
        self.assertNotEqual(seed_for("x", 1), seed_for("y", 1))

    def test_asmc_genetic_map_units_and_physical_offset(self):
        # Breakpoint at bp 5, rate tripled afterwards. Genetic positions must
        # integrate rates, not interpolate rates or scale the one-based bp.
        rmap = msprime.RateMap(position=[0, 5, 10], rate=[1e-8, 3e-8])
        with tempfile.TemporaryDirectory() as tmp:
            root = asmc_input(Path(tmp), np.array([[0,1,0,1],[1,0,1,0]], dtype=np.uint8),
                              np.array([0,4,5,9]), rmap)
            with gzip.open(root+".map.gz", "rt") as f:
                rows = [line.strip().split("\t") for line in f]
            self.assertTrue(all(len(row)==4 for row in rows))
            self.assertEqual([int(row[3]) for row in rows], [1,5,6,10])
            np.testing.assert_allclose([float(row[2]) for row in rows], [0,4e-6,5e-6,17e-6])


if __name__ == "__main__":
    unittest.main()
