import unittest

import numpy as np

from metrics import bin_piecewise, bin_snps, loss, seed_for
from search import propose


class MetricTests(unittest.TestCase):
    def test_exact_interval_integral(self):
        # Hand integral: first window has 1bp at 2, 2bp at 5; next has 2bp at 5.
        np.testing.assert_allclose(bin_piecewise([0, 1, 5], [2, 5], [0, 3, 5]), [4, 5])

    def test_snp_projection_uses_physical_length(self):
        np.testing.assert_allclose(bin_snps([0, 2, 10], [1, 3, 7], [0, 5, 10]), [2.6, 6.2])

    def test_missing_coverage_rejected(self):
        with self.assertRaises(ValueError):
            bin_snps([1, 10], [2, 5], [0, 10])

    def test_wrong_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            bin_piecewise([0, 3, 2], [1, 2], [0, 2])

    def test_absolute_scale_not_optimized_away(self):
        target = np.array([10., 20., 50., 20.])
        full = np.ones(4, bool)
        self.assertAlmostEqual(loss(target*2, target, full, full), np.log(2))
        self.assertEqual(loss(target, target, full, full), 0)

    def test_holdout_values_do_not_affect_training_score(self):
        y = np.array([10., 10., 20., 20.]); p = y.copy()
        train = np.array([True, True, False, False])
        full = np.ones(4, bool)
        p[~train] *= 10
        self.assertEqual(loss(p, y, train, full), 0)
        self.assertGreater(loss(p, y, ~train, full), 2)

    def test_invalid_predictions_are_not_dropped(self):
        full = np.ones(3, bool)
        with self.assertRaises(ValueError):
            loss([10, np.nan, 5], [10, 5, 5], full, full)

    def test_seed_domains_separate_replicates(self):
        train = {seed_for('train', 12, 'GRK2', i) for i in range(30)}
        validation = {seed_for('validation', 12, 'GRK2', i) for i in range(30)}
        self.assertFalse(train & validation)
        self.assertEqual(len(train), 30)

    def test_proposals_stay_in_simulator_domain(self):
        rng = np.random.default_rng(17)
        best = propose(2, rng, {}, 15000)
        for i in range(1000):
            c = propose(i, rng, best, 15000)
            self.assertTrue(3000 <= c['Ne'] <= 40000)
            if c['model'] != 'neutral':
                self.assertTrue(.001 <= c['s'] <= .3)
                self.assertTrue(0 <= c['age'] <= 5000)
                self.assertTrue(abs(c['offset']) <= 150000)
                self.assertLess(c['start_frequency'], c['end_frequency'])


if __name__ == '__main__':
    unittest.main()
