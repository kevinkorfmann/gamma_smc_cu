"""Independent statistical checks; these do not validate human selection models."""
import math
import unittest

import numpy as np

from abc_reference import RejectionABC, heldout_diagnostics, validate_groups


class TestABC(unittest.TestCase):
    def test_normal_model_support_matches_analytic_answer(self):
        rng = np.random.default_rng(117)
        # Unequal bank counts must not change the declared model prior.
        X = np.concatenate([rng.normal(-1, 1, 60000), rng.normal(1, 1, 30000)])[:, None]
        labels = np.array(["a"] * 60000 + ["b"] * 30000)
        abc = RejectionABC(X, labels, {"a": .3, "b": .7})
        for observed in [-.5, 0, .5]:
            result = abc.infer([observed], fraction=.03)
            pa = .3 * math.exp(-.5 * (observed+1)**2)
            pb = .7 * math.exp(-.5 * (observed-1)**2)
            self.assertAlmostEqual(result["model_support"][0], pa / (pa+pb), delta=.025)

    def test_indistinguishable_models_retain_prior_and_ties(self):
        labels = np.array(["a"] * 100 + ["b"] * 10)
        abc = RejectionABC(np.ones((110, 2)), labels, {"a": .2, "b": .8})
        result = abc.infer([1, 1], fraction=.01)
        np.testing.assert_allclose(result["model_support"], [.2, .8])
        self.assertEqual(len(result["indices"]), 110)

    def test_correlated_summaries_are_numerically_valid(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=1000)
        abc = RejectionABC(np.c_[x, x, np.ones(1000)], np.array(["a"]*500+["b"]*500), {"a":.5,"b":.5})
        result = abc.infer([0, 0, 1])
        self.assertTrue(np.isfinite(result["epsilon"]))
        self.assertAlmostEqual(result["model_support"].sum(), 1)

    def test_chance_level_for_identical_model_summaries(self):
        labels = np.array(["a"]*100+["b"]*100)
        abc = RejectionABC(np.zeros((200, 1)), labels, {"a": .5, "b": .5})
        d = heldout_diagnostics(abc, np.zeros((40, 1)), np.array(["a"]*20+["b"]*20))
        self.assertAlmostEqual(d["prior_weighted_accuracy"], .5)
        self.assertAlmostEqual(d["multiclass_brier"], .5)

    def test_ancestry_leakage_or_pseudoreplication_rejected(self):
        with self.assertRaises(ValueError):
            validate_groups(["x", "y"], ["y", "z"])
        with self.assertRaises(ValueError):
            validate_groups(["x", "x"], ["y", "z"])
        validate_groups(["x", "y"], ["z", "w"])

    def test_invalid_inputs_rejected(self):
        X, labels = np.arange(20).reshape(10,2), np.array(["a"]*5+["b"]*5)
        with self.assertRaises(ValueError):
            RejectionABC(X, labels, {"a": .5, "b": .8})
        abc = RejectionABC(X, labels, {"a": .5, "b": .5})
        for observation in [[np.nan, 0], [1, 2, 3]]:
            with self.assertRaises(ValueError):
                abc.infer(observation)
        with self.assertRaises(ValueError):
            abc.infer([1,2], fraction=0)


if __name__ == "__main__":
    unittest.main()
