"""Unit tests for tmrca_cu.infer() top-level API."""

import numpy as np
import pytest
import msprime

import tmrca_cu


@pytest.fixture(scope="module")
def ts():
    ts = msprime.sim_ancestry(
        samples=10, sequence_length=100_000,
        recombination_rate=1e-8, population_size=10000, random_seed=42)
    return msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)


@pytest.fixture(scope="module")
def genotype_data(ts):
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()], dtype=np.float64)
    return G, pos


class TestInferTreeSequence:
    def test_returns_dict(self, ts):
        result = tmrca_cu.infer(ts)
        assert isinstance(result, dict)
        assert "mean" in result

    def test_shape_all_pairs(self, ts):
        result = tmrca_cu.infer(ts)
        n = ts.num_samples
        S = ts.num_sites
        n_pairs = n * (n - 1) // 2
        assert result["mean"].shape == (S, n_pairs)

    def test_shape_specific_pairs(self, ts):
        pairs = [(0, 1), (2, 3)]
        result = tmrca_cu.infer(ts, pairs=pairs)
        assert result["mean"].shape[1] == 2

    def test_returns_pairs(self, ts):
        pairs = [(0, 1), (5, 3)]
        result = tmrca_cu.infer(ts, pairs=pairs)
        assert result["pairs"] == pairs

    def test_returns_positions(self, ts):
        result = tmrca_cu.infer(ts, pairs=[(0, 1)])
        assert len(result["positions"]) == ts.num_sites

    def test_values_positive(self, ts):
        result = tmrca_cu.infer(ts, pairs=[(0, 1)])
        assert np.all(result["mean"] > 0)

    def test_with_ci(self, ts):
        result = tmrca_cu.infer(ts, pairs=[(0, 1)], mean_only=False)
        assert "lower" in result
        assert "upper" in result
        assert np.all(result["lower"] <= result["mean"])
        assert np.all(result["upper"] >= result["mean"])


class TestInferGenotypeMatrix:
    def test_returns_dict(self, genotype_data):
        G, pos = genotype_data
        result = tmrca_cu.infer(G, pos)
        assert isinstance(result, dict)
        assert "mean" in result

    def test_shape_all_pairs(self, genotype_data):
        G, pos = genotype_data
        n = G.shape[0]
        S = G.shape[1]
        n_pairs = n * (n - 1) // 2
        result = tmrca_cu.infer(G, pos)
        assert result["mean"].shape == (S, n_pairs)

    def test_specific_pairs(self, genotype_data):
        G, pos = genotype_data
        result = tmrca_cu.infer(G, pos, pairs=[(0, 1)])
        assert result["mean"].shape[1] == 1

    def test_custom_params(self, genotype_data):
        G, pos = genotype_data
        result = tmrca_cu.infer(G, pos, mu=1e-8, rho=1e-8, Ne=5000,
                                 pairs=[(0, 1)])
        assert result["mean"].shape[1] == 1
        assert np.all(result["mean"] > 0)


class TestInferConsistency:
    def test_ts_and_matrix_agree(self, ts, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3), (4, 5)]
        r_ts = tmrca_cu.infer(ts, pairs=pairs)
        r_mat = tmrca_cu.infer(G, pos, pairs=pairs)
        np.testing.assert_allclose(r_ts["mean"], r_mat["mean"],
                                    rtol=1e-5, atol=1e-6)

    def test_correlates_with_truth(self, ts):
        pair = (0, 1)
        result = tmrca_cu.infer(ts, pairs=[pair])
        est = result["mean"][:, 0]
        pos = result["positions"]

        # Extract truth
        truth = np.empty(len(pos))
        tree_iter = ts.trees(); tree = next(tree_iter)
        for idx, p in enumerate(pos):
            while p >= tree.interval.right:
                tree = next(tree_iter)
            truth[idx] = tree.tmrca(*pair)

        from scipy.stats import pearsonr
        r = pearsonr(np.log(truth), np.log(np.maximum(est, 1)))[0]
        assert r > 0.5, f"Correlation too low: {r}"


class TestInferEdgeCases:
    def test_single_pair(self, ts):
        result = tmrca_cu.infer(ts, pairs=[(0, 1)])
        assert result["mean"].shape[1] == 1

    def test_empty_pairs(self, ts):
        result = tmrca_cu.infer(ts, pairs=[])
        assert result["mean"].shape[1] == 0
