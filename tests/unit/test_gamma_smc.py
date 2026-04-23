import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gamma_smc_cu
from tests.reference.gamma_smc_numpy import gamma_smc_forward as gamma_smc_numpy


class TestGammaSMCForward:
    """
    Verify GPU Gamma-SMC forward filtering against pure-numpy reference.

    Output arrays are site-major [out_S, n_pairs]. Use .T for [n_pairs, out_S].
    """

    def test_matches_numpy_reference(self, small_simulation, uniform_mu, uniform_rho):
        """GPU output matches numpy reference for a single pair."""
        _, G, positions = small_simulation
        pair = (0, 1)

        ref_mean, ref_lower, ref_upper = gamma_smc_numpy(
            G, positions, pair, Ne=10000.0, mu=uniform_mu, rho=uniform_rho
        )

        result = gamma_smc_cu.gamma_smc_forward(
            G, positions, [pair], Ne=10000.0, mu=uniform_mu, rho=uniform_rho
        )

        # Shape is [S, 1] — squeeze pair dim
        gpu_mean = np.array(result["mean"])[:, 0]
        gpu_lower = np.array(result["lower"])[:, 0]
        gpu_upper = np.array(result["upper"])[:, 0]

        np.testing.assert_allclose(gpu_mean, ref_mean, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(gpu_lower, ref_lower, rtol=1e-2, atol=1e-6)
        np.testing.assert_allclose(gpu_upper, ref_upper, rtol=1e-2, atol=1e-6)

    def test_multiple_pairs(self, small_simulation, uniform_mu, uniform_rho):
        """Batched GPU call matches per-pair numpy reference."""
        _, G, positions = small_simulation
        pairs = [(0, 1), (0, 5), (1, 2)]

        result = gamma_smc_cu.gamma_smc_forward(
            G, positions, pairs, Ne=10000.0, mu=uniform_mu, rho=uniform_rho
        )
        gpu = np.array(result["mean"])  # [S, 3]

        for idx, pair in enumerate(pairs):
            ref_mean, _, _ = gamma_smc_numpy(
                G, positions, pair, Ne=10000.0, mu=uniform_mu, rho=uniform_rho
            )
            np.testing.assert_allclose(gpu[:, idx], ref_mean, rtol=1e-3, atol=1e-6)

    def test_mean_positive(self, small_simulation, uniform_mu, uniform_rho):
        """All mean TMRCA values should be positive."""
        _, G, positions = small_simulation
        result = gamma_smc_cu.gamma_smc_forward(
            G, positions, [(0, 1)], Ne=10000.0, mu=uniform_mu, rho=uniform_rho
        )
        assert np.all(np.array(result["mean"]) > 0)

    def test_ci_contains_mean(self, small_simulation, uniform_mu, uniform_rho):
        """95% CI should contain the mean at every site."""
        _, G, positions = small_simulation
        result = gamma_smc_cu.gamma_smc_forward(
            G, positions, [(0, 1)], Ne=10000.0, mu=uniform_mu, rho=uniform_rho
        )
        mean = np.array(result["mean"])[:, 0]
        lower = np.array(result["lower"])[:, 0]
        upper = np.array(result["upper"])[:, 0]

        assert np.all(lower <= mean), "lower CI exceeds mean"
        assert np.all(mean <= upper), "mean exceeds upper CI"

    def test_lower_nonnegative(self, small_simulation, uniform_mu, uniform_rho):
        """Lower CI bound should be non-negative."""
        _, G, positions = small_simulation
        result = gamma_smc_cu.gamma_smc_forward(
            G, positions, [(0, 1)], Ne=10000.0, mu=uniform_mu, rho=uniform_rho
        )
        assert np.all(np.array(result["lower"]) >= 0)

    def test_different_pairs_differ(self, small_simulation, uniform_mu, uniform_rho):
        """Different pairs should produce different results."""
        _, G, positions = small_simulation
        result = gamma_smc_cu.gamma_smc_forward(
            G, positions, [(0, 1), (0, 5)], Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        )
        m = np.array(result["mean"])  # [S, 2]
        assert not np.allclose(m[:, 0], m[:, 1], atol=1e-3)

    def test_stride(self, small_simulation, uniform_mu, uniform_rho):
        """Strided output should match every-Nth site of full output."""
        _, G, positions = small_simulation
        stride = 4

        full = gamma_smc_cu.gamma_smc_forward(
            G, positions, [(0, 1)], Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho, stride=1
        )
        strided = gamma_smc_cu.gamma_smc_forward(
            G, positions, [(0, 1)], Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho, stride=stride
        )

        full_mean = np.array(full["mean"])[:, 0]
        strided_mean = np.array(strided["mean"])[:, 0]

        expected = full_mean[::stride]
        np.testing.assert_allclose(strided_mean, expected, rtol=1e-5)

    def test_output_shape(self, small_simulation, uniform_mu, uniform_rho):
        """Output arrays have correct shape [out_S, n_pairs]."""
        _, G, positions = small_simulation
        S = len(positions)
        pairs = [(0, 1), (0, 5), (1, 2)]

        result = gamma_smc_cu.gamma_smc_forward(
            G, positions, pairs, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        )

        assert np.array(result["mean"]).shape == (S, 3)
        assert np.array(result["lower"]).shape == (S, 3)
        assert np.array(result["upper"]).shape == (S, 3)

    def test_mean_only_mode(self, small_simulation, uniform_mu, uniform_rho):
        """mean_only=True returns only mean, matches full mode."""
        _, G, positions = small_simulation
        pairs = [(0, 1), (0, 5)]

        full = gamma_smc_cu.gamma_smc_forward(
            G, positions, pairs, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho, mean_only=False
        )
        mean_only = gamma_smc_cu.gamma_smc_forward(
            G, positions, pairs, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho, mean_only=True
        )

        np.testing.assert_allclose(
            np.array(mean_only["mean"]),
            np.array(full["mean"]),
            rtol=1e-5
        )
        assert "lower" not in mean_only
        assert "upper" not in mean_only
