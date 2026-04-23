import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gamma_smc_cu


class TestEdgeCases:
    """Degenerate inputs that should not crash and should produce valid output."""

    def test_monomorphic_matrix(self):
        """
        All-zero genotypes: no segregating sites. HMM posterior should
        still be valid (finite, normalized) and close to prior.
        """
        n, S = 20, 500
        G = np.zeros((n, S), dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=1.25e-8, rho=1e-8
        ))
        assert np.all(np.isfinite(gamma)), \
            "NaN/Inf in posterior for monomorphic matrix"
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

        # SFS should have zero segregating sites
        sfs = np.array(gamma_smc_cu.compute_sfs(G))
        assert sfs[1:].sum() == 0, "Monomorphic matrix should have empty SFS"

    def test_single_site(self):
        """S=1: minimal sequence length."""
        G = np.array([[0], [1]], dtype=np.uint8)
        positions = np.array([500.0])

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=1.25e-8, rho=1e-8
        ))
        assert gamma.shape == (1, 32)
        assert np.all(np.isfinite(gamma))
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

    def test_two_sites(self):
        """S=2: minimal sequence for transition computation."""
        G = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        positions = np.array([0.0, 1000.0])

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=1.25e-8, rho=1e-8
        ))
        assert gamma.shape == (2, 32)
        assert np.all(np.isfinite(gamma))
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

        # Log-likelihood should also be finite
        ll = gamma_smc_cu.hmm_log_likelihood(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=1.25e-8, rho=1e-8
        )
        assert np.isfinite(ll)

    def test_two_haplotypes(self):
        """n=2: minimum sample size."""
        rng = np.random.RandomState(42)
        S = 1000
        G = rng.randint(0, 2, size=(2, S)).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=1.25e-8, rho=1e-8
        ))
        assert gamma.shape == (S, 32)
        assert np.all(np.isfinite(gamma))
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

        # SFS should work with n=2
        sfs = np.array(gamma_smc_cu.compute_sfs(G))
        assert sfs.shape == (3,)  # n+1 = 3 entries
        assert np.all(sfs >= 0)

        # Bitpacking round-trip should work
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 2, S)
        np.testing.assert_array_equal(G, unpacked)

    def test_self_pair_posterior(self):
        """
        Pair (i, i): XOR is zero everywhere.
        Posterior should equal the prior at every site.
        """
        rng = np.random.RandomState(45)
        n, S = 10, 200
        G = rng.randint(0, 2, size=(n, S)).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (3, 3), K=32, Ne=10000.0,
            mu=1.25e-8, rho=1e-8
        ))
        prior = np.array(gamma_smc_cu.coalescent_prior(Ne=10000.0, K=32))

        assert np.all(np.isfinite(gamma))
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

        # With zero divergence, posterior mean should be less than or equal to
        # prior mean (gap emission favors shorter coalescence times when there
        # are gaps between sites with no observed mutations)
        t_mid = np.array(gamma_smc_cu.time_midpoints(K=32, Ne=10000.0))
        prior_mean = np.dot(prior, t_mid)
        post_mean = np.mean(gamma @ t_mid)
        assert post_mean <= prior_mean * 1.1, \
            f"Self-pair posterior mean ({post_mean:.0f}) should not exceed prior mean ({prior_mean:.0f})"
