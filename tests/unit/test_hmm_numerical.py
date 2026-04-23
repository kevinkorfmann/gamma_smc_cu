import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gamma_smc_cu


class TestHMMNumericalStability:
    """
    Edge cases that stress numerical precision: very long sequences,
    extreme TMRCA values, near-zero emissions.
    """

    def test_long_monomorphic_stretch(self):
        """
        10 sites spread over 10 Mb with only 2 mutations.
        Forward probabilities must not underflow to NaN/Inf.
        Verifies log-space rescaling is working.
        """
        n, S = 4, 10
        G = np.zeros((n, S), dtype=np.uint8)
        G[0, 0] = 1   # single mutation at start
        G[0, -1] = 1  # single mutation at end
        positions = np.linspace(0, 10_000_000, S)

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=1.25e-8, rho=1e-8
        ))

        # Must not contain NaN or Inf
        assert np.all(np.isfinite(gamma)), \
            "NaN/Inf in posterior after long monomorphic stretch"
        # Must still sum to 1
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

    def test_dense_mutations(self):
        """
        Every site is a mutation (saturated divergence).
        Posterior should concentrate on large time bins.
        """
        n, S = 4, 5000
        G = np.zeros((n, S), dtype=np.uint8)
        G[0, :] = 1  # haplotype 0 differs from all others at every site
        positions = np.arange(S, dtype=np.float64)

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=1.25e-8, rho=1e-8
        ))
        assert np.all(np.isfinite(gamma)), \
            "NaN/Inf in posterior with dense mutations"

        # Posterior mean TMRCA should be large: past the midpoint of
        # the time grid.
        prior = np.array(gamma_smc_cu.coalescent_prior(Ne=10000.0, K=32))
        K = 32
        # The posterior-weighted average time bin index should be
        # larger than what the prior alone would give.
        mean_bin_posterior = np.mean(gamma @ np.arange(K))
        mean_bin_prior = prior @ np.arange(K)
        assert mean_bin_posterior > mean_bin_prior, \
            (f"Saturated divergence should shift posterior to larger times: "
             f"posterior mean bin={mean_bin_posterior:.1f}, "
             f"prior mean bin={mean_bin_prior:.1f}")

    def test_very_small_ne(self):
        """Ne = 100 -- very recent coalescence, tests small time bins."""
        n, S = 10, 1000
        rng = np.random.RandomState(42)
        G = rng.randint(0, 2, size=(n, S)).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=100.0,
            mu=1.25e-8, rho=1e-8
        ))
        assert np.all(np.isfinite(gamma)), \
            "NaN/Inf in posterior with Ne=100"
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

    def test_very_large_ne(self):
        """Ne = 1e7 -- very ancient coalescence, tests large time bins."""
        n, S = 10, 1000
        rng = np.random.RandomState(43)
        G = rng.randint(0, 2, size=(n, S)).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10_000_000.0,
            mu=1.25e-8, rho=1e-8
        ))
        assert np.all(np.isfinite(gamma)), \
            "NaN/Inf in posterior with Ne=1e7"
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

    def test_identical_haplotypes(self):
        """
        XOR is all zeros (identical pair). The posterior should be valid
        and favor smaller coalescence times (gap emission penalizes large t
        because no mutations were observed in the gaps).
        """
        n, S = 10, 1000
        rng = np.random.RandomState(44)
        G = np.zeros((n, S), dtype=np.uint8)
        G[0, :] = rng.randint(0, 2, S).astype(np.uint8)
        G[1, :] = G[0, :]  # identical
        positions = np.arange(S, dtype=np.float64) * 100

        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=1.25e-8, rho=1e-8
        ))
        t_mid = np.array(gamma_smc_cu.time_midpoints(K=32, Ne=10000.0))
        prior = np.array(gamma_smc_cu.coalescent_prior(Ne=10000.0, K=32))

        assert np.all(np.isfinite(gamma)), \
            "NaN/Inf in posterior for identical haplotypes"
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

        # With no mutations, posterior mean TMRCA should be less than prior mean
        # (gap emission favors shorter coalescence times)
        prior_mean = np.dot(prior, t_mid)
        post_mean = np.mean(gamma @ t_mid)
        assert post_mean < prior_mean, \
            f"Identical pair should have smaller mean TMRCA ({post_mean:.0f}) than prior ({prior_mean:.0f})"
