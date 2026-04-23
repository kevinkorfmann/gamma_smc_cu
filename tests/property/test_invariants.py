import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gamma_smc_cu


class TestInvariants:
    """
    Mathematical invariants that must hold regardless of input.
    These catch subtle bugs that specific test cases might miss.
    """

    def test_pair_symmetry(self, small_simulation, uniform_mu, uniform_rho):
        """hmm_posterior(i,j) == hmm_posterior(j,i) since XOR is symmetric."""
        _, G, positions = small_simulation
        pairs = [(0, 5), (1, 10), (3, 15)]

        for i, j in pairs:
            gamma_ij = np.array(gamma_smc_cu.hmm_posterior(
                G, positions, (i, j), K=32, Ne=10000.0,
                mu=uniform_mu, rho=uniform_rho
            ))
            gamma_ji = np.array(gamma_smc_cu.hmm_posterior(
                G, positions, (j, i), K=32, Ne=10000.0,
                mu=uniform_mu, rho=uniform_rho
            ))
            np.testing.assert_allclose(
                gamma_ij, gamma_ji, rtol=1e-5, atol=1e-6,
                err_msg=f"Posterior not symmetric for pair ({i},{j})"
            )

    def test_posterior_normalization(self, small_simulation,
                                     uniform_mu, uniform_rho):
        """Sum over K time bins = 1 at every site for every pair."""
        _, G, positions = small_simulation
        pairs = [(0, 1), (2, 7), (5, 15)]

        for pair in pairs:
            gamma = np.array(gamma_smc_cu.hmm_posterior(
                G, positions, pair, K=32, Ne=10000.0,
                mu=uniform_mu, rho=uniform_rho
            ))
            sums = gamma.sum(axis=1)
            np.testing.assert_allclose(
                sums, 1.0, rtol=1e-4,
                err_msg=f"Posterior not normalized for pair {pair}"
            )

    def test_sfs_symmetry(self, small_simulation):
        """
        Folded SFS check: for a biallelic site, swapping 0 and 1
        labels should produce the mirrored SFS.
        SFS[k] counts sites with k derived alleles among n haplotypes.
        If we flip all alleles, SFS[k] -> SFS[n-k].
        """
        _, G, _ = small_simulation
        n = G.shape[0]

        sfs_original = np.array(gamma_smc_cu.compute_sfs(G))
        G_flipped = 1 - G
        sfs_flipped = np.array(gamma_smc_cu.compute_sfs(G_flipped))

        # sfs_original[k] should equal sfs_flipped[n-k]
        for k in range(n + 1):
            assert sfs_original[k] == sfs_flipped[n - k], \
                (f"SFS symmetry violated: sfs[{k}]={sfs_original[k]} != "
                 f"sfs_flipped[{n-k}]={sfs_flipped[n - k]}")

    def test_posterior_nonnegative(self, small_simulation,
                                    uniform_mu, uniform_rho):
        """All posterior entries must be non-negative."""
        _, G, positions = small_simulation
        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        ))
        assert np.all(gamma >= 0), "Negative posterior probabilities found"

    def test_log_likelihood_is_finite(self, small_simulation,
                                       uniform_mu, uniform_rho):
        """Log-likelihood should be finite and negative."""
        _, G, positions = small_simulation
        ll = gamma_smc_cu.hmm_log_likelihood(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        )
        assert np.isfinite(ll), "Log-likelihood is not finite"
        assert ll < 0, "Log-likelihood should be negative"
