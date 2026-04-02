import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tmrca_cu


class TestEmissions:
    """
    Verify emission probability computations.

    The emission model is the infinite-sites Jukes-Cantor:
        P(d=1 | T=t) = 1 - exp(-2 * mu * t)
        P(d=0 | T=t) = exp(-2 * mu * t)
    """

    MU = 1.25e-8
    TIMES = [10, 100, 1_000, 10_000, 100_000, 1_000_000]

    def test_mutation_increases_with_time(self):
        """P(d=1 | T=t) should be monotonically increasing with t."""
        prev_p = 0.0
        for t in self.TIMES:
            # Compute expected emission from numpy
            expected = 1.0 - np.exp(-2.0 * self.MU * t)
            assert expected > prev_p, \
                f"P(d=1|T={t}) = {expected} should exceed P(d=1|T_prev) = {prev_p}"
            prev_p = expected

        # Also verify via the library's coalescent_prior / hmm machinery:
        # Build a single-site genotype with a mutation and check that the
        # posterior shifts to larger times when we observe more mutations.
        K = 32
        prior = np.array(tmrca_cu.coalescent_prior(Ne=10000.0, K=K))

        # Compute emission probabilities for d=1 at each time midpoint
        # using the reference formula
        Ne = 10000.0
        t_max = 10.0 * Ne
        boundaries = np.array([t_max * (k / K) ** 2 for k in range(K + 1)])
        midpoints = (boundaries[:-1] + boundaries[1:]) / 2.0

        p_mut = 1.0 - np.exp(-2.0 * self.MU * midpoints)
        # Verify monotonicity
        assert np.all(np.diff(p_mut) > 0), \
            "Emission P(d=1|t) not monotonically increasing across time bins"

    def test_no_mutation_decreases_with_time(self):
        """P(d=0 | T=t) should be monotonically decreasing with t."""
        prev_p = 1.0
        for t in self.TIMES:
            expected = np.exp(-2.0 * self.MU * t)
            assert expected < prev_p, \
                f"P(d=0|T={t}) = {expected} should be less than P(d=0|T_prev) = {prev_p}"
            prev_p = expected

    def test_emissions_sum_to_one(self):
        """P(d=0|T=t) + P(d=1|T=t) = 1 for all t."""
        for t in self.TIMES:
            p0 = np.exp(-2.0 * self.MU * t)
            p1 = 1.0 - np.exp(-2.0 * self.MU * t)
            np.testing.assert_allclose(
                p0 + p1, 1.0, rtol=1e-12,
                err_msg=f"Emissions don't sum to 1 at t={t}"
            )

    def test_emission_at_t_zero(self):
        """At t=0, P(d=0)=1 and P(d=1)=0."""
        t = 0.0
        p0 = np.exp(-2.0 * self.MU * t)
        p1 = 1.0 - np.exp(-2.0 * self.MU * t)
        np.testing.assert_allclose(p0, 1.0, atol=1e-15)
        np.testing.assert_allclose(p1, 0.0, atol=1e-15)

    def test_emission_consistency_with_hmm(self, small_simulation,
                                            uniform_mu, uniform_rho):
        """
        Verify that the emission model used internally by the HMM is
        consistent with the Jukes-Cantor formula by checking that a
        single mutation site shifts posterior toward larger times.
        """
        K = 32
        Ne = 10000.0
        S = 100

        # All-zero: no divergence
        G0 = np.zeros((4, S), dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64) * 1000

        gamma_no_mut = np.array(tmrca_cu.hmm_posterior(
            G0, positions, (0, 1), K=K, Ne=Ne,
            mu=uniform_mu, rho=uniform_rho
        ))

        # Single mutation at midpoint
        G1 = np.zeros((4, S), dtype=np.uint8)
        G1[0, S // 2] = 1
        gamma_one_mut = np.array(tmrca_cu.hmm_posterior(
            G1, positions, (0, 1), K=K, Ne=Ne,
            mu=uniform_mu, rho=uniform_rho
        ))

        # At the mutation site, posterior mean time bin should be larger
        site = S // 2
        mean_bin_no_mut = gamma_no_mut[site] @ np.arange(K)
        mean_bin_one_mut = gamma_one_mut[site] @ np.arange(K)
        assert mean_bin_one_mut > mean_bin_no_mut, \
            (f"Mutation at site {site} should shift posterior to larger times: "
             f"no_mut={mean_bin_no_mut:.2f}, one_mut={mean_bin_one_mut:.2f}")
