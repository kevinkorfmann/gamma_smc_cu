import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import msprime
import gamma_smc_cu


def extract_true_tmrca_at_sites(ts, pair, positions):
    """
    Extract the true TMRCA for a pair of haplotypes at each variant site.

    Parameters
    ----------
    ts : tskit.TreeSequence
    pair : tuple of (int, int)
    positions : np.ndarray of variant positions

    Returns
    -------
    np.ndarray of shape (len(positions),) with true TMRCA values
    """
    i, j = pair
    tmrca = np.zeros(len(positions))
    tree_iter = ts.trees()
    tree = next(tree_iter)

    for s, pos in enumerate(positions):
        # Advance tree to cover this position
        while pos >= tree.interval.right:
            tree = next(tree_iter)
        tmrca[s] = tree.tmrca(i, j)

    return tmrca


@pytest.mark.slow
class TestAccuracyMsprime:
    """
    Validate TMRCA estimation accuracy against msprime ground truth.

    Uses a simple constant-Ne simulation (n=20, 100 kb) and checks
    that HMM posterior mean TMRCA is correlated with the true TMRCA
    from the tree sequence.

    These are statistical tests and may occasionally fail due to
    simulation randomness. The thresholds are set conservatively.
    """

    @pytest.fixture(scope="class")
    def msprime_data(self):
        """Simulate a small tree sequence with msprime."""
        ts = msprime.sim_ancestry(
            samples=10,  # 10 diploid = 20 haploid
            sequence_length=100_000,
            recombination_rate=1e-8,
            population_size=10_000,
            random_seed=42,
        )
        ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)
        G = ts.genotype_matrix().T.astype(np.uint8)
        positions = np.array([v.position for v in ts.variants()])
        return ts, G, positions

    def test_posterior_mean_correlation(self, msprime_data):
        """
        For a few pairs, the posterior mean TMRCA should be positively
        correlated with the true TMRCA. r > 0.5 is a reasonable bar
        for per-site HMM without EP.
        """
        ts, G, positions = msprime_data
        test_pairs = [(0, 1), (0, 5), (2, 10)]

        for pair in test_pairs:
            # Get true TMRCA at variant sites
            true_tmrca = extract_true_tmrca_at_sites(ts, pair, positions)

            # Run HMM posterior
            gamma = np.array(gamma_smc_cu.hmm_posterior(
                G, positions, pair, K=32, Ne=10000.0,
                mu=1.25e-8, rho=1e-8
            ))

            # Compute posterior mean TMRCA at each site
            Ne = 10000.0
            K = 32
            t_max = 10.0 * Ne
            boundaries = np.array([t_max * (k / K) ** 2 for k in range(K + 1)])
            midpoints = (boundaries[:-1] + boundaries[1:]) / 2.0

            est_tmrca = gamma @ midpoints

            # Check correlation
            r = np.corrcoef(true_tmrca, est_tmrca)[0, 1]
            assert r > 0.5, \
                (f"Pair {pair}: correlation r={r:.3f} < 0.5 between "
                 f"estimated and true TMRCA")

    def test_posterior_mean_order_preserved(self, msprime_data):
        """
        Among test pairs, the pair with the highest average true TMRCA
        should also have a higher estimated TMRCA than the pair with
        the lowest average true TMRCA.
        """
        ts, G, positions = msprime_data
        test_pairs = [(0, 1), (0, 5), (2, 10), (3, 15)]

        Ne = 10000.0
        K = 32
        t_max = 10.0 * Ne
        boundaries = np.array([t_max * (k / K) ** 2 for k in range(K + 1)])
        midpoints = (boundaries[:-1] + boundaries[1:]) / 2.0

        true_means = []
        est_means = []

        for pair in test_pairs:
            true_tmrca = extract_true_tmrca_at_sites(ts, pair, positions)
            true_means.append(np.mean(true_tmrca))

            gamma = np.array(gamma_smc_cu.hmm_posterior(
                G, positions, pair, K=K, Ne=Ne,
                mu=1.25e-8, rho=1e-8
            ))
            est_tmrca = gamma @ midpoints
            est_means.append(np.mean(est_tmrca))

        # The pair with max true mean should have a higher estimated mean
        # than the pair with min true mean
        idx_max_true = np.argmax(true_means)
        idx_min_true = np.argmin(true_means)

        assert est_means[idx_max_true] > est_means[idx_min_true], \
            (f"Order not preserved: pair {test_pairs[idx_max_true]} "
             f"true={true_means[idx_max_true]:.0f} est={est_means[idx_max_true]:.0f}, "
             f"pair {test_pairs[idx_min_true]} "
             f"true={true_means[idx_min_true]:.0f} est={est_means[idx_min_true]:.0f}")

    def test_posterior_finite_and_normalized(self, msprime_data):
        """Basic sanity: all posteriors are finite and sum to 1."""
        _, G, positions = msprime_data

        for pair in [(0, 1), (5, 15)]:
            gamma = np.array(gamma_smc_cu.hmm_posterior(
                G, positions, pair, K=32, Ne=10000.0,
                mu=1.25e-8, rho=1e-8
            ))
            assert np.all(np.isfinite(gamma)), \
                f"Non-finite posterior values for pair {pair}"
            np.testing.assert_allclose(
                gamma.sum(axis=1), 1.0, rtol=1e-3,
                err_msg=f"Posterior not normalized for pair {pair}"
            )
