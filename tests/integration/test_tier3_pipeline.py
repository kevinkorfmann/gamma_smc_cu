"""
Integration tests for Tier 3 pipeline: HMM-based TMRCA inference via
CoalescenceEstimator.infer_tmrca().

These tests exercise the full path from estimator construction through
GPU HMM forward-backward to posterior summaries in TMRCAResult.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tmrca_cu
from tmrca_cu.estimator import CoalescenceEstimator, TMRCAResult


class TestInferTMRCA:
    """End-to-end HMM inference through CoalescenceEstimator.infer_tmrca()."""

    def test_result_type(self, small_simulation):
        """infer_tmrca returns a TMRCAResult."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)])
        assert isinstance(result, TMRCAResult)

    def test_output_shapes(self, small_simulation):
        """All output arrays have correct shapes."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        pairs = [(0, 1), (2, 3), (0, 5)]
        result = est.infer_tmrca(pairs=pairs)

        n_pairs = len(pairs)
        S = est.S
        K = 32

        assert result.tmrca_mean.shape == (n_pairs, S)
        assert result.tmrca_lower.shape == (n_pairs, S)
        assert result.tmrca_upper.shape == (n_pairs, S)
        assert result.log_likelihood.shape == (n_pairs,)
        assert result.positions.shape == (S,)
        assert result.time_midpoints.shape == (K,)
        assert len(result.pairs) == n_pairs

    def test_posterior_mean_positive(self, small_simulation):
        """Posterior mean TMRCA should be strictly positive."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)])
        assert np.all(result.tmrca_mean > 0)

    def test_credible_interval_ordering(self, small_simulation):
        """Lower bound <= mean <= upper bound at every site."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)])
        assert np.all(result.tmrca_lower <= result.tmrca_mean + 1e-6)
        assert np.all(result.tmrca_mean <= result.tmrca_upper + 1e-6)

    def test_credible_interval_nonneg(self, small_simulation):
        """Credible interval bounds are non-negative."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)])
        assert np.all(result.tmrca_lower >= 0)
        assert np.all(result.tmrca_upper >= 0)

    def test_log_likelihood_finite(self, small_simulation):
        """Log-likelihood should be finite and negative."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)])
        assert np.all(np.isfinite(result.log_likelihood))
        assert np.all(result.log_likelihood < 0)

    def test_different_pairs_different_results(self, small_simulation):
        """Different pairs should generally produce different TMRCA estimates."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1), (0, 5)])
        assert not np.allclose(
            result.tmrca_mean[0], result.tmrca_mean[1], atol=1.0
        )

    def test_time_midpoints_sorted(self, small_simulation):
        """Time bin midpoints should be monotonically increasing."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)])
        assert np.all(np.diff(result.time_midpoints) > 0)

    def test_positions_preserved(self, small_simulation):
        """Positions in result should match input positions."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)])
        np.testing.assert_array_equal(result.positions, positions)

    def test_pairs_preserved(self, small_simulation):
        """Pair list in result should match input pairs."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        pairs = [(0, 1), (3, 7)]
        result = est.infer_tmrca(pairs=pairs)
        assert result.pairs == pairs

    def test_n_iterations(self, small_simulation):
        """n_iterations should be at least 1."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)], max_iterations=3)
        assert result.n_iterations >= 1


class TestTMRCAAccuracy:
    """
    Check that inferred TMRCA is correlated with true TMRCA from the
    tree sequence.  We don't expect perfect recovery, but the posterior
    mean should track the true signal.
    """

    def test_correlation_with_true_tmrca(self, small_simulation, true_pairwise_tmrca):
        """Posterior mean should be positively correlated with true TMRCA."""
        ts, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)

        for pair, true_tmrca_bp in true_pairwise_tmrca.items():
            result = est.infer_tmrca(pairs=[pair])
            inferred = result.tmrca_mean[0]  # (S,)

            # Map true TMRCA (per-bp) to site positions
            site_positions = positions.astype(int)
            site_positions = np.clip(site_positions, 0, len(true_tmrca_bp) - 1)
            true_at_sites = true_tmrca_bp[site_positions]

            # Smooth both for a fairer comparison (HMM output is already smooth)
            from numpy.lib.stride_tricks import sliding_window_view
            w = min(50, len(inferred) // 4)
            if w < 2:
                continue
            inferred_smooth = np.convolve(inferred, np.ones(w) / w, mode='valid')
            true_smooth = np.convolve(true_at_sites, np.ones(w) / w, mode='valid')

            corr = np.corrcoef(inferred_smooth, true_smooth)[0, 1]
            assert corr > 0.0, (
                f"Pair {pair}: correlation {corr:.3f} between inferred and "
                f"true TMRCA is not positive"
            )

    def test_mean_tmrca_reasonable_scale(self, small_simulation):
        """
        Average posterior mean across sites should be in the right ballpark
        relative to Ne.  E[T_2] = Ne for two haplotypes under constant size.
        """
        _, G, positions = small_simulation
        Ne = 10_000
        est = CoalescenceEstimator(G, positions, Ne=Ne)
        result = est.infer_tmrca(pairs=[(0, 1)])
        mean_tmrca = result.tmrca_mean[0].mean()
        # Allow generous bounds: 0.01*Ne to 10*Ne
        assert 0.01 * Ne < mean_tmrca < 10 * Ne, (
            f"Mean TMRCA {mean_tmrca:.0f} is outside plausible range "
            f"[{0.01*Ne:.0f}, {10*Ne:.0f}]"
        )


class TestFromTreeSequencePipeline:
    """End-to-end: from_tree_sequence -> infer_tmrca."""

    def test_full_pipeline(self, small_simulation):
        """Complete pipeline from tree sequence to TMRCA result."""
        ts = small_simulation[0]
        est = CoalescenceEstimator.from_tree_sequence(ts)
        result = est.infer_tmrca(pairs=[(0, 1)])

        assert isinstance(result, TMRCAResult)
        assert result.tmrca_mean.shape[0] == 1
        assert result.tmrca_mean.shape[1] == est.S
        assert np.all(np.isfinite(result.tmrca_mean))
        assert np.all(result.tmrca_mean > 0)

    def test_pipeline_with_multiple_pairs(self, small_simulation):
        """Pipeline works with multiple pairs."""
        ts = small_simulation[0]
        est = CoalescenceEstimator.from_tree_sequence(ts, Ne=10_000)
        pairs = [(0, 1), (2, 3), (4, 5), (0, 19)]
        result = est.infer_tmrca(pairs=pairs)

        assert result.tmrca_mean.shape[0] == len(pairs)
        assert len(result.log_likelihood) == len(pairs)
        # Each pair should have finite log-likelihood
        assert np.all(np.isfinite(result.log_likelihood))


class TestEdgeCases:
    """Edge cases and boundary conditions for the Tier 3 pipeline."""

    def test_single_site(self):
        """Pipeline handles a single-site genotype matrix."""
        G = np.array([[0], [1], [0], [1]], dtype=np.uint8)
        positions = np.array([1000.0])
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)])
        assert result.tmrca_mean.shape == (1, 1)
        assert np.all(np.isfinite(result.tmrca_mean))

    def test_all_zeros(self):
        """Pipeline handles monomorphic data (all zeros)."""
        G = np.zeros((4, 20), dtype=np.uint8)
        positions = np.arange(20, dtype=np.float64) * 100.0
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca(pairs=[(0, 1)])
        assert result.tmrca_mean.shape == (1, 20)
        assert np.all(np.isfinite(result.tmrca_mean))

    def test_default_pairs_infer(self, small_simulation):
        """infer_tmrca with no explicit pairs uses default pair sampling."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        result = est.infer_tmrca()
        assert result.tmrca_mean.shape[0] > 0
        assert result.tmrca_mean.shape[1] == est.S
