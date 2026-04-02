"""
Integration tests for Tier 1 pipeline: site_pi, pairwise_divergence, tmrca_landscape.

These tests exercise the full path from CoalescenceEstimator construction through
GPU-accelerated divergence computation and back to numpy results.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tmrca_cu
from tmrca_cu.estimator import CoalescenceEstimator


class TestSitePi:
    """Per-site nucleotide diversity via CoalescenceEstimator.site_pi()."""

    def test_shape(self, small_simulation):
        """site_pi returns an array of length S."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        pi = est.site_pi()
        assert pi.shape == (est.S,)

    def test_nonneg(self, small_simulation):
        """pi values must be non-negative."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        pi = est.site_pi()
        assert np.all(pi >= 0)

    def test_max_value(self, small_simulation):
        """pi at a biallelic site is at most 2*0.5*0.5*n/(n-1) = n/(2(n-1))."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        pi = est.site_pi()
        max_pi = est.n / (2.0 * (est.n - 1))
        # pi = 2*p*(1-p)*n/(n-1); maximum at p=0.5 gives n/(2(n-1))
        assert np.all(pi <= max_pi + 1e-10)

    def test_monomorphic_sites(self):
        """Sites with no variation should have pi = 0."""
        G = np.zeros((10, 50), dtype=np.uint8)
        positions = np.arange(50, dtype=np.float64)
        est = CoalescenceEstimator(G, positions)
        pi = est.site_pi()
        np.testing.assert_allclose(pi, 0.0)

    def test_matches_manual_computation(self, small_simulation):
        """site_pi matches a manual numpy computation."""
        _, G, positions = small_simulation
        n = G.shape[0]
        freq = G.astype(np.float64).mean(axis=0)
        expected_pi = 2.0 * freq * (1.0 - freq) * n / (n - 1)

        est = CoalescenceEstimator(G, positions)
        pi = est.site_pi()
        np.testing.assert_allclose(pi, expected_pi, rtol=1e-10)


class TestPairwiseDivergence:
    """Windowed pairwise divergence via GPU prefix scan."""

    def test_shape_single_window(self, small_simulation):
        """Output shape is (n_pairs, S) for a single window size."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        pairs = [(0, 1), (2, 3)]
        div = est.pairwise_divergence(pairs=pairs, window_sizes=[50])
        assert div.shape == (2, est.S)

    def test_shape_multiple_windows(self, small_simulation):
        """Output shape is (n_pairs, S, n_windows) for multiple window sizes."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        pairs = [(0, 1)]
        div = est.pairwise_divergence(pairs=pairs, window_sizes=[50, 100])
        assert div.shape == (1, est.S, 2)

    def test_default_pairs(self, small_simulation):
        """With no pairs specified, a default set is generated."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        div = est.pairwise_divergence()
        assert div.ndim == 2
        assert div.shape[1] == est.S

    def test_nonneg(self, small_simulation):
        """Divergence values must be non-negative."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        div = est.pairwise_divergence(pairs=[(0, 1)], window_sizes=[50])
        assert np.all(div >= 0)

    def test_bounded_by_one(self, small_simulation):
        """Windowed divergence (fraction of differing sites) is in [0, 1]."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        div = est.pairwise_divergence(pairs=[(0, 1)], window_sizes=[50])
        assert np.all(div <= 1.0 + 1e-6)

    def test_identical_pair(self, small_simulation):
        """A haplotype compared with itself should have zero divergence."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        div = est.pairwise_divergence(pairs=[(0, 0)], window_sizes=[50])
        np.testing.assert_allclose(div, 0.0, atol=1e-6)

    def test_larger_window_is_smoother(self, small_simulation):
        """Larger window should produce less variable divergence."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        pairs = [(0, 1)]
        div_small = est.pairwise_divergence(pairs=pairs, window_sizes=[10])
        div_large = est.pairwise_divergence(pairs=pairs, window_sizes=[200])
        # Standard deviation should be lower for larger window
        assert np.std(div_large) <= np.std(div_small) + 1e-6


class TestTMRCALandscape:
    """Genome-wide TMRCA landscape estimation."""

    def test_shape(self, small_simulation):
        """Landscape returns one value per site."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        landscape = est.tmrca_landscape(n_pairs=10, window_bp=5000)
        assert landscape.shape == (est.S,)

    def test_nonneg(self, small_simulation):
        """TMRCA estimates should be non-negative."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        landscape = est.tmrca_landscape(n_pairs=10, window_bp=5000)
        assert np.all(landscape >= 0)

    def test_reasonable_scale(self, small_simulation):
        """
        With Ne=10000, average TMRCA should be around Ne for randomly sampled
        pairs, roughly within an order of magnitude.
        """
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions, Ne=10_000)
        landscape = est.tmrca_landscape(n_pairs=50, window_bp=20000)
        mean_tmrca = landscape.mean()
        # E[T_2] = Ne for two haplotypes under constant size
        assert 100 < mean_tmrca < 200_000, (
            f"Mean landscape TMRCA {mean_tmrca:.0f} is outside plausible range"
        )


class TestSFS:
    """Site frequency spectrum via CoalescenceEstimator.sfs()."""

    def test_shape(self, small_simulation):
        """SFS has n+1 entries."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        sfs = est.sfs()
        assert sfs.shape == (est.n + 1,)

    def test_total_equals_num_sites(self, small_simulation):
        """Sum of SFS should equal total number of sites."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        sfs = est.sfs()
        assert sfs.sum() == est.S

    def test_nonneg(self, small_simulation):
        """All SFS entries are non-negative."""
        _, G, positions = small_simulation
        est = CoalescenceEstimator(G, positions)
        sfs = est.sfs()
        assert np.all(sfs >= 0)


class TestFromTreeSequence:
    """Construction from a tskit TreeSequence."""

    def test_basic_construction(self, small_simulation):
        """from_tree_sequence produces a valid estimator."""
        ts, G, positions = small_simulation
        est = CoalescenceEstimator.from_tree_sequence(ts)
        assert est.n == G.shape[0]
        assert est.S == G.shape[1]
        np.testing.assert_array_equal(est.genotypes, G)
        np.testing.assert_array_equal(est.positions, positions)

    def test_explicit_parameters(self, small_simulation):
        """Explicitly provided mu/rho/Ne override defaults."""
        ts = small_simulation[0]
        est = CoalescenceEstimator.from_tree_sequence(
            ts, mu=2e-8, rho=3e-8, Ne=5000
        )
        assert est.mu == 2e-8
        assert est.rho == 3e-8
        assert est.Ne == 5000

    def test_ne_estimation(self, small_simulation):
        """Auto-estimated Ne should be in a reasonable range."""
        ts = small_simulation[0]
        est = CoalescenceEstimator.from_tree_sequence(ts)
        # True Ne is 10000; Watterson estimate should be in the right ballpark
        assert 1000 < est.Ne < 100_000


class TestInputValidation:
    """Input validation in CoalescenceEstimator constructor."""

    def test_rejects_1d_genotypes(self):
        with pytest.raises(ValueError, match="2D"):
            CoalescenceEstimator(
                np.zeros(100, dtype=np.uint8),
                np.arange(100, dtype=np.float64),
            )

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="sites"):
            CoalescenceEstimator(
                np.zeros((10, 50), dtype=np.uint8),
                np.arange(60, dtype=np.float64),
            )

    def test_rejects_2d_positions(self):
        with pytest.raises(ValueError, match="1D"):
            CoalescenceEstimator(
                np.zeros((10, 50), dtype=np.uint8),
                np.zeros((50, 2), dtype=np.float64),
            )
