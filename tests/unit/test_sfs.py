import numpy as np
import pytest
import gamma_smc_cu
from tests.reference.sfs_numpy import compute_sfs_numpy


class TestSFS:
    """Verify SFS computation against numpy reference and tskit."""

    def test_sfs_matches_numpy(self, small_simulation):
        _, G, _ = small_simulation
        cuda_sfs = np.array(gamma_smc_cu.compute_sfs(G))
        np_sfs = compute_sfs_numpy(G)
        np.testing.assert_array_equal(cuda_sfs, np_sfs)

    def test_sfs_matches_tskit(self, small_simulation):
        ts, G, _ = small_simulation
        cuda_sfs = np.array(gamma_smc_cu.compute_sfs(G))
        tskit_afs = ts.allele_frequency_spectrum(
            polarised=True, span_normalise=False
        )
        # tskit AFS has length n+1 and counts site spans (float)
        # Our SFS counts discrete sites
        # For simple simulations these should match
        np.testing.assert_array_equal(cuda_sfs, tskit_afs.astype(int))

    def test_sfs_sums_to_num_sites(self, small_simulation):
        _, G, _ = small_simulation
        sfs = np.array(gamma_smc_cu.compute_sfs(G))
        n = G.shape[0]
        # Entries 1..n-1 should sum to total segregating sites
        assert sfs[1:n].sum() == G.shape[1]

    def test_sfs_monomorphic_input(self):
        """All-zero matrix should give empty SFS (all sites fixed at 0)."""
        G = np.zeros((20, 100), dtype=np.uint8)
        sfs = np.array(gamma_smc_cu.compute_sfs(G))
        # All sites have allele count 0
        assert sfs[0] == 100
        assert sfs[1:].sum() == 0

    def test_sfs_all_ones(self):
        """All-ones matrix: all sites fixed at n."""
        G = np.ones((20, 100), dtype=np.uint8)
        sfs = np.array(gamma_smc_cu.compute_sfs(G))
        assert sfs[20] == 100
        assert sfs[:20].sum() == 0
