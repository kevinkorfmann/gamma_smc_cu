import numpy as np
import pytest
import tmrca_cu


class TestPrefixScan:
    """Verify pairwise XOR prefix scan against numpy reference."""

    def test_prefix_scan_correctness(self, small_simulation):
        """Compare CUDA prefix scan to numpy cumsum of XOR."""
        _, G, _ = small_simulation
        pairs = [(0, 1), (2, 3), (0, 10)]

        for i, j in pairs:
            xor = np.bitwise_xor(G[i], G[j]).astype(np.int64)
            expected_prefix = np.cumsum(xor)

            cuda_prefix = tmrca_cu.pairwise_prefix_scan(G, [(i, j)])[0]
            np.testing.assert_array_equal(
                cuda_prefix, expected_prefix,
                err_msg=f"Mismatch for pair ({i}, {j})"
            )

    def test_multiple_pairs_at_once(self, small_simulation):
        """Batch of pairs matches individual results."""
        _, G, _ = small_simulation
        pairs = [(0, 1), (2, 3), (5, 10), (0, 19)]

        batch_result = tmrca_cu.pairwise_prefix_scan(G, pairs)

        for idx, (i, j) in enumerate(pairs):
            xor = np.bitwise_xor(G[i], G[j]).astype(np.int64)
            expected = np.cumsum(xor)
            np.testing.assert_array_equal(batch_result[idx], expected)

    def test_symmetry(self, small_simulation):
        """prefix_scan(i,j) == prefix_scan(j,i) since XOR is symmetric."""
        _, G, _ = small_simulation
        p_ij = tmrca_cu.pairwise_prefix_scan(G, [(0, 5)])[0]
        p_ji = tmrca_cu.pairwise_prefix_scan(G, [(5, 0)])[0]
        np.testing.assert_array_equal(p_ij, p_ji)

    def test_self_pair_is_zero(self, small_simulation):
        """XOR of haplotype with itself is zero everywhere."""
        _, G, _ = small_simulation
        prefix = tmrca_cu.pairwise_prefix_scan(G, [(3, 3)])[0]
        np.testing.assert_array_equal(prefix, np.zeros_like(prefix))

    def test_final_value_equals_hamming(self, small_simulation):
        """Last element of prefix = total Hamming distance."""
        _, G, _ = small_simulation
        for i, j in [(0, 1), (3, 7)]:
            prefix = tmrca_cu.pairwise_prefix_scan(G, [(i, j)])[0]
            hamming = np.sum(G[i] != G[j])
            assert prefix[-1] == hamming
