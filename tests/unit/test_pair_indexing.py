import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'include'))


def pair_to_ij(p):
    """Python equivalent of the C pair_to_ij."""
    import math
    r = (1.0 + math.sqrt(1.0 + 8.0 * p)) / 2.0
    i = int(r)
    if i * (i - 1) // 2 > p:
        i -= 1
    j = p - i * (i - 1) // 2
    return i, j


def ij_to_pair(i, j):
    """Python equivalent of the C ij_to_pair."""
    if i < j:
        i, j = j, i
    return i * (i - 1) // 2 + j


class TestPairIndexing:
    """Test pair <-> (i,j) index mapping."""

    def test_roundtrip(self):
        """ij_to_pair -> pair_to_ij roundtrip for small n."""
        for n in [5, 10, 50]:
            p = 0
            for i in range(1, n):
                for j in range(i):
                    assert ij_to_pair(i, j) == p
                    ii, jj = pair_to_ij(p)
                    assert (ii, jj) == (i, j), f"pair {p} -> ({ii},{jj}) != ({i},{j})"
                    p += 1

    def test_symmetry(self):
        """ij_to_pair(i,j) == ij_to_pair(j,i)."""
        for i in range(10):
            for j in range(i):
                assert ij_to_pair(i, j) == ij_to_pair(j, i)

    def test_total_pairs(self):
        """Total number of pairs is n*(n-1)/2."""
        for n in [2, 10, 100]:
            max_p = ij_to_pair(n - 1, n - 2)
            assert max_p == n * (n - 1) // 2 - 1
