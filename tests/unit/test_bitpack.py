import numpy as np
import pytest
import gamma_smc_cu


class TestBitpacking:
    """Verify bitpacking preserves genotype information exactly."""

    def test_roundtrip_small(self, small_simulation):
        """Pack and unpack, verify identity."""
        _, G, _ = small_simulation
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, G.shape[0], G.shape[1])
        np.testing.assert_array_equal(G, unpacked)

    def test_non_multiple_of_64(self):
        """Sites count not divisible by 64 — verify padding is zero-filled."""
        G = np.random.RandomState(42).randint(0, 2, size=(10, 100)).astype(np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 10, 100)
        np.testing.assert_array_equal(G, unpacked)

    def test_all_zeros(self):
        """Monomorphic zero matrix."""
        G = np.zeros((50, 1000), dtype=np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 50, 1000)
        np.testing.assert_array_equal(G, unpacked)

    def test_all_ones(self):
        """Monomorphic one matrix."""
        G = np.ones((50, 1000), dtype=np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 50, 1000)
        np.testing.assert_array_equal(G, unpacked)

    def test_single_site(self):
        """Edge case: S=1."""
        G = np.array([[0], [1], [1], [0]], dtype=np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 4, 1)
        np.testing.assert_array_equal(G, unpacked)

    def test_single_haplotype(self):
        """Edge case: n=1."""
        G = np.random.RandomState(43).randint(0, 2, size=(1, 500)).astype(np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 1, 500)
        np.testing.assert_array_equal(G, unpacked)

    def test_exactly_64_sites(self):
        """Exactly one word per haplotype."""
        G = np.random.RandomState(44).randint(0, 2, size=(10, 64)).astype(np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        assert packed.shape == (10, 1)
        unpacked = gamma_smc_cu.unpack(packed, 10, 64)
        np.testing.assert_array_equal(G, unpacked)

    def test_exactly_128_sites(self):
        """Exactly two words per haplotype."""
        G = np.random.RandomState(45).randint(0, 2, size=(5, 128)).astype(np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        assert packed.shape == (5, 2)
        unpacked = gamma_smc_cu.unpack(packed, 5, 128)
        np.testing.assert_array_equal(G, unpacked)
