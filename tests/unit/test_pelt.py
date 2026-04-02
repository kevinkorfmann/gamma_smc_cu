"""
Unit tests for the PELT changepoint detection algorithm.

Tests the numpy reference implementation directly, and (when the CUDA module
is available) validates GPU output against the reference.

NOTE on signal strength: With mu=1.25e-8, the expected mutation probability
at a single segregating site is 2*mu*T. For T=10000 gen, p~0.00025 per bp.
To get enough signal, we use large inter-site spacing (1000-5000 bp) or
higher synthetic mutation rates in tests.
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.reference.pelt_numpy import NumpyPELT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_piecewise_signal(segment_rates, segment_lengths, bp_spacing=1000.0,
                          mu=1.25e-8, rng=None):
    """
    Generate a synthetic XOR signal with known piecewise-constant TMRCA.

    Parameters
    ----------
    segment_rates : list of float
        True TMRCA (in generations) for each segment.
    segment_lengths : list of int
        Number of sites in each segment.
    bp_spacing : float
        Physical distance in bp between consecutive segregating sites.
    mu : float
        Mutation rate.
    rng : np.random.Generator or None

    Returns
    -------
    xor_signal : np.ndarray, shape (S,), dtype uint8
    positions  : np.ndarray, shape (S,), dtype float64
    true_tmrca : np.ndarray, shape (S,), dtype float64
    """
    if rng is None:
        rng = np.random.default_rng(12345)

    S = sum(segment_lengths)
    xor_signal = np.zeros(S, dtype=np.uint8)
    true_tmrca = np.zeros(S, dtype=np.float64)

    offset = 0
    for tmrca, length in zip(segment_rates, segment_lengths):
        # Per-site mutation prob accounts for the bp spacing between sites.
        # At a segregating site, the XOR difference probability per inter-site
        # interval of bp_spacing bp is:
        #   p = 1 - exp(-2 * mu * T * bp_spacing)
        # But for PELT, we model count ~ Poisson(2*mu*T * total_bp_length).
        # So generate mutations with rate per site = 2*mu*T for 1-bp spacing,
        # but since our positions are bp_spacing apart, the per-site prob
        # for each segregating site reflecting its "catchment area" is:
        p_mut = 1.0 - np.exp(-2.0 * mu * tmrca * bp_spacing)
        xor_signal[offset:offset + length] = rng.binomial(
            1, min(p_mut, 1.0), size=length
        ).astype(np.uint8)
        true_tmrca[offset:offset + length] = tmrca
        offset += length

    positions = np.arange(S, dtype=np.float64) * bp_spacing

    return xor_signal, positions, true_tmrca


# ---------------------------------------------------------------------------
# Tests: NumpyPELT
# ---------------------------------------------------------------------------

class TestNumpyPELT:
    """Tests for the pure-numpy PELT reference implementation."""

    def test_single_segment_uniform(self):
        """A uniform-rate signal should yield one (or very few) segments."""
        mu = 1.25e-8
        tmrca = 10000.0
        S = 1000
        bp_spacing = 2000.0
        rng = np.random.default_rng(42)

        p_mut = 1.0 - np.exp(-2.0 * mu * tmrca * bp_spacing)
        xor = rng.binomial(1, p_mut, size=S).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * bp_spacing

        pelt = NumpyPELT(mu=mu)
        segments = pelt.detect(xor, positions)

        # Should find very few segments (ideally 1)
        assert len(segments) >= 1
        assert len(segments) <= 10, f"Expected ~1 segment for uniform rate, got {len(segments)}"

        # The overall TMRCA should be in the right ballpark
        total_count = sum(seg[3] for seg in segments)
        total_length = positions[-1] - positions[0]
        overall_tmrca = total_count / (2.0 * mu * total_length) if total_count > 0 else 0
        assert 3000 < overall_tmrca < 30000, f"Overall TMRCA {overall_tmrca} out of range"

    def test_known_changepoints(self):
        """Signal with two distinct rate regimes should produce a changepoint."""
        mu = 1.25e-8
        bp_spacing = 3000.0  # 3 kb between segregating sites

        # Low TMRCA (few mutations) then high TMRCA (many mutations)
        xor, positions, true_tmrca = make_piecewise_signal(
            segment_rates=[2000.0, 100000.0],
            segment_lengths=[500, 500],
            bp_spacing=bp_spacing,
            mu=mu,
            rng=np.random.default_rng(99),
        )

        pelt = NumpyPELT(mu=mu)
        segments = pelt.detect(xor, positions)

        # Should find at least 2 segments
        assert len(segments) >= 2, f"Expected >=2 segments for two-rate signal, got {len(segments)}"

        # The changepoint should be roughly near site 500
        boundaries = [seg[0] for seg in segments] + [segments[-1][1]]
        dists = [abs(b - 500) for b in boundaries]
        closest = min(dists)
        assert closest < 200, (
            f"Expected a changepoint near site 500, closest boundary at distance {closest}"
        )

    def test_no_mutations(self):
        """All-zero signal: PELT should return segments with TMRCA=0."""
        mu = 1.25e-8
        S = 200
        xor = np.zeros(S, dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64) * 1000.0

        pelt = NumpyPELT(mu=mu)
        segments = pelt.detect(xor, positions)

        assert len(segments) >= 1
        for seg in segments:
            assert seg[2] == 0.0, f"Expected TMRCA=0 for zero-mutation segment, got {seg[2]}"
            assert seg[3] == 0, f"Expected count=0, got {seg[3]}"

    def test_all_mutations(self):
        """All-ones signal: should give segments with high TMRCA."""
        mu = 1.25e-8
        S = 200
        xor = np.ones(S, dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64) * 1000.0

        pelt = NumpyPELT(mu=mu)
        segments = pelt.detect(xor, positions)

        assert len(segments) >= 1
        for seg in segments:
            if seg[3] > 0:
                assert seg[2] > 0, "Non-zero mutation count should give positive TMRCA"

    def test_segment_tmrca_values(self):
        """MLE TMRCA should be close to the true value for long segments."""
        mu = 1.25e-8
        true_tmrca_val = 20000.0
        S = 1000
        bp_spacing = 2000.0
        rng = np.random.default_rng(77)

        p_mut = 1.0 - np.exp(-2.0 * mu * true_tmrca_val * bp_spacing)
        xor = rng.binomial(1, p_mut, size=S).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * bp_spacing

        pelt = NumpyPELT(mu=mu)
        segments = pelt.detect(xor, positions)

        # Weighted average TMRCA across segments
        total_length = 0.0
        weighted_tmrca = 0.0
        for seg_start, seg_end, tmrca_mle, n_mut in segments:
            seg_len = positions[seg_end] - positions[seg_start]
            weighted_tmrca += tmrca_mle * seg_len
            total_length += seg_len
        avg_tmrca = weighted_tmrca / total_length if total_length > 0 else 0

        # Should be within a factor of 2 of truth
        assert 0.5 * true_tmrca_val < avg_tmrca < 2.0 * true_tmrca_val, (
            f"Average TMRCA {avg_tmrca:.0f} too far from truth {true_tmrca_val:.0f}"
        )

    def test_three_segments(self):
        """Three distinct TMRCA regimes should yield >=3 segments."""
        mu = 1.25e-8
        bp_spacing = 3000.0

        xor, positions, _ = make_piecewise_signal(
            segment_rates=[2000.0, 100000.0, 5000.0],
            segment_lengths=[500, 500, 500],
            bp_spacing=bp_spacing,
            mu=mu,
            rng=np.random.default_rng(321),
        )

        pelt = NumpyPELT(mu=mu)
        segments = pelt.detect(xor, positions)

        # Should find at least 3 segments (may find more due to noise)
        assert len(segments) >= 3, f"Expected >=3 segments, got {len(segments)}"

    def test_detect_from_prefix(self):
        """detect_from_prefix should give the same result as detect."""
        mu = 1.25e-8
        rng = np.random.default_rng(55)
        S = 500
        xor = rng.binomial(1, 0.3, size=S).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * 1000.0

        pelt = NumpyPELT(mu=mu)

        seg1 = pelt.detect(xor, positions)

        # Build prefix that matches what detect uses internally:
        # detect builds prefix[0]=0, prefix[s] = sum(xor[0:s]) for s=1..S
        # detect_from_prefix expects prefix[s] = cumulative count at index s
        # In detect: cost(a,b) uses prefix[b]-prefix[a] where prefix is length S+1.
        # In detect_from_prefix: same but prefix is length S, and prefix[0]=0,
        # prefix[s] = prefix[s-1] + xor[s].
        # The key difference: detect's prefix includes xor[0] at index 1,
        # but detect_from_prefix's prefix has prefix[0]=0 and adds xor[s] for s>=1.
        # These are equivalent because cost(0, s) uses prefix[s]-prefix[0].
        prefix_for_fn = np.zeros(S, dtype=np.int64)
        for i in range(1, S):
            prefix_for_fn[i] = prefix_for_fn[i - 1] + int(xor[i])

        seg2 = pelt.detect_from_prefix(prefix_for_fn, positions)

        # Should give the same number of segments and same boundaries
        assert len(seg1) == len(seg2), (
            f"detect gave {len(seg1)} segments, detect_from_prefix gave {len(seg2)}"
        )
        for s1, s2 in zip(seg1, seg2):
            assert s1[0] == s2[0], f"Start mismatch: {s1[0]} vs {s2[0]}"
            assert s1[1] == s2[1], f"End mismatch: {s1[1]} vs {s2[1]}"

    def test_penalty_effect(self):
        """Higher penalty should produce fewer or equal segments."""
        mu = 1.25e-8
        bp_spacing = 3000.0

        xor, positions, _ = make_piecewise_signal(
            segment_rates=[2000.0, 80000.0, 5000.0],
            segment_lengths=[500, 500, 500],
            bp_spacing=bp_spacing,
            mu=mu,
            rng=np.random.default_rng(888),
        )

        pelt = NumpyPELT(mu=mu)
        seg_low = pelt.detect(xor, positions, penalty=1.0)
        seg_high = pelt.detect(xor, positions, penalty=100.0)

        assert len(seg_low) >= len(seg_high), (
            f"Low penalty gave {len(seg_low)} segments but high penalty gave {len(seg_high)}"
        )

    def test_segment_coverage(self):
        """Segments should cover all sites without gaps or overlaps."""
        mu = 1.25e-8
        rng = np.random.default_rng(101)
        S = 500
        xor = rng.binomial(1, 0.1, size=S).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * 1000.0

        pelt = NumpyPELT(mu=mu)
        segments = pelt.detect(xor, positions)

        assert len(segments) >= 1

        # First segment starts at 0
        assert segments[0][0] == 0, f"First segment starts at {segments[0][0]}, expected 0"

        # Last segment ends at S-1
        assert segments[-1][1] == S - 1, (
            f"Last segment ends at {segments[-1][1]}, expected {S - 1}"
        )

        # Consecutive segments are contiguous
        for i in range(len(segments) - 1):
            assert segments[i][1] == segments[i + 1][0], (
                f"Gap between segment {i} (end={segments[i][1]}) "
                f"and segment {i+1} (start={segments[i+1][0]})"
            )

    def test_empty_signal(self):
        """Empty input should return empty segments list."""
        pelt = NumpyPELT(mu=1.25e-8)
        segments = pelt.detect(np.array([], dtype=np.uint8),
                               np.array([], dtype=np.float64))
        assert segments == []

    def test_single_site(self):
        """Single-site signal is degenerate; should not crash."""
        pelt = NumpyPELT(mu=1.25e-8)
        # S=1 means only F[0]=0, no loop iterations, no segments from traceback
        segments = pelt.detect(np.array([1], dtype=np.uint8),
                               np.array([0.0], dtype=np.float64))
        # Implementation may return 0 segments for single site (no interval)
        assert isinstance(segments, list)


# ---------------------------------------------------------------------------
# Tests: GPU vs numpy reference (skip if CUDA module not available)
# ---------------------------------------------------------------------------

try:
    import tmrca_cu
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


@pytest.mark.skipif(not HAS_CUDA, reason="tmrca_cu CUDA module not available")
class TestPELTGPU:
    """Tests that compare GPU PELT output against the numpy reference."""

    def test_matches_numpy_reference(self):
        """GPU PELT output should match numpy reference on synthetic data."""
        mu = 1.25e-8
        rng = np.random.default_rng(42)
        S = 2000
        xor = rng.binomial(1, 0.2, size=S).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * 1000.0
        penalty = np.log(S)

        # Numpy reference
        pelt_np = NumpyPELT(mu=mu)
        segments_np = pelt_np.detect(xor, positions, penalty=penalty)

        # Build prefix for GPU (same convention as detect_from_prefix)
        prefix = np.zeros(S, dtype=np.int64)
        for i in range(1, S):
            prefix[i] = prefix[i - 1] + int(xor[i])

        # GPU
        result = tmrca_cu.pelt_changepoint(
            prefix.reshape(1, S), positions, 1, mu, penalty
        )
        n_seg_gpu = result['n_segments'][0]
        gpu_starts = result['seg_starts'][0, :n_seg_gpu]
        gpu_ends = result['seg_ends'][0, :n_seg_gpu]

        # Same number of segments
        assert n_seg_gpu == len(segments_np), (
            f"GPU found {n_seg_gpu} segments, numpy found {len(segments_np)}"
        )

        # Same boundaries
        np_starts = [s[0] for s in segments_np]
        np_ends = [s[1] for s in segments_np]
        np.testing.assert_array_equal(gpu_starts, np_starts)
        np.testing.assert_array_equal(gpu_ends, np_ends)

    def test_gpu_tmrca_values(self):
        """GPU TMRCA MLE values should match numpy reference."""
        mu = 1.25e-8
        rng = np.random.default_rng(77)
        S = 3000
        xor = rng.binomial(1, 0.15, size=S).astype(np.uint8)
        positions = np.arange(S, dtype=np.float64) * 2000.0
        penalty = np.log(S)

        pelt_np = NumpyPELT(mu=mu)
        segments_np = pelt_np.detect(xor, positions, penalty=penalty)

        prefix = np.zeros(S, dtype=np.int64)
        for i in range(1, S):
            prefix[i] = prefix[i - 1] + int(xor[i])

        result = tmrca_cu.pelt_changepoint(
            prefix.reshape(1, S), positions, 1, mu, penalty
        )
        n_seg = result['n_segments'][0]
        gpu_tmrca = result['seg_tmrca'][0, :n_seg]

        np_tmrca = np.array([s[2] for s in segments_np], dtype=np.float32)
        np.testing.assert_allclose(gpu_tmrca, np_tmrca, rtol=1e-3)
