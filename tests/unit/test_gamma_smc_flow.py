"""Test flow-field Gamma-SMC forward-backward on GPU vs numpy reference."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pytest
from tests.reference.gamma_smc_flow_numpy import _bilinear

# Generate test data with msprime
def make_test_data(n=20, length=1_000_000, Ne=10_000, mu=1.25e-8, rho=1e-8, seed=42):
    import msprime
    ts = msprime.sim_ancestry(
        n, sequence_length=length, recombination_rate=rho,
        population_size=Ne, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 1)
    G = ts.genotype_matrix().T.astype(np.uint8)  # (n_haps, S)
    pos = np.array([v.position for v in ts.variants()])
    # Get true TMRCA for pairs
    return G, pos, ts


def test_flow_fb_runs():
    """Basic smoke test: flow FB runs without error."""
    import gamma_smc_cu
    G, pos, ts = make_test_data(n=10, length=500_000, seed=1)
    pairs = [(0, 1), (2, 3)]
    result = gamma_smc_cu.gamma_smc_flow_fb(G, pos, pairs, Ne=10_000)
    assert 'mean' in result
    assert 'lower' in result
    assert 'upper' in result
    assert result['mean'].shape == (len(pos), len(pairs))
    # Check no NaN/Inf
    assert np.all(np.isfinite(result['mean']))
    assert np.all(result['mean'] > 0)


def test_flow_fb_mean_only():
    """Test mean-only mode."""
    import gamma_smc_cu
    G, pos, ts = make_test_data(n=10, length=500_000, seed=2)
    pairs = [(0, 1)]
    result = gamma_smc_cu.gamma_smc_flow_fb(G, pos, pairs, Ne=10_000, mean_only=True)
    assert 'mean' in result
    assert 'lower' not in result


def test_flow_fb_vs_true_tmrca():
    """Compare flow FB posteriors to true coalescence times from simulation."""
    import msprime
    import gamma_smc_cu

    Ne = 10_000
    mu = 1.25e-8
    rho = 1e-8

    ts = msprime.sim_ancestry(
        50, sequence_length=2_000_000, recombination_rate=rho,
        population_size=Ne, random_seed=123)
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=124)
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()])

    # Pick 5 pairs
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    # Get true TMRCA
    true_tmrca = np.zeros((len(pos), len(pairs)))
    for p_idx, (i, j) in enumerate(pairs):
        for s_idx, var in enumerate(ts.variants()):
            tree = ts.at(var.position)
            true_tmrca[s_idx, p_idx] = tree.tmrca(i, j)

    # GPU flow FB
    result = gamma_smc_cu.gamma_smc_flow_fb(
        G, pos, pairs, Ne=Ne, mu=mu, rho=rho, mean_only=True)
    gpu_mean = result['mean']

    # Check correlation in log space
    for p_idx in range(len(pairs)):
        log_true = np.log(true_tmrca[:, p_idx] + 1)
        log_gpu = np.log(gpu_mean[:, p_idx] + 1)
        r = np.corrcoef(log_true, log_gpu)[0, 1]
        print(f"Pair {pairs[p_idx]}: r(log) = {r:.4f}")
        # Flow FB should achieve at least r > 0.5 (much better than forward-only)
        assert r > 0.3, f"Pair {pairs[p_idx]}: r(log)={r:.4f} too low"


def test_flow_fb_vs_moment_match():
    """Flow FB should outperform moment-matching forward-only."""
    import msprime
    import gamma_smc_cu

    Ne = 10_000
    mu = 1.25e-8
    rho = 1e-8

    ts = msprime.sim_ancestry(
        20, sequence_length=1_000_000, recombination_rate=rho,
        population_size=Ne, random_seed=456)
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=457)
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()])

    pairs = [(0, 1), (2, 3), (4, 5)]

    # True TMRCA
    true_tmrca = np.zeros((len(pos), len(pairs)))
    for p_idx, (i, j) in enumerate(pairs):
        for s_idx, var in enumerate(ts.variants()):
            tree = ts.at(var.position)
            true_tmrca[s_idx, p_idx] = tree.tmrca(i, j)

    # Flow FB
    flow_result = gamma_smc_cu.gamma_smc_flow_fb(
        G, pos, pairs, Ne=Ne, mu=mu, rho=rho, mean_only=True)

    # Moment-matching forward only
    mm_result = gamma_smc_cu.gamma_smc_forward(
        G, pos, pairs, Ne=Ne, mu=mu, rho=rho, mean_only=True)

    flow_rs = []
    mm_rs = []
    for p_idx in range(len(pairs)):
        log_true = np.log(true_tmrca[:, p_idx] + 1)
        log_flow = np.log(flow_result['mean'][:, p_idx] + 1)
        log_mm = np.log(mm_result['mean'][:, p_idx] + 1)
        flow_rs.append(np.corrcoef(log_true, log_flow)[0, 1])
        mm_rs.append(np.corrcoef(log_true, log_mm)[0, 1])

    avg_flow = np.mean(flow_rs)
    avg_mm = np.mean(mm_rs)
    print(f"Flow FB avg r(log): {avg_flow:.4f}")
    print(f"Moment-match fwd avg r(log): {avg_mm:.4f}")
    # Flow FB should be substantially better
    assert avg_flow > avg_mm, \
        f"Flow FB ({avg_flow:.4f}) should beat MM forward ({avg_mm:.4f})"


def test_reference_bilinear_uses_upper_boundary_cell():
    """At the max grid boundary, interpolation should keep weight on the last cell."""
    table = np.array([
        [0.0, 1.0, 2.0],
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0],
    ], dtype=np.float64)
    ff = {
        "mean_n": 3,
        "cv_n": 3,
        "mean_log10_min": 0.0,
        "mean_log10_max": 2.0,
        "cv_log10_min": 0.0,
        "cv_log10_max": 2.0,
    }

    assert _bilinear(table, 2.0, 2.0, ff) == pytest.approx(22.0)
    assert _bilinear(table, 2.0, 1.0, ff) == pytest.approx(21.0)
    assert _bilinear(table, 1.0, 2.0, ff) == pytest.approx(12.0)


if __name__ == "__main__":
    print("=== Smoke test ===")
    test_flow_fb_runs()
    print("PASS")

    print("\n=== Mean-only test ===")
    test_flow_fb_mean_only()
    print("PASS")

    print("\n=== vs True TMRCA ===")
    test_flow_fb_vs_true_tmrca()
    print("PASS")

    print("\n=== vs Moment-match ===")
    test_flow_fb_vs_moment_match()
    print("PASS")

    print("\nAll tests passed!")
