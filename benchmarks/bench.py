"""
Light benchmarking of tmrca_cu GPU kernels on A100.
Measures wall-clock time for key operations across varying problem sizes.
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tmrca_cu

# ── helpers ──────────────────────────────────────────────────────────────
def make_data(n, S, seed=42):
    rng = np.random.RandomState(seed)
    G = rng.randint(0, 2, size=(n, S)).astype(np.uint8)
    positions = np.arange(S, dtype=np.float64) * 100.0
    return G, positions


def bench(fn, warmup=2, repeats=5):
    """Time fn() with warmup, return median seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times)


# ── 1. Bitpack + prefix scan throughput vs S ─────────────────────────────
def bench_prefix_scan():
    n = 100
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    results = []
    for S in sizes:
        G, pos = make_data(n, S)
        t = bench(lambda: tmrca_cu.pairwise_prefix_scan(G, pairs))
        results.append((S, t))
        print(f"  prefix_scan  S={S:>7,}  {t*1e3:8.2f} ms")
    return results


# ── 2. SFS throughput vs n ───────────────────────────────────────────────
def bench_sfs():
    S = 50_000
    ns = [20, 50, 100, 200, 500, 1000]
    results = []
    for n in ns:
        G, _ = make_data(n, S)
        t = bench(lambda: tmrca_cu.compute_sfs(G))
        results.append((n, t))
        print(f"  sfs          n={n:>5,}  {t*1e3:8.2f} ms")
    return results


# ── 3. HMM posterior vs S (single pair) ─────────────────────────────────
def bench_hmm_single():
    n = 20
    sizes = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
    results = []
    for S in sizes:
        G, pos = make_data(n, S)
        t = bench(
            lambda: tmrca_cu.hmm_posterior(G, pos, (0, 1), K=32, Ne=10000.0,
                                           mu=1.25e-8, rho=1e-8),
            warmup=1, repeats=3,
        )
        results.append((S, t))
        print(f"  hmm_single   S={S:>7,}  {t*1e3:8.2f} ms")
    return results


# ── 4. HMM batched: throughput vs n_pairs ────────────────────────────────
def bench_hmm_batched():
    n, S = 200, 10_000
    G, pos = make_data(n, S)
    pair_counts = [1, 10, 50, 100, 500, 1000]
    results = []
    for np_ in pair_counts:
        rng = np.random.RandomState(0)
        pair_set = set()
        while len(pair_set) < np_:
            a, b = sorted(rng.choice(n, 2, replace=False))
            pair_set.add((int(a), int(b)))
        pairs = sorted(pair_set)[:np_]
        t = bench(
            lambda: tmrca_cu.hmm_posterior_batched(
                G, pos, pairs, K=32, Ne=10000.0, mu=1.25e-8, rho=1e-8),
            warmup=1, repeats=3,
        )
        results.append((np_, t))
        tput = np_ * S / t / 1e6
        print(f"  hmm_batched  pairs={np_:>5,}  {t*1e3:8.2f} ms  ({tput:.1f} M site·pairs/s)")
    return results


# ── 5. Windowed divergence vs window size ────────────────────────────────
def bench_windowed_div():
    n, S = 100, 100_000
    G, pos = make_data(n, S)
    pairs = [(i, i + 1) for i in range(0, 20, 2)]
    windows = [10, 50, 100, 500, 1000, 5000]
    results = []
    for w in windows:
        t = bench(lambda: tmrca_cu.windowed_divergence(G, pairs, w))
        results.append((w, t))
        print(f"  windowed_div W={w:>5,}  {t*1e3:8.2f} ms")
    return results


# ── 6. Bitpack throughput vs n ───────────────────────────────────────────
def bench_bitpack():
    S = 100_000
    ns = [20, 50, 100, 500, 1000, 2000]
    results = []
    for n in ns:
        G, _ = make_data(n, S)
        t = bench(lambda: tmrca_cu.bitpack(G))
        results.append((n, t))
        gbps = n * S / t / 1e9
        print(f"  bitpack      n={n:>5,}  {t*1e3:8.2f} ms  ({gbps:.1f} Gbits/s)")
    return results


# ── run all ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("tmrca_cu benchmarks — NVIDIA A100 80GB, CUDA_VISIBLE_DEVICES=0")
    print("=" * 60)

    print("\n[1/6] Prefix scan vs sequence length (4 pairs, n=100)")
    r_prefix = bench_prefix_scan()

    print("\n[2/6] SFS vs sample size (S=50k)")
    r_sfs = bench_sfs()

    print("\n[3/6] HMM posterior (single pair) vs sequence length")
    r_hmm = bench_hmm_single()

    print("\n[4/6] HMM posterior (batched) vs number of pairs (S=10k)")
    r_batch = bench_hmm_batched()

    print("\n[5/6] Windowed divergence vs window size (10 pairs, S=100k)")
    r_wdiv = bench_windowed_div()

    print("\n[6/6] Bitpack throughput vs sample size (S=100k)")
    r_bitpack = bench_bitpack()

    # ── save for plotting ────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(__file__), "bench_results.npz")
    np.savez(
        out,
        prefix=np.array(r_prefix),
        sfs=np.array(r_sfs),
        hmm_single=np.array(r_hmm),
        hmm_batched=np.array(r_batch),
        windowed_div=np.array(r_wdiv),
        bitpack=np.array(r_bitpack),
    )
    print(f"\nResults saved to {out}")
