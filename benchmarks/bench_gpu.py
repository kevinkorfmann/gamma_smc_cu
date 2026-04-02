"""
GPU benchmark: timing + accuracy for 1Mb and 10Mb.
Uses run_fb_summary for timing (matches existing Table 1 methodology).
Uses run_fb/run_fwd for accuracy (per-pair results needed for correlation).
Schweiger times hardcoded from prior runs.
Outputs bench_results.json for replotting.
"""
import time, os, sys, json, gc
import numpy as np
import msprime
from scipy.stats import pearsonr

sys.path.insert(0, "/sietch_colab/kkor/tmrca.cu/python")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tmrca_cu import _core

MU, RHO, NE = 1.25e-8, 1e-8, 10000
FF = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"
OUT = "/sietch_colab/kkor/tmrca.cu/benchmarks"

# ── Existing Schweiger times (not re-run) ─────────────────────
SCHWEIGER = {
    (10, 1):   {"t_sw": 6.2, "r_sw": 0.815},
    (20, 1):   {"t_sw": 6.5, "r_sw": 0.790},
    (50, 1):   {"t_sw": 6.4, "r_sw": 0.755},
    (100, 1):  {"t_sw": 6.7, "r_sw": 0.797},
    (200, 1):  {"t_sw": 8.7, "r_sw": 0.800},
    (500, 1):  {"t_sw": 25.1, "r_sw": 0.796},
    (1000, 1): {"t_sw": 89.2, "r_sw": 0.764},
    (10, 10):  {"t_sw": 6.4, "r_sw": 0.820},
    (20, 10):  {"t_sw": 6.8, "r_sw": None},
    (50, 10):  {"t_sw": 8.2, "r_sw": 0.810},
    (100, 10): {"t_sw": 14.8, "r_sw": 0.800},
    (200, 10): {"t_sw": 45.3, "r_sw": 0.790},
    (500, 10): {"t_sw": None, "r_sw": None},
    (1000, 10):{"t_sw": None, "r_sw": None},
}

def simulate(n_hap, seq_len, seed=42):
    ts = msprime.sim_ancestry(samples=n_hap // 2, sequence_length=seq_len,
        recombination_rate=RHO, population_size=NE, random_seed=seed)
    return msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)

def true_t(ts, i, j, positions):
    t = np.empty(len(positions))
    tit = ts.trees(); tree = next(tit)
    for idx, p in enumerate(positions):
        while p >= tree.interval.right: tree = next(tit)
        t[idx] = tree.tmrca(i, j)
    return t

def r_log(x, y):
    lx = np.log(np.maximum(x, 1e-10))
    ly = np.log(np.maximum(y, 1e-10))
    return float(pearsonr(lx, ly)[0])

def bench(fn, reps=5):
    fn()  # warmup
    times = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return min(times)

def run_fb_summary_chunked(ctx, pairs, S, chunk_size=10000):
    site_sum = np.zeros(S, dtype=np.float64); total = 0
    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i:i+chunk_size]
        r = ctx.run_fb_summary(chunk)
        site_sum += r["site_mean"].astype(np.float64) * len(chunk)
        total += len(chunk)
    return site_sum / total

# ── Run ───────────────────────────────────────────────────────
configs = []
for sl_mb in [1, 10]:
    for nh in [10, 20, 50, 100, 200, 500, 1000]:
        configs.append((nh, sl_mb * 1_000_000, sl_mb))

results = []

for nh, sl, sl_mb in configs:
    npp = nh * (nh - 1) // 2
    tag = f"n={nh:>4}, {sl_mb:>3}Mb, {npp:>7} pairs"
    print(f"\n{tag} ...", end=" ", flush=True)

    ts = simulate(nh, sl, seed=42)
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()], dtype=np.float64)
    n, S = G.shape
    all_pairs = [(i, j) for i in range(n) for j in range(i)]
    print(f"S={S}", end=" ", flush=True)

    ctx = _core.FlowContext(G, pos, float(NE), MU, RHO, FF, 0)
    chunk_sz = 10000

    # ── Timing: run_fb_summary (matches Table 1 methodology) ─
    if npp <= 20000:
        t_fb = bench(lambda: ctx.run_fb_summary(all_pairs), reps=5)
    else:
        t_fb = bench(lambda: run_fb_summary_chunked(ctx, all_pairs, S, chunk_sz), reps=5)
    print(f"fb={t_fb:.4f}s", end=" ", flush=True)

    # ── Timing: run_fwd (chunked for large n) ────────────────
    if npp <= 20000:
        t_fwd = bench(lambda: ctx.run_fwd(all_pairs, True), reps=5)
    else:
        def run_fwd_chunked():
            for i in range(0, npp, chunk_sz):
                ctx.run_fwd(all_pairs[i:i+chunk_sz], True)
        t_fwd = bench(run_fwd_chunked, reps=5)
    print(f"fwd={t_fwd:.4f}s", end=" ", flush=True)

    # ── Accuracy: sample 40 test pairs ────────────────────────
    rng = np.random.RandomState(42)
    n_test = min(40, npp)
    test_set = set()
    while len(test_set) < n_test:
        a, b = sorted(rng.choice(n, 2, replace=False))
        test_set.add((int(a), int(b)))
    test_pairs = sorted(test_set)

    fb_out = _core.gamma_smc_flow_cached_fb(
        G, pos, test_pairs, float(NE), MU, RHO, FF, True, 0)["mean"]
    r_fb_list = [r_log(true_t(ts, p[0], p[1], pos), fb_out[:, i])
                 for i, p in enumerate(test_pairs)]
    r_fb = float(np.mean(r_fb_list))

    fwd_out = _core.gamma_smc_flow_cached_fwd(
        G, pos, test_pairs, float(NE), MU, RHO, FF, True, 0)["mean"]
    r_fwd_list = [r_log(true_t(ts, p[0], p[1], pos), fwd_out[:, i])
                  for i, p in enumerate(test_pairs)]
    r_fwd = float(np.mean(r_fwd_list))

    print(f"r_fb={r_fb:.3f} r_fwd={r_fwd:.3f}", end=" ", flush=True)

    sw = SCHWEIGER.get((nh, sl_mb), {})
    t_sw = sw.get("t_sw"); r_sw = sw.get("r_sw")
    speedup = t_sw / t_fb if t_sw and t_fb > 0 else None
    if speedup:
        print(f"speedup={speedup:.0f}x", end="", flush=True)

    row = {
        "n": nh, "seq_len_mb": sl_mb, "seq_len": sl,
        "n_pairs": npp, "n_sites": S,
        "t_fb": round(t_fb, 4), "t_fwd": round(t_fwd, 4),
        "r_fb": round(r_fb, 4), "r_fwd": round(r_fwd, 4),
        "r_fb_per_pair": [round(x, 4) for x in r_fb_list],
        "r_fwd_per_pair": [round(x, 4) for x in r_fwd_list],
        "t_sw": t_sw, "r_sw": r_sw,
        "speedup": round(speedup, 1) if speedup else None,
    }
    results.append(row)
    del ctx, G, ts; gc.collect()

out_path = os.path.join(OUT, "bench_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\n{'='*105}")
print(f"{'n':>5} {'Mb':>4} {'pairs':>8} {'S':>7} {'t_fb(s)':>9} {'t_fwd(s)':>9} "
      f"{'t_sw(s)':>9} {'speedup':>8} {'r_fb':>7} {'r_fwd':>7} {'r_sw':>7}")
print("-"*105)
for r in results:
    t_sw_s = f"{r['t_sw']:.1f}" if r['t_sw'] else "---"
    r_sw_s = f"{r['r_sw']:.3f}" if r['r_sw'] else "---"
    sp_s = f"{r['speedup']:.0f}x" if r['speedup'] else "---"
    print(f"{r['n']:>5} {r['seq_len_mb']:>4} {r['n_pairs']:>8} {r['n_sites']:>7} "
          f"{r['t_fb']:>9.4f} {r['t_fwd']:>9.4f} {t_sw_s:>9} {sp_s:>8} "
          f"{r['r_fb']:>7.3f} {r['r_fwd']:>7.3f} {r_sw_s:>7}")
print(f"\nSaved to {out_path}")
