"""
Multi-GPU scaling benchmark for tmrca.cu.
"""
import time, os, sys, json
import numpy as np
import msprime
import concurrent.futures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, "/sietch_colab/kkor/tmrca.cu/python")
from tmrca_cu import _core
from tmrca_cu.multigpu import MultiGPUFlowContext

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

MU, RHO, NE = 1.25e-8, 1e-8, 10000
FF = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"
OUT = "/sietch_colab/kkor/tmrca.cu/benchmarks"
N_GPUS_AVAIL = _core.get_device_count()
print(f"Available GPUs: {N_GPUS_AVAIL}", flush=True)

def simulate(n_hap, seq_len, seed=42):
    ts = msprime.sim_ancestry(samples=n_hap // 2, sequence_length=seq_len,
        recombination_rate=RHO, population_size=NE, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()], dtype=np.float64)
    return G, pos

def all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i)]

configs = [
    (100, 1_000_000, "n=100, 1Mb"),
    (200, 1_000_000, "n=200, 1Mb"),
    (500, 1_000_000, "n=500, 1Mb"),
    (1000, 1_000_000, "n=1000, 1Mb"),
    (200, 5_000_000, "n=200, 5Mb"),
    (500, 5_000_000, "n=500, 5Mb"),
]

n_repeats = 5
gpu_counts = [1, 2, 3]
results = []

def run_on_ctx(ctx, pairs):
    return ctx.run_fb(pairs, mean_only=True)

for n_hap, seq_len, label in configs:
    n_pairs = n_hap * (n_hap - 1) // 2
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {label} ({n_pairs:,} pairs)", flush=True)
    print(f"{'='*60}", flush=True)

    G, pos = simulate(n_hap, seq_len)
    pairs = all_pairs(n_hap)

    # Create contexts for all GPUs upfront
    contexts = []
    for gid in range(min(3, N_GPUS_AVAIL)):
        _core.set_device(gid)
        ctx = _core.FlowContext(G, pos, float(NE), MU, RHO, FF, 0)
        ctx.run_fb(pairs[:min(100, n_pairs)], mean_only=True)  # warmup
        contexts.append(ctx)

    for n_gpu in gpu_counts:
        if n_gpu > N_GPUS_AVAIL:
            continue

        # Split pairs across GPUs
        chunk = (n_pairs + n_gpu - 1) // n_gpu
        gpu_pairs = []
        for i in range(n_gpu):
            s = i * chunk
            e = min(s + chunk, n_pairs)
            gpu_pairs.append(pairs[s:e] if s < e else [])

        times = []
        for rep in range(n_repeats):
            if n_gpu == 1:
                t0 = time.perf_counter()
                contexts[0].run_fb(pairs, mean_only=True)
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpu) as ex:
                    futures = [ex.submit(run_on_ctx, contexts[i], gpu_pairs[i])
                               for i in range(n_gpu)]
                    for f in futures:
                        f.result()
                t1 = time.perf_counter()
            times.append(t1 - t0)

        best = min(times)
        median = np.median(times)
        print(f"  {n_gpu} GPU(s): best={best:.4f}s  median={median:.4f}s", flush=True)

        results.append({
            "label": label, "n_hap": n_hap, "seq_len": seq_len,
            "n_pairs": n_pairs, "n_gpus": n_gpu,
            "best_s": best, "median_s": median, "times": times,
        })

    del contexts

# Print results table
print(f"\n\n{'='*80}")
print("MULTI-GPU BENCHMARK RESULTS")
print(f"{'='*80}")
print(f"{'Config':<20} {'Pairs':>10} {'1-GPU (s)':>10} {'2-GPU (s)':>10} {'3-GPU (s)':>10} {'2x speedup':>12} {'3x speedup':>12}")
print("-" * 80)

from itertools import groupby
for label, grp in groupby(results, key=lambda r: r["label"]):
    grp = list(grp)
    row = {"label": label, "n_pairs": grp[0]["n_pairs"]}
    for r in grp:
        row[f"{r['n_gpus']}gpu"] = r["best_s"]
    t1 = row.get("1gpu", float("nan"))
    t2 = row.get("2gpu", float("nan"))
    t3 = row.get("3gpu", float("nan"))
    sp2 = t1 / t2 if t2 > 0 else float("nan")
    sp3 = t1 / t3 if t3 > 0 else float("nan")
    print(f"{label:<20} {row['n_pairs']:>10,} {t1:>10.4f} {t2:>10.4f} {t3:>10.4f} {sp2:>11.2f}x {sp3:>11.2f}x")

# Save JSON
with open(os.path.join(OUT, "multigpu_bench.json"), "w") as f:
    json.dump(results, f, indent=2)

# Figure
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(configs)))

ax = axes[0]
for idx, (n_hap, seq_len, label) in enumerate(configs):
    tbg = {}
    for r in results:
        if r["label"] == label: tbg[r["n_gpus"]] = r["best_s"]
    gpus = sorted(tbg.keys())
    ax.plot(gpus, [tbg[g] for g in gpus], "o-", color=colors[idx], markersize=3, label=label)
ax.set_xlabel("Number of GPUs")
ax.set_ylabel("Wall-clock time (s)")
ax.set_xticks([1, 2, 3])
ax.set_yscale("log")
ax.legend(fontsize=5, loc="upper right")
ax.set_title("a", fontweight="bold", loc="left", fontsize=8)

ax = axes[1]
for idx, (n_hap, seq_len, label) in enumerate(configs):
    tbg = {}
    for r in results:
        if r["label"] == label: tbg[r["n_gpus"]] = r["best_s"]
    if 1 not in tbg: continue
    t1 = tbg[1]
    gpus = sorted(tbg.keys())
    ax.plot(gpus, [t1/tbg[g] for g in gpus], "o-", color=colors[idx], markersize=3, label=label)
ax.plot([1, 2, 3], [1, 2, 3], "k--", linewidth=0.5, alpha=0.4, label="ideal")
ax.set_xlabel("Number of GPUs")
ax.set_ylabel("Speedup over 1 GPU")
ax.set_xticks([1, 2, 3])
ax.set_yticks([1, 2, 3])
ax.set_ylim(0.8, 3.5)
ax.legend(fontsize=5, loc="upper left")
ax.set_title("b", fontweight="bold", loc="left", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_multigpu.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig7_multigpu.png"), bbox_inches="tight", dpi=300)
print(f"\nFigure saved to {OUT}/fig7_multigpu.{{pdf,png}}")
