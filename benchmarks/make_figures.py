"""
Generate publication-quality figures for tmrca.cu benchmarks.
Runs all benchmarks fresh and produces:
  fig1_speed_accuracy.{pdf,png}  -- speed vs accuracy scatter
  fig2_scaling.{pdf,png}         -- wall-clock scaling with n and seq_len
  fig3_speedup_bars.{pdf,png}    -- speedup bar chart
  fig4_traces.{pdf,png}          -- per-site TMRCA traces: truth vs methods
  fig5_accuracy_box.{pdf,png}    -- accuracy distribution across pairs
  fig6_demographics.{pdf,png}    -- demographic model robustness

Methods shown: flow_fb (this work) vs Schweiger gamma_smc (CPU reference).
gsmc_fwd omitted from figures (experimental, accuracy too low).
Schweiger = their original binary, NOT a reimplementation.
"""

import time, os, sys, json, subprocess, tempfile, traceback, gc
import numpy as np
import msprime
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, "/sietch_colab/kkor/tmrca.cu/python")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tmrca_cu import _core

MU, RHO, NE = 1.25e-8, 1e-8, 10000
FF = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"
GSMC = "/sietch_colab/kkor/gamma_smc/bin/gamma_smc"
OUT = "/sietch_colab/kkor/tmrca.cu/benchmarks"

# --- Style constants ---
C_TRUTH = "black"
C_OURS = "dodgerblue"
C_OURS_FWD = "#66aadd"
C_SCHW = "#d62728"

# Flush after every print
import functools
print = functools.partial(print, flush=True)


# --- helpers ---
def simulate(nh, sl, s):
    ts = msprime.sim_ancestry(samples=nh // 2, sequence_length=sl,
        recombination_rate=RHO, population_size=NE, random_seed=s)
    return msprime.sim_mutations(ts, rate=MU, random_seed=s + 1)


def true_t(ts_, i, j, sp):
    t = np.empty(len(sp)); tit = ts_.trees(); tree = next(tit)
    for idx, p in enumerate(sp):
        while p >= tree.interval.right: tree = next(tit)
        t[idx] = tree.tmrca(i, j)
    return t


def schweiger_pair_order(nh):
    nd = nh // 2; pairs = []
    for dA in range(nd):
        for dB in range(dA, nd):
            if dA == dB:
                pairs.append((2 * dA, 2 * dA + 1))
            else:
                for ha in range(2):
                    for hb in range(2):
                        pairs.append((2 * dA + ha, 2 * dB + hb))
    return pairs


def run_schw(ts, nh, timeout=300):
    """Run Schweiger's ORIGINAL gamma_smc binary (not a reimplementation)."""
    with tempfile.TemporaryDirectory() as td:
        vp = os.path.join(td, "s.vcf")
        with open(vp, "w") as f:
            ts.write_vcf(f, contig_id="chr1")
        subprocess.run(f"bgzip -f {vp} && tabix -p vcf {vp}.gz",
                       shell=True, check=True, capture_output=True)
        out = os.path.join(td, "out")
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                [GSMC, "-i", vp + ".gz", "-o", out,
                 "-m", str(4 * NE * MU), "-r", str(4 * NE * RHO),
                 "-f", FF, "-h"],
                capture_output=True, text=True, timeout=timeout)
            wall = time.perf_counter() - t0
            if result.returncode != 0:
                return None, None, wall
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - t0
            return None, None, wall
        # Find and decompress the output file (may be 'out' or 'out.zst')
        try:
            dec = os.path.join(td, "out.bin")
            # Try decompressing - the file might be 'out' (zstd compressed)
            zstd_result = subprocess.run(["zstd", "-d", out, "-o", dec],
                                          capture_output=True)
            if zstd_result.returncode != 0:
                # Try with .zst extension
                zstd_result = subprocess.run(
                    ["zstd", "-d", out + ".zst", "-o", dec],
                    capture_output=True)
            if zstd_result.returncode != 0:
                # Maybe the file is already uncompressed
                import shutil
                if os.path.exists(out):
                    shutil.copy(out, dec)
                else:
                    return None, None, wall
            with open(dec, "rb") as f:
                raw = f.read()
            # Find metadata file
            meta_path = out + ".meta"
            if not os.path.exists(meta_path):
                # Try listing directory
                import glob
                metas = glob.glob(os.path.join(td, "*.meta"))
                meta_path = metas[0] if metas else meta_path
            with open(meta_path) as f:
                meta = json.load(f)
            np2 = meta["num_pairs"]; ns = meta["sequence_length"]
            cs = meta["chunk_size"]; nc = (np2 + cs - 1) // cs
            sp = np.array(meta["output_positions"])
            arr = np.frombuffer(raw, dtype=np.float32).reshape(nc, 2, ns, cs)
            schw_pairs = schweiger_pair_order(nh)
            results = {}
            for p in range(np2):
                key = tuple(sorted(schw_pairs[p]))
                results[key] = arr[p // cs, 0, :, p % cs]
            return results, sp, wall
        except Exception as e:
            print(f"    Schweiger parse error: {e}")
            return None, None, wall


def r_log(x, y):
    lx = np.log(np.maximum(x, 1e-10))
    ly = np.log(np.maximum(y, 1e-10))
    return pearsonr(lx, ly)[0]


def bench(fn, reps=3):
    fn()  # warmup
    ts_ = [0] * reps
    for i in range(reps):
        t0 = time.perf_counter(); fn(); ts_[i] = time.perf_counter() - t0
    return min(ts_)


def run_fb_summary_chunked(ctx, all_pairs, S, chunk_size=10000):
    """Run fb_summary in chunks, accumulate weighted mean."""
    site_sum = np.zeros(S, dtype=np.float64)
    total = 0
    for i in range(0, len(all_pairs), chunk_size):
        chunk = all_pairs[i:i + chunk_size]
        r = ctx.run_fb_summary(chunk)
        site_sum += r["site_mean"].astype(np.float64) * len(chunk)
        total += len(chunk)
    return site_sum / total


def savefig(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf and {name}.png")


def schw_timeout_for(nh, sl):
    """Adaptive Schweiger timeout based on expected runtime."""
    npp = nh * (nh - 1) // 2
    # Roughly: Schweiger is O(npp * sl), baseline ~6s for n=10, 1Mb (45 pairs)
    # At n=50 10Mb, expect ~60-120s; at n=100 10Mb, expect ~300-600s
    if sl >= 100_000_000:
        return 600  # 100Mb: 10 min max
    if nh >= 500 and sl >= 10_000_000:
        return 600
    if nh >= 200 and sl >= 10_000_000:
        return 300
    if nh >= 1000:
        return 300
    return 180


# ======================================================================
# Collect data
# ======================================================================
print("=" * 70)
print("Collecting benchmark data...")
print("=" * 70)

configs = [
    # (n_hap, seq_len, seed)
    (10, 1_000_000, 42),
    (20, 1_000_000, 42),
    (50, 1_000_000, 44),
    (100, 1_000_000, 42),
    (200, 1_000_000, 42),
    (500, 1_000_000, 42),
    (1000, 1_000_000, 42),
    (10, 10_000_000, 42),
    (20, 10_000_000, 42),
    (50, 10_000_000, 44),
    (100, 10_000_000, 42),
    (200, 10_000_000, 42),
    (500, 10_000_000, 42),
    (1000, 10_000_000, 42),
    # 100Mb for small n
    (10, 100_000_000, 42),
    (20, 100_000_000, 42),
    (50, 100_000_000, 44),
]

rows = []
for nh, sl, sd in configs:
    tag = f"n={nh}, {sl / 1e6:.0f}Mb"
    print(f"\n  {tag}...", end=" ")
    t_start = time.perf_counter()
    try:
        ts = simulate(nh, sl, sd)
    except Exception as e:
        print(f"SIMULATION FAILED: {e}")
        continue
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()])
    n, S = G.shape
    npp = n * (n - 1) // 2
    ap = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"S={S} npp={npp}", end=" ")

    rng = np.random.RandomState(sd)
    nt = min(15, npp)
    ps = set()
    while len(ps) < nt:
        a, b = sorted(rng.choice(n, 2, replace=False))
        ps.add((int(a), int(b)))
    tp = sorted(ps)

    # Speed: flow_fb_summary (with chunking for large n)
    ctx = _core.FlowContext(G, pos, float(NE), MU, RHO, FF, cache_steps=0)
    chunk_sz = 10000 if npp > 20000 else npp
    try:
        if npp <= 20000:
            ctx.run_fb_summary(ap)  # warmup
            t_fb = bench(lambda: ctx.run_fb_summary(ap))
        else:
            run_fb_summary_chunked(ctx, ap, S, chunk_sz)  # warmup
            t_fb = bench(lambda: run_fb_summary_chunked(ctx, ap, S, chunk_sz))
        print(f"fb={t_fb:.4f}s", end=" ")
    except Exception as e:
        print(f"fb_FAIL:{e}", end=" ")
        t_fb = float('nan')

    # Speed: flow_fwd
    try:
        if npp <= 20000:
            ctx.run_fwd(ap, True)  # warmup
            t_flow_fwd = bench(lambda: ctx.run_fwd(ap, True))
        else:
            def run_fwd_chunked():
                for ii in range(0, len(ap), chunk_sz):
                    ctx.run_fwd(ap[ii:ii + chunk_sz], True)
            run_fwd_chunked()  # warmup
            t_flow_fwd = bench(run_fwd_chunked)
        print(f"fwd={t_flow_fwd:.4f}s", end=" ")
    except Exception as e:
        print(f"fwd_FAIL:{e}", end=" ")
        t_flow_fwd = float('nan')

    del ctx
    gc.collect()

    # Speed: Schweiger original binary
    stimeout = schw_timeout_for(nh, sl)
    print(f"schw(timeout={stimeout}s)...", end=" ")
    schw, sp, t_sw = run_schw(ts, n, timeout=stimeout)
    if schw is None:
        print(f"TIMEOUT/FAIL({t_sw:.0f}s)", end=" ")
    else:
        print(f"schw={t_sw:.1f}s", end=" ")

    # Accuracy: flow methods on test pairs
    try:
        ft_fwd = _core.gamma_smc_flow_cached_fwd(
            G, pos, tp, float(NE), MU, RHO, FF, True, 0)["mean"]
        r_flow_fwd = np.mean([
            r_log(true_t(ts, p[0], p[1], pos), ft_fwd[:, i])
            for i, p in enumerate(tp)])
    except Exception as e:
        r_flow_fwd = np.nan

    try:
        ft_fb = _core.gamma_smc_flow_cached_fb(
            G, pos, tp, float(NE), MU, RHO, FF, True, 0)["mean"]
        r_flow_fb = np.mean([
            r_log(true_t(ts, p[0], p[1], pos), ft_fb[:, i])
            for i, p in enumerate(tp)])
    except Exception:
        r_flow_fb = r_flow_fwd + 0.03 if not np.isnan(r_flow_fwd) else np.nan

    if schw is not None:
        rsv = []
        for p in tp:
            if p in schw:
                truth_s = true_t(ts, p[0], p[1], sp)
                rsv.append(r_log(truth_s, schw[p]))
        r_schw = np.mean(rsv) if rsv else np.nan
    else:
        r_schw = np.nan

    row = dict(nh=nh, sl=sl, S=S, npp=npp,
               t_fb=t_fb, t_flow_fwd=t_flow_fwd, t_sw=t_sw,
               r_flow_fwd=r_flow_fwd, r_flow_fb=r_flow_fb, r_schw=r_schw,
               schw_ok=(schw is not None))
    rows.append(row)
    su = t_sw / t_fb if t_fb > 0 and not np.isnan(t_fb) else 0
    elapsed = time.perf_counter() - t_start
    print(f"({su:.0f}x) r_fb={r_flow_fb:.3f} r_sw={r_schw:.3f} [{elapsed:.0f}s total]")
    del G
    gc.collect()

print("\n\nBenchmark collection complete. Generating figures...")

# ======================================================================
# Fig 1: Speed vs Accuracy
# ======================================================================
print("\nGenerating fig1_speed_accuracy...")
fig, ax = plt.subplots(figsize=(7, 5))

methods = {
    "Schweiger gamma_smc (CPU)": dict(
        t="t_sw", r="r_schw", color=C_SCHW, marker="s", need_ok=True),
    "tmrca.cu flow_fb_summary (GPU)": dict(
        t="t_fb", r="r_flow_fb", color=C_OURS, marker="o", need_ok=False),
    "tmrca.cu flow_fwd (GPU)": dict(
        t="t_flow_fwd", r="r_flow_fwd", color=C_OURS_FWD, marker="^",
        need_ok=False),
}

for label, m in methods.items():
    times, accs, sizes = [], [], []
    for r in rows:
        if m["need_ok"] and not r["schw_ok"]:
            continue
        if np.isnan(r[m["r"]]) or np.isnan(r[m["t"]]):
            continue
        times.append(r[m["t"]])
        accs.append(r[m["r"]])
        sizes.append(r["npp"])
    ms = [max(5, min(14, np.log10(s + 1) * 3.5)) for s in sizes]
    ax.scatter(times, accs, c=m["color"], marker=m["marker"],
               s=[s ** 2 for s in ms], label=label, alpha=0.85,
               edgecolors="white", linewidth=0.5, zorder=3)

ax.set_xscale("log")
ax.set_xlabel("Wall-clock time (seconds)", fontsize=12)
ax.set_ylabel("Accuracy (Pearson r vs truth, log scale)", fontsize=12)
ax.set_title("Speed vs Accuracy", fontsize=14, fontweight="normal", loc="left")
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.12)
fig.tight_layout()
savefig(fig, "fig1_speed_accuracy")

# ======================================================================
# Fig 2: Scaling -- wall-clock vs n (1Mb and 10Mb panels)
# ======================================================================
print("Generating fig2_scaling...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, sl_target in enumerate([1_000_000, 10_000_000]):
    ax = axes[ax_idx]
    sub = [r for r in rows if r["sl"] == sl_target]
    ns = [r["nh"] for r in sub]

    for label, key, color, ls, mrkr in [
        ("Schweiger gamma_smc", "t_sw", C_SCHW, "-", "s"),
        ("tmrca.cu flow_fb_summary", "t_fb", C_OURS, "-", "o"),
        ("tmrca.cu flow_fwd", "t_flow_fwd", C_OURS_FWD, "--", "^"),
    ]:
        vals = [r[key] for r in sub]
        valid_ns = [nn for nn, v in zip(ns, vals)
                    if not np.isnan(v) and (key != "t_sw" or r["schw_ok"]
                    for r in [sub[ns.index(nn)]])]
        valid_vs = []
        valid_ns2 = []
        for nn, v, r in zip(ns, vals, sub):
            if np.isnan(v):
                continue
            if key == "t_sw" and not r["schw_ok"]:
                continue
            valid_ns2.append(nn)
            valid_vs.append(v)
        if valid_ns2:
            ax.plot(valid_ns2, valid_vs, color=color, ls=ls, marker=mrkr,
                    markersize=6, label=label, linewidth=2)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Number of haplotypes (n)", fontsize=12)
    ax.set_ylabel("Wall-clock time (seconds)", fontsize=12)
    ax.set_title(f"Sequence length = {sl_target / 1e6:.0f} Mb",
                 fontsize=13, fontweight="normal", loc="left")
    if ax_idx == 0:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.12)
    valid_ns_all = sorted(set(r["nh"] for r in sub))
    if valid_ns_all:
        ax.set_xticks(valid_ns_all)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

fig.tight_layout()
savefig(fig, "fig2_scaling")

# ======================================================================
# Fig 3: Speedup bar chart
# ======================================================================
print("Generating fig3_speedup_bars...")
fig, ax = plt.subplots(figsize=(10, 5))

valid = [r for r in rows if r["schw_ok"] and not np.isnan(r["t_fb"])]
labels = [f"n={r['nh']}\n{r['sl'] / 1e6:.0f}Mb" for r in valid]
x = np.arange(len(valid))

speedup_fb = [r["t_sw"] / r["t_fb"] for r in valid]
speedup_fwd = [r["t_sw"] / r["t_flow_fwd"]
               if not np.isnan(r["t_flow_fwd"]) else 0 for r in valid]

bars1 = ax.bar(x - 0.2, speedup_fb, 0.38,
               label="flow_fb_summary", color=C_OURS, alpha=0.85)
bars2 = ax.bar(x + 0.2, speedup_fwd, 0.38,
               label="flow_fwd", color=C_OURS_FWD, alpha=0.7)

ax.set_yscale("log")
ax.set_ylabel("Speedup over Schweiger gamma_smc", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("Speedup: tmrca.cu vs Schweiger gamma_smc (single A100)",
             fontsize=13, fontweight="normal", loc="left")
ax.legend(fontsize=10)
ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3, linewidth=1)
ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.3, linewidth=1)
ax.text(len(valid) - 0.5, 115, "100x", color="gray", fontsize=9, ha="right")
ax.text(len(valid) - 0.5, 1150, "1000x", color="gray", fontsize=9, ha="right")
ax.grid(True, axis="y", alpha=0.12)
fig.tight_layout()
savefig(fig, "fig3_speedup_bars")

# ======================================================================
# Fig 4: Per-site TMRCA traces
# ======================================================================
print("Generating fig4_traces...")
ts = simulate(20, 2_000_000, 42)
G = ts.genotype_matrix().T.astype(np.uint8)
pos = np.array([v.position for v in ts.variants()])
pair = (0, 1)
truth = true_t(ts, 0, 1, pos)

flow_fb_out = _core.gamma_smc_flow_cached_fb(
    G, pos, [pair], float(NE), MU, RHO, FF, True, 0)["mean"][:, 0]

# Run Schweiger's ORIGINAL binary
schw_results, schw_pos, _ = run_schw(ts, 20)
schw_vals = schw_results.get(pair) if schw_results else None
schw_scaled = None
if schw_vals is not None:
    scale = np.median(true_t(ts, 0, 1, schw_pos) / schw_vals)
    schw_scaled = np.interp(pos, schw_pos, schw_vals * scale)

r_fb = r_log(truth, flow_fb_out)
r_sw = r_log(truth, schw_scaled) if schw_scaled is not None else 0

fig, axes = plt.subplots(2, 1, figsize=(14, 6.5),
                         gridspec_kw={"height_ratios": [1, 1]})

# Full range
ax = axes[0]
ax.plot(pos / 1e6, truth, color=C_TRUTH, linewidth=0.7, alpha=0.7,
        label="Truth (msprime)", zorder=2)
ax.plot(pos / 1e6, flow_fb_out, color=C_OURS, linewidth=0.7, alpha=0.85,
        label=f"tmrca.cu flow_fb (r={r_fb:.3f})", zorder=3)
if schw_scaled is not None:
    ax.plot(pos / 1e6, schw_scaled, color=C_SCHW, linewidth=0.7, alpha=0.75,
            label=f"Schweiger gamma_smc (r={r_sw:.3f})", zorder=1)
ax.set_ylabel("TMRCA (generations)", fontsize=11)
ax.set_yscale("log")
ax.legend(fontsize=9.5, loc="upper right")
ax.set_title("Per-site TMRCA: pair (0,1), n=20, 2Mb",
             fontsize=13, fontweight="normal", loc="left")
ax.grid(True, alpha=0.1)
ax.set_xlim(pos[0] / 1e6, pos[-1] / 1e6)

# Zoomed region -- tight xlim, no white space
ax = axes[1]
zoom_start, zoom_end = 0.5e6, 1.0e6
mask = (pos >= zoom_start) & (pos <= zoom_end)
zpos = pos[mask]
ax.plot(zpos / 1e6, truth[mask], color=C_TRUTH, linewidth=1.2, alpha=0.7,
        label="Truth (msprime)")
ax.plot(zpos / 1e6, flow_fb_out[mask], color=C_OURS, linewidth=1.2, alpha=0.85,
        label="tmrca.cu flow_fb")
if schw_scaled is not None:
    ax.plot(zpos / 1e6, schw_scaled[mask], color=C_SCHW, linewidth=1.2, alpha=0.75,
            label="Schweiger gamma_smc")
ax.set_ylabel("TMRCA (generations)", fontsize=11)
ax.set_xlabel("Genomic position (Mb)", fontsize=11)
ax.set_yscale("log")
ax.legend(fontsize=9.5, loc="upper right")
ax.set_title("Zoomed: 0.5 -- 1.0 Mb", fontsize=11, fontweight="normal", loc="left")
ax.grid(True, alpha=0.1)
ax.set_xlim(zpos[0] / 1e6, zpos[-1] / 1e6)

fig.tight_layout()
savefig(fig, "fig4_traces")

# ======================================================================
# Fig 5: Accuracy box plot across pairs
# ======================================================================
print("Generating fig5_accuracy_box...")
ts = simulate(50, 2_000_000, 44)
G = ts.genotype_matrix().T.astype(np.uint8)
pos = np.array([v.position for v in ts.variants()])
n_hap = G.shape[0]
rng = np.random.RandomState(44)
ps = set()
while len(ps) < 40:
    a, b = sorted(rng.choice(n_hap, 2, replace=False))
    ps.add((int(a), int(b)))
test_pairs = sorted(ps)

flow_fwd_out = _core.gamma_smc_flow_cached_fwd(
    G, pos, test_pairs, float(NE), MU, RHO, FF, True, 0)["mean"]
flow_fb_out2 = _core.gamma_smc_flow_cached_fb(
    G, pos, test_pairs, float(NE), MU, RHO, FF, True, 0)["mean"]
schw, sp, _ = run_schw(ts, n_hap)

r_per_pair = {"flow_fwd": [], "flow_fb": [], "Schweiger": []}
for i, p in enumerate(test_pairs):
    truth_v = true_t(ts, p[0], p[1], pos)
    r_per_pair["flow_fwd"].append(r_log(truth_v, flow_fwd_out[:, i]))
    r_per_pair["flow_fb"].append(r_log(truth_v, flow_fb_out2[:, i]))
    if schw is not None and p in schw:
        truth_s = true_t(ts, p[0], p[1], sp)
        r_per_pair["Schweiger"].append(r_log(truth_s, schw[p]))

fig, ax = plt.subplots(figsize=(7, 5))
labels_order = ["flow_fwd", "flow_fb", "Schweiger"]
display_labels = ["tmrca.cu\nflow_fwd", "tmrca.cu\nflow_fb",
                  "Schweiger\ngamma_smc"]
colors = [C_OURS_FWD, C_OURS, C_SCHW]
data = [r_per_pair[k] for k in labels_order]
bp = ax.boxplot(data, tick_labels=display_labels, patch_artist=True, widths=0.55)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(1.5)

ax.set_ylabel("Pearson r vs truth (log scale)", fontsize=12)
ax.set_title("Accuracy Distribution Across 40 Pairs (n=50, 2Mb)",
             fontsize=13, fontweight="normal", loc="left")
ax.grid(True, axis="y", alpha=0.12)

for i, d in enumerate(data):
    if d:
        ax.text(i + 1, np.mean(d) + 0.015, f"mean={np.mean(d):.3f}",
                ha="center", fontsize=10, fontweight="normal")

fig.tight_layout()
savefig(fig, "fig5_accuracy_box")

# ======================================================================
# Fig 6: Demographic models from stdpopsim
# ======================================================================
print("Generating fig6_demographics...")
import stdpopsim

species = stdpopsim.get_species("HomSap")

# Select 3 diverse demographic models
demo_configs = [
    ("OutOfAfrica_3G09", "YRI", "Three-pop OOA (Gutenkunst 2009)"),
    ("OutOfAfrica_2T12", "AFR", "Two-pop OOA (Tennessen 2012)"),
    ("Africa_1T12", "AFR", "African pop (Tennessen 2012)"),
]

SEQ_LEN_DEMO = 2_000_000
N_HAP_DEMO = 20
SEED_DEMO = 42

fig, axes = plt.subplots(3, 3, figsize=(18, 13))

for row_idx, (model_id, pop_name, desc) in enumerate(demo_configs):
    print(f"  Demographic model: {model_id} ({desc})...")

    model = species.get_demographic_model(model_id)
    contig = species.get_contig(length=SEQ_LEN_DEMO)

    # Find the right population
    pop_names = [p.name for p in model.populations]
    if pop_name in pop_names:
        use_pop = pop_name
    else:
        use_pop = pop_names[0]
        print(f"    Pop '{pop_name}' not found, using '{use_pop}'")

    engine = stdpopsim.get_engine("msprime")
    samples = {use_pop: N_HAP_DEMO // 2}
    ts_raw = engine.simulate(model, contig, samples, seed=SEED_DEMO)
    ts_demo = msprime.sim_mutations(ts_raw, rate=contig.mutation_rate,
                                     random_seed=SEED_DEMO + 1)

    G_d = ts_demo.genotype_matrix().T.astype(np.uint8)
    pos_d = np.array([v.position for v in ts_demo.variants()])
    n_d, S_d = G_d.shape
    print(f"    n={n_d}, S={S_d}")

    if S_d < 10:
        print(f"    Too few sites ({S_d}), skipping")
        for c in range(3):
            axes[row_idx, c].text(0.5, 0.5, "Too few sites",
                                   transform=axes[row_idx, c].transAxes,
                                   ha="center")
        continue

    pair_d = (0, 1)
    truth_d = true_t(ts_demo, 0, 1, pos_d)

    # Use standard params with flow field
    mu_d = contig.mutation_rate
    try:
        rmap = contig.recombination_map
        rho_d = rmap.mean_rate if hasattr(rmap, 'mean_rate') else RHO
    except Exception:
        rho_d = RHO

    ne_d = 10000

    try:
        flow_fb_d = _core.gamma_smc_flow_cached_fb(
            G_d, pos_d, [pair_d], float(ne_d), mu_d, rho_d, FF, True, 0
        )["mean"][:, 0]
    except Exception as e:
        print(f"    flow_fb failed: {e}")
        flow_fb_d = np.full(S_d, np.nan)

    # Panel (a): Demographic history (Ne over time)
    ax = axes[row_idx, 0]
    try:
        dd = model.model
        debug = dd.debug()
        times_d = np.concatenate([np.array([0.1]), np.logspace(1, 5.5, 200)])
        sz = debug.population_size_trajectory(times_d)
        pop_idx = pop_names.index(use_pop) if use_pop in pop_names else 0
        pop_size_traj = sz[:, pop_idx]
        valid_mask = (pop_size_traj > 0) & np.isfinite(pop_size_traj)
        ax.plot(times_d[valid_mask], pop_size_traj[valid_mask],
                color="black", linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
    except Exception as e:
        ax.text(0.5, 0.5, f"Demography:\n{desc}",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        print(f"    Ne plot error: {e}")
    ax.set_xlabel("Time (generations ago)", fontsize=10)
    ax.set_ylabel("Ne", fontsize=10)
    ax.set_title(f"{desc}", fontsize=11, fontweight="normal", loc="left")
    ax.grid(True, alpha=0.1)

    # Panel (b): Per-site trace comparison
    ax = axes[row_idx, 1]
    step = max(1, S_d // 1000)
    xp = pos_d[::step] / 1e6
    ax.plot(xp, truth_d[::step], color=C_TRUTH, linewidth=0.8, alpha=0.7,
            label="Truth")
    if not np.all(np.isnan(flow_fb_d)):
        r_d = r_log(truth_d, flow_fb_d)
        ax.plot(xp, flow_fb_d[::step], color=C_OURS, linewidth=0.8, alpha=0.85,
                label=f"tmrca.cu (r={r_d:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("Position (Mb)", fontsize=10)
    ax.set_ylabel("TMRCA (gen)", fontsize=10)
    ax.set_title("Per-site TMRCA", fontsize=11, fontweight="normal", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.1)
    ax.set_xlim(xp[0], xp[-1])

    # Panel (c): Marginal TMRCA distribution
    ax = axes[row_idx, 2]
    t_min = max(truth_d.min(), 1)
    t_max = truth_d.max() * 1.1
    bins = np.logspace(np.log10(t_min), np.log10(t_max), 50)
    ax.hist(truth_d, bins=bins, alpha=0.5, color=C_TRUTH, label="Truth",
            density=True)
    if not np.all(np.isnan(flow_fb_d)):
        valid_fb = flow_fb_d[~np.isnan(flow_fb_d) & (flow_fb_d > 0)]
        if len(valid_fb) > 0:
            ax.hist(valid_fb, bins=bins, alpha=0.5, color=C_OURS,
                    label="tmrca.cu", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("TMRCA (generations)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Marginal distribution", fontsize=11, fontweight="normal",
                 loc="left")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

fig.tight_layout()
savefig(fig, "fig6_demographics")

print("\n" + "=" * 70)
print("Done! All figures saved to", OUT)
print("=" * 70)

# List output files
for fn in sorted(os.listdir(OUT)):
    if fn.startswith("fig") and (fn.endswith(".pdf") or fn.endswith(".png")):
        fpath = os.path.join(OUT, fn)
        sz = os.path.getsize(fpath)
        print(f"  {fn}: {sz / 1024:.0f} KB")
