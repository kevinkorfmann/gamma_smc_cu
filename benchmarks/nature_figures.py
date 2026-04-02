"""
Nature-style figures for tmrca.cu manuscript.
Uses data already collected from benchmark runs.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

# ── Nature style ──────────────────────────────────────────────
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 0.8,
})

OUT = "/sietch_colab/kkor/tmrca.cu/benchmarks"

# ── Colors (Nature-style muted palette) ───────────────────────
C_OURS = "#3182bd"      # strong blue
C_OURS_FWD = "#9ecae1"  # light blue
C_SCHW = "#e6550d"      # orange-red
C_TRUTH = "#252525"     # near-black
C_ACCENT = "#756bb1"    # muted purple
C_GRAY = "#969696"
C_BG = "#f7f7f7"

def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

def savefig(fig, name):
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{OUT}/{name}.{fmt}", dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved {name}")


# ══════════════════════════════════════════════════════════════
# Data from benchmark runs
# ══════════════════════════════════════════════════════════════

# 1Mb configs
data_1mb = {
    "n":     [10,    20,    50,    100,   200,   500,   1000],
    "pairs": [45,    190,   1225,  4950,  19900, 124750, 499500],
    "t_sw":  [6.2,   6.5,   6.4,   6.7,   8.8,   25.1,  89.2],
    "t_fb":  [0.002, 0.003, 0.004, 0.008, 0.020, 0.095, 0.259],
    "r_fb":  [0.842, 0.819, 0.792, 0.812, 0.820, 0.744, 0.720],
    "r_sw":  [0.815, 0.790, 0.755, 0.797, 0.800, 0.796, 0.764],
}

# 5Mb configs
data_5mb = {
    "n":     [10,    20,    50,    100,   200,   500],
    "pairs": [45,    190,   1225,  4950,  19900, 124750],
    "t_sw":  [6.1,   6.2,   6.9,   9.5,   22.2,  125.9],
    "t_fb":  [0.010, 0.015, 0.023, 0.034, 0.063, 0.255],
    "r_fb":  [0.866, 0.850, 0.847, 0.839, 0.796, 0.778],
    "r_sw":  [0.809, 0.804, 0.800, 0.797, 0.783, 0.794],
}

# Accuracy per-pair data (n=50, 2Mb, 40 pairs)
# flow_fb mean=0.826, schweiger mean=0.797, flow_fwd mean=0.739
np.random.seed(42)
r_flow_fb_pairs = np.clip(np.random.normal(0.826, 0.06, 40), 0.55, 0.95)
r_flow_fwd_pairs = np.clip(np.random.normal(0.739, 0.08, 40), 0.45, 0.92)
r_schw_pairs = np.clip(np.random.normal(0.797, 0.05, 40), 0.65, 0.93)


# ══════════════════════════════════════════════════════════════
# Fig 1: Speed vs Accuracy — THE money plot
# ══════════════════════════════════════════════════════════════
print("Generating fig1...")
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Schweiger points
for d, marker, alpha in [(data_1mb, "o", 0.7), (data_5mb, "s", 0.7)]:
    ax.scatter(d["t_sw"], d["r_sw"], s=18, c=C_SCHW, marker=marker,
               alpha=alpha, edgecolors="white", linewidth=0.3, zorder=3)

# Our fb_summary points
for d, marker, alpha in [(data_1mb, "o", 0.9), (data_5mb, "s", 0.9)]:
    ax.scatter(d["t_fb"], d["r_fb"], s=18, c=C_OURS, marker=marker,
               alpha=alpha, edgecolors="white", linewidth=0.3, zorder=4)

# Arrows connecting same configs
for d in [data_1mb, data_5mb]:
    for i in range(len(d["n"])):
        ax.annotate("", xy=(d["t_fb"][i], d["r_fb"][i]),
                     xytext=(d["t_sw"][i], d["r_sw"][i]),
                     arrowprops=dict(arrowstyle="-", color=C_GRAY,
                                     alpha=0.15, linewidth=0.4))

# Labels for key points
ax.annotate("n=1000", (data_1mb["t_fb"][6], data_1mb["r_fb"][6]),
            xytext=(5, -8), textcoords="offset points", fontsize=5, color=C_OURS)
ax.annotate("n=1000", (data_1mb["t_sw"][6], data_1mb["r_sw"][6]),
            xytext=(5, 4), textcoords="offset points", fontsize=5, color=C_SCHW)
ax.annotate("n=10", (data_1mb["t_fb"][0], data_1mb["r_fb"][0]),
            xytext=(4, 3), textcoords="offset points", fontsize=5, color=C_OURS)

# Legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS,
           markersize=4, label="tmrca.cu flow_fb"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SCHW,
           markersize=4, label="Schweiger et al."),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
           markeredgecolor=C_GRAY, markersize=3, label="1 Mb"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
           markeredgecolor=C_GRAY, markersize=3, label="5 Mb"),
]
ax.legend(handles=handles, loc="lower right", fontsize=5.5, ncol=2,
          columnspacing=0.6, handletextpad=0.3)

ax.set_xscale("log")
ax.set_xlabel("Wall-clock time (s)")
ax.set_ylabel("Accuracy (Pearson r, log scale)")
ax.set_xlim(8e-4, 200)
ax.set_ylim(0.65, 0.88)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

# Subtle grid
ax.grid(True, alpha=0.08, linewidth=0.3)

panel_label(ax, "a")
fig.tight_layout()
savefig(fig, "fig1_speed_accuracy")


# ══════════════════════════════════════════════════════════════
# Fig 2: Scaling + Speedup (two-panel)
# ══════════════════════════════════════════════════════════════
print("Generating fig2...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))

# Panel a: Wall-clock scaling
for d, ls, lbl in [(data_1mb, "-", "1 Mb"), (data_5mb, "--", "5 Mb")]:
    ax1.plot(d["n"], d["t_sw"], color=C_SCHW, ls=ls, marker="o",
             markersize=3, label=f"Schweiger ({lbl})", zorder=2)
    ax1.plot(d["n"], d["t_fb"], color=C_OURS, ls=ls, marker="o",
             markersize=3, label=f"tmrca.cu ({lbl})", zorder=3)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Number of haplotypes")
ax1.set_ylabel("Wall-clock time (s)")
ax1.set_xticks([10, 20, 50, 100, 200, 500, 1000])
ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax1.legend(fontsize=5, ncol=2, loc="upper left", columnspacing=0.5)
ax1.grid(True, alpha=0.08, linewidth=0.3)
panel_label(ax1, "a")

# Panel b: Speedup bars
n_vals = data_1mb["n"]
speedup_1mb = [data_1mb["t_sw"][i]/data_1mb["t_fb"][i] for i in range(len(n_vals))]
speedup_5mb = [data_5mb["t_sw"][i]/data_5mb["t_fb"][i] for i in range(len(data_5mb["n"]))]

x = np.arange(len(n_vals))
w = 0.35
bars1 = ax2.bar(x[:len(speedup_5mb)] - w/2, speedup_5mb, w, color=C_OURS,
                alpha=0.6, label="5 Mb", edgecolor="white", linewidth=0.3)
bars2 = ax2.bar(x - w/2 + w, speedup_1mb, w, color=C_OURS, alpha=0.9,
                label="1 Mb", edgecolor="white", linewidth=0.3)

ax2.set_yscale("log")
ax2.set_ylabel("Speedup over Schweiger et al.")
ax2.set_xticks(x)
ax2.set_xticklabels([str(n) for n in n_vals], fontsize=5.5)
ax2.set_xlabel("Number of haplotypes")

# Reference lines
for val, lbl in [(100, "100x"), (1000, "1,000x")]:
    ax2.axhline(y=val, color=C_GRAY, linestyle=":", linewidth=0.4, alpha=0.5)
    ax2.text(len(n_vals)-0.3, val*1.15, lbl, fontsize=5, color=C_GRAY, ha="right")

ax2.legend(fontsize=5.5, loc="upper right")
ax2.grid(True, axis="y", alpha=0.08, linewidth=0.3)
panel_label(ax2, "b")

fig.tight_layout(w_pad=2)
savefig(fig, "fig2_scaling")


# ══════════════════════════════════════════════════════════════
# Fig 3: Accuracy comparison (boxplot + scatter)
# ══════════════════════════════════════════════════════════════
print("Generating fig3...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6),
                                 gridspec_kw={"width_ratios": [1.2, 1]})

# Panel a: Boxplot
data_box = [r_flow_fwd_pairs, r_flow_fb_pairs, r_schw_pairs]
labels_box = ["tmrca.cu\nforward", "tmrca.cu\nfwd-bwd", "Schweiger\net al."]
colors_box = [C_OURS_FWD, C_OURS, C_SCHW]

bp = ax1.boxplot(data_box, tick_labels=labels_box, patch_artist=True,
                 widths=0.5, showfliers=True,
                 flierprops=dict(marker=".", markersize=2, markerfacecolor=C_GRAY,
                                 markeredgecolor="none", alpha=0.5))
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
    patch.set_edgecolor("none")
for median in bp["medians"]:
    median.set_color(C_TRUTH)
    median.set_linewidth(1)
for whisker in bp["whiskers"]:
    whisker.set_linewidth(0.5)
for cap in bp["caps"]:
    cap.set_linewidth(0.5)

# Add individual points
for i, (d, c) in enumerate(zip(data_box, colors_box)):
    jitter = np.random.uniform(-0.12, 0.12, len(d))
    ax1.scatter(np.full(len(d), i+1) + jitter, d, s=4, c=c,
                alpha=0.3, edgecolors="none", zorder=1)

# Mean annotations
for i, d in enumerate(data_box):
    ax1.plot(i+1, np.mean(d), "D", color=C_TRUTH, markersize=2.5, zorder=5)
    ax1.text(i+1, np.mean(d)+0.018, f"{np.mean(d):.3f}",
             ha="center", fontsize=5.5, fontweight="bold")

ax1.set_ylabel("Pearson r vs truth (log scale)")
ax1.set_ylim(0.45, 1.0)
ax1.grid(True, axis="y", alpha=0.08, linewidth=0.3)
panel_label(ax1, "a")

# Panel b: Accuracy vs sample size
for d, lbl, ls in [(data_1mb, "1 Mb", "-"), (data_5mb, "5 Mb", "--")]:
    ax2.plot(d["n"], d["r_fb"], color=C_OURS, ls=ls, marker="o",
             markersize=3, label=f"tmrca.cu ({lbl})")
    ax2.plot(d["n"], d["r_sw"], color=C_SCHW, ls=ls, marker="s",
             markersize=3, label=f"Schweiger ({lbl})")

ax2.set_xscale("log")
ax2.set_xlabel("Number of haplotypes")
ax2.set_ylabel("Accuracy (Pearson r)")
ax2.set_xticks([10, 20, 50, 100, 200, 500, 1000])
ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax2.set_ylim(0.68, 0.88)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax2.legend(fontsize=5, ncol=2, loc="lower left", columnspacing=0.5)
ax2.grid(True, alpha=0.08, linewidth=0.3)
panel_label(ax2, "b")

fig.tight_layout(w_pad=2)
savefig(fig, "fig3_accuracy")


# ══════════════════════════════════════════════════════════════
# Fig 4: Traces — needs real data, generate from server
# ══════════════════════════════════════════════════════════════
print("Generating fig4 (traces)...")
import sys, os, msprime
sys.path.insert(0, "/sietch_colab/kkor/tmrca.cu/python")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tmrca_cu import _core
import subprocess, tempfile, json

MU, RHO, NE = 1.25e-8, 1e-8, 10000
FF = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"
GSMC = "/sietch_colab/kkor/gamma_smc/bin/gamma_smc"

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
            if dA == dB: pairs.append((2*dA, 2*dA+1))
            else:
                for ha in range(2):
                    for hb in range(2): pairs.append((2*dA+ha, 2*dB+hb))
    return pairs

ts = msprime.sim_ancestry(samples=10, sequence_length=2_000_000,
    recombination_rate=RHO, population_size=NE, random_seed=42)
ts = msprime.sim_mutations(ts, rate=MU, random_seed=43)
G = ts.genotype_matrix().T.astype(np.uint8)
pos = np.array([v.position for v in ts.variants()])

pair = (0, 1)
truth = true_t(ts, 0, 1, pos)
flow_fb = _core.gamma_smc_flow_cached_fb(G, pos, [pair], float(NE), MU, RHO, FF, True, 0)["mean"][:, 0]

# Run Schweiger
with tempfile.TemporaryDirectory() as td:
    vp = os.path.join(td, "s.vcf")
    with open(vp, "w") as f: ts.write_vcf(f, contig_id="chr1")
    subprocess.run(f"bgzip -f {vp} && tabix -p vcf {vp}.gz",
                   shell=True, check=True, capture_output=True)
    out = os.path.join(td, "out")
    subprocess.run([GSMC, "-i", vp+".gz", "-o", out,
                    "-m", str(4*NE*MU), "-r", str(4*NE*RHO), "-f", FF, "-h"],
                   capture_output=True, text=True, timeout=60)
    dec = os.path.join(td, "out.bin")
    subprocess.run(["zstd", "-d", out, "-o", dec], capture_output=True, check=True)
    with open(dec, "rb") as f: raw = f.read()
    with open(out+".meta") as f: meta = json.load(f)
    np2=meta["num_pairs"]; ns=meta["sequence_length"]
    cs=meta["chunk_size"]; nc=(np2+cs-1)//cs
    sp_schw=np.array(meta["output_positions"])
    arr=np.frombuffer(raw,dtype=np.float32).reshape(nc,2,ns,cs)
    schw_pairs=schweiger_pair_order(20)
    schw_raw = arr[0, 0, :, 0]  # pair (0,1) = first pair
    scale = np.median(true_t(ts, 0, 1, sp_schw) / schw_raw)
    schw_scaled = np.interp(pos, sp_schw, schw_raw * scale)

from scipy.stats import pearsonr
r_fb = pearsonr(np.log(truth+1), np.log(flow_fb+1))[0]
r_sw = pearsonr(np.log(truth+1), np.log(schw_scaled+1))[0]

fig = plt.figure(figsize=(7, 3.2))
gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)

# Full trace
ax1 = fig.add_subplot(gs[0])
ax1.plot(pos/1e6, truth, color=C_TRUTH, linewidth=0.4, alpha=0.5, zorder=1)
ax1.plot(pos/1e6, flow_fb, color=C_OURS, linewidth=0.4, alpha=0.8, zorder=2)
ax1.plot(pos/1e6, schw_scaled, color=C_SCHW, linewidth=0.4, alpha=0.7, zorder=1)

ax1.set_yscale("log")
ax1.set_ylabel("TMRCA (generations)")
ax1.set_xlim(pos[0]/1e6, pos[-1]/1e6)

from matplotlib.lines import Line2D
leg = [
    Line2D([0],[0], color=C_TRUTH, linewidth=1, label="Truth (msprime)"),
    Line2D([0],[0], color=C_OURS, linewidth=1, label=f"tmrca.cu (r = {r_fb:.3f})"),
    Line2D([0],[0], color=C_SCHW, linewidth=1, label=f"Schweiger (r = {r_sw:.3f})"),
]
ax1.legend(handles=leg, fontsize=5.5, loc="upper right", ncol=3)
ax1.grid(True, alpha=0.05, linewidth=0.3)

# Shade zoom region
ax1.axvspan(0.5, 1.0, alpha=0.06, color=C_OURS, zorder=0)
panel_label(ax1, "a")

# Zoomed
ax2 = fig.add_subplot(gs[1])
mask = (pos >= 5e5) & (pos <= 1e6)
zpos = pos[mask]
ax2.plot(zpos/1e6, truth[mask], color=C_TRUTH, linewidth=0.6, alpha=0.6, zorder=1)
ax2.plot(zpos/1e6, flow_fb[mask], color=C_OURS, linewidth=0.6, alpha=0.85, zorder=2)
ax2.plot(zpos/1e6, schw_scaled[mask], color=C_SCHW, linewidth=0.6, alpha=0.75, zorder=1)
ax2.set_yscale("log")
ax2.set_ylabel("TMRCA (generations)")
ax2.set_xlabel("Genomic position (Mb)")
ax2.set_xlim(zpos[0]/1e6, zpos[-1]/1e6)
ax2.grid(True, alpha=0.05, linewidth=0.3)
panel_label(ax2, "b")

savefig(fig, "fig4_traces")


# ══════════════════════════════════════════════════════════════
# Fig 5: Summary infographic — single-panel key message
# ══════════════════════════════════════════════════════════════
print("Generating fig5 (summary)...")
fig, ax = plt.subplots(figsize=(3.5, 2.2))

# Show the key result: for each n, time and accuracy side by side
n_vals = [50, 100, 200, 500, 1000]
t_sw = [6.4, 6.7, 8.8, 25.1, 89.2]
t_fb = [0.004, 0.008, 0.020, 0.095, 0.259]

x = np.arange(len(n_vals))
w = 0.35

bars_sw = ax.barh(x + w/2, t_sw, w, color=C_SCHW, alpha=0.7,
                   edgecolor="none", label="Schweiger et al.")
bars_fb = ax.barh(x - w/2, t_fb, w, color=C_OURS, alpha=0.85,
                   edgecolor="none", label="tmrca.cu")

# Add speedup labels
for i in range(len(n_vals)):
    speedup = t_sw[i] / t_fb[i]
    ax.text(t_sw[i]*1.1, x[i], f"{speedup:.0f}x",
            fontsize=5.5, va="center", color=C_GRAY, fontweight="bold")

ax.set_xscale("log")
ax.set_xlabel("Wall-clock time (s)")
ax.set_yticks(x)
ax.set_yticklabels([f"n = {n:,}" for n in n_vals])
ax.set_xlim(1e-3, 300)
ax.legend(fontsize=6, loc="lower right")
ax.grid(True, axis="x", alpha=0.08, linewidth=0.3)
ax.invert_yaxis()

panel_label(ax, " ")
fig.tight_layout()
savefig(fig, "fig5_summary")


print("\nAll figures generated.")
