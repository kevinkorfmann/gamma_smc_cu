"""
Generate benchmark plots from bench_results.npz.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

out_dir = os.path.dirname(__file__)
data = np.load(os.path.join(out_dir, "bench_results.npz"))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("tmrca_cu kernel benchmarks — NVIDIA A100 80GB", fontsize=14, fontweight="bold")

# Consistent styling
kw = dict(marker="o", linewidth=2, markersize=7)
fill_kw = dict(alpha=0.12)

# ── 1. Prefix scan vs S ─────────────────────────────────────────────────
ax = axes[0, 0]
d = data["prefix"]
ax.plot(d[:, 0] / 1e3, d[:, 1] * 1e3, color="C0", **kw)
ax.set_xlabel("Sequence length (k sites)")
ax.set_ylabel("Time (ms)")
ax.set_title("Prefix scan (4 pairs, n=100)")
ax.set_xscale("log")
ax.grid(True, alpha=0.3)

# ── 2. SFS vs n ─────────────────────────────────────────────────────────
ax = axes[0, 1]
d = data["sfs"]
ax.plot(d[:, 0], d[:, 1] * 1e3, color="C1", **kw)
ax.set_xlabel("Sample size (n haplotypes)")
ax.set_ylabel("Time (ms)")
ax.set_title("SFS computation (S=50k)")
ax.set_xscale("log")
ax.grid(True, alpha=0.3)

# ── 3. HMM single pair vs S ─────────────────────────────────────────────
ax = axes[0, 2]
d = data["hmm_single"]
ax.plot(d[:, 0] / 1e3, d[:, 1] * 1e3, color="C2", **kw)
ax.set_xlabel("Sequence length (k sites)")
ax.set_ylabel("Time (ms)")
ax.set_title("HMM forward-backward (1 pair)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

# ── 4. HMM batched: time + throughput vs pairs ──────────────────────────
ax = axes[1, 0]
d = data["hmm_batched"]
n_pairs = d[:, 0]
time_s = d[:, 1]
tput = n_pairs * 10_000 / time_s / 1e6  # M site·pairs/s

color1 = "C3"
color2 = "C4"
ln1 = ax.plot(n_pairs, time_s * 1e3, color=color1, **kw, label="Time")
ax.set_xlabel("Number of pairs")
ax.set_ylabel("Time (ms)", color=color1)
ax.tick_params(axis="y", labelcolor=color1)
ax.set_xscale("log")

ax2 = ax.twinx()
ln2 = ax2.plot(n_pairs, tput, color=color2, **kw, label="Throughput", linestyle="--")
ax2.set_ylabel("Throughput (M site·pairs/s)", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc="center left", fontsize=8)
ax.set_title("HMM batched (S=10k)")
ax.grid(True, alpha=0.3)

# ── 5. Windowed divergence vs window ─────────────────────────────────────
ax = axes[1, 1]
d = data["windowed_div"]
ax.plot(d[:, 0], d[:, 1] * 1e3, color="C5", **kw)
ax.set_xlabel("Window size (sites)")
ax.set_ylabel("Time (ms)")
ax.set_title("Windowed divergence (10 pairs, S=100k)")
ax.set_xscale("log")
ax.grid(True, alpha=0.3)

# ── 6. Bitpack throughput vs n ───────────────────────────────────────────
ax = axes[1, 2]
d = data["bitpack"]
tput = d[:, 0] * 100_000 / d[:, 1] / 1e9  # Gbits/s
ax.plot(d[:, 0], tput, color="C6", **kw)
ax.set_xlabel("Sample size (n haplotypes)")
ax.set_ylabel("Throughput (Gbit/s)")
ax.set_title("Bitpack throughput (S=100k)")
ax.set_xscale("log")
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(out_dir, "benchmarks.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

# ── Also make a summary table ────────────────────────────────────────────
print("\n" + "=" * 65)
print("Summary: peak throughput / key numbers")
print("=" * 65)

d = data["hmm_batched"]
peak_tput = d[-1, 0] * 10_000 / d[-1, 1] / 1e6
print(f"  HMM batched peak:   {peak_tput:.1f} M site·pairs/s  (1000 pairs × 10k sites)")

d = data["hmm_single"]
print(f"  HMM 100k sites:     {d[-1, 1]*1e3:.0f} ms  (single pair)")

d = data["prefix"]
print(f"  Prefix scan 500k:   {d[-1, 1]*1e3:.1f} ms  (4 pairs)")

d = data["bitpack"]
peak_bp = d[-1, 0] * 100_000 / d[-1, 1] / 1e9
print(f"  Bitpack peak:       {peak_bp:.1f} Gbit/s  (n=2000, S=100k)")

d = data["sfs"]
print(f"  SFS n=1000:         {d[-1, 1]*1e3:.1f} ms  (S=50k)")
