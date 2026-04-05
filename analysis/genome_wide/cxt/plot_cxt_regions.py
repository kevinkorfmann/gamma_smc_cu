#!/usr/bin/env python
"""Plot cxt TMRCA profiles for the 6 S3 Fig regions."""

import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS = os.path.dirname(os.path.abspath(__file__)) + "/results"
OUTFILE = os.path.dirname(os.path.abspath(__file__)) + "/fig_cxt_sweep_regions.png"

REGIONS = [
    ("SH2B3_ALDH2", "TSI", "SH2B3-ALDH2 (EUR)", "chr12"),
    ("CYP3A", "FIN", "CYP3A (EUR)", "chr7"),
    ("FADS1", "ITU", "FADS1 (SAS)", "chr11"),
    ("CLEC6A", "CDX", "CLEC6A (EAS)", "chr12"),
    ("ABCC11", "CHB", "ABCC11 (EAS)", "chr16"),
    ("TRPV6_KEL", "CHB", "TRPV6-KEL (EAS)", "chr7"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.ravel()

for idx, (region, pop, title, chrom) in enumerate(REGIONS):
    ax = axes[idx]
    fname = os.path.join(RESULTS, f"cxt_{region}_{pop}.npz")
    data = np.load(fname, allow_pickle=True)

    start = int(data["start"])
    end = int(data["end"])

    if "log_tmrca_raw" in data:
        # New format: (n_items, n_reps, n_windows) with multiple blocks
        raw = data["log_tmrca_raw"]
        blocks = data["blocks"]
        n_blocks = len(blocks)
        n_reps = raw.shape[0]  # first dim appears to be reps based on shape
        windows_per_block = raw.shape[-1]

        # Reshape: raw is (n_reps, n_blocks*n_pairs, windows_per_block)
        # or (n_items, n_reps, windows_per_block)
        # Let's figure out the layout from index_map
        index_map = data["index_map"]

        # raw shape: (n_reps, n_items, n_windows) where n_items = n_blocks * n_pairs
        # Transpose to (n_items, n_reps, n_windows) for easier grouping
        raw_t = np.transpose(raw, (1, 0, 2))

        # Group by block, average over pairs and reps
        block_means = []
        for b in range(n_blocks):
            mask = index_map[:, 0] == b
            block_data = raw_t[mask]  # (n_pairs_in_block, n_reps, n_windows)
            block_mean = block_data.mean(axis=(0, 1))
            block_std = block_data.mean(axis=1).std(axis=0)
            block_means.append((block_mean, block_std))

        # Concatenate blocks into a continuous profile
        all_mean = np.concatenate([m[0] for m in block_means])
        all_std = np.concatenate([m[1] for m in block_means])
        n_windows_total = len(all_mean)
    else:
        # Old format: single block, 500 windows
        all_mean = data["pop_mean_log_tmrca"]
        all_std = data["pop_std_log_tmrca"]
        n_windows_total = len(all_mean)
        # Old format only covers first ~1Mb
        end = start + 1_000_000

    # Build position axis
    positions = np.linspace(start, end, n_windows_total)
    tmrca = np.exp(all_mean)
    tmrca_upper = np.exp(all_mean + all_std)
    tmrca_lower = np.exp(all_mean - all_std)

    ax.plot(positions / 1e6, tmrca, color="steelblue", linewidth=0.8)
    ax.fill_between(positions / 1e6, tmrca_lower, tmrca_upper,
                     color="steelblue", alpha=0.2)
    ax.set_title(f"{title}", fontsize=10, fontweight="normal", loc="left")
    ax.set_xlabel(f"{chrom} (Mb)", fontsize=8)
    ax.set_ylabel("TMRCA (generations)", fontsize=8)
    ax.tick_params(labelsize=7)

    # Mark the sweep trough
    min_idx = np.argmin(tmrca)
    ax.axvline(positions[min_idx] / 1e6, color="red", linewidth=0.5,
               linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(OUTFILE, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTFILE}")
plt.close()
