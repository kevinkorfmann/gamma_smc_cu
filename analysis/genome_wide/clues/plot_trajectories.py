#!/usr/bin/env python3
"""
Plot CLUES allele frequency trajectories for known + novel selection signals.
Creates a multi-panel figure showing how swept alleles changed frequency through time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os

# Generation time in years
GEN_TIME = 29

def load_clues_output(prefix):
    """Load CLUES output: epochs, freqs, posterior."""
    epochs = np.load(f"{prefix}.epochs.npy")
    freqs = np.load(f"{prefix}.freqs.npy")
    post = np.load(f"{prefix}.post.npy")
    return epochs, freqs, post

def compute_trajectory(epochs, freqs, post):
    """
    Compute the maximum a posteriori frequency trajectory
    and credible intervals from the CLUES posterior.

    post is log-posterior: shape (n_epochs, n_freq_bins)
    freqs: frequency bin centers
    epochs: time bins (in generations, going into the past)
    """
    # Convert from log space
    posterior = np.exp(post - np.max(post, axis=1, keepdims=True))
    # Normalize each epoch
    posterior = posterior / posterior.sum(axis=1, keepdims=True)

    # MAP trajectory
    map_idx = np.argmax(posterior, axis=1)
    map_traj = freqs[map_idx]

    # Mean trajectory
    mean_traj = np.sum(posterior * freqs[np.newaxis, :], axis=1)

    # 95% credible intervals
    ci_low = np.zeros(len(epochs))
    ci_high = np.zeros(len(epochs))
    for i in range(len(epochs)):
        cumsum = np.cumsum(posterior[i])
        ci_low[i] = freqs[np.searchsorted(cumsum, 0.025)]
        ci_high[i] = freqs[np.searchsorted(cumsum, 0.975)]

    return mean_traj, map_traj, ci_low, ci_high

def plot_single_trajectory(ax, prefix, gene_name, color, label_extra=""):
    """Plot a single gene's frequency trajectory on the given axes."""
    epochs, freqs, post = load_clues_output(prefix)
    mean_traj, map_traj, ci_low, ci_high = compute_trajectory(epochs, freqs, post)

    # Convert generations to years ago (kya)
    times_kya = epochs * GEN_TIME / 1000

    # Plot credible interval
    ax.fill_between(times_kya, ci_low, ci_high, alpha=0.2, color=color)
    # Plot mean trajectory
    ax.plot(times_kya, mean_traj, color=color, linewidth=2, label=gene_name)

    ax.set_xlim(max(times_kya), 0)  # Time runs right to left (past → present)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Derived allele frequency", fontsize=10)
    ax.set_xlabel("Time (thousand years ago)", fontsize=10)

    # Add gene label
    if label_extra:
        ax.set_title(f"{gene_name} — {label_extra}", fontsize=11, loc='left', fontweight='normal')
    else:
        ax.set_title(gene_name, fontsize=11, loc='left', fontweight='normal')

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.grid(axis='y', alpha=0.2)

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_fig = sys.argv[2] if len(sys.argv) > 2 else "clues_trajectories.pdf"

    # Define loci
    loci = [
        ("LCT", "Lactase persistence (EUR)", "#2166ac"),
        ("SLC24A5", "Skin pigmentation (EUR)", "#b2182b"),
        ("EDAR", "Hair/sweat morphology (EAS)", "#1b7837"),
        ("GRK2", "Cardiovascular regulation (SAS+EUR) — NOVEL", "#e08214"),
    ]

    # Check which results exist
    available = []
    for gene, desc, color in loci:
        prefix = os.path.join(results_dir, f"{gene}_clues")
        if os.path.exists(f"{prefix}.post.npy"):
            available.append((gene, desc, color, prefix))
        else:
            print(f"WARNING: No CLUES output for {gene}, skipping")

    if not available:
        print("ERROR: No CLUES results found")
        sys.exit(1)

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]

    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']

    for i, (gene, desc, color, prefix) in enumerate(available):
        ax = axes[i]
        plot_single_trajectory(ax, prefix, gene, color, label_extra=desc)

        # Panel label
        ax.text(-0.08, 1.05, f"$\\bf{{{panel_labels[i]}}}$",
                transform=ax.transAxes, fontsize=14, va='bottom')

    plt.tight_layout()

    # Save
    fig.savefig(output_fig, dpi=300, bbox_inches='tight')
    fig.savefig(output_fig.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_fig}")
    print(f"Saved: {output_fig.replace('.pdf', '.png')}")

if __name__ == "__main__":
    main()
