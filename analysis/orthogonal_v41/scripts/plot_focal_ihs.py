#!/usr/bin/env python3
"""Extract checked regional iHS data, or render the documentation's iHS tracks.

Extraction uses whole-chromosome normalization before selecting a region.
Run extraction/analysis on the data host through its batch scheduler. Plotting
can also be repeated from the small, published CSVs without the cohort inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_selscan_per_gene import load_selscan_table, normalize_within_bins

CASES = {
    "GRK2": {"chromosome": 11, "genes": ["GRK2"], "populations": ["GIH", "BEB", "IBS"]},
    "TREM2": {"chromosome": 6, "genes": ["TREML1", "TREM2"], "populations": ["IBS"]},
}
POPULATIONS = {
    "GIH": "Gujarati Indian",
    "BEB": "Bengali",
    "IBS": "Iberian",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(repository, output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "assembly": "GRCh38",
        "coordinate_convention": "1-based closed intervals",
        "flank_bp": 100000,
        "normalization": {
            "scope": "whole chromosome, separately per population, before regional extraction",
            "frequency": "VCF-coded allele; not verified ancestral/derived polarity",
            "bin_edges": np.linspace(0.0, 1.0, 21).tolist(),
            "closed": "right, include lowest endpoint",
            "minimum_finite_scores_per_bin": 50,
            "ddof": 0,
            "normalizer_sha256": sha256(Path(__file__).with_name("aggregate_selscan_per_gene.py")),
        },
        "cases": {},
    }
    for case, spec in CASES.items():
        chrom = spec["chromosome"]
        gene_path = repository / f"analysis/genome_wide/cache/genes/chr{chrom}_genes.tsv"
        genes = pd.read_csv(gene_path, sep="\t")
        selected = genes.loc[genes.gene_name.isin(spec["genes"]), ["gene_name", "start", "end"]]
        selected = selected.sort_values("start")
        assert len(selected) == len(spec["genes"]), (case, selected)
        start, end = int(selected.start.min()) - 100000, int(selected.end.max()) + 100000
        detail = {
            "chromosome": chrom,
            "window_start": start,
            "window_end": end,
            "genes": selected.to_dict("records"),
            "gene_annotation_sha256": sha256(gene_path),
            "tracks": [],
        }
        frames = []
        for pop in spec["populations"]:
            relative = f"analysis/orthogonal_v41/selscan/chr{chrom}_{pop}/ihs.ihs.out"
            source = repository / relative
            whole = normalize_within_bins(load_selscan_table(str(source), "ihs"), "ihs")
            region = whole.loc[whole.pos.between(start, end)].copy()
            assert not region.empty
            # Keep every returned record, including records sharing a position.
            region["source_row"] = region.index + 1
            region["population"] = pop
            region["chromosome"] = chrom
            frames.append(region.rename(columns={
                "pos": "position", "freq": "coded_allele_frequency", "ihs": "raw_ihs",
                "ihs_norm": "normalized_ihs", "ihs_norm_abs": "abs_normalized_ihs",
            })[["chromosome", "position", "population", "source_row", "coded_allele_frequency",
                "raw_ihs", "normalized_ihs", "abs_normalized_ihs"]])
            counts = {}
            for row in selected.itertuples():
                scores = region.loc[region.pos.between(row.start, row.end), "ihs_norm_abs"]
                finite = scores[np.isfinite(scores)]
                counts[row.gene_name] = {
                    "returned_rows": len(scores), "finite_scores": len(finite),
                    "scores_above_2": int((finite > 2).sum()),
                    "max_abs_normalized_ihs": float(finite.max()) if len(finite) else None,
                }
            detail["tracks"].append({
                "population": pop, "raw_source": relative, "raw_source_sha256": sha256(source),
                "chromosome_returned_rows": len(whole),
                "chromosome_finite_normalized_scores": int(np.isfinite(whole.ihs_norm).sum()),
                "region_returned_rows": len(region),
                "region_distinct_positions": int(region.pos.nunique()),
                "region_finite_normalized_scores": int(np.isfinite(region.ihs_norm).sum()),
                "gene_counts": counts,
            })
        target = output / f"{case.lower()}_ihs.csv"
        pd.concat(frames, ignore_index=True).to_csv(target, index=False, float_format="%.17g")
        detail.update(csv=target.name, csv_sha256=sha256(target))
        metadata["cases"][case] = detail
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Arial", "sans-serif"], "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#68717b", "axes.labelcolor": "#263238",
        "text.color": "#263238", "xtick.color": "#45525e", "ytick.color": "#45525e",
        "axes.linewidth": 0.7, "svg.fonttype": "none", "svg.hashsalt": "focal-ihs",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data / "provenance.json").read_text())
    validation = {}
    for case, detail in metadata["cases"].items():
        source = data / detail["csv"]
        assert sha256(source) == detail["csv_sha256"], "Plot source differs from manifest"
        df = pd.read_csv(source)
        finite = df.loc[np.isfinite(df.normalized_ihs)].copy()
        assert np.allclose(finite.normalized_ihs.abs(), finite.abs_normalized_ihs)
        populations = [track["population"] for track in detail["tracks"]]
        top = max(3, int(np.ceil(finite.abs_normalized_ihs.max() + .25)))
        fig, axes = plt.subplots(len(populations), 1, sharex=True, sharey=True,
                                 figsize=(8, 1.9 * len(populations) + 1.05), squeeze=False)
        fig.subplots_adjust(left=.10, right=.98, bottom=.62/fig.get_figheight(),
                            top=1-.60/fig.get_figheight(), hspace=.48)
        plotted = {}
        for index, (pop, ax) in enumerate(zip(populations, axes[:, 0])):
            subset = finite.loc[finite.population.eq(pop)].sort_values("position")
            track = detail["tracks"][index]
            assert len(subset) == track["region_finite_normalized_scores"]
            in_genes = np.zeros(len(subset), dtype=bool)
            for gene in detail["genes"]:
                mask = subset.position.between(gene["start"], gene["end"]).to_numpy()
                assert int(mask.sum()) == track["gene_counts"][gene["gene_name"]]["finite_scores"]
                in_genes |= mask
                ax.axvspan(gene["start"] / 1e6, gene["end"] / 1e6,
                           color="#e6eef3", linewidth=0, zorder=0)
            for mask, color, size in [(~in_genes, "#8799a6", 10), (in_genes, "#0072B2", 28)]:
                points = ax.scatter(subset.loc[mask, "position"] / 1e6,
                                    subset.loc[mask, "abs_normalized_ihs"],
                                    color=color, s=size, linewidths=0, zorder=3)
                assert len(points.get_offsets()) == int(mask.sum())
            ax.axhline(2, color="#64717a", linestyle=(0, (4, 4)), linewidth=.8, zorder=1)
            ax.set_title(f"{pop} · {POPULATIONS[pop]}", loc="left", fontsize=11, pad=8)
            ax.set_title(f"{len(subset):,} iHS scores", loc="right", fontsize=9, pad=8)
            ax.set_ylim(0, top)
            ax.set_ylabel("|normalized iHS|")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            plotted[pop] = len(subset)
        first = axes[0, 0]
        for index, gene in enumerate(detail["genes"]):
            center = (gene["start"] + gene["end"]) / 2e6
            offset = 0 if len(detail["genes"]) == 1 else (-.016 if index == 0 else .016)
            first.annotate(gene["gene_name"], xy=(center, 1), xycoords=("data", "axes fraction"),
                           xytext=(center + offset, 1.25), textcoords=("data", "axes fraction"),
                           ha="center", va="center", fontsize=11, fontstyle="italic",
                           arrowprops={"arrowstyle": "-", "color": "#68717b", "lw": .7})
        last = axes[-1, 0]
        last.set_xlim(detail["window_start"] / 1e6, detail["window_end"] / 1e6)
        last.xaxis.set_major_locator(MaxNLocator(nbins=5))
        last.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        last.set_xlabel(f"Chromosome {detail['chromosome']} position (Mb, GRCh38)", labelpad=10)
        stem = output / f"{case.lower()}_ihs"
        fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        plt.close(fig)
        validation[case] = {"source_sha256": sha256(source), "plotted_scores": plotted,
                            "svg_sha256": sha256(stem.with_suffix(".svg")),
                            "png_sha256": sha256(stem.with_suffix(".png"))}
    (output / "plot_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract")
    extraction.add_argument("--repository", type=Path, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    rendering = commands.add_parser("plot")
    rendering.add_argument("--data", type=Path, required=True)
    rendering.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.repository, args.output)
    else:
        plot(args.data, args.output)


if __name__ == "__main__":
    main()
