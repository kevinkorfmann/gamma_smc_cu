"""Genes of interest for the v4.1 orthogonal validation work.

Each entry is (gene_name, chromosome, focal_population, group).
group is one of: novel, positive, neutral.

The 5 neutral controls are picked dynamically as the 5 genes whose
min within-population rank is closest to 0.50 (one per superpopulation),
chosen from the new genome_wide_stats.csv.
"""

import os
import pandas as pd

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
STATS = os.path.join(REPO, "analysis/genome_wide/results/genome_wide_stats.csv")
SD_FLAG = os.path.join(REPO, "analysis/genome_wide/postprocess/genes_sd_flag.csv")

NOVEL = [
    ("GRK2",     11, "GIH", "novel"),
    ("BPIFA2",   20, "GIH", "novel"),
    ("SLC6A15",  12, "CHS", "novel"),
    ("CCDC92",   12, "CDX", "novel"),
    ("CLEC6A",   12, "CDX", "novel"),
]

POSITIVE = [
    ("SLC24A5",  15, "GBR", "positive"),
    ("LCT",       2, "CEU", "positive"),
    ("EDAR",      2, "CHB", "positive"),
    ("ABCC11",   16, "CHB", "positive"),
    ("KITLG",    12, "MXL", "positive"),
]

SUPERPOPS = {
    "AFR": ["YRI","LWK","GWD","MSL","ESN","ACB","ASW"],
    "EUR": ["CEU","TSI","FIN","GBR","IBS"],
    "EAS": ["CHB","JPT","CHS","CDX","KHV"],
    "SAS": ["GIH","PJL","BEB","STU","ITU"],
    "AMR": ["MXL","PUR","CLM","PEL"],
}


def neutral_controls():
    """Pick 5 neutral controls: per superpop, the gene whose min_rank is
    closest to 0.50. Excludes SD-flagged genes and genes already in
    NOVEL/POSITIVE."""
    stats = pd.read_csv(STATS)
    sd = pd.read_csv(SD_FLAG)
    sd_set = set(sd[sd["is_sd"]]["gene_name"])
    seen = {g[0] for g in NOVEL + POSITIVE}

    chosen = []
    used_chromosomes = set()
    for sp in ["AFR", "EUR", "EAS", "SAS", "AMR"]:
        sp_pops = SUPERPOPS[sp]
        # genes whose min_pop is in this superpop
        sub = stats[stats["min_pop"].isin(sp_pops)].copy()
        sub = sub[~sub["gene_name"].isin(sd_set)]
        sub = sub[~sub["gene_name"].isin(seen)]
        # exclude unannotated Ensembl gene IDs
        sub = sub[~sub["gene_name"].str.startswith("ENSG")]
        sub["abs_dev"] = (sub["min_rank"] - 0.5).abs()
        sub = sub.sort_values("abs_dev")
        # Pick the closest-to-0.5 gene whose chromosome we haven't used
        for _, r in sub.iterrows():
            if r["chr"] in used_chromosomes:
                continue
            chosen.append((str(r["gene_name"]), int(r["chr"]),
                           str(r["min_pop"]), "neutral"))
            seen.add(r["gene_name"])
            used_chromosomes.add(int(r["chr"]))
            break
    return chosen


def all_genes():
    return NOVEL + POSITIVE + neutral_controls()


if __name__ == "__main__":
    for g in all_genes():
        print(*g, sep="\t")
