#!/bin/bash
#SBATCH --job-name=tmrca-diag
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/genome_wide/logs/diagnose_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

python - <<'PY'
import pandas as pd
import numpy as np

NEW = "analysis/genome_wide/results/genome_wide_ranks.csv"
OLD = "analysis/genome_wide/old_genome_wide_stats.csv"

new = pd.read_csv(NEW)
print(f"=== New ranks file: {new.shape} ===")
print(f"Columns: {list(new.columns)[:10]}...")
print()

# Populations in order
POPS = ["ACB","ASW","BEB","CDX","CEU","CHB","CHS","CLM","ESN","FIN","GBR","GIH",
        "GWD","IBS","ITU","JPT","KHV","LWK","MSL","MXL","PEL","PJL","PUR","STU","TSI","YRI"]
SUPERPOPS = {
    "AFR": ["YRI","LWK","GWD","MSL","ESN","ACB","ASW"],
    "EUR": ["CEU","TSI","FIN","GBR","IBS"],
    "EAS": ["CHB","JPT","CHS","CDX","KHV"],
    "SAS": ["GIH","PJL","BEB","STU","ITU"],
    "AMR": ["MXL","PUR","CLM","PEL"],
}

print("=== Known sweep investigation ===")
# Known sweeps with their expected population(s)
expected = {
    "SLC24A5": ("EUR", "light skin"),
    "HERC2":   ("EUR", "eye color"),
    "LCT":     ("EUR", "lactase persistence"),
    "MCM6":    ("EUR", "lactase persistence"),
    "EDAR":    ("EAS", "ectodysplasin"),
    "EPAS1":   ("EAS", "hypoxia"),
    "ADH1B":   ("EAS", "alcohol"),
    "TRPV6":   ("EAS", "calcium"),
    "KITLG":   ("EUR/EAS", "skin/hair"),
    "OCA2":    ("EUR", "eye color"),
    "TYRP1":   ("EUR", "skin"),
    "HBB":     ("AFR", "sickle cell"),
    "DARC":    ("AFR", "malaria P. vivax"),
    "APOL1":   ("AFR", "trypanosomiasis"),
    "CD36":    ("AFR", "malaria"),
    "G6PD":    ("AFR", "malaria"),
}

for gene, (superpop, label) in expected.items():
    row = new[new["gene_name"] == gene]
    if row.empty:
        print(f"{gene:10s} [{superpop}] {label}: NOT FOUND")
        continue
    row = row.iloc[0]
    # Get TMRCAs for all pops
    tmrca_cols = [f"{p}_tmrca" for p in POPS]
    rank_cols = [f"{p}_rank" for p in POPS]
    tmrcas = row[tmrca_cols].values.astype(float)
    ranks = row[rank_cols].values.astype(float)

    # Find lowest by each superpop
    best_per_superpop = {}
    for sp, sp_pops in SUPERPOPS.items():
        sp_ranks = [(p, row[f"{p}_rank"]) for p in sp_pops if not pd.isna(row[f"{p}_rank"])]
        if sp_ranks:
            best_pop, best_rank = min(sp_ranks, key=lambda x: x[1])
            best_per_superpop[sp] = f"{best_pop}:{best_rank:.3f}"

    min_rank = row["min_rank"] if "min_rank" in row else np.nanmin(ranks)
    min_pop = row["min_pop"] if "min_pop" in row else POPS[np.nanargmin(ranks)]
    summary = " ".join(f"{sp}={v}" for sp, v in best_per_superpop.items())
    print(f"{gene:10s} [{superpop}] min={min_rank:.4f} ({min_pop}) | {summary}")

print()
print("=== Raw TMRCA values for HBB (sickle cell) ===")
hbb = new[new["gene_name"] == "HBB"]
if not hbb.empty:
    row = hbb.iloc[0]
    print(f"HBB chr{row['chr']}:{row['start']}-{row['end']}")
    for sp, sp_pops in SUPERPOPS.items():
        for p in sp_pops:
            col = f"{p}_tmrca"
            if col in row:
                v = row[col]
                r = row[f"{p}_rank"]
                print(f"  {p} ({sp}): TMRCA={v:.0f}, rank={r:.4f}")

print()
print("=== Raw TMRCA values for SLC24A5 (light skin, EUR) ===")
slc = new[new["gene_name"] == "SLC24A5"]
if not slc.empty:
    row = slc.iloc[0]
    print(f"SLC24A5 chr{row['chr']}:{row['start']}-{row['end']}")
    for sp, sp_pops in SUPERPOPS.items():
        for p in sp_pops:
            col = f"{p}_tmrca"
            if col in row:
                v = row[col]
                r = row[f"{p}_rank"]
                print(f"  {p} ({sp}): TMRCA={v:.0f}, rank={r:.4f}")

print()
print("=== Raw TMRCA values for LCT (lactase, EUR) ===")
lct = new[new["gene_name"] == "LCT"]
if not lct.empty:
    row = lct.iloc[0]
    print(f"LCT chr{row['chr']}:{row['start']}-{row['end']}")
    for sp, sp_pops in SUPERPOPS.items():
        for p in sp_pops:
            col = f"{p}_tmrca"
            if col in row:
                v = row[col]
                r = row[f"{p}_rank"]
                print(f"  {p} ({sp}): TMRCA={v:.0f}, rank={r:.4f}")

# Check distribution of TMRCA values per population - maybe some pops have shifted scales
print()
print("=== Per-population TMRCA distribution ===")
for p in POPS:
    col = f"{p}_tmrca"
    if col in new.columns:
        v = new[col].dropna()
        if len(v) > 0:
            print(f"  {p}: n={len(v)}, median={v.median():.0f}, "
                  f"p5={v.quantile(0.05):.0f}, p95={v.quantile(0.95):.0f}")
PY
