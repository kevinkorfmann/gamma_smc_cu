#!/bin/bash
#SBATCH --job-name=tmrca-analyze
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/genome_wide/logs/analyze_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

RESULTS_DIR="analysis/genome_wide/results"

echo "=========================================="
echo "Step 1: Aggregate (primary stat: geom_mean)"
echo "=========================================="
python analysis/genome_wide/aggregate.py

echo ""
echo "=========================================="
echo "Step 2: Top 30 candidates (geom_mean)"
echo "=========================================="
python -c "
import pandas as pd
df = pd.read_csv('${RESULTS_DIR}/genome_wide_stats.csv')
print(f'Total genes: {len(df)}')
print()
print('Top 30 lowest min_rank:')
print(df.head(30).to_string(index=False))
print()
print('Distribution of min_rank:')
print(df['min_rank'].describe())
"

echo ""
echo "=========================================="
echo "Step 3: Known-sweep validation (geom_mean)"
echo "=========================================="
python -c "
import pandas as pd
df = pd.read_csv('${RESULTS_DIR}/genome_wide_stats.csv')
controls = ['LCT','MCM6','EDAR','EPAS1','SLC24A5','TRPV6','ADH1B','HERC2',
            'OCA2','TYRP1','KITLG','G6PD','DARC','APOL1','CD36','HBB','FY']
hit = df[df['gene_name'].isin(controls)].sort_values('min_rank')
print(f'Found {len(hit)} of {len(controls)} known-sweep genes')
print()
print(hit[['gene_name','chr','start','end','min_rank','min_pop']].to_string(index=False))
print()
n_below_10 = (hit['min_rank'] < 0.10).sum()
n_below_05 = (hit['min_rank'] < 0.05).sum()
n_below_01 = (hit['min_rank'] < 0.01).sum()
print(f'{n_below_10}/{len(hit)} below 10% threshold')
print(f'{n_below_05}/{len(hit)} below 5% threshold')
print(f'{n_below_01}/{len(hit)} below 1% threshold')
"

echo ""
echo "=========================================="
echo "Step 4: Novel findings (geom_mean)"
echo "=========================================="
python -c "
import pandas as pd
df = pd.read_csv('${RESULTS_DIR}/genome_wide_stats.csv')
novel = ['GRK2','ADRBK1','BPIFA2','SLC6A15','CCDC92']
hit = df[df['gene_name'].isin(novel)].sort_values('min_rank')
print(f'Found {len(hit)} of {len(novel)} novel-candidate genes')
print()
print(hit[['gene_name','chr','start','end','min_rank','min_pop','rank_range']].to_string(index=False))
"

echo ""
echo "=========================================="
echo "Step 5: Compare new vs archived (2026-04-09) run"
echo "=========================================="
python -c "
import pandas as pd
new = pd.read_csv('${RESULTS_DIR}/genome_wide_stats.csv')
old = pd.read_csv('analysis/genome_wide/old_genome_wide_stats.csv')
if old.columns[0] == 'Unnamed: 0':
    old = old.rename(columns={'Unnamed: 0': 'gene_name'})
print(f'New run genes: {len(new)}')
print(f'Old run genes: {len(old)}')
merged = new.merge(old, on='gene_name', suffixes=('_new','_old'))
print(f'Shared genes: {len(merged)}')
print()
corr = merged[['min_rank_new','min_rank_old']].corr().iloc[0,1]
print(f'Correlation (min_rank new vs old): {corr:.4f}')

new_below = merged['min_rank_new'] < 0.10
old_below = merged['min_rank_old'] < 0.10
agree_below = (new_below & old_below).sum()
new_only = (new_below & ~old_below).sum()
old_only = (~new_below & old_below).sum()
print(f'Genes below 10% in both: {agree_below}')
print(f'New only (below 10%): {new_only}')
print(f'Old only (below 10%): {old_only}')
print()
print('Top 20 new candidates with old ranks:')
sorted_new = merged.sort_values('min_rank_new').head(20)
cols = ['gene_name','min_rank_new','min_pop_new','min_rank_old','min_pop_old']
cols = [c for c in cols if c in sorted_new.columns]
print(sorted_new[cols].to_string(index=False))
"

echo ""
echo "=========================================="
echo "Step 6: Alternative statistics from NPZ"
echo "=========================================="
for STAT in p5 p10 min frac_below_1000; do
    echo ""
    echo "--- Stat: ${STAT} ---"
    python analysis/genome_wide/reaggregate_from_npz.py --stat ${STAT}
done

echo ""
echo "=========================================="
echo "All steps complete"
echo "=========================================="
ls -la ${RESULTS_DIR}/genome_wide_*.csv
