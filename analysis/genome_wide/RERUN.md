# Reproducible inference and candidate rerun

These CLIs read the supplied cache and write only to explicit new output roots.
They do not run Slurm or modify the original analysis. Use identical code and
explicit kernel settings for each host. GPU work on Betty must run through Slurm.

```sh
python analysis/genome_wide/infer_chromosome.py --chr 22 --populations YRI \
  --cache-dir /data/new-cache --samples /data/samples.txt \
  --output-dir /runs/run-id/genome/results \
  --core-block-sites 65536 --flank-sites 8192 --pair-chunk 512

python analysis/akbari_479_tmrca/infer_akbari_windows.py --chr 22 --populations YRI \
  --cache-dir /data/new-cache --samples /data/samples.txt \
  --output-dir /runs/run-id/akbari/results --leads /data/leads_grch38.tsv \
  --core-block-sites 65536 --flank-sites 8192 --pair-chunk 512 \
  --slice-half-bp 500000 --aggregate-half-bp 25000

python analysis/genome_wide/build_candidates.py \
  --results-dir /runs/run-id/genome/results \
  --sd-track /data/genomicSuperDups.txt.gz \
  --output-dir /runs/run-id/genome/postprocess
```

`GAMMA_CACHE_DIR`, `GAMMA_SAMPLES`, and `GAMMA_OUTPUT_DIR` can provide the first
three paths. The default physical mutation and recombination rates are
1.25e-8 and 1e-8 per base per generation. Each chromosome/population uses its
whole-chromosome mean within-diploid heterozygosity as theta, then
`Ne_hat = theta/(4*physical_mu)`. Both decoders receive the physical rates,
this Ne, and `auto_estimate_theta=False`; generation scaling is therefore
`2*Ne_hat = theta/(2*physical_mu)`. Regional slices reuse the full-chromosome
calibration. The denominator defaults to last chromosome position + 1, matching
the reference no-mask convention; `--sequence-length` can supply an explicit
contig length. This does not remove linked-selection/demographic confounding.

The scripts support `cache/parsed/chrN/{G,positions,sample_ids}.npy` with a
`READY.json` from `analysis/rerun/cache.py`, falling back to historical NPZ.
Extracted genotype arrays use true memory mapping. Completion markers record
calibration, block settings, source/input fingerprints, software environment
and CSV/NPZ checksums. `--resume` accepts only identical completed outputs;
partial or mismatched outputs must be inspected and moved aside explicitly.

Postprocessing reads raw floating-point NPZ accumulators, so printed rounding
never determines rank ties. By default all 22 x 26 completed inputs are required.
`--ranks-csv` permits independent postprocessing audits of an existing atlas.
The SD mask merges UCSC intervals before measuring unique base pairs and converts
GENCODE 1-based closed coordinates to UCSC 0-based half-open coordinates.
Candidate clustering uses true single linkage, including nested genes. The
23 reference genes are explicit in `build_candidates.py`. Outputs include all
ranks, SD coverage, reference loci, stages 4/5, cluster membership and the seven
threshold-sensitivity settings.

Galwey effective counts use the Spearman matrix calculated on genes complete in
all 26 populations. Per-continent scores are `max(rank)**n_eff`; the reported
five-continent score is `min(1, 5*min(score))`. These are descriptive heuristics,
not p-values; no FDR interpretation or calibrated significance is claimed.
Focal populations are chosen deterministically within the qualifying continent
with the smallest descriptive score. This differs from arbitrary first-continent
selection in some historical scripts and is recorded in output columns.

CPU regression tests:

```sh
python -m pytest -q analysis/genome_wide/tests/test_rerun.py
```
