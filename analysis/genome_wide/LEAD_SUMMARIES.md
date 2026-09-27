# Lead-variant summaries from one chromosome pass

The production genome-wide runner can summarize the Akbari lead windows during
exactly the same full-chromosome posterior pass as the gene summaries:

```sh
python analysis/genome_wide/infer_chromosome.py \
  --chr 22 --populations YRI --cache-dir /inputs/fresh_cache \
  --samples /inputs/samples.txt --output-dir /new/genome \
  --core-block-sites 65536 --flank-sites 8192 --pair-chunk 512 \
  --lead-variants /inputs/akbari_lead_variants_grch38.tsv --lead-half-bp 25000
```

The cache, GENCODE genes, and GRCh38 lead positions use one-based coordinates.
Intervals include both endpoints. A lead centered at `c` includes retained sites
from `max(1, c-25000)` through `c+25000`. Every pair's posterior comes from the
full chromosome under its population/chromosome calibration. No separate
+/-500 kb input slice or second GPU analysis is used. The finite core/flank
sensitivity of the full-chromosome decoder still requires its usual preflight.

Existing gene NPZ field names and shapes are unchanged. Additional `lead_*`
arrays contain IDs, centers, Akbari statistics, site counts, pair counts, linear
and log sums, squared log sums, minimum pair summaries, histograms, separate
histogram tails, and exact unbinned counts below 1000 generations for per-pair
geometric and arithmetic means. `lead_window_half_bp` records the interval
width, `lead_inference_context` identifies full-chromosome inference, and the
completion metadata includes the lead table's SHA256 in the input identity.
The option and window width participate in resume identity checks.

All posterior means must be finite and positive; values below one generation
are preserved. Float64 reductions occur within one gene/window at a time.
Histograms approximate arbitrary quantiles within their bins, with explicit
underflow and overflow; only the separately retained threshold counts are
unbinned. At least two retained sites are required for each gene/window.

Once the completed chromosome/population NPZ files are collected, extract lead
summaries without inference:

```sh
python analysis/genome_wide/lead_summaries.py \
  --results-dir /new/genome --output-dir /new/akbari
```

This writes `lead_tmrca.csv` (one lead/population per row),
`lead_tmrca_wide.csv` (lead rows and population geometric-mean columns), and
`chrN/POP.csv` tables. All times are generations. `manifest.json` records input
and output hashes, expected completeness, and the single lead-table identity.
The default requires all 22 chromosomes and 26 populations. Explicit
`--chromosomes`/`--populations` select a smaller requested scope;
`--allow-partial` labels missing files as a diagnostic incomplete collection.
Existing output directories are never replaced.

`infer_akbari_windows.py` is retained as a historical regional entrypoint and
is not needed for this production workflow. The combined runner regression
test uses known position-dependent posteriors, checks inclusive window
membership and all six sample pairs, confirms the gene array shapes stay
unchanged, and verifies that extraction performs no additional inference.
