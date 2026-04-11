# Primary statistic recommendation

We compute five per-gene per-population summaries from the per-pair
TMRCA distribution within each gene region:

| stat | definition | strength | weakness |
|---|---|---|---|
| geom_mean | exp(mean(log TMRCA)) over all (pair, site) tuples | matches old run; balanced view | dilutes very partial sweeps |
| p5 | 5th percentile of per-pair geometric-mean TMRCA | catches sweeps at frequency >= sqrt(0.05) ~= 22% | noisier (single-point) |
| p10 | 10th percentile | similar to p5 with more support | similar |
| min | youngest per-pair TMRCA | maximum sensitivity | very noisy, single point |
| frac_below_1000 | fraction of pairs with mean TMRCA < 1000 generations | direct sweep-haplotype frequency | threshold-dependent |

**Recommendation: report `geom_mean` as the headline statistic and
`frac_below_1000` as the supporting statistic.**

Rationale:
- `geom_mean` is what the original archive pipeline used (verified in
  `archive_2026_04_09/genome_wide_local/cxt/run_cxt_region.py`,
  via `np.exp(log_tmrca_raw.mean(...))`). Comparing to the archive
  is meaningful only with `geom_mean`.
- `frac_below_1000` is a direct, interpretable measure of sweep
  haplotype frequency: "what fraction of within-population pairs in
  this gene have an inferred TMRCA below 1000 generations?". A high
  value means many pairs share a recent common ancestor at this gene,
  i.e. a sweep haplotype is at high frequency.
- Reporting both lets the reader see two complementary views: a
  centrality-based estimate (geom_mean) and a frequency-based
  estimate (frac_below_1000).
- `p5` / `p10` / `min` are kept as additional sanity checks. `p5`
  and `p10` are most sensitive for partial sweeps but tend to be
  noisier on small genes with few segregating sites.
