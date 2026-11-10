# Genome-wide rank diagnostics and haplotype comparisons

This analysis addresses genomic covariate dependence and complementary
haplotype-statistic information in the corrected September 2026 atlas.
Run data analysis and haplotype inference through Slurm on Betty. Inputs,
outputs and logs are separate from the reusable source files.

## Execution order and inputs

`stage_inputs.py --workspace PATH --output NEW_INPUTS` freezes the corrected
manuscript atlas, 572 selected inference accumulators, BGS annotations, historical
iHS summaries and exact stdpopsim recombination maps. It expects the original
workspace layout; a checkout of this source directory alone does not contain
those data. The separate analysis recovery runbook explains their regeneration.

Run `covariates.py` on those inputs; run `haplotype_job.py` for each
chromosome/population using the strict chromosome cache and original sample
mapping. Then run `normalize_haplotypes.py` per population, `compare.py`, and
`audit_results.py`. `format_covariate_table.py` and `plot_figure_2x2.py` consume the
completed summaries. Each CLI exposes its required paths with `--help`.
The `slurm_*.sh` files are the original Betty launch recipes; adapt filesystem
paths and allocation settings for another cluster.

CPU-only definition checks, from the software repository root:

```bash
python -m unittest discover -s analysis/rank_diagnostics -p 'test_*.py'
```

These need NumPy, pandas, SciPy and scikit-allel. Running them does not generate
or validate the full empirical results.

## Analysis specification (fixed before inspecting results)

* Preserve the corrected atlas's ranks, 19,119 gene identities, coordinates
  and unique-base segmental-duplication mask. Report each of 26 populations
  separately; population labels are not independent replicates.
* Use retained segregating-site counts from the exact 572 full-chromosome
  accumulator files. Verify reconstructed geometric means against the atlas.
* Covariates: gene-body length, retained-site count, overlap-weighted deCODE
  GRCh38 recombination rate, and Buffalo–Kern YRI CADD6/deCODE B-prime.
  Require at least 95% map coverage, preserve gaps, and exclude SD genes.
  CEU B-prime, Barroso regulatory/phastCons B and Pyrho-YRI recombination
  are annotation sensitivities, not independent biological replications.
* Report marginal and partial Spearman correlations; the latter residualize
  ranked variables against the other three ranked covariates. Use genomic
  chromosome jackknife uncertainty rather than independent-gene p-values.
* Report descriptive focal ranks among up to 500 covariate-nearest genes on
  other chromosomes. Matching uses four covariate percentile coordinates;
  maximum per-coordinate distance is 0.20 and at least 100 controls are
  required. These are sensitivity summaries, not selection probabilities.
* Re-normalize the existing manuscript iHS raw scores in the documented
  chromosome/population frequency bins. Missing normalized scores remain
  missing. Count returned and finite scores separately. These archived iHS
  inputs retain their historical genotype preparation; new XP-EHH uses the
  corrected complete biallelic phased SNP cache. Disclose that distinction.
  Input-audit amendment before any result inspection: historical iHS files
  contain duplicate physical positions. Normalize the full original score
  stream, then exclude every duplicated position and intersect core positions
  with the corrected SNP cache. Record the removed count. Historical haplotype
  construction is unchanged, so this is alignment of scored cores, not a fresh
  iHS inference from the corrected haplotypes.
* Calculate whole-chromosome XP-EHH for each population against YRI; use CEU
  as the explicitly labelled reference for YRI. Retain sites segregating in
  the pooled two-population comparison, including sites fixed within either
  population. No MAF cutoff. Physical-distance integration corresponds to a
  constant recombination map; min EHH 0.05, gap scale 20 kb, maximum gap
  200 kb, no edge-truncated scores. Standardize all finite XP-EHH scores
  across the 22 autosomes separately for each population/reference contrast.
* Summaries distinguish closed gene bodies and disjoint flanks of 100 kb
  per side; 500-kb flanks are a sensitivity. All EHH inference uses the
  whole chromosome. Primary summaries are fractions |normalized iHS| > 2
  and normalized XP-EHH > 2 (positive means longer focal haplotypes).
  Maximum-score summaries provide a sensitivity.
* Compare correlations on pairwise-complete genes and show score coverage
  separately. Missing scores must never be coded as zero or weak evidence.
* Compare low-TMRCA genes (original rank <1%) with up to 20 non-low-TMRCA
  controls on other chromosomes, exactly matched on gene-body iHS
  availability, with covariate percentile caliper 0.15; require >=5 controls.
  Report computability and empirical top-5% haplotype-summary status
  separately. Report unmatched genes rather than relaxing thresholds.

Retained-site count is partly a consequence of diversity and selection.
Conditioning on it can remove biological signal; adjusted associations and
matched ranks do not establish absence of confounding or causal effects.
Reference recombination/BGS maps are shared genomic annotations, not maps
fitted separately to all study populations. Neither gene-level ranking nor
agreement between statistics supplies calibrated selection FDR or power.

Source documentation:
https://scikit-allel.readthedocs.io/en/stable/stats/selection.html
https://popsim-consortium.github.io/stdpopsim-docs/stable/catalog.html
https://doi.org/10.1371/journal.pgen.1011144
