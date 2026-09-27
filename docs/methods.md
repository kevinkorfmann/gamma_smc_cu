# Methods

## Time calibration

Gamma-SMC CUDA approximates each pair's local coalescence-time posterior by a
Gamma distribution. The implementation builds on
[Gamma-SMC](https://github.com/regevs/gamma_smc); the GPU implementation and
the biological interpretation of its outputs need separate validation.

By default, `auto_estimate_theta=True` estimates the scaled mutation rate from
the mean heterozygous-site count across adjacent diploid haplotype pairs,
divided by the last retained coordinate plus one. It sets the scaled
recombination rate using the supplied `rho / mu` ratio. The model uses a
constant recombination rate; these APIs do not accept a local genetic map.

Output time is calibrated using `theta / (2 × physical_mu)` generations per
coalescent unit. `physical_mu` defaults to `mu` in this mode. `Ne` is then a
bookkeeping parameter, not an independently estimated demographic history.
Changing the physical mutation rate changes the reported time scale.

For known supplied rates, use `auto_estimate_theta=False`. Fixed mode normally
uses `2 × Ne` generations per coalescent unit. If supplying externally estimated
effective rates, also supply the intended `physical_mu`; the output metadata
records the resulting conversion. Rate estimation falls back to supplied rates
when adjacent diploid heterozygosity cannot be estimated. Inspect metadata
and preserve it with every result.

There is no callable-base mask argument. Cropped intervals, missing sequence,
relatedness, phase errors and local rate variation can affect inference and
calibration. Full-chromosome analysis avoids an accidental cropped-window span
convention; it does not remove those modeling limitations.

## Context and memory

`infer`, `iter_infer_batches`, `export_dense` and `RegionMomentContext` use the
full supplied sequence context. Checkpoint/replay recomputes forward states
while bounding the stored state; it does not add site-block boundaries.
`infer_blockwise` with multiple site blocks instead approximates context using
flanks. Its accuracy depends on flank length and the dataset.

Dense float32 means require `4 × retained sites × requested pairs` bytes,
before headers, coordinates, pair indices and working memory. Streaming bounds
working memory but cannot make a dense output small. Interval summaries are
preferable when dense posteriors are not needed.

## Gene summaries and ranks

The corrected atlas takes the geometric mean of site-level posterior means
over gene-body retained sites and within-population haplotype pairs. At least
two retained sites are required. It does not impose a one-generation floor.
Each population's finite gene summaries are ranked in ascending order with
average ranks for ties. Percentile denominators are population-specific.

Segmental-duplication coverage uses the union of overlapping UCSC intervals,
avoiding double-counted bases. Genes with at least 50% covered bases are masked
after ranking. Candidate clustering describes spatial reporting units and
does not infer linkage blocks or independent selection events.

The source of these rules is
[`build_candidates.py`](https://github.com/kevinkorfmann/gamma_smc_cu/blob/main/analysis/genome_wide/build_candidates.py).
Descriptive population-replication scores account for rank correlations in
their formula, but are not calibrated p-values or FDR estimates.

## iHS and nSL comparisons

The retained runs used **selscan v3.0.0**, separately per chromosome and
population, with `--ihs` or `--nsl`, `--hap`, `--map`, `--out`, and `--threads 4`.
The writer preserved VCF reference/alternate 0/1 coding and supplied a constant
1 cM/Mb map. It removed population-monomorphic sites. Logs confirm a 0.05 MAF
filter, removal of low-frequency variants from haplotype construction,
EHH cutoff 0.05, gap scale 20 kb and maximum gap 200 kb. Edge-truncated
calculations were skipped. Default maximum extensions were 1 Mb for iHS
and 100 sites for nSL. Other settings used that version's defaults.

Raw scores were normalized separately per chromosome/population in 20
frequency bins of width 0.05. A bin needed at least 50 finite raw scores and
positive finite population standard deviation (`ddof=0`). Frequency refers to
the coded allele, **not verified derived-allele frequency**. The tool does not
infer ancestral polarity; see the [selscan documentation](https://github.com/szpiech/selscan).

For each closed gene interval without flanks, summaries were maximum absolute
normalized score and fraction of finite normalized scores exceeding 2 in
absolute value. Unnormalizable sites are excluded from that fraction's
denominator. The historical `n_ihs_sites` and `n_nsl_sites` columns count returned
raw-score rows and need not equal finite normalized-score counts. Genes with
no rows for either statistic are omitted. Descending ranks use average ties,
place missing summaries at the bottom, and divide by the number of gene rows
retained for that population.

These are gene summaries of the retained runs; they are not interchangeable
with published window-based iHS scans, and missing scores are not negative
evidence for selection.

## Reading the findings

The current screen retains 17,996 genes after duplication masking, 620 after
the minimum-rank <1% rule, 579 after continental replication, and 540 after
reference-locus exclusion. Clustering gives 143 reporting loci (82 multi-gene
clusters and 61 singletons). Changing the rank cutoff to 0.5%, 2% or 5% gives
58, 233 or 277 loci. TREM2 is a separate case study and does not pass the 1% rule.

These empirical ranks prioritize shallow-coalescence regions; selection-specific
false-positive rates and detection power for the full pipeline remain uncalibrated.
Recombination/BGS tracks provide context, and statistics from the same genotypes
are not independent confirmations. Historical sharing intervals condition on
haplotype resampling. CLUES2 case-study intervals and statistics are nominal and
unadjusted for post hoc locus/marker selection.

The runtime comparison uses three timed runs after warmup, with different native
output formats and buffered writes. A separate synchronized CUDA export took
86.10 seconds for a 79.18 GB float32 payload. Accuracy uses one simulated 5 Mb
sequence per configuration and 190 pairs from 20 haplotypes; pairs are not
independent simulation replicates. The release preserves the supplied mutation
rates, which differ from the current model catalogue in 11 configurations.

The dated release contains checked summary tables and code. Raw cohort VCFs,
full dense posteriors and all intermediate inference outputs are not bundled.
The earlier deposit (165 loci and 1,296 duplication-masked genes) is historical;
the corrected release contains 143 loci and 1,123 masked genes.
