# Corrected findings · 2026-09-27

This versioned summary release accompanies the revised gene-resolution atlas.
It supersedes the earlier root-level numbered findings directories for current
interpretation: **143 loci, 1,123 duplication-masked genes, 19,119 gene rows**.
The historical deposit remains available and contains different results.

Use the Git tag `findings-2026-09-27` when citing this deposit. `MANIFEST.json`
records file checksums, row counts, source-code identity, and validation scope.
Compressed CSVs are ordinary gzip files and can be read directly by pandas.

## Data dictionary

| File | Interpretation |
| --- | --- |
| `genome_wide_ranks.csv.gz` | Full-precision gene atlas: identifiers, GRCh38 coordinates, 26 `*_tmrca` values, 26 `*_rank` fractions, duplication flags, descriptive scores. |
| `genome_wide_stats.csv.gz` | Compact identity, coordinate, rank-extrema and duplication-mask table. |
| `stage4_genes.csv` | 540 genes passing masking, rank, continental-replication and reference-exclusion rules. |
| `stage5_loci.csv` | 143 spatial reporting units, representative genes, member names, interval spans and focal populations. |
| `cluster_members.csv` | Gene-to-reporting-unit membership. |
| `reference_loci.csv` | The 23 heterogeneous reference genes; not a set of independent established sweeps. |
| `top50_sd_masked.csv` | First 50 genes after excluding duplication-masked genes, ordered by minimum rank. |
| `threshold_global_sensitivity.csv` | Full cascade at stage-2 thresholds 0.5%, 1%, 2%, 5%. |
| `threshold_focal_sensitivity.csv` | Focal-gene behavior under the threshold sensitivity analysis. |
| `threshold_sensitivity.csv` | Additional rank, replication and clustering configurations. |
| `descriptive_replication_scores.csv.gz` | Correlation-adjusted descriptive population scores, not p-values or FDR. |
| `population_rank_correlations.csv` | Spearman correlations among complete population-rank vectors. |
| `aggregation_comparison.csv` | Arithmetic/geometric rank comparisons for selected genes. |
| `heatmap_matrix_kya.csv`, `heatmap_row_order.csv` | Corrected coalescence summaries at 474 ancient-DNA lead windows; 28 years/generation. Row annotations do not identify causal genes. |
| `accuracy.csv` | 14 retained simulation configurations; pairwise accuracy summaries and execution/calibration metadata. |
| `runtime.csv` | 72 native-workflow timings: four series × six pair counts × three repetitions. |
| `clues_focal.csv` | All 13 focal CLUES2 fits with allele counts, nominal intervals and converter reconciliation counts. |

Gene coordinates are one-based, closed GRCh38 intervals. `*_tmrca` values are
geometric means of posterior mean TMRCA in **generations**, over retained
site/pair combinations; at least two sites per gene are required. Ranks are
ascending fractional ranks among finite genes in each population, computed
before duplication masking with average ties. A value of `0.01` means 1%.
Missing fields indicate unavailable values, not zeros. `sd_unique_bp_fraction`
uses the union of duplication intervals; `is_sd=True` at coverage ≥0.5.

The cascade uses minimum rank <0.01, all populations in at least one continental
group at rank <0.05, exclusion of genes overlapping the reference-gene ±500 kb
intervals, and merging of adjacent remaining intervals separated by ≤1 Mb.
Representatives minimize population rank, with gene ID breaking ties. The
`focal_population` is a descriptive label within a qualifying continental group.
A reporting unit is not necessarily an independent selective event.

## Provenance and reproduction

The atlas and calibrated benchmarks were retained from the corrected September
2026 analysis, with the atlas independently checked against all 572 retained
chromosome/population accumulators. Raw accumulator data are not included here.
The public generation code is:

- `analysis/genome_wide/infer_chromosome.py` and related campaign tools for inference;
- `analysis/genome_wide/build_candidates.py` and `sd_mask.py` for ranks, masks and candidates;
- `benchmarks/test_suite_stdpopsim/` for simulation accuracy;
- `benchmarks/pairwise_scaling/` for native runtime comparisons;
- `analysis/orthogonal_v41/scripts/` for haplotype summaries;
- `analysis/rerun/` and `analysis/relate_clues/` for related workflows.

From the repository root, run `python findings_database/verify_release.py`.
This uses only the standard library and verifies released file checksums, row
counts, population ranks, all representative identities, the candidate cascade,
threshold sensitivities and selected headline summaries. It does not independently
rerun the original preprocessing or inference and does not validate biological
causality. Original large inputs are needed for a full raw-data rerun.

## Interpretation and limits

The 143 loci are a descriptive screen of shallow coalescence, not a calibrated
selection test. Neutral/BGS false positives and selected-simulation detection
power for the full pipeline remain unmeasured. Demography, rates, ascertainment,
callability and relatedness can affect the summaries. Gene-body score absence
in iHS is not negative evidence for selection. Haplotype and coalescent summaries
from the same genomes are not independent confirmations.

Runtime ratios compare different native output formats with buffered writes;
they are not matched durable-storage comparisons. Simulation accuracy uses one
sequence per configuration, not 190 independent simulations. CLUES2 intervals
and statistics are nominal, model-dependent and unadjusted for post hoc marker
and locus selection. They do not establish causal variants or selected genes.

The release excludes raw VCFs, full dense outputs, intermediate inference
accumulators, complete CLUES2 trees/trajectories, and current recombination/BGS
context packages. Those exclusions are explicit to distinguish a summary deposit
from a complete reproduction archive.
