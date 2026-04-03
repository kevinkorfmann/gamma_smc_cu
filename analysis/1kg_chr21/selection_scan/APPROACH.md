# Selection scan via within-population TMRCA ranking

## The problem with raw population comparisons

Comparing TMRCA at a gene between populations picks up demography first:
African populations have deeper coalescence at *every* gene due to larger
long-term Ne, not because of selection. A gene like CRYAA shows up as
"top differential" simply because it's a gene where the demographic
signal is cleanest — not because anything interesting happened there.

## The fix: within-population gene ranking

For each population independently:
1. Rank all 214 genes by their mean TMRCA within that population
2. Convert ranks to percentiles (0 = shallowest coalescence, 1 = deepest)

Now compare percentiles across populations:
- A gene at the 50th percentile in every population = demographic signal only
- A gene at the 95th percentile in YRI but 50th in CEU = something is
  maintaining diversity at this locus specifically in Africans (balancing selection?)
- A gene at the 5th percentile in CEU but 50th in YRI = recent sweep in Europeans

The percentile normalizes out the population-level mean, isolating
gene-specific deviations.

## What to look for

- **Recent sweep**: gene with unusually LOW rank (recent coalescence) in
  one population but normal rank in others. The pairs within that
  population coalesced recently at this gene because a beneficial allele
  swept through.

- **Balancing selection**: gene with unusually HIGH rank (deep coalescence)
  in one or all populations. Diversity is maintained by selection.

- **Background selection / linked selection**: genes in low-recombination
  regions may have systematically low ranks across all populations.

## Output

- `gene_rank_heatmap.png`: heatmap of gene rank percentiles across populations
- `rank_outliers.png`: genes with the largest rank differences between populations
- `top_sweep_candidates.csv`: genes ranked by within-population z-score
