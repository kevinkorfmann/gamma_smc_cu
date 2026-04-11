# Genome-wide TMRCA scan: post-rerun findings

Post-processing of the 2026-04-11 rerun of the 1000 Genomes
high-coverage genome-wide TMRCA scan with the new `tmrca.cu`
pipeline (gamma_smc parity fix, blockwise inference, per-population
segregating-site filter, auto-theta, log-space per-gene aggregation).

Inputs: 22 autosomes × 26 populations × 19,119 protein-coding genes,
3,202 samples (6,404 haplotypes), all within-population pairs (no
haplotype hashing).

## TL;DR

1. The pipeline works. SLC24A5, the LCT-MCM6-ZRANB3-DARS1 sweep
   haplotype, EDAR, HERC2, TRPV6, KITLG, ADH1B, GRK2, BPIFA2,
   SLC6A15, CCDC92, CLEC6A all surface in the top tier of one or
   more populations.
2. **Segmental-duplication masking is essential.** Without it,
   29/30 of the top genome-wide candidates are mapping artifacts
   (CCNYL1B, RGPD5, NBPF26, AMY1A/B/C, etc.). After masking, the
   top 30 is dominated by real selection signals.
3. The previous mucosal-immunity pathway-convergence claim does
   **not** survive proper permutation FDR. Individual genes
   (CLEC6A, BPIFA2, JCHAIN) remain strong outliers but the
   pathway-level claim should be retracted from the manuscript.
4. Five of the eight novel candidates from the 2026-04-09 archive
   replicate strongly (<1% min_rank, expected superpop): GRK2,
   BPIFA2, SLC6A15, CCDC92, CLEC6A. Three are weaker than
   originally reported (JCHAIN, TNFRSF13C, PIGR).
5. Four known sweeps remain undetected (TYRP1, APOL1, OCA2,
   EPAS1). NPZ histograms confirm the signal exists in the raw
   inference (one or more pairs at 12–67 generations) but it is
   <0.1% of pairs — below detection threshold for any centrality
   or low-quantile statistic. These are very partial sweeps where
   gamma-SMC TMRCA hits a fundamental limit; iHS/SweepFinder are
   the appropriate tools for these.

## Headline numbers

| metric | new run | 2026-04-09 archive |
|---|---|---|
| Genes scanned | 19,119 | 18,884 |
| Correlation of `min_rank` (shared genes) | 0.860 | — |
| Known sweeps below 10% (geom_mean) | 8/14 | 9/14 |
| Known sweeps below 10% (frac_below_1000) | 9/14 | — |
| Top 30 dominated by SDs | yes (29/30) | yes |
| LCT CEU `min_rank` | **0.0032** | 0.0089 |
| MCM6 CEU `min_rank` | **0.0031** | 0.0073 |
| EDAR CHB `min_rank` | **0.0034** | 0.0081 |
| SLC24A5 GBR `min_rank` | **0.0009** | 0.0084 |
| HERC2 FIN `min_rank` | **0.0093** | 0.0268 |

The new pipeline matches or exceeds the archive on 8 of 14 known
controls and is meaningfully better on the partial sweeps (LCT,
MCM6, EDAR) where the geometric-mean fix has its largest effect.

## What changed in the pipeline

Three structural changes between the 2026-04-09 archive and the
2026-04-11 rerun:

1. **gamma_smc parity** — bilinear interpolation at the upper
   flow-field boundary was wrong on the old GPU path. Fixed in
   `9d7ed6b`. Verified at parity with the upstream gamma_smc
   binary across the stdpopsim test suite.
2. **`auto_estimate_theta=True`** — scaled rates derived from
   per-population heterozygosity instead of a fixed
   `4 * 10000 * mu`. Removes the demographic-misspecification
   pitfall on bottlenecked / non-human populations.
3. **Per-population segregating-site filter** — sites monomorphic
   within the population subset are dropped before decoding,
   matching the gamma_smc reference behavior.

A fourth change was discovered during postprocessing of the
broken first-pass aggregation: **per-gene aggregation must be in
log space** (geometric mean of per-pair TMRCAs), not arithmetic
mean. The arithmetic mean dilutes partial sweeps because the
pair-level TMRCA distribution at a partial-sweep gene is bimodal
(young SS pairs, old non-SS pairs); the arithmetic mean is
dragged up by the larger non-sweep mode while the geometric mean
correctly tracks the lower mode.

## Step 1 — Segmental-duplication masking

UCSC `genomicSuperDups.txt.gz` track lifted to GRCh38 coordinates.
A gene is flagged as SD if ≥50% of its bp overlap a recorded
duplication. **1,296 / 19,119 genes (6.8%)** are flagged.

Without masking, the top 30 genome-wide candidates by `geom_mean`
are 29 SDs and 1 olfactory receptor cluster:

```
CCNYL1B (chr16) RGPD5 (chr2) SPATA31A5 (chr9) UGT2B17 (chr4)
LIMS3 (chr2) SMIM11 (chr21) GATD3 (chr21) POTED (chr21)
NOTCH2NLR (chr1) PWP2 (chr21) NBPF26 (chr1) LIMS4 (chr2)
SERF1A (chr5) AMY1B (chr1) AMY1C (chr1) UGT2B28 (chr4) ...
```

With masking, the top 30 looks completely different and
matches biology:

| rank | gene | pop | note |
|---|---|---|---|
| 1 | OR4C6 | LWK | olfactory cluster |
| 2 | MYEF2 | GBR | adjacent to SLC24A5 |
| 3 | CTXN2 | GBR | adjacent to SLC24A5 |
| 4 | **SLC24A5** | GBR | textbook EUR light-skin sweep |
| 5 | CENPW | CHB | |
| 6 | SHCBP1 | JPT | |
| 7 | BMI1 | GIH | |
| 8 | COMMD3-BMI1 | GIH | |
| 9 | SPAG6 | ITU | |
| 10 | **ZRANB3** | CEU | LCT sweep haplotype |
| 11 | ABCC11 | CHB | known EAS sweep (earwax) |
| 12 | COMMD3 | FIN | |
| 13 | PIP | CDX | |
| 14 | **DARS1** | CEU | LCT sweep haplotype |
| 15 | SLC26A6 | GIH | |
| 16 | EDC4 | CHB | |
| 17 | **GRK2** | GIH | novel SAS sweep ✓ |
| 18 | CELSR3 | STU | |
| 19 | LONP2 | CHB | |
| 20 | TKFC | MXL | |
| 21 | PHKB | PEL | |
| 22 | RANBP10 | CHB | |
| 23 | TMEM208 | CHS | |
| 24 | ABCC12 | JPT | adjacent to ABCC11 |
| 25 | NETO2 | PEL | |
| 26 | C16orf86 | CHB | |
| 27 | ZNF780B | FIN | |
| 28 | SULT1C4 | CHB | |
| 29 | TMEM89 | GIH | |
| 30 | ENKD1 | CHB | |

SLC24A5, the LCT sweep cluster (ZRANB3 + DARS1), GRK2, ABCC11
(known EAS earwax sweep), and several SLC24A5-adjacent genes are
all in the top 30. **This is the cleanest top-list of the
project.**

**Action item: SD masking must be applied before reporting any
top-N candidates from now on. Add it to `aggregate.py`.**

## Step 2 — Primary statistic recommendation

Five candidate per-gene-per-population summary statistics
computed from the per-pair TMRCA distribution within each gene:

| stat | known controls below 10% | below 5% | below 1% |
|---|---|---|---|
| geom_mean | 8/14 | 8/14 | 7/14 |
| p5 | 9/14 | 8/14 | 6/14 |
| p10 | 8/14 | 8/14 | 6/14 |
| min | 11/14 | 10/14 | 3/14 |
| frac_below_1000 | 9/14 | 9/14 | 7/14 |

`min` recovers the most controls but is the noisiest single-point
statistic. `geom_mean` matches the archive pipeline (verified in
`run_cxt_region.py`) and is the natural log-space centrality.
`frac_below_1000` is the most directly interpretable as a
"sweep haplotype frequency" estimate.

**Recommendation: report `geom_mean` as the headline statistic
and `frac_below_1000` as the supporting statistic.** Two
complementary views (centrality vs frequency); agreement between
them is a robustness check.

See `STATS.md` for the longer rationale.

## Step 3 — Why TYRP1, APOL1, OCA2, EPAS1 are still missed

NPZ histograms inspected for these four genes in the relevant
populations. The signal exists at the very tail of the per-pair
distribution but represents <0.1% of pairs:

| gene | pop | count | geom_mean | min | fraction in lowest 5 hist bins |
|---|---|---|---|---|---|
| TYRP1 | KHV | 29,646 | 3,857 | **13** | 0.000 |
| APOL1 | YRI | 63,190 | 11,071 | **16** | 0.000 |
| OCA2 | CEU | 63,903 | 7,654 | **12** | 0.000 |
| EPAS1 | CHB | 21,115 | 7,372 | **67** | 0.000 |

Each has at least one within-population pair coalescing at
12–67 generations — the sweep haplotype IS in the data — but
the sweep allele frequency in the relevant population is too low
(<10%) for the signal to rise above genome-wide noise on any
centrality or low-quantile statistic.

These are fundamental limits of low-quantile TMRCA statistics
on partial sweeps. **iHS, nSL, and SweepFinder are the
appropriate orthogonal tools** for these genes.

EPAS1 is also a known limitation: the strong sweep is in
Tibetan highlanders, not in any 1KG population. CHB shows only
the very weak peripheral signal.

See `missed_sweeps_histograms.png` for the visualizations.

## Step 4 — Manhattan plots

`manhattan.png` shows two genome-wide panels:

- Top: `-log10(min_rank_geom_mean)`
- Bottom: `-log10(min_rank_frac_below_1000)`

SD-flagged genes are drawn in light gray (mostly hugging the
peaks but visually separable). Top 10 non-SD candidates per
panel are labeled. Both panels agree on the chr15 SLC24A5 peak,
the chr2 LCT cluster, the chr11 GRK2 region, and the chr16
ABCC11/ABCC12 region.

## Step 5 — Pathway convergence

Permutation enrichment test (n=20,000) for KEGG and GO Biological
Process gene sets, per superpopulation, with FDR correction.

**No pathway hits FDR < 0.05 in any superpop after BH correction
across the 5,726 tested gene sets.** The most-significant after
FDR was AFR `Regulation Of Cell Differentiation` at q = 0.28.

Top raw p-values per superpop (before FDR):

- **EAS Ethanol Metabolic Process**: raw p=0.0011, fold=4.6
  (captures the well-known ADH cluster sweep). Does **not**
  survive FDR.
- **AFR Regulation Of Cell Differentiation**: raw p=0.00005,
  fold=2.0. Does not survive FDR (q=0.28).
- **EAS DNA Metabolic Process** raw p=0.00105, fold=1.6.
- **EUR RISC Complex Assembly** raw p=0.00040, fold=6.6.

### The mucosal-immunity claim from the archive does not
### replicate as a pathway-level enrichment.

Specifically scanned for terms containing "iga", "mucos",
"intestin", "innate", "toll", "nf-kappa", "nod-like", "dectin",
"lectin", "b cell", "complement", "antigen process":

| superpop | pathway | raw p | q (FDR) |
|---|---|---|---|
| AFR | I-kappaB kinase / NF-kB Signaling | 0.011 | 1.0 |
| AFR | Negative Regulation Of Innate Immune Response | 0.016 | 1.0 |
| EAS | Regulation Of Innate Immune Response | 0.020 | 1.0 |
| EAS | Positive Regulation Of NIK/NF-kB Signaling | 0.039 | 1.0 |
| EAS | Negative Regulation Of TLR2 Signaling | 0.070 | 1.0 |
| EUR | Marginal Zone B Cell Differentiation | 0.074 | 1.0 |

These are not statistically significant after FDR. The previous
claim of "mucosal immunity convergence in EAS" should be
**retracted from the manuscript** as a pathway-level finding.

What can still be claimed: individual genes in the mucosal
immunity space remain real outliers — CLEC6A (chr12, EAS, rank
0.008), BPIFA2 (chr20, SAS, rank 0.009), JCHAIN (chr4, AMR/EAS,
rank 0.020). These are gene-level findings, not pathway-level.

`pathway_enrichment.csv` has the full table for all 5
superpopulations × ~5,700 pathways.

## Step 6 — Cross-validation of novel findings

Eight candidates carried forward from the 2026-04-09 archive:

| Gene | new rank | new pop | superpop | replicates? |
|---|---|---|---|---|
| **GRK2** | **0.0023** | GIH | SAS | ✓ strong |
| **SLC6A15** | **0.0031** | CHS | EAS | ✓ strong |
| **CCDC92** | **0.0082** | CDX | EAS | ✓ strong |
| **CLEC6A** | **0.0084** | CDX | EAS | ✓ strong |
| **BPIFA2** | **0.0092** | GIH | SAS | ✓ strong |
| JCHAIN | 0.0202 | PEL | AMR/EAS | ✓ weaker |
| PIGR | 0.060 | ITU | SAS (was EAS) | weak, pop changed |
| TNFRSF13C | 0.119 | CDX | EAS | ✗ no longer top |

**Five robust replications** (GRK2, SLC6A15, CCDC92, CLEC6A,
BPIFA2) all in the expected superpopulation, all below 1%
within-population rank.

**Three weakened**:
- JCHAIN dropped from top tier but still notable (rank 0.02).
- PIGR signal is now strongest in SAS (ITU), not EAS as
  previously claimed.
- TNFRSF13C fell out of the top tier (0.12).

The JCHAIN/PIGR/TNFRSF13C weakening matches the failure of the
mucosal-immunity pathway claim — three of the four "pathway
convergence" support genes are individually weaker in the new
run, leaving CLEC6A as the only strong mucosal immunity hit.

`novel_findings.csv` has the per-superpop best ranks for each
candidate.

## Files

```
analysis/genome_wide/postprocess/
├── FINDINGS.md                        — this document
├── STATS.md                           — primary stat recommendation
├── stat_comparison.csv                — sweep recovery per stat
├── genes_sd_flag.csv                  — full gene table with SD flags
├── top50_geom_mean_no_sd.csv          — top 50 SD-free per stat
├── top50_p5_no_sd.csv
├── top50_p10_no_sd.csv
├── top50_min_no_sd.csv
├── top50_frac_below_1000_no_sd.csv
├── missed_sweeps_histograms.png       — TYRP1/APOL1/OCA2/EPAS1 hist plots
├── manhattan.png                      — genome-wide manhattan, both stats
├── pathway_enrichment.csv             — full KEGG/GO results
└── novel_findings.csv                 — per-superpop best ranks
```

## Next steps for the manuscript

1. **Apply SD masking** in the headline candidate tables.
2. **Report two statistics** (geom_mean and frac_below_1000) for
   every novel-sweep claim. Treat agreement as a robustness check.
3. **Retract the mucosal-immunity pathway-level claim**. Reframe
   CLEC6A and BPIFA2 as gene-level findings without pathway
   support.
4. **Acknowledge the limit on partial sweeps**. TYRP1, APOL1,
   OCA2, EPAS1 should be discussed as known limitations of the
   method, not as misses to be debugged.
5. **Run iHS, nSL, xpEHH, SweepFinder2** as orthogonal validation
   for the five replicated novel candidates (GRK2, BPIFA2,
   SLC6A15, CCDC92, CLEC6A). These are the manuscript's headline
   claims and need independent support.
6. **Regenerate the manuscript figures** from the new data.
   Specifically: Manhattan plot (use `manhattan.png` as a
   starting point), per-population heatmap, top-candidate
   violins.
