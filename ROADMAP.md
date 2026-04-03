# Roadmap: from proof of concept to publication

## Current state

- tmrca.cu: working GPU tool, validated accuracy, multi-GPU support
- 1000 Genomes chr21: 830k pairs, 214 genes, selection scan framework
- Manuscript v2: compelling narrative but results are chr21-only

## What's missing (in priority order)

### 1. Genome-wide analysis (all 22 autosomes)

**Why:** Chr21 has 214 protein-coding genes — the smallest autosome.
Genome-wide gives ~20,000 genes. Selection scans have statistical power
proportional to the number of features tested. With 214 genes, a
5th-percentile hit is 1 out of 10 — barely significant. With 20,000
genes, it's 1 out of 1,000 — compelling.

**What to do:**
- Download all 22 autosome VCFs (~15 GB total compressed)
- Parse each with scikit-allel, cache as .npz (~30 min one-time)
- Run inference per chromosome, per population (estimate: 13 min × 22 ≈ 5 hours on 3 GPUs)
- Aggregate into a single genome-wide AnnData: 830k pairs × ~20k genes
- The PCA/UMAP on 20k features will separate populations unsupervised
- The selection scan on 20k genes gives genome-wide significance

**Computational cost:** ~5 hours GPU, ~100 GB disk for the h5ad.
Feasible in one overnight run.

**Expected outcome:** Clean unsupervised population separation (no
supervision needed), genome-wide Manhattan plot of sweep candidates,
comparison with known selection signals.

### 2. Cross-validation against known selection signals

**Why:** Without this, our sweep candidates are just "genes with unusual
TMRCA." A reviewer will ask if these are known selection targets or
false positives. If they overlap known hits, the method is validated.
If they're novel, that's a discovery. Either way, we need the comparison.

**What to do:**
- Compile a list of known positive selection targets from the literature:
  - iHS hits from Voight et al. 2006 (and updates)
  - PBS hits from Yi et al. 2010 (high-altitude adaptation)
  - Grossman et al. 2013 composite of multiple signals (CMS)
  - 1000 Genomes Phase 3 selection scan results
  - Specific known sweeps: LCT (lactase), SLC24A5 (skin pigmentation),
    EDAR (hair/teeth), EPAS1 (altitude), HBB (malaria)
- For each known sweep gene, check its within-population rank in our
  genome-wide scan. Does it show up as an outlier in the expected
  population?
- Compute enrichment: are our top-ranked genes enriched for known
  selection targets versus random genes?
- Report sensitivity and specificity at various rank thresholds

**Expected outcome:** If LCT shows up at the 1st percentile in EUR
and SLC24A5 shows up at the 2nd percentile in EUR, the method works.
If they don't, we have a problem.

### 3. Between-population pairs

**Why:** Within-population pairs capture diversity but not divergence.
Between-population pairs (e.g., CEU × YRI) estimate divergence times
directly. This enables:
- TMRCA-based F_ST analogs (per-gene divergence)
- Introgression detection (genes where between-pop TMRCA is unexpectedly
  recent → recent gene flow)
- Admixture mapping (which genomic regions show recent shared ancestry
  between specific populations?)

**What to do:**
- For selected population pairs (e.g., CEU×YRI, CEU×CHB, YRI×CHB),
  compute all between-population haplotype pairs
- CEU (358 haps) × YRI (356 haps) = 127,448 between-pop pairs
- Run inference same as within-pop
- Compare between-pop TMRCA distributions to within-pop
- Genome scan: genes where between-pop TMRCA is unusually recent
  (introgression candidates) or deep (ancient divergence)

**Computational cost:** Similar to within-pop per pair, but fewer pairs
per comparison (~100k). A handful of key population pairs would suffice.

**Expected outcome:** TMRCA-based divergence times that correlate with
known population split times. Possible introgression signals in
admixed populations (AMR populations showing recent CEU ancestry at
specific genes).

### 4. Something only pairwise TMRCA can do

**Why:** Right now everything in the paper could arguably be done with
windowed F_ST, PBS, or iHS. We need a result that is uniquely enabled
by having the full pairwise TMRCA matrix.

**Candidates:**
- **Pairwise TMRCA-based kinship:** The gene-level TMRCA between two
  individuals is a direct estimate of their genealogical relatedness
  at that locus. Averaging across genes gives a TMRCA-based kinship
  matrix. Does this outperform IBD-based or genotype-based kinship
  for GWAS? Probably not for common variants, but for rare variant
  association it might — because TMRCA captures shared history even
  without shared variants.

- **Per-gene coalescent rate estimation:** For each gene and population,
  the distribution of pairwise TMRCAs estimates the local coalescent
  rate (= 1/2N_e). Plotting this across the genome gives a local N_e
  landscape — which genes have reduced diversity (background selection)
  and which have elevated diversity (balancing selection)? This is
  similar to what PSMC does for one pair but now averaged over thousands
  of pairs, giving much higher resolution.

- **TMRCA heterogeneity within populations:** For a given gene, some
  pairs within a population may have very recent coalescence while
  others have deep coalescence. This bimodality suggests the presence
  of distinct haplotype groups — possibly maintained by balancing
  selection or reflecting substructure. The full pairwise matrix
  captures this; summary statistics like F_ST average it away.

- **Temporal layering:** Different genes coalesce at different times.
  By sorting genes by their population-mean TMRCA, you get a temporal
  profile of evolutionary history — recent genes (recent sweeps or
  bottleneck effects), intermediate genes (drift), ancient genes
  (balanced polymorphism or introgression from archaic lineages).

**Recommendation:** The per-gene coalescent rate landscape is the most
compelling unique contribution. It's a natural output of the pairwise
TMRCA matrix, it can't be computed from genotype data alone (you need
actual time estimates), and it directly connects to population genetics
theory (N_e landscape = selection + recombination map).

### 5. Biological interpretation of sweep candidates

**Why:** Finding genes with unusual TMRCA is statistics. Explaining why
they're unusual is biology. A PLOS Comp Bio paper needs both.

**What to do for each top candidate:**
- Gene function and expression pattern (from UniProt, GTEx)
- Known disease associations (OMIM, GWAS Catalog)
- Known selection signals at this locus from other studies
- Local recombination rate (is the signal driven by a recombination
  coldspot rather than selection?)
- Haplotype structure at the locus (EHH, bifurcation diagrams)
- Overlap with known regulatory elements (ENCODE)

**For the paper:** A table of top 10 genome-wide candidates with
gene function, population, rank, and supporting evidence from other
methods. Plus 2-3 detailed case studies with haplotype-level analysis.

## Execution timeline

| Phase | What | Time | Compute |
|-------|------|------|---------|
| 1 | Download + parse all 22 VCFs | 1 day | CPU only |
| 2 | Genome-wide inference | 1 day | 3× A100, overnight |
| 3 | Genome-wide AnnData + PCA/UMAP | 2 hours | CPU |
| 4 | Selection scan + cross-validation | 1 day | CPU |
| 5 | Between-population pairs (3 comparisons) | 4 hours | 3× A100 |
| 6 | Biological interpretation | 2 days | literature |
| 7 | Manuscript v3 | 2 days | writing |
| **Total** | | **~1 week** | |

## What changes in the paper

- Title stays (it's good)
- Results expand from 1 chromosome to genome-wide
- Selection scan gains statistical power and validation
- New results section: between-population divergence
- New results section: per-gene coalescent rate landscape
- Discussion gains "what TMRCA can do that genotypes can't"
- Supplementary: full table of genome-wide sweep candidates
