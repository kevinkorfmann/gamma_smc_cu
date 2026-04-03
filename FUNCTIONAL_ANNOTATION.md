# Functional annotation of TMRCA sweep candidates

## Motivation

Finding genes with unusual population-specific TMRCA is statistics.
Connecting those genes to functional impact — disease risk, protein
function, clinical relevance — turns it into biology. This plan
describes how to annotate sweep candidates with functional databases.

## Data sources

### 1. AlphaMissense (pathogenicity prediction)

Google DeepMind's AlphaMissense provides per-variant pathogenicity
scores for all possible single amino acid substitutions in the human
proteome.

- **Data**: `AlphaMissense_hg38.tsv.gz` (~7 GB)
  - URL: https://zenodo.org/records/8208688
- **Use**: For each sweep candidate gene, count the number of
  AlphaMissense "likely pathogenic" variants (score > 0.564) that
  are segregating in the 1000 Genomes data, stratified by population.
  A gene under recent selection that harbors pathogenic variants
  suggests adaptive evolution at a functionally constrained locus.
- **Analysis**: Correlate within-population TMRCA rank percentile
  with the number of segregating pathogenic variants. Do genes with
  recent coalescence have more or fewer pathogenic variants? (More =
  possible adaptive introgression of functional variants. Fewer =
  possible purifying selection sweep.)

### 2. Polygenic risk scores (PGS Catalog)

The PGS Catalog (https://www.pgscatalog.org) provides curated
polygenic risk score weights for thousands of traits.

- **Data**: PGS scoring files (per-variant weights)
- **Use**: For each sweep candidate gene:
  - Which PRS traits have variants in this gene?
  - Are those variants population-stratified in frequency?
  - Does the TMRCA signal correlate with PRS transferability
    problems? (Genes where TMRCA diverges between populations are
    genes where PRS weights may not transfer.)
- **Analysis**: Population-specific TMRCA at PRS-relevant genes
  could explain why PRS trained in EUR populations perform poorly
  in AFR populations — if the underlying haplotype structure
  (captured by TMRCA) is different, the tag SNP associations break.

### 3. ClinVar / OMIM (disease associations)

- **Data**: ClinVar VCF + OMIM gene-disease mappings
- **Use**: Are sweep candidate genes enriched for disease-associated
  genes? Which specific conditions? A gene under selection for a
  disease-resistance allele would show up as both a TMRCA outlier
  and a ClinVar hit.
- **Example**: If CFAP298 (our top KHV sweep candidate) has ClinVar
  entries for ciliopathy, that connects the selection signal to
  respiratory/reproductive function — plausible adaptive phenotype.

### 4. GWAS Catalog

- **Data**: NHGRI-EBI GWAS Catalog (https://www.ebi.ac.uk/gwas/)
- **Use**: Overlap sweep candidates with GWAS hits. A gene that is
  both a TMRCA outlier and a GWAS hit for a population-stratified
  trait is strong evidence for adaptive evolution at a medically
  relevant locus.
- **Analysis**: For the top 100 genome-wide sweep candidates, count
  GWAS associations. Compare to background rate (random genes).

### 5. Gene Ontology / pathway enrichment

- **Data**: GO annotations, KEGG pathways
- **Use**: Are sweep candidates enriched for specific pathways?
  Common themes in selection scans: immune response, metabolism,
  skin/hair pigmentation, lactase persistence, altitude adaptation.
- **Tools**: g:Profiler or DAVID for enrichment analysis.

### 6. Known selection databases

- **Data**:
  - dbPSHP (database of Positive Selection in Human Populations)
  - SelScan outputs from 1000 Genomes (iHS, nSL, XP-EHH)
  - Composite of Multiple Signals (CMS) from Grossman et al. 2013
- **Use**: Direct overlap. What fraction of our top candidates are
  known selection targets? What fraction of known targets do we
  recover? This gives sensitivity/specificity for our method
  compared to established approaches.

## Analysis plan

### Step 1: Genome-wide TMRCA (prerequisite)

Run tmrca.cu on all 22 autosomes. Without this, we only have 214
genes and limited statistical power for enrichment analyses.

### Step 2: Build annotation table

For each of ~20,000 protein-coding genes:
- Within-population TMRCA rank percentile (26 populations)
- Max rank range across populations (selection score)
- Population with lowest rank (sweep population)
- Number of AlphaMissense pathogenic variants (segregating in 1kGP)
- Number of ClinVar pathogenic/likely pathogenic variants
- Number of GWAS Catalog associations
- Number of PGS Catalog scores involving this gene
- Known positive selection (yes/no, from dbPSHP/CMS)
- GO terms, KEGG pathways

### Step 3: Correlation analyses

1. **TMRCA rank vs AlphaMissense burden**: Do genes under recent
   selection carry more or fewer pathogenic variants? Scatter plot +
   correlation per population.

2. **TMRCA rank vs GWAS hit density**: Are sweep candidates enriched
   for GWAS hits? Fisher's exact test.

3. **TMRCA rank vs PRS transferability**: For PRS traits with known
   EUR→AFR performance drop, do the genes driving the drop show
   population-specific TMRCA signals?

4. **Pathway enrichment of sweep candidates**: GO/KEGG enrichment
   of top 100 genes by rank range. Compare to known selection
   enrichments (immune, metabolism, pigmentation).

5. **Recovery of known sweeps**: Sensitivity analysis — what
   fraction of known positive selection targets fall in our top
   1%/5%/10% by rank range?

### Step 4: Case studies (2-3 genes in depth)

For the most compelling candidates (genome-wide), write a detailed
analysis:
- Gene function and expression
- Population-specific TMRCA distribution (violin plot)
- AlphaMissense landscape across the gene
- Segregating pathogenic variants and their population frequencies
- Comparison with iHS/PBS at this locus
- Haplotype structure (EHH decay plot)

## What this adds to the paper

- **Table**: Top 20 genome-wide sweep candidates with functional
  annotations (gene, population, TMRCA rank, AlphaMissense count,
  GWAS hits, ClinVar, known selection status)
- **Figure**: Scatter of TMRCA rank vs AlphaMissense burden,
  colored by population
- **Figure**: Enrichment barplot — GO terms of sweep candidates
- **Figure**: 2-3 detailed case studies
- **Result**: "X% of known selection targets are recovered by
  TMRCA ranking" — validates the method
- **Result**: "Sweep candidates are enriched for immune/metabolic
  pathways" — connects to biology
- **Result**: "TMRCA divergence at PRS genes partially explains
  cross-population PRS transferability" — clinical relevance

## Feasibility

Most of this is database lookups and statistical tests — no heavy
computation beyond the genome-wide TMRCA inference (step 1).
The annotation table can be built in a few hours with pandas.
The main bottleneck is the genome-wide inference (~5 hours GPU).
Everything else is analysis and writing.
