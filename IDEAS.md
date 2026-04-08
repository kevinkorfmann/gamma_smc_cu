# Idea: Population-specific selection × protein constraint

## Core question

Are genes under recent population-specific selection enriched for
functionally important (constrained) variants? Or are they depleted?

## Why this is interesting

The naive expectation: constrained genes (many possible pathogenic
variants per AlphaMissense) should be under purifying selection →
deep coalescence → high TMRCA rank in all populations equally.

The interesting case: a constrained gene with RECENT coalescence in
one population but not others. This means a functionally important
variant swept through that population — possibly adaptive despite
being at a constrained locus. These are the variants that cause
PRS transferability problems across populations.

## Data needed

1. **Per-gene TMRCA rank per population** — from tmrca.cu on 1kGP
   (already have chr21, need genome-wide)
2. **AlphaMissense scores** — per-variant pathogenicity for all
   possible missense substitutions (Zenodo, 7 GB)
3. **1000 Genomes variant frequencies** — which AlphaMissense variants
   actually segregate, at what frequency, in which populations
4. **Gene constraint scores** — pLI, LOEUF from gnomAD
5. **Established selection statistics** — iHS, PBS, XP-EHH from
   published 1kGP selection scans

## The analysis

### Level 1: Gene-level correlation (simple, fast)

For each of ~20,000 protein-coding genes:
- x = AlphaMissense constraint (mean pathogenicity of possible missense,
  or number of likely-pathogenic possible variants)
- y = TMRCA rank range across populations (our selection score)

Scatter plot. Is there a correlation? If constrained genes have LOW
rank range (neutral across populations), that's purifying selection
working as expected. If some constrained genes have HIGH rank range,
those are selection-at-constrained-loci candidates.

### Level 2: Population-stratified (the money analysis)

For each gene × population:
- TMRCA rank percentile (within-population)
- Number of segregating AlphaMissense "likely pathogenic" variants
- Population-specific allele frequencies of those variants

Question: In the population where a gene has unusually recent TMRCA,
do the pathogenic variants have unusual allele frequencies?
(Higher than expected = adaptive sweep of a "pathogenic" variant.
 Lower than expected = selective sweep purged the pathogenic variants.)

### Level 3: PRS connection

For traits with known EUR-AFR PRS transferability gap:
- Identify the genes contributing most to PRS in EUR
- Check their TMRCA rank in AFR
- If TMRCA diverges at PRS genes → the haplotype structure differs →
  tag SNP associations break → PRS doesn't transfer

This directly connects population TMRCA to a clinical problem.

## What makes this a paper

- **Novel combination**: nobody has correlated pairwise TMRCA with
  protein-level pathogenicity predictions across populations
- **Clinically relevant**: directly addresses PRS transferability
- **Actionable**: identifies specific genes where population-specific
  selection has altered the frequency of functionally important variants
- **Clean framing**: "does evolution respect protein constraint
  boundaries, and does this vary across populations?"
- **tmrca.cu is a tool, not the paper**: it enables the analysis but
  the contribution is the biological finding

## Possible venues

- Nature Genetics (if genome-wide results are strong and PRS connection holds)
- PLOS Genetics (more methods-tolerant)
- Genome Research (computational genomics audience)
- eLife (if the framing is broad enough)

## Relationship to the current project

This could be:
1. A section in the PLOS Comp Bio paper (adds biological depth)
2. A standalone short paper (Letters/Brief Communications format)
3. The nucleus of a separate, larger study

Option 1 is easiest and makes the PLOS Comp Bio paper much stronger.
Option 3 is the most impactful but requires more work and a broader
collaboration (someone with PRS expertise).
