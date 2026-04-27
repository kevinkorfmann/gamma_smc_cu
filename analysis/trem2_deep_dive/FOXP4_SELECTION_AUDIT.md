# FOXP4 — is it widely known as a selection target?

## TL;DR
**No.** FOXP4 is a household name in COVID-19 and cancer GWAS, but is **not on any canonical human selection-scan hit list**. One 2024 bioinformatics paper (Wang et al., *Funct Integr Genomics*) claims "distinct genetic diversity patterns and positive selection signatures" at FOXP4 across 1000G populations, but it is not resolved to specific populations and does not use classical haplotype scans. This means our IBS-specific TMRCA signal at 6p21.1 is **genuinely novel as a selection finding**, while resting on a deep, well-established disease/GWAS biology base.

## Canonical selection scans checked (FOXP4 absent)

| Scan | Method | Ref | FOXP4 hit? |
|---|---|---|---|
| Voight et al. 2006 *PLoS Biol* | iHS, 1M SNPs | Voight 2006 | No |
| Sabeti et al. 2007 *Nature* | LRH + XP-EHH, HapMap2 | Sabeti 2007 | No |
| Pickrell et al. 2009 *Genome Res* | CLR / iHS, HGDP | Pickrell 2009 | No |
| Grossman et al. 2013 *Cell* | CMS composite | Grossman 2013 | No |
| Field et al. 2016 *Science* | SDS, UK10K | Field 2016 | No |
| Racimo 2016, Browning 2018, Skov 2020 | Archaic introgression | Various | No |
| Akbari et al. 2026 *Nature* | aDNA time-series, POSTERIOR≥0.99 | Akbari 2026 | **No (subthreshold 0.75 at FOXP4 intron rs11760063)** |

## Selection databases

- **PopHumanScan** (Murga-Moreno 2019, *NAR*): catalog of adaptation signals across 22 non-admixed 1000G populations using 8 statistics. Server times out on direct query, but the curated literature around it does not cite FOXP4 as a strong adaptation region at 6p21.1. *Conclusion: not prominently flagged.*
- **dbPSHP** (Li 2014, *NAR*): 15,472 manually curated selection loci from 132 publications. Literature searches for "FOXP4" + "dbPSHP" and "FOXP4" + "selection" return the one Wang 2024 paper only — nothing from the dbPSHP curation corpus.
- **1000 Genomes Selection Browser 1.0** (Pybus 2014): FOXP4 region not cited in their flagged regions.

## The one existing selection claim

**Wang et al. 2024, *Funct Integr Genomics*, "Deciphering the role of FOXP4 in long COVID…"** (DOI: 10.1007/s10142-024-01451-7).

- Analyzed 1000G Phase 3 across 26 populations.
- Reported "distinct genetic diversity patterns and positive selection signatures" for FOXP4.
- Identified the haplotype **CA** (rs1886814 + rs2894439) linked to long-COVID severity.
- **Does not report population-resolved iHS/XP-EHH signals, does not highlight IBS specifically, and is primarily a drug-repurposing bioinformatics paper.**

So the "FOXP4 is under selection" claim in the literature amounts to **one paper making a diffuse claim**, not a population-resolved, haplotype-based selection scan.

## Context: the FOXP2 precedent

FOXP2 (sister gene) has a famous selection-signature history (Enard 2002 *Nature*; later partly retracted by Atkinson 2018 PMC6128738 showing no recent selection signal after controlling for demography). FOXP4 does **not** have a primate-lineage selection story — its human-chimp divergence rate is unremarkable.

## Implication

The IBS-specific TMRCA collapse at 6p21.1 (r_IBS/non-AFR ≈ 1.41, CI [1.35, 1.49]; sweep-allele frequency 82% in IBS vs 96% in JPT; focal variant at chr6:41,470,132) is:

- **Population-novel**: no prior scan has resolved a selection signal to IBS at this locus.
- **Gene-novel for selection**: FOXP4 has not been flagged by the canonical scans (iHS/XP-EHH/CMS/SDS/aDNA).
- **Biologically-anchored**: strong COVID-severity GWAS, lung epithelial regulation, cis-eQTLs in lung and brain.
- **Compatible with a very recent sweep**: consistent with the Akbari subthreshold signal (too recent / too IBS-specific for their pooled ancient-European panel).

This is a stronger, not weaker, story than the original "TREM2 selection" framing. We gain:
- Well-established disease mechanism (severe COVID / long COVID / lung regeneration)
- Plausible geographic driver (Iberian respiratory-disease history — plague, influenza, tuberculosis)
- Genuine novelty as a selection target

We lose:
- Clean mapping to one coding-variant story like TREM2 R47H. The sweep appears regulatory (peak between genes, likely eQTL).
