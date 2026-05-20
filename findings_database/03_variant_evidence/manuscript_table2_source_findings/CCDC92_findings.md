# Deep investigation: CCDC92/ZNF664 — EAS metabolic sweep (chr12:123.5-124.1Mb)

## Summary

An EAS-specific regional sweep of ~500kb at chr12q24.31, strongest
in CDX and JPT populations. Contains CCDC92 and ZNF664, both GWAS
hits for metabolic traits.

## TMRCA signal

| Gene | EAS mean | EUR mean | AFR mean | Min | Pop |
|------|----------|----------|----------|-----|-----|
| ZNF664 | 0.9% | 4.2% | 10.4% | 0.3% | CDX |
| GTF2H3 | 3.6% | 5.3% | 6.3% | 0.3% | JPT |
| CCDC92 | 1.7% | 3.8% | 5.7% | 0.4% | CDX |
| EIF2B1 | 4.5% | 2.9% | 8.5% | 0.5% | JPT |
| DDX55 | 4.7% | 3.3% | 9.3% | 0.6% | JPT |

## Variant analysis

Region (chr12:123.5-124.1Mb): 15,670 variants analyzed.

### FST (EAS vs non-EAS)

- Mean FST: 0.005
- Max FST: 0.198
- EAS-enriched (AF>50% EAS, <30% others): **18**
- EAS-depleted (AF<10% EAS, >30% others): **265**

The 265:18 depletion/enrichment ratio is very strong — the second
strongest sweep signature after BPIFA2 (127:2 total, but higher
ratio). The 265 depleted variants indicate a very clean sweep.

### Most differentiated variant: chr12:123,959,862

Within/near ZNF664:
- EAS: CDX=93%, CHB=89%, CHS=94%, JPT=88%, KHV=93% (~92%)
- SAS: BEB=60%, GIH=47%, ITU=46%, PJL=47%, STU=48% (~50%)
- EUR: CEU=48%, FIN=57%, GBR=49%, IBS=46%, TSI=32% (~46%)
- AFR: all <7% (~5%)

Pattern: EAS ~92% → SAS ~50% → EUR ~46% → AFR ~5%.
A derived allele nearly fixed in EAS, at intermediate frequency in
EUR/SAS, and rare in AFR.

## Gene functions

**CCDC92** (Coiled-Coil Domain Containing 92):
- GWAS hit for waist-hip ratio and BMI-adjusted adiponectin
- Associated with increased triglycerides and decreased HDL
- CCDC92 knockout reduces obesity and insulin resistance in mice
  (Li et al. 2023, iScience)
- Noncoding SNPs reduce CCDC92 expression in subcutaneous adipose

**ZNF664** (Zinc Finger Protein 664):
- GWAS hit for triglyceride levels
- Associated with HDL-C metabolism

**GTF2H3**: General transcription factor IIH subunit 3 (DNA repair)

## Biological hypothesis

The EAS sweep at CCDC92/ZNF664 likely represents dietary/metabolic
adaptation. Possible drivers:
1. **Rice agriculture:** High-carbohydrate diet → selection on lipid
   metabolism and insulin sensitivity
2. **Thrifty genotype:** Metabolic efficiency during agricultural
   transition with seasonal food scarcity
3. **Body composition:** CCDC92 affects fat distribution (WHR GWAS)

The CCDC92 mouse knockout showing reduced obesity is compelling —
the swept haplotype may increase CCDC92 function, optimizing
lipid metabolism for an East Asian dietary context.

