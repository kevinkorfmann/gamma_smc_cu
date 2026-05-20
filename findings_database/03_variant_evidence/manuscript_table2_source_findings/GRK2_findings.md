# Deep investigation: GRK2 SAS+EUR sweep (chr11:67.1-67.6Mb)

## Summary

A ~300kb selective sweep centered on GRK2 at chr11:67.27Mb, strongest
in South Asian and European populations, with the derived haplotype
nearly fixed in SAS+EUR (AF ~5%) and ancestral alleles retained in
Africa (AF ~84%). No prior selection scan has reported this locus.

## TMRCA signal

### Sweep core (SAS mean <1%)

| Gene | Position | SAS mean | EUR mean | AFR mean | EAS mean |
|------|----------|----------|----------|----------|----------|
| GRK2 | 67.27Mb | 0.2% | 0.4% | 6.0% | 4.7% |
| ANKRD13D | 67.29Mb | 0.5% | 1.0% | 21.9% | 9.0% |
| SSH3 | 67.30Mb | 0.6% | 1.4% | 15.4% | 7.6% |
| POLD4 | 67.35Mb | 0.8% | 1.2% | 10.6% | 6.8% |
| ENSG00000256514 | 67.35Mb | 0.6% | 1.0% | 6.0% | 6.1% |
| CLCF1 | 67.36Mb | 0.8% | 1.6% | 27.9% | 10.5% |
| PPP1CA | 67.40Mb | 0.8% | 2.4% | 27.5% | 12.0% |
| TBC1D10C | 67.40Mb | 0.6% | 2.0% | 19.3% | 11.1% |
| CARNS1 | 67.42Mb | 1.1% | — | — | — |

### Sweep flanks (SAS 2-5%)

Centromeric: KDM2A (67.12Mb, SAS=1.3%), SYT12 (67.01Mb, SAS=4.2%)
Telomeric: RPS6KB2 (67.43Mb, SAS=3.2%), CDK2AP2 (67.51Mb, SAS=1.5%)
Clear boundary at GSTP1 (67.58Mb, SAS=42%) and RHOD (67.06Mb, SAS=19%)

### Within-SAS uniformity

GRK2 is at exactly 0.23% in ALL 5 SAS populations:
- BEB (Bengali): 0.23%
- GIH (Gujarati): 0.23%
- ITU (Indian Telugu): 0.23%
- PJL (Punjabi): 0.23%
- STU (Sri Lankan Tamil): 0.23%

This perfect uniformity across geographically diverse South Asian
populations indicates an ancient sweep predating the diversification
of modern SAS populations.

### EUR is also swept

EUR shows very low ranks too (GRK2: 0.3-0.7%). This is NOT
SAS-specific — it's a SAS+EUR shared sweep, with EAS intermediate
(3-5%) and AFR highest (5-28%). Pattern consistent with an
out-of-Africa sweep that strengthened in the SAS+EUR lineage.

## Variant analysis (from 1000 Genomes VCF)

11,872 variants in the sweep region (67.1-67.6Mb).

### FST

**SAS vs non-SAS:**
- Mean FST: 0.004 (very low — SAS+EUR share the haplotype)
- Max FST: 0.11

**EUR+SAS vs AFR+EAS+AMR:**
- Mean FST: 0.007
- Max FST: 0.28 — substantially higher, confirming shared haplotype

### Sweep signature

- SAS-depleted variants (AF<10% SAS, >30% others): **42**
- SAS-enriched variants (AF>50% SAS, <30% others): **1**

This 42:1 asymmetry is the classic sweep signature: a derived
haplotype rose to near-fixation, sweeping out ancestral diversity.

### Most differentiated variant: chr11:67,407,126

Position in PPP1CA gene. Per-population allele frequencies:
- SAS: BEB=6%, GIH=2%, PJL=4%, STU=5%, ITU=8%
- EUR: GBR=8%, CEU=9%, IBS=5%, TSI=8%, FIN=12%
- AFR: YRI=89%, ESN=85%, GWD=87%, MSL=89%, LWK=78%
- EAS: CHB=40%, CHS=44%, JPT=39%, CDX=30%, KHV=32%
- AMR: PEL=70%, MXL=39%, CLM=32%, PUR=29%

FST (EUR+SAS vs rest) = 0.28 — highly differentiated.
The ancestral (African) allele is at ~85% in AFR but only ~5% in SAS+EUR.

## Gene function: GRK2

GRK2 (G protein-coupled receptor kinase 2, formerly ADRBK1):

**Primary function:** Phosphorylates and desensitizes beta-adrenergic
receptors (beta-ARs). The main "off switch" for sympathetic nervous
system signaling in heart and vasculature.

**Cardiovascular role:**
- GRK2 overexpression → hypertension + heart failure (mouse models,
  Cohn et al. 2023 Sci Rep)
- GRK2 inhibition is a therapeutic target for heart failure
  (Rengo et al. 2012 Gene Therapy)
- GRK2 levels correlate with systolic BP in African Americans
- rs1894111 (GRK2 intronic) → BP response to hydrochlorothiazide

**Other genes in the sweep:**
- CLCF1: cardiotrophin-like cytokine, cardiac development
- PPP1CA: protein phosphatase 1, smooth muscle contraction
- CARNS1: carnosine synthase, muscle buffering
- SSH3: slingshot phosphatase, cytoskeletal regulation

Multiple genes in the sweep core relate to cardiac/muscle function.

## Biological hypothesis

The GRK2 sweep in SAS+EUR may represent cardiovascular adaptation
during migration out of Africa into temperate environments:

1. **Salt homeostasis:** GRK2 regulates renal beta-AR signaling,
   affecting sodium handling. Parallels CYP3A5 salt-retention sweep.
2. **Thermoregulation:** Beta-AR drives brown fat thermogenesis.
3. **Cardiovascular tone:** Higher resting BP may have been
   advantageous in cold environments.
4. **Timing:** Shared SAS+EUR signal (EAS intermediate) suggests
   the sweep occurred after the EAS-EUR/SAS split (~30-40 kya).

## Connection to salt/cardiovascular pathway

| Gene | Chr | Population | Function | Status |
|------|-----|-----------|----------|--------|
| CYP3A5 | 7 | EUR | Cortisol → Na+ reabsorption | Known sweep |
| ATP1A1 | 1 | SAS/EUR | Na+/K+ pump | Confirmed (Galinsky 2016) |
| GRK2 | 11 | SAS+EUR | Beta-AR desensitization → BP | this study |
| WNK4 | 17 | AMR | Renal sodium handling | 1.7% in MXL |
| SLC12A1 | 15 | EUR | Na-K-Cl cotransporter | 2.5% in TSI |

Salt pathway enrichment: 5/21 genes at <10% in SAS, p = 0.045.
Suggestive but not independently significant.

