# Deep investigation: BPIFA2 — SAS salivary antimicrobial sweep

## Summary

BPIFA2 (parotid secretory protein) shows a strong SAS-specific signal
(0.4% ITU, SAS mean 0.8%) with 79% range. It is the ONLY member of
the 9-gene BPIF antimicrobial family that is swept in SAS. The signal
extends the mucosal immunity narrative from intestinal (EAS) to oral
(SAS) defense.

## TMRCA signal

| Population | BPIFA2 rank |
|-----------|-------------|
| ITU (Indian Telugu) | 0.4% |
| GIH (Gujarati) | 0.6% |
| STU (Sri Lankan Tamil) | 0.9% |
| BEB (Bengali) | 1.1% |
| PJL (Punjabi) | 1.1% |
| **SAS mean** | **0.8%** |
| MXL (Mexican) | 0.7% |
| FIN (Finnish) | 2.4% |
| EUR mean | 3.0% |
| EAS mean | 5.6% |
| AMR mean | 2.9% |
| AFR mean | 63.0% |

Strong SAS signal with moderate EUR/AMR signal. AFR is very high (63%)
indicating the ancestral state has deep coalescence in Africa.

## BPIF gene family analysis

| Gene | SAS | EAS | EUR | AFR | Role |
|------|-----|-----|-----|-----|------|
| **BPIFA2** | **0.8%** | 5.6% | 3.0% | 63.0% | **Salivary antimicrobial** |
| BPIFA3 | 10.1% | 22.7% | 14.9% | 77.8% | Oral/nasal |
| BPIFB1 | 15.0% | 30.7% | 39.0% | 74.1% | Nasal/lung |
| BPIFB2 | 22.0% | 8.3% | 34.6% | 59.8% | Upper airway |
| BPIFB6 | 39.4% | 28.0% | 54.3% | 78.0% | Upper airway |
| BPIFA1 | 50.6% | 61.1% | 68.3% | 89.3% | Nasal/airway (SPLUNC1) |
| BPIFB3 | 51.0% | 26.9% | 72.5% | 56.0% | Upper airway |
| BPIFB4 | 63.2% | 32.6% | 85.1% | 80.9% | Lung |
| BPIFC | 88.5% | 87.4% | 90.6% | 81.7% | Ubiquitous |

Only BPIFA2 is swept. The family shows a gradient from swept (BPIFA2)
to normal (BPIFC), with the most SAS-specific signal at the gene
with the most salivary-specific expression.

## Variant analysis (from 1000 Genomes)

BPIF cluster region (chr20:31.7-33.7Mb): 47,836 variants.

### FST

- BPIFA2 gene body: 599 variants, mean FST = 0.016, max FST = 0.061
- Broader region: max FST (SAS vs non-SAS) = 0.091
- SAS-depleted variants (AF<10% SAS, >30% others): **127**
- SAS-enriched variants (AF>50% SAS, <30% others): **2**

The 127:2 depletion/enrichment ratio is a strong sweep signature
(even stronger asymmetry than GRK2's 42:1).

### Most differentiated variant: chr20:32,140,133

This variant is upstream of BPIFA2 (in the BPIFB region at 32.1Mb):
- SAS: BEB=60%, GIH=52%, ITU=50%, PJL=40%, STU=57% (~50%)
- EUR: CEU=56%, FIN=64%, GBR=52%, IBS=56%, TSI=50% (~56%)
- AFR: YRI=100%, ESN=100%, GWD=100%, MSL=100%, LWK=100% (~99%)
- EAS: CDX=100%, CHB=99%, JPT=100%, CHS=100%, KHV=98% (~99%)

Interesting: the most differentiated variant is NOT at BPIFA2 itself
but at the BPIFB cluster, and the SAS+EUR allele frequency is ~50%
while AFR+EAS is ~100%. This suggests the sweep is on a broader
haplotype affecting the entire BPIF cluster, but BPIFA2 shows the
strongest TMRCA signal because it's the functional target.

### BPIFA2-specific variant: chr20:33,326,225

Within the BPIFA2 gene:
- SAS: 3.2%, EUR: 3.9%, EAS: 12.9%, AFR: 83.4%
- FST (SAS vs non-SAS) = 0.085
- This variant is nearly absent in SAS+EUR but common in AFR —
  consistent with a swept-out ancestral allele

## Biological function

BPIFA2 (formerly PSP/SPLUNC2):
- Major salivary protein secreted by parotid gland
- First line of defense against oral pathogens
- Binds LPS, bacteria, yeast, and HDL
- Acts as a salivary surfactant (reduces surface tension)
- Anti-inflammatory: affects LPS signaling
- Anticandidal activity (Shiba et al. 2004)

**Connection to mucosal immunity narrative:**

| Population | Pathway | Genes | Surface |
|-----------|---------|-------|---------|
| EAS | Fungal recognition + IgA | CLEC6A, TRAF6, TNFRSF13C, JCHAIN | Intestinal |
| SAS | Salivary antimicrobial | BPIFA2 | Oral |

Both represent pathogen-driven adaptation at mucosal surfaces, but
in different populations and through different molecular mechanisms.

**Additional mucosal gene: PIGR**
- Polymeric immunoglobulin receptor, SAS mean = 8.9%
- Transports IgA across mucosal epithelia
- At 5.7% in ITU — same SAS population as BPIFA2
- If PIGR is also swept, it connects salivary defense (BPIFA2)
  to IgA transport (PIGR) in SAS populations

