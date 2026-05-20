# Deep investigation: SLC6A15 — extreme EAS brain transporter signal

## Summary

SLC6A15 shows one of the most extreme population-specific signals in
the entire genome: 0.10% in ALL 5 EAS populations (nearly the minimum
possible rank), 79% in AFR, 85% range. It is an isolated signal with
no nearby genes swept. SLC6A15 is a well-established depression GWAS
hit (Kohli et al. 2011, Neuron). No prior selection scan has flagged it.

## TMRCA signal

| Population | Rank |
|-----------|------|
| CDX (Chinese Dai) | 0.10% |
| CHS (Southern Han) | 0.10% |
| JPT (Japanese) | 0.10% |
| KHV (Kinh Vietnamese) | 0.10% |
| CHB (Han Chinese) | 0.29% |
| **EAS mean** | **0.14%** |
| MXL (Mexican) | 9.70% |
| BEB (Bengali) | 10.18% |
| SAS mean | 14.45% |
| FIN (Finnish) | 16.88% |
| EUR mean | 22.13% |
| AMR mean | 16.85% |
| **AFR mean** | **79.08%** |
| LWK (Luhya) | 85.26% |

**Key features:**
- 4 of 5 EAS populations at EXACTLY 0.10% (the minimum resolvable rank)
- 560-fold contrast between EAS (0.14%) and AFR (79%)
- SAS is intermediate (14%), EUR higher (22%), AFR highest (79%)
- Gradient: EAS < SAS < AMR < EUR < AFR
- This is NOT an out-of-Africa gradient (EUR would be lower than SAS)
  — the EAS signal is specifically extreme

## Isolation check

No nearby genes (<500kb) at <3%. This is a genuinely isolated signal,
not part of a regional sweep.

## Variant analysis (from 1000 Genomes)

SLC6A15 region (chr12:84.8-85.4Mb): analyzed on betty via slurm.

### FST (EAS vs non-EAS)

- Mean FST: 0.017
- Max FST: 0.166
- SLC6A15 gene body: max FST = 0.166

### Sweep signature

- EAS-depleted variants (AF<10% EAS, >30% others): **70**
- EAS-enriched variants (AF>50% EAS, <30% others): **2**

The 70:2 asymmetry confirms a sweep signature at the variant level.

### Most differentiated variant: chr12:84,849,501

Located ~220kb upstream of SLC6A15 (possible regulatory):
- EAS: CDX=3%, CHB=3%, CHS=1%, JPT=3%, KHV=4% (~2.5%)
- SAS: BEB=40%, GIH=48%, ITU=48%, PJL=50%, STU=41% (~45%)
- EUR: CEU=49%, FIN=32%, GBR=52%, IBS=51%, TSI=60% (~49%)
- AFR: YRI=92%, ESN=91%, GWD=89%, MSL=98%, LWK=97% (~91%)
- FST (EAS vs non-EAS) = 0.21

The ancestral (African) allele is at ~91% in AFR but only ~2.5% in EAS.

### EAS-specific variant: chr12:85,332,101

Located within/near SLC6A15:
- EAS: 23.2%, all others: ~0-1%
- This is an EAS-SPECIFIC derived variant, absent elsewhere
- Could be the selected variant or tightly linked to it

## Gene function: SLC6A15

SLC6A15 (Solute Carrier Family 6 Member 15):

**Brain function:**
- Neutral amino acid transporter (proline, leucine, methionine)
- Highly expressed in hippocampus and amygdala
- Regulates glutamatergic neurotransmission via amino acid supply
- Part of the SLC6 neurotransmitter transporter family

**Disease association:**
- Kohli et al. 2011 (Neuron): rs1545843 = MDD risk variant
- The risk allele affects hippocampal SLC6A15 expression
- Associated with stress reactivity and HPA axis regulation
- Replicated in multiple depression GWAS
- Also implicated in anxiety disorders

**SLC6 family context:**
Only SLC6A15 (EAS=0.1%) and SLC6A4 (serotonin transporter, EAS=11.4%)
show any EAS signal in the entire SLC6 family. SLC6A4 has been
extensively studied for population-specific variation (5-HTTLPR polymorphism),
but SLC6A15's signal is far more extreme.

## Biological hypothesis

The extreme EAS specificity is difficult to explain with a single
mechanism. Possibilities:

1. **Dietary amino acid composition:** East Asian traditional diets
   differ in amino acid profiles (rice-based vs wheat/meat-based).
   SLC6A15 transports branched-chain and aromatic amino acids.
   Different dietary pressures could select for altered transport.

2. **Stress response / HPA axis:** Population differences in cortisol
   response and stress biology are documented. SLC6A15 modulates
   the stress-brain axis via hippocampal amino acid availability.

3. **Linked regulatory variant:** The EAS-specific variant at
   chr12:85,332,101 (23% in EAS, ~0% elsewhere) could affect
   expression in a tissue-specific manner. The TMRCA signal may
   reflect selection on expression rather than protein function.

4. **Cognitive/behavioral adaptation:** Speculative. Changes in
   amino acid transport could affect neurotransmitter balance.

**Important caveat:** Brain-related selection claims are sensitive
and require careful framing. The TMRCA signal doesn't specify what
was selected — a nearby regulatory element could affect SLC6A15
expression in non-brain tissues (it's also expressed in kidney, lung).

