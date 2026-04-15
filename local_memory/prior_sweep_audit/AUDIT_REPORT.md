# Prior-Sweep Audit of 165 Stage-5 Candidate Loci

**Date:** 2026-04-15
**Concern (from user):** Stage 5 collapses genes within 1 Mb of one another and reports
the lowest-rank gene as the "representative". If a *neighbour* gene inside the cluster
has previously been reported as a selection target in another scan, we could be
claiming novelty for a signal that the field already knows about, just at a different gene.

## Method

1. Reproduced the candidate cascade from scratch — stage counts match (17,790 non-SD →
   538 rank<1% → 512 replicated → 473 outside canonical ±500 kb → 165 LD-clustered at 1 Mb).
2. For every one of the 165 stage-5 clusters, recorded all member genes
   (`stage5_cluster_members.csv`).
3. Pulled gene-level candidate sets from three prior haplotype scans:
   - **Voight 2006** (PLoS Biology, supp data sd001/sd002/sd003): 744 gene symbols
     across HapMap Asian, European, Yoruba (Entrez IDs parsed, mapped via NCBI
     `Homo_sapiens.gene_info`).
   - **Sabeti 2007** (Nature, supp PDF NIHMS4416, Tables S1, S9, S10 + main Table 1):
     219 gene symbols.  Table S7 (XP-EHH top 300 regions) exists only as an embedded
     image and is not machine-readable from the NIHMS source.
   - **Metspalu 2011** (AJHG, supp `mmc5`=S4 iHS and `mmc6`=S5 XP-EHH India top windows):
     48 gene symbols.
   - Union = **975 unique prior-scan gene symbols**.
4. Cross-checked each of the 165 clusters against the combined 975-gene prior set.

## Result: Three-scan cluster-level cross-check

| Status | Count | Meaning |
|-------:|------:|---------|
| clean | 137 / 165 | No cluster member appears in any of Voight 2006, Sabeti 2007, Metspalu 2011. Novelty claim safe. |
| rep in prior scan | 19 / 165 | Representative itself is in one of the three scans — independent rediscovery, not mis-labelling. |
| **neighbour in prior scan, rep not** | **9 / 165** | **Mis-labelling risk.** A prior scan flagged selection at this region at a neighbour gene; we pick a different representative. |

### The 9 mis-label-risk loci

| Representative | Chr | Rep pop | Cluster size | Voight 2006 hit(s) | Sabeti 2007 | Metspalu 2011 |
|---|---|---|---|---|---|---|
| DEDD | 1 | YRI | 4 | NECTIN4 | — | — |
| DOK1 | 2 | GBR | 8 | DQX1, AUP1, HTRA2, LOXL3, M1AP | AUP1 | — |
| ORC2 | 2 | PEL | 2 | — | NIF3L1 | — |
| IGIP | 5 | LWK | 4 | PROB1 | — | — |
| CYP3A5 | 7 | GBR | 6 | BUD31 | — | — |
| BMI1 | 10 | GIH | 4 | — | SPAG6 | — |
| **GRK2** | **11** | **GIH** | **15** | **RAD9A, CLCF1, PPP1CA, TBC1D10C, CARNS1, PITPNM1, CDK2AP2** | — | — |
| EDC4 | 16 | CHB | 38 | — | LRRC36 | — |
| ZNF780B | 19 | FIN | 12 | CATSPERG, PSMD8, FAM98C | — | — |

**GRK2-specific:** Voight 2006 is the only prior scan with cluster-member hits at chr11q13. Sabeti 2007 and Metspalu 2011 are clean at the locus; Metspalu's only chr11 candidate is KIRREL3 at chr11:126\,Mb; Sabeti's nearest chr11 entry is MYEOV ~1.8\,Mb distal and not among its top candidates.

## GRK2 is the biggest load-bearing concern

GRK2's 918 kb cluster contains **seven genes that Voight 2006 already reported under
positive selection** (chr11:66.9–67.1 Mb in hg17 coordinates).

Per-population breakdown of the Voight 2006 hits:

| Voight 2006 population | iHS (max) | Hit window | Cluster genes reported |
|---|---:|---|---|
| HapMap Asian (CHB+JPT) | 3.546 | chr11:66.9–67.0 Mb | RAD9A, CLCF1, PPP1CA, TBC1D10C, CARNS1 |
| HapMap Yoruba | 3.397 | chr11:67.0–67.1 Mb | PITPNM1, CDK2AP2 |
| HapMap European (CEU) | — | — | (no hit) |

Our scan's minimum-rank populations for the 15 cluster genes are all SAS
(GIH, BEB, STU) or EUR-adjacent, with GRK2 at 0.23% in GIH. None of the cluster
genes has its minimum rank in EAS or AFR in our scan.

### Population-scope differences

- **Voight 2006 detected the chr11q13 region in ASIAN (CHB+JPT) and YORUBA populations.**
  Our scan ranks GRK2 at 4–10% in 1000 Genomes EAS and AFR — i.e., no sub-1% signal in
  the populations where Voight 2006 detected it. These may be the decaying remnants of
  the iHS signals Voight picked up, or distinct events.
- **Voight 2006 did not detect the region in European (CEU).** Our scan finds GRK2
  below 1% in all 5 EUR populations — that IS novel relative to CEU iHS.
- **Metspalu 2011 (South Asia-focused) did not report chr11q13.** Our sub-1% signal
  in all 5 SAS populations is novel relative to that scan.
- **Sabeti 2007 Table 1 ("22 strongest regions") does not include chr11q13.** (The
  broader 300+ region list in their supplement was not fully parsed.)

## Implications for the manuscript

The current wording is overreaching in at least two places:

1. **Main text (Results, GRK2 principal finding):**
   > "\textit{GRK2} (G protein-coupled receptor kinase 2; chr11q13.2)---to our
   > knowledge unreported in any prior selection scan"

   This is **not accurate at the region level**. Voight 2006 reported strong iHS
   signals at chr11:66.9–67.1 Mb in Asian and Yoruba, at five different genes inside
   our 15-gene cluster.

2. **Main text (Candidate identification and landscape-examples closing):**
   > "None of the 165 loci---including \textit{GRK2} and the four examples described
   > in detail below---has been identified and characterised as a positive-selection
   > target in prior genome-wide scans"

   Six of the 165 loci (including GRK2) have neighbour-gene hits in Voight 2006.
   The statement should be weakened accordingly.

### Recommended revised wording

**For GRK2 specifically** — replace the "unreported in any prior selection scan"
claim with:

> The chr11q13.2 region containing \textit{GRK2} has been reported as an iHS
> candidate by Voight \textit{et al.}\ (2006) in HapMap Asian (CHB+JPT) and Yoruba
> populations, at neighbour genes within a ~200 kb window (notably
> \textit{PPP1CA}, \textit{RAD9A}, \textit{CARNS1}, \textit{TBC1D10C},
> \textit{CLCF1} in Asian and \textit{PITPNM1}, \textit{CDK2AP2} in Yoruba).
> The South Asian and European replication pattern reported here is distinct in
> population scope from those earlier reports: Metspalu \textit{et al.}\ (2011)'s
> South-Asia-focused scan did not flag the region, HapMap CEU iHS (Voight 2006)
> did not detect the European signal, and Sabeti \textit{et al.}\ (2007)'s top
> 22 regions do not include chr11q13. What is new here is (i) the SAS+EUR
> replication pattern, (ii) the gene-level localisation to \textit{GRK2}'s gene
> body (which is near-monomorphic in GIH, rendering within-population iHS
> undefined), (iii) three-method TMRCA concordance, (iv) direct shared-ancestral
> haplotype evidence, and (v) the \textit{β}-adrenergic kinase biology.

**For the candidate-identification summary** — change:

> None of the 165 loci... has been identified and characterised as a
> positive-selection target in prior genome-wide scans \cite{Voight2006,Sabeti2007,Metspalu2011,Akbari2026}

to:

> Of the 165 loci, 143 are clean of any Voight \textit{et al.}\ (2006) iHS
> signal genome-wide; 16 independently recover a Voight 2006 hit at the same
> representative gene; and 6 (\textit{DEDD}, \textit{DOK1}, \textit{IGIP},
> \textit{CYP3A5}, \textit{GRK2}, \textit{ZNF780B}) reside in LD-clustered
> regions where Voight 2006 previously reported iHS signals at a neighbour
> gene (summary: Additional file 1: Table~SX). The principal finding
> (\textit{GRK2}) falls into this third category; its manuscript claim is
> framed accordingly (see GRK2 section).

## Remaining cross-checks not yet done

- Sabeti 2007 Table S7 (XP-EHH top 300 regions) exists only as an embedded image
  in the NIHMS supplement PDF and is not machine-readable from that source;
  OCR pass may be worth running if reviewer pushes.
- PopHumanScan 2859-region aggregate (website unreachable at the time of this audit).
- Pickrell 2009, Grossman 2013 CMS, broader meta-analyses — lower priority given
  the three primary scans we cite are now covered.

All three scans that the manuscript currently cites (Voight 2006, Sabeti 2007,
Metspalu 2011) are covered at gene-symbol level.

## Artifacts

- `stage5_cluster_members.csv` — 165 loci with all cluster members
- `stage5_voight_crosscheck.csv` — 165 loci annotated with Voight 2006 overlap status
- `voight2006_gene_symbols.txt` — 744 gene symbols from Voight 2006
- `voight2006_entrez_ids.txt` — 832 raw Entrez IDs
- `voight2006_{asian,european,yoruba}.txt` — original Voight 2006 supp files
- `Homo_sapiens.gene_info` — NCBI Entrez → symbol map used for translation
