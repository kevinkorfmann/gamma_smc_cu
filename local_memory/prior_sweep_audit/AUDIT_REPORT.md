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
3. Pulled the three prior scans the manuscript cites:
   - **Voight 2006** (PLoS Biology) — parsed supplementary data files
     (sd001/sd002/sd003, 100 kb windows with Entrez gene IDs for Asian, European,
     Yoruba), mapped Entrez → symbol via NCBI `Homo_sapiens.gene_info` → **744 unique
     gene symbols** flagged by Voight 2006 iHS.
   - **Sabeti 2007** (Nature, Table 1, "22 strongest regions") — chr and gene names
     pulled from the paper text (full supplement download blocked by paywall but the
     Table 1 list is the load-bearing reference).
   - **Metspalu 2011** (AJHG, South Asia-focused) — main-text candidate list.
4. Cross-checked each of the 165 clusters against the 744-gene Voight 2006 set.

## Result: Cluster-level cross-check

| Status | Count | Meaning |
|-------:|------:|---------|
| clean | 143 / 165 | No cluster member appears in Voight 2006 — novelty claim safe. |
| rep in Voight | 16 / 165 | Representative itself is in Voight 2006 — we are independently re-detecting, not mis-labelling. Fine for validation. |
| **neighbour in Voight, rep not** | **6 / 165** | **Mis-labelling risk.** Voight 2006 flagged selection at this region at a neighbour gene; we pick a different representative; naive reading of our paper would claim novelty. |

### The 6 mis-label-risk loci

| Representative | Chr | Rep pop | Cluster size | Span (kb) | Voight 2006 hit(s) inside cluster |
|---|---|---|---|---|---|
| DEDD | 1 | YRI | 4 | 94.7 | NECTIN4 |
| DOK1 | 2 | GBR | 8 | 322.7 | DQX1, AUP1, HTRA2, LOXL3, M1AP |
| IGIP | 5 | LWK | 4 | 741.9 | PROB1 |
| CYP3A5 | 7 | GBR | 6 | 271.4 | BUD31 |
| **GRK2** | **11** | **GIH** | **15** | **918.5** | **RAD9A, CLCF1, PPP1CA, TBC1D10C, CARNS1, PITPNM1, CDK2AP2** |
| ZNF780B | 19 | FIN | 12 | 1,771.1 | CATSPERG, PSMD8, FAM98C |

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

- Sabeti 2007 full supplementary table (beyond Table 1's 22 regions)
- Metspalu 2011 supplementary tables (top 20 iHS + XP-EHH windows — most could be
  obtained by hand-parsing their supplementary PDF)
- PopHumanScan 2859-region aggregate (website unreachable at the time of this audit)
- Pickrell 2009, Grossman 2013 CMS, broader meta-analyses

If any of these adds hits to the GRK2 cluster, the wording above should be
further softened. If they all come back clean, the above is sufficient.

## Artifacts

- `stage5_cluster_members.csv` — 165 loci with all cluster members
- `stage5_voight_crosscheck.csv` — 165 loci annotated with Voight 2006 overlap status
- `voight2006_gene_symbols.txt` — 744 gene symbols from Voight 2006
- `voight2006_entrez_ids.txt` — 832 raw Entrez IDs
- `voight2006_{asian,european,yoruba}.txt` — original Voight 2006 supp files
- `Homo_sapiens.gene_info` — NCBI Entrez → symbol map used for translation
