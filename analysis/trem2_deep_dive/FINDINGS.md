# TREM2-IBS deep dive — three verification checks

Exploratory only. Not integrated into main.tex.

## Coordinates (GRCh38)

| Entity | chr6 position | Source |
|---|---|---|
| TREM2 gene body | 41,158,488–41,163,186 | Ensembl |
| TREML1 | 41,149,167–41,155,553 | Ensembl |
| TREML2 | 41,189,749–41,201,149 | Ensembl |
| TREML4 | 41,228,339–41,238,882 | Ensembl |
| TREM1 | 41,267,100–41,286,682 | Ensembl |
| NCR2 | 41,335,608–41,350,889 | Ensembl |
| **FOXP4** | **41,546,363–41,602,384** | Ensembl |
| MDFI | 41,636,840–41,654,246 | Ensembl |
| TFEB | 41,683,978–41,736,259 | Ensembl |
| **Our focal sweep variant** | **41,470,132** | `trem2_haplotype_sharing.json` |
| **Our most-differentiated variant** | **41,485,209** | `TREM2_IBS.json` (ΔAF = −0.40) |

**Critical observation:** the focal sweep variant is **>307 kb downstream of TREM2** and only **~61 kb upstream of FOXP4**.  The most-differentiated FST variant is **~322 kb from TREM2** and **~61 kb from FOXP4**. Labelling this signal "TREM2" reflects the 1 Mb window centered on TREM2, not the gene most proximal to the peak.

## 1. Akbari 2026 aDNA posterior at this region

Source: `private/manuscript/v5/tests/fixtures/akbari_TREM2.tsv` (Akbari et al. 2026 extracted window, chr6:40,626,344–41,630,783, 4,059 variants, 3,322 PASS).

**Gene-body posteriors:**
- TREM2 gene body ±10 kb: max POSTERIOR = 0.32 (MAF<0.01 warning), 0.14 among PASS. **Flat.**
- At focal sweep variant (nearest: rs12665636 @ 41,469,379, PASS): **POSTERIOR = 0.028**. **Flat.**
- At most-differentiated variant (nearest: rs62396705 @ 41,485,841, PASS): **POSTERIOR = 0.187**. **Flat.**

**Subthreshold peaks in the window (POSTERIOR ≥ 0.5):**

| rsID | pos | POSTERIOR | AF | ŝ | Nearest gene |
|---|---|---|---|---|---|
| rs11760063 | 41,522,016 | **0.746** | 0.081 | 0.0090 | **FOXP4 intron** |
| rs144423624 | 41,338,397 | 0.689 | 0.012 | 0.0223 | NCR2 region |
| rs115179763 | 41,233,358 | 0.603 | 0.013 | 0.0204 | TREML4 region |
| rs150620688 | 41,282,621 | 0.592 | 0.012 | 0.0210 | TREM1 region |

**None reach Akbari's FDR=0.99 threshold** → TREM2 correctly absent from their main hit list. But the **subthreshold peak sits inside FOXP4** (rs11760063, VEP: `intron_variant`, `ENSG00000137166`).

**Framing implication:** the earlier hypothesis ("TREM2 not in Akbari because too recent / IBS-specific") is partly supported — no high-confidence hit — but the subthreshold structure points to a nearby gene (FOXP4) rather than TREM2 itself.

## 2. Archaic introgression (Sprime / Skov / admixfrog) at the region

Checks performed:
- Browning 2018 Sprime 1000G catalog (Mendeley dataset y7hyt83vxr): not directly readable via WebFetch; full segment list would need local download.
- Targeted literature search for introgressed haplotypes at chr6:41.1–41.7 Mb, TREM2, TREML1/2/4, NCR2, FOXP4: **no published reports** flag the TREM cluster or FOXP4 as an adaptively introgressed region in Europeans (checked: Racimo 2017, Vernot 2016, Browning 2018, Jagoda 2018, Skov 2020 Icelandic, McArthur 2021, Villanea 2025 bioRxiv introgression-map comparison).
- No published Zeberg/Pääbo-style introgressed-risk-haplotype for FOXP4 (contrast: chr3 LZTFL1 COVID, chr12 OAS COVID).

**Conclusion:** no evidence in the published catalogs that this sweep is archaic introgression. A definitive null requires running Sprime or admixfrog on 1000G IBS+YRI locally for this window — feasible future work; the input VCF `trem2_pm500kb.vcf.gz` is already available.

## 3. GTEx eQTL + coding overlay

- GTEx REST API: TREM2 whole-blood cis-eQTL query for gene ENSG00000095970 returns **0 records** in v8. TREM2 has modest expression in blood; strong microglia/brain expression dominates the regulatory story — GTEx bulk brain tissues dilute microglial signal.
- **FOXP4 has multiple reported eQTLs** in the same window: lead variant rs9367106 (41,515,652) is a cis-eQTL for FOXP4 in lung and brain hippocampus, and is the GWAS lead for both severe COVID-19 and long COVID (Kousathanas 2022; Lammi 2023, *Nat Genet*). rs9367106 falls **45 kb from our focal variant**.
- LD between rs9367106 and rs2496644 (R²=0.88 in EUR) places most of the FOXP4 GWAS/eQTL signal in a ~100 kb block overlapping the right flank of our swept region.
- **Coding variants:** no protein-coding variants in TREM2 itself (41.16 Mb) show unusual differentiation in `TREM2_IBS.json`; none of the canonical AD variants (R47H rs75932628, R62H rs143332484, H157Y rs2234255) appear among top differentiated variants in the window.

## 4. Revised interpretation

The signal previously labelled "TREM2-IBS" is almost certainly **not driven by TREM2**.

Proximity, Akbari subthreshold structure, and GTEx eQTL evidence all point to **FOXP4** as the more plausible functional target. Secondary candidates (TREM1, NCR2, TREML2/4) cannot be ruled out — the full TREM-FOXP4 cluster is in LD.

Recommended re-labelling for the Discussion: **"6p21.1 TREM–FOXP4 cluster sweep in IBS"**, with FOXP4 highlighted as the strongest biological candidate given its independent selection signature, COVID-severity eQTL, and lung/respiratory trait footprint.

## 5. Implications for the v4.1 novelty audit

Memory `project_grk2_not_novel.md` states: "only TREM2 (IBS) is fully clean gene+pop-novel."

This claim is **still technically defensible at the window level** — no prior scan hits this 6p21.1 block in IBS — but the *gene* attribution is likely wrong. Before keeping the TREM2 label in the manuscript, we should either:
  (a) rename to TREM–FOXP4 cluster and discuss FOXP4 as the functional candidate, or
  (b) run fine-mapping / eQTL colocalization on the sweep haplotype to commit to a gene.

Either way, the current framing ("TREM2 loss-of-function is under strong purifying selection, therefore positive signal must be regulatory") should be replaced with a framing that names FOXP4 as the leading candidate and treats TREM-cluster involvement as an alternative.

## 6. Concrete follow-ups before manuscript edits

1. Run Sprime on `trem2_pm500kb.vcf.gz` + a YRI outgroup to confirm no archaic segment.
2. Identify which variants define the swept haplotype (haplotype frequency ≥ 0.5 in IBS, < 0.1 in YRI/JPT) and test for colocalization with FOXP4 cis-eQTLs (rs9367106 block).
3. Check whether our most-differentiated variant (41,485,209) is in LD with rs9367106 / rs2496644 in IBS.
4. Cross-check Akbari's full (not fixture) results for chr6:41.0–41.7 Mb — the fixture may be truncated; a subthreshold POSTERIOR 0.75 is already suggestive.
5. Decide on locus name: "TREM2" (current), "TREM–FOXP4 cluster" (neutral), or "FOXP4" (committal). The cluster name is safest pending colocalization.

## References

- Akbari et al. 2026 — aDNA time-series selection scan (supplementary data used).
- Ensembl REST — gene coordinates and VEP.
- Kousathanas et al. 2022 *Nature* — severe COVID FOXP4 locus.
- Lammi et al. 2023 *Nat Genet* — long COVID FOXP4 rs9367106.
- Jonsson 2013 / Guerreiro 2013 *NEJM* — TREM2 R47H (reference for why TREM2 is not the likely driver: the sweep misses the coding region).
- Colonna 2023 *Nat Rev Immunol* — TREM biology.
- Browning 2018 Sprime; Skov 2020 Icelandic introgression map — absence of published flag at this region.
