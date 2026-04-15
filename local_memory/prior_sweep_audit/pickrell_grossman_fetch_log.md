# Pickrell 2009 + Grossman 2013 supplement fetch log — 2026-04-15

## Pickrell et al. 2009 (Genome Research, DOI 10.1101/gr.087577.108)

- **PMC ID clarification:** task brief gave PMC2698831 but that is the `Sabeti et al. 2007` PMC entry; Pickrell's actual PMC is **PMC2675971**. Genome Research supplement is publicly hosted so PMC was not needed.
- **Successful path:** `https://genome.cshlp.org/content/suppl/2009/03/25/gr.087577.108.DC1/supp.pdf` — HTTP 200, 2.6 MB, no auth required.
- **File:** `pickrell_supp.pdf`. Contains all the candidate-region gene tables on pages 10–31 (per-population candidate regions with `chr:Mb.start:Mb.end  GENE1,GENE2,...` layout).

## Grossman et al. 2013 (Cell, DOI 10.1016/j.cell.2013.01.035)

- **Elsevier CDN attempts** (`https://ars.els-cdn.com/content/image/1-s2.0-S0092867413001001-mmcN.{pdf,zip,xlsx,xls,doc,docx}` for N=1..7) — all 404. No files exposed under the S00928... DOI basename.
- **PMC PoW-protected path:** solved the `cloudpmc-viewer-pow` SHA-256 challenge (difficulty 4, cookie `<challenge>,<nonce>`), fetched:
  - `grossman_S01.pdf`  7.23 MB  main SI PDF (18 pp)
  - `grossman_S02.xlsx` 51 KB   154 genome-wide CMS regions (107 with gene annotations)
  - `grossman_S03.xlsx` 51 KB   438 localized CMS regions (235 with gene annotations)
  - `grossman_S04.xlsx` 38 KB   functional enrichment
  - `grossman_S05.xlsx` 39 KB   pathway enrichments
  - `grossman_S06.xlsx` 32 KB   lincRNA (36 with gene annotations)
  - `grossman_S07.xlsx` 41 KB   35 nonsyn variants (protein symbols)
  - `grossman_S08.xlsx` 105 KB  eQTL annotations
  - `grossman_S09.xlsx` 97 KB   enhancer/promoter annotations
  - `grossman_S10.xlsx` 67 KB   GWAS overlaps

## Parse summary

- Pickrell 2009 — **2,572 unique gene symbols** from 1,229 candidate-region lines.
- Grossman 2013 — **610 unique gene symbols** from 107 + 235 + 36 + 35 annotated regions/variants.
