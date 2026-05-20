# tmrca.cu findings database

A self-contained deposit of every numerical claim in the manuscript, organized so a
reader can match any number, table, or figure in the paper to a primary CSV/TXT/JSON
file in this directory.

`MANIFEST.csv` is the master index: one row per file or directory with the manuscript
location it backs, a short description, and the producer script.

## Layout

| Folder | What it contains | Manuscript anchor |
|---|---|---|
| `01_genome_wide_scan/` | Per-gene TMRCA + within-population rank for all 19,119 protein-coding genes × 26 populations; SD-overlap flags; pathway enrichment; recurring constants. | Methods §Genome-wide scan, §SD masking; Fig 2-3; basis of every rank-based claim. |
| `02_tables/` | CSV equivalents of the LaTeX tables in the manuscript and SI. | Tables 1, 2, S2, S3, S4 + FDR q-values + population sample counts. |
| `03_variant_evidence/` | (a) `all_genes_summary.csv`: 12-gene flat CSV with window-level (no AF cutoff) FST + depleted/enriched counts. (b) `per_gene_json/`: 12 raw per-gene records. (c) `manuscript_table2_source_findings/`: the 5 narrative FINDINGS.md files whose AF-cutoff counts (42:1, 127:2, 70:2, 265:18, …) are quoted in Table 2. | Methods §Variant-level confirmation (lines 865-876); Table 2 / Table 3 orthogonal columns. |
| `04_haplotype_stats/` | Garud H12 (genome-wide and ±500 kb gene-centred), XP-EHH vs YRI for the candidate gene set. | Methods §Haplotype-based orthogonal statistics; Table S2 H12 column; Table S3; Fig S7c. |
| `05_clues_inference/` | CLUES2 trajectory inference text outputs (`*_result_inference.txt`, `*_freqs.txt`, `*_post.txt`, `*_CI.txt`) for LCT, GRK2, CCDC92, BPIFA2, SLC6A15, CLEC6A, C11orf65, TREM2. | Methods §CLUES; Figs 4-7. |
| `06_trem2_case_study/` | Per-population H12, haplotype-sharing JSON, three-method concordance, OoA Δ-AF for the TREML1/TREM2 deep dive. Also GRK2 companion files. | Section 2.5 (TREML1/TREM2 case study); GRK2 deep dive Fig 5. |
| `07_prior_literature_audit/` | The full literature review behind the novelty claims: 60 per-resource markdown reports; the 7-round audit summary; per-gene novelty review for the 5 candidates; stage-4 / stage-5 prior-scan flag CSVs; Table S5 source catalog; pre-submission citation audit. | Section 2.5 ("52 distinct selection-scan resources audited across seven progressively-deeper rounds"), Discussion §Novelty, Table S5. |
| `08_akbari_aDNA_replication/` | Akbari 2026 lead variants in both GRCh37 (original) and GRCh38 (lifted) frames; the TMRCA × time matrix used to render the concordance heatmap; per-chromosome gene-window TSVs. | Discussion §Akbari 2026 cross-validation; Fig 8. |

## Reproducing key numbers from the paper

**Caption of Table S4** ("SD-flagged genes (1,296 of 19,119, 6.8%)"):
```python
import pandas as pd
sd = pd.read_csv("01_genome_wide_scan/sd_flag.csv")
n, N = int(sd["is_sd"].sum()), len(sd)
print(n, N, f"{100*n/N:.4f}%")   # 1296 19119 6.7786%
```

**Table S4 row 17 — GRK2, GIH, 0.23%**:
```python
top = pd.read_csv("02_tables/table_s4_top50_sd_masked.csv")
top.iloc[16][["gene_name","chr","min_rank","min_pop","start"]]
# GRK2  11  0.002278  GIH  67266473   →  0.23 %, 67.3 Mb
```

**Table 2 — GRK2 42:1 depleted:enriched ratio, max FST 0.28**:
- Curated AF-cutoff numbers in `02_tables/table2_novel_candidates.csv` row 1.
- Provenance narrative: `03_variant_evidence/manuscript_table2_source_findings/GRK2_findings.md`
  (search "SAS-depleted variants" / "FST = 0.28").
- Companion no-cutoff window-level numbers (n_depleted, max_fst, etc.) in
  `03_variant_evidence/all_genes_summary.csv` and per-gene `GRK2_GIH.json`.

**Section 2.5 — "52 distinct selection-scan resources audited across seven progressively-deeper rounds"**:
- Per-resource reports: `07_prior_literature_audit/source_reports/01_voight_2006.md` …
  `60_google_scholar_phrase.md` (numbered chronologically).
- Round-by-round summary: `07_prior_literature_audit/literature_audit_findings.md`
  and `round7_deep_audit.md`.

## What is NOT in this deposit

- Raw 1000 Genomes phased VCFs (publicly available; URLs in Methods §Data).
- Per-population genome-wide selscan iHS/nSL outputs (~80 MB; available at
  `analysis/orthogonal_v41/selscan_genelevel/` in the source repository).
- Inference checkpoints / `.npz` model weights (binary, not CSV/TXT-friendly).

## Build provenance

The CSVs in `02_tables/` are the file equivalents of the LaTeX tables produced by
`private/manuscript/v5/tables/gen_tables.py`. The flat variant-evidence CSV in
`03_variant_evidence/all_genes_summary.csv` is reproducible from the 12 JSONs in
`per_gene_json/` — see the `pd.json_normalize`-style flattening logic in this
repository's deposit-builder commit.

Last refreshed: 2026-04-27.
