# 46 — FASTER-NN (van den Belt & Alachiotis 2025 *Comm Biol*)

- **Cite**: needed — bib entry does not exist
- **Method / cohort**: CNN-based selection scan / 1000G CEU
- **TREM2 / TREML1 / 6:41 Mb**: no — nearest chr6 hit is 17 Mb away (chr6:58,515,003)
- **How checked**:
  1. SI list parsed: `MOESM1_ESM.docx` (description), `MOESM2_ESM.pdf` (top-0.5% **purifying** selection regions on 1000G CEU), `MOESM3_ESM.pdf` (Reporting Summary). **No SI4 or SI5** — Round-6 caveat closed in R7.
  2. Coordinate intersect of SI2 chr6 entries against chr6:41,149,167–41,163,186 (TREML1/TREM2 block) — zero overlap.
  3. Figshare 26139454: 1.8 GB simulation training/test data only.
- **Confidence**: 🆕 needs new bib entry. DOI `10.1038/s42003-025-07480-7` verified by audit.
- **User action**: verify `vandenBelt2025` (full title + authors).

**Important caveat to keep in the table cell.** FASTER-NN's only published real-data scan is **purifying** selection, not positive — the row's "no" should retain the qualifier. The cell already says "(nearest chr6 hit 17 Mb away)" which is the correct framing. ROUND_7 §A1 + FINDINGS.md round-6 final blocks.
