# 06 — Johnson & Voight 2018 (iHS refinement, 26 1000G panels)

- **Cite**: `JohnsonVoight2018` (exists; already cited in table)
- **Method / cohort**: iHS over 26 1000G panels
- **TREM2 / TREML1 / 6:41 Mb**: no / no / 100-kb top-1% rank fraction = 8.7% best (CLM); below cutoff in all 26 populations
- **How checked**:
  1. Downloaded 1.2 GB Zenodo archive `JohnsonEA_iHSscores.tar.gz` (`zenodo.org/api/records/7842512/files/...`)
  2. Per-SNP |iHS|>2.0 grep inside TREML1/TREM2 gene bodies (38 + 41 SNPs; top |iHS|=2.37 PEL, 2.91 MSL)
  3. Re-implemented J&V's published candidate criterion (top-1% of 100-kb windows by fraction or count of |iHS|>2 SNPs); CLM ranked 143/1645 (8.7%) — does not pass.
- **Confidence**: ✅ verified
- **User action**: none — already cited

**Notes.** Strongest single-resource raw-data re-analysis in the audit. Closes Round-6 Gap A3. ROUND_7_DEEP_AUDIT.md §A3.
