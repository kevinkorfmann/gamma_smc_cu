# Findings database

**Current release: [2026-09-27](releases/2026-09-27/README.md).**
Use this version for the corrected atlas: 19,119 gene rows, 1,123 genes masked
for segmental duplication and 143 reporting loci. `LATEST.json` identifies the
release; its manifest provides checksums. Interpretation and usage are in
[the documentation](../docs/findings.md).

The numbered directories `01_` through `08_`, root `MANIFEST.csv`, and
[historical README](README_historical_2026-04-27.md) belong to the earlier deposit.
They retain older calibration, masks and/or case-study results and do not
represent the current manuscript. In particular, the old locus and duplication
counts are 165 and 1,296. Historical FDR-labelled columns must not be treated as
calibrated significance for the corrected atlas. Do not mix releases.

Verify the current deposit from the repository root:

```bash
python findings_database/verify_release.py
```

Raw cohort genotypes and all intermediate inference outputs are not bundled.
Each release states its own evidence and reproducibility limits.
