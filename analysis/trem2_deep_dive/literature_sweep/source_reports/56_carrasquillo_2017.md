# 56 — Carrasquillo et al. 2017 *J Hum Genet*

- **Cite**: ⚠️ table currently cites `\cite{Guerreiro2013}` — **WRONG**. Guerreiro 2013 is the original NEJM TREM2-AD paper (already correctly used in main.tex line 348). Carrasquillo 2017 is a different paper (regulatory variant rs9357347-C eQTL × AD).
- **Method / cohort**: eQTL + AD / brain
- **TREM2 / TREML1**: not selection — regulatory variant rs9357347-C tied to TREM expression in brain
- **How checked**: paper inspection
- **Confidence**: ⚠️ uncertain — **bug in current table**
- **User action**:
  1. Drop `\cite{Guerreiro2013}` from row 56 (it does not match the row description).
  2. Either (a) add proposed `Carrasquillo2017` to `references.bib` and cite it here, or (b) drop the row entirely (it is curated/not-selection so non-essential to the audit).

**Notes.** This is the only row in the table where the existing `\cite{}` is **incorrect**. FINDINGS.md round-4 line 162.
