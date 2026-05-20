# Cross-check: main.tex + si.tex prose vs. `table_s_prior_scans.tex`

Generated 2026-04-27 alongside the 60 per-source reports. This file lists
every place the manuscript prose makes a claim that the audit table does
or does not back up — bugs as well as confirmations.

## 🚨 Bugs found

### Bug 1 — Mathieson 2015 cited in prose, missing from table

- `main.tex` line 356 — `\cite{Mathieson2015}` listed among the cross-checked
  aDNA scans alongside Akbari 2026, Speidel 2019 Relate, Field 2016, etc.
- `si.tex` line 260 — explicit "Mathieson 2015" in the audit summary
  enumerating "all major ancient-DNA scans 2015–2026".
- `tables/table_s_prior_scans.tex` — **no row for Mathieson 2015** anywhere
  in the 60 rows.
- `Mathieson2015` is in `references.bib`, so the citation resolves; but the
  audit table the prose cross-references claims to be exhaustive (60 rows)
  and excludes a paper the prose lists by name.

**Fix.** Two options:
- Add a row 14a/insert "Mathieson et al. 2015 *Nature*, aDNA 230 ancient
  Eurasians, no TREM2/TREML1 hit" — and renumber 14–60 → 15–61.
- OR drop "Mathieson 2015" from main.tex line 356 and si.tex line 260 prose
  if the audit didn't actually inspect that paper. Audit `FINDINGS.md` and
  `ROUND_7_DEEP_AUDIT.md` do not have a Mathieson 2015 sub-section, so
  option 2 is the cleaner fix unless the user has notes confirming the
  paper was actually checked.

⚠️ **User must decide before submission** — either the table is not
exhaustive of what's in the prose, or the prose claims a check the audit
docs don't record.

### Bug 2 — Le 2022 vs Le 2024 year mismatch

- `main.tex` line 356 — `Le~2022~\cite{Le2022}`
- `references.bib` — bib key is `Le2022`
- `si.tex` line 260 — "Le 2024"
- `tables/table_s_prior_scans.tex` row 13 — printed year "2024"
- `FINDINGS.md` table line 27 — "Le et al. *Nat Commun* 2024"
- `ROUND_7_DEEP_AUDIT.md` row 13 — "Le et al. *Nat Commun* 2024 aDNA"

The bib key + main-text prose say 2022; the SI prose, the table, and the
audit docs say 2024. Three vs three.

⚠️ **User must resolve.** Likely outcome: the paper is genuinely
"Le et al. 2024 *Nat Commun*" and the bib key + main-text prose are wrong;
in which case rename `Le2022` → `Le2024` (or keep the key, fix the prose).
If the paper *is* 2022, the table + SI + audit prose are wrong.

### Bug 3 — Row 56 cites the wrong paper

- `tables/table_s_prior_scans.tex` row 56 — describes Carrasquillo et al.
  2017 *J Hum Genet* (regulatory variant rs9357347-C eQTL × AD), but cites
  `\cite{Guerreiro2013}` (the 2013 NEJM TREM2 R47H paper).
- These are two different papers; the cell text and the cite do not match.

**Fix.** Drop the cite, or add the proposed `Carrasquillo2017` entry
(see `proposed_bib_additions.bib`) and replace.

### Bug 4 — SI prose lists "Quintana-Murci 2010" but row 30 is Barreiro & Quintana-Murci

- `si.tex` line 260 — "six dedicated immune-gene-selection reviews
  (Quintana-Murci 2010/2019/2020, Quach/Barreiro 2016, Deschamps 2016,
  Bentham 2025)".
- `table_s_prior_scans.tex` row 30 — "Barreiro & Quintana-Murci *NRG* 2010".
- These reference the same paper (the 2010 NRG immune-evolution review),
  but the SI prose drops Barreiro from the author abbreviation. Cosmetic.

**Fix.** Replace SI prose "Quintana-Murci 2010" with "Barreiro &
Quintana-Murci 2010" for consistency with row 30.

## ✅ Confirmations

The manuscript prose makes the following claims about specific subset
sizes; all are independently consistent with `table_s_prior_scans.tex`:

- "5 further haplotype-based scans" (main.tex 90, 154, 269, 356, 414) =
  Voight 2006 + Sabeti 2007 + Metspalu 2011 + Pickrell 2009 + Grossman
  2013 — confirmed against rows 1, 2, 25, 3, 4 of the table.
- "PopHumanScan + 5-scan prior-lit hit" (main.tex 222) — same set + row 7.
- "60 distinct selection-scan resources audited across seven progressively-deeper
  rounds" (si.tex 260, table caption) — confirmed: table has exactly 60 rows.
- "Six dedicated immune-gene-selection reviews" (si.tex 260) — rows 26, 27,
  28, 29, 30, 31 = QM 2019 + QM 2020 + Quach/Barreiro 2016 +
  Deschamps 2016 + Barreiro & Quintana-Murci 2010 + Bentham 2025 = six.
- "Regional cohort scans (AGVP, GCAT, Han Chinese, Bolivian, Iberian)"
  (si.tex 260) — rows 14 (GCAT/Iberian), 23 (Wu Han Chinese), 34 (AGVP),
  36–37 (Bolivian Lindo + Harris). The "Iberian" entry is GCAT itself, so
  GCAT and Iberian overlap; that's intentional.
- The TREM2 quantitative numbers used in the case study (Akbari 2026 max
  POSTERIOR 0.07/0.10/0.60; PEL/CLM iHS rank 8.7%/p90–p95) are sourced
  consistently from rows 6, 15, 47.

## ⚠️ Remaining table-vs-prose loose ends

- The SI prose at line 260 explicitly lists 7 of the 8 "modern ARG- and
  ML-based methods" (Speidel 2019, SIA, HaploSweep, Flex-Sweep, GRoSS,
  FineMAV, FASTER-NN). Counting: rows 9, 19, 17, 18, 20, 22, 46 = 7.
  iSAFE (row 21) is NOT listed in the SI prose — should the prose include
  it for completeness, or is iSAFE intentionally excluded as scope-limited
  (it tests on 22 known sweeps rather than scanning genome-wide)?
- The SI prose says "Quintana-Murci 2010/2019/2020" — three entries — but
  the table has only QM 2019 + QM 2020 (rows 26, 27); the 2010 entry is in
  row 30 attributed to **Barreiro &** Quintana-Murci 2010. See Bug 4.
