# Per-source verification index — `table_s_prior_scans.tex` (final state)

**Final state (2026-04-27 post-apply):** 52 rows in the table, 51 `\cite{}`
commands. Companion files: 60 per-source `NN_*.md` reports (kept under
original numbering for audit trail), `verified_bib_additions.bib` (the
agent-generated verified citations) and `proposed_bib_additions.bib` (the
original draft with `USER VERIFY` flags).

## What was applied

### Removed rows (8 total)

- Original rows 50, 51, 55, 57, 58, 59, 60 — the 7 web/dead/phrase
  resources you flagged.
- Original row 37 (Harris 2023 PNAS Bolivia) — the verifier could not
  locate a 2023 PNAS Bolivia paper by Harris on PubMed/Scholar; row
  dropped rather than cite an unverifiable reference.

### Display + journal corrections (audit-doc errors fixed)

| Old row | New row | Change |
|---|---|---|
| 23 | 23 | First author corrected: "Wu" → "Luo" (verified via DOI 10.1016/j.scib.2023.08.027) |
| 27 | 27 | Authorship corrected: solo "Quintana-Murci" → "Barreiro & Quintana-Murci" (verified DOI 10.1007/s00439-020-02167-x) |
| 29 | 29 | Journal corrected: *Cell* → *AJHG* (verified DOI 10.1016/j.ajhg.2015.11.014) |
| 31 | 31 | First author corrected: "Bentham" → "Nandakumar" (verified DOI 10.1093/molbev/msaf016) |
| 32 | 32 | Resolved: Patin & Quintana-Murci 2025 *Annu Rev Immunol*; year 2024 → 2025 |
| 33 | 33 | Journal corrected: *Trends Genet* → *Curr Opin Immunol* (DOI 10.1016/j.coi.2014.05.001) |
| 38 | 37 | Resolved: Vernot & Akey 2014 *Science* (over 2016 alternative); year "2014/16" → "2014" |
| 39 | 38 | Resolved: Racimo et al. 2017 *MBE* (over 2018 alternative); year "2017/18" → "2017" |
| 42 | 41 | Resolved: Villanea et al. 2025 *Science* MUC19 paper |
| 56 | 52 | Journal corrected: *J Hum Genet* → *Alz & Demen* (DOI 10.1016/j.jalz.2016.10.005); also wrong-cite `Guerreiro2013` dropped, replaced with `Carrasquillo2017` |

### Citations now resolved

51 of 52 rows carry `\cite{}`. The exception is row 22 (FineMAV) — kept
uncited because the audit doc's "Kozlowski 2022 BMC Bioinform" could not
be located; the original FineMAV paper is Szpak et al. 2018 *Genome Biol*
(DOI 10.1186/s13059-017-1380-2). The user must confirm which paper row 22
actually references before it can be cited.

### Bib entries added to `references.bib`

30 new entries appended to `private/manuscript/v5/references.bib`. All
have verified DOIs from the agent verification pass:

```
Zhang2024haplosweep, Lauterbur2023, Hejase2022sia, RefoyoMartinez2019,
Akbari2018isafe, Wu2023scibull, Pybus2014, BarreiroQM2020, Quach2016,
Deschamps2016, Barreiro2010, Bentham2025, Patin2025, FumagalliSironi2014,
Gurdasani2015, Choudhury2020, Lindo2018, VernotAkey2014, Racimo2017,
Browning2018sprime, Skov2020, Villanea2025muc19, vandenBelt2025,
SalazarTortosa2023, Colbran2026, Barton2026convergent, Li2014dbpshp,
Sims2017, Bellenguez2022, Carrasquillo2017
```

### Prose updates

- `main.tex` line 356: count `60 → 52`; dropped Mathieson 2015 reference
  (not in audit docs and not in table).
- `si.tex` line 251: count `60 → 52`; dropped Mathieson 2015 from prose;
  fixed Le year `2024 → 2022`; fixed `Quintana-Murci 2010/2019/2020` to
  list each author pair correctly; aDNA-scan year range `2015–2026 →
  2021–2026`; dropped `, ALZFORUM R47H` and trailing
  `, and explicit phrase-level web checks` to match the row removals.

## Build verification

- `latexmk -pdf si.tex` → 23 pages, no undefined-citation warnings.
- `latexmk -pdf main.tex` → 34 pages, no undefined-citation warnings.
- One pre-existing `Float too large for page` warning (unrelated to this
  task) at main.tex line 240.

## Outstanding flags for the user

These are intentional — places where the audit data and verifier disagreed
or could not be uniquely resolved:

1. **Row 22 FineMAV** — uncited. Verify whether the table refers to Szpak
   et al. 2018 *Genome Biol* (the original method paper) or a later
   implementation. The audit doc said "Kozlowski 2022 BMC Bioinform" but
   no such paper was located.
2. **Original row 37 (Harris 2023 PNAS Bolivia)** — DROPPED. If you have
   the actual citation, it can be re-added; verifier could not find it on
   PubMed/Scholar.
3. **Row 17 / 31 / 23 first-author rename mismatches** — the bib keys are
   `Zhang2024haplosweep` / `Bentham2025` / `Wu2023scibull` but the
   verified first authors are Zhao / Nandakumar / Luo respectively. The
   table display now shows the correct author; the bib key is just an
   internal identifier and is fine as-is. If you'd prefer matching keys
   (`Zhao2024haplosweep`, `Nandakumar2025`, `Luo2023scibull`), it's a
   simple sed across `references.bib` + `tables/*.tex`.
4. **Row 41 (Villanea 2025)** — resolved as the *Science* MUC19 paper by
   the verifier. If your audit was actually targeting an
   "introgression-map comparison" review, the alternate match is Chen,
   Velazquez-Arcelay & Capra 2026 *MBE* — let me know and I'll swap.
5. **Rows 47–48 (Colbran, Barton bioRxiv)** — DOIs use the new `10.64898`
   bioRxiv prefix (real, not a typo as previously suspected). Confirmed
   via PubMed PMIDs.
