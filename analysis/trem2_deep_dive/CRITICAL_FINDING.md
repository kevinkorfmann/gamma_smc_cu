# CRITICAL — "TREM2 IBS-specific sweep" claim does not hold up

Stop-the-press finding before submitting CLUES compute. Re-examining our own scan outputs:

## 1. The sweep is pan-non-African, NOT IBS-specific

`genome_wide_ranks.csv` — TREM2 within-population rank (%):

| Pop | TREM2 | TREML1 | FOXP4 |
|---|---|---|---|
| CEU | 0.50 | 0.72 | 27.6 |
| FIN | 0.33 | 0.54 | 46.2 |
| GBR | 0.37 | 0.48 | 29.3 |
| **IBS** | **0.34** | **0.58** | **35.7** |
| TSI | 0.47 | 1.08 | 30.7 |
| JPT | 0.41 | 0.44 | 33.9 |
| CHB | 0.72 | 0.69 | 33.9 |
| GIH | 0.90 | 1.32 | 28.6 |
| YRI | 1.57 | 0.64 | 22.3 |

Every non-African population ranks TREM2 below 1%. **IBS is not an outlier among Europeans or Eurasians.** The v5 panel (a) title "TREM2 within-population rank — IBS-specific" is not supported by the data — IBS at 0.34% is tied with FIN (0.33%) and GBR (0.37%).

FOXP4: rank 22–46% everywhere. **No selection signal at FOXP4 in any population.** FOXP4 can now be dropped from consideration.

## 2. H12 peak is inside TREM2 gene body, not near 41.47 Mb

H12 (400-SNP windows) across chr6:41.0–41.7 Mb:

| Pop | n_hap | max H12 | position | H12 at 41.47 Mb focal |
|---|---|---|---|---|
| IBS | 210 | 0.649 | **41.150 Mb** | 0.010 |
| CEU | 198 | 0.647 | 41.150 Mb | 0.016 |
| FIN | 198 | 0.690 | 41.150 Mb | 0.016 |
| GBR | 182 | 0.701 | 41.150 Mb | 0.012 |
| TSI | 214 | 0.583 | 41.156 Mb | 0.012 |
| JPT | 208 | 0.726 | 41.156 Mb | 0.012 |
| CHB | 206 | 0.653 | 41.215 Mb | 0.010 |
| YRI | 216 | 0.176 | 41.019 Mb | 0.017 |

- All non-AFR populations show the **same H12 peak at ~41.15 Mb**, which sits **10 kb upstream of TREM2 gene body** (41,158,506–41,163,186).
- The "focal" at 41,470,132 and "most-differentiated" at 41,485,209 have **H12 ≈ 0.01** (baseline) in every population — they are not on the swept haplotype.
- YRI is flat everywhere (max 0.176 vs ~0.6–0.7 elsewhere), confirming the sweep is OoA.

## 3. What actually happened

The chain of previous errors:

1. The `TREM2_IBS.json` "most-differentiated" variant was picked by maximum FST between IBS and non-IBS. But since the sweep is **nearly fixed across all non-African populations**, FST at the true sweep variants is near zero between IBS and other Eurasian pops. The algorithm therefore picked up variants in *other* LD blocks that happen to differ between IBS and the rest — noise, not signal.
2. The haplotype-sharing "focal" at 41,470,132 was chosen from the most-differentiated variant position and propagated the same error.
3. The v5 figure labels this "IBS-specific" based on the 26-population rank panel, but the numbers (0.34% IBS vs 0.33% FIN vs 0.37% GBR) do not support that claim.

## 4. Corrected interpretation

There IS a real selection signal at 6p21.1 — just not the one we were telling.

- **Location**: TREM2 / TREML1 locus, H12 peak at ~41.15 Mb.
- **Populations**: shared across all non-African populations (shared OoA ancestry).
- **Novelty status**: **not novel.** A pan-OoA TREM2-region signal would have been picked up by any classical haplotype scan. We should expect Voight/Sabeti/Pickrell hits if we look.
- **FOXP4 connection**: none. FOXP4 ranks 20-46% everywhere and shows no H12 signal.

## 5. What to do with the manuscript

The TREM2 case study as currently framed in v5 has three issues that must be reconciled:

1. **"IBS-specific" framing is incorrect**; the rank data shows it's pan-non-AFR.
2. **The focal-variant selection for the TMRCA panel and haplotype-sharing panel points at 41.47 Mb**, which is outside the actual H12 sweep block.
3. **The FOXP4 angle I was developing is not supported by the data.** Withdraw it.

Options:
- **(i)** Re-frame as "pan-non-AFR sweep at the TREM2/TREML1 locus" — honest but loses novelty (this would be an already-known shared OoA signal).
- **(ii)** Check whether the TREM2 signal passes the novelty audit at all given its shared OoA character — if other scans have hit TREM2 before, it's not novel and should be removed from the novel-hits list.
- **(iii)** Look for a **different** gene in v5 that actually has IBS-specific haplotype-structure support, and reassign the "IBS case study" slot.

## 6. CLUES on FOXP4 — abort recommended

Running CLUES on FOXP4 in IBS is now expected to return **null** (no selection). Given H12 peak at FOXP4 is baseline (≈0.03) and TMRCA rank is 35.7%, there is no sweep to detect. Submitting the job to Betty would waste compute and produce a defensible but uninformative null.

If the user still wants the CLUES run as a formal null control (e.g., "FOXP4 CLUES showed no evidence of selection, ruling it out as a sweep target"), the scripts are ready in `analysis/relate_clues/scripts/` modeled on the TREM2 versions — I can submit them. But the expected outcome is unambiguous.

## 7. Recommended immediate next actions

Before touching CLUES or main.tex:

1. **Verify the v5 novelty-audit results**: was TREM2 actually checked against prior OoA-wide selection scans (iHS, XP-EHH) or only against "IBS-specific" hits? If the latter, the novelty claim collapses with this finding.
2. **Re-run the genome-wide novelty check excluding pan-OoA sweeps** to see which signals survive.
3. **Find the true sweep variant** inside the 41.15 Mb H12 peak (probably something in TREM2 intron 1 or the TREML1-TREM2 intergenic region). If CLUES is still useful, it should be run on *that* variant in any non-AFR population, not on 41,470,132 or FOXP4.

FOXP4 CLUES scripts are prepared but not submitted pending decision.
