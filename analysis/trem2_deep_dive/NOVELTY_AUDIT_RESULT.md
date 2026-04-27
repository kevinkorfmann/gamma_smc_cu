# Novelty audit — TREM2 pan-OoA selection, not IBS-novel

## 1. Our own iHS scan (`selscan_genelevel/`) flags TREM2/TREML1 in SAS

| Gene | SAS hits (rank_frac_ihs, top 5%) | EUR | EAS |
|---|---|---|---|
| TREM2 | BEB 0.001, STU 0.004 (top 0.1-0.4%) | **no computable iHS** (0 sites) | no computable iHS |
| TREML1 | BEB 0.001, ITU 0.002, STU 0.003, GIH 0.036 | bland (~0.74) | bland (~0.72) |

Selection signal at TREM2/TREML1 in South Asians is clear in our **own** data and would be picked up by any published iHS scan. **The TREM cluster is not a novel selection target.**

The absence of iHS signal in EUR/EAS is a power artifact — 0 computable iHS sites means TREM2 doesn't have enough common derived variants with extended EHH to compute the statistic there. It does NOT mean no selection — H12 evidence below tells the real story.

## 2. No IBS-specific variant exists inside the H12 peak

Top 15 variants by |dAF(IBS vs other EUR)| at 41.10–41.20 Mb (where H12 peaks):

- Max dAF(IBS vs CEU/FIN/GBR/TSI) = **0.060** (chr6:41,199,359) — **noise level**.
- IBS is indistinguishable from other Europeans at every common variant in the H12 peak.

Top 15 variants by |dAF(IBS vs AFR)|: max = **0.523** at chr6:41,166,068 (G>A). This is a classical **OoA-shared sweep variant** — AFR AF = 0.95, non-AFR AF ≈ 0.43. Located 3 kb downstream of TREM2 gene body.

## 3. The "IBS-specific" label is not supported

- TREM2 rank: IBS 0.34% vs FIN 0.33% vs GBR 0.37% — IBS is tied, not an outlier.
- H12 peak: IBS 0.649 vs FIN 0.690 vs GBR 0.701 — IBS is actually *lower* than other EUR.
- No IBS-specific variant in the H12 peak.

Conclusion: **the "IBS-specific TREM2 sweep" does not exist.** What exists is a shared OoA sweep near TREM2, with comparable signal across all non-African populations.

## 4. Consequence for manuscript

The v5 TREM2 case study needs either:

1. **Withdraw.** TREM2 fails the novelty audit when checked against pan-OoA scans.
2. **Re-purpose as a positive control**: "our scan recovers the known pan-OoA TREM2 sweep, replicating iHS hits in BEB/STU/ITU/GIH and extending detection to EUR/EAS via TMRCA where iHS is underpowered." This is defensible and scientifically useful — it shows the method works on a known target.
3. **Find a different IBS-specific signal** to occupy the "IBS case study" slot. Would require re-sorting the gene×pop matrix for genes where IBS rank is substantially lower than other EUR pops (not tied as here).

Option 2 is the honest path. It gives up novelty but preserves the scientific usefulness of the signal — demonstrating that TMRCA detects a sweep where haplotype-based scans lose power due to recombination decay.

## 5. Decision on Betty CLUES submission

Given the audit:

- **FOXP4 CLUES**: null guaranteed — no selection signal anywhere at FOXP4 (rank 20–46% in all populations, H12 = baseline). Submitting is wasteful unless you want a formal null paragraph in the manuscript.
- **TREM2 OoA CLUES**: scientifically meaningful — would give sweep age/strength for the pan-OoA TREM2 signal, which is now our best honest story. Focal variant: chr6:41,166,068 (G>A, AFR 0.95 → EUR 0.43).

Recommendation:
- **Submit CLUES at chr6:41,166,068 in IBS** (the true swept variant) to get sweep parameters for the pan-OoA TREM2 signal. This fits Option 2 above.
- **Do not submit CLUES on FOXP4** — no signal to find.
