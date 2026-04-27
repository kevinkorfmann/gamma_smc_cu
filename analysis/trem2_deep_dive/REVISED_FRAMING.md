# Revised TREM2 framing for v5 — pan-OoA sweep missed by classical scans

## The claim that survives

**TREM2 harbors a pan-OoA selective sweep that classical haplotype scans miss due to the compact (4.7 kb) gene body. TMRCA recovers the signal because it does not depend on haplotype-length decay.**

This framing:

- Gives up the IBS-specific label (not supported — IBS 0.34% tied with FIN 0.33%, GBR 0.37%).
- Keeps novelty at the literature level — no prior publication flags TREM2 as a selection target; canonical catalog searches return nothing.
- Demonstrates a genuine method advantage — iHS has 0 computable sites inside TREM2 in every EUR/EAS population; our scan catches the sweep anyway.
- Turns the "missing prior detection" from a weakness into the *point* of the case study.

## Evidence supporting the pan-OoA sweep (verified locally)

| Evidence | Observation |
|---|---|
| H12 peak position | chr6:41.15 Mb (~10 kb upstream of TREM2 gene body) |
| H12 peak value | 0.58–0.72 across all non-AFR (YRI 0.18) |
| H12 at non-sweep control (41.47 Mb) | ~0.01 in every population |
| iHS top 0.1–0.4% hits (our scan) | BEB, STU, ITU, GIH (TREML1); BEB, STU (TREM2) |
| iHS in EUR/EAS | 0 computable sites — gene is too short |
| Allele-frequency pattern | Most differentiated allele (chr6:41,166,068 G>A): AFR 0.95 → non-AFR 0.43 |
| IBS vs other-EUR differentiation at sweep | Max \|dAF\| = 0.06 (noise) |
| Akbari 2026 aDNA posterior | Flat (max 0.14 PASS at gene body) — consistent with ancient sweep |

## Method story

TREM2 is an unusually clean case for advocating TMRCA over classical haplotype scans:

- The gene is 4.7 kb — below iHS's minimum window for meaningful EHH decay in short-LD populations (EUR, EAS).
- Our own iHS scan confirms the power problem: 0 computable sites inside the gene in every EUR/EAS population.
- iHS picks up the surrounding TREM cluster via TREML1 in SAS (where LD is longer), so the signal IS there — just hidden from non-SAS classical scans.
- TMRCA doesn't depend on EHH decay; it catches the sweep across all non-AFR.

This is a publishable methods-paper point: *classical scans have a gene-size blind spot; TMRCA closes it.*

## The 41.47 Mb leftover signal

The `TREM2_IBS.json` max-dAF variant at chr6:41,485,209 (|dAF|=0.40 vs non-IBS) is genuinely differentiated but is NOT a sweep:
- r² to all common variants in TREM2/FOXP4 ≤ 0.03
- H12 at the position = 0.01 (baseline)
- Not colocalized with the actual swept block at 41.15 Mb

What it probably is: background selection boundary, ancient segregating polymorphism, or cryptic IBS structure. Not a sweep. Should be acknowledged in the manuscript as a separate differentiation feature, not presented as part of the TREM2 selection story.

## What changed from earlier drafts

- **FOXP4 angle**: withdrawn. FOXP4 ranks 20–46% in all populations; no selection signal anywhere. The earlier FOXP4 hypothesis was driven by the erroneous "most-differentiated variant" focal selection.
- **"IBS-specific"**: withdrawn. IBS is tied with FIN/GBR at ~0.3–0.4% rank.
- **TREM2 itself**: kept as the case study, but reframed as a pan-OoA sweep.
- **Biology**: the TREM2 microglia / AD / Nasu-Hakola biology story still anchors the Discussion — just without the "IBS respiratory immunity" twist that doesn't survive.

## CLUES on Betty — submitted

`sbatch analysis/relate_clues/scripts/04f_extract_clues2_trem2_ooa.sh`:
- **Job ID 5484941** (PENDING as of 2026-04-24)
- Focal variant chr6:41,166,068 (G>A), derived AF 0.429 in IBS
- Population: IBS (infrastructure already set up; signal is pan-OoA so any non-AFR works)
- Output: `analysis/relate_clues/clues/TREM2_OoA/trem2_ooa_result_inference.txt`
- Stuck prior job 5479965 left in queue (`DependencyNeverSatisfied`) — harmless PD, will be cleaned up separately.

Expected outcome (informed guess before running):
- Sweep age likely pre- or peri-OoA (>40 kya) given AFR/non-AFR AF split
- Selection coefficient modest (allele at ~0.43 in non-AFR, not fixed, so either s weak or sweep incomplete)
- If CLUES dates it <10 kya → surprising, would suggest ongoing selection; would warrant a separate discussion

## FOXP4 CLUES — not submitted

Explicit decision: FOXP4 has no signal in any population (rank 20-46%, no H12, no iHS). Running CLUES on FOXP4 would produce a defensible but uninformative null. Not a good use of Betty compute. If reviewers specifically ask for the null, we can run it then.
