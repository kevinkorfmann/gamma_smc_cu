# Reproducing the manuscript end-to-end

This guide walks through every step needed to reproduce the paper
*"Pairwise coalescence-time inference localises a shared West Eurasian
sweep haplotype at chr11q13.2 (GRK2)"* from scratch, in order.

Every stage lists: **inputs**, **command**, **expected outputs**,
**expected runtime**, and **what the output is used for downstream**.

Compute profile assumed:
- Development / unit tests: single workstation with one NVIDIA GPU ≥ A40
- Full pipeline: SLURM cluster with ≥ 8 B200 GPU slices (we used UPenn's
  `betty` / `dgx-b200` + `b200-mig45`).

Approximate end-to-end wall clock: **~1 day of cluster compute**
(genome-wide scan dominates; everything else is minutes to hours).

---

## Stage 0. Environment

```bash
git clone https://github.com/kevinkorfmann/tmrca.cu
cd tmrca.cu
pixi install          # resolves conda env (~5 min on first run)
pixi run build        # CUDA + pybind11 build (~3 min)
pixi run pytest tests/statistical/  # 34 paper-stats unit tests, <1 s
```

Output: `python/gamma_smc_cu/_core.cpython-3.12-*.so`,
`python/gamma_smc_cu/libgamma_smc_cu_kernels.so`.

To verify the install:

```bash
pixi run python -c "import gamma_smc_cu; print(gamma_smc_cu.__file__)"
```

## Stage 1. Cross-species benchmark (accuracy + speed parity)

Produces Fig S1 / S2 (cross-species accuracy scatter + chromosome-22
speed comparison).

```bash
cd benchmarks/test_suite_stdpopsim/

# Run all 14 stdpopsim configs (8 species)
pixi run python run_one.py --config-idx 0
# ... or loop over all 14:
for i in $(seq 0 13); do pixi run python run_one.py --config-idx $i; done

# Aggregate + plot
pixi run python aggregate_and_plot.py
```

**Inputs:** none (stdpopsim generates simulations on the fly)
**Outputs:** `results/config_{0..13}.npz`, `figures/test_suite_summary.csv`,
`figures/accuracy_scatter.pdf`, `figures/speed_comparison.pdf`
**Runtime:** ~2 h for all 14 configs on a single A100; each config runs
both `gamma_smc_cu` and the reference `gamma_smc` binary (installed
from the upstream gamma_smc GitHub).
**Downstream use:** Fig 1 (fig:tool), Fig S1, Fig S2; the **132× median
speedup** and **r = 0.876 vs 0.874** numbers in Results §2.1.

## Stage 2. 1000 Genomes data preparation

Download the 30× high-coverage release (Byrska-Bishop 2022) and cache
to NPZ for fast access. This must be done on betty (GRCh38 VCFs are
several TB).

```bash
# On betty
cd /vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/

# 1. Download VCFs from IGSR / NYGC
#    ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/
#    (see data/downloads.sh; not under version control)

# 2. Parse + bitpack to NPZ cache
sbatch slurm_build.sh
```

**Inputs:** per-chromosome phased VCFs from IGSR
**Outputs:** `cache/parsed/chr{1..22}.npz` with bitpacked `G`, `positions`,
`sample_ids`
**Runtime:** ~30 min total across 22 chromosomes on `genoa-std-mem`
(CPU, not GPU)
**Downstream:** every analysis below reads these NPZ caches.

## Stage 3. Genome-wide TMRCA scan

22 autosomes × 26 populations = 572 (chr, pop) SLURM tasks; each runs
the full blockwise decoder across the chromosome for all within-pop
pairs, aggregates to gene level, and writes per-gene CSV + per-pair-
histogram NPZ.

```bash
# On betty
cd /vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/
sbatch --array=0-571 slurm_infer_all.sh      # b200-mig45, ~1-10 h wall-clock
# (or slurm_infer_dgx.sh for full B200 GPUs with --gres=gpu:B200:1)
```

**Inputs:** `cache/parsed/chr{N}.npz`, `data/samples.txt` (pop-map)
**Outputs:** `results/chr{N}/{POP}.csv` + `.npz` (per-gene geom-mean
TMRCA, log-sum accumulator, 50-bin histogram) for all 22×26=572 cells
**Runtime:** per-chromosome wall time 0.5 h (chr22) to 9.5 h (chr5/chr6);
end-to-end wall clock a few hours when chromosomes run concurrently.
**Downstream:** Stage 4 aggregation.

## Stage 4. Aggregation, within-population ranking, SD masking

```bash
sbatch slurm_postprocess.sh         # genoa-std-mem, ~20 min
# calls: python aggregate.py
#        python postprocess.py
```

**Inputs:** all 572 `results/chr{N}/{POP}.csv/.npz`
**Outputs:**
- `postprocess/genome_wide_stats.csv` (19,119 genes × 26 pops with
  gene-level geom-mean TMRCA, per-pop rank percentile, min-pop)
- `postprocess/genome_wide_candidates_filtered.csv` (after SD masking
  at 50% UCSC genomicSuperDups overlap threshold)

**Runtime:** ~20 min.
**Downstream:** candidate cascade, ranking claims, FDR.

## Stage 5. Candidate selection cascade + FDR

```bash
# All locally; reads postprocess CSVs
cd private/manuscript/v4.1/verify/
pixi run python 19_candidate_cascade.py    # cascade counts 19119 -> 165
pixi run python 21_fdr.py                   # q_hier, stage-2 ranks
pixi run python 13_replication_correlation.py  # Galwey n_eff per continent
```

**Outputs:**
- `private/manuscript/v4.1/tables/fdr_qvalues.csv` (17,823 genes with
  per-gene p, BH-adjusted q, hierarchical q_hier)
- `private/manuscript/v4.1/tables/stage5_loci_with_stats.csv`
  (the 165-locus stage-5 set with orthogonal-statistic percentiles)
- Spearman ρ matrices per continent (printed), n_eff values (paper
  numbers: 1.95/1.71/1.78/1.64/1.98, SAS+EUR combined 2.64)

**Runtime:** a few minutes.
**Downstream:** Table S2, Fig:manhattan, all FDR/q_hier claims.

## Stage 6. Orthogonal validation pipelines

All under `analysis/orthogonal_v41/`; each is a SLURM array.

### 6a. Per-variant evidence (ΔAF, Hudson FST, depleted:enriched)

```bash
sbatch analysis/orthogonal_v41/scripts/slurm_variant_evidence.sh
# fills per-gene JSON summaries
```

**Runtime:** ~30 min
**Outputs:** `analysis/orthogonal_v41/results/variant_evidence/{GENE}_{POP}.json`
(max FST, depleted:enriched ratio, most-differentiated variant position
and frequencies)

### 6b. selscan iHS/nSL

```bash
sbatch analysis/orthogonal_v41/scripts/slurm_selscan_array.sh
sbatch analysis/orthogonal_v41/scripts/slurm_aggregate_selscan.sh
```

**Runtime:** ~4 h
**Outputs:** per-chromosome `*.ihs.out.norm`, `*.nsl.out.norm`;
per-gene fraction-extreme aggregated CSV

### 6c. Garud's H_12 genome-wide

```bash
sbatch analysis/orthogonal_v41/scripts/slurm_h12_array.sh
```

**Runtime:** ~2 h

### 6d. ASMC comparison at focal loci

```bash
sbatch analysis/orthogonal_v41/scripts/slurm_asmc_setup.sh
sbatch analysis/orthogonal_v41/scripts/slurm_asmc_array.sh
```

**Runtime:** ~8 h
**Outputs:** ASMC per-pair mean TMRCA for the 5 main-text + 5 positive
control + 5 neutral = 15 focal loci

### 6e. Three-method TMRCA concordance (`gamma_smc_cu` + `cxt` + ASMC)

```bash
sbatch analysis/orthogonal_v41/scripts/slurm_three_method.sh
```

**Outputs:** `analysis/orthogonal_v41/three_method/{GENE}_{POP}_novel.npz`
containing the three-method per-pair mean TMRCA arrays. Fed to
Fig S4 and to verify/17_three_method_concordance.py.

## Stage 7. Relate + CLUES2

Standalone pipeline under `analysis/relate_clues/scripts/`; each
numbered script is a SLURM submission.

```bash
cd /vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/relate_clues/
./scripts/00_setup.sh                    # install Relate + CLUES2
./scripts/01_prepare_inputs.sh           # download GRCh38 ancestral FASTA + recomb map
./scripts/02_run_relate.sh               # genome-wide Relate ARG
./scripts/03_popsize.sh                  # estimate Ne(t)
./scripts/04d_extract_grk2.sh            # extract GRK2 window
./scripts/05d_clues_grk2.sh              # CLUES2 on chr11:67,407,126
./scripts/06_neutral_controls.sh         # C11orf65 neutral control
```

**Runtime:** ~2 days total (Relate is the bottleneck; runs over multiple
weeks' worth of cluster time but in practice 2-3 days on `b200-mig45`).

**Outputs:**
- `results/clues/GRK2/grk2_result_inference.txt` — ŝ=0.018, 95% CI
  [0.014, 0.023], −log10 p=21 (Fig 3, Discussion)
- `results/clues/neutrals/c11orf65/...` — ŝ=0.007, −log10 p=4.9
- Parallel CLUES2 runs for LCT / CCDC92 / CLEC6A / SLC6A15 / BPIFA2

## Stage 8. Akbari 2026 cross-check

```bash
cd /vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/akbari_479_tmrca/

# 1. Download Akbari supplementary from Harvard Dataverse
#    doi:10.7910/DVN/7RVV9N -> akbari_lead_variants.tsv (GRCh37)

# 2. Lift GRCh37 -> GRCh38 (the manuscript's 1000G cache is GRCh38)
pixi run python lift_akbari_positions.py
# Produces akbari_lead_variants_grch38.tsv; 474/474 variants lifted cleanly.

# 3. Compute pairwise TMRCA at Akbari peaks ±25 kb across all 26 1000G pops
sbatch --array=0-571 slurm_infer_akbari_pop.sh
# 22 chr x 26 pop = 572 tasks, ~2-3 h on b200-mig45

# 4. Build + plot the heatmap (Fig 2 in main.tex: fig:akbari_heatmap)
pixi run python make_heatmap.py
```

**Outputs:** `results/chr{N}/{POP}.csv` (Akbari-lead TMRCA), 
`heatmap_matrix_kya.csv`, `figure_heatmap.pdf`.
**Downstream:** Fig:akbari_heatmap + Akbari confirmation narrative.

## Stage 9. Verification suite (mandatory before any figure/table update)

```bash
cd private/manuscript/v4.1/verify/
for s in 0*_*.py 1*_*.py 2*_*.py; do
    pixi run python $s 2>&1 | tail -5
done
```

**Inputs:** postprocess outputs + orthogonal-validation outputs
**Outputs:** PASS/FAIL per every numeric claim in the paper across 26
scripts (sample composition, cascade counts, FDR, three-method
concordance, haplotype sharing, etc.)
**Runtime:** ~3 min.
**Requirement:** **all 26 scripts must PASS** before regenerating figures
or building the manuscript.

## Stage 10. Figure generation

All figures in the paper are regenerated from scratch by scripts in
`private/manuscript/v4.1/figures/gen_fig_*.py`. Each is independent and
reads only the postprocess + verify + orthogonal outputs.

```bash
cd private/manuscript/v4.1/figures/

# Fig 1 (tool parity) — main.tex fig:tool
pixi run python gen_fig_accuracy_si.py
# Fig 2 (pipeline schematic) — main.tex fig:pipeline
pixi run python gen_fig_pipeline.py
# Fig 3 (Manhattan) — main.tex fig:manhattan
pixi run python gen_fig_manhattan.py
pixi run python gen_fig_manhattan_sweeps.py
# Fig 4 (Akbari heatmap) — main.tex fig:akbari_heatmap
# (produced by analysis/akbari_479_tmrca/make_heatmap.py, copied into figures/)
# Fig 5 (GRK2 deep dive) — main.tex fig:grk2
pixi run python gen_fig_grk2_dive.py
# Fig 6 (landscape: LCT + four exemplars) — main.tex fig:landscape
pixi run python gen_fig_landscape.py

# SI figures
pixi run python gen_fig_all_genes.py
pixi run python gen_fig_grk2_cluster_tmrca.py
pixi run python gen_fig_known_sweeps.py
pixi run python gen_fig_lct.py
pixi run python gen_fig_sd_masking.py
pixi run python gen_fig_si_missed_sweeps.py
pixi run python gen_fig_three_method_compact.py
pixi run python gen_fig_novel_clues.py
```

**Runtime:** ~10 min total.
**Outputs:** `figures/fig_*.pdf`, `figures/fig_*.png` (used by main.tex
and si.tex).

## Stage 11. Table generation

```bash
cd private/manuscript/v4.1/tables/
pixi run python gen_tables.py
```

**Outputs:** `table{1,2,3}_*.tex` (known sweeps, novel findings,
orthogonal evidence summary) + `table_s{2,3}_*.tex` (the 165-locus
community resource, gene catalog).

## Stage 12. Compile the manuscript

```bash
cd private/manuscript/v4.1/
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# SI
pdflatex -interaction=nonstopmode si.tex
bibtex si
pdflatex -interaction=nonstopmode si.tex
pdflatex -interaction=nonstopmode si.tex
```

**Outputs:** `main.pdf` (29 pages), `si.pdf` (tables + figures).
**Runtime:** ~10 s per pass.

## Stage 13. (Optional) Unit tests for continuous integration

```bash
pixi run pytest tests/ -v
```

Covers:
- CUDA kernel correctness (`tests/unit/`)
- HMM forward/backward (`tests/unit/test_hmm*.py`, `test_emissions.py`)
- Blockwise stitching (`tests/integration/test_tier1_pipeline.py`)
- Multi-GPU (`tests/integration/test_multigpu.py`)
- Paper reference numbers (`tests/statistical/test_paper_formulas.py`,
  `test_paper_reference.py`) — the 34 tests that reproduce Galwey n_eff,
  q_hier rankings, cascade arithmetic, binomial sensitivity, GRK2
  gene-body monomorphism, rs11604662 build shift, etc.

**Runtime:** `tests/statistical/` alone runs in <1 s; full suite including
CUDA kernels is ~3 min on an A100.

---

## Legacy / audit artifacts

- `private/manuscript/v4.1/audit/BUG_HUNT_FINDINGS_2026_04_23.md` —
  record of the Akbari GRCh37/GRCh38 coordinate bug (fixed) and other
  bug-hunt findings.
- `private/manuscript/v4.1/audit/FIG5_AKBARI_ANNOTATION_2026_04_23.md` —
  details of the Akbari tick-bar addition to Fig:grk2 and Fig:landscape.
- `legacy/` — archived pre-v4.1 material (docs website, demo notebooks,
  pre-Akbari-liftover code).

## Troubleshooting

**"No CUDA device available"** — `nvidia-smi` to check; set
`CUDA_VISIBLE_DEVICES=0` explicitly for single-GPU runs.

**"auto_estimate_theta produced inflated TMRCA"** — you're running on a
small slice (<1 Mb). The Akbari pipeline's `infer_akbari_windows.py`
pre-computes chromosome-wide theta once and freezes it; copy that
pattern for any slice-based analysis. See
`analysis/akbari_479_tmrca/STRATEGY.md` for details.

**"Pipeline mentions `tmrca_cu` import"** — historical: the Python
package was renamed to `gamma_smc_cu` on 2026-04-23. Re-`pixi install`
should pick up the new package; old scripts in `legacy/` may still
reference `tmrca_cu`.

**"Akbari coordinates don't match GRK2 gene body on GRCh38"** — make sure
you've run `lift_akbari_positions.py` before `infer_akbari_windows.py`;
Akbari publishes positions on GRCh37.

## Runtime summary

| Stage | What | Wall clock | Cluster? |
|---|---|---|---|
| 0 | Env + build | 10 min | local |
| 1 | stdpopsim parity benchmark | 2 h | local A100 |
| 2 | Download + cache 1000G | 30 min | CPU |
| 3 | Genome-wide scan | ~3 h | 572-task B200 array |
| 4 | Aggregation + ranking | 20 min | CPU |
| 5 | Cascade + FDR | 5 min | local |
| 6 | Orthogonal validation | ~8 h (overlappable) | cluster |
| 7 | Relate + CLUES2 | ~2 d | cluster |
| 8 | Akbari cross-check | 3 h | B200 array |
| 9 | Verify suite | 3 min | local |
| 10 | Figures | 10 min | local |
| 11 | Tables | 1 min | local |
| 12 | Compile manuscript | 1 min | local |

**Grand total:** ~1 day if everything runs in parallel, ~4-5 days if
strictly serial (Relate stage is the long pole).
