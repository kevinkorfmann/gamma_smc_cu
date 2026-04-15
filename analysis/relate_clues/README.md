# Relate + CLUES2: Allele frequency trajectories at GRK2 and LCT

Temporal inference of selection at the GRK2 shared West Eurasian sweep
and the LCT lactase persistence sweep, using Relate (Speidel et al. 2019)
for genealogy inference and CLUES2 (Vaughn & Nielsen 2024) for allele
frequency trajectory estimation.

## Pipeline steps

| Step | Script | SLURM | Time | Depends on |
|---|---|---|---|---|
| 0 | `00_setup.sh` | genoa, 16G, 2h | ~30 min | — |
| 1 | `01_prepare_inputs.sh` | genoa, 64G, 2h | ~30 min | Step 0 |
| 2 | `02_run_relate.sh` | genoa, 192G, 48h | 6-24h/chr | Step 1 |
| 3 | `03_popsize.sh` | genoa, 128G, 8h | 1-2h/chr | Step 2 |
| 4 | `04_extract_sample.sh` | genoa, 64G, 4h | ~30 min | Step 3 |
| 5 | `05_run_clues2.sh` | genoa, 16G, 1h | ~5 min | Step 4 |

## Submission order

```bash
cd /vast/projects/smathi/cohort/kkor/tmrca.cu

# Step 0: setup (downloads)
JOB0=$(sbatch --parsable analysis/relate_clues/scripts/00_setup.sh)

# Step 1: prepare (depends on setup)
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 analysis/relate_clues/scripts/01_prepare_inputs.sh)

# Step 2: Relate inference — 2 array tasks (chr2 + chr11), depends on prepare
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 analysis/relate_clues/scripts/02_run_relate.sh)

# Step 3: population size — depends on Relate
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 analysis/relate_clues/scripts/03_popsize.sh)

# Step 4: extract + sample — depends on popsize
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 analysis/relate_clues/scripts/04_extract_sample.sh)

# Step 5: CLUES2 — depends on extract
sbatch --dependency=afterok:$JOB4 analysis/relate_clues/scripts/05_run_clues2.sh
```

## Loci

| Gene | Chr | Window | Focal SNP | Pop | Notes |
|---|---|---|---|---|---|
| GRK2 | 11 | 67.2-67.5 Mb | 67,407,126 | GIH | Shared W Eurasian sweep |
| LCT | 2 | 135.7-136.0 Mb | 135,851,076 | CEU | Positive control (rs4988235) |
| KCNQ1 | 11 | 2.4-2.7 Mb | — | GIH | Neutral control |

## References

- Speidel et al. 2019, Nat Genet 51:1321 (Relate)
- Vaughn & Nielsen 2024, Mol Biol Evol 41:msae156 (CLUES2)
- Stern et al. 2019, PLoS Genet 15:e1008384 (original CLUES)
