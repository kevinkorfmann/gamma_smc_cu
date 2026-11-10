# Detection and localisation benchmark

Read `PROTOCOL.md` for the fixed experimental design and limitations. This
experiment compares the CPU Gamma-SMC reference statistic with fresh iHS and
XP-EHH on shared simulated inputs; it is separate from GPU runtime benchmarks
and empirical candidate-profile fitting.

`freeze.py --out <new-directory>` creates the 1,000-replicate manifest. Copy
all `.py`, `.sh`, and `.md` files here into its `source/` directory before
execution. Each production task validates these hashes before running.
`production_betty.sh` runs simulation and scoring for one Slurm array index;
`analyze_betty.sh` must depend on successful completion of the entire array.
The Betty recipes document the original environment; edit a fresh copy and
freeze a new manifest for other environments.

The five technical pilots and `test_metrics.py` validate genotype sharing,
XP-EHH direction, native posterior decoding, coordinates, missing scores,
strict tied thresholds, interval summaries, and tie-aware localisation.
Pilots are excluded from normalisation, calibration, and evaluation.

Outputs retain independent replicate IDs, input/software hashes, accepted
allele-frequency trajectories, site counts, missingness, per-replicate scores,
held-out neutral false-positive rates, power, and localisation errors. The
production protocol is immutable once launched. Any analysis repair must
preserve the original source and document its effect.

## Target-aligned sweep signatures

`postprocess/sweep_profiles.py` extracts true and inferred geometric-mean TMRCA
on the same retained sites and focal pairs, with maximum-score iHS and XP-EHH,
for every evaluation replicate. Run extraction through `profiles_betty.sh` on
Slurm (30 array tasks of 20 replicates, then one dependent aggregation job).
The aggregate checks 150 direct all-pair tree calculations and original target
score availability. `postprocess/plot_sweep_profiles.py` renders the four-history
aggregate-profile display, retained as a superseded layout. The active SI figure is
rendered by `postprocess/plot_combined_sweeps.py --data <source-data-package> --out <figures>`.
It combines the first unsmoothed replicate per history (extracted on Slurm by
`raw_examples_betty.sh`) with all-replicate detection, availability and localisation.
This is a post-evaluation visualisation and does not modify the frozen experiment.
