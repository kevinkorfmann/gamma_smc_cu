# Corrected benchmark protocol (schema 2)

Run from a frozen software checkout after building the CUDA extension. Requires
`numpy`, `scipy`, `msprime`, `stdpopsim`, `tskit`, the upstream `gamma_smc` binary,
and `bgzip`, `tabix`, `zstd`. Point `GAMMA_SMC_BIN` at the recorded upstream binary,
or pass `--gamma-smc-bin`. `--flow-field` can select the matching flow table.
The runner records source hashes, binary hashes, package versions, hardware,
scheduler allocation, the invocation and effective rates.

CPU-only scientific regressions:

```sh
python -m pytest -q benchmarks/test_suite_stdpopsim/tests
```

The retained comparison set is **0–8 and 10–14**, not 0–13. Inspect it without a
GPU or upstream binary:

```sh
python benchmarks/test_suite_stdpopsim/run_one.py --list-configs
```

Run one full configuration (20 haplotypes, 5 Mb, seed 42 unless overridden):

```sh
python benchmarks/test_suite_stdpopsim/run_one.py --config-idx 0 \
  --results-dir /absolute/path/new-run \
  --gpu-repeats 3 --cpu-repeats 3
```

`--results-dir` creates `config_000` under the run root; alternatively
`--output-dir` names the exact per-configuration directory. That directory must
not already exist, even after failure. Use a new attempt directory to retry.
This prevents replacement or mixing of historical evidence. Existing original
results are not read or modified. `--sequence-length 100000` is a smoke-test
override and is recorded in the configuration; keep these outputs separate.

On Betty submit only through Slurm, from the immutable checkout root:

```sh
sbatch --array=0-8,10-14 benchmarks/test_suite_stdpopsim/slurm_array.sh \
  --results-dir /absolute/path/new-betty-run
```

Override partition/GPU/CPU/memory limits to match current cluster policy.
`BENCH_REPO_ROOT`, `BENCH_ENV_PREFIX` and `BENCH_PYTHON` allow separate code and
environment locations. On a non-Slurm host run the Python command directly,
selecting only the GPU assigned to this task with `CUDA_VISIBLE_DEVICES`.

## Scientific contract

- stdpopsim performs exactly one mutation layer, at the declared contig mutation
  rate; the configured recombination rate is explicitly supplied too.
- VCF POS = original tree-sequence position + 1, including position zero.
- Only strict biallelic ACGT SNPs with both alleles observed are retained.
  Genotypes and positions are checked against the source tree sequence.
- GPU uses the calibrated public inference API. CPU means are converted from
  coalescent units using the CPU metadata's theta/(2 * physical mutation rate).
  CPU and GPU scaled rates must agree within metadata rounding tolerance.
- Pair identities come from the upstream metadata. Predictions and truth match
  exact retained coordinates; no positional interpolation is permitted.
- The input tree sequence, normalized VCF, genotype/pair/site mapping, truth,
  both implementations' physical-generation predictions, CPU compressed raw
  posteriors and metadata, and per-pair metrics remain available for replay.
  Correlation, natural-log RMSE and direct method agreement are all reported.

## Timing and blockwise sensitivity

GPU times include the public API's preprocessing, allocations, decoding and
host transfers, following one full-workload warmup. CPU times include the
subprocess, input parsing, initialization, decoding and output serialization,
following one full-workload warmup. Median, all individual repetitions and
reference input compression/index time are saved. These are API/subprocess
costs, not identically scoped kernel-only times; report the distinction when
interpreting speedup. Simulation and artifact archival are outside those times.

The default sensitivity stage compares full-sequence GPU predictions against
cores of 16,384 and 65,536 sites, each with flanks 2,048, 8,192 and 32,768 sites,
on 20 deterministic pairs spread across the pair list. All blockwise predictions
and blocks are saved. It reports absolute log differences overall, within 64
sites of boundaries and over 512 sites from boundaries. Cases that do not
actually split the sequence are marked `not_multiblock`, not validated.

Each measured setting passes the diagnostic thresholds only when the 99th
percentile absolute log difference is <=0.01 and the maximum <=0.1. These are
explicit numerical diagnostics, not a claim of biological gene-rank stability.
Failures are printed and saved as `blockwise_tolerance_failed: true`. Add
`--require-blockwise-tolerance` to exit 3 after retaining all results when any
setting fails. Never interpret `status: ok` for benchmark completion as a pass
for every flank size. Choose production settings after inspecting the measured
boundary errors and separately validate empirical gene-rank stability.

For shorter/longer grids, use `--blockwise-core-sites 16384,65536`,
`--blockwise-flanks 2048,8192,32768`, and `--blockwise-pairs 20`.
An empty `--blockwise-flanks ''` skips the optional sensitivity stage explicitly.

After every retained ID completes, aggregate into a new directory:

```sh
python benchmarks/test_suite_stdpopsim/aggregate_and_plot.py \
  --results-dir /absolute/path/new-run --output-dir /absolute/path/new-figures
```

Aggregation requires the complete 14-ID set unless `--allow-incomplete` is
explicit. It rejects duplicate IDs and mixing historical/corrected schemas.
Keep different hosts and simulation seeds in separate directories.
