# Reproducible same-panel runtime comparison

`run_matched_panel.py` prepares one full population marker panel, then measures
GPU Gamma-SMC, upstream CPU Gamma-SMC, and ASMC on exactly the same distinct
haplotype pairs. Every result records the panel-manifest hash and pair-file hash.
Changing pair count never changes markers or background haplotypes. This runner
supersedes the incomparable pair subsets, marker filters, and repeated-pair ASMC
proxy in the historical `run_final_benchmark.py`.

Use an immutable source snapshot with its Git checkout or verified
`SOURCE_MANIFEST.json`. The latter must contain a full `git_commit` and a `files`
mapping from relative paths to SHA256 (or objects containing `sha256`). Python
requires NumPy, SciPy and stdpopsim for shared provenance helpers; preparation
also needs `bgzip` and `tabix`. Method runs additionally need the built calibrated
GPU package, upstream CPU binary, or installed `asmc-asmc`, respectively. All
output directories must be new. Schedule Betty work through Slurm; select the
allocated GPU through the environment, not by modifying the runner.

Prepare the full chr22 YRI panel once, using the **documented coordinate base of
the input cache** (required flag; no guessed offset). Copy the complete prepared
directory unchanged to the other host if needed:

```sh
python benchmarks/pairwise_scaling/run_matched_panel.py prepare \
  --cache-dir /data/fresh_cache \
  --samples-file /data/samples.txt --positions-base 1 \
  --population YRI --chromosome 22 --output-dir /results/runtime/panel
```

The new raw-VCF cache uses `--positions-base 1` and memory-mapped arrays under
`parsed/chr22/`. A legacy NPZ may be supplied with `--parsed-npz` only when its
positions are strictly increasing and duplicate-free; historical caches with
duplicate positions are rejected rather than reused or silently deduplicated.

The single preparation filter requires complete phased binary genotypes and
both alleles in the full population. It retains arrays, source-column indices,
VCF, Oxford haplotypes, sample names and a constant-rate map. GPU/CPU positions
are zero-based; VCF and ASMC positions equal those coordinates plus one. Rates
default to mutation `1.25e-8` and recombination `1e-8` per generation/base, with
within-diploid theta estimated once over `last_zero_based_position + 1`. GPU
and CPU use that same theta and scaled recombination rate explicitly. ASMC
retains the model in its separately hashed decoding-quantities file; comparisons
here concern runtime, not demographic-model equivalence or accuracy.

Measure the same pair set separately for each method on its intended node:

```sh
python benchmarks/pairwise_scaling/run_matched_panel.py run \
  --panel-dir /results/runtime/panel --method gpu --pair-set within \
  --flow-field /software/python/gamma_smc_cu/default_flow_field.txt \
  --output-dir /results/runtime/gpu_within
python benchmarks/pairwise_scaling/run_matched_panel.py run \
  --panel-dir /results/runtime/panel --method cpu --pair-set within \
  --gamma-smc-bin /software/gamma_smc/bin/gamma_smc \
  --flow-field /software/python/gamma_smc_cu/default_flow_field.txt \
  --output-dir /results/runtime/cpu_within
python benchmarks/pairwise_scaling/run_matched_panel.py run \
  --panel-dir /results/runtime/panel --method asmc --pair-set within \
  --decoding-quantities /data/validated.decodingQuantities.gz \
  --output-dir /results/runtime/asmc_within
```

All runs use one full-workload warmup followed by three repetitions. ASMC passes
**distinct pairs together in batches** (default 32), not repeated decodes of a
single pair. The CPU's `-w` mode permits within-diploid pairs while retaining the
full cohort. Its disjoint `-S`/`-T` mode permits cross-group pairs, again retaining
the full cohort in the union. The precise upstream pair order is retained and
validated against CPU metadata; ASMC returned identities are validated as well.

For 356 haplotypes / 178 diploid samples:

| Pair set | Distinct pairs | Meaning |
| --- | ---: | --- |
| `within` | 178 | one pair per diploid |
| `cross:1` | 708 | first diploid versus the remaining 177 |
| `cross:5` | 3,460 | first 5 diploids versus remaining 173 |
| `cross:15` | 9,780 | first 15 diploids versus remaining 163 |
| `cross:89` | 31,684 | first 89 diploids versus remaining 89 |
| `all` | 63,190 | every unordered haplotype pair |

Every result explicitly distinguishes a measured subset from measured all-pairs
coverage. No extrapolation is produced. Incomplete method coverage cannot be
represented as a measured comparison at a different pair count. Full all-pairs
runs can create substantial output; select those explicitly after subset smoke
checks. One measured set of predictions is retained by default, with the exact
output hash/size from every repetition. `--retain-all-predictions` keeps all
repetitions and the warmup. Native outputs are always generated and timed before
optional removal, so this option does not change the timed workload.

GPU inference defaults to full-sequence decoding in outer pair batches of 256,
bounding both host outputs and device scratch. This repeated API-call strategy
is recorded and is not interchangeable with an all-pairs single API call. To
measure a production blockwise configuration, select
`--gpu-mode blockwise --core-sites 65536 --flank-sites 8192`; label it accordingly
and pair the timing with the separate boundary-sensitivity assessment.

Native API boundaries differ. The result stores each of these explicitly:

- `native_total`: prepared native-format input loading/initialization, decoding,
  host results and native prediction serialization. Disk formats differ:
  compressed alpha/beta for CPU, uncompressed posterior means for GPU and ASMC.
- `decode`: CPU upstream emissions/forward/backward timers (millisecond printed
  precision); GPU public API wall time including preprocessing, allocation and
  device/host transfer; ASMC `decode_pairs` plus copying results. These are **not
  identical kernel-only scopes**.
- `output`: upstream compressed output timing or NumPy prediction writes. The
  panel's VCF/Oxford conversion costs are separately retained in `panel.json`.

The raw repetitions, quartiles, warmup, stdout/stderr, actual CPU/GPU models,
package and source versions, binary/data hashes, CPU pair metadata and first
measured predictions are retained. A strict workload check fails if any method
returns a different marker count, positions, or pair identities. Results must be
reported with these timing scopes and the actual recorded hardware.

## Optimized dense CUDA export

Use the direct exporter for full-context mean files with overlapping inference,
transfers and concurrent batch writers:

```sh
python benchmarks/pairwise_scaling/run_matched_panel.py run \
  --panel-dir /results/runtime/panel --method gpu --pair-set all \
  --gpu-mode export --gpu-ids 0 --gpu-pair-batch 8192 \
  --checkpoint-sites 4096 --export-workers-per-gpu 8 --export-buffers 3 \
  --core-sites 6000000 --flank-sites 0 --repeats 3 \
  --flow-field python/gamma_smc_cu/default_flow_field.txt \
  --output-dir /results/runtime/new_cuda_export
```

The complete core must cover the panel, with no flanks. Tune the pair batch for
the requested pair count; the exporter has bounded staging per worker, and
multiple workers consume additional host/GPU memory. See [the batching guide](../../docs/batching.md).
`native_total` includes all prediction writes. GPU stage timers overlap, so no
separate additive decode/output split is reported. The default uses ordinary
buffered file writes. `--durable-output` adds file and directory sync calls and
must be reported as a distinct timing scope.

The [26 September dense-export comparison](dense_export_20260926/README.md)
contains the updated figure, all plotted repetitions, input/pair hashes,
selected batch sizes and reproduction commands. It reruns CUDA and explicitly
reuses the verified unchanged CPU/ASMC baselines on the same host. Its figure
script rejects extrapolation, mismatched panels/pairs/hosts, missing coverage
and incomplete all-pairs endpoints.

CPU-only regression checks:

```sh
python -m pytest -q benchmarks/pairwise_scaling/tests \
  benchmarks/test_suite_stdpopsim/tests
```

## Fresh decoding quantities

The original external `CEU_50` filename does not establish its model or mutation
rate. Current upstream documentation states that older CEU decoding quantities
used diploid sizes where haploid sizes were required. The corrected built-in
population histories in PrepareDecoding are haploid and use a mutation rate of
`1.65e-8`. For a run at `1.25e-8`, both demographic times and haploid population
sizes must be multiplied by `1.65/1.25`, preserving mutation-scaled history.

Install the explicitly pinned, current packages in the campaign environment:

```sh
python -m pip install 'asmc-asmc==1.4.0' 'asmc-preparedecoding==2.2.5'
```

Generate new quantities with the corrected CEU reference demography, retaining
that reference choice rather than silently changing the biological model:

```sh
python benchmarks/pairwise_scaling/prepare_asmc_decoding.py \
  --demography CEU --samples 300 --quantiles 50 --mu 1.25e-8 \
  --output-dir /new/asmc_data/CEU_csfs300
```

Use `/new/asmc_data/CEU_csfs300/decoding.decodingQuantities.gz` explicitly in the
runtime command. The generator exports and hashes the original and rescaled
haploid demography, DQ file, CSFS, discretization and intervals. Its manifest
records requested and actual state counts, CSFS sample count, package version,
mutation rates, rescaling, timing and source documentation. `--demography YRI`
or another supported population makes a documented alternative model. A
separate `--samples 50` setup can preserve a regional comparison's smaller CSFS
configuration; this number is **haploid CSFS samples**, not a requirement to
observe 50 diploid individuals.

The built-in UKBB frequency information is retained in the DQ provenance; runs
must request ASMC `sequence` mode for unascertained sequence emissions. These
choices should be stated in runtime and biological comparisons. Generating DQ
is CPU work; use a CPU Slurm allocation on Betty.

Primary documentation: [ASMC corrected model notice](https://github.com/PalamaraLab/ASMC/blob/main/docs/asmc.md#decoding-quantities-decodingquantitiesgz),
[haploid demography and mutation-rate calibration](https://github.com/PalamaraLab/ASMC_data#demographies),
and [PrepareDecoding API](https://github.com/PalamaraLab/PrepareDecoding/blob/main/docs/api.md).
