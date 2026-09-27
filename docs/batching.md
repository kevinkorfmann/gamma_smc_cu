# Stream full-context inference across pair batches

`iter_infer_batches` prepares the marker panel and flow field once per selected
GPU. It keeps the whole chromosome context for every pair, and yields bounded
posterior batches in the requested pair order. While the caller writes a batch,
the next batch can run on the GPU. The iterator preserves the calibrated time
units and optional posterior/CI outputs of `infer_blockwise`.

```python
from contextlib import closing
import numpy as np
from gamma_smc_cu import iter_infer_batches

with closing(iter_infer_batches(
    G, positions, pairs=pairs, pair_batch_size=1024, gpu_ids=[0, 1],
    mu=1.25e-8, rho=1e-8, auto_estimate_theta=True,
)) as batches:
    for batch in batches:
        np.save(f"means_{batch['pair_offset']:08d}.npy", batch["mean"])
```

Choose devices explicitly: omit `gpu_ids` to use only the current GPU. Device
IDs are relative to `CUDA_VISIBLE_DEVICES`. Discard consumed batches to keep
memory bounded. Use `closing` when a loop may stop early. Without checkpointing, native inference caps
GPU pair chunks to available memory; the host result size is controlled by
`pair_batch_size`. Larger batches are not necessarily faster: benchmark the
actual panel and GPU.

The matched-panel benchmark supports the same path:

```sh
python benchmarks/pairwise_scaling/run_matched_panel.py run \
  --panel-dir /path/to/panel --method gpu --pair-set all \
  --gpu-mode streaming --gpu-ids 0 1 --gpu-pair-batch 8192 \
  --checkpoint-sites 4096 --output-workers 4 \
  --core-sites 6000000 --flank-sites 0 --repeats 3 \
  --flow-field python/gamma_smc_cu/default_flow_field.txt \
  --output-dir /path/to/new-results
```

This large-batch example needs substantial host RAM (see below). The core
size must cover all panel sites, with zero flanks. Report the measured
end-to-end wall time: it includes loading, context setup, inference, transfers,
validation, and NPY writes. The output writer overlaps inference, so its time
must not be added to GPU time. The benchmark records no isolated decode timer
for this mode. Compare numerical outputs and keep device count explicit when
comparing performance.

## Exact checkpoint/replay

Pass `checkpoint_sites=4096` to the iterator (or `--checkpoint-sites 4096` to
its benchmark) to save sparse forward checkpoints, replay one tile at a time,
and carry backward state between tiles. Tiles never reset the chromosome's
recurrence. This bounds GPU state memory by approximately
`pairs * (sites / tile_sites + tile_sites)` instead of `pairs * sites`, at the
cost of an additional forward sweep. The host result remains the full matrix.

This permits larger pair batches, which can improve GPU utilization. The
checkpoint path checks free device memory and raises an error if the requested
batch does not fit; reduce `pair_batch_size` or the tile size in that case.
Host memory must also accommodate the yielded and prefetched full matrices.
On the tested 313,277-site panel, an 8,192-pair float32 mean batch alone is
10.3 GB. Do not retain every batch unless the host has enough memory.

Checkpoint decoding applies generation calibration and checks every posterior
mean for finite, positive values on the GPU. This avoids extra full-matrix CPU
passes; a failed check raises an error. The benchmark can use bounded concurrent
NPY writers with `--output-workers 2` (GPU streaming mode only). Each outstanding
writer retains one host matrix, in addition to the decoder's prefetched batches.
Writer timings overlap and cannot be added to obtain elapsed wall time.

## Direct dense NPY export

`export_dense` writes every calibrated per-site, per-pair float32 posterior
mean without creating a chromosome-sized host result. It overlaps exact
checkpoint inference, GPU-to-host transfers into pinned buffers, and writes
to standard NumPy files. Pair order and all retained sites are preserved.

```python
from gamma_smc_cu import export_dense

manifest = export_dense(
    G, positions, pairs=pairs, output_dir="new-dense-output",
    pair_batch_size=8192, tile_sites=4096, staging_buffers=3, workers_per_gpu=2,
    gpu_ids=[0], mu=mu, rho=rho, physical_mu=physical_mu,
)
```

The output directory must be new. It contains `positions.npy`, `pairs.npy`,
and `mean_batchNNNNN.npy` arrays shaped `(retained sites, batch pairs)`.
`manifest.json` records calibration, shapes, pair offsets and timings, and is
published only after all files finish successfully. A failed run raises an
error and has no complete manifest; finished batches may remain. This API
exports posterior means only and requires POSIX file support.

Pinned staging memory per active worker is at most
`4 * tile_sites * pair_batch_size * staging_buffers` bytes. GPU workspace is
checked against free device memory. Larger pair batches can therefore fit
without allocating the entire corresponding host matrix. Tune batch and tile
sizes against the actual panel, device and filesystem. Multiple workers per
GPU write separate batch files concurrently; each has its own context and
staging buffers. GPU-local CPU placement
is automatic where available, respects existing CPU allocations, and leaves
the calling thread's affinity unchanged; disable with `numa_local=False`.

By default, writes have the same persistence scope as ordinary `np.save`:
the OS may still cache dirty pages after return. Set `durable=True` to include
file and directory `fsync` before return. Stage timers overlap; measure the
outer call for end-to-end elapsed time. The matched-panel runner supports
`--gpu-mode export --export-buffers 3 --export-workers-per-gpu 2`, with `--gpu-pair-batch`, `--gpu-ids`
and `--checkpoint-sites` as above. Use `--durable-output` for explicit sync.
`--output-workers` controls the separate streaming path, not this exporter.

## Gene and interval summaries without a dense intermediate

When downstream analysis needs each pair's average time and average log time
within intervals, `RegionMomentContext` performs those reductions on the GPU.
It computes the same full-chromosome posterior, applies the same float32
physical-time calibration, and accumulates the interval moments in float64.

```python
from gamma_smc_cu import RegionMomentContext

context = RegionMomentContext(G, positions, mu=mu, rho=rho,
                              physical_mu=physical_mu)
# Bounds index the retained, segregating positions in context.positions.
# Translate genomic coordinates with searchsorted, using their coordinate system.
regions = [(start_index, stop_index)]  # half-open, nonempty site intervals
result = context.run(pairs, regions, tile_sites=4096)
linear_per_pair = result['region_mean']       # (regions, pairs)
logarithmic_per_pair = result['region_mean_log']
```

The pair dimension remains available for histograms, minima, threshold counts,
and quantiles of these per-pair interval summaries. This mode does not save
individual site posteriors, confidence intervals, or posterior parameters.
Overlapping intervals are supported. Double-precision reduction order and log
implementations can cause small rounding differences from NumPy reductions.
Check threshold and histogram agreement on representative production inputs.

The chromosome analysis has an opt-in path that preserves its CSV/NPZ schema,
calibration, gene statistics, and lead-window statistics:

```sh
python analysis/genome_wide/infer_chromosome.py \
  --chr 22 --populations YRI \
  --cache-dir /path/to/cache --samples /path/to/samples.txt \
  --lead-variants /path/to/leads.tsv --output-dir /path/to/new-results \
  --core-block-sites 6000000 --flank-sites 0 --pair-chunk 63190 \
  --gpu-region-moments --checkpoint-sites 4096
```

Choose `--pair-chunk` to fit the actual chromosome, region count and GPU memory.
Use a smaller chunk for longer chromosomes or larger cohorts. The default
pair chunk remains unchanged for compatibility. Full-chromosome context and
zero flanks are required for this mode.

## Measured output scopes

On a YRI chromosome 22 panel (356 haplotypes, 313,277 retained markers, 63,190
pairs), the complete gene/lead workflow on one V100S took a median 6.44 seconds
(6.62, 6.39, 6.44 across three runs), versus 348.51 seconds for the original CUDA
workflow in a matched run (54.2×). Both
included process startup, input loading, calibration, full inference, final
statistics, and CSV/NPZ output. All 43 NPZ fields and 447 gene rows agreed:
integer counts and histograms exactly, floating summaries within 4.7e-15
relative error. The final B200 path took a median 6.96 seconds over three runs;
its outputs passed the same comparison. These are measurements of this panel,
not scaling estimates for other chromosomes or cohorts.

The separate per-pair region-matrix benchmark, which writes about 448 MB of
moment arrays, took medians of 3.84 seconds on one V100S and 2.17 seconds on
two V100S GPUs. It excludes the chromosome analysis's final aggregation and
therefore is not interchangeable with the complete workflow timing above.

Dense export of every site-by-pair float32 mean (79.2 GB) took 36.47, 27.36,
and 28.09 seconds using two V100S GPUs, 8,192-pair batches, 4,096-site
checkpoints and four concurrent NPY writers. The median is 28.09 seconds:
12.6× faster than the original 355.33-second CUDA dense-output baseline on
one V100S. Each full matrix was checked bit-for-bit against that baseline.
A one-V100S/two-writer screening run took 46.59 seconds (7.6×), and a one-B200/
four-writer run took 66.94 seconds. These timings include output writes and
finite/positive checks; neither the original nor new runs force filesystem
`fsync`. Reference equality comparisons run outside the timed section.

The direct exporter further reduced this full dense workload to 18.80, 15.62,
and 15.14 seconds on one V100S (median 15.62 seconds, 22.8× the original CUDA
baseline). These runs used 8,192-pair batches, 4,096-site tiles, three staging
slots and eight workers on the GPU. Every value matched the original export.
Pinned staging totaled 3.11 GB; this excludes the operating system's file cache.
Two V100S GPUs with four workers each took a median 16.64 seconds, showing that
another GPU did not improve this output-bound workload in the repeated test.

These measurements use ordinary buffered writes. A separate one-GPU run with
`durable=True`, including all file/directory sync calls, took 86.10 seconds and
also matched exactly. Do not present the buffered timings as durable-storage
throughput. Output size and storage speed remain important limits even when
inference and transfers overlap.

The independent [matched-panel comparison](../benchmarks/pairwise_scaling/dense_export_20260926/README.md)
then measured a 15.91-second CUDA median (15.91, 18.21, 15.19 seconds after a
full-workload warmup). It covers six pair counts and includes the verified
single-process CPU Gamma-SMC and one/eight-worker ASMC baselines, with complete
timing data and native output-scope descriptions.

Keep these output scopes and device counts explicit: the roughly 54× gene/
lead workflow improvement preserves its final statistics, while the dense
export result also retains every per-site posterior mean.
