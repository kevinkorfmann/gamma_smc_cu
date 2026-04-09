# Blockwise FB

`tmrca_cu.infer_blockwise()` is a memory-bounded variant of the standard
forward-backward decoder. It produces **the same output** as
{func}`infer` (byte-identical, when configured correctly) but its peak GPU
memory is bounded by one block instead of the full sequence × pairs forward
buffer. This is what makes cohort-sized inputs (1000 Genomes scale and up)
feasible on a single GPU.

## What `infer()` does and why it runs out of memory

The standard `infer()` runs the textbook FB:

1. Forward sweep over all $S$ sites, **storing the forward state at every site**
   in a buffer of shape `(S, P, 2)` floats — `8 × S × P` bytes.
2. Backward sweep over all $S$ sites, combining with the forward buffer to
   write the posterior.

For chr2 of 1KG (5M sites) with all 5008-haplotype pairs (~12.5 M pairs), the
forward buffer alone is `8 × 5e6 × 12.5e6 ≈ 500 TB`. The C++ rescues you by
chunking pairs, so each chunk is ~100–1000 pairs and the buffer drops to ~40
GB per chunk — but you still need 12 500 chunks, each rerunning forward+backward
over the entire 5M-site sequence. That's the time bottleneck.

Even before pair chunking, you eat the chr2 genotype matrix
(`5008 × 5e6 ÷ 8 ≈ 3.1 GB`), the position array (`5e6 × 8 = 40 MB`), the
flow-field cache (~5 MB), and a per-pair-chunk forward buffer of multiple GB.
On a 24 GB A10 you don't have headroom for a useful pair chunk; on an 80 GB
H100 you do but you've already given up most of the GPU to scratch.

Blockwise inverts the trade-off: split the **site** axis instead.

## What blockwise does

Pick a core block size $B$ (`core_block_sites`) and a flank size $F$
(`flank_sites`). Decompose the $S$ sites into core blocks
$[0, B), [B, 2B), \ldots$ and decode each block on a **padded** window
$[\text{core\_start} - F,\ \text{core\_stop} + F)$.

```
sites 0 ............................................................. S
       ┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
       │ core 0  ││ core 1  ││ core 2  ││ core 3  ││ core 4  │
       └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘
       ◄────►            ◄──┴──┐    ┌──┴──►            ◄────►
        flank             flank       flank             flank
       ◄══════════padded window for block 2══════════►
```

For each block:

1. Run a fresh forward+backward over the **padded** window — `(B + 2F)` sites
   instead of $S$. The forward buffer is `(B + 2F) × P × 8` bytes.
2. Throw away the flank parts of the output.
3. Copy the core part `[core_start, core_stop)` into the final
   `(S, P)` host array.

The kernels for this live in `gamma_smc_flow.cu` as
`gamma_smc_cached_forward_block_kernel` and
`gamma_smc_cached_backward_block_kernel<bool WRITE_CI>`. The block-window
generation, per-block scratch allocation, and stitching live in
`FlowContext::run_fb_blockwise()` in `src/bindings.cpp`.

## Why flanks matter (the burn-in)

The forward kernel initializes the per-pair state to $(m, c) = (0, 0)$ at the
start of every block. That's the marginal-prior fixed point — *not* the true
forward state at `padded_start`. Without flanking history, every block starts
from the wrong distribution and the per-block posterior is invalid for the
first few hundred sites.

The flanks are a **burn-in**: by the time the forward kernel walks the $F$
flank sites and reaches `core_start`, the state has converged to whatever the
full-sequence forward sweep would have produced there, *modulo* the small
contribution from sites that came before `padded_start`. Symmetrically for
the backward pass.

How much flank is enough depends on the HMM's autocorrelation length. For
Gamma-SMC at typical human-genetics parameters
($\rho = 10^{-8}$, $N_e = 10^4$, median 1KG SNP spacing ≈ 300 bp) the
forward state converges within a few hundred sites. The default
`flank_sites=2048` is conservative — the bench shows
`flank_sites=1024` already gives byte-identical output to `infer()` across
input shapes. Drop it lower at your own risk; pad it more if you suspect
unusual rho or sparse sites.

`flank_sites=0` with **multiple blocks** is broken and the Python wrapper
rejects it with a `ValueError`. The single-block case (`core_block_sites >=
n_sites`) is allowed because there is nothing to stitch — it just runs one
big block end-to-end and is numerically identical to `infer()`.

## Defaults that just work

The default invocation is

```python
result = tmrca_cu.infer_blockwise(G, positions, pairs=pairs)
```

with:

| parameter           | default       | meaning                                       |
|---------------------|---------------|-----------------------------------------------|
| `core_block_sites`  | `'auto'`      | query free GPU memory and pick the largest block that fits all pairs in one batch (clamped to `[1024, 32768]` or to `n_sites` if everything fits in one block) |
| `flank_sites`       | `2048`        | burn-in on either side of the core            |
| `pair_batch_size`   | `-1`          | C++ auto-chunks pairs to fit GPU memory; the Python wrapper caps this at `n_pairs` so the C++ doesn't over-allocate |
| `max_streams`       | `1`           | single CUDA stream; opt into `2` for ~2x speedup at ~2x peak scratch |
| `mean_only`         | `True`        | skip CI bounds; saves ~40% wall time and 2/3 of output bytes |
| `verbose`           | `False`       | if True, print the chosen block sizing and memory estimate |

On a GPU with plenty of headroom this collapses to a single full-sequence
block — the output is exactly what `infer()` would produce. On a GPU with
tight memory, the auto-sizer picks a smaller block and lets the C++
pair-chunker handle further memory pressure. Either way you get a correct
answer; the only thing that varies is the wall time.

## How `core_block_sites='auto'` picks a value

The wrapper queries free GPU bytes via the new `_core.cuda_mem_info()`
binding (which wraps `cudaMemGetInfo()`), reserves headroom, and solves for
the largest core block that lets every pair fit in a single batch.

```text
bytes_per_pair_per_padded_site = 4 * (2 + n_arrays)
                               = 12      # mean only
                               = 20      # with CI

headroom    = max(512 MB, 20% × free_bytes)
budget      = free_bytes - headroom
padded_max  = budget // (bytes_per_pair_site × n_pairs)
core_max    = padded_max - 2 × flank_sites

if core_max ≥ n_sites:   choose n_sites           # single block, == infer()
elif core_max ≤ 0:       choose 1024              # min block, C++ pair-chunks
else:                    choose 2^floor(log2(core_max))   # clamped to [1024, 32768]
```

The factor `2 + n_arrays` is the per-block GPU cost: 2 floats for the forward
buffer (mean + cv) plus 1 (mean only) or 3 (with CI) floats for the output.

The `2^floor(log2(...))` rounding gives a power-of-two block size for cache
friendliness; the `[1024, 32768]` clamp avoids pathologically small blocks (too
much per-block overhead) and pathologically large blocks (no benefit over
`infer()`).

If `cuda_mem_info()` fails for any reason — no GPU, driver error, monkeypatched
in tests — the wrapper falls back to `core_block_sites=8192` and emits a
`UserWarning`.

## How `pair_batch_size=-1` is handled

The C++ backend allocates per-block scratch buffers sized by the chunk cap
**before** it knows how many pairs the call has. With `pair_batch_size=-1`,
the C++ would use `compute_max_fb_block_chunk()`, which on a roomy GPU can be
much larger than the actual `n_pairs` and produces over-sized cudaMallocs that
nearly exhaust VRAM.

The Python wrapper guards against this by capping `pair_batch_size=-1` to
`n_pairs` before forwarding the call:

```python
effective_pair_batch_size = pair_batch_size
if effective_pair_batch_size == -1:
    effective_pair_batch_size = max(1, n_pairs)
```

The C++ then takes `min(effective_pair_batch_size, auto_chunk_cap)`, so:

- with few pairs and plenty of memory: scratch sized for `n_pairs` (cheap),
- with many pairs and tight memory: scratch sized by `auto_chunk_cap` (the
  C++ pair-chunker takes over),
- never anything in between.

## Numbers from the bench

Same setup as in [CUDA optimizations](cuda.md): single MIG slice of a B200,
synthetic uint8 genotype data with fixed seed, mean-only output. Wall time in
seconds, GPU peak Δ over baseline in MB.

| Grid case | shape | n_pairs | infer | blk_default | blk_streams2 |
|---|---|---:|---:|---:|---:|
| S_small_20x50k    | 20 × 50k   |   190 |   0.077 s / 112 MB | 0.091 s / 36 MB | 0.062 s / 112 MB |
| M_med_30x200k     | 30 × 200k  |   435 |   0.337 s / 998 MB | 0.832 s / 38 MB | 0.444 s / 406 MB |
| W_wide_50x100k    | 50 × 100k  | 1 225 |   0.276 s / 1404 MB | 1.155 s / 36 MB | 0.550 s / 540 MB |
| L_large_40x500k   | 40 × 500k  |   780 |   1.021 s / 4472 MB | 3.476 s / 44 MB | 2.141 s / 1568 MB |

Read the rightmost three columns: `blk_default`'s GPU memory delta is
**essentially constant ~40 MB regardless of input size**, while `infer()`'s
scales linearly. For inputs that comfortably fit `infer()`'s budget, `infer()`
is faster — there is real per-block overhead. For inputs that don't fit,
blockwise is the only option, and `max_streams=2` recovers about half the
overhead by overlapping kernel exec with H2D/D2H copies.

## When to use blockwise

| situation                                                        | use                            |
|-------------------------------------------------------------------|--------------------------------|
| `S × n_pairs × 8` fits comfortably in GPU memory                 | `infer()`                      |
| Cohort-sized: 1000G + chr2 + all pairs                            | `infer_blockwise()`, defaults  |
| You want a memory-bounded runtime ceiling regardless of input    | `infer_blockwise()`, defaults  |
| You're deciding whether to invest in a bigger GPU                | `infer_blockwise()` first      |
| Tiny inputs, small GPU                                            | `infer()` (less per-call setup)|

The defaults make `infer_blockwise(G, pos, pairs=pairs)` correct on any GPU.
The worst case is "slower than `infer()`", never "out of memory" or "wrong
answer".

## Verification

The test suite at `tests/unit/test_infer.py::TestInferBlockwise` covers:

- input validation (rejecting `flank_sites=0` with multi-block, non-int
  block sizes, invalid stream counts, missing pairs),
- the auto-sizer (mocked `cuda_mem_info` returning a fixed budget),
- the warning path for over-sized blocks,
- the `pair_batch_size=-1` wrapper cap,
- byte-equivalence between `infer_blockwise(core_block_sites=n_sites,
  flank_sites=0)` and `infer()` (single-block degenerate case),
- byte-equivalence between single-stream and multi-stream blockwise.

Cross-build verification (paper-v3-frozen vs HEAD) confirmed all md5 hashes
match across a small grid of input shapes for both `mean_only=True` and
`mean_only=False`. See the bench script at `benchmarks/bench_blockwise.py`.
