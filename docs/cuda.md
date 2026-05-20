# CUDA optimizations

This page documents the GPU-side decisions in `gamma_smc_cu`. The
[algorithm](algorithm.md) page covers what we compute; this one covers how the
implementation makes that computation fit on a GPU and run at hardware roofline.

## Bird's-eye view of the kernels

The hot path is forward-backward decoding for many `(haplotype_i, haplotype_j)`
pairs over a long sequence of sites. There is one independent FB chain per
pair. The implementation maps that to GPU as:

- **One CUDA thread = one pair.** Each thread runs the entire forward sweep
  over all $S$ sites, then the backward sweep, in registers. No cooperation
  between threads.
- **Block of 256 threads = 256 pairs.** Block size is fixed; the grid is
  $\lceil n_{\text{pairs}} / 256 \rceil$. Occupancy is bounded by register
  pressure (~64 regs/thread typical), giving ~4 blocks per SM at full
  occupancy.
- **Cooperative shared-memory load** of the flow-field table happens once at
  the top of the kernel; the table is small (~20 KB) and reused by every
  thread in the block.

The kernel definitions live in `src/kernels/gamma_smc_flow.cu`. Multiple
specializations exist (uncached, multi-step cached, fp16-cached, texture-based,
forward-only, blockwise) and each is selected by the C++ launcher in
`src/bindings.cpp` depending on which `FlowContext` method the Python wrapper
calls.

## Bitpacked genotypes

A phased haplotype matrix `G[n_haps × n_sites]` of `uint8` is **1 bit of useful
information per cell**. We pack it into `uint64[n_haps × ceil(S/64)]`:

```
packed[h][w] = bit b is set iff G[h][w*64 + b] == 1
```

The kernel never touches the unpacked matrix. Per-site site-emission becomes:

```cuda
int w = s >> 6;          // word index
int bit = s & 63;        // bit position within the word
if (w != cur_word) {     // reload only on word boundary
    xor_w = packed[hi*n_words + w] ^ packed[hj*n_words + w];
    cur_word = w;
}
bool het = (xor_w >> bit) & 1ULL;
```

Two consequences worth pointing out:

1. **Words are reused across 64 consecutive sites.** Inside a block of 64 sites
   the kernel does one global-memory load and 64 cheap register tests. Genotype
   bandwidth drops by 64× compared to reading `uint8`s.
2. **The `^` is the entire emission predicate.** A heterozygous site for a
   pair $(i, j)$ is exactly the bitwise XOR being set at that position. There
   is no per-pair preprocessing — all $\binom{n}{2}$ pair difference patterns
   are derived on the fly from the same shared bitpacked array.

The bitpack itself is computed once on host upload by `bitpack_kernel`
(`src/kernels/bitpack.cu`); after that, only the packed array travels to GPU
memory.

## The cached transition kernel

The naive per-site transition runs `flow_field_advance()` (3–10 sub-iterations
of `bilinear → multiply-add → clamp`) every step. For chr2 at typical SNP
density that's tens of millions of bilinear lookups per pair, all reading the
same 51×50 grid. The arithmetic is fast but the L2 traffic and the
adaptive sub-step loop are not.

The cached kernel precomputes, **on host, in double precision, once per
context construction**, a stack of "n-step" snapshots of the transition
operator:

```
cache[step n][grid_row r][grid_col c] = (m', c') after applying the transition
                                         operator n times starting from the
                                         grid point (m_r, c_c).
```

Layout: `float[n_max_steps × FF_GRID]` for `mean`, same for `cv`. With the
default `MAX_STEP_RHO = 0.1` and a typical chr2 max gap of ~10 kb, the cache
needs only ~256 layers and fits in ~5 MB total.

At decode time, a transition for a gap of $\Delta\mathrm{bp}$ becomes a
**single** bilinear lookup at the right cache layer, with no adaptive
sub-stepping in the device kernel:

```cuda
int gap_steps = (int)(positions[s] - positions[s-1] + 0.5);
cache_advance(m, c, cache_mean, cache_cv, gap_steps, n_max_steps);
```

For gaps larger than the cache depth, `cache_advance()` decomposes them into
chunks of `n_max_steps`. This happens in `< 1%` of sites in 1KG-density data,
so the overhead is negligible.

The performance win is massive: the inner loop drops from ~50 ALU ops per
site to ~10, and the kernel becomes memory-bound on the cache reads instead.
Which is then itself the next thing to optimize.

## Reducing flow-cache memory traffic

The bilinear lookup needs the four corner values around the current $(m, c)$
point. With separate `cache_mean[]` and `cache_cv[]` arrays, that's
**8 float reads per cache_advance call** (4 corners × 2 fields). Several
variants in the kernel file shrink that:

| variant                       | reads per lookup | bytes per lookup | when used                |
|-------------------------------|------------------|------------------|--------------------------|
| separate `float`              | 8 × float        | 32 B             | baseline cached path     |
| interleaved `float2`          | 4 × float2       | 32 B (1 ld128)   | `cache_bilinear_f2`      |
| half-precision `__half2`      | 4 × half2        | 16 B             | `cache_bilinear_h2`      |
| layered 2D texture (fp16)     | 1 tex fetch      | hardware path    | `cache_advance_tex`      |

**`float2` interleaving** halves the number of L2 transactions (8 single-float
reads vectorize into 4 wide reads). The data layout is identical from the
math standpoint — only the in-memory ordering changes.

**`__half2` packing** halves the bytes per lookup. Conversion to fp32 is free
(`__half22float2` is two HW instructions). Accuracy loss is below the
flow-field's own grid quantization, so end-to-end TMRCAs are unchanged at
fp32 output precision.

**Texture-based bilinear** maps the cache to a `cudaTextureObject_t` with
hardware bilinear filtering. A single `tex2DLayered<float2>(...)` call replaces
~26 ALU instructions of software bilinear. This is the fastest forward-only
path; the FB path still uses the explicit `__ldg` variants because the texture
unit is under-utilised when the kernel is bandwidth-bound elsewhere.

## Site-major output layout

The forward buffer and the output mean/lower/upper arrays are stored in
**site-major** order:

```
out[s × n_pairs + pid]
```

with consecutive threads (= consecutive pair IDs) writing consecutive memory.
That's a 256-byte coalesced write per warp on the forward-state store and on
the output write, which is the kind of access the global-memory unit is built
for.

Pair-major would be the alternative — each thread writes a contiguous
`out[pid][s..s+1]` slice — but adjacent threads would then write to addresses
$S$ apart, serializing into 32 separate transactions per warp. We measured
this and it costs ~5× wall time on the backward pass.

## Forward buffer

The standard FB needs the forward state at every site to combine with the
backward state on the way back. That buffer is:

```
fwd_buf[2 × S × n_pairs]   // 2 floats: (m, c)
```

= `8 × S × n_pairs` bytes. For chr2 at 5 M sites and 1 K pairs that's 40 GB —
you'd run out of GPU memory before getting to the backward sweep. Two
mitigations are in place at the C++ level inside `FlowContext`:

1. **Pair chunking.** `compute_max_fb_chunk()` calls `cudaMemGetInfo()`, leaves
   ~512 MB of headroom, and computes the largest pair-chunk that fits the
   forward buffer + output arrays. The C++ then loops over pair chunks,
   running a complete forward+backward per chunk.
2. **`run_fwd()` (forward-only).** If you only want a smoothed mean per site
   from the forward filter and don't need the backward correction (i.e. the
   filtered, not smoothed, posterior), `run_fwd()` skips the buffer entirely
   and writes the mean directly. This is the fastest path on the GPU and is
   what the texture-based kernel above is paired with.

The third mitigation is the **blockwise FB**, which is significant enough to
have [its own page](blockwise.md). It splits the *site* axis instead of the
pair axis, so the per-block forward buffer is bounded by the block size and
the kernel can decode arbitrarily long sequences regardless of `n_pairs`.

## Persistent `FlowContext`

`FlowContext` is the C++ class behind `gamma_smc_cu._core.FlowContext`. It owns,
on the GPU, for the lifetime of the Python object:

- the bitpacked genotype matrix `d_packed_`,
- the position array `d_pos_`,
- the precomputed flow-field cache (`d_cache_mean`, `d_cache_cv`),
- a pre-allocated forward-buffer slab (grown on demand to the max pair-chunk
  size),
- a persistent CUDA stream.

The cost of constructing the context is dominated by the host-side cache build
(double-precision flow-field iteration over 256 layers) — typically a few
hundred milliseconds. After that, every `run_fb()` / `run_fb_blockwise()`
call reuses the same GPU resident state. There is **no per-call CUDA setup
overhead** for repeated calls on the same data with different pair selections —
which is the regime the cohort and chunked workflows live in.

## Multi-stream blockwise

The blockwise kernel can be launched on multiple CUDA streams in parallel.
Each stream gets its own per-block scratch buffers, and the launcher loops over
blocks-then-pair-chunks round-robin across streams. Stream $k$'s D2D output
copy is overlapped with stream $k+1$'s forward kernel:

```
stream 0:  fwd0  bwd0  D2D0
stream 1:        fwd1  bwd1  D2D1
stream 2:              fwd2  bwd2  D2D2
```

In the bench, `max_streams=2` recovers ~50% of the gap between blockwise and
the full `infer()` baseline on cohort-shaped inputs, at the cost of ~2× the
per-block scratch (a few hundred MB). `max_streams=1` is the default; opt into
2 when you know the workload is large enough to benefit.

## Multi-GPU

`MultiGPUFlowContext` (in `python/gamma_smc_cu/multigpu.py`) builds one
`FlowContext` per visible GPU and partitions a pair list across them. Each GPU
runs an independent forward-backward; the host reassembles the per-GPU output
slabs into the final `(n_sites, n_pairs)` array. This is embarrassingly
parallel — there is no cross-GPU communication during the kernel — and gives
near-linear speedup with GPU count for any pair set large enough to amortize
the per-context setup.

## Putting numbers on it

The bench in `benchmarks/bench_repro.py` (also see the entries in
`_scratch/docs_old/_static/`) reports wall time and GPU peak memory across a
small grid of input shapes for `infer()` (full FB) and `infer_blockwise()`
(several configs). On a single MIG slice of a B200 (45 GB), for a synthetic
40 × 500k input × 780 pairs:

| variant                       | wall time | GPU peak (Δ) |
|-------------------------------|----------:|-------------:|
| `infer()` mean only           |    1.02 s |     4 472 MB |
| `infer()` mean + CI           |    1.97 s |     7 448 MB |
| `infer_blockwise()` default   |    3.48 s |        44 MB |
| `infer_blockwise()` 2 streams |    2.14 s |     1 568 MB |

The lesson is in the rightmost column: blockwise's GPU footprint is essentially
**constant in input size**, where `infer()`'s scales linearly until you fall
off the GPU. For inputs that fit in `infer()`'s budget, `infer()` is faster.
For inputs that don't, blockwise is the only option — and that's exactly the
cohort regime.

## Where this lives in the code

| concern                          | file                                       |
|----------------------------------|--------------------------------------------|
| flow-field grid + load           | `include/gamma_smc_cu/flow_field.h`, `src/flow_field.cpp` |
| forward / backward / cached / fp16 / texture / blockwise kernels | `src/kernels/gamma_smc_flow.cu` |
| bitpack / unpack / pairwise XOR  | `src/kernels/bitpack.cu`                   |
| `FlowContext` C++ class          | `src/bindings.cpp` (around line 2390)      |
| Python wrappers                  | `python/gamma_smc_cu/infer.py`                 |
| multi-GPU dispatcher             | `python/gamma_smc_cu/multigpu.py`              |
