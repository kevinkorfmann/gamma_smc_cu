# API reference

The two top-level entry points are `tmrca_cu.infer()` and
`tmrca_cu.infer_blockwise()`. Both wrap the same C++ `FlowContext` machinery —
`infer()` runs forward-backward over the full sequence, `infer_blockwise()`
runs it in padded site blocks. See [Algorithm](algorithm.md) for what they
compute and [Blockwise FB](blockwise.md) for when to use which.

## `tmrca_cu.infer`

```python
tmrca_cu.infer(
    G_or_ts,
    positions=None,
    mu=1.25e-8,
    rho=1e-8,
    Ne=10000,
    pairs=None,
    flow_field_path=None,
    mean_only=True,
    return_posterior=False,
)
```

Estimate pairwise TMRCA at every site for the requested pairs using
forward-backward decoding of the Gamma-SMC HMM.

**Parameters**

`G_or_ts` *(array-like or `tskit.TreeSequence`)*
: Either a phased haplotype matrix of shape `(n_haplotypes, n_sites)` and
  dtype `uint8` (values 0 or 1), or a tree sequence. If a tree sequence, the
  genotype matrix and site positions are extracted automatically.

`positions` *(array-like, optional)*
: Site positions in bp, shape `(n_sites,)`, dtype `float64`. Required when
  `G_or_ts` is a matrix; ignored when it's a tree sequence.

`mu` *(float, default `1.25e-8`)*
: Per-site per-generation mutation rate.

`rho` *(float, default `1e-8`)*
: Per-site per-generation recombination rate. Used at flow-field cache build
  time, not per call — changing it requires a fresh `FlowContext`.

`Ne` *(float, default `10000`)*
: Effective population size. Output TMRCAs are in *real generations*, scaled
  by `2 * Ne`.

`pairs` *(list of `(int, int)`, optional)*
: Pairs of haplotype indices to decode. Defaults to all $n(n-1)/2$ pairs.
  Pass an empty list to get a `(n_sites, 0)` shaped output.

`flow_field_path` *(str, optional)*
: Path to the flow-field file. Defaults to a bundled
  `default_flow_field.txt`.

`mean_only` *(bool, default `True`)*
: If `False`, also compute 95% Wilson-Hilferty CI bounds (`lower`, `upper`).

`return_posterior` *(bool, default `False`)*
: If `True`, also return the per-site combined Gamma posterior
  parameters as `posterior_alpha` / `posterior_beta` (in scaled coalescent
  time `T_scaled = T / (2*Ne)`). Reconstructable mean is
  `(alpha / beta) * 2 * Ne`; arbitrary quantiles via
  `scipy.stats.gamma(alpha, scale=2*Ne/beta).ppf(q)`. See
  [Algorithm](algorithm.md) for the parameterization.

**Returns**

A `dict` with keys:

| key                | shape                | dtype     | needs                          |
|--------------------|----------------------|-----------|--------------------------------|
| `mean`             | `(n_sites, n_pairs)` | `float32` | always                         |
| `lower`            | `(n_sites, n_pairs)` | `float32` | `mean_only=False`              |
| `upper`            | `(n_sites, n_pairs)` | `float32` | `mean_only=False`              |
| `posterior_alpha`  | `(n_sites, n_pairs)` | `float32` | `return_posterior=True`        |
| `posterior_beta`   | `(n_sites, n_pairs)` | `float32` | `return_posterior=True`        |
| `positions`        | `(n_sites,)`         | `float64` | always                         |
| `pairs`            | `list[(int, int)]`   |           | always                         |

## `tmrca_cu.infer_blockwise`

```python
tmrca_cu.infer_blockwise(
    G_or_ts,
    positions=None,
    mu=1.25e-8,
    rho=1e-8,
    Ne=10000,
    pairs=None,
    flow_field_path=None,
    mean_only=True,
    core_block_sites="auto",
    flank_sites=2048,
    pair_batch_size=-1,
    max_streams=1,
    verbose=False,
    return_posterior=False,
)
```

Memory-bounded forward-backward decoder. Output is identical to
{func}`infer` (byte-equal under sensible configs). See [Blockwise FB](blockwise.md)
for the mechanism.

**Extra parameters beyond `infer()`**

`core_block_sites` *(int or `"auto"`, default `"auto"`)*
: Sites kept from each block. `"auto"` queries free GPU memory and picks the
  largest power-of-two block that lets all pairs fit in a single batch
  (clamped to `[1024, 32768]`, or to `n_sites` if everything fits in one
  block). Pass an int to override.

`flank_sites` *(int, default `2048`)*
: Burn-in sites on either side of the core. Must be `> 0` whenever the
  sequence is split into more than one block, otherwise per-block
  forward/backward passes start from the wrong prior. Rejected at the
  wrapper level.

`pair_batch_size` *(int, default `-1`)*
: Maximum pairs per kernel launch. `-1` lets the C++ backend auto-chunk pairs
  to fit available GPU memory; the Python wrapper caps this at `n_pairs` so
  the C++ doesn't over-allocate scratch. Pass a positive int to cap manually.

`max_streams` *(int, default `1`)*
: Number of concurrent CUDA streams. `2` typically halves wall time on large
  inputs at the cost of ~2x peak per-block scratch.

`verbose` *(bool, default `False`)*
: If `True`, print the chosen block sizing and memory estimate.

`return_posterior` *(bool, default `False`)*
: If `True`, also return the per-site combined Gamma posterior parameters
  as `posterior_alpha` and `posterior_beta`. **Currently supported only
  with `max_streams=1`** — passing `max_streams>1` together with
  `return_posterior=True` raises a `ValueError`.

**Required**

Unlike `infer()`, `infer_blockwise()` **requires `pairs` to be passed
explicitly**. There is no default-to-all-pairs behaviour in v1.

**Returns**

Same shape as `infer()` plus an extra `blocks` array describing the block
windows used:

| key                | shape                | dtype     | needs                          |
|--------------------|----------------------|-----------|--------------------------------|
| `mean`             | `(n_sites, n_pairs)` | `float32` | always                         |
| `lower`            | `(n_sites, n_pairs)` | `float32` | `mean_only=False`              |
| `upper`            | `(n_sites, n_pairs)` | `float32` | `mean_only=False`              |
| `posterior_alpha`  | `(n_sites, n_pairs)` | `float32` | `return_posterior=True`        |
| `posterior_beta`   | `(n_sites, n_pairs)` | `float32` | `return_posterior=True`        |
| `blocks`           | `(n_blocks, 4)`      | `int32`   | always                         |
| `positions`        | `(n_sites,)`         | `float64` | always                         |
| `pairs`            | `list[(int, int)]`   |           | always                         |

The `blocks` array stores `(core_start, core_stop, padded_start, padded_stop)`
for every block — useful for debugging or for stitching custom outputs.

## `tmrca_cu._core.cuda_mem_info`

```python
free_bytes, total_bytes = tmrca_cu._core.cuda_mem_info()
```

Returns the free and total bytes on the active CUDA device. Wraps
`cudaMemGetInfo()`. Used by `infer_blockwise()`'s auto-sizer; exposed in case
you want to inspect or implement your own block sizing logic.

## Lower-level entry points

These bypass the Python wrappers and call the C++ `FlowContext` directly. Use
them only if you need to amortize context construction across many `run_*`
calls on the same data, or if you want fine-grained control over chunking.

```python
ctx = tmrca_cu.FlowContext(
    G,                                  # uint8 (n_haps, n_sites)
    positions,                          # float64 (n_sites,)
    Ne=10000.0,
    mu=1.25e-8,
    rho=1e-8,
    flow_field_path=...,
    cache_steps=0,                       # 0 means auto, otherwise n_max_steps
)

# Forward-only: fastest, no backward correction (filtered, not smoothed)
result = ctx.run_fwd(pairs, mean_only=True)

# Standard FB: smoothed posterior, full forward buffer
result = ctx.run_fb(pairs, mean_only=True)

# Blockwise FB: bounded GPU memory, multi-block decoding with flanks
result = ctx.run_fb_blockwise(
    pairs,
    core_block_sites=8192,
    flank_sites=2048,
    pair_batch_size=256,
    max_streams=1,
    mean_only=True,
)
```

`FlowContext` holds the bitpacked genotypes, position array, and the
precomputed multi-step flow-field cache on the GPU for its entire lifetime.
The cost of construction is dominated by the cache build (a few hundred ms in
double-precision on host); subsequent calls reuse all GPU state.

## Multi-GPU

```python
from tmrca_cu import MultiGPUFlowContext

mgc = MultiGPUFlowContext(G, positions, Ne=10000, mu=1.25e-8, rho=1e-8)
result = mgc.run_fb(pairs)
```

`MultiGPUFlowContext` builds one `FlowContext` per visible CUDA device and
partitions the pair list across them. Set `CUDA_VISIBLE_DEVICES` to control
which GPUs are used. There is no cross-GPU communication during decoding —
the pair partition is embarrassingly parallel — so scaling is near-linear in
GPU count for any pair set large enough to amortize setup.
