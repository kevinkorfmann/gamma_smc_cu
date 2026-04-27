# tmrca.cu

GPU-accelerated pairwise coalescence-time inference. A CUDA implementation of the
Gamma-SMC HMM (Schweiger and Durbin, 2023) that decodes pairwise TMRCA at every
segregating site for hundreds-of-thousands of pairs in a single GPU pass.

tmrca.cu is **at parity with the reference gamma_smc binary on accuracy**
across a [15-config stdpopsim cross-species benchmark](test_suite.md) while
delivering **25×–190× end-to-end speedups**.

```python
import tmrca_cu

result = tmrca_cu.infer(ts)                              # from a tree sequence
result = tmrca_cu.infer(G, positions, mu=1.25e-8)        # from a genotype matrix
result["mean"]    # (n_sites, n_pairs) float32 — posterior mean TMRCA in generations
result["pairs"]   # list of (i, j) haplotype index pairs
```

## Install

```bash
git clone https://github.com/kevinkorfmann/tmrca.cu
cd tmrca.cu
pixi install
pixi run build
```

Requires an NVIDIA GPU with compute capability ≥ 8.0 (A40 / A100 / H100 / B200 / RTX 3090+) and the CUDA toolkit. The build is configured for `sm_80`; override with `-DCMAKE_CUDA_ARCHITECTURES=...` if you need a different arch.

## Run

The one-line API is `tmrca_cu.infer(...)`. It accepts either a `tskit.TreeSequence`
or a phased genotype matrix `G` of shape `(n_haplotypes, n_sites)` plus a 1D
position array.

```python
import numpy as np
import tmrca_cu

# Phased haplotypes: 0/1, shape (n_haps, n_sites), dtype uint8
G = np.load("haps.npy").astype(np.uint8)
positions = np.load("positions.npy").astype(np.float64)  # bp coordinates

result = tmrca_cu.infer(
    G, positions,
    mu=1.25e-8,         # per-site per-generation mutation rate
    rho=1e-8,           # per-site per-generation recombination rate
    Ne=10000,           # effective population size (scales the output)
    pairs=[(0, 1), (2, 3)],   # optional; defaults to all n*(n-1)/2 pairs
    mean_only=True,     # set to False to also return 95% Wilson-Hilferty CI
)

mean = result["mean"]              # (n_sites, n_pairs) float32 generations
positions = result["positions"]    # (n_sites,)
pairs = result["pairs"]            # list[(int, int)]
```

Inputs must be **phased and biallelic**. Polarization (which allele is ancestral)
does not matter — the model only looks at heterozygosity (the XOR between
haplotypes), which is invariant to allele coding.

### Cohort-sized data

For cohort inputs (1000 Genomes scale and up), use {func}`infer_blockwise`. It
runs the same forward-backward in **padded site blocks** so peak GPU memory is
bounded by one block instead of the full sequence × pairs forward buffer:

```python
result = tmrca_cu.infer_blockwise(
    G, positions,
    pairs=pairs,
    # All defaults are auto-tuning friendly:
    #   core_block_sites='auto'  -> picks largest block that fits all pairs
    #   pair_batch_size=-1       -> C++ auto-chunks pairs
    #   flank_sites=2048         -> burn-in for forward/backward
    verbose=True,
)
```

See [Blockwise FB](blockwise.md) for the full mechanism, the trade-offs, and how
the auto-sizer chooses block dimensions from `cudaMemGetInfo`.

## What this site covers

- [Examples](examples.md) — end-to-end snippets and figures for each of
  the three output modes (`mean_only`, with CI, full Gamma posterior).
- [Algorithm](algorithm.md) — the Gamma-SMC HMM, the moment-matched
  approximation, the flow-field transition operator, what `infer()` actually
  computes.
- [CUDA optimizations](cuda.md) — bitpacked genotypes, cached transitions,
  thread-per-pair layout, half-precision flow caches, texture-memory bilinear
  interpolation, multi-stream overlap.
- [Blockwise FB](blockwise.md) — how `infer_blockwise()` decomposes the site
  axis, why flanks are needed, the auto-sizer math, when to use it.
- [stdpopsim test suite](test_suite.md) — cross-species benchmark of
  `tmrca.cu` vs the reference `gamma_smc` binary, how to rerun it on a SLURM
  cluster, and the latest per-config results.
- [API reference](api.md) — Python entry points and their parameters.

```{toctree}
:hidden:
:maxdepth: 1

self
examples
algorithm
cuda
blockwise
test_suite
api
```
