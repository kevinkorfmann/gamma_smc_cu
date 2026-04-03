# tmrca.cu

GPU-accelerated pairwise coalescence time estimation.

Estimate the time to the most recent common ancestor (TMRCA) at every
segregating site for every pair of haplotypes — two orders of magnitude
faster than existing methods, with improved accuracy from a full
forward-backward posterior.

```{image} ../speed_comparison.png
:alt: Speed comparison
:width: 700px
```

## Quick start

```python
import numpy as np
import msprime
from tmrca_cu import FlowContext

# Simulate (or load your own haplotype data)
ts = msprime.sim_ancestry(samples=100, sequence_length=1_000_000,
    recombination_rate=1e-8, population_size=10000, random_seed=42)
ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)

G = ts.genotype_matrix().T.astype(np.uint8)  # (n_haplotypes, n_sites)
pos = np.array([v.position for v in ts.variants()])

# Create GPU context (loads data + flow field onto GPU once)
ctx = FlowContext(G, pos, Ne=10000, mu=1.25e-8, rho=1e-8,
                  flow_field_path="path/to/default_flow_field.txt")

# Estimate TMRCA for any set of pairs
pairs = [(0, 1), (2, 3), (10, 50)]
result = ctx.run_fb(pairs, mean_only=True)

tmrca = result["mean"]  # shape [n_sites, n_pairs]
```

That's it. Three lines after loading data: create context, pick pairs, get TMRCAs.

## All pairs at once

```python
all_pairs = [(i, j) for i in range(n) for j in range(i)]

# Per-site TMRCA averaged across all pairs (GPU-side reduction)
summary = ctx.run_fb_summary(all_pairs)
site_mean = summary["site_mean"]  # shape [n_sites]
```

For 200 haplotypes (19,900 pairs) over 1 Mb, this takes **20 milliseconds**.

## Installation

```bash
# Clone
git clone https://github.com/kevinkorfmann/tmrca.cu
cd tmrca.cu

# Build (requires CUDA toolkit + pybind11)
pixi install
pixi run build

# Test
pixi run test-unit
```

Requires an NVIDIA GPU with compute capability 8.0+ (A100, H100, RTX 3090+).

## API

### `FlowContext`

The main entry point. Holds genotype data and the flow field cache on GPU.

```python
ctx = FlowContext(G, positions, Ne, mu, rho, flow_field_path, cache_steps=0)
```

| Method | Returns | Use case |
|--------|---------|----------|
| `ctx.run_fb(pairs)` | `{"mean": [S, P]}` | Forward-backward posterior (best accuracy) |
| `ctx.run_fwd(pairs)` | `{"mean": [S, P]}` | Forward-only (fastest) |
| `ctx.run_fb_summary(pairs)` | `{"site_mean": [S]}` | Per-site average across pairs |

All methods accept `mean_only=False` to also return `"lower"` and `"upper"` (95% credible interval).

### Multi-GPU

```python
from tmrca_cu.multigpu import MultiGPUFlowContext

ctx = MultiGPUFlowContext(G, pos, Ne, mu, rho, ff_path)
result = ctx.run_fb(pairs)  # pairs split across GPUs automatically
```

### From a tree sequence

```python
from tmrca_cu import CoalescenceEstimator

est = CoalescenceEstimator.from_tree_sequence(ts)
result = est.infer_tmrca(pairs)
```

## How it works

`tmrca.cu` implements the Gamma-SMC model of
[Schweiger and Durbin (2023)](https://doi.org/10.1101/gr.277665.122):
at each site, the TMRCA posterior is a Gamma distribution parameterized
by (log-mean, log-CV), propagated through recombination events via a
precomputed flow field lookup table.

Our implementation adds:

- **Forward-backward posterior**: conditions on the full sequence, not just
  upstream sites (r = 0.820 vs 0.791 for forward-only)
- **GPU parallelism**: one CUDA thread per pair, thousands of pairs processed
  simultaneously
- **Fused backward reduction**: per-site summary computed on GPU, minimal
  data transfer

## Demographic robustness

The flow field is precomputed under constant Ne but generalizes well to
non-equilibrium demographies (r = 0.81-0.88 under diverse stdpopsim models).

For variable N(t), use the harmonic-mean Ne estimated from heterozygosity:

```python
theta = mean_heterozygosity  # from your data
Ne_harmonic = theta / (4 * mu)
ctx = FlowContext(G, pos, Ne=Ne_harmonic, ...)
```

See [DEMOGRAPHY.md](https://github.com/kevinkorfmann/tmrca.cu/blob/main/DEMOGRAPHY.md) for details.

## Citation

```bibtex
@article{korfmann2025tmrcacu,
  title={tmrca.cu: GPU-accelerated pairwise coalescence time estimation},
  author={Korfmann, Kevin and Mathieson, Sara},
  year={2025}
}
```

```{toctree}
:hidden:
:maxdepth: 1

self
```
